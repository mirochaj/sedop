"""
annealer.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-01-28.

Description: Driver script for a code to find the optimal discrete spectrum for a continuous 
spectrum of interest, over a region in column density space of interest.
     
"""

import time, sys, re, os, h5py
import numpy as np
import sedop.mods as sm

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    print("Module mpi4py not found.  No worries, we'll just run in serial.")
    rank = 0
    size = 1
    
try:
    from progressbar import ProgressBar, Percentage, RotatingMarker
    pb = True
    widget = ["rt1d: ", Percentage(), ' ', Bar(marker = RotatingMarker()), ' ', ETA(), ' ']
except ImportError:
    print("Module progressbar not found.")
    pb = False    

start = time.time()

# Construct list of parameter files to run
all_pfs = []

# Check to see if we're in batch mode
if sys.argv[1].strip('-') == 'b':   
    f = open(sys.argv[2], 'r')
    for line in f: 
        if not line.strip(): 
            continue
        all_pfs.append(line.strip())
    f.close()
else:
    all_pfs.append(sys.argv[1])

# Loop over parameter files and perform minimization
for i, pfile in enumerate(all_pfs):  
    if rank == 0: 
        print("Starting Monte-Carlo simulation #{0} of {1}...".format(i + 1, len(all_pfs)))
    
    # Read in parameter file, convert to dictionary
    pf = sm.ReadParameterFile(pfile)
    
    # Alter progressbar
    pb &= bool(pf['ProgressBar'])
        
    # Initialize classes that do all the heavy lifting
    ri = sm.RateIntegrals(pf)
    anneal = sm.Anneal(pf)

    # Set up directory hierarchy now rather than later - won't delete anything so be careful
    # The reason this looks really gross to accomodate the general case where OutputDirectory
    # is more than one level away from the CWD.
    made = False
    if rank == 0:
        output = pf['OutputDirectory']
        subout = []
        if re.search('/', output):
            while re.search('/', output):
                i = output.find('/')
                subout.append(output[0:i])
                
                if len(subout) == 1: 
                    if os.path.exists(subout[0]) is False: 
                        os.mkdir(subout[0])
                else: 
                    out = ""
                    for element in subout: 
                        out += "/{0}".format(element)
                                        
                    if os.path.exists(out) is False:
                        os.mkdir(out)
                        made = True
                         
                output = output[i+1:]
        else:
            if not os.path.exists(pf['OutputDirectory']):
                os.mkdir(pf['OutputDirectory'])
                        
    # Make all processors wait here until the directory hierarchy has been built               
    if size > 1:
        MPI.COMM_WORLD.barrier() 
        
    # Determine which integrals we're doing the optimization for
    Integrals = []
    if type(pf['RateIntegral']) is not list: 
        pf['RateIntegral'] = [pf['RateIntegral']]
    for integral in pf['RateIntegral']:
        if integral == 0: 
            Integrals.append(ri.Phi)
        if integral == 1: 
            Integrals.append(ri.Psi)
        if integral == 2: 
            Integrals.append(ri.PhiWiggle)   
        if integral == 3: 
            Integrals.append(ri.PsiWiggle)
        if integral == 4: 
            Integrals.append(ri.PhiHat)   
        if integral == 5: 
            Integrals.append(ri.PsiHat)            
        
    # Set minimization method    
    if type(pf['MinimizationMethod']) is not list: 
        MinMethod = [pf['MinimizationMethod']]
    else: 
        MinMethod = pf['MinimizationMethod']

    # Some useful parameters
    NumberOfWalks = int(pf['NumberOfWalks'])
    MultiSpecies = int(pf['MultiSpecies'])                    # 0 or 1
    N = pf['HINumberOfColumns'] * pf['HeINumberOfColumns']    # total number of column points
    
    if MultiSpecies and N == pf['HINumberOfColumns']:
        raise Exception('MultiSpeces > 0 but no helium columns specified! Exiting.')
    
    SourceMinEnergy = pf['SpectrumMinEnergy']
    if type(SourceMinEnergy) is not float:
        SourceMinEnergy = min(SourceMinEnergy)
        
    # The photo-ionization and heating integrals will be identical for all species 
    # if the lower energy cutoff in the spectrum is above all ionization thresholds.
    # Account for this.    
    E_threshold = np.array([13.6, 24.6, 54.4])
    if np.all(SourceMinEnergy > E_threshold[1:]) or (pf['Species'] == 1):
        Species = 1 # equiv. to saying Phi_HI = Phi_HeI = Phi_HeII and same for Psi
    else:
        Species = 2 # hydrogen and helium
                    
    # Determine where this source first becomes optically thin (with decreasing n_col)
    ncol_temp = np.logspace(15, 22, 71)

    if rank == 0:
        print("Isolating optically thin limit...")
        if pb:
            pbar = ProgressBar(widgets = widget, 
                maxval = len(Integrals) * (MultiSpecies + 1) * len(ncol_temp)).start()
        
    limit_small_tau = np.zeros([len(Integrals), Species])     # Value of integral from N = 3 * [0]
    small_tau_trans = np.zeros([len(Integrals), Species, 71]) # Where in N does tau become > 0?
    for i, integral in enumerate(Integrals):
        for j in range(Species):
            mask = np.zeros(3)
            mask[j] = 1
            for k, column in enumerate(ncol_temp):
                
                global_i = i * Species * len(ncol_temp) + \
                    j * len(ncol_temp) + k + 1
                    
                if global_i % size != rank:
                    continue
                    
                if k == 0:
                    limit_small_tau[i][j] = integral([0] * 3, species = j, continuous = True)      
                                
                cols = np.array([column] * 3)
                small_tau_trans[i][j][k] = integral(np.multiply(cols, mask), species = j, continuous = True)                                
          
                if pb and rank == 0:
                    pbar.update(global_i)
                    
    if rank == 0 and pb:
        pbar.finish()                
          
    limit_small_tau = MPI.COMM_WORLD.allreduce(limit_small_tau, limit_small_tau)
    small_tau_trans = MPI.COMM_WORLD.allreduce(small_tau_trans, small_tau_trans)
          
    # Compare integral values to their optically thin limits                                
    ncol_trans = 1e15 * np.ones([len(Integrals), 2]) # [Gamma [H, He], Heat [H, He]]     
    for i, integral in enumerate(Integrals):
        for j in range(Species):
            
            k = 0
            while small_tau_trans[i][j][k] >= 0.999 * limit_small_tau[i][j]: 
                k += 1
            
            ncol_trans[i][j] = (ncol_temp[k] + ncol_temp[k + 1]) / 2.
                                                        
    # Set lower bounds of integration based on optically thin transition
    HIColumnMin = min(zip(*ncol_trans)[0])
    HIColumnMax = pf["HIColumnMax"]
    HeIColumnMin = min(zip(*ncol_trans)[1])
    HeIColumnMax = pf["HeIColumnMax"]
    
    if rank == 0:
        print("Minimization for hydrogen between n_H = {0} - {1} cm^2.".format(HIColumnMin, HIColumnMax))
        if MultiSpecies: 
            print("Minimization for helium between n_He = {0} - {1} cm^2.".format(HeIColumnMin, HeIColumnMax))
    
        # Construct reference array ("correct" values for RateIntegrals)            
        print('Constructing continuous \'reference\' integrals...')
    
    colHI = np.logspace(np.log10(HIColumnMin), np.log10(HIColumnMax), pf['HINumberOfColumns'])
    if not MultiSpecies:
        colHeI = np.array([0])
    else:
        colHeI = np.logspace(np.log10(HeIColumnMin), np.log10(HeIColumnMax), pf['HeINumberOfColumns'])
    
    if pb and rank == 0:
        pbar = ProgressBar(widgets = widget, 
            maxval = len(Integrals) * 2 * N).start()
    
    ref = np.zeros([len(Integrals), Species, N])
    if MinMethod != [0]:    
        # Construct all possible combinations of column densities        
        col = np.zeros(len(colHI) * len(colHeI), tuple)
        i = 0
        for nHeI in colHeI:
            for nHI in colHI:
                col[i] = (nHI, nHeI, 0.0)
                i += 1
        
        # Compute reference values 
        for i, integral in enumerate(Integrals):
            for j in range(Species):
                for k, column in enumerate(col): 
                    
                    global_i = i * Species * len(col) + \
                        j * len(col) + k + 1
                        
                    if global_i % size != rank:
                        continue    
                    
                    ref[i][j][k] = integral(column, species = j, continuous = True)
                    
                    if pb and rank == 0:
                        pbar.update(global_i)
                        
        if pb and rank == 0:
            pbar.finish()
            
        ref = MPI.COMM_WORLD.allreduce(ref, ref)  
            
    # Discrete solution storage and cost function definition
    # Automatically sums costs over species
    disc = np.zeros_like(ref)
    disc_thin = np.zeros_like(limit_small_tau)
    def f(E, F):    
        for i, integral in enumerate(Integrals):
            for j in range(Species):
                disc_thin[i][j] = integral([0] * 3, E = E, F_E = F, species = j, continuous = False)

                if MinMethod != [0]: 
                    for k, column in enumerate(col): 
                        disc[i][j][k] = integral(column, E = E, F_E = F, species = j, continuous = False)
            
        cost = 0
        for method in MinMethod: 
            for i, integral in enumerate(Integrals):
                            
                # Add max (absolute) deviation in log-space between disrete and continuous solutions in optically thin limit
                if method == 0: 
                    cost += np.sum(np.abs(np.log10(limit_small_tau[i][0:]) - np.log10(disc_thin[i][0:])))
                                                                                                                        
                # Add mean (absolute) deviation in log-space between disrete and continuous solutions for tau > 0
                if method == 1:
                    cost += np.mean(np.abs(np.log10(ref[i][0:][0:]) - np.log10(disc[i][0:][0:])))
                                                                                         
                # Add max (absolute) deviation in log-space between disrete and continuous solutions for tau > 0
                if method == 2: 
                    cost2arr = np.abs(np.log10(ref[i][0:][0:]) - np.log10(disc[i][0:][0:]))
                    cost += np.max(cost2arr)
                 
                if method == 3:
                    cost2arr = np.abs(ref[i][0][0:] - disc[i][0][0:]) / ref[i][0][0:]
                    cost += np.max(cost2arr)                        

        return cost
                        
    # Prepare output directory.
    # See if file exists.  If Clobber == False, change suffix and write new file.
    fn = '%s/%s.h5' % (pf["OutputDirectory"], pf["ResultsFileName"])
    if os.path.exists(fn):
        if not pf['ClobberPreviousResults']:                    
            for i, element in enumerate(pf["ResultsFileName"]):
                if element.isdigit():
                    break
                    
            if not element.isdigit():
                raise IOError('%s exists.  Set ClobberPreviousResults = 1 to overwrite.' % fn)        
                    
            ct = int(pf["ResultsFileName"][i:]) + 1
            basename = pf["ResultsFileName"][0:i]
            while os.path.exists(fn):
                fn = '%s/%s%s.h5' % (pf["OutputDirectory"], basename, ct)
                ct += 1
                    
    if size > 1:
        MPI.COMM_WORLD.barrier()                
    
    # Run minimization                
    E_initial_guesses, F_initial_guesses, E_final_guesses, F_final_guesses, cost = anneal.minimize(f, fn)            
                     
    # Write data.     
    if rank == 0:                        
        result = h5py.File(fn, 'w')
        
        # Store data
        result.create_dataset('E_initial_guesses', data = E_initial_guesses)
        result.create_dataset('F_initial_guesses', data = F_initial_guesses)
        result.create_dataset('E_final_guesses', data = E_final_guesses)
        result.create_dataset('F_final_guesses', data = F_final_guesses)
        result.create_dataset('cost', data = cost)
        
        # Store parameter file
        pfgrp = result.create_group('parameters')
        for key in pf.keys():
            pfgrp.create_dataset(key, data = pf[key])
        
        # These are not actually parameters -- we compute them automatically
        pfgrp.create_dataset('HIColumnMin', data = HIColumnMin)
        pfgrp.create_dataset('HeIColumnMin', data = HeIColumnMin)
                    
        result.close()
                
    del E_initial_guesses, F_initial_guesses, E_final_guesses, F_final_guesses, cost
        
    if size > 1:
        MPI.COMM_WORLD.barrier()
        
end = time.time()

print("Run time: {0} minutes".format(round((end - start) / 60, 2)))

