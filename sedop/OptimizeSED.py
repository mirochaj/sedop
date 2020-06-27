"""
annealer.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-01-28.

Description: Driver script for a code to find the optimal discrete spectrum for a continuous 
spectrum of interest, over a region in column density space of interest.
     
"""

import re
import os
import sys
import h5py
import time
import shutil
import numpy as np
from .Anneal import Anneal
from .ProgressBar import ProgressBar
from .RateIntegrals import RateIntegrals
from .RadiationSource import RadiationSource
from .SetDefaultParameterValues import SetDefaultParameterValues

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1       

class OptimizeSED(object):
    def __init__(self, **kwargs):
        self.pf = SetDefaultParameterValues()
        
        for kwarg in kwargs:
            if kwarg not in self.pf:
                print("WARNING: Unrecognized parameter `{}`.".format(kwarg))
                continue
            
            self.pf[kwarg] = kwargs[kwarg]
            
    @property
    def src(self):
        if not hasattr(self, '_src'):
            self._src = RadiationSource(self.pf)
        return self._src
            
    def _prep_output_dir(self, output_dir):
        
        # The reason this looks really gross to accomodate the general case 
        # where OutputDirectory is more than one level away from the CWD.
        
        made = False
        if rank == 0:
            output = output_dir
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
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
        
        # Make all processors wait here until the directory hierarchy 
        # has been built.
        if size > 1:
            MPI.COMM_WORLD.barrier()
        
    def run(self, prefix, nsteps=10000, ntrials=1, clobber=False,
        output_dir='.', restart=False):
        """
        Run simulated annealing `ntrials` times for `nsteps` each to find
        optimal discrete representation of some SED.
        
        Parameters
        ----------
        prefix : str
            Output filename prefix.
        output_dir : str
            Output directory.
        clobber : bool
            Overwrite previous result with same filename if it exists?
        nsteps : int
            Number of steps to take in each independent trial.
        ntrials : int
            Number of trials to run.
        restart : bool
            If True, will simply augment pre-existing data outputs with 
            more trials. Assumes all other parameters are identical.
        
        Returns
        -------
        A dictionary containing:
        

        """
        
        # Prepare output directory.
        fn = '%s/%s.hdf5' % (output_dir, prefix)

        if os.path.exists(fn) and (not clobber) and (not restart):
            raise IOError("File {} exists! Delete or re-run with clobber=True to overwrite.".format(fn))
        elif restart and (not os.path.exists(fn)):
            raise IOError("File {} not found, though restart=True.".format(fn))
        elif restart:
            assert (not clobber), "If restart=True, set clobber=False."

        nsteps = int(nsteps)
        ntrials = int(ntrials)
    
        start = time.time()

        ##
        # Print some info about the run.
        if rank == 0: 
            s = 'Will run {} trials with {} steps each.'.format(ntrials, nsteps)
            print("Starting Monte-Carlo simulations. {}".format(s))
        
        # Grab parameter file.
        pf = self.pf
        
        # Initialize classes that do all the heavy lifting
        ri = RateIntegrals(pf)
        anneal = Anneal(pf, self.src)
        
        # Set up directory hierarchy now rather than later - won't delete 
        # anything so be careful.
        self._prep_output_dir(output_dir)

        # Determine which integrals we're doing the optimization for
        Integrals = []
        if type(pf['rate_integral']) is not list: 
            pf['rate_integral'] = [pf['rate_integral']]
        for integral in pf['rate_integral']:
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
        if type(pf['method']) is not list: 
            MinMethod = [pf['method']]
        else:
            MinMethod = pf['method']

        # Some shortcuts
        MultiSpecies = int(pf['multispecies'])
        # Total number of column points
        N = pf['NHInum'] * pf['NHeInum']

        if MultiSpecies and N == pf['NHInum']:
            raise Exception('multispecies > 0 but no helium columns specified! Exiting.')

        SourceMinEnergy = pf['spectrum_Emin']
        if type(SourceMinEnergy) is not float:
            SourceMinEnergy = min(SourceMinEnergy)

        # The photo-ionization and heating integrals will be identical for all 
        # species if the lower energy cutoff in the spectrum is above all 
        # ionization thresholds. Account for this.    
        E_threshold = np.array([13.6, 24.6, 54.4])
        if np.all(SourceMinEnergy > E_threshold[1:]) or (pf['species'] == 1):
            Species = 1 # equiv. to saying Phi_HI = Phi_HeI = Phi_HeII and same for Psi
        else:
            Species = 2 # hydrogen and helium
                        
        # Determine where this source first becomes optically thin (with 
        # decreasing n_col)
        ncol_temp = np.logspace(15, 22, 71)
        
        if rank == 0:
            print("Isolating optically thin limit...")
            
        pbar = ProgressBar(len(Integrals) * (MultiSpecies + 1) * len(ncol_temp))
        pbar.start()
            
        # Value of integral from N = 3 * [0]    
        limit_small_tau = np.zeros([len(Integrals), Species])
        # Where in N does tau become > 0?
        small_tau_trans = np.zeros([len(Integrals), Species, 71]) 
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
                        limit_small_tau[i][j] = integral([0] * 3, 
                            species=j, continuous=True)
                                    
                    cols = np.array([column] * 3)
                    small_tau_trans[i][j][k] = integral(np.multiply(cols, mask), 
                        species=j, continuous=True)                                
              
                    pbar.update(global_i)
                        
        pbar.finish()                
              
        limit_small_tau = \
            MPI.COMM_WORLD.allreduce(limit_small_tau, limit_small_tau)
        small_tau_trans = \
            MPI.COMM_WORLD.allreduce(small_tau_trans, small_tau_trans)
              
        # Compare integral values to their optically thin limits
        # [Gamma [H, He], Heat [H, He]]
        ncol_trans = 1e15 * np.ones([len(Integrals), 2])
        for i, integral in enumerate(Integrals):
            for j in range(Species):

                k = 0
                while small_tau_trans[i][j][k] >= 0.999 * limit_small_tau[i][j]: 
                    k += 1

                ncol_trans[i][j] = (ncol_temp[k] + ncol_temp[k + 1]) / 2.

        # Set lower bounds of integration based on optically thin transition
        HIColumnMin = min(list(zip(*ncol_trans))[0])
        HIColumnMax = pf["NHImax"]
        HeIColumnMin = min(list(zip(*ncol_trans))[1])
        HeIColumnMax = pf["NHeImax"]
        
        if rank == 0:
            s = '{:.2e} - {:.2e} cm^2.'.format(HIColumnMin, HIColumnMax)
            print("Minimization for hydrogen between N_HI = {}".format(s))
            if MultiSpecies:
                s = '{:.2e} - {:.2e} cm^2.'.format(HeIColumnMin, HeIColumnMax)
                print("Minimization for helium between N_HeI = {}".format(s))

            # Construct reference array ("correct" values for RateIntegrals)
            print('Constructing continuous \'reference\' integrals...')

        colHI = np.logspace(np.log10(HIColumnMin), np.log10(HIColumnMax), pf['NHInum'])
        if not MultiSpecies:
            colHeI = np.array([0])
        else:
            colHeI = np.logspace(np.log10(HeIColumnMin), np.log10(HeIColumnMax), pf['NHeInum'])

        pbar = ProgressBar(maxval=len(Integrals) * 2 * N)
        pbar.start()

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

                        ref[i][j][k] = integral(column, species=j, 
                            continuous=True)

                        pbar.update(global_i)

            pbar.finish()

            ref = MPI.COMM_WORLD.allreduce(ref, ref)  
                
        # Discrete solution storage and cost function definition
        # Automatically sums costs over species
        disc = np.zeros_like(ref)
        disc_thin = np.zeros_like(limit_small_tau)
        def f(E, F):    
            for i, integral in enumerate(Integrals):
                for j in range(Species):
                    disc_thin[i][j] = integral([0] * 3, E=E, F_E=F, 
                        species=j, continuous=False)
        
                    if MinMethod != [0]: 
                        for k, column in enumerate(col): 
                            disc[i][j][k] = integral(column, E=E, F_E=F, 
                                species=j, continuous=False)
                
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

        if size > 1:
            MPI.COMM_WORLD.barrier()

        ##
        # Run minimization
        results = anneal.minimize(f, nsteps, ntrials)

        # Re-organize the walks
        num = self.pf['num_bins']
        Esteps = np.zeros((num, ntrials, nsteps))
        Fsteps = np.zeros((num, ntrials, nsteps))
        for i in range(ntrials):
            Esteps[:,i,:] = results['walks'][:,0,:,i]
            Fsteps[:,i,:]  = results['walks'][:,1,:,i]

        results['Esteps'] = Esteps
        results['Fsteps'] = Fsteps
        del results['walks']

        # Write data.
        if rank == 0:
            if restart:
                results_prev = {}
                f = h5py.File(fn, 'r')
                results_prev['Ei'] = np.array(f[('Ei')])
                results_prev['Fi'] = np.array(f[('Fi')])
                results_prev['Ef'] = np.array(f[('Ef')])
                results_prev['Ff'] = np.array(f[('Ff')])
                results_prev['cost'] = np.array(f[('cost')])
                results_prev['Esteps'] = np.array(f[('Esteps')])
                results_prev['Fsteps'] = np.array(f[('Fsteps')])

                f.close()
                shutil.copy(fn, fn+'.copy')

                results_new = {}
                for key in results:
                    results_new[key] = \
                        np.hstack((results_prev[key], results[key]))

                print("Stacked new results with previous run.")
                results = results_new
                    
            # Store data
            f = h5py.File(fn, 'w')
            f.create_dataset('Ei', data=results['Ei'])
            f.create_dataset('Fi', data=results['Fi'])
            f.create_dataset('Ef', data=results['Ef'])
            f.create_dataset('Ff', data=results['Fi'])
            f.create_dataset('cost', data=results['cost'])
            f.create_dataset('Esteps', data=results['Esteps'])
            f.create_dataset('Fsteps', data=results['Fsteps'])
            
            # Store parameter file
            pfgrp = f.create_group('parameters')
            for key in pf.keys():
                pfgrp.create_dataset(key, data=pf[key])
            
            # These are not actually parameters -- we compute them automatically
            pfgrp.create_dataset('NHI_min', data=HIColumnMin)
            pfgrp.create_dataset('NHeI_min', data=HeIColumnMin)
            
            f.close()
                        
            print("Wrote {}.".format(fn))
            
            # Clean-up
            if restart:
                os.remove(fn+'.copy')

        if size > 1:
            MPI.COMM_WORLD.barrier()

        end = time.time()

        if rank == 0:
            print("Run time: {0} minutes".format(round((end - start) / 60, 2)))
            print("Results available in `results` attribute for safe-keeping.")
        
        self.results = results
        
        return self.results

