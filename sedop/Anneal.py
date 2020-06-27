"""
Anneal.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-02-10.

Description: Simulated annealing baby.
 
"""

import numpy as np
import copy, h5py, os
from .ProgressBar import ProgressBar

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

class Anneal(object):
    def __init__(self, pf, src):
        self.pf = pf
        self.src = src
        self.Nbins = int(pf['num_bins'])
                  
        self.Gamma = pf['anneal_gamma']
        self.afreq = self.Nbins * pf['anneal_freq']
        self.maxEstep = pf["anneal_Estep"]
        self.maxFstep = pf["anneal_Fstep"]
                        
        # Set up initial guesses for bin placement and normalization
        # (may be overriden later but that's OK)
        self.E = np.linspace(self.src.Emin, self.src.Emax, self.Nbins)
        #self.F = self.src.Spectrum(self.E) / np.sum(self.src.Spectrum(self.E))
        self.F = np.ones(self.Nbins) / self.Nbins
                       
    def initial_guess(self):
        """
        For each new random walk, we'll start with a different initial guess.  
        We'll divide the spectrum into self.Nbins sections, and put our initial 
        guess for each energy bin in one of those segments.
        """
        
        # If we have a multi-frequency spectrum
        if self.Nbins > 1:
            
            # Generate random energy bins
            E = np.zeros(self.Nbins)
            for i in range(self.Nbins): 
                if self.pf['guess_log']:
                    E[i] = 10**(np.random.rand() * (np.log10(self.src.Emax) - np.log10(self.src.Emin)) \
                        + np.log10(self.src.Emin))
                else:
                    E[i] = np.random.rand() * (self.src.Emax - self.src.Emin) \
                        + self.src.Emin         
                                    
            # Normalization split equally to start
            F = np.ones(self.Nbins) / self.Nbins
                        
        # If we have a single energy bin, this is easy
        else:
            if self.pf['guess_log']:
                E = np.array([10**(np.random.rand() * (np.log10(self.src.Emax) - np.log10(self.src.Emin)) \
                    + np.log10(self.src.Emin))])
            else:
                E = np.array([np.random.rand() * (self.src.Emax - self.src.Emin) \
                    + self.src.Emin])
                    
            F = np.array([np.random.rand()])
        
        return E, F
    
    def guess(self, E, F, loc, max_dE, max_dF):
        """
        Returns new guess for E, F given last E, F and loc.
        """
                
        # If loc is less than Nbins, we should vary bin placement
        if loc < self.Nbins: 
            
            E_llim = self.src.Emin
            E_ulim = self.src.Emax
            
            # Generate new guess
            if self.pf["guess_gaussian"]:
                E[loc] = max(min(np.random.normal(E[loc], max_dE), E_ulim), E_llim)
            else:
                
                # Restrict guess to be within self.minEstep
                E_llim = max(E_llim, E[loc] - max_dE)
                E_ulim = min(E_ulim, E[loc] + max_dE)    
                        
                E[loc] = np.random.rand() * (E_ulim - E_llim) + E_llim
                                    
        # Otherwise, vary bin normalization    
        else: 
            
            # For multi-frequency case
            if self.Nbins > 1:
                
                # Change must be smaller than the limit we've set (self.maxFstep)
                F_llim = max(0.0, F[loc - self.Nbins] - max_dF)
                F_ulim = min(1.0, F[loc - self.Nbins] + max_dF)
                
                if self.pf["guess_gaussian"]:
                    F[loc - self.Nbins] = max(min(np.random.normal(F[loc - self.Nbins], max_dF), 1), 0)
                elif self.pf["anneal_logFstep"] and (F[loc - self.Nbins] <= 0.01):
                    F_llim = max(self.pf["anneal_logFmin"], 
                        np.log10(F[loc - self.Nbins]) + self.pf["anneal_logFstep_max"])
                    F_ulim = min(-1.3, np.log10(F[loc - self.Nbins]) - self.pf["anneal_logFstep_max"])    
                   
                    F[loc - self.Nbins] = 10**(np.random.rand() * (F_ulim - F_llim) + F_llim)
                else:
                    F[loc - self.Nbins] = np.random.rand() * (F_ulim - F_llim) + F_llim
                      
                # Always renormalize if somehow we exceed 100% of the bolometric luminosity
                if np.sum(F) > 1: 
                    F = F / np.sum(F)
            
            # For 1-bin case    
            else: 
                F[0] = np.random.rand()                
                                        
        return E, F
        
    def minimize(self, f, nsteps=10000, ntrials=5):
        """
        Minimize function f using Monte-Carlo ("Simulated Annealing") technique.
        """    
                           
        E_initial_guesses = []
        F_initial_guesses = []
        E_final_guesses = []
        F_final_guesses = []            
        temperature = [] 
        cost = []   
        
        best_trial = f(self.E, self.F) 
        
        E = self.E
        F = self.F
        
        # For each random walk
        walks = []
        for i in np.arange(ntrials):
            
            # If this walk belongs to another proc, move on
            if i % size != rank: 
                continue
            
            maxEstep = self.maxEstep
            maxFstep = self.maxFstep
            
            # Store guess history for each walk if TrackWalks > 0
            #if self.pf['TrackWalks']:
            #    basename = fn.rstrip('.hdf5')
            #    newfn = '%s_%s%s.hdf5' % (basename, self.pf["ProcessorDumpName"], rank)
            #                        
            #    if os.path.exists(newfn) and i == rank:
            #        if not self.pf["ClobberPreviousResults"]:
            #            raise IOError('%s exists.  Set ClobberPreviousResults = 1 to overwrite.' % newfn)
            #        else:
            #            os.system('rm -f %s' % newfn)    
            #        
            #    history = h5py.File(newfn, 'a')                                    
        
            # Generate initial guesses for E, F -- or use best solutions from previous walk
            if self.pf["guess_memory"] and i > 0:
                pass
            else:    
                E, F = self.initial_guess()
                
            last_cost = f(E, F)                            
                                    
            # Append initial guesses to initial guesses lists        
            E_initial_guesses.append(E)
            F_initial_guesses.append(F)        
                    
            # Set up guess history for this walk
            E_guess_history = np.zeros([nsteps, self.Nbins], float)
            F_guess_history = np.zeros([nsteps, self.Nbins], float)
            cost_history = np.zeros(nsteps, float)
            acceptance_history = np.zeros([nsteps, self.Nbins * 2], int)
            E_guess_history[0] = E
            F_guess_history[0] = F
            cost_history[0] = last_cost
            acceptance_history[0][0:self.Nbins] = 0
            
            # Initial value for temperature
            T = last_cost * self.pf["anneal_T0"]
            Tnot = copy.deepcopy(T)
            temperature = [T]        
                        
            E_best = self.E
            F_best = self.F                       
                                                                    
            if rank == 0:
                print("Random walk #{0}...".format(i + 1))
                
            pbar = ProgressBar(nsteps)
            pbar.start()    
            
            best_walk = 1 * last_cost    # is this a good idea?    
            for j in range(nsteps):
                
                # Progress bar
                pbar.update(float(j))
                
                # Which parameter are we varying?
                loc = int(np.random.rand() * 2 * self.Nbins)

                # Calculate probability
                E, F = self.guess(E, F, loc, maxEstep, maxFstep)
                next_cost = f(E, F)
                p = np.exp(-1. * (next_cost - last_cost) / T)
                                    
                # (Possibly) alter annealing schedule (only matters if our guess got worse)
                # Decreasing T faster means we are less likely to explore worse guesses
                if (j % self.afreq == 0) and (j != 0):                         
                    T = self.Gamma * T
                    temperature.append(T)
                    
                # Sort elements in E and F for convenient storage
                newE = []
                newF = []
                tmpE = list(E)
                tmpF = list(F)
                while True:    
                    pos = tmpE.index(min(tmpE))
                    newE.append(tmpE[pos])
                    newF.append(tmpF[pos])
                    tmpE.pop(pos)
                    tmpF.pop(pos)
                    
                    if len(newE) == self.Nbins: 
                        break
                    
                E = np.array(newE)
                F = np.array(newF)
                # If we made a good guess, compare future progress to this result
                acceptance_history[j][0:self.Nbins] = 0
                if np.random.rand() < p or j == 0: 
                    last_cost = next_cost
                    E_guess_history[j] = E
                    F_guess_history[j] = F  
                    cost_history[j] = next_cost
                    acceptance_history[j][loc] = 2
                else:
                    E = np.array(E_guess_history[j - 1])
                    F = np.array(F_guess_history[j - 1])
                    E_guess_history[j] = E_guess_history[j - 1]
                    F_guess_history[j] = F_guess_history[j - 1]
                    cost_history[j] = cost_history[j - 1]
                    acceptance_history[j][loc] = 1    
                                                
                # Best minimum obtained in this random walk
                best_walk = np.minimum(last_cost, best_walk)
                                                                                      
                # Track best E, F values obtained so far over all trials                                    
                if np.all(last_cost - best_trial) < 0:
                    E_best = E
                    F_best = F                      

            pbar.finish()
                                                                                            
            E_final_guesses.append(E_guess_history[j])
            F_final_guesses.append(F_guess_history[j])
            cost.append(last_cost)
            
            walks.append((E_guess_history, F_guess_history))


        # This could all be post-processed...but let's do it here.
        E_initial_guesses = MPI.COMM_WORLD.allreduce(E_initial_guesses, E_initial_guesses)
        F_initial_guesses = MPI.COMM_WORLD.allreduce(F_initial_guesses, F_initial_guesses)
        E_final_guesses = MPI.COMM_WORLD.allreduce(E_final_guesses, E_final_guesses)
        F_final_guesses = MPI.COMM_WORLD.allreduce(F_final_guesses, F_final_guesses)
        cost = MPI.COMM_WORLD.allreduce(cost, cost)    
                                
        E_initial_guesses = np.array(list(zip(*E_initial_guesses)))
        F_initial_guesses = np.array(list(zip(*F_initial_guesses)))
        E_final_guesses   = np.array(list(zip(*E_final_guesses)))
        F_final_guesses   = np.array(list(zip(*F_final_guesses)))
        
        results = {'Ei': E_initial_guesses, 'Ef': E_final_guesses,
            'Fi': F_initial_guesses, 'Ff': F_final_guesses, 
            'cost': np.array(cost), 'walks': np.array(walks).swapaxes(0, 3)}


        return results
        
    
