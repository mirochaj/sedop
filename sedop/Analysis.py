"""
Analysis.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Oct 31 09:31:51 2011

Description: Functions to calculate various quantities from our sedop datasets.

"""

import h5py
import numpy as np
import pylab as pl
import itertools as it
import matplotlib.gridspec as gridspec

try: 
    from scipy.integrate import simps as simpson
except ImportError: 
    print('Need scipy to compute integrals.')

h = 6.626068 * 1e-27 			# Planck's constant - [h] = erg*s
c = 29979245800.0 				# Speed of light - [c] = cm / s
k_B = 1.3806503 * 1e-16			# Boltzmann's constant - [k_B] = erg / K
erg_per_ev = 1.60217646e-12

colors = ['k', 'blue', 'red', 'green', 'cyan', 'magenta', 'yellow']
symbols = ['o', 's', '^', '+', 'x']

class Analyze(object):
    def __init__(self, prefix, pf=None):
        """
        Initialize our analysis environment, read in data.
            pf = Parameter file (either a dictionary or a text file)
            rdir = Results directory (if supplied, will read in all .h5 files)
            merge = List of parameter files to merge together
            load_walks = Load random walk histories?
        """
        
        self.prefix = prefix
        self.pf = pf
        
        #self.ds = Dataset(pf, load_walks = load_walks, merge = merge, read_all = read_all)
        #self.pf = self.ds.pf
        #self.results = self.ds.results
        #
        ## Load random walks (all steps of each random walk, actually)
        #if load_walks and self.pf['TrackWalks']: 
        #    self.walks = self.ds.walks
        #
        ## Initialize classes we might need
        #self.rs = sm.RadiationSource(self.pf)
        #self.ri = sm.RateIntegrals(self.pf)
        #
        ## Simulation parameters
        #self.N = int(self.pf['NumberOfBins'])
        #self.Nw = int(self.pf['NumberOfWalks'])
        #
        ## Make lists of the species and rate integrals used for this calculation
        #if type(self.pf["Species"]) is not list: 
        #    self.Species = [int(self.pf["Species"])]    
        #else: 
        #    self.Species = []
        #    for element in self.pf["Species"]: 
        #        self.Species.append(int(element))
        #   
        #self.Species = [0, 1]                    
        #self.RateIntegral = [0, 1]
        #        
        ## Set up column densities and integral values
        #self.HIColumnMin = self.pf["HIColumnMin"]
        #self.HIColumnMax = self.pf["HIColumnMax"]
        #self.HeIColumnMin = self.pf["HeIColumnMin"]
        #self.HeIColumnMax = self.pf["HeIColumnMax"]
        #self.HINumberOfColumns = self.pf["HINumberOfColumns"]
        #self.HeINumberOfColumns = self.pf["HeINumberOfColumns"]
        #        
        #self.integral = []
        #for element in self.RateIntegral:
        #    if element == 0: 
        #        self.integral.append(self.ri.Phi)
        #    if element == 1: 
        #        self.integral.append(self.ri.Psi)
        #
        ## Here are the bins we actually used to evaluate the discrete spectra 
        #self.colHI = np.logspace(np.log10(self.HIColumnMin), np.log10(self.HIColumnMax), self.HINumberOfColumns)
        #self.colHeI = np.logspace(np.log10(self.HeIColumnMin), np.log10(self.HeIColumnMax), self.HeINumberOfColumns)
        #
        ## Calculate a few things automatically - start with stats for E/F solutions
        #self.stats()                    
        #
        ## Compute continuous integrals for this source, and 'best' integral solutions
        #self.residuals = {}
        #self.integral_solutions = {}    
        #if auto_compute:
        #    self.standard_integrals()  
            
    @property
    def data(self):
        if not hasattr(self, '_data'):
            
            if type(self.prefix) is str:
                results = {}
                f = h5py.File(self.prefix+'.hdf5', 'r')
                results['Ei']   = np.array(f[('E_initial_guesses')])
                results['Ef']   = np.array(f[('E_final_guesses')])
                results['Fi']   = np.array(f[('F_initial_guesses')])
                results['Ff']   = np.array(f[('F_final_guesses')])
                results['cost'] = np.array(f[('cost')])
        
                # Take parameter file from results output
                pf = {}
                for key in f['parameters'].keys():
                    if type(f['parameters'][key].value) is np.ndarray:
                        pf[key] = list(f['parameters'][key].value)
                    else:
                        pf[key] = f['parameters'][key].value    
        
                f.close()
                
                self._data = results
                self._pf = pf
            else:
                self._data = self.prefix
                
        return self._data        
        
    def standard_integrals(self):
        """
        Compute standard integrals -- continuous heating + ionization, and best solutions.
        """    
        
        # Compute some integrals - both continuous and discrete, and residuals
        
        for i in self.Species:
            for j in self.RateIntegral:
                self.integral_solutions['sp%iint%i_c' % (i, j)] = self.compute_integral(intnum = j)
                self.integral_solutions['sp%iint%i_best' % (i, j)] = self.compute_integral(E = self.best['E'], 
                    F = self.best['F'], continuous = False, intnum = j)
                self.residuals['sp%iint%i_log' % (i, j)] = \
                    np.log10(self.integral_solutions['sp%iint%i_best' % (i, j)]['integral']) - \
                    np.log10(self.integral_solutions['sp%iint%i_c' % (i, j)]['integral'])
                self.residuals['sp%iint%i_rel' % (i, j)] = \
                    (self.integral_solutions['sp%iint%i_best' % (i, j)]['integral'] - \
                    self.integral_solutions['sp%iint%i_c' % (i, j)]['integral']) / \
                    self.integral_solutions['sp%iint%i_c' % (i, j)]['integral']
                                        
    def compute_integral(self, nHI=None, nHeI=1.0, nHeII=1.0, species=0, E=None, 
        F=None, linsp=0, continuous=True, intnum=0, npoints=200, save='save1'): 
        """
        Compute the value of the continuous/discrete solutions.  By default, use 200 data points.
        Note, only one of nHI, nHeI, or nHeII can be None - that is the dimension sliced over.
        linsp > 0 if we want that many linearly spaced bins, keeping normalizations consistent with spectrum
        """
        
        columns = []
        
        if linsp > 0:
            E = np.linspace(self.pf['SourceMinEnergy'], self.pf['SourceMaxEnergy'], linsp)
            F = self.rs.SpecificIntensity(E) / np.sum(self.rs.SpecificIntensity(E)) 
        else:    
            E = np.array(E)
            F = np.array(F)
        
        # We need to slice through column density space - case 1, slices through HeI and HeII
        if nHI == None:
            self.species_slice = 0
            for col in np.logspace(np.log10(self.HIColumnMin), np.log10(self.HIColumnMax), npoints): 
                columns.append([col, nHeI, nHeII])
            
        elif nHeI == None:   
            self.species_slice = 1
            for col in np.logspace(np.log10(self.HeIColumnMin), np.log10(self.HeIColumnMax), npoints): 
                columns.append([nHI, col, nHeII])
        
        elif nHeII == None:   
            self.species_slice = 2
            for col in np.logspace(np.log10(self.HeIIColumnMin), np.log10(self.HeIIColumnMax), npoints): 
                columns.append([nHI, nHeI, col])
        
        else:
            raise ValueError("I have no idea what you're trying to do.")

        # Compute integral value as function of column density
        solution = np.zeros(npoints)
        for i, col in enumerate(columns): 
            solution[i] = self.integral[intnum](col, E = E, F_E = F, species = species, continuous = continuous)
                
        # Return solution, nHI, nHeI, and nhEII        
        if self.species_slice == 0:
            self.nHI = np.array(zip(*columns)[0])
            self.nHeI = nHeI
            self.nHeII = nHeII
        elif self.species_slice == 1:
            self.nHI = nHI
            self.nHeI = np.array(zip(*columns)[1])
            self.nHeII = nHeII    
        elif self.species_slice == 2:
            self.nHI = nHI
            self.nHeI = nHeI
            self.nHeII = np.array(zip(*columns)[2])
            
        if save is not None:
            self.integral_solutions[save] = {'nHI': self.nHI, 'nHeI': self.nHeI, 'nHeII': self.nHeII, 'integral': solution, 'E': E, 'F': F}
        
        return {'nHI': self.nHI, 'nHeI': self.nHeI, 'nHeII': self.nHeII, 'integral': solution, 'E': E, 'F': F}
        
    def stats(self, Ebins=50., Fbins=50, weights=False, normed=True, exclude=0, 
        logbins=False):
        """
        Calculate the average, mode, and rms spread for each energy bin.
        """
        
        # Weight PDFs by value of cost function or not?
        if weights: 
            weights = 1. / self.results['cost']
        else: 
            weights = None
        
        shape = (self.N, int(self.pf['NumberOfSteps']))

        # Initialize dictionaries for results
        self.pdf = {'E': np.zeros(self.N, np.ndarray), 'F': np.zeros(self.N, np.ndarray)}
        self.cdf = {'E': np.zeros(self.N, np.ndarray), 'F': np.zeros(self.N, np.ndarray)}
        self.bins = {'E': np.zeros(self.N, np.ndarray), 'F': np.zeros(self.N, np.ndarray)}
        self.mode = {'E': np.zeros(self.N), 'F': np.zeros(self.N)}
        self.median = {'E': np.zeros(self.N), 'F': np.zeros(self.N)}
        self.rms = {'E': np.zeros(self.N), 'F': np.zeros(self.N)}
        self.avg = {'E': np.zeros(self.N), 'F': np.zeros(self.N)}
        self.best = {'E': np.zeros(self.N), 'F': np.zeros(self.N)}
        self.sigma = {'E': np.zeros(self.N, np.ndarray), 'F': np.zeros(self.N, np.ndarray)}
        
        if self.pf['InitialGuessMemory']:
            
            if type(Ebins) in [float, np.float64]:
                logEmin = np.log10(self.rs.Emin)
                logEmax = np.log10(self.rs.Emax)
                if logbins or (logEmax - logEmin > 1):
                    Ebins = 10**np.arange(round(logEmin), logEmax+Ebins, Ebins)
                else:
                    Ebins = np.arange(round(self.rs.Emin), self.rs.Emax+Ebins, 
                        Ebins)
                                        
            if type(Fbins) in [float, np.float64]:
                Fbins = np.arange(0, 1, Fbins)
            else:
                Fbins = np.linspace(0, 1, Fbins)
    
            Ebins, E, Fbins, F = self.walk_stats_1d(Ebins=Ebins, Fbins=Fbins, 
                exclude=exclude)
            
            for i in range(self.N):
                self.bins['E'][i] = Ebins
                self.bins['F'][i] = Fbins
                self.pdf['E'][i] = E[i]
                self.pdf['F'][i] = F[i]      
                                
                self.cdf['E'][i] = np.cumsum(self.pdf['E'][i]) / np.sum(self.pdf['E'][i])
                self.cdf['F'][i] = np.cumsum(self.pdf['F'][i]) / np.sum(self.pdf['F'][i])
                
                self.mode['E'][i] = self.bins['E'][i][list(self.pdf['E'][i]).index(max(self.pdf['E'][i]))]
                self.mode['F'][i] = self.bins['F'][i][list(self.pdf['F'][i]).index(max(self.pdf['F'][i]))]
                
                self.median['E'][i] = np.interp(0.5, self.cdf['E'][i], self.bins['E'][i])
                self.median['F'][i] = np.interp(0.5, self.cdf['F'][i], self.bins['F'][i])                
                
                self.sigma['E'][i] = [np.interp(0.16, self.cdf['E'][i], self.bins['E'][i]),
                    np.interp(0.84, self.cdf['E'][i], self.bins['E'][i])]
                self.sigma['F'][i] = [np.interp(0.16, self.cdf['F'][i], self.bins['F'][i]),
                    np.interp(0.84, self.cdf['F'][i], self.bins['F'][i])]   
                
                for j in range(self.Nw):
                    if j < exclude:
                        continue
                        
                    self.avg['E'][i] += np.mean(self.walks[j].E[i])
                    self.avg['F'][i] += np.mean(self.walks[j].F[i])
                    
                    self.rms['E'][i] = np.sqrt(np.mean((self.walks[j].E[i] - self.mode['E'][i])**2))
                    self.rms['E'][i] = np.sqrt(np.mean((self.walks[j].F[i] - self.mode['F'][i])**2))
                    
                self.avg['E'][i] /= (self.Nw - exclude)    
                self.avg['F'][i] /= (self.Nw - exclude)
                
            index = np.argmin(self.results['cost'])
            for i, binnum in enumerate(self.results['Ef']):
                self.best['E'][i] = self.results['Ef'][i][index]
                self.best['F'][i] = self.results['Ff'][i][index]
                
            self.cost = self.results['cost'][index]
                                
        else:    
        
            for i in range(self.N):
                pdf, bin_edges = np.histogram(self.results['Ef'][i], bins = Ebins, normed = False, weights = weights)
                            
                self.pdf['E'][i] = pdf
                if normed: 
                    self.pdf['E'][i] = self.pdf['E'][i] / float(np.sum(pdf))
                    
                self.cdf['E'][i] = np.cumsum(self.pdf['E'][i]) / np.sum(self.pdf['E'][i])    
                self.bins['E'][i] = rebin(bin_edges)
                self.mode['E'][i] = self.bins['E'][i][list(self.pdf['E'][i]).index(max(self.pdf['E'][i]))]
                self.avg['E'][i] = np.mean(self.results['Ef'][i])
                
                if not self.pf['FixNormalization']:
                    pdf, bin_edges = np.histogram(self.results['Ff'][i], bins = Fbins, normed = False, weights = weights)
                    self.pdf['F'][i] = pdf
                    if normed: 
                        self.pdf['F'][i] = self.pdf['F'][i] / float(np.sum(pdf))
                    
                    self.cdf['F'][i] = np.cumsum(self.pdf['F'][i]) / float(np.sum(self.pdf['F'][i]))
                    self.bins['F'][i] = rebin(bin_edges)                
                    self.mode['F'][i] = self.bins['F'][i][list(self.pdf['F'][i]).index(max(self.pdf['F'][i]))]
                    self.avg['F'][i] = np.mean(self.results['Ff'][i])     
                else:
                    self.pdf['F'][i] = self.rs.Spectrum(self.pdf['E'][i])
                    self.cdf['F'][i] = np.cumsum(self.pdf['F'][i]) / float(np.sum(self.pdf['F'][i]))
                    self.bins['F'][i] = rebin(bin_edges)
                    self.mode['F'][i] = self.rs.Spectrum(self.bins['E'][i][list(self.pdf['F'][i]).index(max(self.pdf['F'][i]))])  
                    self.avg['F'][i] = self.rs.Spectrum(np.mean(self.results['Ef'][i]))
                        
            # Compute 'best' solutions
            index = np.argmin(self.results['cost'])
            for i, binnum in enumerate(self.results['Ef']):
                self.best['E'][i] = self.results['Ef'][i][index]
                if self.pf['FixNormalization']:
                    self.best['F'] = self.rs.Spectrum(self.best['E']) / np.sum(self.rs.Spectrum(self.best['E']))
                else:
                    self.best['F'][i] = self.results['Ff'][i][index]                
            
            self.cost = self.results['cost'][index]
            
            # Compute RMS spread for each bin
            for i in range(self.N):
                self.rms['E'][i] = np.sqrt(np.mean((self.results['Ef'][i] - self.best['E'][i])**2))
                if not self.pf['FixNormalization']:
                    self.rms['F'][i] = np.sqrt(np.mean((self.results['Ff'][i] - self.best['F'][i])**2)) 
                else: 
                    self.rms['F'][i] = self.rs.Spectrum(np.sqrt(np.mean((self.results['Ff'][i] - self.best['F'][i])**2)))        
                         
            # Compute median
            for i in range(self.N): 
                self.median['E'][i] = self.bins['E'][i][np.argmin(np.abs(self.cdf['E'][i] - 0.5))]
                self.median['F'][i] = self.bins['F'][i][np.argmin(np.abs(self.cdf['F'][i] - 0.5))]
                         
            # Compute 68% regions for each bin
            for i in range(self.N):             
                E1 = np.argmin(np.abs(self.cdf['E'][i] - 0.16))
                E2 = np.argmin(np.abs(self.cdf['E'][i] - 0.84))
                self.sigma['E'][i] = [self.median['E'][i] - self.bins['E'][i][E1], self.bins['E'][i][E2] - self.median['E'][i]]
                
                F1 = np.argmin(np.abs(self.cdf['F'][i] - 0.16))
                F2 = np.argmin(np.abs(self.cdf['F'][i] - 0.84))
                self.sigma['F'][i] = [self.median['F'][i] - self.bins['F'][i][F1], self.bins['F'][i][F2] - self.median['F'][i]]
                    
    def compare(self, E=None, F=None, species=0, integral=0, nHI=None, 
        nHeI=1.0, nHeII=1.0, plot_components=True, annotate=True, color='b'):
        """
        Comparison of integrals.
        """    
        
        if E is None:
            E = self.best['E']
        if F is None:
            F = self.best['F']
            
        if species == 0:
            s = 'HI'
        elif species == 1:
            s = 'HeI'
        else:
            s = 'HeII'    
        
        continuous = self.compute_integral(nHI=nHI, nHeI=nHeI, nHeII=nHeII, 
            continuous=True, intnum=integral, species=species)
            
        discrete = self.compute_integral(nHI=nHI, nHeI=nHeI, nHeII=nHeII, 
            E=E, F=F, continuous=False, intnum=integral, species=species)    
            
        self.residuals['logerr'] = np.log10(discrete['integral']) - np.log10(continuous['integral'])
        self.residuals['err'] = (discrete['integral'] - continuous['integral']) / continuous['integral']
        
        if not nHI:
            x = continuous['nHI']
            xax = 'nHI'
            ann = r'$n_{\mathrm{HeI}} = 10^{%i} \ \mathrm{cm^{-2}}$' % np.log10(nHeI)
        elif not nHeI:
            x = continuous['nHeI']
            xax = 'nHeI'
            ann = r'$n_{\mathrm{HI}} = 10^{%i} \ \mathrm{cm^{-2}}$' % np.log10(nHI)
        else:
            x = continuous['nHeII']
            xax = 'nHeII'    
            
        # Figure out labels
        if xax == 'nHI':
            xlabel = r'$N_{\mathrm{HI}} \ (cm^{-2})$'
        elif xax == 'nHeI':
            xlabel = r'$N_{\mathrm{HeI}} \ (cm^{-2})$'    
        
        if integral == 0:
            ylabel = r'$\Phi_{\mathrm{%s}}$' % s
        elif integral == 1:
            ylabel = r'$\Psi_{\mathrm{%s}}$' % s
        
        self.gs = gridspec.GridSpec(2, 1, height_ratios = [3, 1], hspace = 0)
        
        self.ax1 = pl.subplot(self.gs[0])
        self.ax2 = pl.subplot(self.gs[1])
        
        # Continuous integral in black
        self.ax1.loglog(continuous[xax], continuous['integral'], color = 'k')

        # 'Best' solution in blue        
        mask = np.arange(0, len(discrete[xax]) - 1)
        mask[mask % 4 != 0] = 0
        mask[0] = 1
        mask[mask != 0] = 1
        self.mask = mask
        self.ax1.scatter(discrete[xax][mask == 1], discrete['integral'][mask == 1], color = color, marker = '+', s = 100)
        
        # Errors
        self.ax2.semilogx(discrete[xax], 100 * ((discrete['integral'] - continuous['integral']) / continuous['integral']), color = color)
        
        # Annotate
        #self.ax.annotate(ann, (continuous[xax][10], continuous['integral'][10]), va = 'bottom', ha = 'left')
        
        # Components
        if plot_components:
            if len(discrete['E']) > 1:
                for i, binnum in enumerate(discrete['E']):
                    results = self.compute_integral(E=[E[i]], F=[F[i]], 
                        species=species, continuous=False, intnum=integral, 
                        nHI=nHI, nHeI=nHeI, nHeII=nHeII)
                        
                    self.ax1.loglog(x, results['integral'], color = color, ls = '--')
                    
                    if annotate:
                        
                        self.ax1.annotate(r'$E = %g \ \mathrm{eV}$' % round(discrete['E'][i], 2), 
                            (results['nHI'][10], results['integral'][10]), 
                            va = 'top', ha = 'left')
        
        self.ax1.set_xscale('log')
        self.ax1.set_yscale('log')
        self.ax1.set_ylabel(ylabel)
        self.ax1.set_xticklabels([])
        self.ax1.set_xlim(min(continuous[xax]), max(continuous[xax]))
        self.ax1.set_ylim(0.5 * min(continuous['integral']), 2 * max(continuous['integral']))           
        
        self.ax2.set_xscale('log')
        self.ax2.set_yscale('linear')
        self.ax2.set_xlim(min(continuous[xax]), max(continuous[xax]))
        self.ax2.set_ylim(-10, 10)
        self.ax2.set_yticks([-10, -5, 0, 5, 10])
        self.ax2.set_ylabel(r'$\% \ \mathrm{Error}$')
        self.ax2.set_xlabel(xlabel)  
        pl.draw()     
        
    def walk_stats_1d(self, Ebins=None, Fbins=None, exclude=0, weights=False,
        log=False):
        """
        Histogram energy and normalization results (separately) for all bins from 
        all random walks with index > exclude.  Setting exclude = 1 effectively 
        removes the 'burn in' stage of the minimization.
        
            returns: Ebins, Ehists, Fbins, Fhists
            
        Should we bother with marginalization?    
        
        Weighting isn't quite right at the moment - run with weights = False.
        """           
        
        # Default bin edges
        if Ebins is None:
            Ebins = self.default_energy_bins(log=log)
        if Fbins is None:
            Fbins = self.default_norm_bins()
                
        Ehist = {}
        Fhist = {}
        for i in range(self.N):
            Ehist[i] = np.zeros(len(Ebins) - 1)
            Fhist[i] = np.zeros(len(Fbins) - 1)
            for j in range(self.Nw):            
                if j < exclude:
                    continue
                
                weight = None    
                if weights:    
                    weight = 1. / self.walks[j].cost
                    
                Ehist[i] += np.histogram(self.walks[j].E[i], bins = Ebins, normed = True, weights = weight)[0]
                Fhist[i] += np.histogram(self.walks[j].F[i], bins = Fbins, normed = True, weights = weight)[0]
                
        # Re-normalize after the fact - is this kosher?      
        for i in range(self.N):
            Ehist[i] /= float(np.sum(Ehist[i]))
            Fhist[i] /= float(np.sum(Fhist[i]))
                
        return rebin(Ebins), Ehist, rebin(Fbins), Fhist               
                      
    def walk_stats_2d(self, Ebins = None, Fbins = None, exclude = 0, normed = True, log = False,
        initial_step = 0):
        """
        2D histogram of energy and normalization results for all bins from all random walks 
        with index > exclude.  Setting exclude = 1 effectively removes the 
        'burn in' stage of the minimization.
        
            returns: Ebins, Ehists, Fbins, Fhists
            
        Should we bother with marginalization?    
        """                         
        
        # Default bin edges
        if Ebins is None:
            Ebins = self.default_energy_bins(log = log)
        if Fbins is None:
            Fbins = self.default_norm_bins()
                   
        hist = np.zeros([self.N, len(Fbins) - 1, len(Ebins) - 1])  
        for i in range(self.N):
            for j in range(self.Nw):
                
                if j < exclude:
                    continue
                        
                hist[i] += self.hist2d(walk = j, bin = i, bins = [Ebins, Fbins], 
                    normed = False, weights = None, initial_step = initial_step)[0]    
                
        return rebin(Ebins), rebin(Fbins), hist                
        
    def walk_plot_1d(self, bin=0, EorF='E', walk=0, color = None):
        """
        Plot evolution of EorF for given bin and walk.
        """
                              
        if color is None:                        
            color = colors[bin]                        
        
        self.ax = pl.subplot(111)
        
        i = int(np.random.rand() * len(self.walks))
        if EorF is 'E': 
            self.ax.plot(self.walks[walk].E[bin], color = color)
        if EorF is 'F': 
            self.ax.plot(self.walks[walk].F[bin], color = color)

        pl.draw()
        
    def walk_plot_2d(self, bin = 0, walk = 0, color = None, initial_step = 0):
        """
        Show random walk path in 2D space for given bin, walk, and initial_step.
        For example, setting initial_step = 5000 will result in a plot showing
        steps 5000 and up.
        """    

        if color is None:                        
            color = colors[bin]                        
                          
        self.ax = pl.subplot(111)
        
        self.ax.plot(self.walks[walk].E[bin][initial_step:], self.walks[walk].F[bin][initial_step:], color = color)
        self.ax.set_xlabel(r'$h\nu_{%i}$' % bin)
        self.ax.set_ylabel(r'$I_{\nu_{%i}}$' % bin)
        
        pl.draw()
        
    def contour(self, bin = 0, walk = 0, Ebins = None, Fbins = None, initial_step = 0, 
        normed = False, weights = None, exclude = None, log = False):
        """
        Contour plot of E vs. F. for given bin.  If exclude == None, the contour plot
        will be for all steps > initial_step in a single random walk (given by arg 'walk').
        Otherwise, if exclude is not None, will be a contour plot including all random
        walks with index >= exclude.
        """
        
        bins = [Ebins, Fbins]
        
        if exclude is not None:
            E, F, hist = self.walk_stats_2d(Ebins = Ebins, Fbins = Fbins, exclude = exclude, 
                normed = normed, log = log, initial_step = initial_step)
            hist = hist[bin] 
        else:    
            hist, F, E = self.hist2d(bin = bin, walk = walk, bins = bins, 
                normed = normed, weights = weights, initial_step = initial_step)
                          
        self.ax = pl.subplot(111)
        self.ax.contour(E, F, hist)
        self.ax.set_xlabel(r'$h\nu_{%i}$' % bin)
        self.ax.set_ylabel(r'$I_{\nu_{%i}}$' % bin)
        pl.draw()        
        
    def hist1d(self, EorF='E', annotate=None):
        """
        Histogram PDFs of best solutions.
        """
        
        if not hasattr(self, 'ax'):
            self.ax = pl.subplot(111)
        
        for i in range(self.N):
            self.ax.plot(self.bins[EorF][i], self.pdf[EorF][i], color = colors[i], ls = '-', drawstyle = 'steps-mid')
        
        if annotate is not None:
            if annotate == 'best':
                self.ax.plot(2 * [self.best[EorF][i]], [0, 2 * max(self.pdf[EorF][i])], ls = '--', color = colors[i])
            
            if annotate == 'mode':
                self.ax.plot(2 * [self.mode[EorF][i]], [0, 2 * max(self.pdf[EorF][i])], ls = '--', color = colors[i])
            
            if annotate == 'median':
                self.ax.plot(2 * [self.mode[EorF][i]], [0, 2 * max(self.pdf[EorF][i])], ls = '--', color = colors[i])
            
        self.ax.set_xlabel(EorF)
        self.ax.set_ylabel('PDF')    
            
        pl.draw()    
        
    def hist2d(self, bin=0, walk = 0, bins = None, initial_step = 0, normed = False, weights = None):
        """
        Compute the 2D histogram of E vs. F.
        
            returns: histogram, Fbins, Ebins.
        """
        
        if bins is None or [None, None]:
            bins = [np.linspace(self.rs.Emin, self.rs.Emax, 1 + (self.rs.Emax - self.rs.Emin) / 0.5), 
                np.linspace(0, 1, 101)]
        
        hist, xedges, yedges = np.histogram2d(self.walks[walk].E[bin][initial_step:], self.walks[walk].F[bin][initial_step:], 
            bins = bins, normed = normed, weights = weights)
        
        return np.transpose(hist), rebin(yedges), rebin(xedges)    
        
    def cost_plot(self, walk = 0):
        """
        Plot the value of the cost function vs. step in a random walk.
        Requires TrackWalks = 1.
        """
        
        self.ax = pl.subplot(111)
        self.ax.semilogy(self.walks[walk].cost)
        
        pl.draw()
        
    def L2norm(self, E = None, F = None, species = 0, integral = 0, 
        nHI = None, nHeI = 1.0, nHeII = 1.0, npoints = 100):
        """
        Compute L^2-norm for continuous vs. discrete quantities.
        Now, doing this with a discrete sum for the sake of speed.
        """    
        
        continuous = self.compute_integral(nHI = nHI, nHeI = nHeI, nHeII = nHeII, intnum = integral,
            continuous = True, npoints = npoints)
        
        if not E:
            E = self.best['E']
        if not F:
            F = self.best['F']
        
        discrete = self.compute_integral(E = E, F = F, 
            nHI = nHI, nHeI = nHeI, nHeII = nHeII, intnum = integral,
            continuous = False, npoints = npoints)  
            
        cont = np.log10(continuous['integral'])
        disc = np.log10(discrete['integral'])            
            
        l2norm_abs = np.sqrt(np.sum((cont - disc)**2))      
        l2norm_rel = l2norm_abs / np.sqrt(np.sum(cont**2))
            
        return l2norm_abs, l2norm_rel     
        
    def default_energy_bins(self, log = False):
        """
        Default bins for energy.
        """
        if log:
            return np.logspace(np.log10(self.rs.Emin), np.log10(self.rs.Emax), 101)
        else:    
            return np.linspace(round(self.rs.Emin), self.rs.Emax, (self.rs.Emax - round(self.rs.Emin)) / 0.5 + 1)
    
    def default_norm_bins(self, log = False):
        """
        Default bins for energy.
        """
        
        if log:
            return np.logspace(self.pf['LogMinimumNormalization'], 0, 501)
        else:
            return np.linspace(0, 1, 101)           
        
def rebin(bins, center = False):
    """
    Take in an array of bin edges (centers) and convert them to bin centers (edges).
        center: Input bin values refer to bin centers?
        
    """
    
    bins = np.array(bins)

    if center:
        imax = bins.size - 1
        result = np.zeros(bins.size + 1)
        for i, element in enumerate(bins): 
            if i == imax:
                result[i] = bins[i] + (bins[i] - bins[i - 1]) / 2.
            else:
                result[i] = bins[i] - (bins[i + 1] - bins[i]) / 2.
    else:
        result = np.zeros(bins.size - 1)
        for i, element in enumerate(result): 
            result[i] = (bins[i] + bins[i + 1]) / 2.
            
    return result        
    

        