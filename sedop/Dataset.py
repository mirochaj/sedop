"""
Dataset.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-03-11.

Description: Construct a sedop dataset object.
"""

import re, h5py, os
import numpy as np
import pylab as pl
import sedop.mods as sm
from scipy.integrate import simps
from sedop.analysis.RandomWalk import RandomWalk
from .Multiplot import *

Global = os.getcwd()

c = 29979245800.0
h = 6.626068e-27
k_B = 1.3806503e-16
erg_per_ev = 1.60217646e-12

class Dataset(object):
    def __init__(self, pf, read_all=True, merge=None, load_walks=False):
        """
        Initialize our analysis environment, read in data.
            pf = Parameter file (either a dictionary or a text file)
            rdir = Results directory (if supplied, will read in all .hdf5 files)
            merge = List of parameter files to merge together
            load_walks = Load random walk histories?
        """
        
        self.pf_fn = pf
        
        if pf[0:2] != './':
            pf = './%s' % pf
        
        # Set up global simulation directory
        i = pf.rfind('/')
        self.glo = '%s/%s' % (Global, pf[0:i])
          
        # Read pf, instantiate classes we might need later.
        if type(pf) is dict: 
            self.pf = pf
        else: 
            self.pf = sm.ReadParameterFile(pf)
        
        if self.pf['OutputDirectory'] is not None:
            self.gd = '%s/%s' % (self.glo, self.pf["OutputDirectory"])
        else:
            self.gd = '%s' % (self.glo)        
                        
        # Cumulative results file     
        self.cr = "%s/%s.hdf5" % (self.gd, self.pf["ResultsFileName"]) 
                                                
        """ 
        May just want to grab all data outputs in some directory.
        Assumed to all have the same parameters (just split up for the 
        sake of time, perhaps).  Will choke if h5 files containing walks are
        in the same directory, since it looks for every file ending in '.hdf5'
        """
        if read_all:
            
            crs = []
            tmp = os.listdir(self.gd)
            for element in tmp:
                if element == '%s.hdf5' % self.pf["ResultsFileName"]:
                    continue
                
                if re.search('.hdf5', element) and not re.search(self.pf["ProcessorDumpName"], element):
                    crs.append("%s/%s" % (self.gd, element)) 
                                    
        # May want to merge multiple runs - pfs here is all pfs except the main one
        if merge is not None:        
            
            if type(merge) is not list: 
                merge = [merge]
            
            pfs = []
            for entry in merge:
                pfs.append(entry)
                
            crs = [] 
            for entry in pfs:
                thispf = sm.ReadParameterFile(entry)
                crs.append("%s/%s.hdf5" % (self.gd, thispf["ResultsFileName"]))
                                                
        # Read in files (one for each processor) containing random walks ('guess_history_proc1.hdf5' for example)
        if load_walks:
            of_list = os.listdir(self.gd)
            self.of_list = []
            for fn in of_list:
                if re.search(self.pf["ProcessorDumpName"], fn):
                    self.of_list.append("{0}/{1}".format(self.gd, fn))
                            
            ds = {}       
            for fn in self.of_list:
                f = h5py.File(fn)
                for j, element in enumerate(f.keys()):
                    ID = element.strip('walk')
                    print(j, element)
                    ds[int(ID)] = RandomWalk(f[(element)], self.pf) 
                
                f.close()
            
            self.walks = ds

        # Always load in final results (Ei, Ef, Fi, Ff, cost)    
        results = {}
        f = h5py.File(self.cr, 'r')
        results['Ei']   = np.array(f[('E_initial_guesses')])
        results['Ef']   = np.array(f[('E_final_guesses')])
        results['Fi']   = np.array(f[('F_initial_guesses')])
        results['Ff']   = np.array(f[('F_final_guesses')])
        results['cost'] = np.array(f[('cost')])
        
        # Take parameter file from results output
        new_pf = {}
        for key in f['parameters'].keys():
            if type(f['parameters'][key].value) is np.ndarray:
                new_pf[key] = list(f['parameters'][key].value)
            else:
                new_pf[key] = f['parameters'][key].value    
        
        f.close()    
        self.pf = new_pf
        
        if (merge or read_all) is not None:
            
            stored = ['E_initial_guesses', 'E_final_guesses', 'F_initial_guesses', 'F_final_guesses']
            
            for entry in crs:
                f = h5py.File(entry, 'r')

                for i, datum in enumerate(['Ei', 'Ef', 'Fi', 'Ff']):
                                        
                    tmp = np.zeros([self.pf['NumberOfBins'], len(results['cost']) + len(f['cost'].value)]) 
                    
                    for j in xrange(int(self.pf['NumberOfBins'])):                        
                        tmp[j] = np.concatenate((results[datum][j], f[stored[i]].value[j]))
                                        
                    results[datum] = tmp

                results['cost'] = np.concatenate((results['cost'], f['cost'].value))
                                                                
                f.close()    
                     
        self.results = results 
        
        
        
    