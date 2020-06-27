"""
RandomWalk.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-03-11.

Description: Construct a random walk object.  This will be used in the Dataset class.   
"""

import numpy as np

class RandomWalk(object):
    def __init__(self, rw, pf):
        """
        Turns an hdf5 file object into attributes of the RandomWalk object!
        
        In other words, these is responsible for reading in all of the
        guess_history files, so we can look at the entire evolution of 
        random walks.
        """
        
        self.N = (len(rw) - 2) // 3
        
        self.E = np.zeros((self.N, int(pf['NumberOfSteps'])))
        self.F = np.zeros((self.N, int(pf['NumberOfSteps'])))
        self.A = np.zeros((self.N, int(pf['NumberOfSteps'])))
        
        for i, walk in enumerate(np.arange(self.N)):
            self.E[i] = rw['e{0}'.format(i)].value
            self.A[i] = rw['acc{0}'.format(i)].value
            self.F[i] = rw['f{0}'.format(i)].value
        
        self.cost = rw['cost'].value
        try: 
            self.T = rw['temp'].value
        except KeyError: 
            pass
            
            