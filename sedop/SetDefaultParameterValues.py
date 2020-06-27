"""
SetDefaultParameterValues.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-01-28.

Description: Complete parameter list with default values.  Stored as a python 
dictionary, read in when we initialize the parameter space.

"""

from numpy import inf

def SetDefaultParameterValues():
    pf = \
    {
    
     # Output
     "progressbar": 1,

     # Methodology
     "method": [0, 1, 2],  
     "rate_integral": [0, 1],  
     "approx_cross_sections": 0,
     "species": 1,  
     "multispecies": 0,  
     "secondary_ionization": 0,
     "num_bins": 1,
     
     # Column density regime
     "NHImax": 1e20,  
     "NHeImax": 0,  
     "NHInum": 20,  
     "NHeInum": 1,
     
     # Annealing
     "anneal_gamma": 0.98,  
     "anneal_penalty": 1.,  
     "anneal_freq": 10,  
     "anneal_Estep": 5,  
     "anneal_logEstep": 0,
     "anneal_Fstep": 0.05,  
     "anneal_logFstep_max": -1,  
     "anneal_logFstep": 0,
     "anneal_logFmin": -6,
     "anneal_T0": 0.25,
     
     # Guess guiding
     "guess_log": 0,  
     "guess_gaussian": 0,
     "guess_memory": 0,
     
                         
     # Source parameters
     "source_type": 1,  
     "source_Qdot": 5e48,  
     "source_temperature": 1e5,  
     "source_mass": 1e3,  
     "source_epsilon": 0.1,
     "source_isco": 0,  
     "source_time_evolution": 0,
     "source_Rmax": 1e3,
     "source_age": 0,  
       
     # Spectral parameters
     "spectrum_discrete": 0,
     "spectrum_file": 'None',
     "spectrum_type": 1, 
     "spectrum_fraction": 1.,   
     "spectrum_alpha": 1.5,  
     "spectrum_Emin": 13.6,  
     "spectrum_Emax": 1e2,  
     "spectrum_EminNorm": 0.01,  
     "spectrum_EmaxNorm": 5e2,  
     "spectrum_logN": -inf,  
     "spectrum_fcol": 1.7,
        
    }

    return pf