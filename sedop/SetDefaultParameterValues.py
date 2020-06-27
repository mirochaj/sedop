"""
SetDefaultParameterValues.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-01-28.

Description: Complete parameter list with default values.  Stored as a python 
dictionary, read in when we initialize the parameter space.

"""

def SetDefaultParameterValues():
    pf = \
    {
    
     # Output
     "ProgressBar": 1,
     "OutputDirectory": '.', 
     "ProcessorDumpName": 'guesses_proc',  
     "ResultsFileName": 'results',  
     "TrackWalks": 1,  
     
     # Methodology
     "MinimizationMethod": [0, 1, 2],  
     "ApproximateCrossSections": 0,
     "RateIntegral": [0, 1],  
     "Species": 1,  
     "MultiSpecies": 0,  
     "SecondaryIonization": 0,
     "NumberOfBins": 1,  
     
     # Annealing
     "TemperatureGamma": 0.98,  
     "CostPenaltyFactor": 1.,  
     "AnnealingFrequencyPerBin": 10,  
     "MaximumEnergyStep": 5,  
     "LogarithmicEnergyStep": 0,
     "MaximumNormalizationStep": 0.05,  
     "MaximumLogNormalizationStep": -1,  
     "LogarithmicNormStep": 0,
     "LogMinimumNormalization": -6,
     
     # Guess guiding
     "InitialGuessLogarithmic": 0,  
     "GaussianGuess": 0,
     "InitialGuessMemory": 0,
     "InitialTemperatureDecrement": 1,
                         
     # Source parameters
     "SourceType": 1,  
     "SourceTemperature": 1e5,  
     "SourceMass": 1e3,  
     "SourceRadiativeEfficiency": 0.1,
     "SourceISCO": 0,  
     "SourceTimeEvolution": 0,
     "SourceDiskMaxRadius": 1e3,
       
     # Spectral parameters
     "DiscreteSpectrum": 0,
     "SpectrumFile": 'None',
     "SpectrumType": 1, 
     "SpectrumFraction": 1,   
     "SpectrumPowerLawIndex": 1.5,  
     "SpectrumMinEnergy": 13.6,  
     "SpectrumMaxEnergy": 1e2,  
     "SpectrumMinNormEnergy": 0.01,  
     "SpectrumMaxNormEnergy": 5e2,  
     "SpectrumPhotonLuminosity": 5e48,  
     "SpectrumAbsorbingColumn": 0,  
     "SpectrumColorCorrectionFactor": 1.7,
     
     "FixedAge": 0,
     
     # Column density regime
     "HIColumnMax": 1e20,  
     "HeIColumnMax": 0,  
     "HINumberOfColumns": 20,  
     "HeINumberOfColumns": 1     
    }

    return pf