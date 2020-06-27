"""
RateIntegrals.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-01-28.

Description: Here are all the integrals/sums that appear in our RT rate equations.
     
"""

import numpy as np
from scipy.integrate import quad
from .RadiationSource import *
from .SecondaryElectrons import *
from .ComputeCrossSections import PhotoIonizationCrossSection

np.seterr(all = 'ignore')

params = np.array([[4.298e-1, 5.475e4, 3.288e1, 2.963, 0.0, 0.0, 0.0],
                   [13.61, 9.492e2, 1.469, 3.188, 2.039, 4.434e-1, 2.136],
                   [1.72, 1.369e4, 3.288e1, 2.963, 0.0, 0.0, 0.0]])

E_th = [13.6, 24.6, 54.4]   # Threshold energies for photoionization (HI, HeI, HeII)

class RateIntegrals(object): 
    def __init__(self, pf):
        self.pf = pf
        self.rs = RadiationSource(pf)
        self.esec = SecondaryElectrons(pf)
        self.N = pf['NumberOfBins']
        self.MultiSpecies = pf['MultiSpecies']
        self.approx = pf['ApproximateCrossSections']
        
    def PhotoIonizationCrossSection(self, E, species = 0):
        """ 
        Returns photoionization cross section for HI, HeI, or HeII from the fits of
        Verner et al. 1996.  HI is the first 7-element sub-array in 'params', HeI
        is the second, and HeII is the third.  In order, the coefficients in these arrays are:
            
            E_0, sigma_0, y_a, P, y_w, y_0, y_1
            
        Also:
            species = 0 for HI
            species = 1 for HeI
            species = 2 for HeII
            
        Note: The units are cm^2.
        
        Also, it is OK if E is a numpy array of energies.
        
        """        
        
        if self.approx and species == 0:
            return 6.346e-18 * (13.6 / E)**3    # coefficient 7 or 8 may be better...
            
        else:    
            x = np.divide(E, params[species][0]) - params[species][5]
            y = np.sqrt(np.multiply(x, x) + params[species][6]**2)
            F_y = (np.power((x - 1.0), 2) + params[species][4]**2) * np.power(y, (0.5 * params[species][3] - 5.5)) * \
                np.power((1.0 + np.sqrt(np.divide(y, params[species][2]))), -params[species][3])
                                    
        return params[species][1] * F_y * 1e-18    
        
    def OpticalDepth(self, E, ncol):
        """
        Returns the optical depth at energy E due to column densities of HI, HeI, and HeII, which
        are stored in the variable 'ncol' as a three element array.
        """
                        
        if type(E) is float:
            E = [E]
                               
        tau = np.zeros_like(E)
        for i, energy in enumerate(E):
            tmp = 0
            
            if energy >= E_th[0]:
                tmp += self.PhotoIonizationCrossSection(energy, 0) * ncol[0]
                
            if self.MultiSpecies:    
                if energy >= E_th[1]:
                    tmp += self.PhotoIonizationCrossSection(energy, 1) * ncol[1]
                if energy >= E_th[2]:
                    tmp += self.PhotoIonizationCrossSection(energy, 2) * ncol[2]    

            tau[i] = tmp     
        
        if len(E) > 1:                
            return tau 
        else:
            return tau[0]        
            
    def Phi(self, n, E = 0.0, F_E = 1.0, species = 0, continuous = True):
        """
        Integral over energy to determine the photoionization rate for a given species and set of absorbing columns.
        
        Energy dependence only necessary for discrete calculation.  Same for F_E.  They must be numpy arrays.
        
        n = [nHI, nHeI]
        
        """
         
        if continuous:            
            if self.pf['SpectrumFile'] != 'None':
                return np.trapz(self.rs.Spectrum()[self.rs.i_Eth[species]:] * \
                    np.exp(-self.OpticalDepth(self.rs.E[self.rs.i_Eth[species]:], n)) / \
                    (self.rs.E[self.rs.i_Eth[species]:] * erg_per_ev), self.rs.E[self.rs.i_Eth[species]:])
            else:
                integrand = lambda EE: 1e-10 * self.rs.Spectrum(EE) * \
                    np.exp(-self.OpticalDepth(EE, n)) / (EE * erg_per_ev)    
                return 1e10 * quad(integrand, max(E_th[species], self.rs.Emin), self.rs.Emax, epsrel = 1e-8, epsabs = 1e-8)[0]
        else:
            return np.sum(F_E * np.exp(-self.OpticalDepth(E, n)) / E / erg_per_ev)

    def Psi(self, n, E = 0.0, F_E = 1.0, species = 0, continuous = True):
        """
        Integral over energy to determine the photo-heating rate do to electrons from photo-ionizations of species.
        """
    
        if continuous:
            if self.pf['SpectrumFile'] != 'None':
                return np.trapz(self.rs.Spectrum()[self.rs.i_Eth[species]:] * \
                    np.exp(-self.OpticalDepth(self.rs.E[self.rs.i_Eth[species]:], n)), 
                    self.rs.E[self.rs.i_Eth[species]:])
            else:
                integrand = lambda EE: self.rs.Spectrum(EE) * \
                    np.exp(-self.OpticalDepth(EE, n))
                                                                        
                return quad(integrand, max(E_th[species], self.rs.Emin), self.rs.Emax, epsrel = 1e-8, epsabs = 1e-8)[0]
        
        else:
            return np.sum(F_E * \
                np.exp(-self.OpticalDepth(E, n)))
                            
    def PhiWiggle(self, n, xHII, E = 0.0, F_E = 1.0, species = 0, continuous = True):
        """
        Defined in documentation. For advanced secondary ionization and heating.
        """              
                
        if continuous:
            integrand = lambda EE: self.esec.DepositionFraction(EE, xHII, channel = species + 1) * \
                self.rs.Spectrum(EE) * \
                np.exp(-self.OpticalDepth(EE, n)) / EE
                                                                    
            return quad(integrand, max(E_th[species], self.rs.Emin), self.rs.Emax, epsrel = 1e-8, epsabs = 1e-8)[0]
        
        else:
            return np.sum(self.esec.DepositionFraction(EE, xHII, channel = species + 1) * F_E * \
                np.exp(-self.OpticalDepth(E, n)) / EE)        
                        
    def PhiWiggle(self, ncol, species = 0, donor_species = 0, xHII = 0.0):
        """
        Equation 2.18 in the rt1d manual.
        """        
        
        Ej = E_th[donor_species]
        
        # Otherwise, continuous spectrum                
        integrand = lambda E: 1e10 * \
            self.esec.DepositionFraction(E - Ej, xHII, channel = species + 1) * \
            self.rs.Spectrum(E) * \
            np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
            (E * erg_per_ev)
                    
        c = self.esec.Energies >= max(Ej, self.rs.Emin)
        c &= self.esec.Energies <= self.rs.Emax
                        
        y = []
        for E in self.esec.Energies[c]:
            y.append(integrand(E))
             
        return 1e-10 * \
            np.trapz(np.array(y), self.esec.Energies[c])    
    
    def PsiWiggle(self, ncol, species = 0, donor_species = 0, xHII = 0.0):            
        """
        Equation 2.19 in the rt1d manual.
        """        
        
        Ej = E_th[donor_species]
        
        # Otherwise, continuous spectrum    
        integrand = lambda E: 1e20 * \
            self.esec.DepositionFraction(E - Ej, xHII, channel = species + 1) * \
            self.rs.Spectrum(E) * \
            np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
                      
        c = self.esec.Energies >= max(Ej, self.rs.Emin)
        c &= self.esec.Energies <= self.rs.Emax
                        
        y = []
        for E in self.esec.Energies[c]:
            y.append(integrand(E))
             
        return 1e-20 * \
            np.trapz(np.array(y), self.esec.Energies[c])          
                              
    def PhiHat(self, ncol, species = 0, donor_species = None, xHII = 0.0):
        """
        Equation 2.20 in the rt1d manual.
        """        
        
        Ei = E_th[species]
        
        # Otherwise, continuous spectrum                
        integrand = lambda E: 1e10 * \
            self.esec.DepositionFraction(E - Ei, xHII, channel = 0) * \
            self.rs.Spectrum(E) * \
            np.exp(-self.SpecificOpticalDepth(E, ncol)[0]) / \
            (E * erg_per_ev)
               
        c = self.esec.Energies >= max(Ei, self.rs.Emin)
        c &= self.esec.Energies <= self.rs.Emax       
                                                
        y = []
        for E in self.esec.Energies[c]:
            y.append(integrand(E))
             
        return 1e-10 * \
            np.trapz(np.array(y), self.esec.Energies[c])          
                
    def PsiHat(self, ncol, species = 0, donor_species = None, xHII = 0.0):            
        """
        Equation 2.21 in the rt1d manual.
        """        
        
        Ei = E_th[species]
        
        # Otherwise, continuous spectrum    
        integrand = lambda E: 1e20 * \
            self.esec.DepositionFraction(E - Ei, xHII, channel = 0) * \
            self.rs.Spectrum(E) * \
            np.exp(-self.SpecificOpticalDepth(E, ncol)[0])
         
        c = self.esec.Energies >= max(Ei, self.rs.Emin)
        c &= self.esec.Energies <= self.rs.Emax
                        
        y = []
        for E in self.esec.Energies[c]:
            y.append(integrand(E))
             
        return 1e-20 * \
            np.trapz(np.array(y), self.esec.Energies[c])   
                          
            