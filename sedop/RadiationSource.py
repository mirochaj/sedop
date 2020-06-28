"""
RadiationSource.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2011-01-28.

Description: Radiation source class, contains all functions needed to calculate the spectrum
of an object.  Stole from rt1d, with slight modification
     
"""

import re, h5py
import numpy as np
from .Constants import *
from scipy.integrate import quad
from .ComputeCrossSections import PhotoIonizationCrossSection as sigma_E

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

np.seterr(all = 'ignore')   # exp overflow occurs when integrating BB - will return 0 as it should for x large

SchaererTable = {
                "Mass": [5, 9, 15, 25, 40, 60, 80, 120, 200, 300, 400, 500, 1000], 
                "Temperature": [4.44, 4.622, 4.759, 4.85, 4.9, 4.943, 4.97, 4.981, 4.999, 5.007, 5.028, 5.029, 5.026],
                "Luminosity": [2.87, 3.709, 4.324, 4.89, 5.42, 5.715, 5.947, 6.243, 6.574, 6.819, 6.984, 7.106, 7.444]
                }

small_number = 1e-3
big_number = 1e5
ls = ['-', '--', ':', '-.']
E_th = [13.6, 24.6, 54.4]

"""
SourceType = 0  (monochromatic)     Just need DiscreteSpectrum and PhotonLuminosity
SourceType = 1  (star)              Need temperature, PhotonLuminosity
SourceType = 2  (popIII star)       Need mass
SourceType = 3  (BH)                Need Mass, epsilon

SpectrumType = 0 (monochromatic)    
SpectrumType = 1 (blackbody)                    
SpectrumType = 2 (blackbody, but temperature and luminosity from Schaerer)
SpectrumType = 3 (multi-color disk)
SpectrumType = 4 (simple power-law)
"""

def RadiationSource(pf):
    if pf['spectrum_file'] is not None:
        return RadiationSourceUserDefined(pf)
    elif pf['spectrum_func'] is not None:
        return pf['spectrum_func']
    else:
        return RadiationSourceModel(pf)
                                
class RadiationSourceModel(object):
    def __init__(self, pf):
        self.pf = pf
        
        self.SpectrumPars = listify(pf)
        
        self.N = len(self.SpectrumPars['type'])
        
        # Cast types to int to avoid indexing complaints
        self.SpectrumPars['Type'] = \
            [int(elem) for elem in self.SpectrumPars['type']]
        
        self.Emin = min(self.SpectrumPars['Emin'])
        self.Emax = min(self.SpectrumPars['Emax'])        
        
        if self.N == 1:
            self.Type = self.SpectrumPars['type'][0]
                       
        self.last_renormalized = 0
        
        # SourceType 0, 1, 2
        self.Lph = pf['source_Qdot']
        
        # SourceType = 1, 2
        self.T = pf['source_temperature']
        
        # SourceType >= 3
        self.M = self.M0 = pf['source_mass']
        self.epsilon = pf['source_epsilon']
        
        if 3 in self.SpectrumPars['Type']:
            self.r_in = self.DiskInnermostRadius(self.M0)
            self.r_out = pf['source_Rmax'] * self.GravitationalRadius(self.M0)
            self.fcol = self.SpectrumPars['fcol'][self.SpectrumPars['type'].index(3)]
            self.T_in = self.DiskInnermostTemperature(self.M0)
            self.T_out = self.DiskTemperature(self.M0, self.r_out)
        
        # Number of ionizing photons per cm^2 of surface area for BB of temperature self.T.  
        # Use to solve for stellar radius (which we need to get Lbol).  The factor of pi gets rid of the / sr units
        if pf['source_type'] in [1, 2]:
            self.LphNorm = np.pi * 2. * (k_B * self.T)**3 * \
                quad(lambda x: x**2 / (np.exp(x) - 1.), 
                13.6 * erg_per_ev / k_B / self.T, big_number, epsrel = 1e-12)[0] / h**3 / c**2 
            self.R = np.sqrt(self.Lph / 4. / np.pi / self.LphNorm)        
            self.Lbol = 4. * np.pi * self.R**2 * sigma_SB * self.T**4
        else:
            self.Lbol = self.BolometricLuminosity(0.0)           
             
        # Normalize spectrum
        self.LuminosityNormalizations = self.NormalizeSpectrumComponents(0.0)
          
    def GravitationalRadius(self, M):
        """
        Half the Schwartzchild radius.
        """
        return G * M * g_per_msun / c**2
        
    def SchwartzchildRadius(self, M):
        return 2. * self.GravitationalRadius(M)    
        
    def MassAccretionRate(self, M = None):
        return self.BolometricLuminosity(0, M = M) / self.epsilon / c**2        
        
    def DiskInnermostRadius(self, M):      
        """
        Inner radius of disk.  Unless SourceISCO > 0, will be set to the 
        inner-most stable circular orbit for a BH of mass M.
        """
        if not self.pf['source_isco']:
            return 6. * self.GravitationalRadius(M)
        else:
            return self.pf['source_isco']     
            
    def DiskInnermostTemperature(self, M):
        """
        Temperature (in Kelvin) at inner edge of the disk.
        """
        return (3. * G * M * g_per_msun * self.MassAccretionRate(M) / \
            8. / np.pi / self.DiskInnermostRadius(M)**3 / sigma_SB)**0.25
    
    def DiskTemperature(self, M, r):
        return ((3. * G * M * g_per_msun * self.MassAccretionRate(M) / \
            8. / np.pi / r**3 / sigma_SB) * \
            (1. - (self.DiskInnermostRadius(M) / r)**0.5))**0.25
            
    def BlackHoleMass(self, t):
        """
        Compute black hole mass after t (seconds) have elapsed.  Relies on 
        initial mass self.M, and (constant) radiaitive efficiency self.epsilon.
        """        
        
        return self.M0 * np.exp(((1.0 - self.epsilon) / self.epsilon) * t / t_edd)         
                
    def Intensity(self, E, i, Type, t):
        """
        Return quantity *proportional* to fraction of bolometric luminosity emitted
        at photon energy E.  Normalization handled separately.
        """
        
        if Type == 0:
            Lnu = self.F[0]
        elif Type in [1, 2]:
            Lnu = self.BlackBody(E)
        elif Type == 3:
            Lnu = self.MultiColorDisk(E, i, Type, t)
        elif Type == 4: 
            Lnu = self.PowerLaw(E, i, Type, t)    
        else:
            Lnu = 0.0
            
        if self.SpectrumPars['logN'][i] > 0:
            return Lnu * np.exp(-10**self.SpectrumPars['logN'][i] * (sigma_E(E, 0) + y * sigma_E(E, 1)))   
        else:
            return Lnu     
                
    def Spectrum(self, E, t=0.0, only=None):
        """
        Return fraction of bolometric luminosity emitted at energy E.
        """        
        
        if type(E) in [float, np.float64]:
            E = [E]
        
        # Renormalize if t > 0 
        if t != self.last_renormalized:
            self.last_renormalized = t
            self.M = self.BlackHoleMass(t)
            self.r_in = self.DiskInnermostRadius(self.M)
            self.r_out = self.pf['souce_Rmax'] * self.GravitationalRadius(self.M)
            self.T_in = self.DiskInnermostTemperature(self.M)
            self.T_out = self.DiskTemperature(self.M, self.r_out)
            self.Lbol = self.BolometricLuminosity(t)
            self.LuminosityNormalizations = self.NormalizeSpectrumComponents(t)    
        
        emission = np.zeros_like(E)
        for h, bin in enumerate(E):
            for i, Type in enumerate(self.SpectrumPars['type']):
                if not (self.SpectrumPars['Emin'][i] <= E[h] <= self.SpectrumPars['Emax'][i]):
                    continue
                    
                if only is not None and Type != only:
                    continue 
                    
                emission[h] += self.LuminosityNormalizations[i] * \
                    self.Intensity(E[h], i, Type, t) / self.Lbol
            
        return emission
        
    def BlackBody(self, E, T = None):
        """
        Returns specific intensity of blackbody at self.T.
        """
        
        if T is None:
            T = self.T
        
        nu = E * erg_per_ev / h
        return 2.0 * h * nu**3 / c**2 / (np.exp(h * nu / k_B / T) - 1.0)
        
    def PowerLaw(self, E, i, Type, t = 0.0):    
        """
        A simple power law X-ray spectrum - this is proportional to the *energy* emitted
        at E, not the number of photons.  
        """

        return E**-self.SpectrumPars.PowerLawIndex[i]
    
    def MultiColorDisk(self, E, i, Type, t = 0.0):
        """
        Soft component of accretion disk spectra.
        """         
        
        # If t > 0, re-compute mass, inner radius, and inner temperature
        if t > 0 and self.pf['source_time_evolution'] > 0 and t != self.last_renormalized:
            self.M = self.BlackHoleMass(t)
            self.r_in = self.DiskInnermostRadius(self.M)
            self.r_out = self.pf['source_Rmax'] * self.GravitationalRadius(self.M)
            self.T_in = self.DiskInnermostTemperature(self.M)
            self.T_out = self.DiskTemperature(self.M, self.r_out)
        
        integrand = lambda T: (T / self.T_in)**(-11. / 3.) * self.BlackBody(E, T) / self.T_in
        return quad(integrand, self.T_out, self.T_in)[0]
                            
    def NormalizeSpectrumComponents(self, t = 0):
        """
        Normalize each component of spectrum to some fraction of the bolometric luminosity.
        """
        
        Lbol = self.BolometricLuminosity(t)
        
        normalizations = np.zeros(self.N)
        for i, component in enumerate(self.SpectrumPars['Type']):
            integral, err = quad(self.Intensity, self.SpectrumPars['EminNorm'][i], 
                self.SpectrumPars['EmaxNorm'][i], args = (i, component, t,))
            normalizations[i] = self.SpectrumPars['fraction'][i] * Lbol / integral
            
        return normalizations
        
    def BolometricLuminosity(self, t = 0.0, M = None):
        """
        Returns the bolometric luminosity of a source in units of erg/s.  For accreting black holes, the 
        bolometric luminosity will increase with time, hence the optional 't' argument.
        """
        
        if self.pf['source_type'] == 1:
            return self.Lbol
        
        if self.pf['source_type'] == 2:
            return 10**SchaererTable["Luminosity"][SchaererTable["Mass"].index(self.M)] * lsun
            
        if self.pf['source_type'] > 2:
            Mnow = self.BlackHoleMass(t)
            if M is not None:
                Mnow = M
            return self.epsilon * 4.0 * np.pi * G * Mnow * g_per_msun * m_p * c / sigma_T
            
    def PlotSpectrum(self, color = 'k', components = True, t = 0, normalized = True,
        bins = 100, mp = None):
        import pylab as pl
        
        if not normalized:
            Lbol = self.BolometricLuminosity(t)
        else: 
            Lbol = 1
        
        E = np.logspace(np.log10(min(self.SpectrumPars['EminNorm'])), 
            np.log10(max(self.SpectrumPars['EmaxNorm'])), bins)
        F = []
        
        for energy in E:
            F.append(self.Spectrum(energy, t = t))
        
        if components and self.N > 1:
            EE = []
            FF = []
            for i, component in enumerate(self.SpectrumPars['type']):
                tmpE = np.logspace(np.log10(self.SpectrumPars['Emin'][i]), 
                    np.log10(self.SpectrumPars['Emax'][i]), bins)
                tmpF = []
                for energy in tmpE:
                    tmpF.append(self.Spectrum(energy, t = t, only = component))
                
                EE.append(tmpE)
                FF.append(tmpF)
        
        if mp is None:
            self.ax = pl.subplot(111)
        else:
            self.ax = mp
                    
        self.ax.loglog(E, np.array(F) * Lbol, color = color, ls = ls[0])
        
        if components and self.N > 1:
            for i in range(self.N):
                self.ax.loglog(EE[i], np.array(FF[i]) * Lbol, color = color, ls = ls[i + 1])
        
        self.ax.set_ylim(1e-3 * np.max(F), 1.1 * np.max(F))
        self.ax.set_xlabel(r'$h\nu \ (\mathrm{eV})$')
        
        if normalized:
            self.ax.set_ylabel(r'$L_{\nu} / L_{\mathrm{bol}}$')
        else:
            self.ax.set_ylabel(r'$L_{\nu}$')
                
        pl.draw()        
                
class RadiationSourceUserDefined(object):
    def __init__(self, pf):
        self.pf = pf
        self.fn = pf['spectrum_file']
        
        self.initialize()
        
    def initialize(self):
        """
        Read data from file.  Should be two columns: energy (eV), and the
        specific luminosity (erg/s/eV) in each bin.  If hdf5, datasets should
        be called 'E' and 'L_E'.  If ASCII, names don't matter, but E should
        be first column, L_E the second.
        """                 
        
        # Can actually just supply as arrays too.
        if type(self.fn) in [tuple, list, np.array]:
            self.E, self.L_E = self.fn
        elif re.search('hdf5', self.fn) or re.search('h5', self.fn):
            f = h5py.File(self.fn)
            self.E = f['E'].value
            self.L_E = f['L_E'].value
            
            #if len(self.L_E) > 1:
            #    self.t = self.Age = f['time_yr'].value * s_per_yr
            #    self.Nt = len(self.t)
            #    i = self.get_time_index(self.pf['source_age'] * s_per_myr)
            #    self.L_E = self.L_E[i]
            
            f.close()    
            if rank == 0:
                print("Read {}.".format(self.fn))            
                        
        else:
            self.E, self.L_E = np.loadtxt(self.fn, unpack=True)
            if rank == 0:
                print("Read {}.".format(self.fn))
            
        self.Emin = max(np.min(self.E), E_th[0])
        self.Emax = np.max(self.E)
        
        # Threshold indices
        self.i_Eth = np.zeros(3, dtype=int)
        for i, energy in enumerate(E_th):
            loc = np.argmin(np.abs(energy - self.E))
            
            if self.E[loc] < energy:
                loc += 1
            
            self.i_Eth[i] = int(loc)
            
        self.Lbol = self.BolometricLuminosity()    
                        
    def Spectrum(self, E=None, t=0.0):
        """
        Return (normalized) specific luminosity.
        """
        
        return self.L_E / self.Lbol
        
    def Intensity(self, E = None):
        return self.L_E
        
    def BolometricLuminosity(self, t=0.0):
        return np.trapz(self.L_E, self.E)
        
    def get_time_index(self, t):
        i = np.argmin(np.abs(t - self.t))
        return max(min(i, self.Nt - 2), 0)
                
def listify(pf):
    """
    Turn any Spectrum parameter into a list, if it isn't already.
    """            
    
    if type(pf['spectrum_type']) is not list:
        ntypes = 1
    else:
        ntypes = len(pf['spectrum_type'])
    
    Spectrum = {}    
    for par in pf.keys():
        if par[0:8] != 'spectrum':
            continue
            
        new_name = par[9:]
        if type(pf[par]) is not list:
            Spectrum[new_name] = [pf[par]]
        else:
            Spectrum[new_name] = pf[par]
            
    return Spectrum    

