""" 

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2009-09-01.

Description: Contains various constants that may be of use.

Notes: 
      -All units are cgs unless stated otherwise.

"""

from math import pi

# General 
h = 6.626068e-27     			# Planck's constant - [h] = erg*s
h_bar = h / (2 * pi) 			# H-bar - [h_bar] = erg*s
c = 29979245800.0 				# Speed of light - [c] = cm/s
k_B = 1.3806503e-16			    # Boltzmann's constant - [k_B] = erg/K
G = 6.673e-8     				# Gravitational constant - [G] = cm^3/g/s^2
e = 1.60217646e-19   			# Electron charge - [e] = C
m_e = 9.10938188e-28     		# Electron mass - [m_e] = g
m_p = 1.67262158e-24    		# Proton mass - [m_p] = g
m_n = 1.67492729e-24            # Neutron mass - [m_n] = g
sigma_T = 6.65e-25			    # Cross section for Thomson scattering - [sigma_T] = cm^2
alpha = 1 / 137.035999070 		# Fine structure constant - unitless

# Stefan-Boltzmann constant - [sigma_SB] = erg / cm^2 / deg^4 / s
sigma_SB = 2.0 * pi**5 * k_B**4 / 15.0 / c**2 / h**3     

# Hydrogen 
A10 = 2.85e-15 				    # HI 21cm spontaneous emission coefficient - [A10] = Hz
E10 = 5.9e-6 				    # Energy difference between hyperfine states - [E10] = eV
m_H = 1.674e-24 			    # Mass of a hydrogen atom - [m_H] = g
nu_0 = 1420.4057e6 			    # Rest frequency of HI 21cm line - [nu_0] = Hz
T_star = 0.068 				    # Corresponding temperature difference between HI hyperfine states - [T_star] = K
a_0 = 5.292e-9 				    # Bohr radius - [a_0] = cm

# Helium
m_He = m_HeI = 2.0 * (m_p + m_n + m_e)
m_HeII = 2.0 * (m_p + m_n) + m_e
Y = 0.2477                      # Primordial helium abundance by mass
y = Y / 4. / (1. - Y)           # Primordial helium abundance by number

# Unit conversions
km_per_pc = 3.08568e13
km_per_mpc = km_per_pc*1e6
km_per_gpc = km_per_mpc*1e3
cm_per_pc = km_per_pc*1e5
cm_per_kpc = cm_per_pc*1e3
cm_per_mpc = cm_per_pc*1e6
cm_per_gpc = cm_per_mpc*1e3
cm_per_km = 1e5
cm_per_rsun = 695500. * cm_per_km
g_per_msun = 1.98892e33
s_per_yr = 365.25*24*3600
s_per_myr = s_per_yr*1e6
s_per_gyr = s_per_myr*1e3
sqdeg_per_std = (180.0**2)/(pi**2)
erg_per_j = 1e-7
erg_per_ev = e/erg_per_j
erg_per_kev = erg_per_ev*1e3

lsun = 3.839e33                                             # Solar luminosity - erg / s
cm_per_rsun = 695500.0 * 1e5                                # Radius of the sun - [cm_per_rsun] = cm
t_edd = 0.45 * 1e9 * s_per_yr                               # Eddington timescale (see eq. 1 in Volonteri & Rees 2005) 

xi = (3. / 7.)**0.5 * (6. / 7.)**3
