'''This module contains functions that calculate the thermal noise of a resonator, 
w.r.t. scattering and recombination events.'''

import SCtheory as SCth
import numpy as np
from scipy.special import i0, i1, k0, k1


def Planck1D(hw, kbT):
    return hw/kbT / (np.exp(hw/kbT) - 1)

def Nyquist(hw, kbT, SCsheet): 
    return 4 * kbT * (Planck1D(hw, kbT) + 1/2 * hw/kbT) * np.real(SCth.Zs(hw, kbT, SCsheet))

def cinductresp_recomb(kbT, hw, sc):
    '''returns the recombination responsivity for changes in kbT, 
    d(s1/|s|)/dkbT, d(s2/|s|)/dkbT,  (unit: µeV^-1)'''
    D = SCth.D(kbT, sc)
    nqp = SCth.nqp(kbT, D, sc)
    xi = hw / (2 * kbT)
    s1, s2 = SCth.cinduct(hw, kbT, sc)
    
    ds1recomb = (nqp / (sc.N0 * hw) 
                 * np.sqrt(2*D**3/(np.pi * kbT**5)) 
                 * np.sinh(xi)*k0(xi))
    ds2recomb = - (nqp / (sc.N0 * hw) 
                   * np.sqrt(np.pi * D**3 / (2 * kbT**5)) 
                   * (np.sqrt(np.pi*kbT/(2*D)) + np.exp(-1*xi)*i0(xi)))
    return ds1recomb/np.abs(s1-1j*s2), ds2recomb/np.abs(s1-1j*s2)

def cinductresp_scat(kbT, hw, sc):
    '''returns d(s1/|s|)/dkbT, d(s2/|s|)/dkbT (unit: µeV^-1)'''
    D = SCth.D(kbT, sc)
    nqp = SCth.nqp(kbT, D, sc)
    xi = hw / (2 * kbT)
    s1, s2 = SCth.cinduct(hw, kbT, sc)
    
    ds1scat = nqp / (sc.N0 * hw) * np.sqrt(D/(2*np.pi*kbT**3)) * \
                (2*xi * (np.sinh(xi)*k1(xi)-np.cosh(xi)*k0(xi))-np.sinh(xi)*k0(xi))
    ds2scat = nqp / (sc.N0 * hw) * np.sqrt(np.pi*D/(8*kbT**3)) * \
                np.exp(-1*xi)*(i0(xi) + 2*xi*(i1(xi) - i0(xi))) 
    return ds1scat/np.abs(s1-1j*s2), ds2scat/np.abs(s1-1j*s2)

def dNqpdkbT(kbT, scvol):
    ''' returns the change in quasiparticles w.r.t. kbT at low temperatures (kbT << Delta) '''
    D = SCth.D(kbT, scvol.SC)
    return SCth.nqp(kbT, D, scvol.SC) * scvol.V * D / kbT**2

def var_total(kbT, scvol):
    ''' returns the variance up to first order in temperature (in kbT/Delta).'''
    D = SCth.D(kbT, scvol.SC)
    return kbT**4 / (scvol.V * SCth.nqp(kbT, D, scvol.SC) * D**2) * (1 - (kbT/D)**2)

def var_recomb(kbT, scvol):
    '''returns the variance of recombination noise, in the low temperature limit (kbT << Delta)'''
    D = SCth.D(kbT, scvol.SC)
    return kbT**4 / (scvol.V * SCth.nqp(kbT, D, scvol.SC) * D**2)

def var_scat(kbT, scvol):
    '''returns the variance of scattering noise, in the low temperature limit (kbT << Delta). 
    This is only valid in the range where tscat << trec, such that the number of quasiparticles
    is constant at these times scales.'''
    return kbT**2 / (scvol.V * SCth.nqp(kbT, SCth.D(kbT, scvol.SC), scvol.SC))
    
    