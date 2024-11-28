import numpy as np
import scipy.constants as const
from scipy.special import lambertw

hbar = const.hbar / const.e * 1e12

def kbTstaroD(sc, ak, w0, Pint, V): 
    '''Returns T*/Delta, as given in Fischer&Catelani2023, which is a measure
    of the width of the distribution function due to microwave photon absorption.
    ak - kinetic induction fraction
    w0 - angular resonance frequency in rad/µs
    Pint - internal power in the resonator in µeV/µs
    V - volume that the quasiparticles occupy in µm^3'''
    return (105/64 * (sc.kbTc/sc.D0)**3 * 
            ak * w0* Pint / V * sc.t0/(2*sc.N0*sc.D0) * hbar / sc.D0**2)**(1/6)

def kbTBstar(sc, ak, w0, Pint, V):
    '''Crossover temperature for high photon/low phonon - low photon/high phonon regimes. 
    See Fischer&Catelani2023.'''
    return kbTstaroD(sc, ak, w0, Pint, V)**3 * sc.D0

def nqp_sat(scvol, ak, w0, Pint, V):
    '''Saturation quasiparticle density due to microwave photon absorption.
    This includes the correction in Tstar/Delta. Note that there is a 
    factor 2.1 missing in eq. (51) in Fischer&Catelani2024.'''
    sc = scvol.SC
    kbTstaroD_value = kbTstaroD(sc, ak, w0, Pint, V)
    correction = (2.1 + kbTstaroD_value*3/4*.88 - 5/32*.77*kbTstaroD_value**2)
    return (.39 * scvol.tesc/sc.tpb * sc.N0 * sc.D0 * 
            kbTstaroD_value**(9/2) * np.exp(-np.sqrt(14/5)*kbTstaroD_value**-3)
            * correction)