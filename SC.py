"""This module includes the physical constants needed for the calculations. Instances of the SC class containt superconductor specific constants, such as the Debye Temperature (TD), Density of States at Fermi surface (N0) etc. """

import scipy.integrate as integrate
import scipy.constants as const
import os
import numpy as np
import warnings


class Superconductor(object):
    '''General class for superconductor material properties.
    Free electron model is assumed.'''

    def __init__(self, name, Tc, TD, N0, rhon, kF,  t0, tpb, cT, cL, rho):
        self.name = name
        self.kbTc = Tc * const.Boltzmann / const.e * 1e6  # critical Temperature in µeV
        self.kbTD = TD * const.Boltzmann / const.e * 1e6  # Debye Energy in µeV
        self.N0 = N0  # Electronic DoS at Fermi Surface [µeV^-1 µm^-3]
        self.rhon = rhon  # normal state resistivity [µOhmcm] (!)
        self.kF = kF  # Fermi wave number [µm^-1]
        self.t0 = t0  # electron-phonon interaction time [µs]
        self.tpb = tpb  # phonon pair-breaking time [µs]
        self.cT = cT # transverse speed of sound
        self.cL = cL # longitudinal speed of sound
        self.rho = rho/const.e*1e-12 # mass density give in kg/m^3, returns in µeV/(µm/µs)**2 µm^-3
        
    @property
    def mstar(self):
        '''effective electron mass in µeV/(µm/µs)^2'''
        return 2 * (const.hbar*1e12/const.e)**2 * self.N0 * np.pi**2/self.kF
        
    @property
    def vF(self):
        '''Fermi velocity in µm/µs'''
        return const.hbar*1e12/const.e * self.kF / self.mstar
    
    @property
    def EF(self):
        '''Fermi energy in µeV'''
        return (const.hbar*1e12/const.e)**2 * self.kF**2 / (2*self.mstar)
    
    @property
    def l_e(self):
        '''electron mean free path in µm'''
        return (3*np.pi**2 * (const.hbar *1e12 / const.e) /
                (self.kF**2 * (self.rhon * 1e10 / const.e) * const.e**2))

    @property
    def lbd0(self):
        """London penetration depth (i.e. at T = 0 K) in µm. 
        This only holds when Mattis-Bardeen can be used 
        (i.e. dirty limit or extreme anomalous limit)"""
        return np.sqrt(
            const.hbar * 1e12 / const.e
            * self.rhon * 1e4
            / (const.mu_0 * 1e6
               * self.D0
               * np.pi)
        )

    @property
    def Vsc(self):
        """Calculates the superconducting coupling strength in BSC-theory 
        from the BSC relation 2D=3.52kbTc."""

        def integrand1(E, D):
            return 1 / np.sqrt(E ** 2 - D ** 2)

        return 1 / (
            integrate.quad(integrand1, self.D0, self.kbTD,
                           args=(self.D0,))[0] * self.N0
        )

    @property
    def Ddata(self):
        """To speed up the calculation for Delta, an interpolation of generated values is used.
        Ddata_{SC}_{Tc}.npy contains this data, where a new one is needed 
        for each superconductor (SC) and crictical temperature (Tc).
        This function returns the Ddata array."""

        Ddataloc = os.path.dirname(__file__) + "/Ddata/"
        if self.name != "":
            Tc = str(
                np.around(self.kbTc / (const.Boltzmann / const.e * 1e6), 3)
            ).replace(".", "_")
            try:
                Ddata = np.load(Ddataloc + f"Ddata_{self.name}_{Tc}.npy")
            except FileNotFoundError:
                Ddata = None
        else:
            Ddata = None
        return Ddata

    @property
    def D0(self):
        """BSC relation"""
        return 1.76 * self.kbTc

    @property
    def xi0(self):
        """BSC Coherence length at T = 0 K (in [µm])"""
        return const.hbar * 1e12 / const.e * self.vF / (np.pi * self.D0)
    
    @property
    def xi_DL(self):
        '''Dirty limit coherence length'''
        if self.xi0/self.l_e > 10:  
            return np.sqrt(self.xi0 * self.l_e)
        else:
            warnings.warn(f'Not in dirty limit xi0={self.xi0}, l={self.l}')

    @property
    def jc(self):
        """Critical current density, in A/µm^2, from Romijn1982"""
        return .75*np.sqrt(
            self.N0 * self.D0**3 /
            (self.rhon * 1e-2 / const.e * const.hbar * 1e12 / const.e)
                           )
    
    @property
    def D(self):
        '''Diffusion constant in µm^2/µs'''
        return 1/(2 * (self.rhon * 1e10 / const.e) * const.e**2 * self.N0)
    
    @property
    def lbd_eph(self):
        '''Electron-phonon coupling constant, (with BSC relation 2D=3.52kbTc)'''
        return (self.N0 * self.Vsc)
        
    @property
    def rhoM(self):
        '''The Mott resisitivty in µOhm cm'''
        return 3 * np.pi**2 * const.hbar / (const.e**2 * self.kF * 1e6) * 1e8
    
    @property
    def cs(self):
        '''effective 3D speed of sound'''
        return (3/(2/self.cT**3 + 1/self.cL**3))**(1/3)


###################### The superconductor objects ##########################
############################################################################


PEC = Superconductor('PEC', 
                     Tc=300,
                     rhon=0,
                     TD=np.nan,
                     N0=np.nan,
                     kF=np.nan,
                     t0=np.nan, 
                     tpb=np.nan,
                     cL=np.nan,
                     cT=np.nan,
                     rho=np.nan)

# Constants from Kaplan1976, Kaplan1976
Al = Superconductor('Al', 
                     Tc=1.2,
                     rhon=0.9,
                     TD=433,
                     N0=1.72e4,
                     kF=1.75e4,
                     t0=.44, 
                     tpb=.28e-3,
                     cL=6.65e3,
                     cT=3.26e3,
                     rho=2.5e3)


# tpb is calculated with Kaplan formula for tph0, with N and N(0) from 
# Abadias2019.
# tau0 is calculated from the power law GR noise lifetimes in LT192,
#     CPWs with tau0 = 4.2 tau_GR(Tc) (see Kaplan1979)
# EF, N0 and vF are calculated with the free electron model, 
#     with the results from the magnetoresistance experiments.
# The res are constants for Ta from Abadias2019 and Magnuson2019
bTa = Superconductor('bTa', 
                     Tc=1.,
                     rhon=206.,
                     TD=266,
                     N0=2.6e4,
                     kF=2.3e4,
                     t0=81e-3, 
                     tpb=.015e-3, 
                     cL=4.34e3,
                     cT=1.73e3,
                     rho=16.6e3)


# Data from Abadia2019 and Magnuson2019
aTa = Superconductor('aTa',
                     Tc=4.4,
                     rhon=8.8,
                     TD=258, 
                     N0=5.7e4,
                     kF=1.6e4,
                     t0=1.78e-3,
                     tpb=.015e-3,
                     cL=3.9e3,
                     cT=2.01e3,
                     rho=17.1e3)

# Film C of Coumou2013 and Kardakova2015
#  The Debye temperature is for T = 300 K,
#        https://doi.org/10.1016/S1006-7191(08)60082-4
# density from Hansen2020
TiN = Superconductor('TiN',
                     Tc=2.7,
                     rhon=253.,
                     TD=579.2,
                     N0=6.17e4,
                     kF=np.nan,
                     t0=4.2*23e-3,
                     tpb=np.nan,
                     cL=np.nan,
                     cT=np.nan,
                     rho=5e3)

# Sidorova
NbTiN = Superconductor('NbTiN',
                       Tc=15.1,
                       rhon=115.,
                       TD=np.nan,
                       N0=3.7e4,
                       kF=np.nan,
                       t0=np.nan,
                       tpb=np.nan,
                       cL=np.nan,
                       cT=np.nan,
                       rho=3e3)

class Sheet(object):
    '''A superconducting sheet with a certain thickness d and phonon escape
    time tesc'''

    def __init__(self, SC, d, tesc=1e-12, tescerr=0):
        self.SC = SC
        self.d = d
        self.tesc = tesc
        self.tescerr = tescerr

    @property
    def Rs(self):
        '''Returns the sheet resistance in µOhm/sq'''
        return self.SC.rhon / (self.d * 1e-4)

    @property
    def Lks(self):
        '''Returns the sheet inductance in pH/sq'''
        return const.hbar * 1e12 / const.e * self.Rs / (np.pi * self.SC.D0)

    def set_tesc(self, Chipnum, KIDnum, **kwargs):
        """sets the tesc attribute to a value calculated from the kidcalc.calc.tesc function,
        which uses GR noise lifetimes at high temperatures."""
        import kidata

        tesc, tescerr = kidata.calc.tesc(
            Chipnum, KIDnum, self, reterr=True, **kwargs)
        self.tesc = tesc
        self.tescerr = tescerr


class Vol(Sheet):
    '''A superconducting sheet of volume V, width w and lenght l'''

    def __init__(self, SC, d, V, w=np.nan, l=np.nan, tesc=1e-12, tescerr=0):
        super().__init__(SC, d, tesc, tescerr)
        self.V = V
        self.w = w
        self.l = l

    def checkV(self):
        return self.w * self.l * self.d == self.V

    @property
    def Ic(self):
        return self.SC.jc * self.w * self.d

###############################################################################


def init_SCvol(Chipnum, KIDnum, SC_class=Al, rhon=np.nan, set_tesc=True, **tesckwargs):
    """This function returns an Volume instance (which is defined with the SC_class
    argument), with the parameters initialized from data.
    Tc, V, and d (thickness) are from the S21 measurement, and tesc is set with
    the set_tesc method which uses kidata.calc.tesc() to estimate the phonon
    escape time."""
    import kidata

    S21data = kidata.io.get_S21data(Chipnum, KIDnum)
    SCvol = Vol(SC_class, V=S21data[0, 14],
                d=S21data[0, 25])
    SCvol.SC.kbTc = 86.17 * S21data[0, 21]
    SCvol.SC.rhon = rhon
    
    if set_tesc:
        SCvol.set_tesc(Chipnum, KIDnum, **tesckwargs)
    return SCvol
