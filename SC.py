"""This module includes the physical constants needed for the calculations. Instances of the SC class containt superconductor specific constants, such as the Debye Temperature (TD), Density of States at Fermi surface (N0) etc. """

import scipy.integrate as integrate
import scipy.constants as const
import os
import numpy as np


class Superconductor(object):
    '''General class for superconductor material properties. '''

    def __init__(self, name, Tc, TD, N0, rhon, EF, vF, t0, tpb):
        self.name = name
        self.kbTc = Tc * const.Boltzmann / const.e * 1e6  # critical Temperature in µeV
        self.kbTD = TD * const.Boltzmann / const.e * 1e6  # Debye Energy in µeV
        self.N0 = N0  # Electronic DoS at Fermi Surface [µeV^-1 µm^-3]
        self.rhon = rhon  # normal state resistivity [µOhmcm]
        self.EF = EF  # Fermi Energy [µeV]
        self.vF = vF  # Fermi velocity [µm/µs]
        self.t0 = t0  # electron-phonon interaction time [µs]
        self.tpb = tpb  # phonon pair-breaking time [µs]

    @property
    def lbd0(self):
        """Calculates the London penetration depth (i.e. at T = 0 K)"""
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
    def xi0BSC(self):
        """BSC Coherence length at T = 0 K (in [µm])"""
        return const.hbar * 1e12 / const.e * self.vF / (np.pi * self.D0)

    @property
    def jc(self):
        """Critical current density, in A/µm^2, from Romijn1982"""
        return .75*np.sqrt(
            self.N0 * self.D0**3 /
            (self.rhon * 1e-2 / const.e * const.hbar * 1e12 / const.e)
                           )


# Sub-classes, which are the actual superconductors
class PEC(Superconductor):
    def __init__(self, Tc=300, rhon=0):
        super().__init__(
            'PEC', 300, np.nan, np.nan, rhon, np.nan, np.nan, np.nan, np.nan)


class Al(Superconductor):
    def __init__(self, Tc=1.2, rhon=0.9):
        """The hardcoded arguments for the
        __init__() method, are standard constants for Al"""
        super().__init__(
            "Al", Tc, 433, 1.72e4, rhon, 11.6e6, 2.03e6, 0.44, 0.28e-3)


class bTa(Superconductor):
    def __init__(self, Tc=1.0, rhon=239.0):
        """The default escape time is calculated with Kaplan1979 for (a)Ta on 
        Sapphire.
        tpb is calculated with Kaplan formula for tph0, with N and N(0) from 
        Abadias2019.
        The hardcoded arguments for the __init__() method, are standard 
        constants for Ta from Abadias2019 
        and Magnuson2019"""
        super().__init__(
            "bTa", Tc, 266, 3.07e4, rhon, np.nan, np.nan, 1.78e-3, 0.015e-3
        )


class aTa(Superconductor):
    def __init__(self, Tc=4.4, rhon=8.8):
        """ The default escape time is calculated with Kaplan1979 for Ta on Sapphire.
        The hardcoded arguments for the
        __init__() method, are standard constants for Ta from Abadia2019 
        and Magnuson2019 """
        super().__init__(
            "aTa", Tc, 250, 5.70e4, rhon, 9.5e6, 0.24e6, 1.78e-3, 0.0227e-3,
        )


class TiN(Superconductor):
    def __init__(self, Tc=2.7, rhon=253.):
        """The default values are set to the film C of
        in Coumou2013 and Kardakova2015.
        The Debye temperature is for T = 300 K,
        https://doi.org/10.1016/S1006-7191(08)60082-4"""
        super().__init__(
            "TiN", Tc, 579.2, 6.17e4, rhon, np.nan, np.nan, 23e-3 * 4.2, np.nan)


class NbTiN(Superconductor):
    def __init__(self, Tc=15.1, rhon=115.):
        super().__init__(
            'NbTiN', Tc, np.nan, 3.7e4, rhon, np.nan, np.nan, np.nan, np.nan)


class Sheet(object):
    '''A superconducting sheet with a certain thickness d and phonon escape
    time tesc'''

    def __init__(self, SC, d, tesc=0, tescerr=0):
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

    def __init__(self, SC, d, V, w=np.nan, l=np.nan, tesc=0, tescerr=0):
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
    SCvol = Vol(SC_class(Tc=S21data[0, 21], rhon=rhon), V=S21data[0, 14],
                d=S21data[0, 25])
    if set_tesc:
        SCvol.set_tesc(Chipnum, KIDnum, **tesckwargs)
    return SCvol
