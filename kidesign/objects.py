import sys
sys.path.insert(1, '../../Coding')
import SC

import numpy as np
import copy
import scipy.special as sp
import scipy.constants as const
from scipy.optimize import minimize_scalar

'''TODO:
- Find and implement Coupler analytic formula
- Refactor KID object with child hyKID and open/shorted
    and quarter/half wave options
- Implement important design characteristics:
    - P_0 (power handling) (bifurcation point)
    - Expected phase noise
    - Expected amplifier noise
    - Expected GR noise
    - Responsivity
'''


class CPW(object):
    """Class to calculate CPW properties, such as vph, Z0 and ak.
    Give S,W,d and l in µm, and Lk in pH/sq"""

    def __init__(self, SCsheet, S, W, eeff, l=1):
        self.SCsheet = SCsheet
        self.S = S * 1e-6
        self.W = W * 1e-6
        self.eeff = eeff
        self.l = l * 1e-6

    # CALCULATED ATTRIBUTES
    @property
    def k(self):
        return self.S / (self.S + 2 * self.W)

    @property
    def gc(self):
        return (
            np.pi
            + np.log(4 * np.pi * self.S / (self.SCsheet.d * 1e-6))
            - self.k * np.log((1 + self.k) / (1 - self.k))
        ) / (4 * self.S * (1 - self.k ** 2) * sp.ellipk(self.k ** 2) ** 2)

    @property
    def gg(self):
        return (
            self.k
            * (
                np.pi
                + np.log(4 * np.pi * (self.S + 2 * self.W) / (self.SCsheet.d * 1e-6))
                - 1 / self.k * np.log((1 + self.k) / (1 - self.k))
            )
            / (4 * self.S * (1 - self.k ** 2) * sp.ellipk(self.k ** 2) ** 2)
        )

    @property
    def xi(self):
        return sp.ellipk(1 - self.k ** 2) / sp.ellipk(self.k ** 2)

    @property
    def Lgl(self):
        return const.mu_0 * self.xi / 4

    @property
    def Lkl(self):
        return self.SCsheet.Lks * 1e-12 * (self.gc + self.gg)

    @property
    def L(self):
        return self.l * (self.Lkl + self.Lgl)

    @property
    def Cl(self):
        return 4 * const.epsilon_0 * self.eeff / self.xi

    @property
    def C(self):
        return self.l * self.Cl

    @property
    def Z0(self):
        return np.sqrt((self.Lgl + self.Lkl) / self.Cl)

    @property
    def vph(self):
        return 1 / np.sqrt((self.Lgl + self.Lkl) * self.Cl)

    @property
    def ak(self):
        return self.Lkl / (self.Lkl + self.Lgl)

    @property
    def fres(self):
        return self.vph / self.l

    # METHODS
    def beta(self, f):
        return 2 * np.pi * f / self.vph

    def ABCD(self, f):
        beta = self.beta(f)
        Z0 = self.Z0
        return np.array(
            [
                [np.cos(beta * self.l), 1j * Z0 * np.sin(beta * self.l)],
                [1j / Z0 * np.sin(beta * self.l), np.cos(beta * self.l)],
            ]
        )


class hyCPW(CPW):
    '''Hybrid CPW, with a different sheet for ground plane (gSCsheet)
    and central line (cSCsheet).'''

    def __init__(self, gSCsheet, cSCsheet, S, W, eeff, l=1):
        CPW.__init__(self, None, S, W, eeff, l=l)
        self.gSCsheet = gSCsheet
        self.cSCsheet = cSCsheet

    @property
    def Lkl(self):
        return (self.cSCsheet.Lks * 1e-12 * self.gc +
                self.gSCsheet.Lks * 1e-12 * self.gg)

    @property
    def gc(self):
        return (
            np.pi
            + np.log(4 * np.pi * self.S / (self.cSCsheet.d * 1e-6))
            - self.k * np.log((1 + self.k) / (1 - self.k))
        ) / (4 * self.S * (1 - self.k ** 2) * sp.ellipk(self.k ** 2) ** 2)

    @property
    def gg(self):
        return (
            self.k
            * (
                np.pi
                + np.log(4 * np.pi * (self.S + 2 * self.W) /
                         (self.gSCsheet.d * 1e-6))
                - 1 / self.k * np.log((1 + self.k) / (1 - self.k))
            )
            / (4 * self.S * (1 - self.k ** 2) * sp.ellipk(self.k ** 2) ** 2)
        )


class Coupler(object):
    '''Object that represents the capcitive coupler from (quarterwave) KID to TL'''

    def __init__(self, Qc, f0, Ztl, Zkid):
        self.Qc = Qc
        self.f0 = f0
        self.Ztl = Ztl
        self.Zkid = Zkid

    @property
    def C(self):
        return 1/np.sqrt(8*np.pi*self.f0**2*self.Ztl*self.Zkid*self.Qc)

    def ABCD(self, f):
        return np.array([[1, 1/(1j*2*np.pi*f*self.C)], [0, 1]])


class hyKID(object):
    '''Hybrid KID object, consisting of a coupler to transmission line with Ztl,
    Capacitive CPW, Inductive CPW, which is shorted (quarter wave). 
    We fix the design resonance frequency (f0) and coupling quality factor (Qc).
    When we fix also the inductor part fully (and Z0 of the transmission line (TL)),
    the method calc_CapL calculates the length needed for the capacitive part and
    set that to that object.'''

    def __init__(self, Qc, f0, CapCPW, IndCPW, Ztl):
        self.f0 = f0
        self.Qc = Qc
        self.CapCPW = CapCPW
        self.IndCPW = IndCPW
        self.Coupler = Coupler(Qc, f0, Ztl, CapCPW.Z0)

    @property
    def fres_simple(self):
        '''Calculate the approximate resonance frequency with 
        standard 1/4sqrt(LC) formula, for two (lossless) LC circuits in series,
        as derived from ABCD materices (without coupler).'''
        gamma = (self.IndCPW.C * self.IndCPW.L +
                 self.CapCPW.C * self.CapCPW.L + 
                 self.IndCPW.L * self.CapCPW.C)
        delta = self.IndCPW.C * self.IndCPW.L * self.CapCPW.C * self.CapCPW.L
        return 1/(4*np.sqrt(
            gamma/2 + np.sqrt(gamma**2/4 - delta)
        ))

    @property
    def fres(self):
        '''Find the resonance frequency by calculating the input impedance
        of the whole hybrid KID (Coupler, CapCPW, IndCPW) via ABCD matrices and equating
        the imaginary part to 0.'''

        def min_fun(f):
            (A, B), (C, D) = (self.Coupler.ABCD(f) @ self.CapCPW.ABCD(f) @
                              self.IndCPW.ABCD(f))
            return np.abs(np.imag(B / D))
        res = minimize_scalar(min_fun,
                              bounds=(
                                  self.fres_simple *
                                  np.array([1 - 5 / np.sqrt(np.pi * self.Qc),
                                            1 + 5 / np.sqrt(np.pi * self.Qc)])
                              ), method='bounded', options={'xatol': 1e-5})
        return res.x

    @property
    def ak(self):
        '''Calculate the kinetic induction fraction, by temporarily changing
        the inductive CPW central line with PEC, and using the formula:
        ak = 1 - (fres/fres,PEC)^2'''
        # save current parameters
        Qc = self.Qc
        IndSC = copy.copy(self.IndCPW.cSCsheet.SC)
        self.Qc = 1e10  # set Qc to high, to mitigate coupling fres shifts
        fres = self.fres  # get fres
        self.IndCPW.cSCsheet.SC = SC.PEC()   # switch to PEC
        fresPEC = self.fres  # get fres,PEC
        # set parameters back
        self.IndCPW.cSCsheet.SC = IndSC
        self.Qc = Qc
        return 1 - (fres/fresPEC)**2

    def set_CapLen(self):
        '''Iteratively calculate the resonance frequency and adjust the
        lenght of the Capacitive CPW length to reach f0'''
        def f0diff(CapLen, self):
            self.CapCPW.l = CapLen
            return np.abs(self.f0 - self.fres)
        res = minimize_scalar(f0diff, args=(self,))
        self.CapCPW.l = res.x
        return self.CapCPW.l


class KID(object):
    """Object to calculate the input impedance of a KID 
    (coupler + transmission line).
    f0 in Hz, Z0 in Ohm."""

    def __init__(self, fres, Qc, Qi, Z0):
        self.fres = fres
        self.Qc = Qc
        self.Qi = Qi
        self.Z0 = Z0

    def Zin(self, f):
        # from pag.76, eq. (3.18) of PdV's thesis
        dx = (f - self.fres) / self.fres
        return self.Z0 * (
            (
                2 * self.Qi * np.sqrt(self.Qc / np.pi) * dx
                - 1j * np.sqrt(1 / (np.pi * self.Qc))
            )
            / (1 + 2j * self.Qi * (dx - np.sqrt(1 / (np.pi * self.Qc))))
        )

    def ABCD(self, f):
        Zin = self.calc_Zin(f)
        return np.array([[1, 0], [1 / Zin, 1]])
