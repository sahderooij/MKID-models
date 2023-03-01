import sys
sys.path.insert(1, '../../Coding')
import SC

import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.special as sp
import scipy.constants as const
from scipy.optimize import minimize_scalar

'''TODO:
- Find and implement Coupler analytic formula (see Besedin2018, but quite hard)
- open/shorted and quarter/half wave options
- Implement important design characteristics:
    - P_0 (power handling) (bifurcation point)
    - Expected phase noise
    - Expected amplifier noise
    - Expected GR noise
    - Responsivity
'''


class CPW(object):
    """Class to calculate CPW properties, such as vph, Z0 and ak.
    Give S,W and l in µm"""

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
    

class KID(object):
    '''Class to make a CPW KID out of a single CPW line'''
    def __init__(self, f0, Qc, CPW, Ztl):
        self.f0 = f0
        self.Qc = Qc
        self.CPW = CPW
        self.Ztl = Ztl
    
    @property
    def Coupler(self):
        return Coupler(self.Qc, self.f0, self.Ztl, self.CPW.Z0)
    
    @property
    def SensSC(self):
        return self.CPW.SCsheet.SC
    
    @SensSC.setter
    def SensSC(self, SC):
        self.CPW.SCsheet.SC = SC
    
    @property
    def fres_simple(self):
        return 1/(4*np.sqrt(self.CPW.L*self.CPW.C))
    
    def fres(self, plot=False):
        '''Find the resonance frequency by calculating the input impedance
        of the whole hybrid KID (Coupler, CPW, IndCPW) via ABCD matrices and equating
        the imaginary part to 0.'''

        def min_fun(f):
            (A, B), (C, D) = self.ABCD(f)
            return np.abs(np.imag(B / D))
        res = minimize_scalar(min_fun,
                              bounds=(
                                  self.fres_simple *
                                  np.array([1/2,
                                            1])
                              ), method='bounded', options={'xatol': 1e-5})
        if plot:
            plt.figure()
            fs = np.linspace(.5, 2, 1000)*self.fres_simple
            A, B, C, D = np.zeros((4, len(fs)), dtype='complex')
            for i, f in enumerate(fs):
                (A[i], B[i]), (C[i], D[i]) = self.ABCD(f)
            plt.plot(fs, np.abs(np.imag(B/D)))
            plt.axvline(self.fres_simple, linestyle='--', color='r', label='fres_simple')
            plt.axvline(res.x, linestyle='--', color='k', label='fres')
            plt.axvline(self.f0, linestyle='--', color='g', label='f0')
            plt.yscale('log')
            plt.legend()
        return res.x
    
    def ak(self, plot=False):
        '''Calculate the kinetic induction fraction, by temporarily changing
        the CPW line with PEC, and using the formula:
        ak = 1 - (fres/fres,PEC)^2'''
        Qc = self.Qc
        SuCo = copy.copy(self.SensSC)
        self.Qc = 1e5  # set Qc to high, to mitigate coupling fres shifts
        fres = self.fres(plot=plot)  # get fres
        if plot:
            plt.title('fres')
        self.SensSC = SC.PEC()   # switch to PEC
        fresPEC = self.fres(plot=plot)  # get fres,PEC
        if plot:
            plt.title('fres, PEC')
        # set parameters back
        self.SensSC = SuCo
        self.Qc = Qc
        return 1 - (fres/fresPEC)**2
    
    def set_CPWlen(self):
        '''Iteratively calculate the resonance frequency and adjust the
        lenght of the Capacitive CPW length to reach f0'''
        def f0diff(CapLen, self):
            self.CPW.l = CapLen
            return np.abs(self.f0 - self.fres())
        res = minimize_scalar(f0diff, args=(self,))
        self.CPW.l = res.x
        return self.CPW.l
    
    def ABCD(self, f):
        return self.Coupler.ABCD(f) @ self.CPW.ABCD(f)
        
    

class hyKID(KID):
    '''Hybrid KID object, consisting of a coupler to transmission line with Ztl,
    Capacitive CPW, Inductive CPW, which is shorted (quarter wave). 
    We fix the design resonance frequency (f0) and coupling quality factor (Qc).
    When we fix also the inductor part fully (and Z0 of the transmission line (TL)),
    the method calc_CapL calculates the length needed for the capacitive part and
    set that to that object.'''

    def __init__(self, f0, Qc, CapCPW, IndCPW, Ztl):
        KID.__init__(self, f0, Qc, CapCPW, Ztl)
        self.IndCPW = IndCPW
    
    @property
    def SensSC(self):
        return self.IndCPW.cSCsheet.SC
    
    @SensSC.setter
    def SensSC(self, SC):
        self.IndCPW.cSCsheet.SC = SC
        
    @property
    def fres_simple(self):
        '''Calculate the approximate resonance frequency 
        for a (lossless) LC circuit, with the sum of the total inductances and
        capacitances of the individual CPW line.'''
        return 1/(4*np.sqrt((self.CPW.L + self.IndCPW.L)*(self.CPW.C + self.IndCPW.C)))
    
    def ABCD(self, f):
        return self.Coupler.ABCD(f) @ self.CPW.ABCD(f) @ self.IndCPW.ABCD(f)
    

class THzKID(hyKID):
    '''This is the same as a hybrid KID, but with an extra CPW section (called here THzCoupCPW)
    that connects to the antenna/filter/THz coupler.'''
    def __init__(self, Qc, f0, CPW, IndCPW, THzCoupCPW, Ztl):
        hyKID.__init__(self, Qc, f0, CPW, IndCPW, Ztl)
        self.THzCoupCPW = THzCoupCPW
        
    @property
    def fres_simple(self):
        return 1/(4*np.sqrt((self.CPW.L + self.IndCPW.L + self.THzCoupCPW.L)*
                            (self.CPW.C + self.IndCPW.C + self.THzCoupCPW.C)))   
    
    def ABCD(self, f):
        return (self.Coupler.ABCD(f) @ self.CPW.ABCD(f) @ 
                self.IndCPW.ABCD(f) @ self.THzCoupCPW.ABCD(f))



# class KID(object):
#     """Object to calculate the input impedance of a KID 
#     (coupler + transmission line).
#     f0 in Hz, Z0 in Ohm."""

#     def __init__(self, fres, Qc, Qi, Z0):
#         self.fres = fres
#         self.Qc = Qc
#         self.Qi = Qi
#         self.Z0 = Z0

#     def Zin(self, f):
#         # from pag.76, eq. (3.18) of PdV's thesis
#         dx = (f - self.fres) / self.fres
#         return self.Z0 * (
#             (
#                 2 * self.Qi * np.sqrt(self.Qc / np.pi) * dx
#                 - 1j * np.sqrt(1 / (np.pi * self.Qc))
#             )
#             / (1 + 2j * self.Qi * (dx - np.sqrt(1 / (np.pi * self.Qc))))
#         )

#     def ABCD(self, f):
#         Zin = self.calc_Zin(f)
#         return np.array([[1, 0], [1 / Zin, 1]])
