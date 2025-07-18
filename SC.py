"""This module includes the physical constants needed for the calculations. Instances of the SC class containt superconductor specific constants, such as the Debye Temperature (TD), Density of States at Fermi surface (N0) etc. """

import scipy.integrate as integrate
import scipy.constants as const
import os
import numpy as np
import warnings
import SCtheory as SCth


class Superconductor(object):
    '''General class for superconductor material properties.
    Free electron model is assumed.'''

    def __init__(self, name, Tc, TD, N0, rhon, kF,  t0, tpb, cT, cL, rho):
        self.name = name
        self.kbTc = Tc * const.Boltzmann / const.e * 1e6  # critical Temperature in µeV
        self.kbTD = TD * const.Boltzmann / const.e * 1e6  # Debye Energy in µeV
        self.N0 = N0  # Electronic DoS at Fermi Surface [µeV^-1 µm^-3], single spin
        self.rhon = rhon  # normal state resistivity [µOhmcm] (!)
        self.kF = kF  # Fermi wave number [µm^-1]
        self.t0 = t0  # electron-phonon interaction time [µs]
        self.cT = cT # transverse speed of sound [µm/µs]
        self.cL = cL # longitudinal speed of sound [µm/µs]
        self.rho = rho/const.e*1e-12 # mass density give in kg/m^3, returns in µeV/(µm/µs)^2 µm^-3
        if tpb is None:
            self.tpb = (1/self.tpb_L +1/self.tpb_T)**-1 # phonon pair-breaking time [µs]
        else:
            self.tpb = tpb

        if t0 is None:
            self.t0 = self.tau0_clean
        
    # electronic properties
        
    @property
    def mstar(self):
        '''effective electron mass in µeV/(µm/µs)^2'''
        return 2 * (const.hbar*1e12/const.e)**2 * self.N0 * np.pi**2/self.kF
    
    @property
    def mstar_rel(self):
        '''effective electron mass relative to free electron mass'''
        return self.mstar / (const.m_e / const.e * 1e6)
        
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
    def rhoM(self):
        '''The Mott resisitivty in µOhm cm'''
        return 3 * np.pi**2 * const.hbar / (const.e**2 * self.kF * 1e6) * 1e8
    
    @property
    def n(self):
        '''charge density (in 1/µm^3)'''
        return self.kF**3 / (3 * np.pi**2)
    
    @property
    def tau(self):
        '''Elastic scattering time, in µs'''
        return self.l_e/self.vF
        
    
    # electron-phonon properties (incl. superconductor properties)

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
        """To speed up the calculation for Delta, 
        an interpolation of generated values is used.
        Ddata_{SC}_{Tc}.npy contains this data. A new data file is needed 
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
        return 1.764 * self.kbTc
    
    @property
    def lbd0(self):
        """London penetration depth (i.e. at T = 0 K) in µm. 
        This only holds when Mattis-Bardeen can be used 
        (i.e. dirty limit or extreme anomalous limit)"""
        return np.sqrt(self.l_e #µm
            * self.rhon * 1e-2 #Ohm µm
            / (const.mu_0 # Ohm µs/µm
               * self.vF # µm/µs)
              ))
    
    @property
    def lbd_eff(self):
        '''Effective penetration depth (µm) in local approximation, at T=0.
        See Tinkham p.102'''
        return self.lbd0*np.sqrt(1+self.xi0/self.l_e)

    @property
    def xi0(self):
        """BSC (Pippard) Coherence length at T = 0 K (in [µm])"""
        return const.hbar * 1e12 / const.e * self.vF / (np.pi * self.D0)
    
    @property
    def xi_DL(self):
        '''Dirty limit coherence length in µm'''
        if self.xi0/self.l_e > 10:  
            return np.sqrt(self.xi0 * self.l_e)
        else:
            warnings.warn(f'Not in dirty limit xi0={self.xi0}, l={self.l_e}')
            
    @property
    def kappa_DL(self):
        '''Dirty limit Ginzberg-Landau parameter at Tc. See Tinkham p.120'''
        if self.xi0 < self.l_e:
            warnings.warn(f'Not in de dirty limit: xi={xi0}, l_e={self.l_e}')
        return .715 * self.lbd0/self.l_e
    
    @property
    def Bc1(self):
        '''Lower critical field with Ginzberg-Landau with kappa>>1, near Tc'''
        if self.kappa_DL < 10:
            warnings.warn(f'Not in the high kappa limit: kappa={self.kappa_DL}')
        return const.Planck/(2*const.e) / (4*np.pi*(self.lbd_eff*1e-6)**2) * np.log(self.kappa_DL)
            
    @property
    def Bc2(self):
        '''Upper critical field at T=0 in T, calculated with dirty limit coherence length'''
        return const.Planck/(2*const.e) / (2* np.pi * (self.xi_DL*1e-6)**2)
    
    @property
    def jc(self):
        """Critical current density, in A/µm^2, from Romijn1982"""
        return 2.75*np.sqrt(
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
    def lbd_eph_clean(self):
        '''Electron-phonon coupling constant for a clean metal, see Keck and Schmid 1976 eq. 49.
        If the Debye energy is taken out of this expression (i.e. divide by kbTD**2), you get the constant 
        b in Kaplan1976 for a clean metal (a^2F(w) = b Omega^2) see eq. 45 in Keck and Schmid.'''
        return self.N0*self.kbTD**2*self.vF**2/(36 * self.rho * self.cL**4)

    @property
    def tau0_clean(self):
        '''Calculates the electron-phonon coupling time with eq. (10) of Kaplan1976, 
        with Z' = 1 + lbd_eph_clean and b=lbd_eph_clean/kbTD**2 
        (see also Keck and Schmid 1979 and Kozorezov2000)'''
        return ((1 + self.lbd_eph_clean) 
                * (self.kbTD/self.kbTc)**2 
                * const.hbar * 1e12 / const.e 
                / (2*np.pi * self.kbTc))

    @property
    def R(self):
        '''Recombination constant in µm^3/µs'''
        return 2*self.D0**2/(self.kbTc**3 * self.t0 * self.N0)
        
    # phonon properties
    
    @property
    def cs(self):
        '''effective 3D speed of sound'''
        return (3/(2/self.cT**3 + 1/self.cL**3))**(1/3)
    
    @property
    def phmfp(self):
        '''averaged phonon mean free path against pair-breaking (in µm)'''
        return self.cs * self.tpb
    
    @property
    def phmfp_L(self):
        '''longitudinal phonon mean free path against pair-breaking (in µm)'''
        return self.cL * self.tpb_L
    
    @property
    def phmfp_T(self):
        '''transverse phonon mean free path against pair-breaking (in µm)'''
        return self.cT * self.tpb_T
    
    @property
    def qLl_e(self):
        '''Longitudinal 2Delta (recombination) phonon disorder parameter (i.e. phonon wave number (2Delta) times the 
        electronic mean free path). If this is much smaller than 1, the superconductor is 
        disordered w.r.t. electron-phonon interaction.'''
        return self.l_e * 2*self.D0 / (const.hbar*1e12/const.e * self.cL)
    
    @property
    def qTl_e(self):
        '''Transverse 2Delta (recombination) phonon disorder parameter.'''
        return self.l_e * 2*self.D0 / (const.hbar*1e12/const.e * self.cT)

    def _tint_L(self, ql):
        return 1 / (self.n * self.mstar / (self.rho * self.tau) 
                * (ql**2 * np.arctan(ql) / (3 * (ql - np.arctan(ql))) - 1) )

    def _tint_T(self, ql):
        zeta = 3 / (2 * ql**3) * ((1 + ql**2)*np.arctan(ql) - ql)
        return 1 / (self.n * self.mstar / (self.rho * self.tau) 
                * (1 / zeta - 1) )
    
    @property
    def tpb_L(self):
        '''2Delta longitudinal phonon inelastic scattering time, from Pippard1955, Kittel1987.
        Works in both the l_e q >> 1 and l_e q << 1 regimes. omega * tau << 1 is assumed, 
        which states that 2Delta/hbar should be much bigger than tau, which is always the case. '''
        return self._tint_L(self.qLl_e)
    
    @property
    def tpb_T(self):
        '''2Delta transverse phonon inelastic scattering time.'''
        return self._tint_T(self.qTl_e)
    
    def qLkbTl_e(self, kbT):
        '''Longitudinal disorder parameter for thermal (scattering) phonons'''
        return self.l_e * kbT / (const.hbar*1e12/const.e * self.cL)
    
    def qTkbTl_e(self, kbT):
        '''Transverse disorder parameter for thermal (scattering) phonons'''
        return self.l_e * kbT / (const.hbar*1e12/const.e * self.cT)

    def tint_T(self, kbT):
        '''Interaction time for a kbT transverse phonon, form Pippard1955, Kittel1987'''
        return self._tint_T(self.qTkbTl_e(kbT))

    def tint_L(self, kbT):
        '''Interaction time for a kbT longitudinal phonon, form Pippard1955, Kittel1987'''
        return self._tint_L(self.qLkbTl_e(kbT))


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

# Constants from Kaplan1976, Kaplan1979
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


# tau0 is calculated from the power law GR noise lifetimes in LT192,
#     CPWs with tau0 = 4.2 tau_GR(Tc) (see Kaplan1979)
# N0 and kF are calculated with the free electron model, 
#     with the results from the magnetoresistance experiments of LT278.
# The rest are constants for Ta from Abadias2019 and Magnuson2019
bTa = Superconductor('bTa', 
                     Tc=.87,
                     rhon=206.,
                     TD=221,
                     N0=2.06e4,
                     kF=1.4e4,
                     t0=81e-3, 
                     tpb=None, 
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
                     tpb=.0227e-3,
                     cL=3.9e3,
                     cT=2.01e3,
                     rho=17.1e3)

# Film C of Coumou2013 and Kardakova2015
#  The Debye temperature is for T = 300 K,
#        https://doi.org/10.1016/S1006-7191(08)60082-4
# density from Hansen2020
# N0 is from Dridi2002
TiN = Superconductor('TiN',
                     Tc=2.7,
                     rhon=253.,
                     TD=579.2,
                     N0=6.3e4,
                     kF=1.05e4,
                     t0=4.2*23e-3,
                     tpb=None,
                     cL=np.nan,
                     cT=np.nan,
                     rho=5.7e3)

# Sidorova2021
NbTiN = Superconductor('NbTiN',
                       Tc=15.1,
                       rhon=115.,
                       TD=np.nan,
                       N0=3.7e4,
                       kF=1.17e4,
                       t0=np.nan,
                       tpb=np.nan,
                       cL=np.nan,
                       cT=np.nan,
                       rho=3e3)

# from Gershenzon1990, film 10 
# kF from Ashcroft&Mermin
Nb = Superconductor('Nb',
                    Tc=8.5,
                    rhon=15,
                    TD=276,
                    N0=7.9e4,
                    kF=2.55e4,
                    t0=np.nan,
                    tpb=np.nan,
                    cL=np.nan,
                    cT=np.nan,
                    rho=np.nan)

# Sidorova2020, film M-2259
NbN = Superconductor('NbN', 
                     Tc=10.74, 
                     rhon=265,
                     TD=176,
                     N0=2.5e4,
                     kF=np.nan,
                     t0=np.nan,
                     tpb=np.nan,
                     cL=np.nan,
                     cT=2.42e3,
                     rho=7.8e3)
                    
                    
                    
                    
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
        '''Returns the normal state sheet resistance in µOhm/sq'''
        return self.SC.rhon / (self.d * 1e-4)

    @property
    def Lks(self):
        '''Returns the sheet inductance in pH/sq'''
        return const.hbar * 1e12 / const.e * self.Rs / (np.pi * self.SC.D0)
    
    @property
    def phtrf(self):
        '''phonon trapping factor, as defined by (1 + tau_esc/tau_pb)'''
        return 1 + self.tesc/self.SC.tpb
    
    @property
    def lbd_Pearl(self):
        '''Pearl length in µm. Checks if the thickness
        is less then the London penetration depth'''
        if self.SC.lbd_eff < self.d:
            warnings.warn(f'''Effective penetration depth is less then thickness: 
            \n lbd0={self.SC.lbd0} µm, d={self.d} µm''')
        return self.SC.lbd_eff**2/self.d
    
    @property
    def qLd(self):
        return self.d * 2*self.SC.D0 / (const.hbar*1e12/const.e * self.SC.cL)
    
    @property
    def qTd(self):
        return self.d * 2*self.SC.D0 / (const.hbar*1e12/const.e * self.SC.cT)

    @property
    def Rbar(self):
        '''Recombination constant, renormalized for phonon trapping in µm^3/µs'''
        return self.SC.R / self.phtrf
    
    def qLkbTd(self, kbT):
        return self.d * kbT / (const.hbar*1e12/const.e * self.SC.cL)
    
    def qTkbTd(self, kbT):
        return self.d * kbT / (const.hbar*1e12/const.e * self.SC.cT)

    def Zs(self, kbT, hw):
        '''Returns the complex sheet impedance of the film, in the dirty limit. In µOhm/sq
        from Kautz1978'''
        s1, s2 = SCth.cinduct(hw, SCth.D(kbT, self.SC, useD0lowT=True), kbT)
        sigma = (s1 - 1j*s2) / self.SC.rhon * 1e-4  # 1/(µOhm µm)
        omg = hw/(const.hbar/const.e*1e12) # 1/(µs)
        return np.sqrt(1j*const.mu_0*1e6 * omg / sigma) / np.tanh(np.sqrt(1j*omg*const.mu_0*1e6*sigma)*self.d)        

    def set_tesc(self, Chipnum, KIDnum, **kwargs):
        """sets the tesc attribute to a value calculated from the kidcalc.calc.tesc function,
        which uses GR noise lifetimes at high temperatures."""
        import kidata

        tesc, tescerr = kidata.calc.tesc(
            Chipnum, KIDnum, self, reterr=True, **kwargs)
        self.tesc = tesc
        self.tescerr = tescerr

    

class Wire(Sheet):
    '''A superconducting wire with thickness d and width w'''
    
    def __init__(self, SC, d, w, tesc=1e-12, tescerr=0):
        super().__init__(SC, d, tesc, tescerr)
        self.w = w

    @property
    def Ic(self):
        '''Critical current in A'''
        return self.SC.jc * self.w * self.d

    @property
    def Bc(self):
        '''Critical field in T for Pearl vortices, see Bronson2006.'''
        if not ((self.lbd_Pearl > self.w/10) and (self.w > self.SC.xi_DL/10)):
            warnings.warn('Lambda_Pearl >> W >> xi does not hold:\n' + 
                          f'lbd_Preal={self.lbd_Pearl} µm\n' + 
                          f'W={self.w} µm\n' + 
                          f'xi_DL={self.SC.xi_DL} µm\n')

        return (2 * (const.Planck/(2*const.e)) / (np.pi * (self.w*1e-6)**2)
                * np.log(2*self.w/(np.pi*self.SC.xi_DL)))
        
class Vol(Wire):
    '''A superconducting sheet of volume V, width w and lenght l'''

    def __init__(self, SC, d, w, l=None, V=None, tesc=1e-12, tescerr=0):
        super().__init__(SC, d, w, tesc, tescerr)
        self.w = w
        if V is None:
            self.l = l
            self.V = self.w * self.l * self.d
        elif l is None:
            self.V = V
            self.l = self.V / (self.w * self.d)
            

    def checkV(self):
        return self.w * self.l * self.d == self.V
            

###############################################################################


def init_SCvol(Chipnum, KIDnum, SC_class=Al, rhon=np.nan, w=np.nan, set_tesc=False, **tesckwargs):
    """This function returns an Volume instance (which is defined with the SC_class
    argument), with the parameters initialized from data.
    Tc, V, and d (thickness) are from the S21 measurement, and tesc is set with
    the set_tesc method which uses kidata.calc.tesc() to estimate the phonon
    escape time."""
    import kidata

    S21data = kidata.io.get_S21data(Chipnum, KIDnum)
    SCvol = Vol(SC_class, w=w, V=S21data[0, 14], # use the area, i.e. V/d as width and length=1
                d=S21data[0, 25])
    SCvol.SC.kbTc = 86.17 * S21data[0, 21]
    SCvol.SC.rhon = rhon
    SCvol.w = w
    
    if set_tesc:
        SCvol.set_tesc(Chipnum, KIDnum, **tesckwargs)
    return SCvol
