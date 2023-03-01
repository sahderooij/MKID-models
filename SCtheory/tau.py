import SCtheory as SCth
import numpy as np
import scipy.integrate as integrate
from scipy.special import zeta, gamma
import scipy.constants as const

def hbar():
    return const.hbar * 1e12 / const.e

def qpstar(T, SCsheet):
    """Calculates the apparent quasiparticle lifetime from Kaplan. 
    See PdV PhD eq. (2.29)"""
    SC = SCsheet.SC
    D_ = SCth.D(const.Boltzmann / const.e * 1e6 * T, SC)
    nqp_ = SCth.nqp(const.Boltzmann / const.e * 1e6 * T, D_, SC)
    return (
        SC.t0 * SC.N0 * SC.kbTc ** 3 / (4 * nqp_ * D_ ** 2) *
        (1 + SCsheet.tesc / SC.tpb)
    )


def qp_kaplan(kbT, SC):
    """Calculates the intrinsic quasiparticle lifetime w.r.t recombination at E=Delta
    from Kaplan1976 and uses the intral form s.t. it holds for all temperatures.
    Still, a^2F(omega)=b omega^2 is assumed."""
    D_ = SCth.D(kbT, SC)

    def integrand(E, D, kbT):
        return (
            E ** 2
            * (E - D)
            / np.sqrt((E - D) ** 2 - D ** 2)
            * (1 + D ** 2 / (D * (E - D)))
            * (SCth.n(E, kbT) + 1)
            * SCth.f(E - D, kbT)
        )

    return (
        SC.t0
        * SC.kbTc ** 3
        * (1 - SCth.f(D_, kbT))
        / integrate.quad(integrand, 2 * D_, np.inf, args=(D_, kbT))[0]
    )


def scat_kaplan(kbT, SC):
    """Calculates the intrinsic quasiparticle lifetime w.r.t scattering at E=Delta
    from Kaplan1976 and uses the intral form s.t. it holds for all temperatures.
    Still, a^2F(omega)=b omega^2 is assumed."""
    D_ = SCth.D(kbT, SC)

    def integrand(E, D, kbT):
        return (
            E ** 2
            * (E + D)
            / np.sqrt((E + D) ** 2 - D ** 2)
            * (1 - D ** 2 / (D * (E + D)))
            * SCth.n(E, kbT)
            * (1 - SCth.f(E + D, kbT))
        )

    return (
        SC.t0
        * SC.kbTc ** 3
        * (1 - SCth.f(D_, kbT))
        / integrate.quad(integrand, 0, np.inf, args=(D_, kbT))[0]
    )

def esc(kbT, tqpstar, SC):
    """Calculates the phonon escape time, based on tqp* via Kaplan. Times are in µs."""
    D_ = SCth.D(kbT, SC)
    nqp_ = SCth.nqp(kbT, D_, SC)
    return SC.tpb * (
        (4 * tqpstar * nqp_ * D_ ** 2) / (SC.t0 * SC.N0 * SC.kbTc ** 3) - 1
    )



############################### Devereaux and Belitz 1991 #################################
def x(SCsheet):
    return SCsheet.d / SCsheet.SC.xi_DL * np.sqrt(3 / (2 * np.pi))


def F(x):
    return ((np.sinh(x) + np.sin(x)) / 
            (np.cosh(x) - np.cos(x)))


def G(x):
    # note that we have to make sure that arctan(tan(x)) = x 
    phi = x*np.sqrt(np.pi)/2
    fldp = phi // (np.pi/2)
    theta = x/2
    fldt = theta // (np.pi/2)
    return ((4 / (np.pi - 1)) * 
            (np.arctan(np.tan(phi % (np.pi/2))/
                       np.tanh(phi)) + fldp * np.pi/2 - 
            1/np.pi * (np.arctan(np.tan(theta % (np.pi/2))/
                                np.tanh(theta) + fldt * np.pi/2))
            ))


def scat_DB_Coulomb(kbT, SCsheet):
    sc = SCsheet.SC
    Z = sc.lbd_eph + 1
    rhohat = sc.rhon / sc.rhoM
    return hbar()/(sc.D0/Z *  rhohat * 
            (3 * np.pi**(7/3)) / (4*np.sqrt(2)*(1+np.pi)) *
            sc.D0/sc.EF * 
            G(x(SCsheet)) * SCsheet.SC.xi0 / SCsheet.d * 
            (kbT / sc.D0)**(1/2) * 
            np.exp(-sc.D0/kbT) )


def scat_eph2(kbT, SCsheet):
    sc = SCsheet.SC
    Z = sc.lbd_eph + 1
    rhohat = sc.rhon / sc.rhoM
    return (hbar() /
            (sc.D0/Z**2 * (1 + 6/np.pi*(4*sc.lbd_eph-3/2)*rhohat) *
            rhohat**(1/2) * 9*np.pi**2*np.sqrt(6)/5 * 
            gamma(7/2) * zeta(7/2) * sc.lbd_eph * 
            (sc.cL/sc.cT)**4 *
            (sc.D0/sc.kbTD)**2 *
            (sc.D0/sc.EF)**(3/2) * 
            F(x(SCsheet)) * SCsheet.SC.xi0 / SCsheet.d *
            (kbT/sc.D0)**(7/2) * 
            (1 + kbT/sc.D0 * SCsheet.d/sc.xi0 * sc.vF/sc.cT * 
            7/(2*np.pi**2) * zeta(9/2) / zeta(7/2) )))


def scat_DB(kbT, SCsheet):
    return 1/(1/scat_eph2(kbT, SCsheet) + 1/scat_DB_Coulomb(kbT, SCsheet))


def db(SCsheet, cb, D):
    '''Generalized e-ph coupling from Devereaux and Belitz 1991, 
    where we calculate the ion density from the sound velocity
    and Debye frequency.
    Note: their $\rho_i$ is a mass density! (not a number density)'''
    sc = SCsheet.SC
    if D==3:
        rhoi = sc.rho
    elif D==2:
        rhoi = sc.rho * SCsheet.d
    return sc.kF**(D+1) / (16*np.pi*cb*rhoi) * hbar()

def _scat_eph_2D(kbT, sc, Z):
    return (hbar()
            /(2*np.sqrt(2)*sc.D0/sc.EF*(kbT/sc.D0)**(5/2)*sc.D0/Z))

def scat_eph_2D_disorder(kbT, SCsheet):
    sc = SCsheet.SC
    Z = sc.lbd_eph + 1 
    return (_scat_eph_2D(kbT, sc, Z) / (
            db(SCsheet, sc.cT, 2) *         
            sc.vF**3 / sc.cT**3 * (1 + (sc.cT/sc.cL)**4) * 
            gamma(7/2) * zeta(7/2) * np.pi / (2*sc.rhon*1e-2/SCsheet.d/4108)*kbT/sc.EF)
           )

def scat_eph_2D(kbT, SCsheet):
    sc = SCsheet.SC
    Z = sc.lbd_eph + 1 
    return (_scat_eph_2D(kbT, sc, Z) / (
            db(SCsheet, sc.cL, 2) * 
            sc.vF**2 / sc.cL**2 * gamma(5/2) * zeta(5/2))
           )

def _recomb_eph_2D(kbT, sc, Z):
    return (hbar()
            /(8*np.pi*sc.D0/Z*np.sqrt(np.pi*kbT/(2*sc.D0)) * np.exp(-sc.D0/kbT))
           )
            
def recomb_eph_2D_disorder(kbT, SCsheet):
    sc = SCsheet.SC
    Z = sc.lbd_eph + 1
    return (_recomb_eph_2D(kbT, sc, Z) / (
        2*sc.vF**3 * (sc.D0/sc.EF)**2 * db(SCsheet, sc.cT, 2) / sc.cT**3 * (1 + (sc.cT/sc.cL)**4) /
        (sc.rhon*1e-2/SCsheet.d/4108)))
        

def recomb_eph_2D(kbT, SCsheet):
    sc = SCsheet.SC
    Z = sc.lbd_eph + 1
    return (_recomb_eph_2D(kbT, sc, Z) / (
            2*sc.D0/sc.EF * db(SCsheet, sc.cL, 2) * sc.vF**2 / (np.pi * sc.cL**2))
           )

### Gap contribution Y to scattering and recombination in 2 and 3D
#only holds in dirty limit
def _eph2_3D(kbT, SCsheet):
    sc = SCsheet.SC
    Z = sc.lbd_eph + 1 
    rhohat = sc.rhon / sc.rhoM
    mu = 1/2
    return hbar()/(
        sc.D0/Z**2 * (rhohat**.5 + 6*(4*sc.lbd_eph - 3*mu)/np.pi*rhohat**(3/2))
        * db(SCsheet, sc.cT, 3) * 2*np.sqrt(6)/(5*np.sqrt(np.pi)) * sc.vF**4/sc.cT**4
        * (sc.D0/sc.EF)**(7/2) * (kbT/sc.D0)**(3/2)
        * (1 + 4/3*(sc.cT/sc.cL)**5)
    )

def recomb_eph2_3D(kbT, SCsheet):
    sc = SCsheet.SC
    return _eph2_3D(kbT, SCsheet)/(
        np.exp(-sc.D0/kbT)
    )

def _eph2_2D(kbT, SCsheet):
    sc = SCsheet.SC
    Z = sc.lbd_eph + 1 
    rhohat = sc.rhon / sc.rhoM
    mu = 1/2
    Y1 = (sc.D0 * sc.rhon*1e-2/SCsheet.d/4108 * np.log(2*sc.kF*sc.xi_DL)
          * (sc.cL * sc.kF / (8*np.pi*sc.kbTD/hbar()) * sc.lbd_eph 
            * (1 - 2/np.pi*np.arcsin(1 - .5*(sc.kbTD/hbar()/(sc.kF*sc.cL))**2))
             - np.pi**-2)
         )
    return hbar()/(
        sc.D0*(1+Y1/sc.D0)*db(SCsheet, sc.cT, 2) * (sc.vF/sc.cT)**3
        * 4/(np.sqrt(2*np.pi)*Z**2) * (sc.D0/sc.EF)**2 * (kbT/sc.D0)**(3/2)
        * (1+(sc.cT/sc.cL)**4)
    )

def recomb_eph2_2D(kbT, SCsheet):
    sc = SCsheet.SC
    return _eph2_2D(kbT, SCsheet)/(
        np.exp(-sc.D0/kbT)
    )

############################### M. Reizer 2000 #################################
# THIS IS FOR IMPURE METALS (dirty superconductors)

def scat_ee_2D_dirty(kbT, SCsheet):
    '''This time contains eq. 42, 43 and 45 of Reizer2000'''
    sc = SCsheet.SC
    nu = 2*sc.N0*SCsheet.d
    kappa = 2*np.pi*nu*const.e**2 * hbar()/const.epsilon_0 * 1e12/const.e
    
    # these rates have units of energy
    scatrate42 = (kbT / (2*np.pi**2*sc.D*nu) * 
                  (1 + sc.lbd_eph**2) * np.exp(-sc.D0/kbT)
                 )
    scatrate43 = (21 / (2*np.pi**6) * np.sqrt(np.pi/2) 
                  * (kbT/sc.D0)**(3/2) * kbT**2 / (sc.D**2*kappa**2*nu))
    scatrate45 = ((np.pi*sc.D0**2*kbT)**(1/3)/(2*np.pi**2*sc.D*nu)**(3/2)
                  * np.exp(-2*sc.D0/(3*kbT))) / hbar()**(3/2)
    
    return hbar()/(
        scatrate42 + scatrate43 + scatrate45
    )
                  

def scat_ee_3D_dirty(kbT, SCsheet):
    sc = SCsheet.SC
    return hbar()/(
        12*np.sqrt(kbT) * sc.D0 * np.exp(-4*sc.D0/(5*kbT)) 
        / (hbar()**(3/2) * np.pi * (np.pi*sc.D)**(3/2) * 2*sc.N0)
    )

def recomb_ee_2D_dirty(kbT, SCsheet):
    '''Recombination term for 2D e-e interaction Reizer2000.
    NOTE: the lambda in Reizer2000 is Vsc, not N0*Vsc'''
    sc = SCsheet.SC
    return hbar()/(
        kbT/(4*np.pi*sc.D*sc.N0*2*SCsheet.d) * (1 + sc.lbd_eph**2) * np.exp(-2*sc.D0/kbT)
    )

def recomb_ee_3D_dirty(kbT, SCsheet):
    sc = SCsheet.SC
    return hbar()/(
        sc.D0**.5 * kbT / (2**(1/4)*(2*np.pi)**2 * sc.D**(3/2) * 2*sc.N0)
        * (1 + sc.lbd_eph**2) * np.exp(-2*sc.D0/kbT)/hbar()**(3/2)
    )

############################### M. Reizer A.V. Sergeyev 1986 #################################
def eph_3D_disorder(kbT, sc):
    '''Electron-phonon inelastic scattering time for disordered normal metals.
    From RezierSergeyev1986, eq. 31. Note the typo in eq. 30!! that must be ^3 in the 
    denominator; check Schmid1973 and SergeevMitin2000'''
    beta = (2 * sc.EF / 3)**2 * sc.N0 / (2 * sc.rho * sc.cL**2)
    return hbar()**4/(
        np.pi**4 * beta / 5 * sc.kF * sc.l_e * kbT**4 / (sc.kF * sc.cL)**3
        * (1 + 3/2*(sc.cL/sc.cT)**5)
    )

def recomb_eph_3D_disorder(kbT, SCsheet):
    sc = SCsheet.SC
    return eph_3D_disorder(sc.kbTc, sc)/(
        np.sqrt(np.pi) * (2*sc.D0/sc.kbTc)**(7/2) * (kbT/sc.kbTc)**.5
        * np.exp(-sc.D0/kbT)
        * 4*np.pi**2
    )

def scat_eph_3D_disorder(kbT, SCsheet):
    sc = SCsheet.SC
    return eph_3D_disorder(sc.kbTc, sc)/(
        gamma(9/2) * zeta(9/2) * (sc.kbTc/(2*sc.D0))**.5
        * (kbT/sc.kbTc)**(9/2)
        * 4*np.pi**2
    )

########################### M. Reizer 1998 ####################################################
# THIS IS FOR 2D, CLEAN METALS
def _B(SCsheet):
    sc = SCsheet.SC
    nu = 2*sc.N0*SCsheet.d
    kappa = 2*np.pi*nu*const.e**2*hbar()/const.epsilon_0 * 1e12/const.e
    pF = 2*sc.EF/sc.vF
    return (np.log(4*sc.EF/((np.sqrt(8) + np.sqrt(3))*sc.D0))
            - np.log((2*pF + kappa)/kappa)
            - 4*pF/(2*pF + kappa))

def scat_ee_2D(kbT, SCsheet):
    sc = SCsheet.SC
    nu = 2*sc.N0*SCsheet.d
    kappa = 2*np.pi*nu*const.e**2*hbar()/const.epsilon_0 * 1e12/const.e
    A1 = 1 + (1 + 2/np.pi)*sc.lbd_eph**2
    return hbar()/(A1 * _B(SCsheet) * sc.D0/(4*sc.EF) 
            * np.sqrt(2*np.pi*sc.D0*kbT) 
            * np.exp(-sc.D0/kbT))

def recomb_ee_2D(kbT, SCsheet):
    sc = SCsheet.SC
    A2 = (1 + sc.lbd_eph**2)*(1 + np.sqrt(2)/4) + np.sqrt(2) * sc.lbd_eph**2
    return hbar()/(
        A2*_B(SCsheet)*kbT*sc.D0/(2*sc.EF)*np.exp(-2*sc.D0/kbT)
    )

    
    
    
    