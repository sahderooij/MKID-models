'''This module implements usefull superconducting theory, 
needed to predict KID behaviour and quasiparticle dynamics. 
Based on the PhD thesis of PdV.
If required, units are in micro: µeV, µm, µs etc.'''

import os
import numpy as np
import scipy.integrate as integrate
from scipy import interpolate
from scipy.optimize import minimize_scalar as minisc
import warnings

def f(E, kbT):
    '''The Fermi-Dirac distribution.'''
    with np.errstate(over='raise',under='ignore'):
        try:
            return 1 / (1 + np.exp(E/kbT))
        except FloatingPointError: #use low temperature approx. if normal fails.
            return np.exp(-E/kbT)

def cinduct(hw, D, kbT):
    '''Mattis-Bardeen equations.'''
    def integrand11(E, hw, D, kbT):
        nume = 2 * (f(E, kbT) - f(E + hw, kbT)) * (E ** 2 + D ** 2 + hw * E)
        deno = hw * ((E ** 2 - D ** 2) * ((E + hw) ** 2 - D ** 2)) ** 0.5
        return nume / deno
        
    def integrand12(E, hw, D, kbT):
        nume = (1 - 2 * f(E + hw, kbT)) * (E ** 2 + D ** 2 + hw * E)
        deno = hw * ((E ** 2 - D ** 2) * ((E + hw) ** 2 - D ** 2)) ** 0.5
        return nume / deno

    def integrand2(E, hw, D, kbT):
        nume = (1 - 2 * f(E + hw, kbT)) * (E ** 2 + D ** 2 + hw * E)
        deno = hw * ((D ** 2 - E ** 2) * ((E + hw) ** 2 - D ** 2)) ** 0.5
        return nume / deno

    s1 = integrate.quad(integrand11, D, np.inf, args=(hw, D, kbT))[0]
    if hw > 2 * D:
        s1 -= integrate.quad(integrand12, D - hw, -D, args=(hw, D, kbT))[0]
    s2 = integrate.quad(integrand2, np.max(
        [D - hw, -D]), D,args=(hw, D, kbT))[0]
    return s1,s2

def Vsc(kbTc,N0,kbTD):
    '''Calculates the superconducting coupling strength in BSC-theory 
    from the BSC relation 2D=3.52kbTc.'''
    D0 = 1.76*kbTc # BSC-relation
    def integrand1(E, D):
        return 1/np.sqrt(E**2-D**2)
    return 1/(integrate.quad(integrand1, D0, kbTD,
                                 args=(D0,))[0]*N0)

def load_Ddata(N0,kbTc,kbTD,kb=86.17):
    '''To speed up the calculation for Delta, an interpolation of generated values is used.
    Ddata_{SC}_{Tc}.npy contains this data, where a new one is needed 
    for each superconductor (SC) and crictical temperature (Tc).
    This function returns the Ddata array.'''
    
    Ddataloc = os.path.dirname(__file__)+ '/Ddata/'
    if (N0 == 1.72e4) & (kbTD == 37312.0):
        SC = 'Al'
    elif (N0 == 4.08e4) & (kbTD == 86.17*246):
        SC = 'Ta'
    else:
        SC = None
    
    if SC is not None:
        Tc = str(kbTc/kb).replace('.','_')
        try:
            Ddata = np.load(Ddataloc + 
                f'Ddata_{SC}_{Tc}.npy')
        except FileNotFoundError:
            Ddata = None
    else: 
        Ddata = None
    return Ddata

def D(kbT, N0, kbTc, kbTD,kb=86.17):
    '''Calculates the thermal average energy gap, Delta. Tries to load Ddata, 
    but calculates from scratch otherwise. Then, it cannot handle arrays.  '''
    Ddata = load_Ddata(N0,kbTc,kbTD)
    if Ddata is not None:
        Dspl = interpolate.splrep(Ddata[0, :], Ddata[1, :], s=0)
        return np.clip(interpolate.splev(kbT, Dspl),0,None)
    else:
        warnings.warn('D takes long.. \n N0={}\n kbTD={}\n Tc={}'.format(N0,kbTD,kbTc/kb))
        _Vsc = Vsc(kbTc,N0,kbTD)
        def integrandD(E, D, kbT, N0, _Vsc):
            return N0 * _Vsc * (1 - 2 * f(E, kbT)) / np.sqrt(E ** 2 - D ** 2)

        def dint(D, kbT, N0, _Vsc, kbTD):
            return np.abs(
                integrate.quad(integrandD, D, kbTD,
                               args=(D, kbT, N0, _Vsc))[0] - 1
            )

        res = minisc(dint, args=(kbT, N0, _Vsc, kbTD))
        if res.success:
            return np.clip(res.x,0,None)

def nqp(kbT, D, N0):
    '''Thermal average quasiparticle denisty. It can handle arrays 
    and uses a low temperature approximation, if appropriate.'''
    if (kbT<D/20).all():
        return 2*N0*np.sqrt(2*np.pi*kbT*D)*np.exp(-D/kbT)
    else:
        def integrand(E, kbT, D, N0):
            return 4 * N0 * E / np.sqrt(E ** 2 - D ** 2) * f(E, kbT)
        if any([type(kbT) is float, type(D) is float,
               type(kbT) is np.float64, type(D) is np.float(64)]):#make sure it can deal with kbT,D arrays
            return integrate.quad(integrand, D, np.inf, args=(kbT, D, N0))[0]
        else:
            assert (kbT.size == D.size),'kbT and D arrays are not of the same size'
            result = np.zeros(len(kbT))
            for i in range(len(kbT)):
                result[i] = integrate.quad(
                    integrand, D[i], np.inf, args=(kbT[i], D[i], N0))[0]
            return result
            
def kbTeff(N_qp, N0, V, kbTc, kbTD):
    '''Calculates the effective temperature (in µeV) at a certain 
    number of quasiparticles.'''
    Ddata = load_Ddata(N0,kbTc,kbTD)
    if Ddata is not None:
        kbTspl = interpolate.splrep(Ddata[2,:],Ddata[0,:])
        return interpolate.splev(N_qp/V,kbTspl)
    else:
        def minfunc(kbT, N_qp, N0, V, kbTc, kbTD):
            Dt = D(kbT, N0, kbTc, kbTD)
            return np.abs(nqp(kbT, Dt, N0) - N_qp/V)
        res = minisc(
            minfunc,
            bounds = (0,1*86.17),
            args=(N_qp, N0, V, kbTc, kbTD), 
            method="bounded",
            options = {'xatol':1e-15}
        )
        if res.success:
            return res.x
    
def beta(lbd0, d, D, D0, kbT):
    '''calculates beta, a measure for how thin the film is, 
    compared to the penetration depth.
    d -- film thickness
    D -- energy gap
    D0 -- energy gap at T=0
    kbT -- temperature in µeV'''
    lbd = lbd0 * 1 / np.sqrt(D / D0 * np.tanh(D / (2 * kbT)))
    return 1 + 2 * d / (lbd * np.sinh(2 * d / lbd))


def Qi(s1, s2, ak, lbd0, d, D, D0, kbT):
    '''Calculates the internal quality factor, 
    from the complex conductivity. See PdV PhD thesis eq. (2.23)'''
    b = beta(lbd0, d, D, D0, kbT)
    return 2 * s2 / (ak * b * s1)


def hwres(s2, hw0, s20, ak, lbd0, d, D, D0, kbT):
    '''Gives the resonance frequency in µeV, from the sigma2,
    from a linearization from point hw0,sigma20. See PdV PhD eq. (2.24)'''
    b = beta(lbd0, d, D, D0, kbT)
    return hw0 * (
        1 + ak * b / 4 / s20 * (s2 - s20)
    )  # note that is a linearized approach

def S21(Qi, Qc, hwread, dhw, hwres):
    '''Gives the complex transmittance of a capacatively coupled
    superconducting resonator (PdV PhD, eq. (3.21)), with:
    hwread -- the read out frequency
    dhw -- detuning from hwread (so actual read frequency is hwread + dhw)
    hwres -- resonator frequency'''
    Q = Qi * Qc / (Qi + Qc)
    dhw += hwread - hwres
    return (Q / Qi + 2j * Q * dhw / hwres) / (1 + 2j * Q * dhw / hwres)

def hwread(hw0, kbT0, ak, lbd0, d, D_, D0, kbT, N0, kbTc, kbTD):
    '''Calculates at which frequency, on probes at resonance. 
    This must be done iteratively, as the resonance frequency is 
    dependent on the complex conductivity, which in turn depends on the
    read frequency.'''
    D_0 = D(kbT0, N0, kbTc, kbTD)
    s20 = cinduct(hw0, D_0, kbT0)[1]

    def minfuc(hw, hw0, s20, ak, lbd0, d, D_, D0, kbT):
        s1, s2 = cinduct(hw, D_, kbT)
        return np.abs(hwres(s2, hw0, s20, ak, lbd0, d, D_, D0, kbT) - hw)

    res = minisc(
        minfuc,
        bracket=(.5*hw0,hw0,2*hw0),
        args=(hw0, s20, ak, lbd0, d, D_, D0, kbT),
        method="brent",
        options={"xtol": 1e-21},
    )
    if res.success:
        return res.x
    
def calc_Nwsg(kbT,V,D,e):
    '''Calculates the number of phonons with the Debye approximation, 
    for Al.'''
    def integrand(E,kbT,V):
        return 3*V*E**2/(2*np.pi*(6.582e-4)**2*(6.3e3)**3*(np.exp(E/kbT)-1))
    return integrate.quad(integrand,e+D,2*D,args=(kbT,V))[0]

def tau_kaplan(T,tesc=.14e-3, 
               t0=.44,
               kb = 86.17,
               tpb = .28e-3,
               N0 = 1.72e4,
               kbTc = 1.2*86.17,
               kbTD = 37312.0,):
    '''Calculates the apparent quasiparticle lifetime from Kaplan. 
    See PdV PhD eq. (2.29)'''
    D0 = 1.76 * kbTc #BSC
    D_ = D(kb*T, N0, kbTc, kbTD)
    nqp_ = nqp(kb*T, D_, N0)
    taukaplan = t0*N0*kbTc**3/(4*nqp_*D_**2)*(1+tesc/tpb) 
    return taukaplan

def kbTbeff(tqpstar,
    V=1000,
    t0=.44,
    kb=86.17,
    tpb=.28e-3,
    N0=1.72e4,
    kbTc = 1.2 * 86.17,
    kbTD=37312.0,
    tesc=0.14e-3,
    plot=False):
    '''Calculates the effective temperature, with a certian 
    quasiparticle lifetime.'''
    D0 = 1.76 * kbTc 
    Nqp_0 = V * t0 * N0 * kbTc ** 3 / \
        (2 * D0 ** 2 * tqpstar) * 0.5 * (1 + tesc / tpb)
    
    return kbTeff(Nqp_0, N0, V, kbTc, kbTD)

def tesc(
    kbT,
    tqpstar,
    t0=.44,
    tpb=.28e-3,
    N0=1.72e4,
    kbTc=1.2 * 86.17,
    kbTD=37312.0
):
    '''Calculates the phonon escape time, based on tqp* via Kaplan. Times are in µs.'''
    
    D_ = D(kbT,N0,kbTc,kbTD)
    nqp_ = nqp(kbT, D_, N0)
    return tpb*((4*tqpstar*nqp_*D_**2)/(t0*N0*kbTc**3)-1)

def nqpfromtau(tau,
               tesc=0,
               kbTc=1.2*86.17,
               t0=.44,
               tpb=.28e-3,
               kb=86.17,
               N0=1.72e4):
    '''Calculates the density of quasiparticles from the quasiparticle lifetime.'''
    D_ = 1.76*kbTc
    return t0*N0*kbTc**3/(2*D_**2*2*tau/(1+tesc/tpb))