"""This module implements usefull superconducting theory, 
needed to predict KID behaviour and quasiparticle dynamics. 
Based on the PhD thesis of PdV.
If required, units are in micro: µeV, µm, µs etc."""

import os
import numpy as np
import scipy.integrate as integrate
from scipy import interpolate
import scipy.constants as const
from scipy.optimize import minimize_scalar as minisc
from scipy.optimize import root_scalar
import warnings
from multiprocess import Pool

import SCtheory.tau, SCtheory.noise, SCtheory.Usadel, SCtheory.qpredistr
import SC


def f(E, kbT):
    """The Fermi-Dirac distribution."""
    with np.errstate(over="raise", under="ignore"):
        try:
            return 1 / (np.exp(E / kbT) + 1)
        except FloatingPointError:  # use low temperature approx. if normal fails.
            return np.exp(-E / kbT)


def n(E, kbT):
    """The Bose-Einstein distribution."""
    with np.errstate(over="raise", under="ignore"):
        try:
            return 1 / (np.exp(E / kbT) - 1)
        except FloatingPointError:  # use low temperature approx. if normal fails.
            return np.exp(-E / kbT)


def _cinduct(hw, kbT, SC, useD0lowT=False):
    """Mattis-Bardeen equations."""
    D_ = D(kbT, SC, useD0lowT)

    def integrand11(E, hw, D_, kbT):
        nume = 2 * (f(E, kbT) - f(E + hw, kbT)) * np.abs(E**2 + D_**2 + hw * E)
        deno = hw * ((E**2 - D_**2) * ((E + hw)**2 - D_**2))**0.5
        return nume / deno

    def integrand12(E, hw, D_, kbT):
        nume = (1 - 2 * f(E + hw, kbT)) * np.abs(E**2 + D_**2 + hw * E)
        deno = hw * ((E**2 - D_**2) * ((E + hw)**2 - D_**2))**0.5
        return nume / deno

    def integrand2(E, hw, D_, kbT):
        nume = (1 - 2 * f(E + hw, kbT)) * np.abs(E**2 + D_**2 + hw * E)
        deno = hw * ((D_**2 - E**2) * ((E + hw)**2 - D_**2))**0.5
        return nume / deno

    s1 = integrate.quad(integrand11, D_, np.inf, args=(hw, D_, kbT))[0]
    if hw > 2 * D_:
        s1 += integrate.quad(integrand12, D_ - hw, -D_, args=(hw, D_, kbT))[0]
    s2 = integrate.quad(integrand2,
                        np.max([D_ - hw, -D_]),
                        D_,
                        args=(hw, D_, kbT))[0]
    return s1, s2


def cinduct(hw, kbT, SC, useD0lowT=False):
    if (isinstance(kbT, (int, float, complex)) 
        and isinstance(hw, (int, float, complex))):
        return _cinduct(hw, kbT, SC, useD0lowT)
    else:
        if isinstance(hw, (int, float, complex)):
            hw = np.ones(len(kbT)) * hw
        elif isinstance(kbT, (int, float, complex)):
            kbT = np.ones(len(hw)) * kbT
        s1, s2 = np.zeros((2, len(kbT)))
        with Pool() as p:
            for i, res in enumerate(
                    p.imap(lambda x: _cinduct(*x, SC, useD0lowT), zip(hw,
                                                                      kbT))):
                s1[i], s2[i] = res
        return s1, s2


def D(kbT, SC, useD0lowT=False):
    """Calculates the thermal average energy gap, Delta. Tries to load Ddata,
    but calculates from scratch otherwise. If Ddata is loaded, it can handle arrays.  """
    if useD0lowT and kbT < SC.kbTc / 2:
        return SC.D0
    else:
        Ddata = SC.Ddata
        if Ddata is not None:
            Dspl = interpolate.splrep(Ddata[0, :], Ddata[1, :], s=0)
            return np.clip(interpolate.splev(kbT, Dspl), 0, None)
        else:
            warnings.warn(
                "D calculation takes long.. \n Superconductor={}\n N0={}\n kbTD={}\n Tc={}"
                .format(SC.name, SC.N0, SC.kbTD,
                        SC.kbTc / (const.Boltzmann / const.e * 1e6)))

            def integrandD(E, D, kbT, SC):
                return SC.N0 * SC.Vsc * (
                    np.tanh(np.sqrt(E**2 + D**2) /
                            (2 * kbT)) / np.sqrt(E**2 + D**2))

            def dint(D, kbT, SC):
                return np.abs(
                    integrate.quad(
                        integrandD, 0, SC.kbTD, args=(D, kbT,
                                                      SC), points=(D,))[0] - 1)

            return root_scalar(dint, args=(kbT, SC), x0=SC.D0,
                               method='secant').root


def nqp(kbT, D_=None, SC=None):
    """Thermal average quasiparticle denisty. It can handle arrays (only if
    both kbT and D_ are arrays) and uses a low temperature approximation, 
    if kbT < Delta/100."""
    if D_ is None:
        D_ = D(kbT, SC)

    if np.array(kbT < D_ / 100).all():
        return 2 * SC.N0 * np.sqrt(2 * np.pi * kbT * D_) * np.exp(-D_ / kbT)
    else:

        def integrand(E, kbT, D_, SC):
            return 4 * SC.N0 * E / np.sqrt(E**2 - D_**2) * f(E, kbT)

        if isinstance(kbT, (int, float, complex)):
            return integrate.quad(integrand, D_, np.inf, args=(kbT, D_, SC))[0]
        else:
            if isinstance(D_, (int, float, complex)):
                D_ = D_ * np.ones(len(kbT))
            assert kbT.size == D_.size, "kbT and D_ arrays are not of the same size"
            result = np.zeros(len(kbT))
            for i in range(len(kbT)):
                result[i] = integrate.quad(integrand,
                                           D_[i],
                                           np.inf,
                                           args=(kbT[i], D_[i], SC))[0]
            return result


def kbTeff(nqp_, SC):
    """Calculates the effective temperature (in µeV) at a certain 
   quasiparticle density."""
    Ddata = SC.Ddata
    if Ddata is not None:
        kbTspl = interpolate.splrep(Ddata[2, :], Ddata[0, :], s=0, k=1)
        return interpolate.splev(nqp_, kbTspl)
    else:

        def minfunc(kbT, nqp_, SC):
            Dt = D(kbT, SC)
            return np.abs(nqp(kbT, Dt, SC) - nqp_)

        res = minisc(
            minfunc,
            bounds=(0, 0.9 * SC.kbTc),
            args=(nqp_, SC),
            method="bounded",
            options={"xatol": 1e-15},
        )
        if res.success:
            return res.x


def Zs(hw, kbT, SCsheet):
    '''The surface impendance of a superconducting sheet with arbitrary 
    thickness. Unit is µOhm'''
    D_ = D(kbT, SCsheet.SC)
    s1, s2 = cinduct(hw, kbT, SCsheet.SC) / SCsheet.SC.rhon
    omega = hw / (const.hbar * 1e12 / const.e)
    return (np.sqrt(1j * const.mu_0 * 1e6 * omega / (s1 - 1j * s2)) / np.tanh(
        np.sqrt(1j * omega * const.mu_0 * 1e6 * (s1 - 1j * s2)) * SCsheet.d))


def beta(kbT, D_, SCsheet):
    """calculates beta, a measure for how thin the film is
    compared to the penetration depth.
    D -- energy gap
    kbT -- temperature in µeV
    SC -- Superconductor object, from the SC class"""
    SC = SCsheet.SC
    lbd = SC.lbd_eff * 1 / np.sqrt(D_ / SC.D0 * np.tanh(D_ / (2 * kbT)))
    return 1 + 2 * SCsheet.d / (lbd * np.sinh(2 * SCsheet.d / lbd))


def Qi(s1, s2, ak, kbT, D_, SCsheet):
    """Calculates the internal quality factor, 
    from the complex conductivity. See PdV PhD thesis eq. (2.23)"""
    b = beta(kbT, D_, SCsheet)
    return 2 * s2 / (ak * b * s1)


def hwres(s2, hw0, s20, ak, kbT, D_, SCsheet):
    """Gives the resonance frequency in µeV, from the sigma2,
    from a linearization from point hw0,sigma20. See PdV PhD eq. (2.24)"""
    b = beta(kbT, D_, SCsheet)
    return hw0 * (1 + ak * b / 4 / s20 * (s2 - s20))  # note that is linearized


def S21(Qi, Qc, hwread, dhw, hwres):
    """Gives the complex transmittance of a capacatively coupled
    superconducting resonator (PdV PhD, eq. (3.21)), with:
    hwread -- the read out frequency
    dhw -- detuning from hwread (so actual read frequency is hwread + dhw)
    hwres -- resonator frequency"""
    Q = Qi * Qc / (Qi + Qc)
    dhw_act = hwread + dhw - hwres
    return (Q / Qi + 2j * Q * dhw_act / hwres) / (1 + 2j * Q * dhw_act / hwres)


def hwread(hw0, kbT0, ak, kbT, SCvol):
    """Calculates at which frequency, one probes at resonance. 
    This must be done iteratively, as the resonance frequency is 
    dependent on the complex conductivity, which in turn depends on the
    read frequency."""
    s20 = cinduct(hw0, kbT0, SCvol.SC)[1]

    def minfuc(hw, hw0, s20, ak, kbT, SCvol):
        s1, s2 = cinduct(hw, kbT, SCvol.SC)
        return np.abs(hwres(s2, hw0, s20, ak, kbT, SCvol) - hw)

    res = minisc(
        minfuc,
        bracket=(0.5 * hw0, hw0, 2 * hw0),
        args=(hw0, s20, ak, kbT, SCvol),
        method="brent",
        options={"xtol": 1e-21},
    )
    if res.success:
        return res.x


def calc_Nwsg(kbT, D_, e, V):
    """Calculates the number of phonons with the Debye approximation, 
    for Al."""

    def integrand(E, kbT, V):
        return (3 * V * E**2 / (2 * np.pi * (const.hbar / const.e * 1e12)**2 *
                                (6.3e3)**3 * (np.exp(E / kbT) - 1)))

    return integrate.quad(integrand, e + D_, 2 * D_, args=(kbT, V))[0]


def kbTbeff(tqpstar, SCsheet, plot=False):
    """Calculates the effective temperature, from a
    quasiparticle lifetime."""
    SC = SCsheet.SC
    nqp_0 = (SC.t0 * SC.N0 * SC.kbTc**3 / (2 * SC.D0**2 * tqpstar) *
             (1 + SCsheet.tesc / SC.tpb) / 2)
    return kbTeff(nqp_0, SC)


def nqpfromtau(tau, SCsheet):
    """Calculates the density of quasiparticles from the quasiparticle lifetime."""
    SC = SCsheet.SC
    return (SC.t0 * SC.N0 * SC.kbTc**3 / (2 * SC.D0**2 * 2 * tau /
                                          (1 + SCsheet.tesc / SC.tpb)))
