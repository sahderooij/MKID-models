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
import warnings


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


def cinduct(hw, D, kbT):
    """Mattis-Bardeen equations."""

    def integrand11(E, hw, D, kbT):
        nume = 2 * (f(E, kbT) - f(E + hw, kbT)) * np.abs(E ** 2 + D ** 2 + hw * E)
        deno = hw * ((E ** 2 - D ** 2) * ((E + hw) ** 2 - D ** 2)) ** 0.5
        return nume / deno

    def integrand12(E, hw, D, kbT):
        nume = (1 - 2 * f(E + hw, kbT)) * np.abs(E ** 2 + D ** 2 + hw * E)
        deno = hw * ((E ** 2 - D ** 2) * ((E + hw) ** 2 - D ** 2)) ** 0.5
        return nume / deno

    def integrand2(E, hw, D, kbT):
        nume = (1 - 2 * f(E + hw, kbT)) * np.abs(E ** 2 + D ** 2 + hw * E)
        deno = hw * ((D ** 2 - E ** 2) * ((E + hw) ** 2 - D ** 2)) ** 0.5
        return nume / deno

    s1 = integrate.quad(integrand11, D, np.inf, args=(hw, D, kbT))[0]
    if hw > 2 * D:
        s1 += integrate.quad(integrand12, D - hw, -D, args=(hw, D, kbT))[0]
    s2 = integrate.quad(integrand2, np.max([D - hw, -D]), D, args=(hw, D, kbT))[0]
    return s1, s2


def D(kbT, SC):
    """Calculates the thermal average energy gap, Delta. Tries to load Ddata,
    but calculates from scratch otherwise. Then, it cannot handle arrays.  """
    Ddata = SC.Ddata
    if Ddata is not None:
        Dspl = interpolate.splrep(Ddata[0, :], Ddata[1, :], s=0)
        return np.clip(interpolate.splev(kbT, Dspl), 0, None)
    else:
        warnings.warn(
            "D calculation takes long.. \n Superconductor={}\n N0={}\n kbTD={}\n Tc={}".format(
                SC.name, SC.N0, SC.kbTD, SC.kbTc / (const.Boltzmann / const.e * 1e6)
            )
        )

        def integrandD(E, D, kbT, SC):
            return SC.N0 * SC.Vsc * (1 - 2 * f(E, kbT)) / np.sqrt(E ** 2 - D ** 2)

        def dint(D, kbT, SC):
            return np.abs(
                integrate.quad(integrandD, D, SC.kbTD, args=(D, kbT, SC), points=(D,))[
                    0
                ]
                - 1
            )

        res = minisc(dint, args=(kbT, SC), method="bounded", bounds=(0, SC.D0))
        if res.success:
            return np.clip(res.x, 0, None)


def nqp(kbT, D, SC):
    """Thermal average quasiparticle denisty. It can handle arrays 
    and uses a low temperature approximation, if kbT < Delta/20."""
    if (kbT < D / 20).all():
        return 2 * SC.N0 * np.sqrt(2 * np.pi * kbT * D) * np.exp(-D / kbT)
    else:

        def integrand(E, kbT, D, SC):
            return 4 * SC.N0 * E / np.sqrt(E ** 2 - D ** 2) * f(E, kbT)

        if any(
            [
                type(kbT) is float,
                type(D) is float,
                type(kbT) is np.float64,
                type(D) is np.float(64),
            ]
        ):  # make sure it can deal with kbT,D arrays
            return integrate.quad(integrand, D, np.inf, args=(kbT, D, SC))[0]
        else:
            assert kbT.size == D.size, "kbT and D arrays are not of the same size"
            result = np.zeros(len(kbT))
            for i in range(len(kbT)):
                result[i] = integrate.quad(
                    integrand, D[i], np.inf, args=(kbT[i], D[i], SC)
                )[0]
            return result


def kbTeff(nqp, SC):
    """Calculates the effective temperature (in µeV) at a certain 
   quasiparticle density."""
    Ddata = SC.Ddata
    if Ddata is not None:
        kbTspl = interpolate.splrep(Ddata[2, :], Ddata[0, :])
        return interpolate.splev(nqp, kbTspl)
    else:

        def minfunc(kbT, Nqp, SC):
            Dt = D(kbT, SC)
            return np.abs(nqp(kbT, Dt, SC) - nqp)

        res = minisc(
            minfunc,
            bounds=(0, 0.9 * SC.kbTc),
            args=(nqp, SC),
            method="bounded",
            options={"xatol": 1e-15},
        )
        if res.success:
            return res.x


def beta(kbT, D, SCsheet):
    """calculates beta, a measure for how thin the film is
    compared to the penetration depth.
    D -- energy gap
    kbT -- temperature in µeV
    SC -- Superconductor object, from the SC class"""
    SC = SCsheet.SC
    lbd = SC.lbd0 * 1 / np.sqrt(D / SC.D0 * np.tanh(D / (2 * kbT)))
    return 1 + 2 * SCsheet.d / (lbd * np.sinh(2 * SCsheet.d / lbd))


def Qi(s1, s2, ak, kbT, D, SCsheet):
    """Calculates the internal quality factor, 
    from the complex conductivity. See PdV PhD thesis eq. (2.23)"""
    b = beta(kbT, D, SCsheet)
    return 2 * s2 / (ak * b * s1)


def hwres(s2, hw0, s20, ak, kbT, D, SCsheet):
    """Gives the resonance frequency in µeV, from the sigma2,
    from a linearization from point hw0,sigma20. See PdV PhD eq. (2.24)"""
    b = beta(kbT, D, SCsheet)
    return hw0 * (1 + ak * b / 4 / s20 * (s2 - s20))  # note that is linearized


def S21(Qi, Qc, hwread, dhw, hwres):
    """Gives the complex transmittance of a capacatively coupled
    superconducting resonator (PdV PhD, eq. (3.21)), with:
    hwread -- the read out frequency
    dhw -- detuning from hwread (so actual read frequency is hwread + dhw)
    hwres -- resonator frequency"""
    Q = Qi * Qc / (Qi + Qc)
    dhw += hwread - hwres
    return (Q / Qi + 2j * Q * dhw / hwres) / (1 + 2j * Q * dhw / hwres)


def hwread(hw0, kbT0, ak, kbT, D, SC):
    """Calculates at which frequency, one probes at resonance. 
    This must be done iteratively, as the resonance frequency is 
    dependent on the complex conductivity, which in turn depends on the
    read frequency."""
    D_0 = D(kbT0, SC)
    s20 = cinduct(hw0, D_0, kbT0)[1]

    def minfuc(hw, hw0, s20, ak, kbT, D, SC):
        s1, s2 = cinduct(hw, D, kbT)
        return np.abs(hwres(s2, hw0, s20, ak, kbT, D, SC) - hw)

    res = minisc(
        minfuc,
        bracket=(0.5 * hw0, hw0, 2 * hw0),
        args=(hw0, s20, ak, kbT, D, SC),
        method="brent",
        options={"xtol": 1e-21},
    )
    if res.success:
        return res.x


def calc_Nwsg(kbT, D, e, V):
    """Calculates the number of phonons with the Debye approximation, 
    for Al."""

    def integrand(E, kbT, V):
        return (
            3
            * V
            * E ** 2
            / (
                2
                * np.pi
                * (const.hbar / const.e * 1e12) ** 2
                * (6.3e3) ** 3
                * (np.exp(E / kbT) - 1)
            )
        )

    return integrate.quad(integrand, e + D, 2 * D, args=(kbT, V))[0]


def tau_kaplan(T, SCsheet):
    """Calculates the apparent quasiparticle lifetime from Kaplan. 
    See PdV PhD eq. (2.29)"""
    SC = SCsheet.SC
    D_ = D(const.Boltzmann / const.e * 1e6 * T, SC)
    nqp_ = nqp(const.Boltzmann / const.e * 1e6 * T, D_, SC)
    taukaplan = (
        SC.t0 * SC.N0 * SC.kbTc ** 3 / (4 * nqp_ * D_ ** 2) *
        (1 + SCsheet.tesc / SC.tpb)
    )
    return taukaplan


def tauqp_kaplan(kbT, SC):
    """Calculates the intrinsic quasiparticle lifetime w.r.t recombination at E=Delta
    from Kaplan1976 and uses the intral form s.t. it holds for all temperatures."""
    D_ = D(kbT, SC)

    def integrand(E, D, kbT):
        return (
            E ** 2
            * (E - D)
            / np.sqrt((E - D) ** 2 - D ** 2)
            * (1 + D ** 2 / (D * (E - D)))
            * (n(E, kbT) + 1)
            * f(E - D, kbT)
        )

    return (
        SC.t0
        * SC.kbTc ** 3
        * (1 - f(D_, kbT))
        / integrate.quad(integrand, 2 * D_, np.inf, args=(D_, kbT))[0]
    )


def tauscat_kaplan(kbT, SC):
    """Calculates the intrinsic quasiparticle lifetime w.r.t scattering at E=Delta
    from Kaplan1976 and uses the intral form s.t. it holds for all temperatures."""
    D_ = D(kbT, SC)

    def integrand(E, D, kbT):
        return (
            E ** 2
            * (E + D)
            / np.sqrt((E + D) ** 2 - D ** 2)
            * (1 - D ** 2 / (D * (E + D)))
            * n(E, kbT)
            * (1 - f(E + D, kbT))
        )

    return (
        SC.t0
        * SC.kbTc ** 3
        * (1 - f(D_, kbT))
        / integrate.quad(integrand, 0, np.inf, args=(D_, kbT))[0]
    )


def kbTbeff(tqpstar, SCsheet, plot=False):
    """Calculates the effective temperature, from a
    quasiparticle lifetime."""
    SC = SCsheet.SC
    nqp_0 = (
        SC.t0
        * SC.N0
        * SC.kbTc ** 3
        / (2 * SC.D0 ** 2 * tqpstar)
        * (1 + SCsheet.tesc / SC.tpb)
        / 2
    )
    return kbTeff(nqp_0, SC)


def tesc(kbT, tqpstar, SC):
    """Calculates the phonon escape time, based on tqp* via Kaplan. Times are in µs."""
    D_ = D(kbT, SC)
    nqp_ = nqp(kbT, D_, SC)
    return SC.tpb * (
        (4 * tqpstar * nqp_ * D_ ** 2) / (SC.t0 * SC.N0 * SC.kbTc ** 3) - 1
    )


def nqpfromtau(tau, SCsheet):
    """Calculates the density of quasiparticles from the quasiparticle lifetime."""
    SC = SCsheet.SC
    return (
        SC.t0
        * SC.N0
        * SC.kbTc ** 3
        / (2 * SC.D0 ** 2 * 2 * tau / (1 + SCsheet.tesc / SC.tpb))
    )
