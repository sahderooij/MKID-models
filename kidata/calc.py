import numpy as np
import warnings
import matplotlib.pyplot as plt

import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar as minisc
from scipy.special import k0, i0
from scipy import interpolate
from scipy.signal import deconvolve

import SCtheory as SCth
import SC as SuperCond

from kidata import io, noise
from kidata.plot import selectPread


# NOTE: all units are in 'micro': µeV, µm, µs etc.
def ak(S21data, SC=None, plot=False, reterr=False, method="df_nonlin", Tmin=None):
    """Calculates the kinetic induction fraction, based on Goa2008, PhD Thesis. 
    Note: this only works for small ak!
    Arguments:
    S21data -- the content of the .csv from the S21-analysis. 
    SC -- Superconductor object, from SC module, default: Al
    plot -- boolean to plot the fit over temperature.
    reterr -- boolean to return fitting error.
    method -- either df or Qi, which is fitted linearly over temperature.
    Tmin -- the temperature from which to start the fit. Default is 1/5 of Tc
    
    Returns:
    ak 
    optionally: the error in ak from fitting."""

    if SC is None:
        SC = SuperCond.Al
        SC.kbTc = 86.17 * S21data[0, 21]
    if Tmin is None:
        Tmin = SC.kbTc / const.Boltzmann * const.e * 1e-6 / 5
        #This only works for high temperature points, as sigma needs to change signaficantly

    # Extract relevant data
    hw = S21data[:, 5] * const.Planck / const.e * 1e6  # µeV
    kbT = S21data[:, 1] * const.Boltzmann / const.e * 1e6  # µeV
    Qi = S21data[:, 4]
    
    hw0 = hw[np.argmin(S21data[:, 1] - Tmin)]
    kbT0 = kbT[np.argmin(S21data[:, 1] - Tmin)]
    Qi0 = Qi[np.argmin(S21data[:, 1] - Tmin)]
    
    # define y to fit:
    if method == "df":
        y = (hw - hw0) / hw0
    elif method == "Qi":
        y = 1 / Qi - 1 / Qi0
    elif method == 'df_nonlin':
        y = (hw0/hw)**2 - 1
        
    # define x to fit:
    x = np.zeros(len(y))
    s0 = SCth.cinduct(hw0, SCth.D(kbT0, SC), kbT0)
    Lk0 = s0[1]/((s0[0]**2+s0[1]**2)*hw0) # This is low T, thin film (d << pen.depth) limit
    for i, kbTi in enumerate(kbT):
        D_0 = SCth.D(kbTi, SC)
        s = SCth.cinduct(hw[i], D_0, kbTi)
        beta = SCth.beta(kbTi, D_0, 
                         SuperCond.Sheet(
                             SC, d=S21data[0, 25]))
        Lk = s[1]/((s[0]**2+s[1]**2)*hw[i])
        if method == "df":
            x[i] = (s[1] - s0[1]) / s0[1] * beta / 4
        elif method == "Qi":
            x[i] = (s[0] - s0[0]) / s0[1] * beta / 2
        elif method == 'df_nonlin':
            x[i] = Lk/Lk0 - 1
            
    # Mask the double measured temperatures, and only fit from Tmin
    mask1 = np.zeros(len(kbT), dtype="bool")
    mask1[np.unique(np.round(S21data[:, 1], decimals=3), return_index=True)[1]] = True
    mask = np.logical_and(mask1, (kbT >= Tmin * const.Boltzmann / const.e * 1e6))

    if mask.sum() <= 5:
        warnings.warn("Not enough high temperature S21data, taking the last 5 points")
        mask = -1*np.linspace(5, 1, 5, dtype=int)
        
    # do the fit with mask:
    fit = curve_fit(lambda t, ak, b: ak * t + b, x[mask], y[mask])
    
    # check if result is correct
    if (fit[0][0] >= 1) or (fit[0][0] < 0):
        warnings.warn(f'\nak is {fit[0][0]}, which is wrong. '+
                      'Probably the temperature is too low, or ak is too close to 1')

    if (fit[0][0] >= .22) and (method == 'df'):
        warnings.warn(
            f'\nThe df method gives an error more than 10%:'
            + f'\n error = {(1/(1-fit[0][0]/2)-1)*100:.2f}%')
    
    if plot:
        plt.figure()
        for xpl, ypl in zip([x, x[mask]], [y, y[mask]]):
            plt.plot(xpl, ypl, "o")
            plt.plot(xpl, fit[0][0] * xpl + fit[0][1])
        plt.legend(["Data", f"Fit, $\\alpha_k={fit[0][0]:.3f}$"])
        if method == "df":
            plt.ylabel(r"$\delta f/f_0$")
            plt.xlabel(r"$\beta \delta \sigma_2/4\sigma_2 $")
        elif method == "Qi":
            plt.ylabel(r"$\delta(1/Q_i)$")
            plt.xlabel(r"$\beta \delta \sigma_1/2\sigma_2 $")
        elif method == 'df_nonlin':
            plt.ylabel(r'$\left(\frac{\omega_0}{\omega}\right)^2 - 1$')
            plt.xlabel(r'$\left(\frac{L_k}{L_{k0}}\right) - 1$')
    if reterr:
        return fit[0][0], np.sqrt(fit[1][0])
    else:
        return fit[0][0]


def Respspl(Chipnum, KIDnum, Pread, phasemethod="f0", ampmethod="Qi", var="cross"):
    """Returns a spline representation of the responsivity (amp, phase or cross) vs Temperature (K).
    Arguments:
    Chipnum,KIDnum,Pread -- define the data to be taken (S21data, pulse and/or noise).
    phasemethod -- defines the method that is used for the responsivity calculation. Options:
        f0 (default) - uses the f0 vs T from the S21-measurement. 
        crossnoise - uses noise PSDs: dTheta/dNqp = crossNL/sqrt(ampNL). (future)
    ampmethod -- defines which method is used for the amplitude responsivity. Options:
        Qi (default) - uses the 1/Qi fit in the S21-analysis, direct from S21 csv
        pulse - computes dA/dTheta from pulses (minimal temp, wavelength) and 
                multiplies phase responsivity with it.
        noise - computes dA/dTheta by dividing the amp and phase PSDs 
                from the noise measurement and multiplies the phase
                responsivity with it (future).
        crossnoise - computes dA/dNqp by dividing the crossPSD with sqrt(phasePSD)
                    (future)"""

    S21data = io.get_S21data(Chipnum, KIDnum, Pread)
    Temp = S21data[:, 1]
    if phasemethod == "f0":
        phaseResp = S21data[:, 10]

    if ampmethod == "Qi":
        ampResp = S21data[:, 18]
    elif ampmethod == "pulse":
        pulsePreadar = io.get_pulsePread(Chipnum, KIDnum)
        pulsePreadselect = pulsePreadar[np.abs(pulsePreadar - Pread).argmin()]
        pulseTemp = io.get_pulseTemp(Chipnum, KIDnum, pulsePreadselect).min()
        pulsewvl = io.get_pulsewvl(Chipnum, KIDnum, pulsePreadselect, pulseTemp).min()
        phasepulse, amppulse = io.get_pulsedata(
            Chipnum, KIDnum, pulsePreadselect, pulseTemp, pulsewvl
        )

        phtau = tau_pulse(phasepulse)
        amptau = tau_pulse(amppulse)
        assert (
            np.abs(1 - phtau / amptau) < 0.1
        ), "Amp and Phase lifetimes differ by more than 10%"
        dAdTheta = -1 * (amppulse / phasepulse)[600 : int(600 + 2 * phtau)].mean()
        ampResp = phaseResp * dAdTheta

    if var == "cross":
        Respspl = interpolate.splrep(Temp, np.sqrt(ampResp * phaseResp), s=0)
    elif var == "amp":
        Respspl = interpolate.splrep(Temp, ampResp, s=0)
    elif var == "phase":
        Respspl = interpolate.splrep(Temp, phaseResp, s=0)
    else:
        raise ValueError(f"'{var}' is not a valid variable")
    return Respspl


def NLcomp(Chipnum, KIDnum, Pread, SCvol=None, method="", var="cross"):
    """Returns a spline representation of a Noise Level (non-dB) vs Temperature (K), 
    with which the measured noise level can be compensated. For example,
    the method \'Resp\' gives the responsivity squared. If the measured 
    noise level is divided by the responsivity squared, one is left with 
    the quasiparticle fluctuation level.
    Arguments:
    Chipnum, KIDnum, Pread -- define which data is to be used (S21data and/or pulse data)
    SCvol -- a Volume object (see SC module), which defines superconductor properties
    method -- defines which level is to be returned. See if statements in the function for the options.
            (future: multiply every individual method stated in the method string)
    var -- gives the type of PSD to be compensated - cross, amp or phase - and is used 
            if \'Reps\' is in the method """

    if SCvol is None and method != "":
        SCvol = SuperCond.init_SCvol(Chipnum, KIDnum, set_tesc=False)

    if method != "":
        S21data = io.get_S21data(Chipnum, KIDnum, Pread)
        if "ak" in method:
            akin = ak(S21data)

        if method == "QakV":
            lvlcompspl = interpolate.splrep(
                S21data[:, 1], (S21data[:, 2] * akin) ** 2 / SCvol.V ** 2, s=0
            )

        elif method == "QaksqrtV":
            lvlcompspl = interpolate.splrep(
                S21data[:, 1], (S21data[:, 2] * akin) ** 2 / (SCvol.V), s=0
            )

        elif method == "QaksqrtVtesc":
            lvlcompspl = interpolate.splrep(
                S21data[:, 1],
                (S21data[:, 2] * akin) ** 2 /
                (SCvol.V * (1 + SCvol.tesc / SCvol.SC.tpb)),
                s=0,
            )

        elif method == "QaksqrtVtescTc":
            lvlcompspl = interpolate.splrep(
                S21data[:, 1],
                (S21data[:, 2] * akin) ** 2
                / (
                    SCvol.V
                    * (1 + SCvol.tesc / SCvol.SC.tpb)
                    * (const.Boltzmann / const.e * 1e6 * S21data[0, 21]) ** 3
                    / (SCvol.SC.D0 / const.e * 1e6) ** 2
                ),
                s=0,
            )

        elif method == "Resp":
            lvlcompspl = interpolate.splrep(
                S21data[:, 1],
                interpolate.splev(
                    S21data[:, 1], Respspl(Chipnum, KIDnum, Pread, var=var)
                )
                ** 2,
            )

        elif method == "RespPulse":
            lvlcompspl = interpolate.splrep(
                S21data[:, 1],
                interpolate.splev(
                    S21data[:, 1],
                    Respspl(Chipnum, KIDnum, Pread, ampmethod="pulse", var=var),
                )
                ** 2,
            )

        elif method == "RespPint":
            Pint = 10 ** (-Pread / 10) * S21data[:, 2] ** 2 / (S21data[:, 3] * np.pi)
            Pint /= Pint[0]
            lvlcompspl = interpolate.splrep(
                S21data[:, 1],
                interpolate.splev(
                    S21data[:, 1], Respspl(Chipnum, KIDnum, Pread, var=var)
                )
                / Pint ** (1 / 2),
                s=0,
            )

        elif method == "RespV":
            lvlcompspl = interpolate.splrep(
                S21data[:, 1],
                interpolate.splev(
                    S21data[:, 1], Respspl(Chipnum, KIDnum, Pread, var=var)
                )
                * SCvol.V,
                s=0,
            )

        elif method == "RespVtescTc":
            kbTc = const.Boltzmann / const.e * 1e6 * S21data[0, 21]
            lvlcompspl = interpolate.splrep(
                S21data[:, 1],
                interpolate.splev(
                    S21data[:, 1], Respspl(Chipnum, KIDnum, Pread, var=var)
                )
                * SCvol.V
                * (1 + SCvol.tesc / SCvol.tpb)
                * (kbTc) ** 3
                / (SCth.D(const.Boltzmann / const.e * 1e6 * S21data[:, 1],
                             SCvol.SC)) ** 2,
                s=0,
            )

        elif method == "RespLowT":
            lvlcompspl = interpolate.splrep(
                S21data[:, 1],
                np.ones(len(S21data[:, 1]))
                * interpolate.splev(
                    S21data[0, 1], Respspl(Chipnum, KIDnum, Pread, var=var)
                ),
            )

        elif method == "Resptres":
            lvlcompspl = interpolate.splrep(
                S21data[:, 1],
                interpolate.splev(
                    S21data[:, 1], Respspl(Chipnum, KIDnum, Pread, var=var)
                )
                ** 2
                * (1 + (S21data[:, 1] * 2 * S21data[:, 2] / S21data[:, 5]) ** 2),
            )
        else:
            raise ValueError("{} is an invalid compensation method".format(method))
    else:
        lvlcompspl = interpolate.splrep(np.linspace(0.01, 10, 10), np.ones(10))
    return lvlcompspl

def Nqp(chip, KID, Pread, spec='cross'):
    '''Calculates the Nqp vs T (in mK) from GR noise and S21 data'''
    specdict = {'amp': 0, 'phase': 1, 'cross': 2}
    fits = io.get_noisefits(chip, KID, Pread)
    tau, tauerr, lvl, lvlerr = fits[:, (specdict[spec] * 4 + 1):(specdict[spec] * 4 + 5)].T
    rspnsv = Respspl(chip, KID, Pread, var=spec)
    Nqp = lvl / (4 * tau * 1e-6) / interpolate.splev(fits[:, 0] * 1e-3, rspnsv)**2
    Nqperr = np.sqrt((Nqp / lvl * lvlerr)**2 + (Nqp / tau * tauerr)**2)
    return fits[:, 0], Nqp, Nqperr
                

def tesc(
    Chipnum,
    KIDnum,
    SCvol=None,
    usePread="max",
    minTemp=200,
    maxTemp=400,
    taunonkaplan=2e2,
    taures=1e1,
    relerrthrs=0.2,
    pltkaplan=False,
    reterr=False,
    defaulttesc=0,
):
    """Calculates the phonon escape time from the GR noise lifetimes and Kaplan.
    Uses data at Pread (default max), and temperatures between minTemp,maxTemp
    (default (200,400)). Only lifetimes between taunonkaplan and taures, and with
    a relative error threshold of relerrthrs are considered.
    From the remaining lifetimes, tesc is calculated and averaged. The error (optional return) 
    is the variance of the tesc values. If this fails, defaulttesc is returned."""

    KIDPrT = io.get_noiseKIDPrT(Chipnum)
    Preads = KIDPrT[KIDPrT[:, 0] == KIDnum, 1]
    Pread = selectPread(usePread, Preads)[0]
    if SCvol is None:
        S21data = io.get_S21data(Chipnum, KIDnum)
        SCvol = SuperCond.init_SC(Chipnum, KIDnum, set_tesc=False)

    fits = io.get_noisefits(Chipnum, KIDnum, Pread)
    mask = (fits[:, 0] < maxTemp) & (fits[:, 0] > minTemp)
    tescar = SCth.tau.esc(
        const.Boltzmann / const.e * 1e6 * fits[:, 0] * 1e-3, fits[:, 9],
        SCvol.SC)
    tescarerr = np.abs(
        SCth.tau.esc(
            const.Boltzmann / const.e * 1e6 * fits[:, 0] * 1e-3,
            fits[:, 10], SCvol.SC) 
        + SCvol.SC.tpb)
    
    mask = ((fits[:, 0] < maxTemp) & (fits[:, 0] > minTemp) 
            & (fits[:, 10]/fits[:, 9] < relerrthrs)
            & (fits[:, 9] < taunonkaplan)
            & (fits[:, 9] > taures))

    if mask.sum() > 0:
        tesc1 = np.mean(tescar[mask])
        tescerr = np.sqrt(
            np.std(tescar[mask]) ** 2
            + ((tescarerr[mask] / (mask.sum())) ** 2).sum()
        )
    else:
        tesc1 = np.nan

    if tesc1 < 0 or np.isnan(tesc1) or tesc1 > 1e-2:
        warnings.warn(
            "tesc ({}) is not valid and set to {} µs. {}, KID{}".format(
                tesc1, defaulttesc, Chipnum, KIDnum
            )
        )
        tesc1 = defaulttesc
        tescerr = 0
        
    SCvol.tesc = tesc1
    if pltkaplan:
        plt.figure()
        plt.errorbar(fits[:, 0], fits[:, 9], yerr=fits[:, 10], capsize=5.0, fmt="o")
        plt.errorbar(fits[mask, 0], fits[mask, 9], fmt="o")
        T = np.linspace(
            fits[mask, 0].min(),
            fits[mask, 0].max(), 100
        )

        taukaplan = SCth.tau.qpstar(T * 1e-3, SCvol)
        plt.plot(T, taukaplan)
        plt.yscale("log")
        plt.ylim(None, 1e4)
        plt.xlabel("T (mK)")
        plt.ylabel(r"$\tau_{qp}^*$ (µs)")
        plt.legend(["Kaplan", "GR Noise Data", "Selected Data"])
        plt.show()
        plt.close()
    if reterr:
        return tesc1, tescerr
    else:
        return tesc1


def get_tescdict(Chipnum, Pread="max"):
    """Returns a dictionary with the escape times of all KIDs in a chip."""
    tescdict = {}
    TDparam = io.get_grTDparam(Chipnum)
    KIDlist = io.get_grKIDs(TDparam)
    for KIDnum in KIDlist:
        tescdict[KIDnum] = tesc(Chipnum, KIDnum, Pread=Pread)
    return tescdict


def NqpfromQi(S21data, uselowtempapprox=True, SC=SuperCond.Al):
    """Calculates the number of quasiparticles from the measured temperature dependence of Qi.
    Returns temperatures in K, along with the calculated quasiparticle numbers. 
    If uselowtempapprox, the complex impedence is calculated directly with a low 
    temperature approximation, else it\'s calculated with the cinduct function in SCth 
    (slow)."""
    ak_ = ak(S21data, SC)
    hw = S21data[:, 5] * const.Plack / const.e * 1e6
    kbT = S21data[:, 1] * const.Boltzmann / const.e * 1e6

    if uselowtempapprox:
        beta_ = SCth.beta(kbT[0], SC.D0, SuperCond.Sheet(
            SC,
            d=S21data[0, 25]
        ))

        def minfunc(kbT, s2s1, hw, D0):
            xi = hw / (2 * kbT)
            return np.abs(
                np.pi
                / 4
                * (
                    (np.exp(D0 / kbT) - 2 * np.exp(-xi) * i0(xi))
                    / (np.sinh(xi) * k0(xi))
                )
                - s2s1
            )

        Nqp = np.zeros(len(kbT))
        for i in range(len(kbT)):
            s2s1 = S21data[i, 4] * (ak_ * beta_) / 2
            res = minisc(
                minfunc,
                args=(s2s1, hw[i], SC.D0),
                bounds=(0, SC.kbTc),
                method="bounded",
            )
            kbTeff = res.x
            Nqp[i] = S21data[0, 14] * SCth.nqp(kbTeff, SC.D0, SC)
        return kbT / (const.Boltzmann / const.e * 1e6), Nqp
    else:

        def minfunc(kbT, s2s1, hw, SC):
            D_ = SCth.D(kbT, SC)
            s1, s2 = SCth.cinduct(hw, D_, kbT)
            return np.abs(s2s1 - s2 / s1)

        Nqp = np.zeros(len(kbT))
        for i in range(len(kbT)):
            D_0 = SCth.D(kbT[i], SC)
            beta_ = SCth.beta(kbT[i], D_0, SuperCond.Sheet(
                SC,
                d=S21data[0, 25]
            ))
            s2s1 = S21data[i, 4] * (ak_ * beta_) / 2
            res = minisc(
                minfunc, args=(s2s1, hw[i], SC), bounds=(0, SC.kbTc), method="bounded"
            )
            kbTeff = res.x
            D_ = SCth.D(kbTeff, SC)
            Nqp[i] = S21data[0, 14] * SCth.nqp(kbTeff, D_, SC)
        return kbT / (const.Boltzmann / const.e * 1e6), Nqp