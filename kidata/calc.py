import numpy as np
import warnings
import matplotlib.pyplot as plt

import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar as minisc
from scipy.special import k0, i0
from scipy import interpolate

from kidcalc import D, beta, cinduct, nqp
import kidcalc
from SC import Al

from kidata import io
from kidata.plot import _selectPread

# NOTE: all units are in 'micro': µeV, µm, µs etc.


def ak(S21data, SC=None, plot=False, reterr=False, method="df"):
    """Calculates the kinetic induction fraction, based on Goa2008, PhD Thesis. 
    Arguments:
    S21data -- the content of the .csv from the S21-analysis. 
    SC -- Superconductor object, from SC module, default: Al
    plot -- boolean to plot the fit over temperature.
    reterr -- boolean to return fitting error.
    method -- either df or Qi, which is fitted linearly over temperature.
    
    Returns:
    ak 
    optionally: the error in ak from fitting."""

    if SC is None:
        SC = Al(Tc=S21data[0, 21], V=S21data[0, 14], d=S21data[0, 25])

    # Extract relevant data
    hw = S21data[:, 5] * const.Plack / const.e * 1e6  # µeV
    kbT = S21data[:, 1] * const.Boltzmann / const.e * 1e6  # µeV
    hw0 = hw[0]

    # define y to fit:
    if method == "df":
        y = (hw - hw0) / hw0
    elif method == "Qi":
        y = 1 / S21data[:, 4] - 1 / S21data[0, 4]

    # Mask the double measured temperatures, and only fit from 250 mK
    mask1 = np.zeros(len(y), dtype="bool")
    mask1[np.unique(np.round(S21data[:, 1], decimals=2), return_index=True)[1]] = True
    mask = np.logical_and(mask1, (kbT >= 0.25 * const.Boltzmann / const.e * 1e6))

    if mask.sum() > 3:
        y = y[mask]
    else:
        warnings.warn("Not enough high temperature S21data, taking the last 10 points")
        y = y[mask1][-10:]

    # define x to fit:
    x = np.zeros(len(y))
    i = 0
    s0 = cinduct(hw0, D(kbT[0], SC), kbT[0])
    for kbTi in kbT[mask]:
        D_0 = D(kbTi, SC)
        s = cinduct(hw[i], D_0, kbTi)
        if method == "df":
            x[i] = (s[1] - s0[1]) / s0[1] * beta(kbTi, D_0, SC) / 4
        elif method == "Qi":
            x[i] = (s[0] - s0[0]) / s0[1] * beta(kbTi, D_0, SC) / 2
        i += 1

    # do the fit:
    fit = curve_fit(lambda t, ak: ak * t, x, y)
    if plot:
        plt.figure()
        plt.plot(x, y, "o")
        plt.plot(x, fit[0] * x)
        plt.legend(["Data", "Fit"])
        if method == "df":
            plt.ylabel(r"$\delta f/f_0$")
            plt.xlabel(r"$\beta \delta \sigma_2/4\sigma_2 $")
        elif method == "Qi":
            plt.ylabel(r"$\delta(1/Q_i)$")
            plt.xlabel(r"$\beta \delta \sigma_1/2\sigma_2 $")

    if reterr:
        return fit[0][0], np.sqrt(fit[1][0])
    else:
        return fit[0][0]


def tau(
    freq, SPR, startf=None, stopf=None, decades=3, minf=1e2, plot=False, retfnl=False
):
    """Fits a Lorentzian to a PSD.
    Arguments:
    freq -- frequency array (in Hz)
    SPR -- power spectral denisty values (in dB)
    startf -- start frequency for the fit, default None: 3 decades lower than stopf
    stopf -- stop frequency for the fit, default the lowest value in the interval 3e2 to 3e4 Hz
    decades -- fitrange length in decades, used to determine startf when startf is None.
    plot -- boolean to show the plot
    retfnl -- boolean to return the noise level as well.
    
    Returns:
    tau -- lifetime in µs
    tauerr -- error in tau
    optionally: noise level (non-dB) and  error in noise level."""

    # Filter nan-values
    freq = freq[~np.isnan(SPR)]
    SPR = SPR[~np.isnan(SPR)]
    freq = freq[SPR != -np.inf]
    SPR = SPR[SPR != -np.inf]
    if stopf is None:
        try:
            bdwth = np.logical_and(
                freq > 10 ** (np.log10(minf) + decades), freq < 2 * freq.max() / 10
            )
            stopf = freq[bdwth][np.real(SPR[bdwth]).argmin()]
        except ValueError:
            stopf = 10 ** (np.log10(minf) + decades)
    if startf is None:
        startf = max(10 ** (np.log10(stopf) - decades), minf)

    # fitting a Lorentzian
    fitmask = np.logical_and(freq >= startf, freq <= stopf)
    fitfreq = freq[fitmask]
    if len(fitfreq) < 10:
        warnings.warn("Too little points in window to do fit.")
        tau = np.nan
        tauerr = np.nan
        N = np.nan
        Nerr = np.nan
    else:
        fitPSD = 10 ** (np.real(SPR[fitmask] - SPR.max()) / 10)
        # notice the normalization for robust fitting

        def Lorspec(f, t, N):
            SN = 4 * N * t / (1 + (2 * np.pi * f * t) ** 2)
            return SN

        try:
            fit = curve_fit(
                Lorspec,
                fitfreq,
                fitPSD,
                bounds=([0, 0], [np.inf, np.inf]),
                p0=(2e-4, 1e4),
            )
            tau = fit[0][0] * 1e6
            tauerr = np.sqrt(np.diag(fit[1]))[0] * 1e6
            N = fit[0][1] * 10 ** (np.real(SPR.max()) / 10)
            Nerr = np.sqrt(np.diag(fit[1]))[1] * 10 ** (np.real(SPR.max()) / 10)
        except RuntimeError:
            tau, tauerr, N, Nerr = np.ones(4) * np.nan

    if plot:
        plt.figure()
        plt.plot(freq[SPR != -140], np.real(SPR), "o")
        if ~np.isnan(tau):
            plt.plot(fitfreq, 10 * np.log10(Lorspec(fitfreq, tau * 1e-6, N)), "r")
            plt.plot(freq, 10 * np.log10(Lorspec(freq, tau * 1e-6, N)), "r--")
        plt.xscale("log")
        plt.show()
        plt.close()

    if retfnl:
        return (
            tau,
            tauerr,
            4 * N * tau * 1e-6,
            np.sqrt((4e-6 * N * tauerr) ** 2 + (4e-6 * Nerr * tau) ** 2),
        )
    else:
        return tau, tauerr


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


def NLcomp(Chipnum, KIDnum, Pread, SC=None, method="", var="cross"):
    """Returns a spline representation of a Noise Level (non-dB) vs Temperature (K), 
    with which the measured noise level can be compensated. For example,
    the method \'Resp\' gives the responsivity squared. If the measured 
    noise level is divided by the responsivity squared, one is left with 
    the quasiparticle fluctuation level.
    Arguments:
    Chipnum, KIDnum, Pread -- define which data is to be used (S21data and/or pulse data)
    SC -- a SuperConductor object (see SC module), which defines superconductor properties
    method -- defines which level is to be returned. See if statements in the function for the options.
            (future: multiply every individual method stated in the method string)
    var -- gives the type of PSD to be compensated - cross, amp or phase - and is used 
            if \'Reps\' is in the method """

    if SC is None and method != "":
        S21data = io.get_S21data(Chipnum, KIDnum, Pread)
        SC = Al(Tc=S21data[0, 21], V=S21data[0, 14], d=S21data[0, 25])

    if method != "":
        S21data = io.get_S21data(Chipnum, KIDnum, Pread)
        if "ak" in method:
            akin = ak(S21data)

        if method == "QakV":
            lvlcompspl = interpolate.splrep(
                S21data[:, 1], (S21data[:, 2] * akin) ** 2 / SC.V ** 2, s=0
            )

        elif method == "QaksqrtV":
            lvlcompspl = interpolate.splrep(
                S21data[:, 1], (S21data[:, 2] * akin) ** 2 / (SC.V), s=0
            )

        elif method == "QaksqrtVtesc":
            lvlcompspl = interpolate.splrep(
                S21data[:, 1],
                (S21data[:, 2] * akin) ** 2 / (SC.V * (1 + SC.tesc / SC.tpb)),
                s=0,
            )

        elif method == "QaksqrtVtescTc":
            lvlcompspl = interpolate.splrep(
                S21data[:, 1],
                (S21data[:, 2] * akin) ** 2
                / (
                    SC.V
                    * (1 + SC.tesc / SC.tpb)
                    * (const.Boltzmann / const.e * 1e6 * S21data[0, 21]) ** 3
                    / (SC.D0 / const.e * 1e6) ** 2
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
                * SC.V,
                s=0,
            )

        elif method == "RespVtescTc":
            kbTc = const.Boltzmann / const.e * 1e6 * S21data[0, 21]
            lvlcompspl = interpolate.splrep(
                S21data[:, 1],
                interpolate.splev(
                    S21data[:, 1], Respspl(Chipnum, KIDnum, Pread, var=var)
                )
                * SC.V
                * (1 + SC.tesc / SC.tpb)
                * (kbTc) ** 3
                / (kidcalc.D(const.Boltzmann / const.e * 1e6 * S21data[:, 1], SC)) ** 2,
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
        Pint = 10 * np.log10(
            10 ** (-1 * Pread / 10) * S21data[0, 2] ** 2 / S21data[0, 3] / np.pi
        )
    else:
        lvlcompspl = interpolate.splrep(np.linspace(0.01, 10, 10), np.ones(10))
    return lvlcompspl


def tau_pulse(
    pulse,
    pulsestart=500,
    sfreq=1e6,
    tfit=(10, 1e3),
    tauguess=500,
    reterr=False,
    plot=False,
):
    """Calculates lifetime from a exponential fit to a pulse.
    Arguments:
    pulse -- pulse data at 1 MHz, with begin of the pulse at 500 (µs)
    pulsestart -- index where the rise time of the pulse is halfway
    sfreq -- sample frequency of the pulse data
    tfit -- tuple to specify the fitting window, default is (10,1e3)
    reterr -- boolean to return error
    plot -- boolean to plot the fit

    Returns:
    tau -- in µs
    optionally the fitting error"""

    t = np.arange(len(pulse)) - pulsestart
    fitmask = np.logical_and(t > tfit[0], t < tfit[1])
    t2 = t[fitmask]
    peak2 = pulse[fitmask]
    try:
        fit = curve_fit(
            lambda x, a, b: b * np.exp(-x / a), t2, peak2,
            p0=(tauguess*1e6/sfreq, peak2[0])
        )
    except RuntimeError:
        fit = [[np.nan, np.nan], np.array([[np.nan, np.nan], [np.nan, np.nan]])]

    if plot:
        plt.figure()
        plt.plot(t, pulse)
        plt.plot(t2, fit[0][1] * np.exp(-t2 / fit[0][0]))
        plt.yscale("log")
        plt.show()
        plt.close()
    if reterr:
        return fit[0][0] * 1e6 / sfreq, np.sqrt(fit[1][0, 0]) * 1e6 / sfreq
    else:
        return fit[0][0] * 1e6 / sfreq


def fit_nonexppulse(
    pulse,
    pulsestart=500,
    sfreq=1e6,
    tsstfit=(200, 1e3),
    tssguess=500,
    tfit=(0, 3e3),
    reterr=False,
    plot=False,
):
    """Fits an equation of the form: 
        x(t) = xi*(1-r)/(exp(t/tss)-r). 
    This is a solution to 
        dx/dt = -R x^2  -x/tss, 
    with r/(1-r)=R*xi*tss (Wang2014). 
    First, tss is estimated from the second part of the pulse decay (tsstfit), 
    after which r is extracted with the full fit window (tfit). Both tfit and
    tsstfit are with respect to the pulsestart.
    Otherwise the fit does not converge.

    Returns:
    tss -- in µs
    R -- in µs^-1
    xi -- in the same units as pulse
    optionally with error"""

    # first extract the steady state (tail) decay time
    tss, tsserr = (
        np.array(tau_pulse(pulse, pulsestart, sfreq, tsstfit, tssguess, reterr=True))
        * 1e-6
        * sfreq
    )

    # set-up for the fit:
    t = np.arange(len(pulse)) - pulsestart
    fitmask = np.logical_and(t > tfit[0], t < tfit[1])
    t2 = t[fitmask]
    peak2 = pulse[fitmask]

    # now do the fit:
    def fitfun(t, r, xi):
        return xi * (1 - r) / (np.exp(t / tss) - r)

    fit = curve_fit(fitfun, t2, peak2, p0=(0.5, peak2[0]), bounds=(0, [1, pulse.max()]))
    R = fit[0][0] / (1 - fit[0][0]) / (fit[0][1] * tss)

    if plot:
        plt.plot(t, pulse, label='data')
        plt.plot(t2, fitfun(t2, fit[0][0], fit[0][1]), 'r', label='1/t-fit')
        plt.plot(t[pulsestart:], fitfun(t[pulsestart:], fit[0][0], fit[0][1]),
                 'r--')
        plt.yscale("log")
        plt.legend()

    if reterr:
        return (
            tss * 1e6 / sfreq,
            tsserr * 1e6 / sfreq,
            R * 1e6 / sfreq,
            np.sqrt(
                (
                    np.array(
                        [
                            1
                            / (1 - fit[0][0])
                            / tss
                            / fit[0][1]
                            * (1 + fit[0][0] / (1 - fit[0][0])),
                            fit[0][0] / (1 - fit[0][0]) / tss ** 2 / fit[0][1],
                            fit[0][0] / (1 - fit[0][0]) / tss / fit[0][1] ** 2,
                        ]
                    )
                    ** 2
                )
                .dot([fit[1][0, 0], tsserr ** 2, fit[1][1, 1]])
                .sum()
            ),
            fit[1][1, 1]
        )
    else:
        return tss * 1e6 / sfreq, R * 1e6 / sfreq, fit[0][1]


def double_exp(pulse, pulsestart=500, sfreq=1e6,
               tfit=None, t1fit=(1, 10), t2fit=(50, 150),
               reterr=False, tauguess=(10, 100), plot=False):
    '''Returns first and second decay times, based on a 4 parameter
    double exponential fit of the form: A1 exp(-t/t1) + A2 exp(-t/t2)'''
    if tfit is None:
        tfit = (0, len(pulse) - pulsestart)

    t1, t1err = tau_pulse(pulse, pulsestart, sfreq, t1fit,
                          tauguess[0]*1e6/sfreq, reterr=True)
    t2, t2err = tau_pulse(pulse, pulsestart, sfreq, t2fit,
                          tauguess[1]*1e6/sfreq, reterr=True)

    def fitfun(t, A1, A2):
        return A1 * np.exp(-t / t1) + A2 * np.exp(-t / t2)

    t = np.arange(len(pulse)) - pulsestart
    fitmask = np.logical_and(t > tfit[0], t < tfit[1])

    fit = curve_fit(
        fitfun, t[fitmask], pulse[fitmask],
        p0=(pulse.max(), pulse.max()*1e-2)
    )
    if plot:
        plt.plot(t, pulse, label='data')
        plt.plot(t[fitmask], fitfun(t[fitmask], *fit[0]), 'r', label='1/t-fit')
        plt.plot(t[pulsestart:], fitfun(t[pulsestart:], *fit[0]),
                 'r--')
        plt.yscale("log")
        plt.legend()

    if reterr:
        pass
    else:
        return t1*1e6/sfreq, t2*1e6/sfreq, fit[0][0], fit[0][1]


def tesc(
    Chipnum,
    KIDnum,
    SC=None,
    usePread="max",
    minTemp=200,
    maxTemp=400,
    taunonkaplan=2e2,
    taures=1e1,
    relerrthrs=0.2,
    pltfit=False,
    pltkaplan=False,
    reterr=False,
    defaulttesc=0,
):
    """Calculates the phonon escape time from the GR noise lifetimes and Kaplan.
    Uses data at Pread (default max), and temperatures between minTemp,maxTemp
    (default (300,400)). Only lifetimes between taunonkaplan and taures, and with
    a relative error threshold of relerrthrs are considered.
    The remaining lifetimes, tesc is calculated and averaged. The error (optional return) 
    is the variance of the remaining lifetimes. If this fails, defaulttesc is returned."""

    TDparam = io.get_grTDparam(Chipnum)
    Pread = _selectPread(usePread, io.get_grPread(TDparam, KIDnum))[0]
    if SC is None:
        S21data = io.get_S21data(Chipnum, KIDnum)
        SC = Al(Tc=S21data[0, 21], V=S21data[0, 14], d=S21data[0, 25])

    Temp = io.get_grTemp(TDparam, KIDnum, Pread)
    Temp = Temp[np.logical_and(Temp < maxTemp, Temp > minTemp)]
    tescar, tescarerr, tqpstar, tqpstarerr = np.zeros((4, len(Temp)))
    for i in range(len(Temp)):
        if pltfit:
            print("{} KID{} -{} dBm T={} mK".format(Chipnum, KIDnum, Pread, Temp[i]))
        freq, SPR = io.get_grdata(TDparam, KIDnum, Pread, Temp[i])
        tqpstar[i], tqpstarerr[i] = tau(freq, SPR, plot=pltfit)

        if tqpstarerr[i] / tqpstar[i] > relerrthrs or (
            tqpstar[i] > taunonkaplan or tqpstar[i] < taures
        ):
            tescar[i] = np.nan
        else:
            tescar[i] = kidcalc.tesc(
                const.Boltzmann / const.e * 1e6 * Temp[i] * 1e-3, tqpstar[i], SC
            )
            tescarerr[i] = np.abs(
                kidcalc.tesc(
                    const.Boltzmann / const.e * 1e6 * Temp[i] * 1e-3, tqpstarerr[i], SC
                )
                + SC.tpb
            )

    if tescar[~np.isnan(tescar)].size > 0:
        tesc1 = np.mean(tescar[~np.isnan(tescar)])
        tescerr = np.sqrt(
            np.std(tescar[~np.isnan(tescar)]) ** 2
            + ((tescarerr[~np.isnan(tescar)] / (~np.isnan(tescar)).sum()) ** 2).sum()
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
    SC.tesc = tesc1
    if pltkaplan:
        plt.figure()
        plt.errorbar(Temp, tqpstar, yerr=tqpstarerr, capsize=5.0, fmt="o")
        mask = ~np.isnan(tescar)
        plt.errorbar(Temp[mask], tqpstar[mask], fmt="o")
        try:
            T = np.linspace(
                Temp[~np.isnan(tqpstar)].min(), Temp[~np.isnan(tqpstar)].max(), 100
            )
        except ValueError:
            T = np.linspace(minTemp, maxTemp, 100)
        taukaplan = kidcalc.tau_kaplan(T * 1e-3, SC)
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


def NqpfromQi(S21data, uselowtempapprox=True, SC=Al()):
    """Calculates the number of quasiparticles from the measured temperature dependence of Qi.
    Returns temperatures in K, along with the calculated quasiparticle numbers. 
    If uselowtempapprox, the complex impedence is calculated directly with a low 
    temperature approximation, else it\'s calculated with the cinduct function in kidcalc 
    (slow)."""
    ak_ = ak(S21data)
    hw = S21data[:, 5] * const.Plack / const.e * 1e6
    kbT = S21data[:, 1] * const.Boltzmann / const.e * 1e6

    if uselowtempapprox:
        beta_ = beta(kbT[0], SC.D0, SC)

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
            Nqp[i] = S21data[0, 14] * nqp(kbTeff, SC.D0, SC)
        return kbT / (const.Boltzmann / const.e * 1e6), Nqp
    else:

        def minfunc(kbT, s2s1, hw, SC):
            D_ = D(kbT, SC)
            s1, s2 = cinduct(hw, D_, kbT)
            return np.abs(s2s1 - s2 / s1)

        Nqp = np.zeros(len(kbT))
        for i in range(len(kbT)):
            D_0 = D(kbT[i], SC)
            beta_ = beta(lbd0, D_0, SC)
            s2s1 = S21data[i, 4] * (ak_ * beta_) / 2
            res = minisc(
                minfunc, args=(s2s1, hw[i], SC), bounds=(0, SC.kbTc), method="bounded"
            )
            kbTeff = res.x
            D_ = D(kbTeff, SC)
            Nqp[i] = S21data[0, 14] * nqp(kbTeff, D_, SC)
        return kbT / (const.Boltzmann / const.e * 1e6), Nqp
