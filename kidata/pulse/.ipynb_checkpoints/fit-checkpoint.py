import numpy as np
import warnings
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import deconvolve

from kidata import io

def deconv_ringtime(pulse, chip, kid, temp, pread=None,
                   numtres=10, fs=1e6, plot=False):
    S21data = io.get_S21data(chip, kid, pread)
    Tind = (S21data[:, 1] - temp).argmin()
    Q = S21data[Tind, 2]
    fres = S21data[Tind, 5]
    tres = Q / (fres * np.pi) * fs
    t = np.arange(numtres*tres)
    impulseresp = np.exp(-t/tres) 
    impulseresp /= impulseresp.sum()
    
    deconv, remain = deconvolve(pulse, impulseresp)
    if plot:
        fig, axs = plt.subplots(2, 1, sharex=True)
        for ax in axs:
            ax.plot(pulse, label='data')
            ax.plot(deconv, label=f'deconv., $\\tau_{{res}}$={tres:.0e} µs')
            ax.plot(remain, label='remain')
        axs[0].legend()
        axs[0].set_yscale('log')
    return deconv
    
def exp(
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


def nonexp(
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
    after which r is extracted with the full fit window (tfit).
    Both tfit and tsstfit are with respect to the pulsestart.

    Returns:
    tss -- in µs
    R -- in µs^-1
    xi -- in the same units as pulse
    optionally with error"""

    # first extract the steady state (tail) decay time
    tss, tsserr = (
        np.array(exp(pulse, pulsestart, sfreq, tsstfit, tssguess, reterr=True))
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

    fit = curve_fit(fitfun, t2, peak2, p0=(0.5, peak2[0]),
                    bounds=([0, 0] , [1, pulse.max()]))
    R = fit[0][0] / (1 - fit[0][0]) / (fit[0][1] * tss)

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(5, 7), sharex=True)
        for ax in axs:
            ax.plot(t, pulse, label='data')
            ax.plot(t2, fitfun(t2, *fit[0]), 'r', label='1/t + exp. fit')
            ax.plot(t[pulsestart:], fitfun(t[pulsestart:], *fit[0]),
                 'r--')
            tssmask = (t > tsstfit[0]) & (t < tsstfit[1])
            ax.plot(t[tssmask], fitfun(t[tssmask], *fit[0]), 'r',
                    linewidth=4., label=f'$\\tau_{{ss}}$ fit-range\n $\\tau_{{ss}}$={tss:.0f} µs')
        axs[0].set_yscale("log")
        axs[0].legend()

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

    t1, t1err = exp(pulse, pulsestart, sfreq, t1fit,
                          tauguess[0]*1e6/sfreq, reterr=True)
    t2, t2err = exp(pulse, pulsestart, sfreq, t2fit,
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
        plt.plot(t[fitmask], fitfun(t[fitmask], *fit[0]), 'r', label='double exp. fit')
        plt.plot(t[pulsestart:], fitfun(t[pulsestart:], *fit[0]),
                 'r--')
        plt.yscale("log")
        plt.legend()

    if reterr:
        pass
    else:
        return t1*1e6/sfreq, t2*1e6/sfreq, fit[0][0], fit[0][1]
