import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def Lorentzian(
    freq, SPR, startf=None, stopf=None, decades=3, minf=1e2, plot=False):
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
    noise level (non-dB) and error in noise level."""

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
        lvl = np.nan
        lvlerr = np.nan
    else:
        fitPSD = 10 ** (np.real(SPR[fitmask] - SPR.max()) / 10)
        # normalise for robust fitting

        def Lorspec(f, t, lvl):
            SN = lvl / (1 + (2 * np.pi * f * t) ** 2)
            return SN

        try:
            fit = curve_fit(
                Lorspec,
                fitfreq,
                fitPSD,
                bounds=([0, 0], [np.inf, np.inf]),
                p0=(1/(2*np.pi*stopf), 1),
            )
            tau = fit[0][0] * 1e6
            tauerr = np.sqrt(np.diag(fit[1]))[0] * 1e6
            lvl = fit[0][1] * 10 ** (np.real(SPR.max()) / 10)
            lvlerr = np.sqrt(np.diag(fit[1]))[1] * 10 ** (np.real(SPR.max()) / 10)
        except RuntimeError:
            tau, tauerr, lvl, lvlerr = np.ones(4) * np.nan
    if plot:
        plt.figure()
        plt.plot(freq, np.real(SPR), "o")
        if ~np.isnan(tau):
            plt.plot(fitfreq, 10 * np.log10(Lorspec(fitfreq, tau * 1e-6, lvl)), "r")
            plt.plot(freq, 10 * np.log10(Lorspec(freq, tau * 1e-6, lvl)), "r--")
        plt.xscale("log")
        plt.show()
        plt.close()
    return tau, tauerr, lvl, lvlerr