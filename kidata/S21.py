import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def getKIDfreqs(freq, S21, prom=2, plot=False):
    S21dB = 20 * np.log10(np.abs(S21))
    loc, props = find_peaks(np.diff(S21dB), prominence=prom)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        axs[0].plot(freq, S21dB, ".")
        axs[0].set_ylabel("$|S_{21}|^2$ (dB)")
        axs[1].plot(freq[1:], np.diff(S21dB), "-")
        axs[1].plot(freq[loc], np.ones(len(loc)) * prom, ".")
        axs[1].set_ylabel("Differences")
        for ax in axs:
            ax.set_xlabel("Frequency")
        fig.tight_layout()

    return freq[loc]


def getKIDdip(freq, S21, approxf0=None, wnd=None, plot=False):
    if wnd is None:
        wnd = np.diff(freq).min() * 1e2
    if approxf0 is None:
        approxf0 = freq[np.abs(S21).argmin()]

    initwnd = (freq > approxf0 - wnd / 2) & (freq < approxf0 + wnd / 2)
    f0guess = freq[initwnd][np.abs(S21)[initwnd].argmin()]
    wndmask = (freq > f0guess - wnd / 2) & (freq < f0guess + wnd / 2)
    f, data = freq[wndmask], S21[wndmask]

    if plot:
        plt.figure()
        plt.plot(freq, 20 * np.log10(np.abs(S21)), ".")
        plt.plot(f, 20 * np.log10(np.abs(data)), ".")
    return f, data


def S21res(f, fr, Ql, Qc_real, Qc_imag=0, a=1, alpha=0, tau=0):
    Qc = Qc_real + 1j * Qc_imag
    Qc = Qc_real * np.exp(1j*Qc_imag)
    mismatch = a * np.exp(1j * alpha)
    delay = np.exp(-1j * 2 * np.pi * f * tau)
    resonator = 1 - (Ql / np.abs(Qc) * np.exp(-1j * np.angle(Qc))) / (
        1 + 2j * Ql * (f / fr - 1)
    )
    return mismatch * delay * resonator

def S21fit(freq, cmplS21, approxf0=None, wnd=None, fitmismatch=True, plot=False):
    if (approxf0 is not None) or (wnd is not None):
        f, data = getKIDdip(freq, cmplS21, approxf0, wnd)
    else:
        f, data = freq, cmplS21

    f0guess = f[np.abs(data).argmin()]
    aguess = (np.abs(data[0]) + np.abs(data[-1])) / 2
    #guess Ql as f0/df(FWHM)
    HM = (aguess - np.abs(data).min()) / 2
    Fl = f[f<f0guess][np.abs(np.abs(data[f<f0guess]) - HM).argmin()]
    Fr = f[f>f0guess][np.abs(np.abs(data[f>f0guess]) - HM).argmin()]
    FWHM = Fr - Fl
    Qlguess = f0guess/FWHM
    Qc_realguess = Qlguess / (1 - np.abs(data[np.abs(data).argmin()]))
    phase = np.unwrap(np.angle(data))
    alphaguess = phase[[0, -1]].mean()
    tauguess = (phase[-1] - phase[0]) / (2 * np.pi * (f[0] - f[-1]))

    def absfit(f, fr, Ql, Qc_real, Qc_imag, a):
        return np.abs(S21res(f, fr, Ql, Qc_real, Qc_imag, a, alphaguess, tauguess))

    fitres = curve_fit(
        absfit,
        f,
        np.abs(data),
        p0=(f0guess, Qlguess, Qc_realguess, 0, aguess),
        bounds=([0, 0, Qlguess*.9, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, 1]),
    )
    
    f0, Ql, Qc_real, Qc_imag, a = fitres[0]
    Qc = Qc_real + 1j*Qc_imag
    Qc = Qc_real*np.exp(1j*Qc_imag)
    
    phfit = curve_fit(
        lambda f, alpha, tau: 
        np.unwrap(np.angle(S21res(
            f, f0, Ql, Qc_real, Qc_imag, a, alpha, tau))), 
        f, phase, p0=(alphaguess, tauguess))
    
    alpha, tau = phfit[0]
    params = (f0, Ql, Qc_real, Qc_imag, a, alpha, tau)
    
    Qi = (1/Ql - np.real(1/Qc))**(-1)

    if plot:
        numplots = 3 if (approxf0 is None and wnd is None) else 4
        fig, axs = plt.subplots(1, numplots, figsize=(3*numplots, 3))
        fitf = np.linspace(f.min(), f.max(), 1000)
        i=0
        if approxf0 is not None or wnd is not None:
            axs[i].plot(freq, 20 * np.log10(np.abs(cmplS21)), ".")
            axs[i].plot(
                fitf, 20 * np.log10(np.abs(S21res(fitf, *params))), "-"
            )
            i += 1
        axs[i].plot(f, 20 * np.log10(np.abs(data)), ".")
        axs[i].plot(
            fitf, 20 * np.log10(np.abs(S21res(fitf, *params))), "-"
        )
        axs[i].plot(
            fitres[0][0],
            20 * np.log10(np.abs(S21res(fitres[0][0], *params))),
            "o",
            label=f'$f_0$={fitres[0][0]:.2e}'
        )
        axs[i].legend() 
        i += 1
        axs[i].plot(f, phase, ".")
        axs[i].plot(fitf, np.unwrap(np.angle(S21res(fitf, *params))), "-")
        axs[i].plot(
            fitres[0][0], np.angle(S21res(fitres[0][0], *params)), "o"
        )
        
        i += 1
        axs[i].plot(np.real(data), np.imag(data), ".")
        axs[i].plot(
            np.real(S21res(fitf, *params)),
            np.imag(S21res(fitf, *params)),
            "-",
        )
        axs[i].plot(
            np.real(S21res(fitres[0][0], *params)),
            np.imag(S21res(fitres[0][0], *params)),
            "o",
        )
        axs[i].axhline(0, linestyle="--", color="k")
        axs[i].axvline(0, linestyle="--", color="k")
        fig.tight_layout()

    return f0, Ql, np.abs(Qc), Qi
