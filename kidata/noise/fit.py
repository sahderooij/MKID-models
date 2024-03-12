import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm.notebook import tnrange
import glob
from ipywidgets import interact
import os
import json

from kidata import io
from . import filters

def Lorentzian(f, t, lvl):
    return lvl / (1 + (2 * np.pi * f * t) ** 2)
    
def Lorspec(
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
        warnings.warn("Too few points in window to do fit.")
        tau = np.nan
        tauerr = np.nan
        lvl = np.nan
        lvlerr = np.nan
    else:
        fitPSD = 10 ** (np.real(SPR[fitmask] - SPR[fitmask].max()) / 10)
        # normalize for robust fitting

        try:
            fit = curve_fit(
                Lorentzian,
                fitfreq,
                fitPSD,
                bounds=([0, 0], [np.inf, np.inf]),
                p0=(1/(2*np.pi*stopf), 1),
            )
            tau = fit[0][0] * 1e6
            tauerr = np.sqrt(np.diag(fit[1]))[0] * 1e6
            lvl = fit[0][1] * 10 ** (np.real(SPR[fitmask].max()) / 10)
            lvlerr = np.sqrt(np.diag(fit[1]))[1] * 10 ** (np.real(SPR[fitmask].max()) / 10)
        except RuntimeError:
            tau, tauerr, lvl, lvlerr = np.ones(4) * np.nan
    if plot:
        plt.figure()
        plt.plot(freq, SPR, "o")
        if ~np.isnan(tau):
            plt.plot(fitfreq, 10 * np.log10(Lorentzian(fitfreq, tau * 1e-6, lvl)), "r")
            plt.plot(freq, 10 * np.log10(Lorentzian(freq, tau * 1e-6, lvl)), "r--")
        plt.xscale("log")
        plt.show()
        plt.close()
    return tau, tauerr, lvl, lvlerr


def Lorspecs(fld, plot=False, fltr50Hz=np.zeros(3), fltramp=[1, 1, 0], fltr1fn=[0, 1, 0],  **fitkwargs):
    '''This function fits Lorentzian spectra to the PSDs that are in 
    the folder 'fld' and saves the values in 'fld/fits'. 
    One output csv file contains the values for all temperatures defined by the filenames '_TmK<>.csv'.
    When fltrampphase is True, the amplitude and phase spectra (index 0 and 1) will be filtered 
        with del_1fnNoise() and del_ampNoise() from noise.filters''' 
    
    resultpath = fld + '/fits'
    
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)
    fnames = io.get_avlfiles(fld, ftype='.csv')
    KIDPrExs = np.unique(fnames[:, :3], axis=0)
    for k in tnrange(len(KIDPrExs), leave=False, desc='KID_Pr_Ex'):
        TmKs = [int(i.split('_')[-1].split('.')[0][3:]) 
                    for i in glob.iglob(fld + '/' + '_'.join(KIDPrExs[k]) + '*.csv')]
        fitres = np.zeros((len(TmKs), 4*3 + 1))
        for t in tnrange(len(TmKs), leave=False, desc='Temp'):
            fitres[t, 0] = TmKs[t]
            specdata = np.loadtxt(fld + '/' + '_'.join(KIDPrExs[k]) + f'_TmK{TmKs[t]}.csv',
                                  delimiter=',', ndmin=2)
            for s in range(3):
                freq, Sxy = (specdata[:, 0], specdata[:, s+1])
                if fltr50Hz[s]:
                    freq, Sxy = filters.del_50Hz(freq, Sxy)
                if fltramp[s]:
                    freq, Sxy = filters.del_ampNoise(freq, Sxy)
                if fltr1fn[s]:
                    freq, Sxy = filters.del_1fnNoise(freq, Sxy)
                if plot:
                    print('_'.join(KIDPrExs[k]) + f', T={TmKs[t]} mK, spec{s}')
                fitres[t, (1+4*s):(1+4*(s+1))] = Lorspec(
                    freq, Sxy, plot=plot, **fitkwargs)
                if plot:
                    print('(tau, tau err, level, level err) =', fitres[t, (1+4*s):(1+4*(s+1))], '\n ------')
        np.savetxt(resultpath + '/' + '_'.join(KIDPrExs[k]) + '.csv',
                   fitres[fitres[:, 0].argsort(), :], delimiter=',', 
                  header=('Temperature (mK), amp tau (µs), amp tau err (µs), amp level (1/Hz), amp level err (1/Hz),'
                          + 'phase tau (µs), phase tau err (µs), phase level (rad.^2/Hz), phase level err (rad.^2/Hz),'
                          + 'cross tau (µs), cross tau err (µs), cross level (rad./Hz), cross level err (rad./Hz)')
                  )
    with open(f"{resultpath}/fitoptions.json", "w") as fp:
        json.dump(fitkwargs, fp)
    np.savetxt(resultpath + '/usedfilters.txt',
              np.array([fltr50Hz, fltramp, fltr1fn]).T,
               delimiter=',', header='Filter 50 Hz, amplifier, 1/f^n  (rows: amp, phase, cross)')

def show(fld, plotfltred=True):
    '''Plots the fitted Lorentzians together with PSD for amp, phase and cross.
    Needs interactive matplotlib back-end''' 
    fig, axs = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
    plt.ion()
    
    usedfilters = np.loadtxt(fld + '/fits/usedfilters.txt', delimiter=',')
    def plotfit(file):
        specs = np.loadtxt(fld + '/' + file, delimiter=',')
        fitres = np.loadtxt(fld + '/fits/' + '_'.join(file.split('_')[:3]) + '.csv', 
                           delimiter=',', ndmin=2)
        TmK = int(file.split('_')[-1].split('.')[0][3:])
        
        ymin = 0
        for i, (ax, spec) in enumerate(zip(axs, ['amp', 'phase', 'cross'])):
            ax.cla()
            ax.plot(specs[:, 0], specs[:, i+1])
            if usedfilters[i, 0]:
                flfr, flspec = filters.del_50Hz(specs[:, 0], specs[:, i+1])
            else:
                flfr, flspec = (specs[:, 0], specs[:, i+1])

            if usedfilters[i, 1]:
                flfr, flspec = filters.del_ampNoise(flfr, flspec)

            if usedfilters[i, 2]:
                flfr, flspec = filters.del_1fnNoise(flfr, flspec)
                
            ax.plot(flfr, flspec,
                   label='filtered')
            tau, tauerr, lvl, lvlerr = fitres[fitres[:, 0] == TmK, 
                                              (4*i + 1):(4*(i+1) + 1)][0]
            pltf = np.logspace(np.log10(specs[:, 0].min()), 
                               np.log10(specs[:, 0].max()))
            
            ax.plot(pltf, 10*np.log10(Lorentzian(pltf, tau*1e-6, lvl)), 
                   label=('fit:\n' 
                          + f' $\\tau={tau:.0f} \pm {tauerr:.1f}~\mu s$\n'
                          + f' level={(10*np.log10(lvl)):.0f} $\pm$ {(10*np.log10(2*(lvl-lvlerr)/lvl)):.0f} dB'))
            ax.set_xlabel('Frequency (Hz)')
            ax.legend(loc=(0, 1), title=spec)
            ymin = np.min((ymin, flspec[np.isfinite(flspec)].min()))
        axs[0].set_ylim(ymin, None)
        axs[0].set_xscale('log')
        axs[0].set_ylabel('PSD (dBc/Hz)')
        fig.suptitle(file)
        fig.tight_layout()
    
    fileids = io.get_avlfileids(fld, ftype='.csv')
    files = [(f'KID{fileids[i, 0]:.0f}_{fileids[i, 1]:.0f}dBm_' 
             + ('_' if fileids[i, 2]==0 else f'Tchip{fileids[i, 2]:.2f}_')
             + f'TmK{fileids[i, 3]:.0f}.csv') for i in range(len(fileids))]
    interact(plotfit, file=files)
            
        