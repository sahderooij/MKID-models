import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
import glob
from tqdm.notebook import tnrange
from ipywidgets import interact


from scipy.optimize import curve_fit
from scipy.signal import deconvolve

from kidata import io

def nonexp_func(t, tss, A, B):
    return A / ((1 + B)*np.exp(t / tss) - 1)

def deconv_ringtime(pulse, tres, numtres=10, plot=False):
    t = np.arange(numtres*tres)
    impulseresp = np.exp(-t/tres) 
    impulseresp /= impulseresp.sum()
    
    deconv, remain = deconvolve(pulse, impulseresp)
    if plot:
        fig, axs = plt.subplots(2, 1, sharex=True)
        for ax in axs:
            ax.plot(pulse, '.', label='data')
            ax.plot(deconv, '.', label=f'deconv., $\\tau_{{res}}$={tres:.0e} µs')
            ax.plot(remain, '.', label='remain')
        axs[0].legend()
        axs[0].set_yscale('log')
    return deconv
    
def exp(
    pulse,
    pulsestart=100,
    sfreq=1e6,
    tfit=(10, 1e3),
    tauguess=100,
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
    except:
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
    pulsestart=100,
    sfreq=1e6,
    tsstfit=None,
    tssguess=100,
    tfit=None,
    reterr=False,
    plot=False,
):
    """Fits an equation of the form: 
        x(t) = A / ((1 + B)exp(t/tss) - 1),
    where A/B is the hight at t=0 and 
    B describes the deviation from exponetial (B = 2Nqp0/dNqp(0))
    ( This is the same as,
        x(t) = xi*(1-r)/(exp(t/tss)-r), 
    which is a solution to 
        dx/dt = -R x^2  -x/tss, 
    with r/(1-r)=R*xi*tss (Wang2014). )
    
    First, tss is estimated from the second part of the pulse decay (tsstfit), 
    after which A and B are extracted with the full fit window (tfit).
    Both tfit and tsstfit are with respect to the pulsestart.

    Returns:
    tss -- in µs
    A -- in the same units as pulse
    B -- arb.
    all with error"""
    
    t = np.arange(len(pulse)) - pulsestart
    
    if tsstfit is None:
        pulset = t[t>0]
        noisestd = np.std(pulse[:pulsestart])
        closemask = np.abs(pulse[t>0] - noisestd*1e-1) < noisestd*1e-2
        if any(closemask):
            tsstfitmax = pulset[closemask][0]
        else:
            tsstfitmax = t.max()
        tsstfitmin = tsstfitmax/2
        tsstfit = (tsstfitmin, tsstfitmax)
    
    if tfit is None:
        tfit = (0, tsstfit[1])

    # first extract the steady state (tail) decay time
    tss, tsserr = (
        np.array(exp(pulse, pulsestart, sfreq, tsstfit, tssguess, reterr=True))
        * 1e-6
        * sfreq
    ) #arb. units 

    # set-up for the fit:
    fitmask = np.logical_and(t > tfit[0], t < tfit[1])
    t2 = t[fitmask]
    peak2 = pulse[fitmask]

    # now do the fit:
    try:
        fit = curve_fit(lambda t, A, B: nonexp_func(t, tss, A, B),
                        t2, peak2, p0=(peak2[0], 1),
                       bounds=([0, 0], [np.inf, np.inf]))
        A, B = fit[0]
        Aerr, Berr = [fit[1][i, i] for i in range(2)]
    except:
        A, B, Aerr, Berr = np.ones(4)*np.nan

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(5, 7), sharex=True)
        for ax in axs:
            ax.plot(t, pulse, '.', label='data')
            ax.axhline(np.std(pulse[:pulsestart])*1e-1, color='k')
            ax.plot(t2, nonexp_func(t2, tss, A, B), 'r', label='1/t + exp. fit')
            ax.plot(t[pulsestart:], nonexp_func(t[pulsestart:],tss, A, B),
                 'r--')
            tssmask = (t > tsstfit[0]) & (t < tsstfit[1])
            ax.plot(t[tssmask], nonexp_func(t[tssmask], tss, A, B), 'r',
                    linewidth=4., label=f'$\\tau_{{ss}}$ fit-range\n $\\tau_{{ss}}$={tss:.0f} (1/fs)')
        axs[0].set_yscale("log")
        axs[0].set_ylim(1e-4, None)
        axs[0].legend()

    if reterr:
        return tss * 1e6 / sfreq, tsserr * 1e6 / sfreq, A, Aerr, B, Berr
    else:
        return tss * 1e6 / sfreq, A, B


def double_exp(pulse, pulsestart=100, sfreq=1e6,
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
        plt.plot(t, pulse, '.', label='data')
        plt.plot(t[fitmask], fitfun(t[fitmask], *fit[0]), 'r', label='double exp. fit')
        plt.plot(t[pulsestart:], fitfun(t[pulsestart:], *fit[0]),
                 'r--')
        plt.yscale("log")
        plt.legend()

    if reterr:
        pass
    else:
        return t1*1e6/sfreq, t2*1e6/sfreq, fit[0][0], fit[0][1]
    
def nonexps(fld, coord='ampphase', **nonexpkwargs):
    resultpath = fld + '/fits'
    
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)
    fnames = io.get_avlfiles(fld, ftype=f'{coord}.csv')
    KIDPrExs = np.unique(fnames[:, :3], axis=0)
    for k in tnrange(len(KIDPrExs), leave=False, desc='KID_Pr_Ex'):
        TmKs = [int(i.split('_')[-3][3:]) 
                    for i in glob.iglob(fld + '/' + '_'.join(KIDPrExs[k]) + f'*{coord}.csv')]
        fitres = np.zeros((len(TmKs), 3*2*2 + 1))
        for t in tnrange(len(TmKs), leave=False, desc='Temp'):
            fitres[t, 0] = TmKs[t]
            avgpulsedata = np.loadtxt(fld + '/' + '_'.join(KIDPrExs[k]) + f'_TmK{TmKs[t]}_avgpulse_{coord}.csv',
                                  delimiter=',', skiprows=1)
            fitres[t, 1:7] = nonexp(1 - avgpulsedata[:, 0], **nonexpkwargs, reterr=True)
            fitres[t, 7:16] = nonexp(avgpulsedata[:, 2], **nonexpkwargs, reterr=True)

        np.savetxt(resultpath + '/' + '_'.join(KIDPrExs[k]) + '.csv',
                   fitres[fitres[:, 0].argsort(), :], delimiter=',', 
                  header=('Temperature (mK), amp tss (µs), amp tss err (µs), amp A (arb.), amp A err (arb.), '
                          + 'amp B (arb.), amp B err (arb.), '
                          + 'phase tss (µs), phase tss err (µs), phase A (arb.), phase A err (arb.), '
                          + 'phase B (arb.), phase B err (arb.),')
                  )
        
def show(fld, pulsestart=100, coords='ampphase'):
    '''Plots the fitted Lorentzians together with PSD for amp, phase and cross.
    Needs interactive matplotlib back-end''' 
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex='col', sharey='row')
    plt.ion()
    
    def plotfit(file):
        avgpulse = np.loadtxt(fld + '/' + file, delimiter=',', skiprows=1)
        t = np.arange(len(avgpulse[:, 0])) - pulsestart
        fitres = np.loadtxt(fld + '/fits/' + '_'.join(file.split('_')[:3]) + '.csv', 
                           delimiter=',', ndmin=2)
        TmK = int(file.split('_')[-3].split('.')[0][3:])
        
        for j in range(2):
            for i, (ax, coord) in enumerate(zip(axs[j, :], ['amp', 'phase'])):
                ax.cla()
                if i==0:
                    ax.plot(t, 1 - avgpulse[:, 0], '.')
                else:
                    ax.plot(t, avgpulse[:, 2], '.')
                        
                tss, tsserr, A, Aerr, B, Berr = fitres[fitres[:, 0] == TmK, 
                                                       (6*i + 1):(6*(i+1) + 1)][0]
                ax.plot(t[t>0], nonexp_func(t[t>0], tss, A, B),
                       label=('fit:\n' 
                              + f' $\\tau={tss:.0f} \pm {tsserr:.1f}~\mu s$\n'
                              + f' A={A:.0e} $\pm$ {Aerr:.0e}\n'
                              + f' B={B:.0e} $\pm$ {Berr:.0e}'))
                if j==0:
                    ax.legend(loc=(0, 1), title=coord)
                if j==1:
                    ax.set_ylim(1e-4, None)
            axs[1, j].set_yscale('log')
            axs[1, j].set_xlabel('Time (1/fs)')
            axs[j, 0].set_ylabel('Response')
        fig.suptitle(file)
        fig.tight_layout()
    
    fileids = io.get_avlfileids(fld, ftype=f'{coords}.csv')
    fileids = np.unique(fileids[:, :4], axis=0)
    files = [(f'KID{fileids[i, 0]:.0f}_{fileids[i, 1]:.0f}dBm_' 
             + ('_' if fileids[i, 2]==0 else f'Tchip{fileids[i, 2]:.2f}_')
             + f'TmK{fileids[i, 3]:.0f}_avgpulse_ampphase.csv') for i in range(len(fileids))]
    interact(plotfit, file=files)
