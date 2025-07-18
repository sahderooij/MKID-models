import numpy as np
import warnings
import matplotlib.pyplot as plt
from ipywidgets import interact

import matplotlib
from scipy import interpolate
import scipy.constants as const

import SC as SuperCond
from kidata import io
from kidata.noise import analysis
from kidata.IQ import to_ampphase

import SCtheory as SCth

def fromfld(fld, std=6, ppd=30):
    """Needs matplotlib widget back-end."""
    fig, ax = plt.subplots()
    plt.ion()

    def plotspec(fileid):
        ax.cla()
        freqs, Saas, Spps, Saps = np.zeros((4, 2), dtype='O')
        fileidsplit = fileid.split('_')
        KID = int(fileidsplit[0][3:])
        Pread = int(fileidsplit[1][:-3])
        T = int(fileidsplit[-1][3:])
        for i, (sfreq, name, nperseg) in enumerate(
            zip([5e4, 1e6], ['med', 'fast'], ['seglen', '1e-4s'])):
            data = io.get_bin(fld, KID, Pread, T, name)
            amp, phase = to_ampphase(data)
            freqs[i], Saas[i], Spps[i], Saps[i] = analysis.PSDs(
                amp, phase, sfreq, nperseg, std, 32, 'fail', plotfail=False)
            
        stitchfreq = 2e4
        freq = np.concatenate((freqs[0][freqs[0] < stitchfreq],
                         freqs[1][freqs[1] > stitchfreq]))
        Saa = np.concatenate((Saas[0][freqs[0] < stitchfreq],
                             Saas[1][freqs[1] > stitchfreq]))
        Spp = np.concatenate((Spps[0][freqs[0] < stitchfreq],
                             Spps[1][freqs[1] > stitchfreq]))
        Sap = np.concatenate((Saps[0][freqs[0] < stitchfreq],
                             Saps[1][freqs[1] > stitchfreq]))
        f, Saa = analysis.logsmooth(freq, Saa, ppd)
        f, Spp = analysis.logsmooth(freq, Spp, ppd)
        f, Sap = analysis.logsmooth(freq, Sap, ppd)
        ax.plot(f, np.real(Saa), label='amp')
        ax.plot(f, np.real(Spp), label='phase')
        ax.plot(f, -1*np.real(Sap), label='cross, neg.')
        ax.plot(f, np.real(Sap), label='cross, pos.')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Noise level (1/Hz)')
        ax.set_xlabel('Frequency (Hz)')
        ax.legend()
        ax.set_title(fileid)
    
    fileids = io.get_avlfileids(fld)
    fileids = np.unique(fileids[:, (0, 1, 2, 4)], axis=0)
    files = [(f'KID{fileids[i, 0]:.0f}_{fileids[i, 1]:.0f}dBm_' 
             + f'_TmK{fileids[i, 3]:.0f}') for i in range(len(fileids))]
    interact(plotspec, fileid=files)


def PSDs(chip, KIDs=None, pltPreads='all', Tminmax=(None, None), specinds=[0, 1, 2]):
    if io.get_datafld().replace('/', '') in chip:
        KIDPrT = io.get_avlfileids(chip, ftype='.csv')[:, (0, 1, -1)].astype('int')
        fld = chip.split('\\')[2]
        chip = chip.split('\\')[1]
    else:
        KIDPrT = io.get_noiseKIDPrT(chip)
        fld = 'Noise_vs_T'
    
    if KIDs is None:
        KIDs = np.unique(KIDPrT[:, 0])
    
    for KID in KIDs:
        Preads = io.selectPread(pltPreads, np.unique(KIDPrT[KIDPrT[:, 0] == KID, 1]))
        for Pread in Preads:
            Temps = KIDPrT[(KIDPrT[:, 0] == KID) & (KIDPrT[:, 1] == Pread), 2]
            if Tminmax != (None, None):
                Temps = Temps[np.logical_and(Temps < Tminmax[1], Temps > Tminmax[0])]
            if len(Temps) > 0:
                fig, axs = plt.subplots(1, len(specinds),
                    figsize=(4*len(specinds), 4),
                    sharex=True,
                    sharey=True)
                fig.suptitle(f"{chip}, KID{KID}, -{Pread}dBm")
                cmap = matplotlib.colormaps.get_cmap("viridis")
                norm = matplotlib.colors.Normalize(Temps.min(), Temps.max())

                for Temp in Temps:
                    PSDs = io.get_noisePSD(chip, KID, Pread, Temp, fld=fld)
                    for i, specind in enumerate(specinds):
                        axs[i].plot(PSDs[:, 0], PSDs[:, specind + 1], color=cmap(norm(Temp)))
                titledict = {0: 'amp', 
                             1: 'phase', 
                             2:'cross', 
                             3:'cross pos.', 
                             4:'cross imag neg.', 
                             5:'cross imag pos.'}
                for ax, specind in zip(axs, specinds):
                    ax.set_title(titledict[specind])
                    ax.set_xlabel('Frequency (Hz)')
                axs[0].set_xscale('log')
                axs[0].set_ylabel('Noise level (dBc/Hz)')
                clb = plt.colorbar(
                        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[-1]
                    )
                clb.ax.set_title("T (mK)")
                plt.show()
            
def fits(chip, KIDs=None, pltPreads='all', Tminmax=(None, None), relerrthrs=.2):
    KIDPr = np.unique(io.get_noiseKIDPrT(chip)[:-1], axis=0)
    if KIDs is None:
        KIDs = np.unique(KIDPr[:, 0])
    
    for KID in KIDs:
        Preads = io.selectPread(pltPreads, np.unique(KIDPr[KIDPr[:, 0] == KID, 1]))
        fig, axs = fig, axs = plt.subplots(2, 3,
                figsize=(12, 4),
                sharex=True,
                sharey='row')
        fig.suptitle(f"{chip}, KID{KID}")
        cmap = matplotlib.colormaps["plasma"]
        norm = matplotlib.colors.Normalize(-Preads.max(), -Preads.min())
        for Pread in Preads:
            fit = io.get_noisefits(chip, KID, Pread)
            if Tminmax != (None, None):
                mask1 = ((fit[:, 0] > Tminmax[0]) & (fit[:, 0] < Tminmax[1]))
            else:
                mask1 = np.ones(len(fit[:, 0]), dtype='bool')

            for i in range(3):
                mask2 = fit[:, 4*i+2]/fit[:, 4*i+1] < relerrthrs
                mask = mask1 & mask2
                for j in range(2):                    
                    axs[j, i].errorbar(fit[mask, 0], fit[mask, (2*j+4*i) + 1], 
                                    yerr=fit[mask, (2*j+4*i) + 2],
                                    color=cmap(norm(-Pread)), fmt='-o')
        for row in range(2):
            axs[row, 0].set_yscale('log')
            clb = plt.colorbar(
                    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[row, -1]
                )
            clb.ax.set_title("$P_{read}$ (dBm)")
        axs[0, 0].set_ylabel('Lifetime (µs)')
        axs[1, 0].set_ylabel('Level (1/Hz)')
        for ax in axs[-1, :]:
            ax.set_xlabel('Temperature (mK)')
        for ax, title in zip(axs[0, :], ['amp', 'phase', 'cross']):
            ax.set_title(title)