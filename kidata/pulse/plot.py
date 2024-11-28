import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from kidata import io


def fit_res(chip, KID, wvl, pltPreads='all', Tminmax=(None, None), relerrthrs=1):
    '''Plots the fit results vs T of the non-exp fit, which is located in 
    '<chip>/Pulse/<wvl>nm/fits'.
    The relative error thershold (relerrthrs) is defined as tsserr/tss of
    the phase, and only points with tsserr/tss less then relerrthrs are plotted.'''
    Preads = io.selectPread(pltPreads, io.get_pulsePread(chip, KID, wvl))
    fig, axs = plt.subplots(4, 2,
            figsize=(8, 8),
            sharex=True,
            sharey='row')
    fig.suptitle(f"{chip}, KID{KID}")
    cmap = matplotlib.colormaps["plasma"]
    norm = matplotlib.colors.Normalize(-Preads.max(), -Preads.min())
    
    for Pread in Preads:
        fits = io.get_pulsefits(chip, KID, Pread, wvl)
        mask = fits[:, 8]/fits[:, 7] < relerrthrs
        for v in range(2):
            for p in range(3):
                axs[p, v].errorbar(fits[mask, 0], fits[mask, 6*v + 2*p + 1],
                                   fits[mask, 6*v + 2*p + 2], 
                                   color=cmap(norm(-Pread)), fmt='o-')
            axs[3, v].plot(fits[mask, 0], fits[mask, 6*v + 3]/fits[mask, 6*v + 4], 
                           'o-', color=cmap(norm(-Pread)))
            
    for v, var in enumerate(['Amplitude', 'Phase']):
        axs[0, v].set_title(var)
        axs[-1, v].set_xlabel('Temperature (mK)')
        
    for p, param in enumerate(['$\\tau_{ss}$ (µs)', 'A', 'B', 'A/B']):
        axs[p, 0].set_yscale('log')
        axs[p, 0].set_ylabel(param)

        clb = plt.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[p, -1]
        )
        clb.ax.set_title("$P_{read}$ (dBm)")
    