import numpy as np
import warnings
import matplotlib.pyplot as plt

import matplotlib
from scipy import interpolate
import scipy.constants as const

import SC as SuperCond
from kidata import io
from kidata import calc
from kidata import noise

import SCtheory as SCth


def Nqp(
    chip,
    KIDs=None,
    pltPread="all",
    spec=["cross"],
    Tminmax=None,
    relerrthrs=0.3,
    pltThrm=True,
    SC=None,
    nqpaxis=True,
    inclpulse=False,
):
    """Plots the number of quasiparticle calculated from the noise levels and lifetimes from PSDs.
    options similar to options in ltnlvl.
    
    pltThrm -- also plot thermal line (needs constants)
    nqpaxis -- also shows density on right axis."""
    
    specdict = {'amp': [0, '-^'], 'phase': [1, '-v'], 'cross': [2, '-o']}
    KIDPrT = io.get_noiseKIDPrT(chip)
    if KIDs is None:
        KIDs = np.unique(KIDPrT[:, 0])
        
    for KID in KIDs:
        fig, ax = plt.subplots()
        fig.suptitle(f'{chip}, KID{KID}')
        Preads = np.unique(KIDPrT[KIDPrT[:, 0] == KID, 1])
        Preadar = io.selectPread(pltPread, Preads)

        cmap = matplotlib.cm.get_cmap("plasma")
        norm = matplotlib.colors.Normalize(-1*Preadar.max(), -1*Preadar.min())
        clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        clb.ax.set_title(r"$P_{read}$ (dBm)")

        for Pread in Preadar:           
            for sp in spec:
                T, Nqp, Nqperr = calc.Nqp(chip, KID, Pread, spec=sp)
                if Tminmax is not None:
                    Tmask = (T > Tminmax[0]) & (T < Tminmax[1])
                else:
                    Tmask = np.ones(len(T), dtype='bool')
                
                mask = Tmask & (Nqperr/Nqp < relerrthrs)
                ax.errorbar(T[mask], Nqp[mask], yerr=Nqperr[mask], 
                            color=cmap(norm(-Pread)), fmt=specdict[sp][1],
                           capsize=2, mec='k')
            if inclpulse:
                try:
                    pfits = io.get_pulsefits(chip, KID, Pread, inclpulse)
                    phresp = interpolate.splev(pfits[:, 0]*1e-3, calc.Respspl(chip, KID, Pread, var='phase'))
                    Nqp = pfits[:, 9] / phresp
                    Nqperr = pfits[:, 10] / phresp
                    mask = Nqperr/Nqp < relerrthrs
                    ax.errorbar(pfits[mask,  0], Nqp[mask], yerr=Nqperr[mask],
                                color=cmap(norm(-Pread)), fmt='-s',
                                   capsize=2, mec='k')
                except:
                    pass
                            
                                           

        if pltThrm:
            if SC is None:
                SC = SuperCond.Al
            SCvol = SuperCond.init_SCvol(chip, KID, SC, set_tesc=False)
            T = np.linspace(*ax.get_xlim(), 100)
            NqpT = np.zeros(100)
            for i in range(len(T)):
                D_ = SCth.D(const.Boltzmann / const.e * 1e6 * T[i] * 1e-3, SCvol.SC)
                NqpT[i] = SCvol.V * SCth.nqp(
                    const.Boltzmann / const.e * 1e6 * T[i] * 1e-3, D_, SCvol.SC
                )
            ax.plot(
                T, NqpT, color='k', 
                zorder=len(ax.lines) + 1, label="Thermal $N_{qp}$"
            )
            ax.legend()

        ax.set_ylabel("$N_{qp}$")
        ax.set_xlabel("Temperature (mK)")
        ax.set_yscale("log")
        if ax.get_ylim()[0] < 1e-1:
            ax.set_ylim(1e-1, None)

        if nqpaxis:

            def nqptoNqp(x):
                return x * SCvol.V

            def Nqptonqp(x):
                return x / SCvol.V

            ax2 = ax.secondary_yaxis("right", functions=(Nqptonqp, nqptoNqp))
            ax2.set_ylabel("$n_{qp}$ ($\\mu m^{-3}$)")
            l, b, w, h = clb.ax.get_position().bounds
            clb.ax.set_position([l + 0.12, b, w, h])



def Qif0(
    Chipnum,
    KIDnum,
    color="Pread",
    Tmax=0.5,
    pltPread="all",
    fracfreq=False,
    fig=None,
    ax12=None,
    xaxis="T",
    **kwargs,
):
    """Plot the internal quality factor and resonance frequency from S21-measurement.
    The color gives different read powers, but can be set to Pint as well.
    If fracfreq is True, the y-axis is df/f0, instead of f0."""
    dfld = io.get_datafld()
    if fig is None or ax12 is None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{Chipnum}, KID{KIDnum}")
    else:
        axs = ax12

    Preadar = io.selectPread(pltPread, io.get_S21Pread(Chipnum, KIDnum))
    if color == "Pread":
        cmap = matplotlib.cm.get_cmap("plasma")
        norm = matplotlib.colors.Normalize(-1.05 * Preadar.max(), -0.95 * Preadar.min())
        clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[-1])
        clb.ax.set_title(r"$P_{read}$ (dBm)")
    elif color == "Pint":
        Pint = np.array(io.get_Pintdict(Chipnum)[KIDnum])
        cmap = matplotlib.cm.get_cmap("plasma")
        norm = matplotlib.colors.Normalize(Pint.min() * 1.05, Pint.max() * 0.95)
        clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
        clb.ax.set_title(r"$P_{int}$ (dBm)")

    for Pread in Preadar:
        S21data = io.get_S21data(Chipnum, KIDnum, Pread)
        if color == "Pread":
            clr = cmap(norm(-Pread))
        elif color == "Pint":
            clr = cmap(norm(Pint[Preadar == Pread][0]))
        else:
            clr = color

        T = S21data[:, 1]

        # set what will be on the x and y axis
        if xaxis == "T":
            xvalues = T * 1e3
            yvaluesQ = S21data[:, 4]
        elif xaxis == "Nqp":
            xvalues = S21data[:, 12]
            yvaluesQ = 1 / S21data[:, 4]
        else:
            raise ValueError("Not a valid xaxis argument")

        axs[0].plot(xvalues[T < Tmax], yvaluesQ[T < Tmax], color=clr, **kwargs)
        if fracfreq:
            axs[1].plot(
                xvalues[T < Tmax],
                (S21data[T < Tmax, 5] - S21data[0, 5]) / S21data[0, 5],
                color=clr,
                **kwargs,
            )
        else:
            axs[1].plot(
                xvalues[T < Tmax], S21data[T < Tmax, 5] * 1e-9, color=clr, **kwargs
            )

    for ax in axs:
        if xaxis == "T":
            ax.set_xlabel("Temperature (mK)")
        elif xaxis == "Nqp":
            ax.set_xlabel("$N_{qp}$")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    if xaxis == "T":
        axs[0].set_ylabel("$Q_i$")
        axs[0].set_yscale("log")
    elif xaxis == "Nqp":
        axs[0].set_ylabel("$1/Q_i$")

    if fracfreq:
        axs[1].set_ylabel("$\delta f_{res}/f_0$")
    else:
        axs[1].set_ylabel("f0 (GHz)")
    fig.tight_layout()


def Qfactors(Chipnum, KIDnum, Pread=None, ax=None):
    """Plots Ql, Qi and Qc over temperature in one figure."""
    S21data = io.get_S21data(Chipnum, KIDnum, Pread)
    T = S21data[:, 1] * 1e3

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(T, S21data[:, 2], label="$Q$")
    ax.plot(T, S21data[:, 3], label="$Q_c$")
    ax.plot(T, S21data[:, 4], label="$Q_i$")
    ax.set_yscale("log")
    ax.set_ylabel("Q-factor")
    ax.set_xlabel("Temperature (mK)")
    ax.legend()


def f0(Chipnum, KIDnum, Pread=None, ax=None):
    """Plots resonance frequency over temperature"""
    S21data = io.get_S21data(Chipnum, KIDnum, Pread)
    T = S21data[:, 1] * 1e3

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(T, S21data[:, 5] * 1e-9)
    ax.set_xlabel("Temperature (mK)")
    ax.set_ylabel("$f_0$ (GHz)")
    ax.ticklabel_format(useOffset=False)


def Qfactorsandf0(Chipnum, KIDnum, Pread=None, fig=None, ax12=None):
    """Plots both Qfactors and resonance frequency over temperature in one figure"""
    if ax12 is None or fig is None:
        fig, ax12 = plt.subplots(1, 2, figsize=(9, 3))
    fig.suptitle("{}, KID{}, -{} dBm".format(Chipnum, KIDnum, Pread))
    Qfactors(Chipnum, KIDnum, Pread, ax=ax12[0])
    f0(Chipnum, KIDnum, Pread, ax=ax12[1])
    fig.tight_layout(rect=(0, 0, 1, 0.9))


def Powers(Chipnum, KIDnum, Pread=None, ax=None):
    """Plots the read power, internal power and absorbed power vs temperature in one figure"""
    S21data = io.get_S21data(Chipnum, KIDnum, Pread)

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title("{}, KID{}, {} dBm".format(Chipnum, KIDnum, S21data[0, 7]))
    Q = S21data[:, 2]
    Qc = S21data[:, 3]
    Qi = S21data[:, 4]
    T = S21data[:, 1] * 1e3
    ax.plot(T, S21data[:, 7], label="$P_{read}$")
    ax.plot(T, S21data[:, 8], label="$P_{int}$")
    ax.plot(
        T,
        10 * np.log10(10 ** (S21data[0, 7] / 10) / 2 * 4 * Q ** 2 / (Qi * Qc)),
        label="$P_{abs}$",
    )
    ax.set_ylabel("Power (dBm)")
    ax.set_xlabel("Temperature (mK)")
    ax.legend()


def PowersvsT(Chipnum, KIDnum, density=False, phnum=False, fig=None, axs=None):
    """Plots the internal and absorbed powers over temperature, for different read powers (color). 
    Options:
    Density -- the powers are devided by the superconductor volume
    phnum -- the powers are expressed in resonator photon occupations"""
    if axs is None or fig is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    Preadar = io.get_S21Pread(Chipnum, KIDnum)
    cmap = matplotlib.cm.get_cmap("plasma")
    norm = matplotlib.colors.Normalize(-1.05 * Preadar.max(), -0.95 * Preadar.min())
    clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[-1])
    clb.ax.set_title(r"$P_{read}$ (dBm)")
    for Pread in Preadar:
        S21data = io.get_S21data(Chipnum, KIDnum, Pread)
        Q = S21data[:, 2]
        Qc = S21data[:, 3]
        Qi = S21data[:, 4]
        T = S21data[:, 1] * 1e3
        Pabs = 10 ** (S21data[0, 7] / 10) / 2 * 4 * Q ** 2 / (Qi * Qc) * 1e-3 / const.e
        Pint = 10 ** (S21data[:, 8] / 10) * 1e-3 / const.e
        if phnum:
            Pabs /= const.Planck / const.e * 1e12 * (S21data[:, 5] * 1e-6) ** 2
            Pint /= onst.Planck / const.e * 1e12 * (S21data[:, 5] * 1e-6) ** 2

        if density:
            Pabs /= S21data[0, 14]
            Pint /= S21data[0, 14]
        axs[1].plot(T, Pabs, color=cmap(norm(-1 * Pread)))
        axs[0].plot(T, Pint, color=cmap(norm(-1 * Pread)))

    title0 = "Internal Power"
    title1 = "Absorbed Power"
    if phnum:
        ylabel = "$N_{\gamma}^{res}$"
    else:
        ylabel = "$eV~s^{-1}$"

    if density:
        ylabel += " $\mu m^{-3}$"
        title0 += " Density"
        title1 += " Density"

    axs[0].set_title(title0)
    axs[1].set_title(title1)
    axs[0].set_ylabel(ylabel)
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    return fig, axs


def tres(Chipnum, KIDnum, Pread=None, ax=None, label=None):
    """Plots the resonator ring-time as a function of temperature."""
    S21data = io.get_S21data(Chipnum, KIDnum, Pread)

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title(f"{Chipnum}, KID{KIDnum}, {S21data[0,7]} dBm")

    T = S21data[:, 1] * 1e3
    w = 2 * np.pi * S21data[:, 5] * 1e-6
    Q = S21data[:, 2]
    tres = 2 * Q / w

    ax.plot(T, tres, label=label)
    ax.set_xlabel("Temperature (mK)")
    ax.set_ylabel(r"$\tau_{res}$ (µs)")
    ax.set_yscale("log")


def Nphres(Chipnum, KIDnum, Pread=None, ax=None, label=None):
    """Plots the number of resonator photons in the resonator over temperature."""
    S21data = io.get_S21data(Chipnum, KIDnum, Pread)

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title(f"{Chipnum}, KID{KIDnum}, {S21data[0,7]} dBm")

    T = S21data[:, 1] * 1e3
    Pint = S21data[:, 8]  # dBm
    w = 2 * np.pi * S21data[:, 5]  # rad./s
    Nphres = 2 * np.pi * 10 ** (Pint / 10) * 1e-3 / (const.hbar * w ** 2)

    ax.plot(T, Nphres, label=label)
    ax.set_xlabel("Temperature (mK)")
    ax.set_ylabel("Number of photons")
    ax.set_yscale("log")


def Nphabsres(Chipnum, KIDnum, Pread=None, ax=None, label=None):
    """Plots the number of absorbed resonator photons over temperature."""
    S21data = io.get_S21data(Chipnum, KIDnum, Pread)
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title(f"{Chipnum}, KID{KIDnum}, {S21data[0,7]} dBm")

    Q = S21data[:, 2]
    Qc = S21data[:, 3]
    Qi = S21data[:, 4]
    T = S21data[:, 1] * 1e3
    Pabs = 10 * np.log(10 ** (S21data[0, 7] / 10) / 2 * 4 * Q ** 2 / (Qi * Qc)) # mW

    w = 2 * np.pi * S21data[:, 5] #rad./s
    Nphabsres = (
        2 * np.pi * 10 ** (Pabs / 10) *1e-3 / (const.hbar * w ** 2) * S21data[0, 14]
    )

    ax.plot(T, Nphabsres, label=label)
    #     ax.plot(T,Pabs)
    ax.set_xlabel("Temperature (mK)")
    ax.set_ylabel("Absorbed photons per cycle")
    ax.set_yscale("log")