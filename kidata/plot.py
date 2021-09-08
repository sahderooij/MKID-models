import numpy as np
import warnings
import matplotlib.pyplot as plt

import matplotlib
from scipy import interpolate
import scipy.constants as const

import SC as SuperCond
from kidata import io
from kidata import calc
from kidata import filters

import kidcalc


def _selectPread(pltPread, Preadar):
    """Function that returns a Pread array, depending on the input pltPread."""
    if type(pltPread) is str:
        if pltPread == "min":
            Pread = np.array([Preadar.max()])
        elif pltPread == "med":
            Pread = np.array([Preadar[np.abs(Preadar.mean() - Preadar).argmin()]])
        elif pltPread == "max":
            Pread = np.array([Preadar.min()])
        elif pltPread == "minmax":
            Pread = np.array([Preadar.max(), Preadar.min()])
        elif pltPread == "minmedmax":
            Pread = np.array(
                [
                    Preadar.max(),
                    Preadar[np.abs(Preadar.mean() - Preadar).argmin()],
                    Preadar.min(),
                ]
            )
        elif pltPread == "all":
            Pread = Preadar[::-1]
        else:
            raise ValueError("{} is not a valid Pread selection".format(pltPread))
    elif type(pltPread) == list:
        Pread = np.array(pltPread)
    elif type(pltPread) == int:
        Pread = np.array([np.sort(Preadar)[pltPread]])
    elif type(pltPread) == np.ndarray:
        Pread = pltPread
    else:
        raise ValueError("{} is not a valid Pread selection".format(pltPread))
    return Pread


def spec(
    Chipnum,
    KIDlist=None,
    pltPread="all",
    spec="all",
    lvlcomp="",
    SC=None,
    SCkwargs={},
    clbar=True,
    cmap=None,
    norm=None,
    del1fNoise=False,
    delampNoise=False,
    del1fnNoise=False,
    suboffres=False,
    plttres=False,
    Tminmax=(0, 500),
    ax12=None,
    xlim=(None, None),
    ylim=(None, None),
):
    """Plots PSDs of multiple KIDs, read powers and temperatures (indicated by color). Every KID has a new figure, which is returned if only one KID is plotted.
    lvlcomp specifies how the noise levels should be compensated (will be a different function in the future). 
    Use Resp to divide by responsivity and obtain quasiparticle fluctuations.
    Use Resptres to compensate for the factor (1+(omega*taures)^2) and get the quasiparticle fluctuations.
    plttres will plot arrow at the frequencies corresponding to the resonator ring time."""
    TDparam = io.get_grTDparam(Chipnum)
    if suboffres:
        TDparamoffres = io.get_grTDparam(Chipnum, offres=True)

    if KIDlist is None:
        KIDlist = io.get_grKIDs(TDparam)
    elif type(KIDlist) is int:
        KIDlist = [KIDlist]

    if spec == "all":
        specs = ["cross", "amp", "phase"]
    elif type(spec) == list:
        specs = spec
    else:
        raise ValueError("Invalid Spectrum Selection")

    for KIDnum in KIDlist:
        if lvlcomp != "":
            if SC is None:
                SC_inst = SuperCond.init_SC(Chipnum, KIDnum, **SCkwargs)
            else:
                SC_inst = SC
        else:
            SC_inst = SuperCond.Al()
        Preadar = _selectPread(pltPread, io.get_grPread(TDparam, KIDnum))
        if ax12 is None:
            fig, axs = plt.subplots(
                len(Preadar),
                len(specs),
                figsize=(4 * len(specs), 4 * len(Preadar)),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            fig.suptitle(f"{Chipnum}, KID{KIDnum}")
        else:
            axs = ax12

        for ax1, Pread in zip(range(len(Preadar)), Preadar):

            axs[ax1, 0].set_ylabel("PSD (dBc/Hz)")
            Temp = io.get_grTemp(TDparam, KIDnum, Pread)
            if suboffres:
                Temp = np.intersect1d(Temp, io.get_grTemp(TDparamoffres, KIDnum, Pread))

            Temp = Temp[np.logical_and(Temp < Tminmax[1], Temp > Tminmax[0])]
            if cmap is None or norm is None:
                cmap = matplotlib.cm.get_cmap("viridis")
                norm = matplotlib.colors.Normalize(Temp.min(), Temp.max())
            if clbar:
                clb = plt.colorbar(
                    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[ax1, -1]
                )
                clb.ax.set_title("T (mK)")
            if plttres:
                S21data = io.get_S21data(Chipnum, KIDnum, Pread)

            for i in range(len(Temp)):
                for (ax2, spec) in zip(range(len(specs)), specs):
                    lvlcompspl = calc.NLcomp(
                        Chipnum, KIDnum, Pread, SC=SC_inst, method=lvlcomp, var=spec
                    )
                    freq, SPR = io.get_grdata(
                        TDparam, KIDnum, Pread, Temp[i], spec=spec
                    )
                    if suboffres:
                        orfreq, orSPR = io.get_grdata(
                            TDparamoffres, KIDnum, Pread, Temp[i], spec=spec
                        )
                        freq, SPR = filters.subtr_spec(freq, SPR, orfreq, orSPR)

                    if delampNoise:
                        freq, SPR = filters.del_ampNoise(freq, SPR)
                    if del1fNoise:
                        freq, SPR = filters.del_1fNoise(freq, SPR)
                    if del1fnNoise:
                        freq, SPR = filters.del_1fnNoise(freq, SPR)

                    SPR[SPR == -140] = np.nan
                    SPR[SPR == -np.inf] = np.nan

                    SPR = 10 * np.log10(
                        10 ** (SPR / 10) / interpolate.splev(Temp[i] * 1e-3, lvlcompspl)
                    )

                    axs[ax1, ax2].plot(freq, SPR, color=cmap(norm(Temp[i])))
                    axs[ax1, ax2].set_xscale("log")
                    axs[ax1, ax2].set_title(spec + ", -{} dBm".format(Pread))
                    axs[-1, ax2].set_xlabel("Freq. (Hz)")

                    if plttres:
                        Tind = np.abs(S21data[:, 1] - Temp[i] * 1e-3).argmin()
                        fres = S21data[Tind, 5] / (2 * S21data[Tind, 2])
                        axs[ax1, ax2].annotate(
                            "",
                            (fres, 1),
                            (fres, 1.25),
                            arrowprops=dict(
                                arrowstyle="simple", color=cmap(norm(Temp[i])), ec="k"
                            ),
                            annotation_clip=False,
                            xycoords=('data', 'axes fraction')
                        )
        axs[0, 0].set_xlim(*xlim)
        axs[0, 0].set_ylim(*ylim)
    #         plt.tight_layout()
    if ax12 is None and len(KIDlist) == 1:
        return fig, axs


def ltnlvl(
    Chipnum,
    KIDlist=None,
    pltPread="all",
    spec="cross",
    Tminmax=None,
    startstopf=(None, None),
    lvlcomp="",
    pltTTc=False,
    delampNoise=False,
    del1fNoise=False,
    del1fnNoise=False,
    suboffres=False,
    relerrthrs=0.2,
    pltKIDsep=True,
    pltthlvl=False,
    pltkaplan=False,
    pltthmfnl=False,
    plttres=False,
    plttscat=False,
    fig=None,
    ax12=None,
    color="Pread",
    pltclrbar=True,
    fmt="-o",
    label=None,
    SC=None,
    SCkwargs={},
    showfit=False,
    savefig=False,
):
    """Plots the results from a Lorentzian fit to the PSDs of multiple KIDs, read powers and temperatures. 
    Two axes: 0: lifetimes 1: noise levels, both with temperature on the x-axis. The color can be specified and
    is Pread by default. 
    Options:
    startstopf -- defines the fitting window
    lvlcomp -- defines how the levels are compensated. Use Resp for responsivity compensation.
        (will be moved in the future)
    del{}Noise -- filter spectrum before fitting.
    relerrthrs -- only plot fits with a relative error threshold in lifetime less than this.
    pltKIDsep -- if True, different KIDs get a new figure.
    pltthlvl -- expected noise level is plotted as dashed line
    pltkaplan -- a kaplan fit (tesc as parameter) is plotted in the lifetime axis.
    pltthmfnl -- a noise level from the fitted lifetime and theoretical Nqp is plotted as well
    plttres -- the resonator ring time is plotted in the lifetime axis.
    ... multiple figure handling options ...
    ... options for the tesc deteremination ...
    showfit -- the fits are displayed in numerous new figures, for manual checking."""

    def _make_fig():
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        return fig, axs

    def _get_cmap(**kwargs):
        if color == "Pread":
            cmap = matplotlib.cm.get_cmap("plasma")
            norm = matplotlib.colors.Normalize(
                -1.05 * kwargs["Preadar"].max(), -0.95 * kwargs["Preadar"].min()
            )
            if pltclrbar:
                clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
                clb.ax.set_title(r"$P_{read}$ (dBm)")
        elif color == "Pint":
            cmap = matplotlib.cm.get_cmap("plasma")
            norm = matplotlib.colors.Normalize(
                kwargs["Pintar"].min() * 1.05, kwargs["Pintar"].max() * 0.95
            )
            if pltclrbar:
                clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
                clb.ax.set_title(r"$P_{int}$ (dBm)")
        elif color == "V":
            cmap = matplotlib.cm.get_cmap("cividis")
            norm = matplotlib.colors.Normalize(
                np.array(list(Vdict.values())).min(),
                np.array(list(Vdict.values())).max(),
            )
            if pltclrbar:
                clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
                clb.ax.set_title(r"Al Vol. ($\mu m^3$)")
        elif color == "KIDnum":
            cmap = matplotlib.cm.get_cmap("Paired")
            norm = matplotlib.colors.Normalize(
                np.array(KIDlist).min(), np.array(KIDlist).max()
            )
            if pltclrbar:
                clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
                clb.ax.set_title("KID nr.")
        else:
            raise ValueError("{} is not a valid variable as color".format(color))
        return cmap, norm

    TDparam = io.get_grTDparam(Chipnum)
    if suboffres:
        TDparamoffres = io.get_grTDparam(Chipnum, offres=True)

    if KIDlist is None:
        KIDlist = io.get_grKIDs(TDparam)
    elif type(KIDlist) is int:
        KIDlist = [KIDlist]

    if color == "Pint":
        Pintdict = io.get_Pintdict(Chipnum)

    if not pltKIDsep:
        if ax12 is None:
            fig, axs = _make_fig()
        else:
            axs = ax12
        if color == "Pint":
            Pintar = np.array([Pintdict[k] for k in KIDlist])
            cmap, norm = _get_cmap(Pintar=Pintar)
        elif color == "V":
            Vdict = io.get_Vdict(Chipnum)
            cmap, norm = _get_cmap(Vdict=Vdict)
        elif color == "Pread":
            Preaddict = io.get_Preaddict(Chipnum)
            Preadar = np.array([Preaddict[k] for k in KIDlist])
            cmap, norm = _get_cmap(Preadar=Preadar)
        elif color == "KIDnum":
            cmap, norm = _get_cmap(KIDlist=KIDlist)

    for KIDnum in KIDlist:
        Preadar = _selectPread(pltPread, io.get_grPread(TDparam, KIDnum))
        if pltKIDsep:
            if ax12 is None:
                fig, axs = _make_fig()
            else:
                axs = ax12

            if len(KIDlist) > 1:
                fig.suptitle(f"KID{KIDnum}")

            if color == "Pread":
                cmap, norm = _get_cmap(Preadar=Preadar)
            elif color == "Pint":
                cmap, norm = _get_cmap(Pintar=np.array(Pintdict[KIDnum]))

        if lvlcomp != "" or pltkaplan or pltthmfnl or pltthlvl or pltTTc:
            if SC is None:
                SC_inst = SuperCond.init_SC(Chipnum, KIDnum, **SCkwargs)
            else:
                SC_inst = SC
        else:
            SC_inst = SuperCond.Al()

        for Pread in Preadar:
            Temp = np.trim_zeros(io.get_grTemp(TDparam, KIDnum, Pread))
            lvlcompspl = calc.NLcomp(
                Chipnum, KIDnum, Pread, SC=SC_inst, method=lvlcomp, var=spec
            )

            if suboffres:
                Temp = np.intersect1d(Temp, io.get_grTemp(TDparamoffres, KIDnum, Pread))

            if Tminmax != None:
                if Tminmax[0] != None:
                    Temp = Temp[Temp > Tminmax[0]]
                if Tminmax[1] != None:
                    Temp = Temp[Temp < Tminmax[1]]
            taut = np.zeros((len(Temp)))
            tauterr = np.zeros((len(Temp)))
            lvl = np.zeros((len(Temp)))
            lvlerr = np.zeros((len(Temp)))
            for i in range(len(Temp)):
                freq, SPR = io.get_grdata(TDparam, KIDnum, Pread, Temp[i], spec)
                if suboffres:
                    orfreq, orSPR = io.get_grdata(
                        TDparamoffres, KIDnum, Pread, Temp[i], spec
                    )
                    freq, SPR = filters.subtr_spec(freq, SPR, orfreq, orSPR)
                if delampNoise:
                    freq, SPR = filters.del_ampNoise(freq, SPR)
                if del1fNoise:
                    freq, SPR = filters.del_1fNoise(freq, SPR)
                if del1fnNoise:
                    freq, SPR = filters.del_1fnNoise(freq, SPR)

                if showfit:
                    print(
                        "{}, KID{}, -{} dBm, T={}, {}".format(
                            Chipnum, KIDnum, Pread, Temp[i], spec
                        )
                    )
                taut[i], tauterr[i], lvl[i], lvlerr[i] = calc.tau(
                    freq,
                    SPR,
                    plot=showfit,
                    retfnl=True,
                    startf=startstopf[0],
                    stopf=startstopf[1],
                )
                if showfit:
                    print(tauterr[i] / taut[i])

                lvl[i] = lvl[i] / interpolate.splev(Temp[i] * 1e-3, lvlcompspl)
                lvlerr[i] = lvlerr[i] / interpolate.splev(Temp[i] * 1e-3, lvlcompspl)

            # Deleting bad fits and plotting:
            mask = ~np.isnan(taut)
            mask[mask] = tauterr[mask] / taut[mask] <= relerrthrs

            if color == "Pread":
                clr = cmap(norm(-1 * Pread))
            elif color == "Pint":
                clr = cmap(norm(Pint))
            elif color == "V":
                clr = cmap(norm(Vdict[KIDnum]))
            elif color == "KIDnum":
                clr = cmap(norm(KIDnum))
            else:
                clr = color

            if pltTTc:
                Temp = Temp / (SC_inst.kbTc / (const.Boltzmann / const.e * 1e6) * 1e3)

            axs[0].errorbar(
                Temp[mask],
                taut[mask],
                yerr=tauterr[mask],
                fmt=fmt,
                capsize=3.0,
                color=clr,
                mec="k",
                label=label if Pread == Preadar[-1] else "",
            )
            axs[1].errorbar(
                Temp[mask],
                10 * np.log10(lvl[mask]),
                yerr=10 * np.log10((lvlerr[mask] + lvl[mask]) / lvl[mask]),
                fmt=fmt,
                capsize=3.0,
                color=clr,
                mec="k",
                label=label if Pread == Preadar[-1] else "",
            )
            if pltthlvl:
                if Tminmax is not None:
                    Tstartstop = Tminmax
                else:
                    Tstartstop = (Temp[mask].min(), Temp[mask].max())
                Ttemp = np.linspace(*Tstartstop, 100)
                if pltTTc:
                    Ttemp = Ttemp * (
                        SC_inst.kbTc / (const.Boltzmann / const.e * 1e6) * 1e3
                    )
                explvl = (
                    interpolate.splev(
                        Ttemp * 1e-3, calc.Respspl(Chipnum, KIDnum, Pread, var=spec)
                    )
                    ** 2
                )
                explvl *= (
                    4
                    * SC_inst.t0
                    * 1e-6
                    * SC_inst.V
                    * SC_inst.N0
                    * (SC_inst.kbTc) ** 3
                    / (2 * (SC_inst.D0) ** 2)
                    * (1 + SC_inst.tesc / SC_inst.tpb)
                    / 2
                )
                explvl /= interpolate.splev(Ttemp * 1e-3, lvlcompspl)
                if pltTTc:
                    Ttemp = Ttemp / (
                        SC_inst.kbTc / (const.Boltzmann / const.e * 1e6) * 1e3
                    )
                (thlvlplot,) = axs[1].plot(
                    Ttemp,
                    10 * np.log10(explvl),
                    color=clr,
                    linestyle="--",
                    linewidth=2.0,
                )
                axs[1].legend((thlvlplot,), (r"Expected noise level",))

            if pltkaplan and Temp[mask].size != 0:
                if Tminmax is not None:
                    Tstartstop = Tminmax
                else:
                    Tstartstop = (Temp[mask].min(), Temp[mask].max())
                T = np.linspace(*Tstartstop, 100)

                if pltTTc:
                    T = T * (SC_inst.kbTc / (const.Boltzmann / const.e * 1e6))
                else:
                    T = T * 1e-3
                taukaplan = kidcalc.tau_kaplan(T, SC_inst)
                if pltTTc:
                    T = T / (SC_inst.kbTc / (const.Boltzmann / const.e * 1e6))
                else:
                    T = T * 1e3

                axs[0].plot(
                    T,
                    taukaplan,
                    color=clr,
                    linestyle="--",
                    linewidth=2.0,
                    label="Kaplan, $\\tau_{qp}$",
                )

            if plttscat:
                if Tminmax is not None:
                    Tstartstop = Tminmax
                else:
                    Tstartstop = (Temp[mask].min(), Temp[mask].max())
                T = np.linspace(*Tstartstop, 100)

                if pltTTc:
                    T = T * (SC_inst.kbTc / (const.Boltzmann / const.e * 1e6))
                else:
                    T = T * 1e-3
                tscat = SC_inst.t0 / (
                    2.277
                    * (SC_inst.kbTc / (2 * SC_inst.D0)) ** 0.5
                    * (T / (SC_inst.kbTc / (const.Boltzmann / const.e * 1e6)))
                    ** (7 / 2)
                )
                if pltTTc:
                    T = T / (SC_inst.kbTc / (const.Boltzmann / const.e * 1e6))
                else:
                    T = T * 1e3
                axs[0].plot(
                    T,
                    tscat,
                    color=clr,
                    linestyle="-.",
                    linewidth=2.0,
                    label="Kaplan, $\\tau_s$",
                )

            if pltthmfnl:
                try:
                    if pltTTc:
                        Temp = Temp * (
                            SC_inst.kbTc / (const.Boltzmann / const.e * 1e6) * 1e3
                        )
                    tauspl = interpolate.splrep(Temp[mask], taut[mask], s=0)
                    T = np.linspace(Temp[mask].min(), Temp[mask].max(), 100)
                    Nqp = np.zeros(len(T))
                    for i in range(len(T)):
                        Nqp[i] = SC_inst.V * kidcalc.nqp(
                            T[i] * 1e-3 * const.Boltzmann / const.e * 1e6,
                            SC_inst.D0,
                            SC_inst,
                        )
                    thmfnl = (
                        4
                        * interpolate.splev(T, tauspl)
                        * 1e-6
                        * Nqp
                        * interpolate.splev(
                            T * 1e-3, calc.Respspl(Chipnum, KIDnum, Pread, var=spec)
                        )
                        ** 2
                    )
                    thmfnl /= interpolate.splev(T * 1e-3, lvlcompspl)
                    if pltTTc:
                        T = T / (SC_inst.kbTc / (const.Boltzmann / const.e * 1e6) * 1e3)
                    (thmfnlplot,) = axs[1].plot(
                        T,
                        10 * np.log10(thmfnl),
                        color=clr,
                        linestyle="--",
                        linewidth=3.0,
                    )
                    axs[1].legend(
                        (thmfnlplot,),
                        ("Thermal Noise Level \n with measured $\\tau_{qp}^*$",),
                    )
                except:
                    warnings.warn(
                        "Could not make Thermal Noise Level, {},KID{},-{} dBm,{}".format(
                            Chipnum, KIDnum, Pread, spec
                        )
                    )
            if plttres:
                S21data = io.get_S21data(Chipnum, KIDnum, Pread)
                (tresline,) = axs[0].plot(
                    S21data[:, 1] / S21data[0, 21] if pltTTc else S21data[:, 1] * 1e3,
                    S21data[:, 2] / (np.pi * S21data[:, 5]) * 1e6,
                    color=clr,
                    linestyle=":",
                )
                axs[0].legend((tresline,), ("$\\tau_{res}$",))
        axs[0].set_yscale("log")
        for i in range(2):
            if pltTTc:
                axs[i].set_xlabel("$T/T_c$")
            else:
                axs[i].set_xlabel("Temperature (mK)")
        axs[0].set_ylabel(r"$\tau_{qp}^*$ (µs)")

        if lvlcomp == "Resp":
            axs[1].set_ylabel(r"Noise Level/$\mathcal{R}^2(T)$ (dB/Hz)")
        elif lvlcomp == "RespV":
            axs[1].set_ylabel(r"Noise Level/$(\mathcal{R}^2(T)V)$ (dB/Hz)")
        elif lvlcomp == "RespVtescTc":
            axs[1].set_ylabel(r"Noise Level/$(\mathcal{R}^2(T)\chi)$ (dB/Hz)")
        elif lvlcomp == "":
            axs[1].set_ylabel(r"Noise Level (dB/Hz)")
        elif lvlcomp == "RespLowT":
            axs[1].set_ylabel(r"Noise Level/$\mathcal{R}^2(T=50 mK)$ (dB/Hz)")
        else:
            axs[1].set_ylabel(r"comp. Noise Level (dB/Hz)")
        plt.tight_layout()
        if savefig:
            plt.savefig("GR_{}_KID{}_{}.pdf".format(Chipnum, KIDnum, spec))
            plt.close()
    if ax12 is None:
        return fig, axs


def Nqp(
    Chipnum,
    KIDnum,
    pltPread="all",
    spec="cross",
    startstopf=(None, None),
    delampNoise=False,
    del1fNoise=False,
    del1fnNoise=False,
    Tmax=500,
    relerrthrs=0.3,
    pltThrm=True,
    pltNqpQi=False,
    splitT=0,
    pltNqptau=False,
    SC=None,
    SCkwargs={},
    nqpaxis=True,
    fig=None,
    ax=None,
    label=None,
    color=None,
    fmt="-o",
):
    """Plots the number of quasiparticle calculated from the noise levels and lifetimes from PSDs.
    options similar to options in ltnlvl.
    
    pltThrm -- also plot thermal line (needs constants)
    pltNqpQi -- plot Nqp from Qi as well (needs constants)
    splitT -- makes NqpQi line dashed below this T
    pltNqptau -- plot Nqp from lifetime only (need constants)
    nqpaxis -- also shows density on right axis."""

    TDparam = io.get_grTDparam(Chipnum)
    if ax is None or fig is None:
        fig, ax = plt.subplots()

    Preadar = _selectPread(pltPread, io.get_grPread(TDparam, KIDnum))

    if Preadar.size > 1:
        cmap = matplotlib.cm.get_cmap("plasma")
        norm = matplotlib.colors.Normalize(-1.05 * Preadar.max(), -0.95 * Preadar.min())
        clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        clb.ax.set_title(r"$P_{read}$ (dBm)")
    if SC is None:
        SC_inst = SuperCond.init_SC(Chipnum, KIDnum, set_tesc=pltNqptau, **SCkwargs)
    else:
        SC_inst = SC
    for Pread in Preadar:
        S21data = io.get_S21data(Chipnum, KIDnum, Pread)

        Respspl = calc.Respspl(Chipnum, KIDnum, Pread, var=spec)

        Temp = io.get_grTemp(TDparam, KIDnum, Pread)
        Temp = Temp[Temp < Tmax]
        Nqp, Nqperr, taut = np.zeros((3, len(Temp)))
        for i in range(len(Temp)):
            freq, SPR = io.get_grdata(TDparam, KIDnum, Pread, Temp[i], spec=spec)
            if delampNoise:
                freq, SPR = filters.del_ampNoise(freq, SPR)
            if del1fNoise:
                freq, SPR = filters.del_1fNoise(freq, SPR)
            if del1fnNoise:
                freq, SPR = filters.del_1fnNoise(freq, SPR)
            taut[i], tauterr, lvl, lvlerr = calc.tau(
                freq, SPR, retfnl=True, startf=startstopf[0], stopf=startstopf[1]
            )
            lvl = lvl / interpolate.splev(Temp[i] * 1e-3, Respspl) ** 2
            lvlerr = lvlerr / interpolate.splev(Temp[i] * 1e-3, Respspl) ** 2
            Nqp[i] = lvl / (4 * taut[i] * 1e-6)
            Nqperr[i] = np.sqrt(
                (lvlerr / (4 * taut[i] * 1e-6)) ** 2
                + (-lvl * tauterr * 1e-6 / (4 * (taut[i] * 1e-6) ** 2)) ** 2
            )
        mask = ~np.isnan(Nqp)
        mask[mask] = Nqperr[mask] / Nqp[mask] <= relerrthrs
        if Preadar.size > 1:
            color = cmap(norm(-1 * Pread))
        elif pltPread == "min":
            color = "purple"
        elif pltPread == "max":
            color = "gold"

        dataline = ax.errorbar(
            Temp[mask],
            Nqp[mask],
            yerr=Nqperr[mask],
            color=color,
            fmt=fmt,
            mec="k",
            capsize=2.0,
            label=label,
        )
        if pltNqptau:
            Nqp_ = SC_inst.V * kidcalc.nqpfromtau(taut, SC_inst)
            (tauline,) = ax.plot(
                Temp[mask],
                Nqp_[mask],
                color=color,
                zorder=len(ax.lines) + 1,
                label="$\\tau_{qp}^*$",
            )
    if pltNqpQi:
        Preadar = io.get_S21Pread(Chipnum, KIDnum)
        for Pread in Preadar:
            S21data = io.get_S21data(Chipnum, KIDnum, Pread)
            T, Nqp = calc.NqpfromQi(S21data)
            mask = np.logical_and(
                T * 1e3 > ax.get_xlim()[0], T * 1e3 < ax.get_xlim()[1]
            )
            totalT = T[mask]
            totalNqp = Nqp[mask]
            if len(Preadar) == 1:
                color = "g"
            else:
                color = cmap(norm(closestPread))
            (Qline,) = ax.plot(
                totalT[totalT > splitT] * 1e3,
                totalNqp[totalT > splitT],
                linestyle="-",
                color=color,
                zorder=len(ax.lines) + 1,
                label="$Q_i$",
            )
            ax.plot(
                totalT[totalT < splitT] * 1e3,
                totalNqp[totalT < splitT],
                linestyle="--",
                color=color,
                zorder=len(ax.lines) + 1,
            )
    if pltThrm:
        T = np.linspace(*ax.get_xlim(), 100)
        NqpT = np.zeros(100)
        for i in range(len(T)):
            D_ = kidcalc.D(const.Boltzmann / const.e * 1e6 * T[i] * 1e-3, SC_inst)
            NqpT[i] = SC_inst.V * kidcalc.nqp(
                const.Boltzmann / const.e * 1e6 * T[i] * 1e-3, D_, SC_inst
            )
        (Thline,) = ax.plot(
            T, NqpT, color="k", zorder=len(ax.lines) + 1, label="Thermal $N_{qp}$"
        )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_ylabel("$N_{qp}$")
    ax.set_xlabel("Temperature (mK)")

    ax.set_yscale("log")

    if nqpaxis:

        def nqptoNqp(x):
            return x * SC_inst.V

        def Nqptonqp(x):
            return x / SC_inst.V

        ax2 = ax.secondary_yaxis("right", functions=(Nqptonqp, nqptoNqp))
        ax2.set_ylabel("$n_{qp}$ ($\\mu m^{-3}$)")
    if Preadar.size > 1:
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

    Preadar = _selectPread(pltPread, io.get_S21Pread(Chipnum, KIDnum))
    if color == "Pread":
        cmap = matplotlib.cm.get_cmap("plasma")
        norm = matplotlib.colors.Normalize(-1.05 * Preadar.max(), -0.95 * Preadar.min())
        clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
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
    """Plots the read power, internal power and absorbed power over temperature in one figure"""
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
    clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
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
    w = 2 * np.pi * S21data[:, 5] * 1e-6  # 1/µs
    Nphres = 2 * np.pi * 10 ** (Pint / 10) / (const.hbar * 1e9 * w ** 2)

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
    Pabs = 10 * np.log(10 ** (S21data[0, 7] / 10) / 2 * 4 * Q ** 2 / (Qi * Qc))

    w = 2 * np.pi * S21data[:, 5] * 1e-6  # 1/µs
    Nphabsres = (
        2 * np.pi * 10 ** (Pabs / 10) / (const.hbar * 1e9 * w ** 2) * S21data[0, 14]
    )

    ax.plot(T, Nphabsres, label=label)
    #     ax.plot(T,Pabs)
    ax.set_xlabel("Temperature (mK)")
    ax.set_ylabel("Absorbed photons per cycle")
    ax.set_yscale("log")
