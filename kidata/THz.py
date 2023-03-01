import kidata as kd
from scipy import constants as const
from scipy.integrate import quad
from scipy.interpolate import splrep, splev
import numpy as np
import os

import SCtheory as SCth

hbar = const.hbar / const.e * 1e12  # µeV µs


class OptSys(object):
    """Class that defines optical system objects, which contains
    attributes that describe the optical system (excluding MKID).
    It has geometric properties (like optical throughput (solidangle * A) ),
    as well as frequency parameters such as filter transmission.
    """

    def __init__(
        self,
        eta_opt=1,
        solidangle=2 * np.pi,
        area=1,
        pol=2,
        filterpath=os.path.dirname(__file__) + "/../../THzAnalysis/filterfiles",
        filterstack=[],
        fminmax=(1e3, 1e7),
        df=1e3,
    ):
        self.eta_opt = eta_opt  # optical efficiency
        self.solidangle = solidangle  # [str], default is hemisphere
        self.area = area  # [µm^2]
        self.filterpath = filterpath
        self.filterstack = filterstack  # list of used filters
        self.fminmax = (
            fminmax  # [MHz] tuple with maximum and minium frequency to analyse
        )
        self.df = df  # [MHz] frequency spacing
        self.pol = pol

        self.set_freq()
        self.set_filters()
        self.set_TX()
        self.set_TXspl()
        self.cfreq = self.get_cfreq()

    def set_filters(self):
        filters = {}
        for fltr in np.unique(self.filterstack):
            fltrdata = np.loadtxt(self.filterpath + "/" + fltr + ".dat")
            fltrdata[:, 0] = (
                const.c * fltrdata[:, 0] * 1e2 * 1e-6
            )  # convert cm^-1 to MHz
            filters[fltr] = {
                "data": fltrdata,
                "spl": splrep(fltrdata[:, 0], fltrdata[:, 1]),
            }
        self.filters = filters

    def set_TX(self):
        """Caculates the transmission of all the filters"""
        self.TX = np.ones(len(self.freq))
        for fltr in self.filterstack:
            self.TX *= splev(self.freq, self.filters[fltr]["spl"], ext=1)

    def set_TXspl(self):
        """Defines a spline interpolation of the calculated transmisstion (TX)"""
        self.TXspl = splrep(self.freq, self.TX)

    def set_freq(self):
        self.freq = np.arange(*self.fminmax, self.df)

    def get_cfreq(self):
        """Calculates the centre frequency as the weighted average of transmission of all the filters"""
        return np.sum(self.freq * self.TX) / self.TX.sum()


ADR1_5THz = OptSys(
    filterstack=["W1256", "B746", "W933", "W1111", "B724", "W1256", "B746", "W1467"],
    fminmax=(5e5, 2e6),
    df=1e3,
)


def Planck(hw, kbT, pol=2):
    """Planck law for black body radiation.
    pol is the number of polarizations. The unit is µeV/µm^2 (or µeV/µs/µm^2/MHz)"""
    return (
            pol
            * hw[:, np.newaxis]
            * (hw[:, np.newaxis] / hbar / (2 * np.pi) / const.c) ** 2
            * SCth.n(hw[:, np.newaxis], kbT)
        )


def Prad(kbT, optsys, method="sum"):
    """Calculates the radiation power falling on the detector
    with a black body at kbT and the optical system defined by optsys.
    Units are in µeV/µs"""
    if method == "int":

        def integrand(hw, kbT, optsys):
            return splev(hw / hbar / (2 * np.pi), optsys.TXspl, ext=1) * Planck(
                hw, kbT, optsys.pol
            )

        return (
            optsys.area
            * optsys.solidangle
            * quad(
                integrand,
                optsys.fminmax[0] * 2 * np.pi * hbar,
                optsys.fminmax[1] * 2 * np.pi * hbar,
                args=(kbT, optsys),
            )[0]
        )
    elif method == "sum":
        return (
                np.sum(
                    optsys.TX[:, np.newaxis]
                    * kd.THz.Planck(optsys.freq * 2 * np.pi * hbar, kbT)
                    * optsys.df
                    * 2
                    * np.pi
                    * kd.THz.hbar,
                axis=0)
                * optsys.solidangle
                * optsys.area
            )
    
    