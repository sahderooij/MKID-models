from scipy import constants as const
from scipy.integrate import quad
from scipy.interpolate import splrep, splev
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

import SCtheory as SCth
import kidata as kd


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
        self.solidangle = solidangle  # [str], default is hemisphere, 
        self.area = area  # [µm^2] 
        # can also be 'lambda^2', for single mode throughput, then use solidangle as coupling efficiency
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
            self.TX *= splev(self.freq, self.filters[fltr]["spl"], ext=3)

    def set_TXspl(self):
        """Defines a spline interpolation of the calculated transmisstion (TX)"""
        self.TXspl = splrep(self.freq, self.TX)

    def set_freq(self):
        self.freq = np.arange(*self.fminmax, self.df)

    def get_cfreq(self):
        """Calculates the centre frequency as the weighted average of transmission of all the filters"""
        return np.sum(self.freq * self.TX) / self.TX.sum()


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
            if optsys.area == 'lambda^2':
                area = (const.c / (hw[:, np.newaxis] / hbar / (2 * np.pi))) ** 2
            else:
                area = optsys.area
                
            return (splev(hw / hbar / (2 * np.pi), optsys.TXspl, ext=3) 
                    * Planck(hw, kbT, optsys.pol)
                    * optsys.solidangle
                    * area
                   )

        return quad(
                integrand,
                optsys.fminmax[0],
                optsys.fminmax[1],
                args=(kbT, optsys),
            )[0]

    elif method == "sum":
        if optsys.area == 'lambda^2': 
            # use lambda^2 troughput within the integration 
            area = (const.c / optsys.freq[:, np.newaxis]) ** 2
        else:
            area = optsys.area
            
        return np.sum(
            optsys.TX[:, np.newaxis]
            * kd.THz.Planck(optsys.freq * hbar * 2 * np.pi, kbT, 
                            pol=optsys.pol)
            * optsys.df
            * optsys.solidangle
            * area,
            axis=0)

def eta_opt(PabsNoiseLevel, kbT, optsys, SC, eta_pb=.57):
    '''Calculates the optical efficiency from the measured fluctuation level of the 
    absorbed power (in (eV/s)^2/Hz, not MHz), while eluminating with a black body at kbT through the optical system 
    defined by optsys. See eq. 5 of Baselmans et al. 2022'''
    if optsys.area == 'lambda^2':
        area = (const.c / optsys.freq[:, np.newaxis]) ** 2
    else:
        area = optsys.area

    hw = optsys.freq * 2 * np.pi * hbar 
    
    exp_shotnoise_gr = np.sum(
        optsys.TX[:, np.newaxis]
        * kd.THz.Planck(hw, kbT, optsys.pol)
        * optsys.solidangle
        * area
        * (2*hw[:, np.newaxis] + 4*SC.D0/eta_pb)
        * optsys.df,
        axis=0)
    
    exp_bunch = np.sum(
        optsys.TX[:, np.newaxis]**2
        * SCth.n(hw[:, np.newaxis], kbT)
        * kd.THz.Planck(hw, kbT, optsys.pol)
        * optsys.solidangle
        * area
        * 2*hw[:, np.newaxis] 
        * optsys.df,
        axis=0)
    return exp_shotnoise_gr / (PabsNoiseLevel * 1e6 - exp_bunch)


def calc_resps(fld, optsys, plot=False):
    '''This calculates the responsivity to radiation power (Prad) from the data in 'fld'.
    It puts the output files in fld/resps/<>.csv, with a seperate file for each KID, read power, 
    and chip temperature (with the same name as the data files).
    If plot is True, it plot the data along with the fits.'''
    
    resultpath = fld + '/resps'
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)
    
    KIDPrExs = kd.io.get_avlfiles(fld + '/2D_BB/*/', ftype='ReImT.dat')[:, :-2]
    for KIDPrEx in KIDPrExs:
        fname = '_'.join(KIDPrEx)
        files = glob.glob(fld + f'/2D_BB/*/{fname}*ReImT.dat')
        resps = np.zeros((len(files), 9))
        for f, file in enumerate(files):
            data = kd.io.read_dat(file)['Data']
            circledat = kd.io.read_dat(file.split('tint')[0] + '_S21dB.dat')
            circledata = list(circledat['Data'].values())[0]
            kidsw = kd.S21.kidsweep(circledata[:, 0], 10**(circledata[:, 1]/20) * np.exp(1j*circledata[:, 2]))
            kidsw.truncate_data() #only use -1 dB bandwidth

            TinK = list(data.keys())[0]
            TBBar = data[TinK][:, 0]
            Pradar = kd.THz.Prad(const.Boltzmann / const.e * 1e6 * TBBar, optsys)
            amp, phase = kd.IQ.to_ampphase(data[TinK][:, 1:3], kidsw.S21circ)
            # do lin. fit
            ampfit, ampfiterr = np.polyfit(Pradar, amp, 1, cov=True)
            phasefit, phasefiterr = np.polyfit(Pradar, phase, 1, cov=True)
            
            #save in numpy array
            resps[f, 0] = TinK
            resps[f, 1:5] = (ampfit[0], np.sqrt(ampfiterr[0, 0]), 
                             phasefit[0], np.sqrt(phasefiterr[0, 0]))
            #add fres and Qfactors from header (future: redo the fit with S21 module)
            resps[f, 5:] = (float(circledat['Header'][2].split('GHz :')[-1]),
                            float(circledat['Header'][3].split(',')[0].split('=')[1].split(' ')[0]),
                            float(circledat['Header'][3].split(',')[1].split('=')[1].split(' ')[0]),
                            float(circledat['Header'][3].split(',')[2].split('=')[1].split(' ')[0]))
                                  
                            
            if plot:       
                plt.figure()
                plt.title(fname + f' $T_{{BB}}$={TinK} K')
                plt.plot(Pradar, amp, color='r', alpha=.5)
                plt.plot(Pradar, np.polyval(ampfit, Pradar),
                        color='r', label=f'$dA/dP_{{rad}}={ampfit[0]:.1e}\pm{np.sqrt(ampfiterr[0, 0]):.1e}$ 1/(eV/s)')
                plt.ylabel('Amplitude', color='r')
                plt.legend(loc=(0,.9))
                plt.twinx()
                plt.plot(Pradar, phase, color='b', alpha=.5)
                plt.plot(Pradar, np.polyval(phasefit, Pradar),
                        color='b', label=f'$d\\theta/dP_{{rad}}={phasefit[0]:.1e}\pm{np.sqrt(phasefiterr[0, 0]):.1e}$ rad./(eV/s)')
                plt.ylabel('Phase (rad.)', color='b')
                plt.legend(loc=(0, 0))
                plt.show()
                plt.close()
        np.savetxt(resultpath + '/' + fname[:-4] + '.csv', resps[resps[:, 0].argsort(), :], delimiter=',', 
               header=('TBB (K), amp resp. (1/(eV/s)), amp resp. err. (1/(eV/s)),'
                       + 'phase resp. (rad/(eV/s)), phase resp. err. (rad/(eV/s)),'
                       + ' Fres (GHz), Q, Qc, Qi')
              )
    