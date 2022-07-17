import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import root
from scipy.signal import find_peaks
from scipy.linalg import eig
import copy

    
class S21circle(object):
    def __init__(self, S21data):
        self.S21 = S21data
        self.xc, self.yc, self.r0 = self.fit_circle()
        
    @property
    def xi(self):
        return self.S21.real

    @property
    def yi(self):
        return self.S21.imag
    
    @property
    def Pc(self):
        return self.xc + 1j*self.yc
    
    @property
    def phase(self, unwrap=True):
        if unwrap:
            return np.unwrap(np.angle(self.S21))
        else:
            return np.angle(self.S21)
        
    @property
    def dB(self):
        return 20*np.log10(np.abs(self.S21))
    
    
    def fit_circle(self, plot=False):
        '''
        Fits a circle to the complex S21data data using an algebraic fit technique. Returns the center point and the radius. 

        For an explanation of the method see the paper  "Efficient and robust analysis of complex scattering data under noise in
        resonators", Probst et al., 2014, doi.org/10.1063/1.4907935.
        Adjusted from a script by Bruno Buijtendorp
        '''

        x = self.S21.real 
        y = self.S21.imag 
        z = x**2 + y**2 

        M__ = lambda i, j : np.dot(i,j)
        M_ = lambda i : np.sum(i)

        M = np.array([[M__(z,z), M__(x,z), M__(y,z), M_(z)],
                      [M__(x,z), M__(x,x), M__(x,y), M_(x)],
                      [M__(y,z), M__(x,y), M__(y,y), M_(y)],
                      [M_(z)   , M_(x)   , M_(y)   , len(x)]])

        B = np.array([[0, 0, 0, -2],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [-2, 0, 0, 0]])

        eival, eivec = eig(M, B, right=True)
        A = np.real(eivec[:, np.where(eival > 0, eival, np.inf).argmin()])
        xc = -A[1]/(2*A[0])
        yc = -A[2]/(2*A[0])
        r0 = 1/(2*np.abs(A[0]))*np.sqrt(A[1]**2 + A[2]**2 - 4*A[0]*A[3]) #this sqrt is needed to compensate numerical errors
        if plot:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(x, y)
            self._plot_circle_and_origin(xc, yc, r0, ax)
        self.xc, self.yc, self.r0 = (xc, yc, r0)
        return xc, yc, r0
    
    def centre(self):
        self.S21 -= self.Pc
    
    def restore(self):
        self.S21 += self.Pc
        
    def get_circsqerr(self):
        return np.sum(self.r0**2 - ((self.xi - self.xc)**2 + (self.yi - self.yc)**2))
    
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(self.xi, self.yi)
        circle = plt.Circle((self.xc, self.yc), self.r0, fill=False, zorder=10)
        ax.add_patch(circle)
        ax.plot(self.xc, self.yc, 'kx')
        
    
class kidsweep(object):
    def __init__(self, freq, S21data):
        self.f = freq
        self.S21circ = S21circle(S21data)
    
    
    def truncate_data(self, fmin=None, fmax=None, plot=False):
        approxf0 = self.f[self.S21circ.dB.argmin()]
        if fmin is None:
            #select -1 dB point
            rlmask = self.f < approxf0
            S21abs = np.abs(self.S21circ.S21[rlmask])
            S21min = S21abs.min()
            S21max = S21abs.max()
            fmin = self.f[rlmask][
                np.abs(S21abs - (S21max - S21min)/2**(1/3)).argmin()]
        if fmax is None:
            #select -1 dB points
            rlmask = self.f > approxf0
            S21abs = np.abs(self.S21circ.S21[rlmask])
            S21min = S21abs.min()
            S21max = S21abs.max()
            fmax = self.f[rlmask][
                np.abs(S21abs - (S21max - S21min)/2**(1/3)).argmin()]
            
        mask = (self.f > fmin) & (self.f < fmax)
        if plot:
            plt.figure()
            plt.plot(self.f, self.S21circ.dB, label='data')
            plt.plot(self.f[mask], self.S21circ.dB[mask], label='trunc. data')
            plt.legend()
            
        self.S21circ.S21 = self.S21circ.S21[mask]
        self.f = self.f[mask]       


    def fit_delay(self):

        def minfunc(tau, freq, S21circ):
            _S21 = copy.deepcopy(S21circ)
            _S21.S21 *= np.exp(1j*2*np.pi*tau*freq)
            _S21.fit_circle()
            return _S21.get_circsqerr()

        self.tau = least_squares(minfunc, 1/self.f[0], args=(self.f, self.S21circ), 
                             bounds=(0, np.inf)).x[0]
        return self.tau

    
    def corr_delay(self, plot=False):
        self.fit_delay()
        
        self.S21wodelay = copy.deepcopy(self.S21circ)
        self.S21wodelay.S21 *= np.exp(1j*2*np.pi*self.tau*self.f)
        self.S21wodelay.fit_circle()
        
        if plot:
            fig, ax = plt.subplots(figsize=(5, 5))
            self.S21circ.plot(ax)
            self.S21wodelay.plot(ax)
            self.plot_origin(ax)

    def _theta_curve(self, f, theta0, Ql, fr):
        return np.unwrap(theta0 + 2*np.arctan(2*Ql*(1-f/fr)))


    def fit_phase(self, plot=False):
        self.S21wodelay.centre()
        
        fitres = curve_fit(self._theta_curve, self.f, self.S21wodelay.phase,
                           p0=(0,
                               self.f.mean()/(self.f.max() - self.f.min()), 
                               self.f.mean()),
                          bounds=([-np.inf, 0, self.f.min()],
                                  [np.inf, np.inf, self.f.max()]))

        self.theta0, self.Ql, self.fres = fitres[0]
        
        if plot:
            plt.figure()
            plt.plot(self.f, self.S21wodelay.phase, label='data')
            plt.plot(self.f, self._theta_curve(self.f, *fitres[0]), label='fit')
            plt.legend()
            
        self.S21wodelay.restore()
        return self.theta0, self.Ql, self.fres
    
    def calibrate(self, plot=False):
        self.corr_delay()
        self.fit_phase()

        self.Poffres = self.S21wodelay.Pc + self.S21wodelay.r0*np.exp(1j*(self.theta0 + np.pi))
        self.S21cal = copy.deepcopy(self.S21circ)
        self.S21cal.S21 *= np.exp(1j*2*np.pi*self.tau*self.f)/self.Poffres
        self.S21cal.fit_circle()

        if plot:
            fig, ax = plt.subplots(1, 2)
            self.S21circ.plot(ax[0])
            self.S21wodelay.plot(ax[0])
            self.S21cal.plot(ax[1])
            ax[0].plot(self.Poffres.real, self.Poffres.imag,'go')
            for axis in ax:
                self.plot_origin(axis)

    def fit_S21(self, plot=False):
        assert hasattr(self, 'S21cal'), 'first calibrate the data!'
        self.phi0 = -1 * np.arcsin(self.S21cal.yc/self.S21cal.r0)
        self.S21cal.centre()
        fitfunc = lambda f, Ql, fr: self._theta_curve(f, self.phi0-np.pi, Ql, fr)
        fitres = curve_fit(fitfunc, self.f, self.S21cal.phase,
                           p0=(self.Ql,  self.fres), 
                           bounds=([0, self.f.min()], [np.inf, self.f.max()]))
        self.S21cal.restore()
        self.Ql, self.fres = fitres[0]
        self.Qc_compl = self.Ql/(2*self.S21cal.r0*np.exp(1j*self.phi0))
        self.Qi = (1/self.Ql - np.real(1/self.Qc_compl))**(-1)
        self.Qc = np.abs(self.Qc_compl)

        if plot:
            fig, ax = plt.subplots(1, 3, figsize=(7, 3))
            resind = np.abs(self.f-self.fres).argmin()
            self.S21cal.plot(ax[2])
            ax[2].plot(self.S21cal.xi[resind], self.S21cal.yi[resind], 'ro')
            ax[2].plot(1, 0, 'go')
            fitS21 = S21res(self.f, self.fres, self.Ql,
                            self.Qc_compl.real, self.Qc_compl.imag)
            ax[2].plot(fitS21.real, fitS21.imag)
            self.plot_origin(ax[2])
            
            self.S21cal.centre()
            ax[1].plot(self.f, self.S21cal.phase)
            self.S21cal.restore()
            ax[1].plot(self.f, self._theta_curve(self.f, self.phi0 - np.pi,
                                                 self.Ql, self.fres))
            ax[1].plot(self.fres, 
                       self._theta_curve(self.f, self.phi0 - np.pi,
                                         self.Ql, self.fres)[resind], 'ro')
            ax[0].plot(self.f, self.S21cal.dB)
            ax[0].plot(self.f, 20*np.log10(np.abs(fitS21)))
            ax[0].plot(self.fres, self.S21cal.dB[resind], 'ro')
            fig.suptitle(f'$f_{{res}}$={self.fres:.3e}, $Q_l$={self.Ql:.0f}, $Q_c$={self.Qc:.0f}, $Q_i$={self.Qi:.0f}')
            fig.tight_layout()
            
    def plot_origin(self, ax):
        ax.axhline(0, color='k', linestyle='--')
        ax.axvline(0, color='k', linestyle='--')
            
    
        

def getKIDfreqs(freq, S21data, prom=2, 
                fminmax=(None, None), plot=False):
    #truncate data with fminmax
    if fminmax[0] is None:
        fmin = freq.min()
    else:
        fmin = fminmax[0]
    if fminmax[1] is None:
        fmax = freq.max()
    else:
        fmax = fminmax[1]
    fminmax = (fmin, fmax)
    S21data = S21data[(freq > fminmax[0]) & (freq < fminmax[1])]
    freq = freq[(freq > fminmax[0]) & (freq < fminmax[1])]
    
    S21datadB = 20 * np.log10(np.abs(S21data))
    loc, props = find_peaks(np.diff(S21datadB), prominence=prom)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        axs[0].plot(freq, S21datadB, ".")
        axs[0].set_ylabel("$|S_{21}|^2$ (dB)")
        axs[1].plot(freq[1:], np.diff(S21datadB), "-")
        axs[1].plot(freq[loc], np.ones(len(loc)) * prom, ".")
        axs[1].set_ylabel("Differences")
        for ax in axs:
            ax.set_xlabel("Frequency")
        fig.tight_layout()

    return freq[loc]


def getKIDdip(freq, S21data, approxf0=None, wnd=None, plot=False):
    if wnd is None:
        wnd = np.diff(freq).min() * 1e2
    if approxf0 is None:
        approxf0 = freq[np.abs(S21data).argmin()]

    initwnd = (freq > approxf0 - wnd / 2) & (freq < approxf0 + wnd / 2)
    f0guess = freq[initwnd][np.abs(S21data)[initwnd].argmin()]
    wndmask = (freq > f0guess - wnd / 2) & (freq < f0guess + wnd / 2)
    f, data = freq[wndmask], S21data[wndmask]

    if plot:
        plt.figure()
        plt.plot(freq, 20 * np.log10(np.abs(S21data)), ".")
        plt.plot(f, 20 * np.log10(np.abs(data)), ".")
    return f, data


def S21res(f, fr, Ql, Qc_real, Qc_imag=0):
    Qc = Qc_real + 1j * Qc_imag
    return 1 - (Ql / np.abs(Qc) * np.exp(-1j * np.angle(Qc))) / (
        1 + 2j * Ql * (f / fr - 1)
    )

def S21fit(freq, cmplS21data, approxf0=None, wnd=None, fitmismatch=True, plot=False):
    if (approxf0 is not None) or (wnd is not None):
        f, data = getKIDdip(freq, cmplS21data, approxf0, wnd)
    else:
        f, data = freq, cmplS21data

    f0guess = f[np.abs(data).argmin()]
    # aguess = (np.abs(data[0]) + np.abs(data[-1])) / 2
    #guess Ql as f0/df(FWHM)
    HM = (1 - np.abs(data).min()) / 2
    Fl = f[f<f0guess][np.abs(np.abs(data[f<f0guess]) - HM).argmin()]
    Fr = f[f>f0guess][np.abs(np.abs(data[f>f0guess]) - HM).argmin()]
    FWHM = Fr - Fl
    Qlguess = f0guess/FWHM
    Qc_realguess = Qlguess / (1 - np.abs(data[np.abs(data).argmin()]))
    phase = np.unwrap(np.angle(data))
    # alphaguess = phase[[0, -1]].mean()
    # tauguess = (phase[-1] - phase[0]) / (2 * np.pi * (f[0] - f[-1]))

    def absfit(f, fr, Ql, Qc_real, Qc_imag):
        return np.abs(S21res(f, fr, Ql, Qc_real, Qc_imag))

    fitres = curve_fit(
        absfit,
        f,
        np.abs(data),
        p0=(f0guess, Qlguess, Qc_realguess, 0),
        bounds=([0, 0, Qlguess*.9, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
    )
    
    f0, Ql, Qc_real, Qc_imag = fitres[0]
    Qc = Qc_real + 1j*Qc_imag
    
#     phfit = curve_fit(
#         lambda f, alpha, tau: 
#         np.unwrap(np.angle(S21res(
#             f, f0, Ql, Qc_real, Qc_imag, a, alpha, tau))), 
#         f, phase, p0=(alphaguess, tauguess))
    
#     alpha, tau = phfit[0]
    params = (f0, Ql, Qc_real, Qc_imag)
    
    Qi = (1/Ql - np.real(1/Qc))**(-1)

    if plot:
        numplots = 3 if (approxf0 is None and wnd is None) else 4
        fig, axs = plt.subplots(1, numplots, figsize=(3*numplots, 3))
        fitf = np.linspace(f.min(), f.max(), 1000)
        i=0
        if approxf0 is not None or wnd is not None:
            axs[i].plot(freq, 20 * np.log10(np.abs(cmplS21data)), ".")
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
