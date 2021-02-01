import numpy as np
import warnings 
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar as minisc
from scipy.special import k0,i0

from kidcalc import D, beta, cinduct, hwread, hwres, kbTeff, nqp
import kidcalc

from kidata import io,filters
from kidata.plot import _selectPread

# NOTE: all units are in 'micro': µeV, µm, µs etc.

def ak(S21data, lbd0=0.092, N0=1.72e4, kbTD=37312.0,plot=False,reterr=False,method='df'):
    '''Calculates the kinetic induction fraction, based on Goa2008, PhD Thesis. 
    Arguments:
    S21data -- the content of the .csv from the S21-analysis. 
    lbd0 -- penetration depth at T=0 in µm (default: 0.092, Al)
    N0 -- Density of states in µeV^{-1}µm^{-3} (default: 1.72e4, Al)
    kbTD -- Debye Energy in µeV (default 37312.0, Al)
    plot -- boolean to plot the fit over temperature.
    reterr -- boolean to return fitting error.
    method -- either df or Qi, which is fitted linearly over temperature.
    
    Returns:
    ak 
    optionally: the error as well.'''
    
    # Extract relevant data
    hw = S21data[:, 5] * 2 * np.pi * 0.6582 * 1e-9 #µeV
    kbT = S21data[:, 1] * 86.17  #µeV

    # Set needed constants
    hw0 = hw[0]
    d = S21data[0, 25]
    kbTc = S21data[0,21] * 86.17
    D0 = 1.76 * kbTc
    
    # define y to fit:
    if method == 'df':
        y = (hw - hw0) / hw0
    elif method == 'Qi':
        y = 1/S21data[:,4] - 1/S21data[0,4]
    
    
    
    # Mask the double measured temperatures, and only fit from 250 mK
    mask1 = np.zeros(len(y), dtype="bool")
    mask1[np.unique(np.round(S21data[:, 1], decimals=2),
                    return_index=True)[1]] = True
    mask = np.logical_and(mask1, (kbT >= .25 * 86.17))
    
    if mask.sum() > 3:
        y = y[mask]
    else:
        warnings.warn('Not enough high temperature S21data, taking the last 10 points')
        y = y[mask1][-10:]


    
    # define x to fit:
    x = np.zeros(len(y))
    i = 0
    s0 = cinduct(hw0, D(kbT[0], N0, kbTc, kbTD), kbT[0])
    for kbTi in kbT[mask]:
        D_0 = D(kbTi, N0, kbTc, kbTD)
        s = cinduct(hw[i], D_0, kbTi)
        if method == 'df':
            x[i] = (s[1] - s0[1]) / s0[1] * beta(lbd0, d, D_0, D0, kbTi)/4
        elif method == 'Qi':
            x[i] = (s[0] - s0[0]) / s0[1] * beta(lbd0, d, D_0, D0, kbTi)/2
        i += 1
    
    #do the fit:
    fit = curve_fit(lambda t, ak: ak * t, x, y)
    if plot:
        plt.figure()
        plt.plot(x,y,'o')
        plt.plot(x,fit[0]*x)
        plt.legend(['Data','Fit'])
        if method == 'df':
            plt.ylabel(r'$\delta f/f_0$')
            plt.xlabel(r'$\beta \delta \sigma_2/4\sigma_2 $')
        elif method == 'Qi':
            plt.ylabel(r'$\delta(1/Q_i)$')
            plt.xlabel(r'$\beta \delta \sigma_1/2\sigma_2 $')

    if reterr:
        return fit[0][0],np.sqrt(fit[1][0])
    else:
        return fit[0][0]

def tau(freq, SPR, startf = None, stopf = None,plot=False,retfnl = False):
    '''Fits a Lorentzian to a PSD.
    Arguments:
    freq -- frequency array (in Hz)
    SPR -- power spectral denisty values (in dB)
    startf -- start frequency for the fit, default None: 3 decades lower than stopf
    stopf -- stop frequency for the fit, default the lowest value in the interval 3e2 to 3e4 Hz
    plot -- boolean to show the plot
    retfnl -- boolean to return the noise level as well.
    
    Returns:
    tau -- lifetime in µs
    tauerr -- error in tau
    optionally: noise level (non-dB) and  error in noise level.'''
    
    #Filter nan-values
    freq = freq[SPR!=-140]
    SPR = SPR[SPR!=-140]
    freq = freq[~np.isnan(SPR)]
    SPR = SPR[~np.isnan(SPR)]
    freq = freq[SPR!=-np.inf]
    SPR = SPR[SPR!=-np.inf]
    if stopf is None:
        bdwth = np.logical_and(freq>3e2,freq<3e4)
        try:
            stopf = freq[bdwth][np.real(SPR[bdwth]).argmin()]
        except ValueError:
            stopf = 25e4
    if startf is None:
        startf = max(10**(np.log10(stopf)-3),1e2)
    
    # fitting a Lorentzian
    fitmask = np.logical_and(freq >= startf, freq <= stopf)
    fitfreq = freq[fitmask]
    if len(fitfreq) < 10:
        warnings.warn('Too little points in window to do fit.')
        tau = np.nan
        tauerr = np.nan
        N = np.nan
        Nerr = np.nan
    else:
        fitPSD = 10**(np.real(SPR[fitmask]-SPR.max())/10) 
        #notice the normalization for robust fitting

        def Lorspec(f, t, N):
            SN = 4 * N * t / (1 + (2 * np.pi * f * t) ** 2)
            return SN
        
        try:
            fit = curve_fit(Lorspec, fitfreq, fitPSD,
                            bounds=([0, 0], [np.inf, np.inf]),
                            p0=(2e-4, 1e4))
            tau = fit[0][0]*1e6
            tauerr = np.sqrt(np.diag(fit[1]))[0]*1e6
            N = fit[0][1]*10**(np.real(SPR.max())/10)
            Nerr = np.sqrt(np.diag(fit[1]))[1]*10**(np.real(SPR.max())/10)
        except RuntimeError:
            tau,tauerr,N,Nerr = np.ones(4)*np.nan

    if plot:
        plt.figure()
        plt.plot(freq[SPR!=-140], np.real(SPR), 'o')
        if ~np.isnan(tau):
            plt.plot(fitfreq, 10*np.log10(Lorspec(fitfreq,tau*1e-6,N)))
        plt.xscale("log")
        plt.show(); plt.close()
        
    if retfnl:
        return tau,tauerr,4*N*tau*1e-6,np.sqrt((4e-6*N*tauerr)**2 + (4e-6*Nerr*tau)**2)
    else:
        return tau,tauerr
    
def tau_pulse(pulse,tfit=(10,1e3),reterr=False,plot=False):
    '''Calculates lifetime from a exponential fit to a pulse.
    Arguments:
    pulse -- pulse data at 1 MHz, with begin of the pulse at 500 (µs)
    tfit -- tuple to specify the fitting window, default is (10,1e3)
    reterr -- boolean to return error
    plot -- boolean to plot the fit
    
    Returns:
    tau -- in µs
    optionally the fitting error'''
    t = (np.arange(len(pulse)) - 500) 
    fitmask = np.logical_and(t > tfit[0], t < tfit[1])
    t2 = t[fitmask]
    peak2 = pulse[fitmask]
    fit = curve_fit(
        lambda x, a, b: b * np.exp(-x / a), t2, peak2, p0=(0.5e3, peak2[0])
    )
    
    if plot:
        plt.figure()
        plt.plot(t, pulse)
        plt.plot(t2, fit[0][1]*np.exp(-t2/fit[0][0]))
        plt.yscale('log')
        plt.show();plt.close()
    if reterr:
        return fit[0][0],np.sqrt(fit[1][0,0])
    else: 
        return fit[0][0]
    
def tesc(Chipnum,KIDnum,Pread='max',
              minTemp=200,maxTemp=400,taunonkaplan=2e2,taures=1e1,relerrthrs=.2,
              pltfit=False,pltkaplan=False,reterr=False,
    t0=.44,
    kb=86.17,
    tpb=.28e-3,
    N0=1.72e4,
    kbTD=37312.0,
    defaulttesc=0):
    '''Calculates the phonon escape time from the GR noise lifetimes and Kaplan.
    Uses data at Pread (default max), and temperatures between minTemp,maxTemp
    (default (300,400)). Only lifetimes between taunonkaplan and taures, and with
    a relative error threshold of relerrthrs are considered.
    The remaining lifetimes, tesc is calculated and averaged. The error (optional return) 
    is the variance of the remaining lifetimes. If this fails, defaulttesc is returned.
    TODO: replace this with a true fitting procedure.'''
    
    TDparam = io.get_grTDparam(Chipnum)
    Pread = _selectPread(Pread,io.get_grPread(TDparam,KIDnum))[0]

    kbTc = io.get_S21data(Chipnum,KIDnum,
                       io.get_S21Pread(Chipnum,KIDnum)[0])[0,21]*kb
    
    Temp = io.get_grTemp(TDparam,KIDnum,Pread)
    Temp = Temp[np.logical_and(Temp<maxTemp,Temp>minTemp)]
    tescar,tescarerr,tqpstar,tqpstarerr = np.zeros((4,len(Temp)))
    for i in range(len(Temp)):
        if pltfit:
            print('{} KID{} -{} dBm T={} mK'.format(
                Chipnum,KIDnum,Pread,Temp[i]))
        freq,SPR = io.get_grdata(TDparam,KIDnum,Pread,Temp[i])
        tqpstar[i],tqpstarerr[i] = tau(freq,SPR,plot=pltfit)
        
        if tqpstarerr[i]/tqpstar[i] > relerrthrs or \
        (tqpstar[i] > taunonkaplan or tqpstar[i] < taures):
            tescar[i] = np.nan
        else:
            tescar[i] = kidcalc.tesc(kb*Temp[i]*1e-3,tqpstar[i],
                             t0,tpb,N0,kbTc,kbTD)
            tescarerr[i] = np.abs(kidcalc.tesc(kb*Temp[i]*1e-3,tqpstarerr[i],
                             t0,tpb,N0,kbTc,kbTD)+tpb)
            
    if tescar[~np.isnan(tescar)].size > 0:
        tesc1 = np.mean(tescar[~np.isnan(tescar)])
        tescerr = np.sqrt(np.std(tescar[~np.isnan(tescar)])**2 + 
                          ((tescarerr[~np.isnan(tescar)]/(~np.isnan(tescar)).sum())**2).sum())
    else:
        tesc1 = np.nan

    if tesc1 < 0 or np.isnan(tesc1) or tesc1 > 1e-2:
        warnings.warn(
            'tesc ({}) is not valid and set to {} µs. {}, KID{}'.format(
                tesc1,defaulttesc,Chipnum,KIDnum))
        tesc1 = defaulttesc
    if pltkaplan:
        plt.figure()
        plt.errorbar(Temp,tqpstar,yerr=tqpstarerr,capsize=5.,fmt='o')
        mask = ~np.isnan(tescar)
        plt.errorbar(Temp[mask],tqpstar[mask],fmt='o')
        try:
            T = np.linspace(Temp[~np.isnan(tqpstar)].min(),
                       Temp[~np.isnan(tqpstar)].max(),100)
        except ValueError:
            T = np.linspace(minTemp,maxTemp,100)
        taukaplan = kidcalc.tau_kaplan(T*1e-3,tesc=tesc1,kbTc=kbTc)  
        plt.plot(T,taukaplan)
        plt.yscale('log')
        plt.ylim(None,1e4)
        plt.xlabel('T (mK)')
        plt.ylabel(r'$\tau_{qp}^*$ (µs)')
        plt.legend(['Kaplan','GR Noise Data', 'Selected Data'])
        plt.show();plt.close()
    if reterr:
        return tesc1,tescerr
    else:
        return tesc1

def get_tescdict(Chipnum,Pread='max'):
    '''Returns a dictionary with the escapetimes of all KIDs in a chip.'''
    tescdict = {}
    TDparam = io.get_grTDparam(Chipnum)
    KIDlist = io.get_grKIDs(TDparam)
    for KIDnum in KIDlist:
        tescdict[KIDnum] = tesc(Chipnum,KIDnum,Pread=Pread)
    return tescdict

def NqpfromQi(S21data,uselowtempapprox=True,lbd0=0.092,kb=86.17,N0=1.72e4,kbTD=37312.0):
    '''Calculates the number of quasiparticles from the measured temperature dependence of Qi.
    Returns temperatures in K, along with the calculated quasiparticle numbers. 
    If uselowtempapprox, the complex impedence is calculated directly with a low 
    temperature approximation, else it\'s calculated with the cinduct function in kidcalc 
    (slow).''' 
    ak_ = ak(S21data)
    hw = S21data[:, 5]*2*np.pi*6.582e-4*1e-6
    d = S21data[0, 25]
    kbTc = S21data[0,21] * kb
    kbT = S21data[:,1]*kb
    D0 = 1.76 * kbTc
    if uselowtempapprox:
        beta_ = beta(lbd0, d, D0, D0, kbT[0])
        def minfunc(kbT,s2s1,hw,D0):
            xi = hw/(2*kbT)
            return np.abs(np.pi/4*((np.exp(D0/kbT)-2*np.exp(-xi)*i0(xi))/\
                                   (np.sinh(xi)*k0(xi)))-s2s1)
        Nqp = np.zeros(len(kbT))
        for i in range(len(kbT)):
            s2s1 = S21data[i,4]*(ak_*beta_)/2
            res = minisc(minfunc,
                         args=(s2s1,hw[i],D0),
                         bounds = (0,kbTc),
                         method='bounded')
            kbTeff = res.x
            Nqp[i] = S21data[0,14]*nqp(kbTeff,D0,N0)
        return kbT/kb,Nqp
    else:
        def minfunc(kbT,s2s1,hw,N0,kbTc,kbTD):
            D_ = D(kbT,N0,kbTc,kbTD)
            s1,s2 = cinduct(hw, D_, kbT)
            return np.abs(s2s1-s2/s1)
        Nqp = np.zeros(len(kbT))
        for i in range(len(kbT)):
            D_0 = D(kbT[i],N0,kbTc,kbTD)
            beta_ = beta(lbd0, d, D_0, D0, kbT[i])
            s2s1 = S21data[i,4]*(ak_*beta_)/2
            res = minisc(minfunc,
                         args=(s2s1,hw[i],N0,kbTc,kbTD),
                         bounds = (0,kbTc),
                         method='bounded')
            kbTeff = res.x
            D_ = D(kbTeff,N0,kbTc,kbTD)
            Nqp[i] = S21data[0,14]*nqp(kbTeff,D_,N0)
        return kbT/kb,Nqp
                   
