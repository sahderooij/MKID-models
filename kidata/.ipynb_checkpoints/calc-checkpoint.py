import numpy as np
import warnings 
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar as minisc

from kidcalc import D, beta, cinduct, hwread, hwres, kbTeff, nqp,Vsc
from kidata import io,filters

def ak(S21data, lbd0=0.092, N0=1.72e4, kbTD=37312.0,plot=False,reterr=False,method='df'):
    # Extract relevant data
    hw = S21data[:, 5] * 2 * np.pi * 0.6582 * 1e-9
    kbT = S21data[:, 1] * 86.17  # µeV

    # Set needed constants
    hw0 = hw[0]
    d = S21data[0, 25]
    kbTc = S21data[0,21] * 86.17
    D0 = 1.76 * kbTc

    # For D calculation:
    def integrand1(E, D):
        return 1 / np.sqrt(E ** 2 - D ** 2)
    Vsc = 1 / (integrate.quad(integrand1, D0, kbTD, args=(D0,))[0] * N0)
    
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
        warnings.warn('Little high temperature S21data')
        y = y[mask1][-10:]
    

    x = np.zeros(len(y))
    i = 0
    s0 = cinduct(hw0, D(kbT[0], N0, Vsc, kbTD), kbT[0])
    for kbTi in kbT[mask]:
        D_0 = D(kbTi, N0, Vsc, kbTD)
        s = cinduct(hw[i], D_0, kbTi)
        if method == 'df':
            x[i] = (s[1] - s0[1]) / s0[1] * beta(lbd0, d, D_0, D0, kbTi)/4
        elif method == 'Qi':
            x[i] = (s[0] - s0[0]) / s0[1] * beta(lbd0, d, D_0, D0, kbTi)/2
        i += 1
    
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
        return fit[0],np.sqrt(fit[1])
    else:
        return fit[0]

def tau(freq, SPR, startf = None, stopf = None, plot=False,retfnl = False):
    #Filter non-values
    freq = freq[SPR!=-140]
    SPR = SPR[SPR!=-140]
    freq = freq[~np.isnan(SPR)]
    SPR = SPR[~np.isnan(SPR)]
    if stopf is None:
        bdwth = np.logical_and(freq>3e2,freq<1e4)
        try:
            stopf = freq[bdwth][np.real(SPR[bdwth]).argmin()+1] #+1 to make sure this point is included 
        except:
            stopf = 2e4
    if startf is None:
        startf = 1e1
    
    # fitting a Lorentzian
    fitmask = np.logical_and(freq > startf, freq < stopf)
    fitfreq = freq[fitmask]
    if len(fitfreq) < 10:
        warnings.warn('Too little points in window to do tau fit.')
        tau = np.nan
        tauerr = np.nan
        N = np.nan
        Nerr = np.nan
    else:
        fitPSD = 10**(np.real(SPR[fitmask]-SPR.max())/10) 
        #notice the normalization

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
        
    if retfnl:
        return tau,tauerr,4*N*tau*1e-6,np.sqrt((4e-6*N*tauerr)**2 + (4e-6*Nerr*tau)**2)
    else:
        return tau,tauerr
    
def tau_peak(peakdata_ph,tfit=None,reterr=False,plot=False):
    t = (np.arange(len(peakdata_ph)) - 500) 
    peak = peakdata_ph
    
    if tfit is None:
        fitmask = np.logical_and(t > 10, t < 1e3)
    else:
        fitmask = np.logical_and(t > tfit[0], t < tfit[1])
    t2 = t[fitmask]
    peak2 = peak[fitmask]
    fit = curve_fit(
        lambda x, a, b: b * np.exp(-x / a), t2, peak2, p0=(0.5e3, peak2[0])
    )
    
    if plot:
        plt.figure()
        plt.plot(t, peak)
        plt.plot(t2, fit[0][1]*np.exp(-t2/fit[0][0]))
        plt.yscale('log')
    if reterr:
        return fit[0][0],np.sqrt(fit[1][0,0])
    else: 
        return fit[0][0]

def tau_kaplan(T,tesc=.14e-3, 
               t0=.44,
               kb = 86.17,
               tpb = .28e-3,
               N0 = 1.72e4,
               kbTc = 1.2*86.17,
               kbTD = 37312.0,):
    D0 = 1.76 * kbTc
    def integrand1(E, D):
        return 1 / np.sqrt(E ** 2 - D ** 2)
    Vsc = 1 / (integrate.quad(integrand1, D0, kbTD, args=(D0,))[0] * N0)
    D_ = D(kb*T, N0, Vsc, kbTD)
    nqp_ = nqp(kb*T, D_, N0)
    taukaplan = t0*N0*kbTc**3/(4*nqp_*D_**2)*(1+tesc/tpb) 
    return taukaplan
    
def kbTbeff(S21data,tqpstar,
    t0=.44,
    kb=86.17,
    tpb=.28e-3,
    N0=1.72e4,
    kbTD=37312.0,
    tesc=0.14e-3,
    plot=False):
    
    kbTc = kb * S21data[0,21]
    D0 = 1.76 * kbTc  # µeV
    V = S21data[0, 14]

    def integrand1(E, D):
        return 1 / np.sqrt(E ** 2 - D ** 2)
    Vsc = 1 / (integrate.quad(integrand1, D0, kbTD, args=(D0,))[0] * N0)
    Nqp_0 = V * t0 * N0 * kbTc ** 3 / \
        (2 * D0 ** 2 * tqpstar) * 0.5 * (1 + tesc / tpb)
    
    return kbTeff(Nqp_0, N0, V, Vsc, kbTD)
    
def _tesc(
    kbT,
    tqpstar,
    t0=.44,
    tpb=.28e-3,
    N0=1.72e4,
    kbTc=1.2 * 86.17,
    kbTD=37312.0
):
    '''Calculates the phonon escape time, based on tqp* via Kaplan. Times are in ns.'''
    Vsc_ = Vsc(kbTc,N0,kbTD)
    
    D_ = D(kbT, N0, Vsc_, kbTD)
    nqp_ = nqp(kbT, D_, N0)
    return tpb*((4*tqpstar*nqp_*D_**2)/(t0*N0*kbTc**3)-1)

def tesc(Chipnum,KIDnum,Pread='max',
              minTemp=220,maxTemp=400,taunonkaplan=2e2,taures=1e1,relerrthrs=.2,
              pltfit=False,pltkaplan=False,reterr=False,
    t0=.44,
    kb=86.17,
    tpb=.28e-3,
    N0=1.72e4,
    kbTD=37312.0,
    defaulttesc=0):
    
    TDparam = io.get_grTDparam(Chipnum)
    
    if Pread == 'max':
        Pread = io.get_grPread(TDparam,KIDnum).min()
    elif Pread == 'min':
        Pread = io.get_grPread(TDparam,KIDnum).max()
    elif Pread == 'med':
        Preadarr = io.get_grPread(TDparam,KIDnum)
        Pread = Preadarr[np.abs(Preadarr.mean()-Preadarr).argmin()]
    elif type(Pread) == int:
        Pread = np.sort(io.get_grPread(TDparam,KIDnum))[Pread]
    else:
        raise ValueError('{} not a valid Pread value'.format(Pread))

    kbTc = io.get_S21data(Chipnum,KIDnum,
                       io.get_S21Pread(Chipnum,KIDnum)[0])[0,21]*kb
    
    Temp = io.get_grTemp(TDparam,KIDnum,Pread)
    tescar = np.zeros(len(Temp))
    tqpstar = np.zeros(len(Temp))
    tqpstarerr = np.zeros(len(Temp))
    for i in range(len(Temp)):
        freq,SPR = io.get_grdata(TDparam,KIDnum,Pread,Temp[i])
        freq,SPR = filters.del_otherNoise(freq,SPR)
        tqpstar[i],tqpstarerr[i] = tau(freq,SPR,stopf=1e5,plot=pltfit)
        if pltfit:
            plt.title('{} KID{} -{} dBm T={} mK'.format(
                Chipnum,KIDnum,Pread,Temp[i]))
            
        if tqpstarerr[i]/tqpstar[i] > relerrthrs or \
        (tqpstar[i] > taunonkaplan or tqpstar[i] < taures) or \
        (Temp[i] < minTemp or Temp[i] > maxTemp):
            tescar[i] = np.nan
        else:
            tescar[i] = _tesc(kb*Temp[i]*1e-3,tqpstar[i],
                             t0,tpb,N0,kbTc,kbTD)
    if tescar[~np.isnan(tescar)].size > 0:
        tesc1 = np.mean(tescar[~np.isnan(tescar)])
        tescerr = np.std(tescar[~np.isnan(tescar)])
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
        T,taukaplan = tau_kaplan(np.linspace(Temp[~np.isnan(tqpstar)].min(),
                   Temp[~np.isnan(tqpstar)].max(),100),tesc=tesc1,kbTc=kbTc)  
        plt.plot(T,taukaplan)
        plt.yscale('log')
        plt.ylim(None,1e4)
        plt.xlabel('T (mK)')
        plt.ylabel(r'$\tau_{qp}^*$ (µs)')
        plt.legend(['Kaplan','GR Noise Data', 'Selected Data'])
    if reterr:
        return tesc1,tescerr
    else:
        return tesc1

def get_tescdict(Chipnum,Pread='max'):
    tescdict = {}
    TDparam = io.get_grTDparam(Chipnum)
    KIDlist = io.get_grKIDs(TDparam)
    for KIDnum in KIDlist:
        tescdict[KIDnum] = tesc(Chipnum,KIDnum,Pread=Pread)
    return tescdict

def NqpfromQi(S21data,lbd0=0.092,kb=86.17,N0=1.72e4,kbTD=37312.0):
    ak_ = ak(S21data)
    hw = S21data[:, 5]*2*np.pi*6.582e-4*1e-6
    d = S21data[0, 25]
    kbTc = S21data[0,21] * kb
    kbT = S21data[:,1]*kb
    D0 = 1.76 * kbTc
    Vsc_ = Vsc(kbTc,N0,kbTD)
    def minfunc(kbT,s2s1,hw,N0,Vsc_,kbTD):
        D_ = D(kbT,N0,Vsc_,kbTD)
        s1,s2 = cinduct(hw, D_, kbT)
        return np.abs(s2s1-s2/s1)
    Nqp = np.zeros(len(kbT))
    for i in range(len(kbT)):
        D_0 = D(kbT[i],N0,Vsc_,kbTD)
        beta_ = beta(lbd0, d, D_0, D0, kbT[i])
        s2s1 = S21data[i,4]*(ak_*beta_)/2
        res = minisc(minfunc,
                     args=(s2s1,hw[i],N0,Vsc_,kbTD),
                     bounds = (0,kbTc),
                     method='bounded')
        kbTeff = res.x
        D_ = D(kbTeff,N0,Vsc_,kbTD)
        Nqp[i] = S21data[0,14]*nqp(kbTeff,D_,N0)
    return kbT/kb,Nqp
                   
def nqpfromtau(tau_,Chipnum,KIDnum,tescPread='max',
                     t0=.44,
                     tpb=.28e-3,
                    kb=86.17,
                    N0 = 1.72e4):
    tesc_ = tesc(Chipnum,KIDnum,Pread=tescPread)
    S21data = io.get_S21data(Chipnum,KIDnum,io.get_S21Pread(Chipnum,KIDnum)[0])
    V = S21data[0,14]
    kbTc = S21data[0,21]*kb
    D_ = 1.76*kbTc
    return t0*N0*kbTc**3/(2*D_**2*2*tau_/(1+tesc_/tpb))