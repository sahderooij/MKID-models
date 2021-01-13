import matplotlib.pyplot as plt
from scipy.signal import savgol_filter,csd,welch
from scipy.interpolate import splrep,splev
from scipy.io import savemat
from scipy.ndimage.filters import uniform_filter1d
import numpy as np
import glob
from tqdm.notebook import tnrange
import warnings

from kidata import io,plot,calc

def to_ampphase(noisedata):
    Amp = np.sqrt(noisedata[:,0]**2+noisedata[:,1]**2)
    Amp /= Amp.mean()
    Phase = (np.pi - \
             (np.arctan2(
        noisedata[:,1],noisedata[:,0]) % (2*np.pi)))
    return Amp,Phase

def subtr_offset(data,plot=False):
    t = np.arange(len(data))
    p = np.polyfit(t[::int(len(data)/1e3)],
                   data[::int(len(data)/1e3)],2) #speed-up the fit by selecting points
    if plot:
        plt.figure()
        plt.plot(t,data)
        plt.plot(t,np.polyval(p,t))
        plt.show()
        plt.close()
    return data - np.polyval(p,t)

def smooth(data,tau,sfreq):
    wnd = int(tau/2*sfreq)
    if wnd % 2 != 1:
        wnd += 1
    return uniform_filter1d(data,wnd,mode='wrap')

def rej_pulses(data,nrsgm=6,nrseg=32,sfreq=50e3,
               smoothdata=False,plot=False):
    if smoothdata:
        #estimate lifetime via initial PSD:
        freq,PSD = welch(data,sfreq,'hamming',nperseg=.5*sfreq)
        tau,tauerr = calc.tau(freq,10*np.log10(PSD),plot=True)
        print(tauerr/tau)
        if tauerr/tau >= .1:
            tau = 1
            warnings.warn('Could not estimate lifetime from PSD, 1 µs is used')
        tau*=1e-6 #convert to seconds
        
        smdata = smooth(data,tau,sfreq)
    else:
        smdata = data
    
    corrdata = subtr_offset(smdata)
    spdata = np.array(np.array_split(corrdata,nrseg))
    thrshld = nrsgm*spdata.std(1).min()
    reject = np.abs(spdata).max(1) > thrshld
    
    if plot:
        fig,ax = plt.subplots()
        t0=0
        for ind in range(nrseg):
            ax.plot((t0+np.arange(len(spdata[ind])))/sfreq,spdata[ind],
                    color='b',alpha=.5 if reject[ind] else 1)
            t0 += len(spdata[ind])
        ax.plot(np.arange(len(data))/sfreq,thrshld*np.ones(len(data)),
                'r',label=f'{nrsgm}$\sigma_{{min}}$-threshold')
        ax.plot(np.arange(len(data))/sfreq,-1*thrshld*np.ones(len(data)),
                'r')
        ax.legend()
        ax.set_xlabel('Time (s)')
        plt.show()
        plt.close()
        
    return np.array(np.array_split(data,nrseg)),reject

def calc_avgPSD(dataarr,reject,dataarr1=None,reject1=None,sfreq=50e3):
    if dataarr1 is None and reject1 is None:
        dataarr1 = dataarr
        reject1 = reject
    rejects = np.logical_or(reject,reject1)
    if rejects.sum() != rejects.size:
        f,psds = csd(dataarr[~rejects],dataarr1[~rejects],
                     window='hamming',nperseg=len(dataarr[0]),
                     nfft=len(dataarr[0]),
                     fs=sfreq,axis=1,scaling='density')
        return f,psds.mean(0)
    else:
        warnings.warn('All parts are rejected')
        return np.full([2,1],np.nan)

def logsmooth(freq,psd,ppd):
    if ~np.isnan(freq).all() and ~np.isnan(psd).all():
        logfmax = np.ceil(np.log10(freq.max()))
        logfmin = np.floor(np.log10(freq[freq.argmin()+1]))

        freqdiv = np.logspace(logfmin,logfmax,int(ppd*(logfmax-logfmin))+1)
        opsd = np.empty(len(freqdiv)-1,dtype='c16')
        of = np.empty(len(freqdiv)-1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            for i in range(len(freqdiv)-1):
                fmask = np.logical_and(freq>freqdiv[i],freq<freqdiv[i+1])
                of[i] = freq[fmask].mean()
                opsd[i] = psd[fmask].mean()
        return of[~np.isnan(of)],opsd[~np.isnan(of)]
    else:
        return np.full([2,1],np.nan)

def do_TDanalysis(Chipnum,
        resultpath = None,
        matname = 'TDresults',
        ppd = 30,
        nrseg = 32,
        nrsgm = 6,
        sfreq=50e3):
    
    if resultpath is None:
        resultpath = io.get_datafld()+f'{Chipnum}/NoiseTDanalyse/'
        
    KIDPrT = np.array([[int(i.split('\\')[-1].split('_')[0][3:]),
                        int(i.split('\\')[-1].split('_')[1][:-3]),
                        int(i.split('\\')[-1].split('_')[4][3:-4])] 
              for i in glob.iglob(io.get_datafld() + \
                                  f'{Chipnum}/Noise_vs_T/TD_2D/*TDmed*.bin')])
    KIDs = np.unique(KIDPrT[:,0])
    TDparam = np.empty((1,len(KIDs)),
                       dtype=[
                           ('kidnr','O'),('Pread','O'),
                           ('Temp','O'),('nrrejectedmed','O'),
                           ('fmtotal','O'),('SPRrealneg','O'),
                           ('SPRrealpos','O'),('SPRimagneg','O'),
                           ('SPRimagpos','O'),('SPPtotal','O'),
                           ('SRRtotal','O')
                       ])

    for i in tnrange(len(KIDs),desc='KID',leave=False):
        TDparam['kidnr'][0,i] = np.array([[KIDs[i]]])

        Preads = np.unique(KIDPrT[KIDPrT[:,0]==KIDs[i],1])
        TDparam['Pread'][0,i],TDparam['Temp'][0,i] = np.zeros((2,len(Preads),np.unique(
            KIDPrT[KIDPrT[:,0]==KIDs[i],1],return_counts=True)[1].max()))

        TDparam['nrrejectedmed'][0,i],TDparam['fmtotal'][0,i],\
        TDparam['SPRrealneg'][0,i],TDparam['SPRrealpos'][0,i],\
        TDparam['SPRimagneg'][0,i],TDparam['SPRimagpos'][0,i],TDparam['SPPtotal'][0,i],\
        TDparam['SRRtotal'][0,i] = np.full((8,len(Preads),np.unique(
            KIDPrT[KIDPrT[:,0]==KIDs[i],1],return_counts=True)[1].max()),np.nan,dtype='O')

        for j in tnrange(len(Preads),desc='Pread',leave=False):
            Temps = np.unique(KIDPrT[np.logical_and(KIDPrT[:,0]==KIDs[i],
                                                        KIDPrT[:,1]==Preads[j]),2])
            for k in tnrange(len(Temps),desc='Temp',leave=False):
                TDparam['Pread'][0,i][j,k] = Preads[j]
                TDparam['Temp'][0,i][j,k] = Temps[k]

                noisedata = io.get_noisebin(Chipnum,KIDs[i],Preads[j],Temps[k])
                amp,phase = to_ampphase(noisedata)
                spamp,rejectamp = rej_pulses(
                    amp,nrsgm=nrsgm,nrseg=nrseg,sfreq=sfreq)
                spphase,rejectphase = rej_pulses(
                    phase,nrsgm=nrsgm,nrseg=nrseg,sfreq=sfreq)
                
                #amp:
                f,SRR = calc_avgPSD(spamp,rejectamp,sfreq=sfreq)
                lsfRR,lsSRR = logsmooth(f,SRR,ppd)
                #phase:
                f,SPP = calc_avgPSD(spphase,rejectphase,sfreq=sfreq)
                lsfPP,lsSPP = logsmooth(f,SPP,ppd)
                #cross:
                f,SPR = calc_avgPSD(spphase,rejectphase,spamp,rejectamp,sfreq=sfreq)
                lsfPR,lsSPR = logsmooth(f,SPR,ppd)
                
                #write to TDparam:
                if all(np.logical_and(lsfRR == lsfPP,lsfPP==lsfPR)):
                    TDparam['nrrejectedmed'][0,i][j,k] = \
                        np.logical_or(rejectamp,rejectphase).sum()
                    TDparam['fmtotal'][0,i][j,k] = lsfRR
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',RuntimeWarning)
                        TDparam['SPRrealneg'][0,i][j,k] = \
                            10*np.log10(-1*np.clip(np.real(lsSPR),None,0))
                        TDparam['SPRrealpos'][0,i][j,k] = \
                            10*np.log10(np.clip(np.real(lsSPR),0,None))
                        TDparam['SPRimagneg'][0,i][j,k] = \
                            10*np.log10(-1*np.clip(np.imag(lsSPR),None,0))
                        TDparam['SPRimagpos'][0,i][j,k] = \
                            10*np.log10(np.clip(np.imag(lsSPR),0,None))
                    TDparam['SPPtotal'][0,i][j,k] = 10*np.log10(np.real(lsSPP))
                    TDparam['SRRtotal'][0,i][j,k] = 10*np.log10(np.real(lsSRR))
                else:
                    warnings.warn('different frequencies, writing nans')
                    TDparam['nrrejectedmed'][0,i][j,k],TDparam['fmtotal'][0,i][j,k],\
                    TDparam['SPRrealneg'][0,i][j,k], TDparam['SPRrealpos'][0,i][j,k],\
                    TDparam['SPRimagneg'][0,i][j,k],TDparam['SPRimagpos'][0,i][j,k],\
                    TDparam['SPPtotal'][0,i][j,k], TDparam['SRRtotal'][0,i][j,k] = \
                        np.full([8,1],np.nan)
                    
    savemat(resultpath + matname + '.mat',{'TDparam':TDparam})   