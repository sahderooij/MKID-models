import matplotlib.pyplot as plt
from scipy.signal import csd,welch
from scipy.interpolate import splrep,splev
from scipy.io import savemat
import numpy as np
import glob
from tqdm.notebook import tnrange
import warnings
import os

from kidata import io,plot,calc,filters
from kidata.IQ import to_ampphase, subtr_offset, smooth

def do_TDanalysis(Chipnum,
        resultpath = None,
        matname = 'TDresults',
        ppd = 30,
        nrseg = 32,
        nrsgm = 6,
        freqs=['med','fast']):
    '''Main function of this module that executes the noise post-processing.
    It writes the number of rejected segments and output PSDs in the same format as 
    the algorithm by PdV (.mat-file).
    Takes:
    Chipnum -- Chip number to preform the analysis on
    resultpath -- output where the resulting .mat-file is written. Default is NoiseTDanalyse folder.
    matname -- name of the output .mat-file. Default is TDresults.
    ppd -- points per decade to downsample the PSDs (default 30).
    nrseg -- number of segments for the pulse rejection.
    nrsgm -- number of standard deviations to reject a segment in pulse rejection (default 6).
    freqs -- list of data types to be used. Default is ['med','fast'], which is both 50 kHz and  
             1 MHz data. These spectra will be stitched together at 20 kHz. The fast data is
             processed with nperseg equal to 0.1 ms of data, as only the high frequency data is used.
    
    Returns:
    Nothing, but writes the .mat-file in the resultpath under matname.'''
    
    if resultpath is None:
        resultpath = io.get_datafld()+f'{Chipnum}/NoiseTDanalyse/'
    
    #find all KIDs, Read powers and Temperatures
    assert glob.glob(io.get_datafld() + f'{Chipnum}/Noise_vs_T/TD_2D/*.bin'), 'Data not found'
    
    KIDPrT = np.array([[int(i.split('\\')[-1].split('_')[0][3:]),
                        int(i.split('\\')[-1].split('_')[1][:-3]),
                        int(i.split('\\')[-1].split('_')[4][3:-4])] 
              for i in glob.iglob(io.get_datafld() + \
                                  f'{Chipnum}/Noise_vs_T/TD_2D/*TD{freqs[0]}*.bin')])
    KIDs = np.unique(KIDPrT[:,0])
    
    #initialize:
    TDparam = np.empty((1,len(KIDs)),
                       dtype=[
                           ('kidnr','O'),('Pread','O'),
                           ('Temp','O'),('nrrejectedmed','O'),
                           ('nrrejectedfast','O'),
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

        TDparam['nrrejectedmed'][0,i],TDparam['nrrejectedfast'][0,i],\
        TDparam['fmtotal'][0,i],TDparam['SPRrealneg'][0,i],TDparam['SPRrealpos'][0,i],\
        TDparam['SPRimagneg'][0,i],TDparam['SPRimagpos'][0,i],TDparam['SPPtotal'][0,i],\
        TDparam['SRRtotal'][0,i] = np.full((9,len(Preads),np.unique(
            KIDPrT[KIDPrT[:,0]==KIDs[i],1],return_counts=True)[1].max()),np.nan,dtype='O')
        
        for j in tnrange(len(Preads),desc='Pread',leave=False):
            Temps = np.unique(KIDPrT[np.logical_and(KIDPrT[:,0]==KIDs[i],
                                                        KIDPrT[:,1]==Preads[j]),2])
            for k in tnrange(len(Temps),desc='Temp',leave=False):
                TDparam['Pread'][0,i][j,k] = Preads[j]
                TDparam['Temp'][0,i][j,k] = Temps[k]
                
                allPSDs = np.zeros((len(freqs),3),dtype=object) #dims: [sfreq,var (amp,phase,cross)]
                rejected = np.zeros((len(freqs),2),dtype=object) #dims: [sfreq, var (amp, phase)]
                for l,freq in enumerate(freqs): 
                    if freq == 'med':
                        sfreq = 50e3
                        nperseg = 'seglen' #optimized to have information in low frequencies
                    elif freq == 'fast':
                        sfreq = 1e6
                        nperseg = '1e-4s' #to optimize S/N in high frequencies
                    else:
                        raise ValueError(f'{freq} is a not supported data sampling frequency')
                        
                    noisedata = io.get_noisebin(Chipnum,KIDs[i],Preads[j],Temps[k],freq=freq)
                    amp,phase = to_ampphase(noisedata)
                    spamp,rejected[l,0] = rej_pulses(
                        amp,nrsgm=nrsgm,nrseg=nrseg,sfreq=sfreq,smoothdata=(freq=='fast'))
                    spphase,rejected[l,1] = rej_pulses(
                        phase,nrsgm=nrsgm,nrseg=nrseg,sfreq=sfreq,smoothdata=(freq=='fast'))                 
                    #amp:
                    allPSDs[l,0] = np.array(calc_avgPSD(spamp,rejected[l,0],
                                                        sfreq=sfreq,nperseg=nperseg)).T
                    #phase:
                    allPSDs[l,1] = np.array(calc_avgPSD(spphase,rejected[l,1],
                                                        sfreq=sfreq,nperseg=nperseg)).T
                    #cross:
                    allPSDs[l,2] = np.array(
                        calc_avgPSD(spphase,rejected[l,1],spamp,rejected[l,0],
                                    sfreq=sfreq,nperseg=nperseg)).T
                    
                    
                PSDs = np.zeros(3,dtype=object) #dims: [var (amp,phase,cross)]
                for varind in range(3):
                    if len(freqs) == 1:
                        PSDs[varind] = logsmooth(np.real(allPSDs[:1,varind][0][:,0]),
                                                 allPSDs[:1,varind][0][:,1],ppd)
                    elif len(freqs) == 2:#check if multiple spectra need to be stichted.
                        medind = (np.array(freqs)=='med')
                        fastind = (np.array(freqs)=='fast')
                        stichted = np.vstack((
                            allPSDs[medind,varind][0][allPSDs[medind,varind][0][:,0] < 2e4,:],
                            allPSDs[fastind,varind][0][allPSDs[fastind,varind][0][:,0] > 2e4,:]))
                        PSDs[varind] = logsmooth(np.real(stichted[:,0]),stichted[:,1],ppd)
                
                #write to TDparam:
                if all(np.logical_and(PSDs[0][0] == PSDs[1][0],PSDs[1][0]==PSDs[2][0])):
                    for freq in freqs:
                        TDparam['nrrejected'+freq][0,i][j,k] = \
                        np.logical_or(rejected[(np.array(freqs)==freq),0][0],
                                      rejected[(np.array(freqs)==freq),1][0]).sum()
                    TDparam['fmtotal'][0,i][j,k] = PSDs[0][0]
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',RuntimeWarning)
                        TDparam['SPRrealneg'][0,i][j,k] = \
                            10*np.log10(-1*np.clip(np.real(PSDs[2][1]),None,0))
                        TDparam['SPRrealpos'][0,i][j,k] = \
                            10*np.log10(np.clip(np.real(PSDs[2][1]),0,None))
                        TDparam['SPRimagneg'][0,i][j,k] = \
                            10*np.log10(-1*np.clip(np.imag(PSDs[2][1]),None,0))
                        TDparam['SPRimagpos'][0,i][j,k] = \
                            10*np.log10(np.clip(np.imag(PSDs[2][1]),0,None))
                    TDparam['SPPtotal'][0,i][j,k] = 10*np.log10(np.real(PSDs[1][1]))
                    TDparam['SRRtotal'][0,i][j,k] = 10*np.log10(np.real(PSDs[0][1]))
                else:
                    warnings.warn('different frequencies, writing nans')
                    TDparam['nrrejectedmed'][0,i][j,k],TDparam['nrrejectedfast'][0,i][j,k],\
                    TDparam['fmtotal'][0,i][j,k],\
                    TDparam['SPRrealneg'][0,i][j,k], TDparam['SPRrealpos'][0,i][j,k],\
                    TDparam['SPRimagneg'][0,i][j,k],TDparam['SPRimagpos'][0,i][j,k],\
                    TDparam['SPPtotal'][0,i][j,k], TDparam['SRRtotal'][0,i][j,k] = \
                        np.full([9,1],np.nan)
                    
    if os.path.exists(resultpath):
        savemat(resultpath + matname + '.mat',{'TDparam':TDparam})   
    else:
        warnings.warn(f'Result path \"{resultpath}\" does not exsist. Making it.')
        os.makedirs(resultpath)
        savemat(resultpath + matname + '.mat',{'TDparam':TDparam})   

def rej_pulses(data,nrsgm=6,nrseg=32,sfreq=50e3,
               smoothdata=False,plot=False):
    '''Pulse rejection algorithm. It devides the time stream into segments and rejects segments based on a threshold. Based on PdV's pulse rejection algorithm.
    
    Arguments:
    data -- the time stream to be analyzed
    nrsgm -- number of times the minimal standard deviation to use as threshold (default 6)
    nrseg -- number of segments to divide the time stream in (default 32)
    sfreq -- sample frequency (default 50 kHz), only needed when plot or smoothdata.
    smoothdata -- boolean to determine to smooth the timestream first. If True, the lifetime is estimated on a Lorenzian fit on the non-rejected full time stream and given to the smooth function. if the fit fails, smoothing is not executed. 
    plot -- boolean to plot time stream with rejection indication.
    
    Returns:
    The splitted input data and a list of booleans which segments are rejected.'''
    
    if smoothdata:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            freq,PSD = welch(data,sfreq,'hamming',nperseg=len(data))
            smf,smPSD = logsmooth(freq,PSD,ppd=32)
            smf,smPSD = filters.del_ampNoise(np.real(smf),np.real(10*np.log10(smPSD)))
            smf,smPSD = filters.del_1fnNoise(smf,smPSD)
            tau,tauerr = calc.tau(smf,smPSD,startf=1e3,stopf=1e5)
            if tauerr/tau >= .2 or np.isnan(tau):
                warnings.warn(f'Could not estimate lifetime from PSD, no smoothing is used.')
                smdata = data
            else: 
                tau*=1e-6
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
        plt.show();plt.close()
        
    return np.array(np.array_split(data,nrseg)),reject

def calc_avgPSD(dataarr,reject,dataarr1=None,reject1=None,sfreq=50e3,
               nperseg='seglen'):
    '''Calculates the average cross PSD of dataarr and dataarr1, 
    from the segments of which both reject and reject1 are False.
    Takes:
    dataarr -- numpy array of segments in first axis and time in the second.
    reject -- boolean list of the same length as the first axis of dataarr.
    dataarr1 -- see dataarr
    reject1 -- see reject
    sfreq -- sample frequency
    nperseg -- number of data points of each sub-segment, 
            passed to the scipy.signal.csd function. Default is 'seglen', which
            sets it to the length of each segment, i.e. no subsegments. For other values,
            use a string of the form: 1e-3s, which will set the number to match 1 ms of data.
    
    Returns:
    frequency -- nan if all parts are rejected
    average PSD -- nan if all parts are rejected.'''
    if dataarr1 is None or reject1 is None:
        dataarr1 = dataarr
        reject1 = reject
    rejects = np.logical_or(reject,reject1)
    
    if nperseg == 'seglen':
        nperseg = len(dataarr[0])
    elif 's' in nperseg:
        nperseg = float(nperseg.replace('s',''))*sfreq
    else:
        raise ValueError(f'{nperseg} is not a valid input')
    
    if rejects.sum() != rejects.size:
        # csd is used, which uses Welch's method. 
        # However, if we choose nperseg as the full length ('seglen'), 
        # so we're only calculating the periodogram with a Hamming window.
        f,psds = csd(dataarr[~rejects],dataarr1[~rejects],
                     window='hamming',nperseg=nperseg,
                     fs=sfreq,axis=1,scaling='density')
        return f,psds.mean(0)
    else:
        warnings.warn('All parts are rejected, returning nans')
        plt.figure()
        t = np.arange(len(dataarr.flatten()))/sfreq
        plt.plot(t,dataarr.flatten())
        plt.plot(t,dataarr1.flatten())
        plt.show();plt.close()
        return np.full([2,1],np.nan)

def logsmooth(freq,psd,ppd):
    '''Down-samples a spectrum to a certain points per decade (ppd).'''
    if ~np.isnan(freq).all() and ~np.isnan(psd).all():
        logfmax = np.ceil(np.log10(freq.max()))
        logfmin = np.floor(np.log10(freq[freq.argmin()+1]))

        freqdiv = np.logspace(logfmin,logfmax,int(ppd*(logfmax-logfmin))+1)
        opsd = np.empty(len(freqdiv)-1,dtype='c16')
        of = np.empty(len(freqdiv)-1)
        with warnings.catch_warnings():  # To suppress warnings about sections being empty
            warnings.simplefilter('ignore',RuntimeWarning)
            for i in range(len(freqdiv)-1):
                fmask = np.logical_and(freq>freqdiv[i],freq<freqdiv[i+1])
                of[i] = freq[fmask].mean()
                opsd[i] = psd[fmask].mean()
        return of[~np.isnan(of)],opsd[~np.isnan(of)]
    else:
        return np.full([2,1],np.nan)

