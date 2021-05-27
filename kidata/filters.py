import numpy as np
import warnings 
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

def del_ampNoise(freq,SPR,plot=False):
    '''Filter to delete amplifier noise from the spectrum. 
    The high frequency noise level is subtracted, if it is lower than mid-frequency range. 
    Otherwise the minimum of the mid-frequency range is subtracted. Only positive values are returned.'''

    #Make it non-dB
    SPRn = 10**(SPR/10)
    #Substract amplifier noise
    startr1 = freq.max()/2
    stopr1 = freq.max()
    if SPRn[(freq>startr1) & (freq<stopr1) & (~np.isnan(SPRn))].size > 0:
        SPRn -= SPRn[(freq>startr1) & (freq<stopr1) & (~np.isnan(SPRn))].max()
            
    #filter positive 
    freqn = freq[SPRn>0]
    SPRn = SPRn[SPRn>0]
    #return to dB
    SPRn = 10*np.log10(SPRn)
    if plot:
        plt.figure()
        plt.plot(freq,SPR)
        plt.plot(startr1,SPR[~np.isnan(SPR)].max(),'ro')
        plt.plot(stopr1,SPR[~np.isnan(SPR)].max(),'ro')

        plt.plot(freqn,SPRn)
        plt.xscale('log')
        plt.legend(['Input','Amp. noise filtered'])
        plt.show();plt.close()
    return freqn,SPRn
            
def del_1fNoise(freq,SPR,plot=False):
    '''1/f noise is subtracted. The height is determined as the average of the first 4 points.'''
    #Delete -140 datapoints
    freq = freq[SPR!=-140]
    SPR = SPR[SPR!=-140]
    #Make it non-dB
    SPRn = 10**(SPR/10)
    
    SPRn -= freq**-.5*np.mean(SPRn[1:4])
    #filter positive 
    freqn = freq[SPRn>0]
    SPRn = SPRn[SPRn>0]
    #return to dB
    SPRn = 10*np.log10(SPRn)
    if plot:
        plt.figure()
        plt.plot(freq,SPR)
        plt.plot(freqn,SPRn)
        plt.xscale('log')
        plt.legend(['Input','1/f filtered'])
        plt.show();plt.close()
    return freqn,SPRn

def del_1fnNoise(freq,SPR,minn=.35,plot=False):
    '''A 1/f^n spectrum is fitted and subtracted if n is higher than minn (default: 0.35).'''
    #Make it non-dB
    SPRn = 10**(SPR/10)
    
    try:
        fit = curve_fit(lambda f,a,b: a*f**(-b),
                        freq[~np.isnan(SPRn)][1:],SPRn[~np.isnan(SPRn)][1:],
                        p0=(SPRn[~np.isnan(SPRn)][1:4].mean(),1))
        if fit[0][1] > minn:
            SPRn -= fit[0][0]*freq**(-fit[0][1])
    except:
        #do nothing
        freqn,SPRn = (freq,SPR)    
    
    #filter positive 
    freqn = freq[SPRn>0]
    SPRn = SPRn[SPRn>0]
    #return to dB
    SPRn = 10*np.log10(SPRn)
    
    if plot:
        plt.figure()
        plt.plot(freq,SPR,label='Input')
        plt.plot(freqn,SPRn,label='Filtered')
        plt.xscale('log')
        plt.plot(freq,10*np.log10(fit[0][0]*freq**(-fit[0][1])),label='fit')
        plt.legend()
        plt.show();plt.close()
    return freqn,SPRn

def del_otherNoise(freq,SPR,plot=False,del1fn=False):
    if del1fn:
        return del_1fnNoise(*del_ampNoise(freq,SPR,plot=plot),plot=plot)
    else:
        return del_1fNoise(*del_ampNoise(freq,SPR,plot=plot),plot=plot)

def subtr_spec(freq,SPR,mfreq,mSPR,plot=False):
    '''Subtract mSPR from SPR. Used to subtract off-resonance spectrum, 
    such as in de Visser et al. 2011, PRL. The spectra are first scaled with
    the highest frequency point, to account for the dip-depth difference when
    off resonance.'''
    #TODO: use splines to avoid freq, mfreq to be the same
    assert all(mfreq == freq), "Off-resonance at different frequencies"
    mask = np.logical_and(mSPR != -140, SPR != -140)
    lSPR = 10**(SPR[mask]/10)
    lmSPR = 10**(mSPR[mask]/10)
    
    #Scale mSPR:
    lmSPR *= lSPR[-1]/lmSPR[-1]
    #Subtract:
    sSPR = lSPR - lmSPR
    sSPR[sSPR<=0] = np.nan
    sSPR[sSPR>0] = 10*np.log10(sSPR[sSPR>0])
    
    if plot:
        plt.figure()
        plt.plot(freq[mask],SPR,label='Original')
        plt.plot(mfreq[mask],10*np.log10(lmSPR),label='Scaled correction spectrum')
        plt.plot(freq[mask],sSPR,label='Corrected Spectrum')
        plt.xscale('log');plt.xlabel('Freq. (Hz)');plt.ylabel('PSD (dBc/Hz)')
        plt.legend();plt.show();plt.close()
    return freq[mask],sSPR    

