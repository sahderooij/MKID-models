import numpy as np
import warnings 
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

def del_ampNoise(freq,SPR,plot=False):
    '''Filter to delete amplifier noise from the spectrum. 
    The high frequency noise level is subtracted, if it is lower than mid-frequency range. 
    Otherwise the minimum of the mid-frequency range is subtracted. Only positive values are returned.'''
    #Delete -140 datapoints
    freq = freq[SPR!=-140]
    SPR = SPR[SPR!=-140]
    #Make it non-dB
    SPRn = 10**(SPR/10)
#     #Substract amplifier noise
#     startr1 = 7e3
#     stopr1 = 1e4
#     startr2 = 3e2
#     stopr2 = 1e4
#     if SPRn[np.logical_and(freq>startr1,freq<stopr1)].size > 0 and \
#         SPRn[np.logical_and(freq>startr2,freq<stopr2)].size > 0:
#         #sometimes amplifier noise is higher.. so check:
#         if SPRn[np.logical_and(freq>startr1,freq<stopr1)].mean() < \
#             SPRn[np.logical_and(freq>startr2,freq<stopr2)].mean():
#             SPRn -= SPRn[np.logical_and(freq>startr1,freq<stopr1)].max()
#         else:
#             SPRn -= SPRn[np.logical_and(freq>startr2,freq<stopr2)].min()
    SPRn -= SPRn[-5]
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

def del_1fnNoise(freq,SPR,plot=False):
    '''A 1/f^n spectrum is fitted and subtracted if n is higher than 0.35.'''
    #Delete -140 datapoints
    freq = freq[SPR!=-140]
    SPR = SPR[SPR!=-140]
    #Make it non-dB
    SPRn = 10**(SPR/10)
    try:
        fit = curve_fit(lambda f,a,b: a*f**(-b),
                        freq[~np.isnan(SPRn)][1:],SPRn[~np.isnan(SPRn)][1:],
                        p0=(SPRn[~np.isnan(SPRn)][1:4].mean(),1))
    except:
        fit = np.array([[np.nan,np.nan]])
    
    if fit[0][1] > .35:
        SPRn -= fit[0][0]*freq**(-fit[0][1])
    
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
        plt.plot(freq[freq<1e2],10*np.log10(fit[0][0]*freq[freq<1e2]**(-fit[0][1])),label='fit')
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

