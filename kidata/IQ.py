'''This module contains functions that handle IQ data-streams.'''

import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
import numpy as np

def to_ampphase(data):
    '''Converts the I and Q variables from the raw .dat file to (normalized) amplitude and phase.
    Takes: 
    data loaded by np.fromfile(path,dtype='>f8').reshape(-1,2)
    
    Returns:
    Amplitude, Phase: numpy arrays'''
    
    Amp = np.sqrt(data[:,0]**2+data[:,1]**2)
    Amp /= Amp.mean()
    Phase = (np.pi - \
             (np.arctan2(
        data[:,1],data[:,0]) % (2*np.pi)))
    return Amp,Phase

def subtr_offset(data,plot=False):
    '''Subtracts a quadratic offset from a time stream (data) to compensate temperature drifts.
    Optionally plots the quadratic fit.'''
    t = np.arange(len(data))
    p = np.polyfit(t[::int(len(data)/1e3)],
                   data[::int(len(data)/1e3)],2) #speed-up the fit by selecting points
    if plot:
        plt.figure()
        plt.plot(t,data)
        plt.plot(t,np.polyval(p,t))
        plt.show();plt.close()
    return data - np.polyval(p,t)

def smooth(data,tau,sfreq):
    '''Smooths a time stream with half of the lifetime (tau) via a moving average.
    Takes:
    data -- time stream data
    tau -- lifetime of pulses in seconds
    sfreq -- sample frequency of the data.'''
    wnd = int(tau/2*sfreq)
    if wnd % 2 != 1:
        wnd += 1
    return uniform_filter1d(data,wnd,mode='wrap')