"""This module contains functions that handle IQ- and (amplitude,phase)-data-streams."""

import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import numpy as np
from ipywidgets import interact
import glob


def norm_radius(data):
    '''Normalize the of the circle to 1. It uses the mean radius of all the data.'''
    return data / np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2).mean()

def to_ampphase(data, S21circle=None):
    """Converts the I and Q variables from the raw .dat file to (normalized)
    amplitude and phase. It assumes that I and Q are calibrated, i.e.
    the circle is rotated and centred around (0, 0).
    If S21circle is passed, it gets the radius of the circle from that data.
    Takes:
    data loaded by np.fromfile(path,dtype='>f8').reshape(-1,2)

    Returns:
    Amplitude, Phase: numpy arrays"""
    
    if S21circle is None:
        ndata = norm_radius(data)
        Amp = np.sqrt(ndata[:, 0] ** 2 + ndata[:, 1] ** 2)
        Phase = np.pi - (np.arctan2(ndata[:, 1], ndata[:, 0]) % (2 * np.pi))
    else:
        Amp = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2) / S21circle.r0
        Phase = np.pi - (np.arctan2(data[:, 1], data[:, 0]) % (2 * np.pi))
    return Amp, Phase



def to_RX(data):
    '''Converts IQ data to the non-linear (Smith) coördinates of Zobrist et al. 2021: 10.1117/1.JATIS.7.1.010501. 
    These coördinates are normalized to the resonance circle.'''
    ndata = norm_radius(data)
    g = ndata[:, 0] + 1j * ndata[:, 1]
    z = (1 + g) / (1 - g)
    return z.real, z.imag
    

def subtr_offset(data, plot=False):
    """Subtracts a 3rd order polynomial (arbritrary) offset from 
    a time stream (data) to compensate temperature drifts.
    Optionally plots the quadratic fit."""
    t = np.arange(len(data))
    p = np.polyfit(
        t[:: int(len(data) / 1e3)], data[:: int(len(data) / 1e3)], 2
    )  # speed-up the fit by selecting points
    if plot:
        plt.figure()
        plt.plot(t, data)
        plt.plot(t, np.polyval(p, t))
        plt.show()
        plt.close()
    return data - np.polyval(p, t)


def smooth(data, tau, sfreq):
    """Smooths a time stream with half of the lifetime (tau) via a
    moving average.
    Takes:
    data -- time stream data
    tau -- lifetime of pulses in seconds
    sfreq -- sample frequency of the data."""
    wnd = int(tau / 2 * sfreq)
    if wnd % 2 != 1:
        wnd += 1
    return uniform_filter1d(data, wnd, mode="wrap")


def view_bins(path, coordinates='ampphase', sel='',
             smoothdata=False, tau=100e-6, sfreq=1e6):
    """A functions that plots the amplitude and phase from IQ data streams that
    are found in the folder indicated by path. Needs matplotlib widget
    back-end."""
    fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.ion()

    def plotbin(file):
        axs[0].cla()
        axs[1].cla()
        data = np.fromfile(path + "\\" + file, dtype=">f8").reshape(-1, 2)
        if coordinates == 'ampphase':
            amp, phase = to_ampphase(data)
            axs[1].set_ylabel("Phase (rad.)")
            axs[0].set_ylabel("Amp. (normalized)")
        elif coordinates == 'RX':
            amp, phase = to_RX(data)
            axs[1].set_ylabel("Reactance (norm.)")
            axs[0].set_ylabel("Resistance (norm.)")
        elif coordinates == 'IQ':
            amp, phase = (data[:, 0], data[:, 1])
            axs[1].set_ylabel("Q (norm.)")
            axs[0].set_ylabel("I (norm.)")
        if smoothdata:
            amp = smooth(amp, tau, sfreq)
            phase = smooth(phase, tau, sfreq)
        axs[1].plot(phase)
        axs[0].plot(amp, color="orange")
        axs[1].set_xlabel("time point")

    interact(
        plotbin, file=np.sort([i.split("\\")[-1]
                              for i in glob.iglob(path + f"\\*{sel}*.bin")])
    )
