import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.stats import norm
from scipy.stats import gaussian_kde

import pandas as pd
from IPython.display import clear_output
from ipywidgets import interact
import glob
import os
from tqdm.notebook import tnrange, tqdm
import warnings
from scipy.signal import savgol_filter
from multiprocess import Pool

from kidata.IQ import to_ampphase, to_RX
from kidata import io


def calc_pulseavg_par(
    filelocation, KIDPrT=None, save_location=None,
    pulse_len=2000, start=500, nrsgm=10, mawnd=10, minmax_proms=None, 
    coord='ampphase', numpulses=100
):
    if KIDPrT is None:
        KIDPrT = np.unique(
            io.get_avlfileids(filelocation)[:, (0, 1, 4)].astype(int), 
            axis=0)

    n_streams = np.array(
        [
            int(i.split("\\")[-1].split("_")[3][5:]) + 1
            for i in glob.iglob(
                filelocation +
                (f"/KID{KIDPrT[0, 0]}_{KIDPrT[0, 1]}dBm"
                 f"__TDvis*_TmK{KIDPrT[0, 2]}.bin"))
        ]
    ).max()

    if minmax_proms is None:
        minmax_proms = select_proms(
            filelocation, KIDPrT, pulse_len, numpulses, coord, nrsgm, mawnd, n_streams)
        
    with Pool() as p:
        for dummy in tqdm(p.imap(lambda file: 
                            calc_pulseavg(
                                filelocation, file, save_location,
                                pulse_len, start, minmax_proms, coord),
                            [KIDPrT[i:(i + 1), :] for i in range(len(KIDPrT[:, 0]))]),
                          desc='KIDPrT', total=len(KIDPrT[:, 0])):
            pass

def calc_pulseavg(
    filelocation, KIDPrT=None, save_location=None,
    pulse_len=2000, start=500, nrsgm=10, mawnd=10, minmax_proms=None, 
    coord='ampphase', sfreq=1e6, numpulses=100
):
    """Script that runs a temp sweep of a certain data set and saves it.
    filelocation: the folder that contains .bin files.
    KIDPrT: array that has KID, Pread and T in columns of the files that 
        need to be processed. Default is all .bin files in folder.
    pulse_len: (int) total amount of points that will be taken for each pulse, incl. prepulse noise
    start: desired amount of noise points before the pulse
    minmax_prom: prominence spread of the desired peaks 
        (run selected_proms to determine these, asks if not given) 
    coord: choose coordinates in which the pulse is analysed: either 'ampphase' or 'RX' 
        (the linear or non-linear, Smith, coordinates).
    prctl: percentile of data points below first guess thershold for peak selection.
    nstream: number of data files to be taken for thershold selection."""

    if KIDPrT is None:
        KIDPrT = np.unique(
            io.get_avlfileids(filelocation)[:, (0, 1, 4)].astype(int), 
            axis=0)

    n_streams = np.array(
        [
            int(i.split("\\")[-1].split("_")[3][5:]) + 1
            for i in glob.iglob(
                filelocation +
                (f"/KID{KIDPrT[0, 0]}_{KIDPrT[0, 1]}dBm"
                 f"__TDvis*_TmK{KIDPrT[0, 2]}.bin"))
        ]
    ).max()
    
    if save_location is None:
        parentfld = "\\".join(glob.glob(filelocation)[0].split("\\")[:-1])
        fldnmlist = parentfld.split('\\')[-1].split('_')
        wvl = np.array(fldnmlist)[[('nm' in i) for i in fldnmlist]]
        save_location = '\\'.join(parentfld.split('\\')[:-1]) + f'\\{wvl[0]}'
        if not os.path.exists(save_location):
            os.mkdir(save_location)

    if minmax_proms is None:
        minmax_proms = select_proms(
            filelocation, KIDPrT, pulse_len, numpulses, coord, nrsgm, mawnd, n_streams)

    KIDs = np.unique(KIDPrT[:, 0])
    for i in tnrange(len(KIDs), desc='KID', leave=False):
        KID = KIDs[i]
        Preads = np.unique(KIDPrT[KIDPrT[:, 0] == KID, 1])
        for j in tnrange(len(Preads), desc='Pread', leave=False):
            Pread = Preads[j]
            temps = KIDPrT[(KIDPrT[:, 0] == KID) & (KIDPrT[:, 1] == Pread), 2]
            for k in tnrange(len(temps), desc='Temp', leave=False):
                # select current needed variables
                temp = temps[k]
                min_prom, max_prom = minmax_proms[KID][Pread][temp]

                # create empty arrays for the storage of data
                all_pulses_phase = []
                all_pulses_amp = []
                stream_num = []
                locations = np.empty(0)
                prominences = np.empty(0)

                # now the stream begins to loop through all your streams
                # and select the pulses which are going to be used

                for n in tnrange(n_streams, desc='stream', leave=False):
                    data = io.get_bin(filelocation, KID, Pread, temp, n)
                    data_info = (f"{filelocation}/KID{KID}_{Pread}dBm"
                                 f"__TDvis{n}_TmK{temp}_info.dat")
                    tres = calctres(data_info, sfreq)
                    
                    if coord == 'ampphase':
                        amp, phase = to_ampphase(data)
                    elif coord == 'RX':
                        infodata = io.read_dat(data_info)
                        Qs = infodata['Header'][3].split('S21min')[0].split(',')
                        Qc = float(Qs[1].split('=')[1])
                        Q = float(Qs[0].split('=')[1])
                        amp, phase = to_RX(data, Q/(2*Qc))
                        
                    min_dist = pulse_len

                    peaks_cur, locs_cur, proms_cur = find_sppeaks(
                        phase,
                        nrsgm,
                        mawnd,
                        pulse_len,
                        start,
                        min_dist,
                        tres,
                        min_prom,
                        max_prom)
                    if len(locs_cur) > 0:
                        cur_pulses_amp = get_amp(
                            amp, locs_cur, pulse_len, start, coord)
                        for m in range(len(locs_cur)):
                            all_pulses_phase.append(peaks_cur[m, :])
                            all_pulses_amp.append(cur_pulses_amp[m, :])
                            stream_num.append(n)
                        locations = np.concatenate(
                            (locations, (np.array(locs_cur))))
                        prominences = np.concatenate(
                            (prominences, np.array(proms_cur)))
                    else:
                        warnings.warn(
                            f'No pulses found in KID{KID}, {temp} mK, {Pread} dBm, stream {n}'
                        )

                # now that we have all the pulses we generate an average pulse
                # and save it, alongside some data of which pulses were used

                if len(locations) > 0:
                    avg_pulse_phase = np.average(all_pulses_phase, 0)
                    std_phase = np.std(all_pulses_phase, 0)
                    avg_pulse_amp = np.average(all_pulses_amp, 0)
                    std_amp = np.std(all_pulses_amp, 0)
                else:
                    warnings.warn(
                        (f"KID{KID}, {temp} mK, {Pread} dBm: No pulses made it "
                         "through filtering, please check filters"))
                    avg_pulse_phase = [0]
                    std_phase = [0]
                    avg_pulse_amp = [0]
                    std_amp = [0]
                    stream_num = [0]
                    locations = [0]
                    prominences = [0]

                # save the information of the pulses
                d = {
                    "Stream": np.array(stream_num),
                    "Location": locations,
                    "Prominence": prominences,
                }
                df = pd.DataFrame(data=d)
                df.to_csv(
                    save_location
                    + f"\\KID{KID}_{Pread}dBm__TmK{temp}_avgpulse_{coord}_info.csv",
                    index=False,
                )

                # save the avg pulse and the STD
                if coord == 'ampphase':
                    d = {
                        "Amp": avg_pulse_amp,
                        "Amp_std": std_amp,
                        "Phase": avg_pulse_phase,
                        "Phase_std": std_phase
                    }
                elif coord == 'RX':
                    d = {
                        "R": avg_pulse_amp,
                        "R_std": std_amp,
                        "X": avg_pulse_phase,
                        "X_std": std_phase
                    }

                df = pd.DataFrame(data=d)
                df.to_csv(
                    save_location +
                    f"\\KID{KID}_{Pread}dBm__TmK{temp}_avgpulse_{coord}.csv",
                    index=False,
                )


def select_proms(filelocation, KIDPrT, pulse_len, numpulses, coord, nrsgm, mawnd, n_streams):
    minmax_proms = {}
    for KID in np.unique(KIDPrT[:, 0]):
        minmax_proms[KID] = {}
        for Pread in np.unique(KIDPrT[KIDPrT[:, 0] == KID, 1]):
            minmax_proms[KID][Pread] = {}
            temps = KIDPrT[(KIDPrT[:, 0] == KID) & (KIDPrT[:, 1] == Pread), 2]
            for temp in temps:
                clear_output(wait=True)
                plt.clf()
                total_prominences = np.empty(0)
                j = 0
                totpulses = 0
                while totpulses < numpulses and j < n_streams:
                    data = io.get_bin(filelocation, KID, Pread, temp, j)
                    data_info = (f"{filelocation}/KID{KID}_{Pread}dBm"
                                 f"__TDvis{j}_TmK{temp}_info.dat")
                    j += 1
                    if coord == 'ampphase':
                        amp, phase = to_ampphase(data)
                    elif coord == 'RX':
                        infodata = io.read_dat(data_info)
                        Qs = infodata['Header'][3].split('S21min')[0].split(',')
                        Qc = float(Qs[1].split('=')[1])
                        Q = float(Qs[0].split('=')[1])
                        amp, phase = to_RX(data, Q/(2*Qc))
                    locs, prominences = find_allpeaklocs(
                        phase, nrsgm, mawnd, pulse_len)
                    if len(locs) >= 1:
                        totpulses += len(locs)
                        total_prominences = np.append(
                            total_prominences, prominences, 0)
                    else:
                        total_prominences = []
                        break
                    

                fig, axs = plt.subplots(1, 2, figsize=(8, 3))
                maphase = np.convolve(phase, np.ones(mawnd) / mawnd, mode="same")
                tmax = 100*pulse_len
                axs[0].plot(phase[:tmax])
                axs[0].plot(maphase[:tmax])
                axs[0].plot(locs[locs < tmax], 
                            maphase[locs[locs < tmax]], 'r.')
                axs[0].set_title(f"KID{KID}, {temp} mK, -{Pread} dBm")
                axs[0].set_xlabel('Time')
                axs[0].set_ylabel('Phase')
                n, bins, patches = plt.hist(total_prominences, bins=30, density=True)
                if len(total_prominences) > 0:
                    kde = gaussian_kde(total_prominences)
                    pdf = kde.pdf(bins)
                    mu = bins[pdf.argmax()]
                    fwhm = peak_widths(pdf, [pdf.argmax()])[0][0] * np.diff(bins)[0]
                    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
                    axs[1].plot(bins, pdf, 'r--', 
                                label=f'Gaussian kde\n R={(mu/fwhm):.1f}')
                    lowbnd = mu - 2*sigma
                    upbnd = mu + 2*sigma
                    axs[1].fill_between(bins[(bins >= lowbnd) & (bins <= upbnd)], 0, 
                    pdf[(bins >= lowbnd) & (bins <= upbnd)], 
                    alpha=0.5, color='r')
                else:
                    lowbnd = np.nan
                    upbnd = np.nan
                axs[1].axvline(upbnd, color='r', linestyle='dashed')
                axs[1].axvline(lowbnd, color='r', linestyle='dashed')
                axs[1].set_title("Histogram of found peaks")
                axs[1].set_xlabel('Prominence')
                axs[1].legend()
                plt.show()
                minp = float(input(f"Lower bound (enter for {lowbnd:.3f})") or lowbnd)
                maxp = float(input(f"Upper bound (enter for {upbnd:.3f})") or upbnd)
                minmax_proms[KID][Pread][temp] = (minp, maxp)
    clear_output()
    return minmax_proms


def find_allpeaklocs(data, nrsgm, mawnd, pulse_len):
    # moving average with a windows of 
    madata = np.convolve(data, np.ones(mawnd) / mawnd, mode="same")
    # estimate the noise standard deviation by lower half
    noisestd_est = np.abs(np.diff(np.percentile(madata, [16, 50]))[0])
    # retrieve all the peaks in the timestream
    locs, peakprops = find_peaks(madata, prominence=nrsgm*noisestd_est)
    prominences = peakprops["prominences"]
    return locs, prominences
    
    
def find_sppeaks(
    data,
    nrsgm,
    mawnd,
    pulse_len,
    start,
    min_dist,
    tres,
    min_prom,
    max_prom,):
    '''Find the single photon peaks in the data'''

    locs, prominences = find_allpeaklocs(data, nrsgm, mawnd, pulse_len)

    if locs.size > 0:
        # filter on distance to other peaks and data edges
        locs, prominences = filter_distance(locs, prominences, min_dist, len(data))
    
        # filter on single photon peaks
        locs, prominences = filter_prominence(locs, prominences, min_prom, max_prom)

    # cut out the desired window (len: pulse_len, data points before the peakstart = start
    peaks, locs = cut_peaks(data, locs, pulse_len, start, mawnd, tres)

    # subtract the offset by the average of the start
    peaks = subtr_offset(peaks, start, tres)

    # return the results
    return peaks, locs, prominences


def get_amp(amp, locations, pulse_len, start, coord):
    all_peaks = np.empty((len(locations), pulse_len))
    for i in range(len(locations)):
        all_peaks[i] = amp[int(locations[i] - start):int(locations[i] - start + pulse_len)]
    return all_peaks


def filter_distance(locs, proms, min_dist, data_len):
    # makes sure the pulses are separated by a distance min_dist
    dist = np.diff(locs)
    distbefore = np.insert(dist, 0, locs[0])
    distafter = np.append(dist, data_len - locs[-1])
    mask = ((distbefore > min_dist)
            & (distafter > min_dist))
    return locs[mask], proms[mask]

def cut_peaks(data, locs, pulse_len, start, mawnd, tres):
    # defines the lenght of the pulse (pulse_len) and the amount of noise beforehand (start)
    # also makes sure the peaks are overlapped correctly by using a difference check with risetime
    all_peaks = np.zeros((len(locs), pulse_len))
    true_locations = np.zeros(len(locs))
    
    for i in range(len(locs)):
        startwnd = data[locs[i] - int(10*tres) - mawnd:locs[i] + int(10*tres)]
        difference = np.roll(startwnd, -int(np.ceil(tres))) - startwnd
        begin = np.argmax(difference[:-int(np.ceil(tres))])
        actual_location = locs[i] - int(10*tres) - mawnd + begin + int(np.ceil(tres))
        all_peaks[i, :] = data[(actual_location - start):(
            actual_location - start + pulse_len)]
        true_locations[i] = actual_location

    return all_peaks, true_locations


def subtr_offset(peaks, start, tres):
    for i in range(len(peaks[:, 0])):
        peaks[i, :] -= np.mean(peaks[i, 0:start - int(10*tres)])
    return peaks


def filter_prominence(locations, prominences, min_prom, max_prom):
    # selects only the desired peaks after the distance check has been done
    mask = (prominences > min_prom) & (prominences < max_prom)
    if sum(mask) > 0:
        prominences = prominences[mask]
        locations = locations[mask]
        return locations, prominences
    else:
        return [], []
        

def calctres(info_loc, sfreq):
    datContent = [i.strip().split() for i in open(info_loc).readlines()]
    Omega_0 = 2 * np.pi * float(datContent[2][4][1:]) * 1e9
    Q = float(datContent[3][0][2:-1])
    tres = 2 * Q / Omega_0 * sfreq # in units of 1 / sample rate
    return tres


def view_pulses(binloc, avgloc, KID, Pread, T, coord='ampphase', pulse_len=500, start=100,
                logscale=True, movavg=False, wnd=9, suboff=False, ymin=1e-3, sfreq=1e6):
    
    strms, locs, proms = np.genfromtxt(avgloc + f'/KID{KID}_{Pread}dBm__TmK{T}_avgpulse_{coord}_info.csv',
        delimiter=',', skip_header=1, unpack=True)
    
    plt.ion()

    def show_pulse(pulse):
        plt.clf()
        strm = strms[pulse]
        loc = locs[pulse]

        data_info = (f"{binloc}/KID{KID}_{Pread}dBm"
                     f"__TDvis{strm}_TmK{T}_info.dat")

        data = io.get_bin(binloc, KID, Pread, T, strm)
        if coord == 'ampphase':
            amp, phase = to_ampphase(data)
            phpulse = phase[int(loc-start):int(loc+pulse_len-start)]
            amppulse = 1-amp[int(loc-start):int(loc+pulse_len-start)]
        elif coord == 'RX':
            infodata = io.read_dat(data_info)
            Qs = infodata['Header'][3].split('S21min')[0].split(',')
            Qc = float(Qs[1].split('=')[1])
            Q = float(Qs[0].split('=')[1])
            amp, phase = to_RX(data, Q/(2*Qc))
            phpulse = X[int(loc-start):int(loc+pulse_len-start)]
            amppulse = R[int(loc-start):int(loc+pulse_len-start)]
        else:
            raise ValueError('coord must be ampphase or RX')
        t = np.arange(len(phpulse)) - start
        if suboff:
            phpulse -= np.mean(phpulse[0:start - 5*calctres(data_info, sfreq)])
            amppulse -= np.mean(amppulse[0:start - 5*calctres(data_info, sfreq)])
        if movavg:
            phpulse = savgol_filter(phpulse, wnd, 0)
            amppulse = savgol_filter(amppulse, wnd, 0)
        plt.plot(t, phpulse, label='phase' if coord == 'ampphase' else 'X')
        plt.plot(t, amppulse, label='1-amp' if coord == 'ampphase' else 'R')
        plt.legend()
        plt.ylim(ymin, None)
        if logscale:
            plt.yscale('log')
        plt.show()

    interact(show_pulse, pulse=np.arange(len(strms)))