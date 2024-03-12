import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.signal import find_peaks
from scipy.stats import norm

import pandas as pd
from IPython.display import clear_output
from ipywidgets import interact
import glob
import os
from tqdm.notebook import tnrange
import warnings
from scipy.signal import savgol_filter

from kidata.IQ import to_ampphase, to_RX
from kidata import io


def calc_pulseavg(
    filelocation, KIDPrT=None, save_location=None,
    pulse_len=2000, start=500, minmax_proms=None, 
    coord='ampphase', prctl=99.99, nstream=5
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
        fldnmlist = parentfld.split('\\')[-1].split(' ')
        wvl = np.array(fldnmlist)[[('nm' in i) for i in fldnmlist]]
        save_location = '\\'.join(parentfld.split('\\')[:-1]) + f'\\{wvl[0]}'
        if not os.path.exists(save_location):
            os.mkdir(save_location)
        

    if minmax_proms is None:
        minmax_proms = select_proms(filelocation, KIDPrT, pulse_len, min(n_streams, nstream), coord, prctl)

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
                    tres = calctres(data_info)
                    
                    if coord == 'ampphase':
                        amp, phase = to_ampphase(data)
                    elif coord == 'RX':
                        amp, phase = to_RX(data)
                    std = np.std(phase)
                    min_dist = 1.5 * pulse_len

                    peaks_cur, locs_cur, proms_cur, amount_cur = findpeaks(
                        phase,
                        3 * std,
                        pulse_len,
                        start,
                        min_dist,
                        tres,
                        min_prom,
                        max_prom)
                    if amount_cur > 0:
                        cur_pulses_amp = get_amp(
                            amp, locs_cur, pulse_len, start, coord)
                        for m in range(amount_cur):
                            all_pulses_phase.append(peaks_cur[m, :])
                            all_pulses_amp.append(cur_pulses_amp[m, :])
                            stream_num.append(n)
                        locations = np.concatenate(
                            (locations, (np.array(locs_cur))))
                        prominences = np.concatenate(
                            (prominences, np.array(proms_cur)))

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


def select_proms(filelocation, KIDPrT=None, pulse_len=2e3, nstream=5, coord='ampphase', prctl=99.99):
    # Shows you nstream timestreams and a hist of the peaks and
    # then asks you to select the wanted prominences

    if KIDPrT is None:
        KIDPrT = io.get_avlbins(filelocation)
    minmax_proms = {}
    maxp, minp = (0, 0)

    for KID in np.unique(KIDPrT[:, 0]):
        minmax_proms[KID] = {}
        for Pread in np.unique(KIDPrT[KIDPrT[:, 0] == KID, 1]):
            minmax_proms[KID][Pread] = {}
            temps = KIDPrT[(KIDPrT[:, 0] == KID) & (KIDPrT[:, 1] == Pread), 2]
            for temp in temps:
                clear_output(wait=True)
                plt.clf()
                total_prominences = np.empty(0)
                for j in range(nstream):
                    data = io.get_bin(filelocation, KID, Pread, temp, j)
                    if coord == 'ampphase':
                        amp, phase = to_ampphase(data)
                    elif coord == 'RX':
                        amp, phase = to_RX(data)
                    peaks, peakheight = find_peaks(
                        phase, prominence=np.percentile(phase, prctl))
                    prominences = peakheight["prominences"]
                    total_prominences = np.append(
                        total_prominences, prominences, 0)
                mu, sigma = norm.fit(total_prominences)

                plt.figure()
                plt.plot(phase)
                plt.plot(peaks, phase[peaks], 'r.')
                plt.title(f"KID{KID}, {temp} mK, {Pread} dBm")
                plt.figure()
                n, bins, patches = plt.hist(total_prominences, bins=30, density=True)
                y = norm.pdf(bins, mu, sigma)
                plt.plot(bins, y, 'r--', label='Gaussian fit')
                plt.title("Histogram of found peaks")
                plt.legend()
                plt.show()
                minp = float(input(f"Lower bound (enter for {(mu - 2*sigma):.3f})") or mu - 2*sigma)
                maxp = float(input(f"Upper bound (enter for {(mu + 2*sigma):.3f})") or mu + 2*sigma)
                minmax_proms[KID][Pread][temp] = (minp, maxp)
    clear_output()
    return minmax_proms


def findpeaks(
    data,
    prom,
    points,
    start,
    min_dist,
    tres,
    min_prom,
    max_prom,):
    data = np.array(data)

    # retrieve all the peaks in the timestream
    peaks, peakheight = find_peaks(data, prominence=prom)
    prominences = peakheight["prominences"]
    threshold = len(data) / (2 * points)

    # variable to ensure you don't include too many peaks so the dist checks removes everything
    while len(peaks) > threshold:
        prom += 0.05
        mask = prominences > prom
        peaks = peaks[mask]
        prominences = prominences[mask]

    # start tracking locations and amount of peaks
    locations = np.copy(peaks)

    # check distance
    peaks, locations, prominences, amount = checkdist(
        peaks, locations, prominences, min_dist
    )

    # cut out the desired window (len: points, data points before the peakstart = start
    peaks, locations, prominences, amount = cut_peaks(
        data, peaks, locations, prominences, amount, points, start, tres
    )

    # beforehand we also selected the little peaks, now we only want to look
    # at the ones that are actually caused by photons and not other random hits

    peaks, locations, prominences, amount = select_prominence(
        peaks, locations, prominences, min_prom, max_prom
    )
    # here we check if we missed any little hits that weren't caught before

    peaks, locations, prominences, amount = checkdoublepeak(
        peaks, locations, prominences, amount, tres
    )

    peaks, locations, prominences, amount = calc_offset(
        peaks, amount, start, locations, data, prominences, tres
    )
    # Now we calculate the offset at the time of the peak by taking the average of the points before the peak (start)

    # return the results
    return peaks, locations, prominences, amount


def get_amp(amp, locations, pulse_len, start, coord):
    all_peaks = np.empty((0, pulse_len))

    for i in range(len(locations)):

        all_peaks = np.append(
            all_peaks,
            np.array(
                [amp[(locations[i] - start): (locations[i] - start + pulse_len)]]
            ),
            0,
        )
        all_peaks[i, :] -= np.average(all_peaks[i, 0: (start - 50)])
        if coord == 'ampphase':
            all_peaks[i, :] += 1
    return all_peaks


def checkdist(peaks, locations, proms, min_dist):
    # makes sure the pulses are separated by a distance min_dist
    new_peaks = []
    new_prominences = []
    for i in range(0, len(peaks) - 1):
        if (
            np.abs(peaks[i] - peaks[i + 1]) > min_dist
            and np.abs(peaks[i] - peaks[i - 1]) > min_dist
        ):
            new_peaks.append(peaks[i])
            new_prominences.append(proms[i])
    locations = np.copy(new_peaks)
    return (new_peaks, locations, new_prominences, len(new_peaks))


def cut_peaks(data, peaks, locations, proms, amount, points, start, tres):
    # defines the lenght of the pulse (points) and the amount of noise beforehand (start)
    # also makes sure the peaks are overlapped correctly by using a difference check with risetime
    all_peaks = np.empty((0, points))
    true_locations = []
    deleted = []

    for i in range(len(peaks)):
        if (start * 2) < peaks[i] < (len(data) - (points) - 100):
            check = np.array(data[peaks[i] - 80: peaks[i] + 10])
            difference = diff(check, int(np.ceil(tres)))
            begin = np.argmax(difference)
            actual_location = peaks[i] - 80 + begin + int(np.ceil(tres))
            all_peaks = np.append(
                all_peaks,
                np.array(
                    [
                        data[
                            (actual_location - start): (
                                actual_location - start + points
                            )
                        ]
                    ]
                ),
                0,
            )
            true_locations.append(actual_location)
        else:
            deleted.append(i)

    proms = np.delete(proms, deleted)
    amount_peaks = len(proms)

    return all_peaks, true_locations, proms, amount_peaks


def calc_offset(peaks, amount, start, locations, data, prominences, tres):
    # create array to store offsets
    offsets = np.zeros(shape=amount)
    # calc offsets
    for i in range(amount):
        used_range = peaks[i, 0:start - int(5*tres)]
        offsets[i] = np.mean(used_range)
        peaks[i, :] -= offsets[i]
    amount = len(locations)
    return peaks, locations, prominences, amount


def select_prominence(peaks, locations, prominences, min_prom, max_prom):
    # selects only the desired peaks after the distance check has been done
    prominences = np.array(prominences)
    mask1 = min_prom < prominences
    mask2 = prominences < max_prom
    mask = np.logical_and(mask1, mask2)
    if sum(mask) > 0:
        peaks = peaks[mask, :]
        prominences = np.array(prominences)[mask]
        locations = np.array(locations)[mask]
        amount = len(locations)

        return peaks, locations, prominences, amount
    else:
        return [], [], [], 0


def diff(peak, num_dif):
    diff = peak[num_dif:] - peak[:-num_dif]
    return diff


def movingaverage(peak, window):
    avg = np.convolve(peak, np.ones(window) / window, mode="same")
    return avg


def checkdoublepeak(peaks, locations, prominences, amount, tres):
    deleted = []
    for i in range(amount):
        peak = peaks[i, :]
        avg = movingaverage(peak, 10)
        std = np.std(avg[0:400])
        mul = 0.1
        check, check1 = find_peaks(
            avg, prominence=mul * prominences[i], distance=30)
        cur_prom = check1["prominences"]
        left_bases = check1["left_bases"]
        while len(check) > 5:
            mul += 0.05

            mask = cur_prom > mul * prominences[i]
            cur_prom = cur_prom[mask]
            check = check[mask]
            left_bases = left_bases[mask]
        if len(check) > 1:
            deleted.append(i)

    peaks = np.delete(peaks, deleted, 0)
    proms = np.delete(prominences, deleted)
    locations = np.delete(locations, deleted)
    amount = len(locations)
    return peaks, locations, proms, amount


def calctres(info_loc):
    datContent = [i.strip().split() for i in open(info_loc).readlines()]
    Omega_0 = 2 * np.pi * float(datContent[2][4][1:]) * 1e9
    Q = float(datContent[3][0][2:-1])
    tres = 2 * Q / Omega_0 * 1e6
    return tres


def view_pulses(Chipnum, KID, Pread, T, wvl, coord='ampphase', pulse_len=500, start=100,
                logscale=True, movavg=False, wnd=9, suboff=False):
    strms, locs, proms = io.get_avgpulseinfo(Chipnum, KID, Pread, T, wvl, coord=coord)
    plt.ion()

    def show_pulse(pulse):
        plt.clf()
        strm = strms[pulse]
        loc = locs[pulse]

        filelocation = io.get_datafld() + f'{Chipnum}\\Pulse\\{wvl}nm\\TD_2D'

        data_info = (f"{filelocation}/KID{KID}_{Pread}dBm"
                     f"__TDvis{strm}_TmK{T}_info.dat")

        data = io.get_bin(filelocation,
                          KID, Pread, T, strm)
        if coord == 'ampphase':
            amp, phase = to_ampphase(data)
            phpulse = phase[int(loc-start):int(loc+pulse_len-start)]
            amppulse = 1-amp[int(loc-start):int(loc+pulse_len-start)]
        elif coord == 'RX':
            R, X = to_RX(data)
            phpulse = X[int(loc-start):int(loc+pulse_len-start)]
            amppulse = R[int(loc-start):int(loc+pulse_len-start)]
        else:
            raise ValueError('coord must be ampphase or RX')
        t = np.arange(len(phpulse)) - start
        if suboff:
            phpulse -= np.mean(phpulse[0:start - 5*calctres(data_info)])
            amppulse -= np.mean(amppulse[0:start - 5*calctres(data_info)])
        if movavg:
            phpulse = savgol_filter(phpulse, wnd, 0)
            amppulse = savgol_filter(amppulse, wnd, 0)
        plt.plot(t, phpulse, label='phase' if coord == 'ampphase' else 'X')
        plt.plot(t, amppulse, label='1-amp' if coord == 'ampphase' else 'R')
        plt.legend()
        if logscale:
            plt.yscale('log')
        plt.show()

    interact(show_pulse, pulse=np.arange(len(strms)))