import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
from IPython.display import clear_output
import glob

from kidata.IQ import to_ampphase
from kidata import io

def calc_pulseavg(
    filelocation,
    pulse_len,
    start,
    minmax_proms,
    kid_property='membrane',
):
    """Please give filelocation in the form: 
    r'location\KID1_{}dBm__TDvis{}_TmK{}.bin' so it formats the dbm, stream and temp
    Please give save_location in the form: 
    r'Location\KID1_{}dBm_{}mK_avgpulse    or something of the like (first dbm then temp)
    Script that runs a temp sweep of a certain data set and saves it.
    n_streams: amount of raw data streams you have
    pulse_len: amount of points that will be taken from each pulse
    start: amount of 'noise' points before the pulse that will be taken into consideration
    min_dist: minimum distance that 2 pulses should have in order to not overlap
    min/max_prom: prominence spread of the desired peaks (Run selected_proms first to determine these)
    KID_property: changes some of the filtering statistics based on what kind of MKID we are using (substrate/membrane)"""

    KIDPrT = io.get_avlbins(filelocation)

    n_streams = np.array(
        [
            int(i.split("\\")[-1].split("_")[3][5:])
            for i in glob.iglob(filelocation + "/*vis*.bin")
        ]
    ).max()
    
    save_location = '\\'.join(glob.glob(filelocation)[0].split('\\')[:-1])

    for KID in np.unique(KIDPrT[:, 0]):
        for Pread in np.unique(KIDPrT[KIDPrT[:, 0] == KID, 1]):
            temps = KIDPrT[(KIDPrT[:, 0] == KID) & (KIDPrT[:, 1] == Pread), 2]
            for i in range(len(temps)):
                # select current needed variables
                temp = temps[i]
                min_prom, max_prom = minmax_proms[KID][Pread][temp]

                # create empty arrays for the storage of data
                all_pulses_phase = []
                all_pulses_amp = []
                stream_num = []
                locations = np.empty(0)
                prominences = np.empty(0)

                # now the stream begins to loop through all your streams and select the pulses which are going to be used
                for i in range(n_streams):
                    data = np.fromfile(
                        f'{filelocation}/KID{KID}_{Pread}dBm__TDvis{i}_TmK{temp}.bin'
                        , dtype=">f8"
                    ).reshape(-1, 2)
                    data_info = f'{filelocation}/KID{KID}_{Pread}dBm__TDvis{i}_TmK{temp}_info.dat'
                    tres = calctres(data_info)

                    amp, phase = to_ampphase(data)
                    std = np.std(phase)
                    min_dist = 1.5*pulse_len

                    peaks_cur, locs_cur, proms_cur, amount_cur = findpeaks(
                        phase,
                        3 * std,
                        pulse_len,
                        start,
                        min_dist,
                        tres,
                        min_prom,
                        max_prom,
                        kid_property,
                    )
                    if amount_cur > 0:
                        cur_pulses_amp = get_amp(amp, locs_cur, pulse_len, start)
                        for j in range(amount_cur):
                            all_pulses_phase.append(peaks_cur[j, :])
                            all_pulses_amp.append(cur_pulses_amp[j, :])
                            stream_num.append(i)
                        locations = np.concatenate((locations, (np.array(locs_cur))))
                        prominences = np.concatenate(
                            (prominences, np.array(proms_cur)))

                # now that we have all the pulses we generate an average pulse and save it, alongside some data of which pulses were used
                if len(locations) > 0:
                    avg_pulse_phase = np.average(all_pulses_phase, 0)
                    std_phase = np.std(all_pulses_phase, 0)
                    avg_pulse_amp = np.average(all_pulses_amp, 0)
                    std_amp = np.std(all_pulses_amp, 0)
                    print(temp, "mK, ran succesful")
                else:
                    print(
                        temp,
                        "{} mK: No pulses made it through filtering, please check filters".format(
                            temp
                        ),
                    )
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
                    save_location + f'\\KID{KID}_{Pread}dBm__mK{temp}_avgpulse_info.csv'
                    , index=False)

                # save the avg puls and the STD
                d = {
                    "Phase": avg_pulse_phase,
                    "Phase_std": std_phase,
                    "Amp": avg_pulse_amp,
                    "Amp_std": std_amp,
                }

                df = pd.DataFrame(data=d)
                df.to_csv(
                    save_location + f'\\KID{KID}_{Pread}dBm__mK{temp}_avgpulse.csv'
                    , index=False)
    return "script ran succesful"


def select_proms(filelocation, nstream=5, maxpulses=100):
    # Shows you 1 timestream and a hist of the peaks and then asks you to select the wanted prominences
    KIDPrT = io.get_avlbins(filelocation)
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
                for j in range(nstream):
                    data = np.fromfile(
                        f'{filelocation}/KID{KID}_{Pread}dBm__TDvis{j}_TmK{temp}.bin'
                        , dtype=">f8"
                    ).reshape(-1, 2)
                    amp, phase = to_ampphase(data)
                    std = np.std(phase)
                    multiplier = 4
                    peaks, peakshight = find_peaks(phase, prominence=multiplier * std)
                    prominences = peakshight["prominences"]
                    while len(peaks) < 5:
                        multiplier -= 1
                        peaks, peakshight = find_peaks(
                            phase, prominence=multiplier * std)

                    while len(peaks) > maxpulses:
                        multiplier += 0.2
                        mask = (prominences) > (multiplier * std)
                        peaks = peaks[mask]
                        prominences = prominences[mask]
                    total_prominences = np.append(total_prominences, prominences, 0)

                plt.figure()
                plt.plot(phase)
                plt.title("Temp: {}, dbm: {}".format(temp, Pread))
                plt.figure()
                plt.hist(total_prominences, bins=30)
                plt.title("Histogram of found peaks")
                plt.show()
                minp = float(
                    input("enter wanted lower bound. Use . for decimals (e.g. 0.6)"))
                maxp = float(
                    input("enter wanted upper bound. Use . for decimals (e.g. 0.6)"))
                minmax_proms[KID][Pread][temp] = (minp, maxp)
    return minmax_proms


def findpeaks(
    data,
    prom,
    points,
    start,
    min_dist,
    tres,
    min_prom,
    max_prom,
    kid_property="substrate",
):
    data = np.array(data)

    # retrieve all the peaks in the timestream
    peaks, peakshight = find_peaks(data, prominence=prom)
    prominences = peakshight["prominences"]
    threshold = len(data) / (2 * points)

    # variable to ensure you don't include too many peaks so the dist checks removes everything
    # met masks proberen ipv opnieuw find_peaks
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

    # beforehand we also selected the little peaks, now we only want to look at the ones that are actually caused by photons
    # and not other random hits

    peaks, locations, prominences, amount = select_prominence(
        peaks, locations, prominences, min_prom, max_prom
    )
    # here we check if we missed any little hits that weren't caught before

    peaks, locations, prominences, amount = checkdoublepeak(
        peaks, locations, prominences, amount, tres
    )

    peaks, locations, prominences, amount = calc_offset(
        peaks, amount, start, locations, data, prominences, kid_property, tres
    )
    # Now we calculate the offset at the time of the peak by taking the average of the points before the peak (start)

    # return the results
    return peaks, locations, prominences, amount


def get_amp(amp, locations, pulse_len, start):
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


def calc_offset(peaks, amount, start, locations, data, prominences, kid_property, tres):
    # create array to store offsets
    offsets = np.zeros(shape=amount)
    deleted = []
    # calc offsets
    for i in range(amount):
        used_range = peaks[i, 0: start - 50]

        offsets[i] = np.mean(used_range)
        if kid_property == "membrane":
            mask = used_range < 0
            if sum(mask) < (0.05 * (start - 50)):
                deleted.append(i)
        peaks[i, :] -= offsets[i]
    proms = np.delete(prominences, deleted)
    locations = np.delete(locations, deleted)
    peaks = np.delete(peaks, deleted, 0)
    amount = len(locations)
    return peaks, locations, proms, amount


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