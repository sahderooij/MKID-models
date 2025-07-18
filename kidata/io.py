import numpy as np
import pandas as pd
import warnings
import scipy.io
import glob
import re


def get_datafld():
    '''This specifies where the data is located. It should have the following structure:
    - Chipnum: name of the measured chip
        - S21: The data from the S21-measurements
            - 2D: Temperature and Power variations
                - KID{KIDnum}_{Pread}dBm_.dat: raw frequency sweeps
                - KID{KIDnum}_{Pread}dBm_Tdep.csv: results for the MATLAB S21analsysis 
        - Noise_vs_T
            - TD_{subfld}: raw .bin files
                - KID{KIDnum}_{Pread}dBm__TD{fast/med/slow}_TmK{Temperature}.bin
            - FFT: other files from the measurement 
                - {subfld}
            subfld is standard '2D', but can be passed to the io functions.
        - NoiseTDanalyse: the results of the noise post-processing.
            - TDresults.mat: same structure as the MATLAB routine for PdV.
        - Pulse
            - wavelength folders (e.g. \\1545nm\\)
                - TD_2D with bin files of the pulse data streams
                - KID{KIDnum}_{Pread}dBm__TmK{Temperature}_avgpluse.csv
                - KID{KIDnum}_{Pread}dBm__TmK{Temperature}_avgpluse)info.csv, which
                are the results of the average pulse calculation from 
                kidata.pulse.calc_avgpulse().'''

    datafld = "D:/"
    assert glob.glob(datafld), 'data folder not found'
    return datafld


def read_dat(path):
    '''This functions reads raw .dat measurement files.
    Takes:
    path -- location of the .dat-file.

    Returns:
    datdata -- a dictionary with (keys:values): 
        - Header: list of lines within the header
        - Data: a dictionary with (keys:values):
            - Temp: measurement values at this temperature (three colums of which the first is GHz)'''
    datdata = {}
    with open(path, 'r') as file:
        datdata['Header'] = []
        nremptylines = 0
        linenr = 0
        nroflines = len(file.readlines())
        file.seek(0)

        while nremptylines < 2:
            line = file.readline().replace('\n', '')
            linenr += 1
            if line == '':
                nremptylines += 1
            else:
                datdata['Header'].append(line)

        datdata['Data'] = {}
        while linenr <= nroflines:
            line = file.readline().replace('\n', '')
            linenr += 1
            if "Temperature" in line:
                Temp = float(line.split(':')[-1])
                nextline = file.readline()
                linenr += 1
                while not any(char.isdigit() for char in nextline):
                    nextline = file.readline()
                    linenr += 1
                datdata['Data'][Temp] = np.fromstring(nextline, sep='\t')
            elif line != '':
                datdata['Data'][Temp] = np.vstack(
                    (datdata['Data'][Temp], np.fromstring(line, sep='\t')))

    return datdata

# General file finding
def get_avlfiles(fld, ftype='.bin'):
    filearray = np.array(
            [
                    '.'.join(i.replace("\\", '/').split("/")[-1].split('.')[:-1]).split("_")
                for i in glob.iglob(fld + f'/*{ftype}')
            ]
        )
    return np.unique(filearray, axis=0)

def get_avlfileids(fld, ftype='.bin'):
    files = get_avlfiles(fld, ftype=ftype)
    bins = np.unique(np.array([[float(str(re.sub('[^\d\.]', '', j) or 0)) 
                                for j in i]
                     for i in files]), axis=0)
    bindf = pd.DataFrame(bins)
    bindf.sort_values(list(bindf.columns))
    return bindf.values

# S21


def get_S21KIDs(Chipnum):
    '''Returns which KIDs are measured in the S21-measurement.'''
    datafld = get_datafld()
    S21fld = datafld + '\\'.join([Chipnum, 'S21', '2D'])
    return np.unique([
        int(i.split('\\')[-1].split('_')[0][3:])
        for i in glob.glob(S21fld + '\\KID*_Tdep.csv')])


def get_S21Pread(Chipnum, KIDnum):
    '''Returns which read powers are measured in the S21-measurement.'''
    datafld = get_datafld()
    S21fld = datafld + '\\'.join([Chipnum, 'S21', '2D'])
    return np.sort([
        int(i.split('\\')[-1].split('_')[1][:-3])
        for i in glob.glob(S21fld + '\\KID{}_*Tdep.csv'.format(KIDnum))])


def get_S21dat(Chipnum, KIDnum, Pread=None):
    '''Returns the contents of the raw .dat-file from the S21-measurement.
    If read power is not given, the highest read power is chosen.'''
    if Pread is None:
        Pread = get_S21Pread(Chipnum, KIDnum)[0]
    datafld = get_datafld()
    path = datafld + "{}/S21/2D/KID{}_{}dBm_.dat".format(
        Chipnum, KIDnum, Pread)
    return read_dat(path)


def get_S21data(Chipnum, KIDnum, Pread=None):
    '''Returns the content of the .csv-file from the S21analysis MATLAB routine in a numpy-array.
    If read power is not given, the highest read power is chosen.'''
    Preadar = get_S21Pread(Chipnum, KIDnum)
    if Pread is None:
        Pread = Preadar[0]
    elif Pread not in Preadar:
        Pread_ = Preadar[np.abs(Preadar - Pread).argmin()]
        warnings.warn(
            'No S21data at this Pread. ' +
            f'Closest value is selected: -{Pread_} dBm instead of -{Pread} dBm')
        Pread = Pread_

    datafld = get_datafld()
    S21file = datafld + "\\".join(
        [
            Chipnum,
            "S21",
            "2D",
            "_".join(["KID" + str(KIDnum), str(int(Pread)) + "dBm", "Tdep.csv"]),
        ]
    )
    S21data = np.genfromtxt(S21file, delimiter=",")[1:, :]
    return S21data

# Pulse

def get_bin(folder, KID, Pread, T, strm):
    if type(strm) != str:
        return np.fromfile(
            folder + f"\\KID{KID}_{Pread}dBm__TDvis{int(strm)}_TmK{T}.bin",
            dtype=">f8").reshape(-1, 2)
    else:
        return np.fromfile(
            folder + f"\\KID{KID}_{Pread}dBm__TD{strm}_TmK{T}.bin",
            dtype=">f8").reshape(-1, 2)


def get_pulseavg(Chipnum, KID, Pread, wvl, T, subfolder='', std=False, coord='ampphase'):
    '''Returns the amplitude and phase data (in that order)
    of the average pulse, calculated from kidata.pulse.calc_avgpulse.
    Note: the baseline is subtracted from amplitude.
    If std is True, the standard deviation is returned, instead of
    the average.
    the coord variable gives which analysis it should take. (either 'ampphase'(default) or 'RX')'''
    data = np.genfromtxt(
        get_datafld() +
        f'{Chipnum}\\Pulse\\{wvl}nm\\{subfolder}KID{KID}_{Pread}dBm__TmK{T}_avgpulse_{coord}.csv',
        delimiter=',', skip_header=1)
    ind = 0
    if std:
        ind += 1
    return data[:, ind], data[:, ind + 2]


def get_pulseavginfo(Chipnum, KID, Pread, wvl, T, subfolder='', coord='ampphase'):
    '''Returns the stream number (vis{nr}), location and prominence of
    the peaks used in the kidata.pulse.calc_avgpulse() function.'''
    strms, locs, proms = np.genfromtxt(
        get_datafld() +
        f'{Chipnum}\\Pulse\\{wvl}nm\\{subfolder}KID{KID}_{Pread}dBm__TmK{T}_avgpulse_{coord}_info.csv',
        delimiter=',', skip_header=1).T
    return strms.astype(int), locs.astype(int), proms


def get_pulsewvl(Chipnum):
    '''Returns which wavelengths are measured at pulse measurements'''
    return np.unique([int(i.split('\\')[-1][:-2])
                      for i in glob.iglob(
                          get_datafld() + f'{Chipnum}\\Pulse\\*nm')])

def get_pulseKIDPrT(Chipnum, wvl, subfolder=''):
    csvs = np.array([[int(i.split('\\')[-1].split('_')[0][3:]),
                      int(i.split('\\')[-1].split('_')[1][:-3]),
                      int(i.split('\\')[-1].split('_')[3][3:])]
                     for i in glob.iglob(get_datafld() + f'{Chipnum}\\Pulse\\{wvl}nm\\{subfolder}*.csv')])
    csvdf = pd.DataFrame(csvs, columns=['KID', 'Pread', 'T']).sort_values(
        by=['KID', 'Pread', 'T'])
    csvdf = csvdf.drop_duplicates()
    return csvdf.values


def get_pulseKIDs(Chipnum, wvl, subfolder=''):
    '''Returns which KIDs are measured at pulse measurements'''
    return np.unique([int(i.split('\\')[-1].split('_')[0][3:])
                     for i in glob.iglob(get_datafld() +
                                         f'{Chipnum}\\Pulse\\{wvl}nm\\{subfolder}*.csv')])


def get_pulsePread(Chipnum, KIDnum, wvl, subfolder=''):
    '''Returns which read powers are measured at pulse measurements'''
    return np.unique([int(i.split('\\')[-1].split('_')[1][:-3])
                      for i in glob.iglob(
                          get_datafld() +
                          f'{Chipnum}\\Pulse\\{wvl}nm\\{subfolder}KID{KIDnum}*.csv')])


def get_pulseTemp(Chipnum, KIDnum, Pread, wvl, subfolder=''):
    '''Returns which temperatures are measured at pulse measurements'''
    return np.unique([int(i.split('\\')[-1].split('_')[3][3:])
                      for i in glob.iglob(
                          get_datafld() +
                          f'{Chipnum}\\Pulse\\{wvl}nm\\{subfolder}KID{KIDnum}_{Pread}dBm*.csv')])

def get_pulsefits(Chipnum, KIDnum, Pread, wvl, subfolder=''):
    '''Returns the fitted values of the nonexp pulse fits in Pulse/wvl/fits'''
    return np.loadtxt(get_datafld() 
              + f'{Chipnum}/Pulse/{wvl}nm/{subfolder}fits/KID{KIDnum}_{Pread}dBm_.csv',
             delimiter=',', ndmin=2) 


# GR Noise

def get_noiseS21dat(Chipnum, KIDnum, Pread=None, dB=True, subfld='2D'):
    '''Returns the content of the raw .dat-file of the S21-sweep in the FFT/2D folder.
    If read power is not specified, the maximum is chosen.
    The values in decibel is default, but linear is also possible (this is not calibrated yet!)'''
    if Pread is None:
        Pread = get_grPread(get_grTDparam(Chipnum), KIDnum)[0]
    datafld = get_datafld()
    path = datafld + f"{Chipnum}/Noise_vs_T/FFT/{subfld}/KID{KIDnum}_{Pread}dBm__{'S21dB' if dB else 'S21'}.dat"
    return read_dat(path)


def get_noisetddat(Chipnum, KIDnum, Pread=None, subfld='2D'):
    '''Returns the content of the raw .dat-file with the time domain noise.'''
    if Pread is None:
        Pread = get_grPread(get_grTDparam(Chipnum), KIDnum)[0]
    datafld = get_datafld()
    path = datafld + f"{Chipnum}/Noise_vs_T/FFT/{subfld}/KID{KID}_{Pread}dBm__td.dat"
    return read_dat(path)


def get_noisebin(Chipnum, KIDnum, Pread=None, T=None, freq='med', subfld='2D'):
    '''Returns the contents of the .bin-file with raw (calibrated) noise in I and Q.'''
    if Pread is None:
        Pread = get_grPread(get_grTDparam(Chipnum), KIDnum)[0]
    if T is None:
        T = get_grTemp(get_grTDparam(Chipnum), KIDnum, Pread)[0]
    datafld = get_datafld()
    path = datafld + \
        f"{Chipnum}/Noise_vs_T/TD_{subfld}/KID{int(KIDnum)}_{int(Pread)}dBm__TD{freq}_TmK{int(T)}.bin"
    return np.fromfile(path, dtype='>f8').reshape(-1, 2)

def get_noisePSD(Chipnum, KIDnum, Pread, T, subfld='2D', fld='Noise_vs_T'):
    '''returns the calculated spectra from .csv file in the _PSDs folder'''
    return np.loadtxt(get_datafld() 
                      + f'{Chipnum}/{fld}/TD_{subfld}_PSDs/KID{KIDnum}_{Pread}dBm__TmK{T}.csv',
                     delimiter=',', ndmin=2)

def get_noisefits(Chipnum, KIDnum, Pread, subfld='2D'):
    return np.loadtxt(get_datafld() 
                  + f'{Chipnum}/Noise_vs_T/TD_{subfld}_PSDs/fits/KID{KIDnum}_{Pread}dBm_.csv',
                 delimiter=',', ndmin=2)

def get_noiseKIDPrT(Chipnum, subfld='2D'):
    avlfiles = get_avlfiles(get_datafld()
                            + f'{Chipnum}/Noise_vs_T/TD_{subfld}_PSDs',
                           ftype='.csv')
    return np.unique([[int(i[0][3:]), int(i[1][:-3]), int(i[3][3:])] 
                      for i in avlfiles], axis=0)

# THz measurement
def get_THzKIDPrTTBB(Chipnum):
    avlfiles = get_avlfiles(get_datafld()
                            + f'{Chipnum}/THz/TD_optNEP2D_BB_PSDs/',
                           ftype='.csv')
    return np.unique([[int(i[0][3:]), int(i[1][:-3]), 
                       float(i[2][5:])*1e3, int(i[3][3:])] 
                  for i in avlfiles], axis=0)
    
def get_THzPSD(Chipnum, KIDnum, Pread, T, TBB):
    '''returns the spectrum in the .csv file in THz/TD_optNEP2D_BB_PSDs folder.
    Give TBB in K and it will find the file that is closest in TmK<>.csv'''
    fld = get_datafld() + f'{Chipnum}/THz/TD_optNEP2D_BB_PSDs/'
    csvs = get_avlfiles(fld, ftype=f'KID{KIDnum}_{Pread}dBm_Tchip{(T*1e-3):.2f}*.csv')
    csv = csvs[np.abs(TBB - np.array([(float(i[3:])*1e-3) for i in csvs[:, -1]])).argmin(), :]
    if np.abs(TBB - np.array([(float(i[3:])*1e-3) for i in csvs[:, -1]])).min() > 1:
        warnings.warn('\nGiven TBB and TBB of data differ by more than 1 K: \n'
                      f' TBB = {TBB} K, TBB_data = {float(csv[-1][3:])*1e-3} K')
    return np.loadtxt(fld + '_'.join(csv) + '.csv',
                     delimiter=',')

def get_THzfits(Chipnum, KIDnum, Pread, T):
    return np.loadtxt(get_datafld() 
                  + f'{Chipnum}/THz/TD_optNEP2D_BB_PSDs/fits/KID{KIDnum}_{Pread}dBm_Tchip{(T*1e-3):.2f}.csv',
                 delimiter=',')

# Dictionaries to find data
def get_Vdict(Chipnum):
    '''Returns a dictionary with (KIDnum:Volume) from the S21 .csv-file.'''
    KIDlist = get_S21KIDs(Chipnum)
    Volumes = {}
    for KIDnum in KIDlist:
        S21data = get_S21data(Chipnum, KIDnum,
                              get_S21Pread(Chipnum, KIDnum)[0])
        Volumes[KIDnum] = S21data[0, 14]
    return Volumes


def get_Pintdict(Chipnum):
    '''Returns a dictionary with:
    KIDnum: Internal Power at lowest Temp., for each measured read power
    from the S21 .csv-file.'''
    KIDlist = get_S21KIDs(Chipnum)
    Pintdict = {}
    for KIDnum in KIDlist:
        S21Pread = np.array(get_S21Pread(Chipnum, KIDnum))
        Pintdict[KIDnum] = []
        for Pread in S21Pread:
            S21data = get_S21data(Chipnum, KIDnum, Pread)
            Q = S21data[0, 2]
            Qc = S21data[0, 3]
            Qi = S21data[0, 4]
            Pintdict[KIDnum].append(
                10*np.log10(10**(-1*Pread/10)*Q**2/Qc/np.pi))
    return Pintdict


def selectPread(pltPread, Preadar):
    """Function that returns a Pread array, depending on the input pltPread."""
    if type(pltPread) is str:
        if pltPread == "min":
            Pread = np.array([Preadar.max()])
        elif pltPread == "med":
            Pread = np.array([Preadar[np.abs(Preadar.mean() - Preadar).argmin()]])
        elif pltPread == "max":
            Pread = np.array([Preadar.min()])
        elif pltPread == "minmax":
            Pread = np.array([Preadar.max(), Preadar.min()])
        elif pltPread == "minmedmax":
            Pread = np.array(
                [
                    Preadar.max(),
                    Preadar[np.abs(Preadar.mean() - Preadar).argmin()],
                    Preadar.min(),
                ]
            )
        elif pltPread == "all":
            Pread = Preadar[::-1]
        else:
            raise ValueError("{} is not a valid Pread selection".format(pltPread))
    elif type(pltPread) == list:
        Pread = np.array(pltPread)
    elif type(pltPread) == int:
        Pread = np.array([np.sort(Preadar)[pltPread]])
    elif type(pltPread) == np.ndarray:
        Pread = pltPread
    else:
        raise ValueError("{} is not a valid Pread selection".format(pltPread))
    return Pread