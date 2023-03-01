import numpy as np
import pandas as pd
import warnings
import scipy.io
import glob


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

    datafld = "D:/MKIDdata/"
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
            if "Temperature in K" in line:
                Temp = float(line.replace('Temperature in K:', ''))
                nextline = file.readline()
                linenr += 1
                while not nextline[0].isdigit() and nextline[0] != '-':
                    nextline = file.readline()
                    linenr += 1
                datdata['Data'][Temp] = np.fromstring(nextline, sep='\t')
            elif line != '':
                datdata['Data'][Temp] = np.vstack(
                    (datdata['Data'][Temp], np.fromstring(line, sep='\t')))

    return datdata

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


def get_avlbins(folder):
    bins = np.unique(np.array([[int(i.split('\\')[-1].split('_')[0][3:]),
                      int(i.split('\\')[-1].split('_')[1][:-3]),
                      int(i.split('\\')[-1].split('_')[4][3:-4])]
                     for i in glob.iglob(folder + '/*.bin')]), axis=0)
    bindf = pd.DataFrame(bins, columns=['KID', 'Pread', 'T']).sort_values(
        by=['KID', 'Pread', 'T'])
    return bindf.values


def get_bin(folder, KID, Pread, T, strm):
    if type(strm) != str:
        return np.fromfile(
            folder + f"\\KID{KID}_{Pread}dBm__TDvis{int(strm)}_TmK{T}.bin",
            dtype=">f8").reshape(-1, 2)
    else:
        return np.fromfile(
            folder + f"\\KID{KID}_{Pread}dBm__TD{strm}_TmK{T}.bin",
            dtype=">f8").reshape(-1, 2)


def get_avgpulse(Chipnum, KID, Pread, T, wvl, subfolder='', std=False, coord='ampphase'):
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


def get_avgpulseinfo(Chipnum, KID, Pread, T, wvl, subfolder='', coord='ampphase'):
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


# GR Noise


def get_noiseS21dat(Chipnum, KIDnum, Pread=None, dB=True, subfld='2D'):
    '''Returns the content of the raw .dat-file of the S21-sweep in the FFT/2D folder.
    If read power is not specified, the maximum is chosen.
    The values in decibel is default, but linear is also possible (this is not calibrated yet!)'''
    if Pread is None:
        Pread = get_grPread(get_grTDparam(Chipnum), KIDnum)[0]
    datafld = get_datafld()
    path = datafld + f"{Chipnum}/Noise_vs_T/FFT/{subfld}/KID{KID}_{Pread}dBm__{'S21dB' if dB else 'S21'}.dat"
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


def get_grTDparam(Chipnum, version=None):
    '''Returns the data struct from the .mat-file generated by noise post-processing.
    version controles the trailing description of the TDresults.mat file as: TDresults_{version}.mat'''
    datafld = get_datafld() + '\\' + Chipnum + "\\NoiseTDanalyse\\"
    if version is not None:
        GRdata = scipy.io.loadmat(datafld + f"TDresults_{version}")
    else:
        GRdata = scipy.io.loadmat(datafld + "TDresults")
    return GRdata["TDparam"]


def get_grKIDs(TDparam):
    '''Returns an array of KID numbers, which are measured and stored with the TDparam data struct.'''
    return np.array([TDparam['kidnr'][0][i][0, 0] for i in range(TDparam['kidnr'].size)])


def get_grPread(TDparam, KIDnum):
    '''Returns an array of read powers, which are measured and stored with the TDparam data struct.'''
    KIDlist = get_grKIDs(TDparam)
    ind = np.where(KIDlist == KIDnum)[0][0]
    Preadar = TDparam["Pread"][0, ind][:, 0]
    return np.array(Preadar[np.nonzero(Preadar)])


def get_grTemp(TDparam, KIDnum, Pread):
    '''Returns an array of temperatures, which are measured and stored with the TDparam data struct.'''
    KIDlist = get_grKIDs(TDparam)
    ind = np.where(KIDlist == KIDnum)[0][0]
    Preadar = TDparam["Pread"][0, ind][:, 0]
    Tempar = TDparam["Temp"][0, ind][np.where(Preadar == Pread), :]
    return Tempar[0, 0][np.nonzero(Tempar[0, 0])]


def get_grdata(TDparam, KIDnum, Pread, Temp, spec='cross'):
    '''Returns the frequency and PSD, read from the TDparam data-struct.
    Default spectrum is cross (Ampliude, Phase) real, negative, but this can be set to:
    cross: 
    amp: 
    phase: 
    crosspos: real positive
    crossimag: imaginary part, all in dB as output.'''

    KIDlist = get_grKIDs(TDparam)
    ind = np.where(KIDlist == KIDnum)[0][0]

    Preadind = np.where(TDparam["Pread"][0, ind][:, 0] == Pread)
    Tempind = np.where(TDparam["Temp"][0, ind][Preadind, :][0, 0] == Temp)
    freq = TDparam["fmtotal"][0, ind][Preadind, Tempind][0, 0][0]
    if spec == 'cross':
        SPR = TDparam["SPRrealneg"][0, ind][Preadind, Tempind][0, 0][0]
    elif spec == 'crosspos':
        SPR = TDparam["SPRrealpos"][0, ind][Preadind, Tempind][0, 0][0]
    elif spec == 'crossimagneg':
        SPR = TDparam["SPRimagneg"][0, ind][Preadind, Tempind][0, 0][0]
    elif spec == 'crossimagpos':
        SPR = TDparam["SPRimagpos"][0, ind][Preadind, Tempind][0, 0][0]
    elif spec == 'phase':
        SPR = TDparam["SPPtotal"][0, ind][Preadind, Tempind][0, 0][0]
    elif spec == 'amp':
        SPR = TDparam["SRRtotal"][0, ind][Preadind, Tempind][0, 0][0]
    else:
        raise ValueError('spec must be \'cross\', \'phase\' or \'amp\'.')
    return freq, SPR


# Dictionaries to find data
def get_Vdict(Chipnum):
    '''Returns a dictionary with (KIDnum:Volume) from the S21 .csv-file.'''
    KIDlist = get_grKIDs(get_grTDparam(Chipnum))
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
    KIDlist = get_grKIDs(get_grTDparam(Chipnum))
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


def get_Preaddict(Chipnum):
    '''Returns a dictionary with:
    KIDnum: read powers used to measure noise.'''
    TDparam = get_grTDparam(Chipnum)
    KIDlist = get_grKIDs(TDparam)
    Preaddict = {}
    for KIDnum in KIDlist:
        Preaddict[KIDnum] = get_grPread(TDparam, KIDnum)
    return Preaddict
