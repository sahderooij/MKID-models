import numpy as np
import warnings 
import matplotlib.pyplot as plt

import scipy.io
import glob

def get_datafld():
    datafld = "D:\\MKIDdata\\"
    assert glob.glob(datafld), 'data folder not found'
    return datafld 

def read_dat(path):
    datdata = {}
    with open(path,'r') as file:
        datdata['Header'] = []
        nremptylines = 0
        linenr = 0
        nroflines = len(file.readlines())
        file.seek(0)

        while nremptylines < 2:
            line = file.readline().replace('\n','')
            linenr += 1
            if line == '':
                nremptylines += 1
            else:
                datdata['Header'].append(line)
        
        datdata['Data'] = {}
        while linenr <= nroflines:
            line = file.readline().replace('\n','')
            linenr += 1
            if "Temperature in K" in line:
                Temp = float(line.replace('Temperature in K:',''))
                nextline = file.readline()
                linenr += 1 
                while not nextline[0].isdigit() and nextline[0] != '-':
                    nextline = file.readline()
                    linenr += 1
                datdata['Data'][Temp] = np.fromstring(nextline,sep='\t')
            elif line != '':
                datdata['Data'][Temp] = np.vstack((datdata['Data'][Temp],np.fromstring(line,sep='\t')))
        
    return datdata

#S21
def get_S21KIDs(Chipnum):
    datafld = get_datafld()
    S21fld = datafld + '\\'.join([Chipnum,'S21','2D'])
    return np.unique([
        int(i.split('\\')[-1].split('_')[0][3:]) 
        for i in glob.glob(S21fld + '\\KID*_Tdep.csv')])

def get_S21Pread(Chipnum,KIDnum):
    datafld = get_datafld()
    S21fld = datafld + '\\'.join([Chipnum,'S21','2D'])
    return np.sort([
        int(i.split('\\')[-1].split('_')[1][:-3]) 
        for i in glob.glob(S21fld + '\\KID{}_*Tdep.csv'.format(KIDnum))])
    
def get_S21dat(Chipnum,KIDnum,Pread=None):
    if Pread is None:
        Pread = get_S21Pread(Chipnum,KIDnum)[0]
    datafld = get_datafld()
    path = datafld + "{}/S21/2D/KID{}_{}dBm_.dat".format(
                                Chipnum,KIDnum,Pread)
    return read_dat(path)

def get_S21data(Chipnum, KIDnum, Pread=None):
    Preadar = get_S21Pread(Chipnum,KIDnum)
    if Pread is None:
        Pread = Preadar[0]
    elif Pread not in Preadar:
        Pread_ = Preadar[np.abs(Preadar - Pread).argmin()]
        warnings.warn(f'No S21data at this Pread. Closest value is selected: -{Pread_} dBm instead of - {Pread} dBm')
        Pread = Pread_
    
    datafld = get_datafld()
    S21file = datafld + "\\".join(
        [
            Chipnum,
            "S21",
            "2D",
            "_".join(["KID" + str(KIDnum), str(Pread) + "dBm", "Tdep.csv"]),
        ]
    )
    S21data = np.genfromtxt(S21file, delimiter=",")[1:, :]
    return S21data

#Pulse
def get_pulseKIDs(Chipnum):
    return np.unique([int(i.split('\\')[-1].split('_')[0][3:]) 
                     for i in glob.glob(get_datafld() + f'{Chipnum}\\*mK\\*.mat')])

def get_pulsePread(Chipnum,KIDnum):
    return np.unique([int(i.split('\\')[-1].split('_')[1][:-3])
                      for i in glob.glob(
                          get_datafld() + f'{Chipnum}\\*mK\\KID{KIDnum}*.mat')])

def get_pulseTemp(Chipnum,KIDnum,Pread):
    return np.unique([int(i.split('\\')[-2][:-2])
                      for i in glob.glob(
                          get_datafld() + f'{Chipnum}\\*mK\\KID{KIDnum}_{Pread}dBm*.mat')])

def get_pulsewvl(Chipnum,KIDnum,Pread,Temp):
    return np.unique([int(i.split('\\')[-1].split('_')[2])
                      for i in glob.glob(
                          get_datafld() + f'{Chipnum}\\{Temp}mK\\KID{KIDnum}_{Pread}dBm*.mat')])    
    
def get_pulsedata(Chipnum,KIDnum,Pread,Tbath,wvlngth,points = 3000):
    datafld = get_datafld()
    peakfile = datafld + "\\".join(
        [
            Chipnum,
            str(Tbath) + "mK",
            "_".join(
                ["KID" + str(KIDnum), str(Pread) + "dBm",
                 str(wvlngth), str(points) + "points"]
            ),
        ]
    )
    peakdata = scipy.io.loadmat(peakfile)
    peakdata_ph = peakdata["pulsemodelfo"][0]
    peakdata_amp = peakdata["pulsemodelfo_amp"][0]
    return peakdata_ph, peakdata_amp

#GR Noise data
def get_noiseS21dat(Chipnum,KIDnum,Pread=None,dB=True):
    if Pread is None:
        Pread = get_grPread(get_grTDparam(Chipnum),KIDnum)[0]
    datafld = get_datafld()
    path = datafld + "{}/Noise_vs_T/FFT/2D/KID{}_{}dBm__{}.dat".format(
                                Chipnum,KIDnum,Pread,'S21dB' if dB else 'S21')
    return read_dat(path)

def get_noisetddat(Chipnum,KIDnum,Pread=None):
    if Pread is None:
        Pread = get_grPread(get_grTDparam(Chipnum),KIDnum)[0]
    datafld = get_datafld()
    path = datafld + "{}/Noise_vs_T/FFT/2D/KID{}_{}dBm__td.dat".format(
                                Chipnum,KIDnum,Pread)
    return read_dat(path)

def get_noisebin(Chipnum,KIDnum,Pread=None,T=None,freq='med'):
    if Pread is None:
        Pread = get_grPread(get_grTDparam(Chipnum),KIDnum)[0]
    if T is None:
        T = get_grTemp(get_grTDparam(Chipnum),KIDnum,Pread)[0]
    datafld = get_datafld()
    path = datafld + f"{Chipnum}/Noise_vs_T/TD_2D/KID{KIDnum}_{Pread}dBm__TD{freq}_TmK{T}.bin"
    return np.fromfile(path,dtype='>f8').reshape(-1,2)

def get_grTDparam(Chipnum,offres=False):
    datafld = get_datafld() + '\\' + Chipnum + "\\NoiseTDanalyse\\"
    if offres:
        GRdata = scipy.io.loadmat(datafld + "TDresults_offres")
    else:
        GRdata = scipy.io.loadmat(datafld + "TDresults")
    return GRdata["TDparam"]

def get_grKIDs(TDparam):
    return np.array([TDparam['kidnr'][0][i][0,0] for i in range(TDparam['kidnr'].size)])

def get_grTemp(TDparam,KIDnum,Pread):
    KIDlist = get_grKIDs(TDparam)
    ind = np.where(KIDlist == KIDnum)[0][0]
    Preadar = TDparam["Pread"][0, ind][:, 0]
    Tempar = TDparam["Temp"][0, ind][np.where(Preadar == Pread), :]
    return Tempar[0,0][np.nonzero(Tempar[0,0])]
    
def get_grPread(TDparam,KIDnum):
    KIDlist = get_grKIDs(TDparam)
    ind = np.where(KIDlist == KIDnum)[0][0]
    Preadar = TDparam["Pread"][0, ind][:, 0]
    return np.array(Preadar[np.nonzero(Preadar)])
    
def get_grdata(TDparam,KIDnum,Pread,Temp,spec='cross'):
    KIDlist = get_grKIDs(TDparam)
    ind = np.where(KIDlist == KIDnum)[0][0]
            
    Preadind = np.where(TDparam["Pread"][0, ind][:, 0] == Pread)
    Tempind = np.where(TDparam["Temp"][0, ind][Preadind, :][0,0] == Temp)
    freq = TDparam["fmtotal"][0, ind][Preadind,Tempind][0, 0][0]
    if spec == 'cross':
        SPR = TDparam["SPRrealneg"][0, ind][Preadind,Tempind][0, 0][0]
    elif spec == 'crosspos':
        SPR = TDparam["SPRrealpos"][0, ind][Preadind,Tempind][0, 0][0]
    elif spec == 'crossimag':
        SPR = TDparam["SPRimagneg"][0, ind][Preadind,Tempind][0, 0][0]
    elif spec == 'phase':
        SPR = TDparam["SPPtotal"][0, ind][Preadind,Tempind][0, 0][0]
    elif spec == 'amp':
        SPR = TDparam["SRRtotal"][0, ind][Preadind,Tempind][0, 0][0]
    else:
        raise ValueError('spec must be \'cross\', \'phase\' or \'amp\'.')
    return freq,SPR

def get_Vdict(Chipnum):
    KIDlist = get_grKIDs(get_grTDparam(Chipnum))
    Volumes = {}
    for KIDnum in KIDlist:
        S21data = get_S21data(Chipnum,KIDnum,
                                       get_S21Pread(Chipnum,KIDnum)[0])
        Volumes[KIDnum] = S21data[0,14]
    return Volumes

def get_Pintdict(Chipnum):
    KIDlist = get_grKIDs(get_grTDparam(Chipnum))
    Pintdict = {}
    for KIDnum in KIDlist:
        S21Pread = np.array(get_S21Pread(Chipnum,KIDnum))
        Pintdict[KIDnum] = []
        for Pread in S21Pread:
            S21data =  get_S21data(Chipnum,KIDnum,Pread)
            Q = S21data[0,2]
            Qc = S21data[0,3]
            Qi = S21data[0,4]
            Pintdict[KIDnum].append(10*np.log10(10**(-1*Pread/10)*Q**2/Qc/np.pi))
    return Pintdict

def get_Preaddict(Chipnum):
    TDparam = get_grTDparam(Chipnum)
    KIDlist = get_grKIDs(TDparam)
    Preaddict = {}
    for KIDnum in KIDlist:
        Preaddict[KIDnum] = get_grPread(TDparam,KIDnum)
    return Preaddict