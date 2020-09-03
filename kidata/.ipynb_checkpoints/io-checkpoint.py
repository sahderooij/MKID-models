import numpy as np
import warnings 
import matplotlib.pyplot as plt

import scipy.io
import glob

def get_datafld():
    return "D:\\MKIDdata\\"

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
    #To make faster/less code: read at once and split with '\n\n' and '\n'
    datafld = get_datafld()
    datdata = {}
    with open(datafld + "{}/S21/2D/KID{}_{}dBm_.dat".format(
                                Chipnum,KIDnum,Pread),'r') as file:
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
                _ = file.readline()
                firstline = file.readline()
                linenr += 2
                datdata['Data'][Temp] = np.fromstring(firstline,sep='\t')
            elif line != '':
                datdata['Data'][Temp] = np.vstack((datdata['Data'][Temp],np.fromstring(line,sep='\t')))
        
    return datdata


def get_S21data(Chipnum, KIDnum, Pread=None):
    if Pread is None:
        Pread = get_S21Pread(Chipnum,KIDnum)[0]
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

def get_peakdata(Chipnum,KIDnum, Pread, Tbath, wvlngth, points = 3000):
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
    return Preadar[np.nonzero(Preadar)]
    
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