import numpy as np
import warnings 
import matplotlib.pyplot as plt

from kidata import io
from kidata import calc

import KID

def init_KID(Chipnum,KIDnum,Pread,Tbath,Teffmethod = 'GR',wvl = None,S21 = False,
                t0=.44,
                kb=86.17,
                tpb=.28e-3,
                N0=1.72e4,
                kbTD=37312.0,
                lbd0=.092):
    TDparam = io.get_grTDparam(Chipnum)
    S21Pread = np.array(io.get_S21Pread(Chipnum,KIDnum))
    closestPread = S21Pread[np.abs(S21Pread - Pread).argmin()]
    S21data = io.get_S21data(Chipnum,KIDnum,closestPread)
    if closestPread != Pread:
        warnings.warn('S21data at another Pread')
    
    Qc = S21data[0, 3]
    hw0 = S21data[0, 5]*2*np.pi*6.582e-4*1e-6
    kbT0 = kb * S21data[0, 1]
    V = S21data[0, 14]
    d = S21data[0, 25]
    kbTc = S21data[0,21] * kb
    ak1 = calc.ak(S21data,lbd0,N0,kbTD)[0]
    
    tesc1 = calc.tesc(Chipnum,KIDnum,t0=t0,kb=kb,tpb=tpb,N0=N0,kbTD=kbTD)
    
    if Teffmethod == 'GR':
        Temp = io.get_grTemp(TDparam,KIDnum,Pread)
        taut = np.zeros(len(Temp))
        for i in range(len(Temp)):
            freq,SPR = io.get_grdata(TDparam,KIDnum,Pread,Temp[i])
            taut[i] = calc.tau(freq,SPR)[0]
        tauspl = interpolate.splrep(Temp[~np.isnan(taut)],taut[~np.isnan(taut)])
        tau1 = interpolate.splev(Tbath,tauspl)
        kbT = calc.kbTbeff(S21data,tau1,t0,kb,tpb,N0,kbTD,tesc1)
    elif Teffmethod == 'peak':
        peakdata_ph,peakdata_amp = io.get_pulsedata(Chipnum,KIDnum,Pread,Tbath,wvl)
        tau1 = calc.tau_pulse(peakdata_ph)
        kbT = calc.kbTbeff(S21data,tau1,t0,kb,tpb,N0,kbTD,tesc1)
    elif Teffmethod == 'Tbath':
        kbT = Tbath*1e-3*kb
    
    if S21:
        return KID.S21KID(S21data,
            Qc=Qc,hw0=hw0,kbT0=kbT0,kbT=kbT,V=V,
                          ak=ak1,d=d,kbTc=kbTc,tesc=tesc1)
    else:
        return KID.KID(
            Qc=Qc, hw0=hw0, kbT0=kbT0, kbT=kbT, V=V, ak=ak1, d=d, kbTc = kbTc,tesc = tesc1)