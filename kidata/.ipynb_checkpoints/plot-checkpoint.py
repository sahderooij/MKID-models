import numpy as np
import warnings 
import matplotlib.pyplot as plt

import matplotlib
from scipy import interpolate
import scipy.io

from kidata import io
from kidata import calc
from kidata import filters

import kidcalc

def _selectPread(pltPread,Preadar):
    '''Function that returns a Pread array, depending on the input pltPread.'''
    if type(pltPread) is str:
        if pltPread == 'min':
            Pread = np.array([Preadar.max()])
        elif pltPread == 'med':
            Pread = np.array([Preadar[np.abs(Preadar.mean()-Preadar).argmin()]])
        elif pltPread == 'max':
            Pread = np.array([Preadar.min()])
        elif pltPread == 'minmax':
            Pread = np.array([Preadar.max(),
                                Preadar.min()])
        elif pltPread == 'minmedmax':
            Pread = np.array([Preadar.max(),
                                Preadar[np.abs(Preadar.mean()-Preadar).argmin()],
                                Preadar.min()])
        elif pltPread == 'all':
            Pread = Preadar[::-1]
        else:
            raise ValueError('{} is not a valid Pread selection'.format(pltPread))
    elif type(pltPread) == list:
        Pread = np.array(pltPread)
    elif type(pltPread) == int:
        Pread = np.array([np.sort(Preadar)[pltPread]])
    elif type(pltPread) == np.ndarray:
        Pread = pltPread
    else:
        raise ValueError('{} is not a valid Pread selection'.format(pltPread))
    return Pread
    

def spec(Chipnum,KIDlist=None,pltPread='all',spec=['cross'],lvlcomp='',comptres=False,clbar=True,
              del1fNoise=False,delampNoise=False,del1fnNoise=False,suboffres=False,
              plttres=False,
              Tminmax=(0,500),ax12=None,
             xlim=(None,None),ylim=(None,None)):
    '''Plots PSDs of multiple KIDs, read powers and temperatures (indicated by color). Every KID has a new figure, which is returned if only one KID is plotted.
    lvlcomp specifies how the noise levels should be compensated (will be a different function in the future). 
    Use Resp to divide by responsivity and obtain quasiparticle fluctuations.
    comptres compensates for the factor (1+(omega*taures)^2), to get the quasiparticle fluctuations.
    plttres will plot arrow at the frequencies corresponding to the resonator ring time.'''
    TDparam = io.get_grTDparam(Chipnum)
    if suboffres:
        TDparamoffres = io.get_grTDparam(Chipnum,offres=True)

    if KIDlist is None:
        KIDlist = io.get_grKIDs(TDparam)
    elif type(KIDlist) is int:
        KIDlist = [KIDlist]
    
    if spec == 'all':
        specs = ['cross','amp','phase']
    elif type(spec) == list:
        specs = spec
    else:
        raise ValueError('Invalid Spectrum Selection')
        
    for KIDnum in KIDlist:
        Preadar = _selectPread(pltPread,io.get_grPread(TDparam,KIDnum))
        if ax12 is None:
            fig,axs = plt.subplots(len(Preadar),len(specs),
                                   figsize=(4*len(specs),4*len(Preadar)),
                                   sharex=True,sharey=True,squeeze=False)
            fig.suptitle(f'{Chipnum}, KID{KIDnum}')
        else:
            axs = ax12
        
        for ax1,Pread in zip(range(len(Preadar)),Preadar):
            if lvlcomp or comptres or plttres:
                S21data = io.get_S21data(Chipnum,KIDnum,Pread)

            axs[ax1,0].set_ylabel('PSD (dBc/Hz)')
            Temp = io.get_grTemp(TDparam,KIDnum,Pread)
            if suboffres:
                Temp = np.intersect1d(Temp,io.get_grTemp(TDparamoffres,KIDnum,Pread))
                
            Temp = Temp[np.logical_and(Temp<Tminmax[1],Temp>Tminmax[0])]
            cmap = matplotlib.cm.get_cmap('viridis')
            norm = matplotlib.colors.Normalize(Temp.min(),Temp.max())
            if clbar:
                clb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),
                                   ax=axs[ax1,-1])
                clb.ax.set_title('T (mK)')
                
            for i in range(len(Temp)):
                for (ax2,spec) in zip(range(len(specs)),specs):                 
                    freq,SPR = io.get_grdata(TDparam,KIDnum,Pread,Temp[i],spec=spec)
                    if suboffres:
                        orfreq,orSPR = io.get_grdata(TDparamoffres,KIDnum,Pread,Temp[i],spec=spec)
                        freq,SPR = filters.subtr_spec(freq,SPR,orfreq,orSPR)
                        
                    if delampNoise:
                        freq,SPR = filters.del_ampNoise(freq,SPR)
                    if del1fNoise:
                        freq,SPR = filters.del_1fNoise(freq,SPR)
                    if del1fnNoise:
                        freq,SPR = filters.del_1fnNoise(freq,SPR)
                        
                    SPR[SPR==-140] = np.nan
                    SPR[SPR==-np.inf] = np.nan

                    if lvlcomp == 'Resp':
                        if spec == 'cross':
                            Respspl = interpolate.splrep(
                                S21data[:,1]*1e3,np.sqrt(S21data[:,10]*S21data[:,18]),s=0)
                        elif spec == 'amp':
                            Respspl = interpolate.splrep(
                                S21data[:,1]*1e3,S21data[:,18],s=0)
                        elif spec == 'phase':
                            Respspl = interpolate.splrep(
                                S21data[:,1]*1e3,S21data[:,10],s=0)
                        
                        SPR = 10*np.log10(10**(SPR/10)/interpolate.splev(Temp[i],Respspl)**2)
                        
                    if comptres:
                        Tind = np.abs(S21data[:,1]-Temp[i]*1e-3).argmin()
                        SPR = 10*np.log10(10**(SPR/10)*\
                                          (1+(freq*2*S21data[Tind,2]/S21data[Tind,5])**2))                        
                        
                    axs[ax1,ax2].plot(freq,SPR,color=cmap(norm(Temp[i])))
                    axs[ax1,ax2].set_xscale('log')
                    axs[ax1,ax2].set_title(spec+ ', -{} dBm'.format(Pread))
                    axs[-1,ax2].set_xlabel('Freq. (Hz)')
                    axs[-1,ax2].set_xlim(*xlim)
                    
                    if plttres:
                        Tind = np.abs(S21data[:,1]-Temp[i]*1e-3).argmin()
                        axs[ax1,ax2].annotate('',(S21data[Tind,5]/(2*S21data[Tind,2]),
                                                  ylim[1]),
                                              (S21data[Tind,5]/(2*S21data[Tind,2]),
                                                  ylim[1]+10),
                                             arrowprops=dict(arrowstyle='->',color=cmap(norm(Temp[i]))),
                                             annotation_clip=False)
                        

            axs[ax1,0].set_ylim(*ylim)
            plt.tight_layout(rect=(0,0,1,1-.12/len(Preadar)))
    if ax12 is None and len(KIDlist) == 1:
        return fig,axs
                    
                    
def ltnlvl(Chipnum,KIDlist=None,pltPread='all',spec='cross',Tminmax=None,startstopf=(None,None),
           lvlcomp='',
                delampNoise=False,del1fNoise=False,del1fnNoise=False,suboffres=False,relerrthrs=.2,
                pltKIDsep=True,pltthlvl=False,pltkaplan=False,pltthmfnl=False,plttres=False,
                fig=None,ax12=None,color='Pread',pltclrbar=True,fmt='-o',label=None,
                defaulttesc=0,tescPread='max',tescpltkaplan=False,tescTminmax=(300,400),
                showfit=False,savefig=False):
    '''Plots the results from a Lorentzian fit to the PSDs of multiple KIDs, read powers and temperatures. 
    Two axes: 0: lifetimes 1: noise levels, both with temperature on the x-axis. The color can be specified and
    is Pread by default. 
    Options:
    startstopf -- defines the fitting window
    lvlcomp -- defines how the levels are compensated. Use Resp for responsivity compensation.
        (will be moved in the future)
    del{}Noise -- filter spectrum before fitting.
    relerrthrs -- only plot fits with a relative error threshold in lifetime less than this.
    pltKIDsep -- if True, different KIDs get a new figure.
    pltthlvl -- expected noise level is plotted as dashed line
    pltkaplan -- a kaplan fit (tesc as parameter) is plotted in the lifetime axis.
    pltthmfnl -- a noise level from the fitted lifetime and theoretical Nqp is plotted as well
    plttres -- the resonator ring time is plotted in the lifetime axis.
    ... multiple figure handling options ...
    ... options for the tesc deteremination ...
    showfit -- the fits are displayed in numerous new figures, for manual checking.'''

    def _make_fig():
        fig, axs = plt.subplots(1,2,figsize=(8,3))
        return fig,axs
    def _get_cmap(**kwargs):
        if color == 'Pread':
            cmap = matplotlib.cm.get_cmap('plasma')
            norm = matplotlib.colors.Normalize(-1.05*kwargs['Preadar'].max(),-.95*kwargs['Preadar'].min())
            clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap))
            clb.ax.set_title(r'$P_{read}$ (dBm)')
        elif color == 'Pint':
            cmap = matplotlib.cm.get_cmap('plasma')
            norm = matplotlib.colors.Normalize(kwargs['Pintar'].min()*1.05,kwargs['Pintar'].max()*.95)
            clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap))
            clb.ax.set_title(r'$P_{int}$ (dBm)')
        elif color == 'V':
            cmap = matplotlib.cm.get_cmap('cividis')
            norm = matplotlib.colors.Normalize(
                np.array(list(Vdict.values())).min(),
                np.array(list(Vdict.values())).max())
            clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap))
            clb.ax.set_title(r'Al Vol. ($\mu m^3$)')
        elif color == 'KIDnum':
            cmap = matplotlib.cm.get_cmap('Paired')
            norm = matplotlib.colors.Normalize(np.array(KIDlist).min(),
                                               np.array(KIDlist).max())
            clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap))
            clb.ax.set_title('KID nr.')
        else:
            raise ValueError('{} is not a valid variable as color'.format(color))
        return cmap,norm
    
    TDparam = io.get_grTDparam(Chipnum)
    if suboffres:
        TDparamoffres = io.get_grTDparam(Chipnum,offres=True)
        
    if KIDlist is None:
        KIDlist = io.get_grKIDs(TDparam)
    elif type(KIDlist) is int:
        KIDlist = [KIDlist]
    
    if color == 'Pint':
        Pintdict = io.get_Pintdict(Chipnum)
    
    if not pltKIDsep:
        if ax12 is None:
            fig,axs = _make_fig()
        else:
            axs = ax12
        if color == 'Pint':
            Pintar = np.array([Pintdict[k] for k in KIDlist])
            cmap,norm = _get_cmap(Pintar=Pintar)
        elif color == 'V':
            Vdict = io.get_Vdict(Chipnum)
            cmap,norm = _get_cmap(Vdict=Vdict)
        elif color == 'Pread':
            Preaddict = io.get_Preaddict(Chipnum)
            Preadar = np.array([Preaddict[k] for k in KIDlist])
            cmap,norm = _get_cmap(Preadar=Preadar)
        elif color == 'KIDnum':
            cmap,norm = _get_cmap(KIDlist=KIDlist)

    for KIDnum in KIDlist:
        Preadar = _selectPread(pltPread,io.get_grPread(TDparam,KIDnum))
        if pltKIDsep:
            if ax12 is None:
                fig,axs = _make_fig()
            else:
                axs = ax12
                
            if len(KIDlist) > 1:
                fig.suptitle(f'KID{KIDnum}')
            
            if color == 'Pread':
                cmap,norm = _get_cmap(Preadar=Preadar)
            elif color == 'Pint':
                cmap,norm = _get_cmap(Pintar=np.array(Pintdict[KIDnum]))
            
        if pltthlvl or 'tesc' in lvlcomp or pltkaplan:
            tesc_ = calc.tesc(Chipnum,KIDnum,
                         defaulttesc=defaulttesc,minTemp=tescTminmax[0],maxTemp=tescTminmax[1],
                         relerrthrs=relerrthrs,Pread=tescPread,pltkaplan=tescpltkaplan)
    
        for Pread in Preadar:
            Temp = np.trim_zeros(io.get_grTemp(TDparam,KIDnum,Pread))
            if lvlcomp != '':
                
                S21data = io.get_S21data(Chipnum,KIDnum,Pread)
                if 'ak' in lvlcomp:
                    akin = calc.ak(S21data)
                V = S21data[0,14]
                if spec == 'cross':
                    Respspl = interpolate.splrep(
                        S21data[:,1]*1e3,np.sqrt(S21data[:,10]*S21data[:,18]),s=0)
                elif spec == 'amp':
                    Respspl = interpolate.splrep(
                        S21data[:,1]*1e3,S21data[:,18],s=0)
                elif spec == 'phase':
                    Respspl = interpolate.splrep(
                        S21data[:,1]*1e3,S21data[:,10],s=0)

                if lvlcomp == 'QakV':
                    sqrtlvlcompspl = interpolate.splrep(
                        S21data[:,1]*1e3,
                        S21data[:,2]*akin/V,s=0)
                elif lvlcomp == 'QaksqrtV': 
                    sqrtlvlcompspl = interpolate.splrep(
                        S21data[:,1]*1e3,
                        S21data[:,2]*akin/np.sqrt(V),s=0)
                elif lvlcomp == 'QaksqrtVtesc':
                    sqrtlvlcompspl = interpolate.splrep(
                        S21data[:,1]*1e3,
                        S21data[:,2]*akin/np.sqrt(V*\
                                                  (1+tesc_/.28e-3)),s=0)
                elif lvlcomp == 'QaksqrtVtescTc':
                    sqrtlvlcompspl = interpolate.splrep(
                        S21data[:,1]*1e3,
                        S21data[:,2]*akin/np.sqrt(V*\
                                                   (1+tesc_/.28e-3)*\
                                                (86.17*S21data[0,21])**3/\
                                                   (S21data[0,15]/1.6e-19*1e6)**2),s=0)
                elif lvlcomp == 'Resp':            
                    sqrtlvlcompspl = Respspl
                    
                elif lvlcomp == 'RespPulse':
                    pulsePreadar = io.get_pulsePread(Chipnum,KIDnum)
                    pulsePreadselect = pulsePreadar[np.abs(pulsePreadar-Pread).argmin()]
                    pulseTemp = io.get_pulseTemp(Chipnum,KIDnum,pulsePreadselect).min()
                    pulsewvl = io.get_pulsewvl(Chipnum,KIDnum,pulsePreadselect,pulseTemp).min()
                    phasepulse,amppulse = io.get_pulsedata(
                        Chipnum,KIDnum,pulsePreadselect,pulseTemp,pulsewvl)
                    
                    phtau = calc.tau_pulse(phasepulse)
                    amptau = calc.tau_pulse(amppulse)
                    assert np.abs(1-phtau/amptau) < .1, 'Amp and Phase lifetimes differ by more than 10%' 
                    dAdTheta = -1*(amppulse/phasepulse)[600:int(600+2*phtau)].mean()

                    if spec == 'cross':
                        Respspl = interpolate.splrep(
                            S21data[:,1]*1e3,np.sqrt(S21data[:,10]**2*dAdTheta),s=0)
                    elif spec == 'amp':
                        Respspl = interpolate.splrep(
                            S21data[:,1]*1e3,S21data[:,10]*dAdTheta,s=0)
                    elif spec == 'phase':
                        Respspl = interpolate.splrep(
                            S21data[:,1]*1e3,S21data[:,10],s=0)
                    sqrtlvlcompspl = Respspl
                    
                elif lvlcomp == 'RespPint':
                    Pint = 10**(-Pread/10)*S21data[:,2]**2/(S21data[:,3]*np.pi)
                    Pint /= Pint[0]
                    sqrtlvlcompspl = interpolate.splrep(
                        S21data[:,1]*1e3,
                        interpolate.splev(S21data[:,1]*1e3,
                                          Respspl)/Pint**(1/4),s=0)
                elif lvlcomp == 'RespV':            
                    sqrtlvlcompspl = interpolate.splrep(
                        S21data[:,1]*1e3,
                        interpolate.splev(S21data[:,1]*1e3,
                                          Respspl)*np.sqrt(V),s=0)
                elif lvlcomp == 'RespVtescTc':   
                    kbTc = 86.17*S21data[0,21]
                    sqrtlvlcompspl = interpolate.splrep(
                        S21data[:,1]*1e3,
                        interpolate.splev(S21data[:,1]*1e3,
                                          Respspl)*\
                        np.sqrt(V*(1+tesc_/.28e-3)*\
                                (kbTc)**3/\
                                (kidcalc.D(86.17*S21data[:,1],1.72e4,kbTc,37312.))**2),s=0)
                elif lvlcomp == 'RespLowT':
                    sqrtlvlcompspl = interpolate.splrep(
                        S21data[:,1]*1e3,np.ones(len(S21data[:,1]))*\
                    interpolate.splev(S21data[0,1]*1e3,Respspl))
                else:
                    raise ValueError('{} is an invalid compensation method'.format(
                        lvlcomp))

                Pint = 10*np.log10(10**(-1*Pread/10)*S21data[0,2]**2/S21data[0,3]/np.pi)
            else:
                sqrtlvlcompspl = interpolate.splrep(
                        Temp,np.ones(len(Temp)))
            
            
            if suboffres:
                Temp = np.intersect1d(Temp,io.get_grTemp(TDparamoffres,KIDnum,Pread))
                
            if Tminmax != None:
                if Tminmax[0] != None:
                    Temp = Temp[Temp > Tminmax[0]]
                if Tminmax[1] != None:
                    Temp = Temp[Temp < Tminmax[1]]
            taut = np.zeros((len(Temp)))
            tauterr = np.zeros((len(Temp)))
            lvl = np.zeros((len(Temp)))
            lvlerr = np.zeros((len(Temp)))
            for i in range(len(Temp)):
                freq,SPR = io.get_grdata(TDparam,KIDnum,Pread,Temp[i],spec)
                if suboffres:
                    orfreq,orSPR = io.get_grdata(TDparamoffres,KIDnum,Pread,Temp[i],spec)
                    freq,SPR = filters.subtr_spec(freq,SPR,orfreq,orSPR)
                if delampNoise:
                    freq,SPR = filters.del_ampNoise(freq,SPR)
                if del1fNoise:
                    freq,SPR = filters.del_1fNoise(freq,SPR)
                if del1fnNoise:
                    freq,SPR = filters.del_1fnNoise(freq,SPR)
                    
                if showfit:
                    print('{}, KID{}, -{} dBm, T={}, {}'.format(
                        Chipnum,KIDnum,Pread,Temp[i],spec))
                taut[i],tauterr[i],lvl[i],lvlerr[i] = \
                    calc.tau(freq,SPR,plot=showfit,retfnl=True,
                             startf=startstopf[0],stopf=startstopf[1])
                if showfit:
                    print(tauterr[i]/taut[i])

                                
                lvl[i] = lvl[i]/interpolate.splev(Temp[i],sqrtlvlcompspl)**2
                lvlerr[i] = lvlerr[i]/interpolate.splev(Temp[i],sqrtlvlcompspl)**2

            #Deleting bad fits and plotting:
            mask = ~np.isnan(taut)
            mask[mask] = tauterr[mask]/taut[mask] <= relerrthrs
            
            if color == 'Pread':
                clr = cmap(norm(-1*Pread))
            elif color == 'Pint':
                clr = cmap(norm(Pint))
            elif color == 'V':
                clr = cmap(norm(Vdict[KIDnum]))
            elif color == 'KIDnum':
                clr = cmap(norm(KIDnum))
            else:
                clr = color
            
            axs[0].errorbar(Temp[mask],taut[mask],
                                     yerr = tauterr[mask],fmt = fmt,capsize = 3.,
                              color=clr,mec='k',label=label if Pread == Preadar[-1] else '')
            axs[1].errorbar(Temp[mask],10*np.log10(lvl[mask]),
                                     yerr = 10*np.log10((lvlerr[mask]+lvl[mask])/lvl[mask]),
                                     fmt = fmt, capsize = 3.,color=clr,mec='k',
                           label=label if Pread == Preadar[-1] else '')
            if pltthlvl:
                if Tminmax is not None:
                    Tstartstop = Tminmax
                else:
                    Tstartstop = (Temp[mask].min(),Temp[mask].max())
                Ttemp = np.linspace(*Tstartstop,100)
                explvl = interpolate.splev(Ttemp,Respspl)**2
                explvl *= 4*.44e-6*S21data[0,14]*1.72e4*(86.17*S21data[0,21])**3/\
                (2*(S21data[0,15]/1.602e-19*1e6)**2)*(1+tesc_/.28e-3)/2
                explvl /= interpolate.splev(Ttemp,sqrtlvlcompspl)**2
                thlvlplot, = axs[1].plot(Ttemp,10*np.log10(explvl),
                                         color=clr,linestyle='--',linewidth=2.)
                axs[1].legend((thlvlplot,),(r'Expected noise level',))
                
            if pltkaplan and Temp[mask].size != 0:
                if Tminmax is not None:
                    Tstartstop = Tminmax
                else:
                    Tstartstop = (Temp[mask].min(),Temp[mask].max())
                T = np.linspace(*Tstartstop,100)*1e-3
                taukaplan = kidcalc.tau_kaplan(T,tesc=tesc_,kbTc=86.17*S21data[0,21])
                kaplanfit, = axs[0].plot(T*1e3,taukaplan,color=clr,linestyle='--',linewidth=2.)
                axs[0].legend((kaplanfit,),('Kaplan',))
                
            if pltthmfnl:
                try:
                    tauspl = interpolate.splrep(Temp[mask],taut[mask],s=0)
                    T = np.linspace(Temp[mask].min(),Temp[mask].max(),100)
                    Nqp = np.zeros(len(T))
                    for i in range(len(T)):
                        Nqp[i] = V*nqp(T[i]*86.17*1e-3,S21data[0,15]/1.602e-19*1e6,1.72e4)
                    thmfnl = 4*interpolate.splev(T,tauspl)*1e-6*\
                        Nqp*interpolate.splev(T,Respspl)**2
                    thmfnl /= interpolate.splev(T,sqrtlvlcompspl)**2
                    thmfnlplot, = axs[1].plot(T,10*np.log10(thmfnl),color=clr,
                                              linestyle='--',linewidth=3.)
                    axs[1].legend((thmfnlplot,),('Thermal Noise Level \n with measured $\\tau_{qp}^*$',))
                except:
                    warnings.warn('Could not make Thermal Noise Level, {},KID{},-{} dBm,{}'.format(
                    Chipnum,KIDnum,Pread,spec))
            if plttres:
                tresline, = axs[0].plot(S21data[:,1]*1e3,
                            S21data[:,2]/(np.pi*S21data[:,5])*1e6,color=clr,linestyle=':')
                axs[0].legend((tresline,),('$\\tau_{res}$',))
                
                    
        axs[0].set_yscale('log')
        for i in range(2):
            axs[i].set_xlabel('Temperature (mK)')
        axs[0].set_ylabel(r'$\tau_{qp}^*$ (µs)')
        
        if lvlcomp == 'Resp':
            axs[1].set_ylabel(r'Noise Level/$\mathcal{R}^2(T)$ (dB/Hz)')
        elif lvlcomp == 'RespV':            
            axs[1].set_ylabel(r'Noise Level/$(\mathcal{R}^2(T)V)$ (dB/Hz)')
        elif lvlcomp == 'RespVtescTc':   
            axs[1].set_ylabel(r'Noise Level/$(\mathcal{R}^2(T)\chi)$ (dB/Hz)')   
        elif lvlcomp == '':
            axs[1].set_ylabel(r'Noise Level (dB/Hz)')
        elif lvlcomp == 'RespLowT':
            axs[1].set_ylabel(r'Noise Level/$\mathcal{R}^2(T=50 mK)$ (dB/Hz)')
        else:
            axs[1].set_ylabel(r'comp. Noise Level (dB/Hz)')
        plt.tight_layout()
        if savefig:
            plt.savefig('GR_{}_KID{}_{}.pdf'.format(Chipnum,KIDnum,spec))
            plt.close()
    if ax12 is None:
        return fig,axs
            
          
def Qif0(Chipnum,KIDnum,color='Pread',Tmax=.5,pltPread='all',fracfreq=False,
        fig=None,ax12=None):
    '''Plot the internal quality factor and resonance frequency from S21-measurement.
    The color gives different read powers, but can be set to Pint as well.
    If fracfreq is True, the y-axis is df/f0, instead of f0.'''
    dfld = io.get_datafld()
    if fig is None or ax12 is None:
        fig,axs = plt.subplots(1,2,figsize=(12,4))
        fig.suptitle(f'{Chipnum}, KID{KIDnum}')
    else:
        axs = ax12
    
    Preadar = _selectPread(pltPread,io.get_S21Pread(Chipnum,KIDnum))
    if color == 'Pread':   
        cmap = matplotlib.cm.get_cmap('plasma')
        norm = matplotlib.colors.Normalize(-1.05*Preadar.max(),-.95*Preadar.min())
        clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap))
        clb.ax.set_title(r'$P_{read}$ (dBm)')
    elif color == 'Pint':
        Pint = np.array(io.get_Pintdict(Chipnum)[KIDnum])
        cmap = matplotlib.cm.get_cmap('plasma')
        norm = matplotlib.colors.Normalize(Pint.min()*1.05,Pint.max()*.95)
        clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap))
        clb.ax.set_title(r'$P_{int}$ (dBm)')
        
    for Pread in Preadar:
        S21data = io.get_S21data(Chipnum,KIDnum,Pread)
        if color == 'Pread':
            clr = cmap(norm(-Pread))
        elif color == 'Pint':
            clr = cmap(norm(Pint[Preadar == Pread][0]))
        T = S21data[:,1]
        axs[0].plot(T[T<Tmax]*1e3,S21data[T<Tmax,4],color=clr)
        if fracfreq:
            axs[1].plot(T[T<Tmax]*1e3,(S21data[T<Tmax,5]-S21data[0,5])/S21data[0,5]*1e5,color=clr)
        else:
            axs[1].plot(T[T<Tmax]*1e3,S21data[T<Tmax,5]*1e-9,color=clr)
        
    for ax in axs:
        ax.set_xlabel('Temperature (mK)')
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
    axs[0].set_ylabel('$Q_i$')
    axs[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    if fracfreq:
        axs[1].set_ylabel('$10^5~\delta f_0/f_0$')
    else:
        axs[1].set_ylabel('f0 (GHz)')
    fig.tight_layout()

def Nqp(Chipnum,KIDnum,pltPread='all',spec='cross',
        startstopf=(None,None),
        delampNoise=False,del1fNoise=False,del1fnNoise=False,Tmax=500,relerrthrs=.3,
        pltThrm=True,pltNqpQi=False,splitT=0,pltNqptau=False,tescPread='max',nqpaxis=True,
        fig=None,ax=None,label=None,clr=None,
        N0 = 1.72e4,
        kbTD = 37312.0,
        kb = 86.17):
    '''Plots the number of quasiparticle calculated from the noise levels and lifetimes from PSDs.
    options similar to options in ltnlvl.
    TODO: delete double code in ltnlvl and Nqp
    
    pltThrm -- also plot thermal line (needs constants)
    pltNqpQi -- plot Nqp from Qi as well (needs constants)
        splitT -- makes NqpQi line dashed below this T
    pltNqptau -- plot Nqp from lifetime only (need constants)
    nqpaxis -- also shows density on right axis.
    '''
    TDparam = io.get_grTDparam(Chipnum)
    if ax is None or fig is None:
        fig,ax = plt.subplots()       
    
    Preadar = _selectPread(pltPread,io.get_grPread(TDparam,KIDnum))
        
    if Preadar.size > 1:
        cmap = matplotlib.cm.get_cmap('plasma')
        norm = matplotlib.colors.Normalize(-1.05*Preadar.max(),-.95*Preadar.min())
        clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),
                           ax=ax)
        clb.ax.set_title(r'$P_{read}$ (dBm)')
        
    for Pread in Preadar:
        S21data = io.get_S21data(Chipnum,KIDnum,Pread)
            
        if spec == 'cross':
            Respspl = interpolate.splrep(
                S21data[:,1]*1e3,np.sqrt(S21data[:,10]*S21data[:,18]),s=0)
        elif spec == 'amp':
            Respspl = interpolate.splrep(
                S21data[:,1]*1e3,S21data[:,18],s=0)
        elif spec == 'phase':
            Respspl = interpolate.splrep(
                S21data[:,1]*1e3,S21data[:,10],s=0)
        
        Temp = io.get_grTemp(TDparam,KIDnum,Pread)
        Temp = Temp[Temp<Tmax]
        Nqp,Nqperr,taut = np.zeros((3,len(Temp)))
        for i in range(len(Temp)):
            freq,SPR = io.get_grdata(TDparam,KIDnum,Pread,Temp[i],spec=spec)
            if delampNoise:
                freq,SPR = filters.del_ampNoise(freq,SPR)
            if del1fNoise:
                freq,SPR = filters.del_1fNoise(freq,SPR)
            if del1fnNoise:
                freq,SPR = filters.del_1fnNoise(freq,SPR)
            taut[i],tauterr,lvl,lvlerr = \
                            calc.tau(freq,SPR,retfnl = True,
                                     startf=startstopf[0],stopf=startstopf[1])
            lvl = lvl/interpolate.splev(Temp[i],Respspl)**2
            lvlerr = lvlerr/interpolate.splev(Temp[i],Respspl)**2
            Nqp[i] = lvl/(4*taut[i]*1e-6)
            Nqperr[i] = np.sqrt((lvlerr/(4*taut[i]*1e-6))**2+\
                                (-lvl*tauterr*1e-6/(4*(taut[i]*1e-6)**2))**2)
        mask = ~np.isnan(Nqp)
        mask[mask] = Nqperr[mask]/Nqp[mask] <= relerrthrs
        if Preadar.size > 1:
            clr = cmap(norm(-1*Pread))
        elif pltPread == 'min':
            clr = 'purple'
        elif pltPread == 'max':
            clr = 'gold'
            
        dataline = ax.errorbar(Temp[mask],Nqp[mask],yerr=Nqperr[mask],
                    color=clr,marker='o',mec='k',capsize=2.,label=label)
        if pltNqptau:
            tesc = calc.tesc(Chipnum,KIDnum,Pread=tescPread)
            Nqp_ = S21data[0,14]*kidcalc.nqpfromtau(taut,tesc,kb*S21data[0,21])
            tauline, = ax.plot(Temp[mask],Nqp_[mask],
                   color=clr,zorder=len(ax.lines)+1,
                    label='$\\tau_{qp}^*$')
    if pltNqpQi:
        Preadar = io.get_S21Pread(Chipnum,KIDnum)
        for Pread in Preadar:
            S21data = io.get_S21data(Chipnum,KIDnum,Pread)
            T,Nqp = calc.NqpfromQi(S21data)
            mask = np.logical_and(T*1e3>ax.get_xlim()[0],T*1e3<ax.get_xlim()[1])
            totalT = T[mask]
            totalNqp = Nqp[mask]
            if len(Preadar) == 1:
                clr = 'g'
            else:
                clr = cmap(norm(closestPread))
            Qline, = ax.plot(totalT[totalT>splitT]*1e3,totalNqp[totalT>splitT],
                    linestyle='-',color=clr,zorder=len(ax.lines)+1,label='$Q_i$')
            ax.plot(totalT[totalT<splitT]*1e3,totalNqp[totalT<splitT],
                    linestyle='--',color=clr,zorder=len(ax.lines)+1)
    if pltThrm:
        T = np.linspace(*ax.get_xlim(),100)
        NqpT = np.zeros(100)
        for i in range(len(T)):
            D_ = kidcalc.D(kb*T[i]*1e-3, N0, kb*S21data[0,21], kbTD)
            NqpT[i] = S21data[0,14]*kidcalc.nqp(kb*T[i]*1e-3, D_, N0)
        Thline, = ax.plot(T,NqpT,color='k',zorder=len(ax.lines)+1,label='Thermal $N_{qp}$')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())         
    
    ax.set_ylabel('$N_{qp}$')
    ax.set_xlabel('Temperature (mK)')

    ax.set_yscale('log') 
    
    if nqpaxis:
        def nqptoNqp(x):
            return x*S21data[0,14]
        def Nqptonqp(x):
            return x/S21data[0,14]

        ax2 = ax.secondary_yaxis('right', functions=(Nqptonqp,nqptoNqp))
        ax2.set_ylabel('$n_{qp}$ ($\\mu m^{-3}$)')
    if Preadar.size > 1:
        l,b,w,h = clb.ax.get_position().bounds
        clb.ax.set_position([l+.12,b,w,h])
    
def Qfactors(Chipnum,KIDnum,Pread=None,ax=None):
    '''Plots Ql, Qi and Qc over temperature in one figure.'''
    S21data = io.get_S21data(Chipnum,KIDnum,Pread)
    T = S21data[:,1]*1e3
    
    if ax is None:
        fig,ax = plt.subplots()
    ax.plot(T,S21data[:,2],label='$Q$')
    ax.plot(T,S21data[:,3],label='$Q_c$')
    ax.plot(T,S21data[:,4],label='$Q_i$')
    ax.set_yscale('log')
    ax.set_ylabel('Q-factor')
    ax.set_xlabel('Temperature (mK)')
    ax.legend()

def f0(Chipnum,KIDnum,Pread=None,ax=None):
    '''Plots resonance frequency over temperature'''
    S21data = io.get_S21data(Chipnum,KIDnum,Pread)
    T = S21data[:,1]*1e3
    
    if ax is None:
        fig,ax = plt.subplots()
    ax.plot(T,S21data[:,5]*1e-9)
    ax.set_xlabel('Temperature (mK)')
    ax.set_ylabel('$f_0$ (GHz)')
    ax.ticklabel_format(useOffset=False)

def Qfactorsandf0(Chipnum,KIDnum,Pread=None,fig=None,ax12=None):
    '''Plots both Qfactors and resonance frequency over temperature in one figure'''
    if ax12 is None or fig is None:
        fig,ax12 = plt.subplots(1,2,figsize=(9,3))
    fig.suptitle('{}, KID{}, -{} dBm'.format(Chipnum,KIDnum,Pread))
    Qfactors(Chipnum,KIDnum,Pread,ax=ax12[0])
    f0(Chipnum,KIDnum,Pread,ax=ax12[1])
    fig.tight_layout(rect=(0,0,1,.9))
    
def Powers(Chipnum,KIDnum,Pread=None,ax=None):
    '''Plots the read power, internal power and absorbed power over temperature in one figure'''
    S21data = io.get_S21data(Chipnum,KIDnum,Pread)

    if ax is None:
        fig,ax = plt.subplots()
        ax.set_title('{}, KID{}, {} dBm'.format(Chipnum,KIDnum,S21data[0,7]))
    Q = S21data[:,2]
    Qc = S21data[:,3]
    Qi = S21data[:,4]
    T = S21data[:,1]*1e3
    ax.plot(T,S21data[:,7],label='$P_{read}$')
    ax.plot(T,S21data[:,8],label='$P_{int}$')
    ax.plot(T,10*np.log10(10**(S21data[0,7]/10)/2*4*Q**2/(Qi*Qc)),label='$P_{abs}$')
    ax.set_ylabel('Power (dBm)')
    ax.set_xlabel('Temperature (mK)')
    ax.legend()

def PowersvsT(Chipnum,KIDnum,density=False,phnum=False,
              fig=None,axs=None):
    '''Plots the internal and absorbed powers over temperature, for different read powers (color). 
    Options:
    Density -- the powers are devided by the superconductor volume
    phnum -- the powers are expressed in resonator photon occupations'''
    if axs is None or fig is None:
        fig,axs = plt.subplots(1,2,figsize=(10,4))
    Preadar = io.get_S21Pread(Chipnum,KIDnum)
    cmap = matplotlib.cm.get_cmap('plasma')
    norm = matplotlib.colors.Normalize(-1.05*Preadar.max(),-.95*Preadar.min())
    clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap))
    clb.ax.set_title(r'$P_{read}$ (dBm)')
    for Pread in Preadar:
        S21data = io.get_S21data(Chipnum,KIDnum,Pread)
        Q = S21data[:,2]
        Qc = S21data[:,3]
        Qi = S21data[:,4]
        T = S21data[:,1]*1e3
        Pabs = 10**(S21data[0,7]/10)/2*4*Q**2/(Qi*Qc)*1e-3/1.602e-19
        Pint = 10**(S21data[:,8]/10)*1e-3/1.602e-19
        if phnum:
            Pabs /= 2*np.pi*6.582e-4*(S21data[:,5]*1e-6)**2
            Pint /= 2*np.pi*6.582e-4*(S21data[:,5]*1e-6)**2
            
        if density:
            Pabs /= S21data[0,14]
            Pint /= S21data[0,14]
        axs[1].plot(T,Pabs,color=cmap(norm(-1*Pread)))
        axs[0].plot(T,Pint,color=cmap(norm(-1*Pread)))
        
    title0 = 'Internal Power'
    title1 = 'Absorbed Power'
    if phnum:
        ylabel = '$N_{\gamma}^{res}$'
    else:
        ylabel = '$eV~s^{-1}$'
        
    if density:
        ylabel += ' $\mu m^{-3}$'
        title0 += ' Density'
        title1 += ' Density'

    axs[0].set_title(title0)
    axs[1].set_title(title1)
    axs[0].set_ylabel(ylabel)
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    return fig,axs
    
    
def Nphres(Chipnum,KIDnum,Pread=None,ax=None,label=None):
    '''Plots the number of resonator photons in the resonator over temperature.'''
    S21data = io.get_S21data(Chipnum,KIDnum,Pread)

    if ax is None:
        fig,ax = plt.subplots()
        ax.set_title(f'{Chipnum}, KID{KIDnum}, {S21data[0,7]} dBm')
    
    
    T = S21data[:,1]*1e3
    Pint = S21data[:,8] #dBm
    hbar = 1.055e-25 #mJ µs
    w = 2*np.pi*S21data[:,5]*1e-6 #1/µs
    Nphres = 2*np.pi*10**(Pint/10)/(hbar*w**2)

    ax.plot(T,Nphres,label=label)
    ax.set_xlabel('Temperature (mK)')
    ax.set_ylabel('Number of photons')
    ax.set_yscale('log')
    
def Nphabsres(Chipnum,KIDnum,Pread=None,ax=None,label=None):
    '''Plots the number of absorbed resonator photons over temperature.'''
    S21data = io.get_S21data(Chipnum,KIDnum,Pread)
    if ax is None:
        fig,ax = plt.subplots()
        ax.set_title(f'{Chipnum}, KID{KIDnum}, {S21data[0,7]} dBm')
    
    Q = S21data[:,2]
    Qc = S21data[:,3]
    Qi = S21data[:,4]
    T = S21data[:,1]*1e3
    Pabs = 10*np.log(10**(S21data[0,7]/10)/2*4*Q**2/(Qi*Qc))
    
    hbar = 1.055e-25 #mJ µs
    w = 2*np.pi*S21data[:,5]*1e-6 #1/µs
    Nphabsres = 2*np.pi*10**(Pabs/10)/(hbar*w**2)*S21data[0,14]

    ax.plot(T,Nphabsres,label=label)
#     ax.plot(T,Pabs)
    ax.set_xlabel('Temperature (mK)')
    ax.set_ylabel('Absorbed photons per cycle')
    ax.set_yscale('log')