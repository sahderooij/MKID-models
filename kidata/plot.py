import numpy as np
import warnings 
import matplotlib.pyplot as plt

import matplotlib
from scipy import interpolate
import scipy.io

from kidata import io
from kidata import calc
from kidata import filters

def spec(Chipnum,KIDlist=None,Pread='all',spec=['cross'],lvlcomp='',clbar=True,
              del1fNoise=False,delampNoise=False,del1fnNoise=False,suboffres=False,
              Tmin=0,Tmax=500,ax12=None,
             xlim=(None,None),ylim=(None,None)):
    TDparam = io.get_grTDparam(Chipnum)
    if suboffres:
        TDparamoffres = io.get_grTDparam(Chipnum,offres=True)

    if KIDlist is None:
        KIDlist = io.get_grKIDs(TDparam)
    
    if spec == 'all':
        specs = ['cross','amp','phase']
    elif type(spec) == list:
        specs = spec
    else:
        raise ValueError('Invalid Spectrum Selection')
        
    for KIDnum in KIDlist:
        allPread = io.get_grPread(TDparam,KIDnum)
        if Pread == 'min':
            Preadar = np.array([allPread.max()])
        elif Pread == 'max':
            Preadar = np.array([allPread.min()])
        elif Pread == 'med':
            Preadarr = io.get_grPread(TDparam,KIDnum)
            Preadar = np.array([Preadarr[np.abs(Preadarr.mean()-Preadarr).argmin()]])
        elif Pread == 'minmax':
            Preadar = [allPread.max(),
                       allPread.min()]
        elif Pread == 'all':
            Preadar = allPread
        elif type(Pread) == np.ndarray:
            Preadar = Pread
        elif type(Pread) == list:
            Preadar = np.array(Pread)
        else:
            raise ValueError('{} is not a valid Pread option'.format(Pread))
        if ax12 is None:
            fig,axs = plt.subplots(len(Preadar),len(specs),
                                   figsize=(6*len(specs),4*len(Preadar)),
                                   sharex=True,sharey=True,squeeze=False)
            fig.suptitle('{}, KID{}'.format(Chipnum,KIDnum))
        else:
            axs = ax12
        
        for ax1,_Pread in zip(range(len(Preadar)),Preadar):
            S21Pread = np.array(io.get_S21Pread(Chipnum,KIDnum))
            closestPread = S21Pread[np.abs(S21Pread - _Pread).argmin()]
            S21data = io.get_S21data(Chipnum,KIDnum,closestPread)
            
            axs[ax1,0].set_ylabel('PSD (dBc/Hz)')
            Temp = io.get_grTemp(TDparam,KIDnum,_Pread)
            if suboffres:
                Temp = np.intersect1d(Temp,io.get_grTemp(TDparamoffres,KIDnum,_Pread))
                
            Temp = Temp[np.logical_and(Temp<Tmax,Temp>Tmin)]
            cmap = matplotlib.cm.get_cmap('viridis')
            norm = matplotlib.colors.Normalize(Temp.min(),Temp.max())
            if clbar:
                clb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),
                                   ax=axs[ax1,-1])
                clb.ax.set_title('T (mK)')
            for i in range(len(Temp)):
                for (ax2,spec) in zip(range(len(specs)),specs):                 
                    freq,SPR = io.get_grdata(TDparam,KIDnum,_Pread,Temp[i],spec=spec)
                    if suboffres:
                        orfreq,orSPR = io.get_grdata(TDparamoffres,KIDnum,_Pread,Temp[i],spec=spec)
                        freq,SPR = filters.subtr_spec(freq,SPR,orfreq,orSPR)
                        
                    if delampNoise:
                        freq,SPR = filters.del_ampNoise(freq,SPR)
                    if del1fNoise:
                        freq,SPR = filters.del_1fNoise(freq,SPR)
                    if del1fnNoise:
                        freq,SPR = filters.del_1fnNoise(freq,SPR)
                        
                    SPR[SPR==-140] = np.nan

                        
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
                        
                    axs[ax1,ax2].plot(freq,SPR,color=cmap(norm(Temp[i])))
                    axs[ax1,ax2].set_xscale('log')
                    axs[ax1,ax2].set_title(spec+ ', -{} dBm'.format(_Pread))
                    axs[-1,ax2].set_xlabel('Freq. (Hz)')
                    axs[-1,ax2].set_xlim(*xlim)
            axs[ax1,0].set_ylim(*ylim)
        if ax12 is None:
            fig.tight_layout(rect=(0,0,1,1-.12/len(Preadar)))
                    
                    
def ltnlvl(Chipnum,KIDlist=None,pltPread='all',spec='cross',Tminmax=None,lvlcomp='',
                delampNoise=False,del1fNoise=False,del1fnNoise=False,suboffres=False,relerrthrs=.3,
                pltKIDsep=True,pltthlvl=False,pltkaplan=False,pltthmfnl=False,
                fig=None,ax12=None,color='Pread',fmt='-o',label=None,
                defaulttesc=.14e-3,tescPread='max',tescpltkaplan=False,
                showfit=False,savefig=False):

    def _make_fig():
        plt.rcParams.update({'font.size':12})
        fig, axs = plt.subplots(1,2,figsize=(9,3))
        return fig,axs
    def _get_cmap(**kwargs):
        if color == 'Pread':
            cmap = matplotlib.cm.get_cmap('plasma')
            norm = matplotlib.colors.Normalize(-1.1*Preadar.max(),-.9*Preadar.min())
            clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap))
            clb.ax.set_title(r'$P_{read}$')
        elif color == 'Pint':
            cmap = matplotlib.cm.get_cmap('plasma')
            norm = matplotlib.colors.Normalize(np.array(Pintdict[KIDlist[k]]).min()*1.1,
                                               np.array(Pintdict[KIDlist[k]]).max()*.9)
            clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap))
            clb.ax.set_title(r'$P_{int}$ (dBm)')
        elif color == 'V':
            cmap = matplotlib.cm.get_cmap('cividis')
            norm = matplotlib.colors.Normalize(
                np.array(list(Vdict.values())).min()/.04/.6,
                np.array(list(Vdict.values())).max()/.04/.6)
            clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap))
            clb.ax.set_title(r'Al l. (µm)')
        elif color == 'KIDnum':
            cmap = matplotlib.cm.get_cmap('gist_rainbow')
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
    elif type(KIDlist) is float:
        KIDlist = [KIDlist]
        
    if color == 'Pint':
        Pintdict = io.get_Pintdict(Chipnum)
    elif color == 'V':
        Vdict = io.get_Vdict(Chipnum)
        
    if not pltKIDsep and ax12 is None:
        fig,axs = _make_fig()
    elif not pltKIDsep:
        axs = ax12
        
    for k in range(len(KIDlist)):
        if pltPread == 'min':
            Preadar = np.array([io.get_grPread(TDparam,KIDlist[k]).max()])
        elif type(pltPread) == int:
            Preadar = np.array([np.sort(io.get_grPread(TDparam,KIDlist[k]))[pltPread]])
        elif pltPread == 'med':
            Preadarr = io.get_grPread(TDparam,KIDlist[k])
            Preadar = np.array([Preadarr[np.abs(Preadarr.mean()-Preadarr).argmin()]])
        elif pltPread == 'max':
            Preadar = np.array([io.get_grPread(TDparam,KIDlist[k]).min()])
        elif pltPread == 'minmax':
            Preadar = np.array([io.get_grPread(TDparam,KIDlist[k]).max(),
                                io.get_grPread(TDparam,KIDlist[k]).min()])
        elif pltPread == 'minmedmax':
            Preadarr = io.get_grPread(TDparam,KIDlist[k])
            Preadar = np.array([io.get_grPread(TDparam,KIDlist[k]).max(),
                                Preadarr[np.abs(Preadarr.mean()-Preadarr).argmin()],
                                io.get_grPread(TDparam,KIDlist[k]).min()])
        elif pltPread == 'all':
            Preadar = io.get_grPread(TDparam,KIDlist[k])[::-1]
        elif type(pltPread) == np.ndarray:
            Preadar = pltPread
        elif type(pltPread) == list:
            Preadar = np.array(pltPread)
        else:
            raise ValueError('{} is not a valid Pread selection'.format(pltPread))
            
        if pltKIDsep and ax12 is None:
            fig,axs = _make_fig()
            cmap,norm = _get_cmap(Preadar=Preadar)
            if len(KIDlist) > 1:
                fig.suptitle('KID{}'.format(KIDlist[k]))
        elif pltKIDsep:
            axs=ax12        

        if pltthlvl or 'tesc' in lvlcomp or pltkaplan:
            tesc_ = calc.tesc(Chipnum,KIDlist[k],
                         defaulttesc=defaulttesc,
                         relerrthrs=relerrthrs,Pread=tescPread,pltkaplan=tescpltkaplan)
    
        for Pread in Preadar:
            S21Pread = np.array(io.get_S21Pread(Chipnum,KIDlist[k]))
            closestPread = S21Pread[np.abs(S21Pread - Pread).argmin()]
            S21data = io.get_S21data(Chipnum,KIDlist[k],closestPread)
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
                Vsc_ = Vsc(kbTc,1.72e4,37312.0)
                sqrtlvlcompspl = interpolate.splrep(
                    S21data[:,1]*1e3,
                    interpolate.splev(S21data[:,1]*1e3,
                                      Respspl)*\
                    np.sqrt(V*(1+tesc_/.28e-3)*\
                            (kbTc)**3/\
                            (D(86.17*S21data[:,1],1.72e4,Vsc_,37312.))**2),s=0)
            elif lvlcomp == '':
                sqrtlvlcompspl = interpolate.splrep(
                    S21data[:,1]*1e3,np.ones(len(S21data[:,1])))
            elif lvlcomp == 'RespLowT':
                sqrtlvlcompspl = interpolate.splrep(
                    S21data[:,1]*1e3,np.ones(len(S21data[:,1]))*\
                interpolate.splev(S21data[0,1]*1e3,Respspl))
            else:
                raise ValueError('{} is an invalid compensation method'.format(
                    lvlcomp))
            
            Pint = 10*np.log10(10**(-1*Pread/10)*S21data[0,2]**2/S21data[0,3]/np.pi)
            Temp = np.trim_zeros(io.get_grTemp(TDparam,KIDlist[k],Pread))
            if suboffres:
                Temp = np.intersect1d(Temp,io.get_grTemp(TDparamoffres,KIDlist[k],Pread))
                
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
                freq,SPR = io.get_grdata(TDparam,KIDlist[k],Pread,Temp[i],spec)
                if suboffres:
                    orfreq,orSPR = io.get_grdata(TDparamoffres,KIDlist[k],Pread,Temp[i],spec)
                    freq,SPR = filters.subtr_spec(freq,SPR,orfreq,orSPR)
                if delampNoise:
                    freq,SPR = filters.del_ampNoise(freq,SPR)
                if del1fNoise:
                    freq,SPR = filters.del_1fNoise(freq,SPR)
                if del1fnNoise:
                    freq,SPR = filters.del_1fnNoise(freq,SPR)

                    
                taut[i],tauterr[i],lvl[i],lvlerr[i] = \
                    calc.tau(freq,SPR,plot=showfit,retfnl=True)
                if showfit:
                    plt.title('{}, KID{}, -{} dBm, T={}, {},\n relerr={}'.format(
                        Chipnum,KIDlist[k],Pread,Temp[i],spec,tauterr[i]/taut[i]))
                    plt.xlim(1e-1,1e5)
                    plt.ylim(-120,-60)
                    plt.show()
                    plt.close()
                                
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
                clr = cmap(norm(Vdict[KIDlist[k]]/.04/.6))
            elif color == 'KIDnum':
                clr = cmap(norm(KIDlist[k]))
            else:
                clr = color
            
            axs[0].errorbar(Temp[mask],taut[mask],
                                     yerr = tauterr[mask],fmt = fmt,capsize = 3.,
                              color=clr,mec='k',label=label)
            axs[1].errorbar(Temp[mask],10*np.log10(lvl[mask]),
                                     yerr = 10*np.log10((lvlerr[mask]+lvl[mask])/lvl[mask]),
                                     fmt = fmt, capsize = 3.,color=clr,mec='k',
                           label=label)
            if pltthlvl:
                Ttemp = np.linspace(Temp[mask].min(),Temp[mask].max(),100)
                explvl = interpolate.splev(Ttemp,Respspl)**2
                explvl *= 4*1e-6*.44*S21data[0,14]*1.72e4*(86.17*S21data[0,21])**3/\
                (2*(S21data[0,15]/1.602e-19*1e6)**2)*(1+tesc_/.28e-3)/2
                explvl /= interpolate.splev(Ttemp,sqrtlvlcompspl)**2
                thlvlplot, = axs[1].plot(Ttemp,10*np.log10(explvl),color=clr,linestyle='--',
                                        label='Theoretical FNL')
                axs[1].legend((thlvlplot,),(r'FNL from responsivity',))
                
            if pltkaplan and Temp[mask].size != 0:
                T = np.linspace(Temp[mask].min(),Temp[mask].max(),100)*1e-3
                taukaplan = calc.tau_kaplan(T,tesc=tesc_,kbTc=86.17*S21data[0,21])
                kaplanfit, = axs[0].plot(T*1e3,taukaplan,color='k',linestyle='-',linewidth=1.)
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
                    axs[1].legend((thmfnlplot,),('Thermal FNL \n with measured $\\tau_{qp}^*$',))
                except:
                    warnings.warn('Could not make Thermal FNL, {},KID{},-{} dBm,{}'.format(
                    Chipnum,KIDlist[k],Pread,spec))
                    
        axs[0].set_yscale('log')
        for i in range(2):
            axs[i].set_xlabel('Temp. (mK)')
        axs[0].set_ylabel(r'$\tau_{qp}^*$ (µs)')
        
        if lvlcomp == 'Resp':
            axs[1].set_ylabel(r'FNL/$\mathcal{R}^2(T)$ (dB/Hz)')
        elif lvlcomp == 'RespV':            
            axs[1].set_ylabel(r'FNL/$(\mathcal{R}^2(T)V)$ (dB/Hz)')
        elif lvlcomp == 'RespVtescTc':   
            axs[1].set_ylabel(r'FNL/$(\mathcal{R}^2(T)\chi)$ (dB/Hz)')   
        elif lvlcomp == '':
            axs[1].set_ylabel(r'FNL (dB/Hz)')
        elif lvlcomp == 'RespLowT':
            axs[1].set_ylabel(r'FNL/$\mathcal{R}^2(T=50 mK)$ (dB/Hz)')
        else:
            axs[1].set_ylabel(r'comp. FNL (dB/Hz)')
        plt.tight_layout()
        if savefig:
            plt.savefig('GR_{}_KID{}_{}.pdf'.format(Chipnum,KIDlist[k],spec))
            plt.close()
    if ax12 is None:
        return fig,axs
            
def rejspec(Chipnum,KIDnum,sigma,Trange = (0,400),sepT = False, spec='SPR'):
    dfld = io.get_datafld()
    for j in range(len(sigma)):
        #Load matfile and extract Pread en temperature
        matfl = scipy.io.loadmat(
            dfld + '\\'+ Chipnum + '\\Noise_vs_T' + '\\TDresults_{}'.format(sigma[j])
        )['TDparam']
        if Chipnum == 'LT139':
            if KIDnum == 1:
                ind = 0
            elif KIDnum == 6:
                ind = 2
            elif KIDnum == 7:
                ind = 3
            else:
                raise ValueError('KID{} is not in LT139'.format(KIDnum))
        else:
            ind = KIDnum - 1 
        Preadar = matfl['Pread'][0,ind][:,1]
        Pread = Preadar[-1]
        Pind = np.where(Preadar == Pread)
        Tar = matfl['Temp'][0,ind][Pind,:][0,0]
        Tmin,Tmax = Trange
        Tar=Tar[np.logical_and(Tar>Tmin,Tar<Tmax)]
        Tar=np.flip(Tar) #To plot the lowest temperatures last        

        #Plot spectrum from each temperature
        if not sepT:
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            cmap = matplotlib.cm.get_cmap('viridis')
            norm = matplotlib.colors.Normalize(Tar.min(),Tar.max())
            percrej = np.zeros(len(Tar))
        for i in range(len(Tar)):
            Tind = np.where(np.flip(Tar) == Tar[i])
            freq = matfl['fmtotal'][0,ind][Pind,Tind][0,0][0]
            if spec == 'SPR':
                SPR = matfl['SPRrealneg'][0,ind][Pind,Tind][0,0][0]
            elif spec == 'SPP':
                SPR = matfl['SPPtotal'][0,ind][Pind,Tind][0,0][0]
            elif spec == 'SRR':
                SPR = matfl['SRRtotal'][0,ind][Pind,Tind][0,0][0]
            else:
                raise ValueError('Choose SPR, SPP or SRR as spec')
            SPR[SPR==-140] = np.nan #delete -104 dBm, because that is minimum.
            if sepT:
                plt.figure(np.floor(i/2),figsize=(10,4))
                plt.subplot(1,2,i%2+1)
                plt.title('KID{}, -{} dBm, {} mK'.format(KIDnum,Pread,Tar[i]))
                plt.plot(freq,SPR)
                plt.legend(sigma,title=r'Thrshld ($\sigma$)')
            else:
                plt.plot(freq,SPR,color=cmap(norm(Tar[i])))
                percrej[i] = 100*matfl['nrrejectedmed'][0,ind][Pind,Tind][0,0]/32
            plt.xscale('log')
        if not sepT:
            plt.title(r'KID{}, $P_{{read}}=$-{} dBm, {}$\sigma$'.format(
                KIDnum,Pread,sigma[j]))
            plt.ylim(-130,-50)
            clb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap))
            clb.ax.set_title('T (mK)')
            plt.subplot(1,2,2)
            plt.title('Rejection Percentage')
            plt.plot(Tar,percrej)
            plt.xlabel('T (mK)')
            plt.ylabel('%')
            plt.ylim(0,100)
            plt.tight_layout()
            
def Qif0(Chipnum,KIDnum,color='Pread',Tmax=.35,pltPread='all'):
    dfld = io.get_datafld()
    fig,axs = plt.subplots(1,2,figsize=(6.2,2.1))
    plt.rcParams.update({'font.size':7})
    if pltPread == 'all':
        Preadar = np.array(io.get_S21Pread(Chipnum,KIDnum))
    elif type(pltPread) == tuple:
        Preadar = np.array(io.get_S21Pread(Chipnum,KIDnum))[pltPread[0]:pltPread[1]]
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
        axs[1].plot(T[T<Tmax]*1e3,S21data[T<Tmax,5]*1e-9,color=clr)
        
    for ax in axs:
        ax.set_xlabel('T (mK)')
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
    axs[0].set_ylabel('$Q_i$')
    axs[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axs[1].set_ylabel('$f_0$ (GHz)')
    fig.tight_layout()

def Nqp(Chipnum,KIDnum,pltPread='all',spec='cross',
             pltThrm=True,pltNqpQi=False,splitT=0,pltNqptau=False,tescPread='max',
             relerrthrs=.4,
             fig=None,ax=None,
            N0 = 1.72e4,
            kbTD = 37312.0,
            kb = 86.17):
    TDparam = io.get_grTDparam(Chipnum)
    if ax is None or fig is None:
        fig,ax = plt.subplots()
        plt.rcParams.update({'font.size':10})

    if pltPread == 'all':
        Preadar = io.get_grPread(TDparam,KIDnum)[::-1]
    elif pltPread == 'minmax':
        Preadar = np.array([io.get_grPread(TDparam,KIDnum).max(),
                            io.get_grPread(TDparam,KIDnum).min()])
    elif pltPread == 'min':
        Preadar = np.array([io.get_grPread(TDparam,KIDnum).max()])
    elif pltPread == 'max':
        Preadar = np.array([io.get_grPread(TDparam,KIDnum).min()])
    elif type(pltPread) == np.ndarray:
        Preadar = pltPread
    elif type(pltPread) == list:
        Preadar = np.array(pltPread)
    else:
        raise ValueError('{} is not a valid Pread selection'.format(pltPread))
        
    if Preadar.size > 1:
        cmap = matplotlib.cm.get_cmap('plasma')
        norm = matplotlib.colors.Normalize(-1.05*Preadar.max(),-.95*Preadar.min())
        clb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),
                           ax=ax)
        clb.ax.set_title(r'$P_{read}$ (dBm)')
        
    for Pread in Preadar:
        S21Pread = np.array(io.get_S21Pread(Chipnum,KIDnum))
        closestPread = S21Pread[np.abs(S21Pread - Pread).argmin()]
        S21data = io.get_S21data(Chipnum,KIDnum,closestPread)
        Vsc_ = Vsc(S21data[0,21]*kb,N0,kbTD)
        if closestPread != Pread:
            warnings.warn('S21data at another Pread')
            
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
        Nqp,Nqperr,taut = np.zeros((3,len(Temp)))
        for i in range(len(Temp)):
            freq,SPR = io.get_grdata(TDparam,KIDnum,Pread,Temp[i],spec=spec)
            freq,SPR = filters.del_otherNoise(freq,SPR)
            taut[i],tauterr,lvl,lvlerr = \
                            calc.tau(freq,SPR,retfnl = True)
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
            
        ax.errorbar(Temp[mask],Nqp[mask]/S21data[0,14],yerr=Nqperr[mask]/S21data[0,14],
                    color=clr,marker='o',mec='k',capsize=2.)
        if pltNqpQi:
            T,Nqp = calc.NqpfromQi(S21data)
            mask = np.logical_and(T*1e3>ax.get_xlim()[0],T*1e3<ax.get_xlim()[1])
            totalT = T[mask]
            totalNqp = Nqp[mask]
            ax.plot(totalT[totalT>splitT]*1e3,totalNqp[totalT>splitT]/S21data[0,14],
                    'g-',zorder=len(ax.lines)+1,label='$n_{qp}$ from $Q_i$')
            ax.plot(totalT[totalT<splitT]*1e3,totalNqp[totalT<splitT]/S21data[0,14],
                    'g--',zorder=len(ax.lines)+1)
        if pltNqptau:
            nqp_ = calc.nqpfromtau(taut,Chipnum,KIDnum,tescPread=tescPread)
            ax.plot(Temp[mask],nqp_[mask],
                   color=clr,zorder=len(ax.lines)+1,
                    label='$n_{qp}$ from $\\tau_{qp}^*$')
    if pltThrm:
        T = np.linspace(*ax.get_xlim(),100)
        nqpT = np.zeros(100)
        for i in range(len(T)):
            D_ = D(kb*T[i]*1e-3, N0, Vsc_, kbTD)
            nqpT[i] = nqp(kb*T[i]*1e-3, D_, N0)
        ax.plot(T,nqpT,color='k',zorder=len(ax.lines)+1,label='Thermal $n_{qp}$')
        ax.set_ylabel('$n_{qp}$ ($\mu m^{-3}$)')
        ax.set_xlabel('T (mK)')
    if pltThrm or pltNqpQi or pltNqptau:
        ax.legend()
    ax.set_yscale('log')    
    
def Qfactors(Chipnum,KIDnum,Pread=None,ax=None):
    if Pread is None:
        Pread = io.get_S21Pread(Chipnum,KIDnum)[0]
    S21data = io.get_S21data(Chipnum,KIDnum,Pread)
    T = S21data[:,1]*1e3
    
    if ax is None:
        fig,ax = plt.subplots()
    ax.plot(T,S21data[:,2],label='$Q$')
    ax.plot(T,S21data[:,3],label='$Q_c$')
    ax.plot(T,S21data[:,4],label='$Q_i$')
    ax.set_yscale('log')
    ax.set_ylabel('Q-factor')
    ax.set_xlabel('Temp. (mK)')
    ax.legend()

def f0(Chipnum,KIDnum,Pread,ax=None):
    S21data = io.get_S21data(Chipnum,KIDnum,Pread)
    T = S21data[:,1]*1e3
    
    if ax is None:
        fig,ax = plt.subplots()
    ax.plot(T,S21data[:,5]*1e-9)
    ax.set_xlabel('Temp. (mK)')
    ax.set_ylabel('$f_0$ (GHz)')
    ax.ticklabel_format(useOffset=False)

def Qfactorsandf0(Chipnum,KIDnum,Pread=None,fig=None,ax12=None):
    if Pread is None:
        Pread = io.get_S21Pread(Chipnum,KIDnum)[0]
    if ax12 is None or fig is None:
        fig,ax12 = plt.subplots(1,2,figsize=(9,3))
    fig.suptitle('{}, KID{}, -{} dBm'.format(Chipnum,KIDnum,Pread))
    Qfactors(Chipnum,KIDnum,Pread,ax=ax12[0])
    f0(Chipnum,KIDnum,Pread,ax=ax12[1])
    fig.tight_layout(rect=(0,0,1,.9))
    
def Powers(Chipnum,KIDnum,Pread=None,ax=None):
    if Pread is None:
        Pread = io.get_S21Pread(Chipnum,KIDnum)[0]
    if ax is None:
        fig,ax = plt.subplots()
    ax.set_title('{}, KID{}, -{} dBm'.format(Chipnum,KIDnum,Pread))
    S21data = io.get_S21data(Chipnum,KIDnum,Pread)
    Q = S21data[:,2]
    Qc = S21data[:,3]
    Qi = S21data[:,4]
    T = S21data[:,1]*1e3
    ax.plot(T,S21data[:,7],label='$P_{read}$')
    ax.plot(T,S21data[:,8],label='$P_{int}$')
    ax.plot(T,10*np.log10(10**(-Pread/10)/2*4*Q**2/(Qi*Qc)),label='$P_{abs}$')
    ax.set_ylabel('Power (dBm)')
    ax.set_xlabel('Temperature (mK)')
    ax.legend()
    fig.tight_layout()