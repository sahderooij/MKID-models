import numpy as np
import datacalc
import kidcalc
import scipy.integrate as integrate
from scipy import interpolate
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib
import warnings

def get_KIDparam(Chipnum,KIDnum,Pread):
    kbTD = 37312.
    N0 = 1.72e4
    S21data = datacalc.get_S21data(Chipnum,KIDnum,Pread)
    V = S21data[0,14]
    kbTc = S21data[0,21]*86.17
    D0 = 1.76*kbTc
    def integrand1(E, D):
                return 1/np.sqrt(E**2-D**2)
    Vsc = 1/(integrate.quad(integrand1, D0, kbTD,
                                     args=(D0,))[0]*N0)
    tesc = datacalc.tesc(Chipnum,KIDnum,Pread)
    return V,kbTc,D0,Vsc,tesc


#####################################################################################
##################################### Models ########################################
#####################################################################################

class Model:
    def __init__(self,V,kbTc,D0,Vsc,tesc):
        self.V = V
        self.kbTc = kbTc
        self.D0 = D0
        self.Vsc = Vsc
        self.tesc = tesc
        #Standard constants
        self.N0 = 1.72e4
        self.t0 = 440e-3
        self.tpb = .28e-3
        self.kbTD = 37312.
        self.kb = 86.17
    
    def calc_spec(self,lvlcal,*args,startf=1,stopf=1e6,points=100,
                  retnumrates=False,PSDs='NN'):
        if retnumrates:
            M,B,num,rates = self.calc_MB(*args,retnumrates=True)
        else:
            M,B = self.calc_MB(*args)
        frqarr = np.logspace(np.log10(startf),np.log10(stopf),points)
        warr = 2*np.pi*frqarr*1e-6 #note the conversion to µs^-1
        sw = np.zeros(len(warr))
        for j in range(len(warr)):
            Gw = 2*np.real(np.linalg.multi_dot([
                np.linalg.inv(M+1j*warr[j]*np.eye(len(M))),
                B,
                np.linalg.inv(M.transpose()-1j*warr[j]*np.eye(len(M)))]))
            if PSDs == 'NN+NNt':
                sw[j] = (Gw[0,0] + Gw[1,0])
            elif PSDs is 'NN':
                sw[j] = Gw[0,0]
            elif PSDs is 'NtNt':
                sw[j] = Gw[1,1]
            elif PSDs is 'NNt':
                sw[j] = Gw[0,1]
            elif PSDs is 'NCNC':
                sw[j] = (Gw[0,0] + Gw[1,0] + Gw[0,1] + Gw[1,1])
            else:
                raise ValueError('{} is not a valid PSDs value'.format(PSDs))
        swdB = 10*np.log10(sw/lvlcal)
        if retnumrates:
            return frqarr,swdB,num,rates
        else:
            return frqarr,swdB
    
    def calc_ltnlvl(self,Tmin,Tmax,
                    lvlcal,*args,points=50,
                    plotspec=False,plotnumrates=False,PSDs='NN'):
        tau = np.full(points,np.nan)
        tauerr = np.full(points,np.nan)
        lvl = np.full(points,np.nan)
        lvlerr = np.full(points,np.nan)
        Temp = np.linspace(Tmin,Tmax,points)

        cmap = matplotlib.cm.get_cmap('viridis')
        norm = matplotlib.colors.Normalize(vmin=Temp.min(),vmax=Temp.max())
        for i in range(len(Temp)):
            if plotnumrates:
                freq,swdB,nums,rates = self.calc_spec(lvlcal,*args,Temp[i],retnumrates=True,
                                                     PSDs=PSDs)
                plt.figure('Nums')
                numcol = ['b','g','r','c','m','y','k']
                for val in nums.values():
                    plt.plot(Temp[i],val,numcol.pop(0)+'.')
                
                plt.figure('Rates')
                ratecol = ['b','g','r','c','m','y','k']
                for val in rates.values():
                    plt.plot(Temp[i],val,ratecol.pop(0)+'.')
                
            else:
                freq,swdB = self.calc_spec(lvlcal,*args,Temp[i],PSDs=PSDs)

            tau[i],tauerr[i],lvl[i],lvlerr[i] = datacalc.tau(freq,swdB,
                                                             plot=False,retfnl=True)
            if plotspec:
                plt.figure('Spectra')
                plt.plot(freq,swdB,color=cmap(norm(Temp[i])))
        if plotspec:
            plt.figure('Spectra')
            plt.xscale('log')
            plt.ylabel('Noise level (dBc/Hz)')
            plt.xlabel('Frequency (Hz)')
            clb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap))
            clb.ax.set_title('T (mK)')
            
        if plotnumrates:
            plt.figure('Nums')
            plt.title('Numbers')
            plt.yscale('log')
            plt.xlabel('Temperature (K)')
            plt.ylabel('Number')
            plt.legend(list(nums.keys()))
            plt.figure('Rates')
            plt.title('Rates')
            plt.yscale('log')
            plt.legend(list(rates.keys()))
            plt.ylabel(r'Rate ($\mu s^{-1}$)')
            plt.xlabel('Temperature (K)')
        return Temp,tau,tauerr,lvl,lvlerr
    
    def calc_phtresp(self,epb,wvl,*args,tStop=2.5e3,points=100,plot=False):
        t = np.linspace(0,tStop,points)
        params,*rest = self.calc_params(*args)
        Nss = root(self.rateeq,np.ones(self.nrRateEqs),args=(t,params),
                   jac=self.jac,method='hybr').x
        Nini = Nss.copy()
        Nini[0]+= epb*(6.582e-4*2*np.pi*3e8/(wvl*1e-3))/params[-1]
        Nevol = integrate.odeint(self.rateeq,Nini,t,args=(params,))
        if plot:
            plt.plot(t,Nevol[:,0]-Nss[0])
            plt.yscale('log')
            plt.xlabel('t (µs)')
            plt.ylabel('Nqp - Nqp0')
            plt.title('Singel Photon ({} nm) Response'.format(wvl))
        return t,Nevol,Nss
    
    #plot fuctions
    def plot_ltlvl(self,T,tau,tauerr,lvl,lvlerr,ax1=None,ax2=None,color='b',fmt=''):
        if ax1 is None or ax2 is None:
            fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
        mask = np.logical_and(lvl/lvlerr > 2,tau/tauerr > 2)
        ax1.set_title('Lifetime')
        ax1.errorbar(T[mask]*1e3,tau[mask],yerr=tauerr[mask],color=color,fmt=fmt)
        ax1.set_yscale('log')
        ax1.set_ylabel(r'$\tau$ $(\mu s)$')
        ax1.set_xlabel(r'T (mK)')
        ax2.set_title('Flat Noise Level')
        ax2.errorbar(T[mask]*1e3,10*np.log10(lvl[mask]),
                     yerr=10*np.log10((lvlerr[mask]+lvl[mask])/lvl[mask]),color=color,fmt=fmt)
        ax2.set_ylabel('Noise level (dBc/Hz)')
        ax2.set_xlabel(r'T (mK)')

        
####################################Trap Detrap############################################

class TrapDetrap(Model):
    def __init__(self,V,kbTc,D0,Vsc,tesc):
        self.nrRateEqs = 3
        super().__init__(V,kbTc,D0,Vsc,tesc)
    #Model equations
    def rateeq(self,Ns,t,params):
        N,Nt,Nw = Ns
        R,V,GB,Gt,Gd,Ges,NTw,eta,P,D = params
        return [-R*N**2/V + 2*GB*Nw - Gt*N + Gd*Nt + eta*P/D,
                Gt*N-Gd*Nt,
                R*N**2/(2*V) - GB*Nw - Ges*(Nw-NTw)]
    
    def jac(self,Ns,t,params):
        N,Nt,Nw = Ns
        R,V,GB,Gt,Gd,Ges,NTw,eta,P,D = params
        return [[-2*R*N/V-Gt, Gd,2*GB],
                [Gt,-Gd,0],
                [R*N/V,0,-GB-Ges]]
    
    def calc_params(self,Gt,Gd,eta,P,T):
        kbT = self.kb*T
        D = kidcalc.D(kbT,self.N0,self.Vsc,self.kbTD)
        NT = kidcalc.nqp(kbT,D,self.N0)*self.V
        R = (2*D/self.kbTc)**3/(4*D*self.N0*self.t0)
        Rstar = R/(1+self.tesc/self.tpb)
        Ges = 1/self.tesc
        GB = 1/self.tpb
        NTw = R*NT**2/(2*self.V*GB)
        return [R,self.V,GB,Gt,Gd,Ges,NTw,eta,P,D],NT,Rstar
    
    def calc_MB(self,Gt,Gd,eta,P,T,retnumrates = False):
        params,NT,Rstar = self.calc_params(Gt,Gd,eta,P,T)
        R,V,GB,Gt,Gd,Ges,NTw,eta,P,D = params
        #Steady state values
        t=0 #dummy var. for rate eq.
        Nqp0,Nt0,Nw0 = root(
            self.rateeq,[NT,NT/2,NTw],args=(t,params),
            tol=1e-12,jac=self.jac,method='hybr').x
        #M and B matrices
        M = np.array([[Gt+2*Rstar*Nqp0/self.V,-Gd],
                      [-Gt,Gd]])
        B = np.array([[Gt*Nqp0+Gd*Nt0+4*Rstar*Nqp0**2/self.V,-Gt*Nqp0-Gd*Nt0],
                      [-Gt*Nqp0-Gd*Nt0,Gt*Nqp0+Gd*Nt0]])  
        if retnumrates:
            return M,B,{'NqpT':NT,
                        'Nqp0':Nqp0,
                        'Nt0':Nt0,
                        'Nw0':Nw0},{'R*NqpT':R*NT,
                                    'R*Nqp0':R*Nqp0,
                                    'Gt':Gt,
                                    'Gd':Gd}
        else:
            return M,B
        
####################################Trap Detrap Ntmax ########################################
class TrapDetrapNtmax(Model):
    def __init__(self,V,kbTc,D0,Vsc,tesc):
        self.nrRateEqs = 3
        super().__init__(V,kbTc,D0,Vsc,tesc)
    #Model equations
    def rateeq(self,Ns,t,params):
        N,Nt,Nw = Ns
        R,V,GB,Gt,Gd,Ntmax,Ges,NTw,eta,P,D = params
        return [-R*N**2/V + 2*GB*Nw - Gt*(Ntmax-Nt)*N + Gd*Nt + eta*P/D,
                Gt*(Ntmax-Nt)*N-Gd*Nt,
                R*N**2/(2*V) - GB*Nw - Ges*(Nw-NTw)]
    
    def jac(self,Ns,t,params):
        N,Nt,Nw = Ns
        R,V,GB,Gt,Gd,Ntmax,Ges,NTw,eta,P,D = params
        return [[-2*R*N/V-Gt*(Ntmax-Nt), Gd + Gt*N,2*GB],
                [Gt*(Ntmax-Nt),-Gd-Gt*N,0],
                [R*N/V,0,-GB-Ges]]
    
    def calc_params(self,Gt,Gd,Ntmax,eta,P,T):
        kbT = self.kb*T
        D = kidcalc.D(kbT,self.N0,self.Vsc,self.kbTD)
        NT = kidcalc.nqp(kbT,D,self.N0)*self.V
        R = (2*D/self.kbTc)**3/(4*D*self.N0*self.t0)
        Rstar = R/(1+self.tesc/self.tpb)
        Ges = 1/self.tesc
        GB = 1/self.tpb
        NTw = R*NT**2/(2*self.V*GB)
        return [R,self.V,GB,Gt,Gd,Ntmax,Ges,NTw,eta,P,D],NT,Rstar
    
    def calc_MB(self,Gt,Gd,Ntmax,eta,P,T,retnumrates = False):
        params,NT,Rstar = self.calc_params(Gt,Gd,Ntmax,eta,P,T)
        R,V,GB,Gt,Gd,Ntmax,Ges,NTw,eta,P,D = params
        #Steady state values
        t=0 #dummy var. for rate eq.
        Nqp0,Nt0,Nw0 = root(
            self.rateeq,[NT,NT/2,NTw],args=(t,params),
            tol=1e-12,jac=self.jac,method='hybr').x
        #M and B matrices
        M = np.array([[Gt*(Ntmax-Nt0)+2*Rstar*Nqp0/self.V,-Gd-Gt*Nqp0],
                      [-Gt*(Ntmax-Nt0),Gd+Gt*Nt0]])
        B = np.array([[Gt*(Ntmax-Nt0)*Nqp0+Gd*Nt0+2*Rstar*(Nqp0**2+NT**2)/self.V,
                       -Gt*(Ntmax-Nt0)*Nqp0-Gd*Nt0],
                      [-Gt*(Ntmax-Nt0)*Nqp0-Gd*Nt0,
                       Gt*(Ntmax-Nt0)*Nqp0+Gd*Nt0]])  
        if retnumrates:
            return M,B,{'Nqp0':Nqp0,'Nt0':Nt0,'Nw0':Nw0},{'R*Nqp0':R*Nqp0,'T*(Ntmax-Nt0)':Gt*(Ntmax-Nt0),'Gd':Gd}
        else:
            return M,B
###################################Trap Detrap Rt##########################################
class TrapDetrapRt(Model):
    def __init__(self,V,kbTc,D0,Vsc,tesc):
        self.nrRateEqs = 3
        super().__init__(V,kbTc,D0,Vsc,tesc)
        
    #Model Equations
    def rateeq(self,Ns,t,params):
        N,Nt,Nw = Ns
        R,V,GB,Gt,Gd,Rt,Ges,NTw,eta,P,D = params
        return [-R*N**2/V-Rt*Nt*N/(2*V) +2*GB*Nw-Gt*N+Gd*Nt+eta*P/D,
                -Rt*Nt*N/(2*V) +Gt*N-Gd*Nt,
                R*N**2/(2*V) - GB*Nw - Ges*(Nw-NTw)]

    def jac(self,Ns,t,params):
        N,Nt,Nw = Ns
        R,V,GB,Gt,Gd,Rt,Ges,NTw,eta,P,D = params
        return [[-2*R*N/V-Gt-Rt*Nt/(2*V),Gd-Rt*N/(2*V),2*GB],
                [Gt-Rt*Nt/(2*V),-Gd-Rt*N/(2*V),0],
                [R*N/V,0,-GB-Ges]]

    def calc_params(self,Gt,Gd,Rt,eta,P,T):
        kbT = self.kb*T
        D = kidcalc.D(kbT,self.N0,self.Vsc,self.kbTD)
        NT = kidcalc.nqp(kbT,D,self.N0)*self.V
        R = (2*D/self.kbTc)**3/(4*D*self.N0*self.t0)
        Rstar = R/(1+self.tesc/self.tpb)
        Ges = 1/self.tesc
        GB = 1/self.tpb
        NTw = R*NT**2/(2*self.V*GB)
        return [R,self.V,GB,Gt,Gd,Rt,Ges,NTw,eta,P,D],NT,Rstar
    
    def calc_MB(self,*args,retnumrates=False):
        params,NT,Rstar = self.calc_params(*args)
        R,V,GB,Gt,Gd,Rt,Ges,NTw,eta,P,D = params

        #Steady state values
        Nqp0,Nt0,Nw0 = root(
            self.rateeq,[NT,NT/2,NTw],
            args=(0,params),jac=self.jac,method='hybr',
            options = {'factor':2}).x
        if any([i<0 for i in [Nqp0,Nt0,Nw0]]):
            Nqp0,Nt0,Nw0 = np.nan,np.nan,np.nan
            warnings.warn('No root solution found',UserWarning)
        #M and B matrices
        M = np.array([[Gt+2*Rstar*Nqp0/V+Rt*Nt0/(2*V),Rt*Nqp0/(2*V)-Gd],
                      [Rt*Nt0/(2*V)-Gt,Gd+Rt*Nqp0/(2*V)]])
        B = np.array([[Gt*Nqp0+Gd*Nt0+2*Rstar*(Nqp0**2+NT**2)/V+Rt*Nqp0*Nt0/(2*V),
                       -Gt*Nqp0-Gd*Nt0],
                      [-Gt*Nqp0-Gd*Nt0,
                       Gt*Nqp0+Gd*Nt0+Rt*Nqp0*Nt0/(2*V)]])
        if retnumrates:
            return M,B,{
                'NqpT':NT,
                'Nqp0':Nqp0,
                'Nt0':Nt0,
                'Nw0':Nw0},{
                'R*NqpT/2V':R*NT/(2*V),
                'R*Nqp0/2V':R*Nqp0/(2*V),
                'Rt*Nt0/2V':Rt*Nt0/(2*V),
                'Gt':Gt,
                'Gd':Gd}
        else:
            return M,B
####################################Trap Detrap Ntmax Rt ####################################
class TrapDetrapNtmaxRt(Model):
    def __init__(self,V,kbTc,D0,Vsc,tesc):
        self.nrRateEqs = 3
        super().__init__(V,kbTc,D0,Vsc,tesc)
    #Model equations
    def rateeq(self,Ns,t,params):
        N,Nt,Nw = Ns
        R,V,GB,Gt,Gd,Ntmax,Rt,Ges,NTw,eta,P,D = params
        return [-R*N**2/V-Rt*N*Nt/(2*V)+ 2*GB*Nw - Gt*(Ntmax-Nt)*N + Gd*Nt + eta*P/D,
                -Rt*N*Nt/(2*V) + Gt*(Ntmax-Nt)*N - Gd*Nt,
                R*N**2/(2*V) - GB*Nw - Ges*(Nw - NTw) ]
    
    def jac(self,Ns,t,params):
        N,Nt,Nw = Ns
        R,V,GB,Gt,Gd,Ntmax,Rt,Ges,NTw,eta,P,D = params
        return [[-2*R*N/V-Rt*Nt/(2*V)-Gt*(Ntmax-Nt), -Rt*N/(2*V) + Gd + Gt*N, 2*GB],
                [Gt*(Ntmax-Nt)-Rt*Nt/(2*V),-Rt*N/(2*V)-Gd-Gt*N,0],
                [R*N/V ,0,-GB-Ges]]
    
    def calc_params(self,Gt,Gd,Ntmax,Rt,eta,P,T):
        kbT = self.kb*T
        D = kidcalc.D(kbT,self.N0,self.Vsc,self.kbTD)
        NT = kidcalc.nqp(kbT,D,self.N0)*self.V
        R = (2*D/self.kbTc)**3/(4*D*self.N0*self.t0)
        Rstar = R/(1+self.tesc/self.tpb)
        Ges = 1/self.tesc
        GB = 1/self.tpb
        NTw = R*NT**2/(2*self.V*GB)
        return [R,self.V,GB,Gt,Gd,Ntmax,Rt,Ges,NTw,eta,P,D],NT,Rstar
    
    def calc_MB(self,*args,retnumrates = False):
        params,NT,Rstar = self.calc_params(*args)
        R,V,GB,Gt,Gd,Ntmax,Rt,Ges,NTw,eta,P,D = params
        #Steady state values
        t=0 #dummy var. for rate eq.
        Nqp0,Nt0,Nw0 = root(
            self.rateeq,[NT,NT/2,NTw],args=(t,params),
            tol=1e-12,jac=self.jac,method='hybr').x
        #M and B matrices
        
        M = np.array([[Gt*(Ntmax-Nt0)+(2*Rstar*Nqp0+.5*Rt*Nt0)/V,-Gd-Gt*Nqp0-Rt*Nqp0/(2*V)],
                      [-Gt*(Ntmax-Nt0)+Rt*Nt0/(2*V),Gd+Gt*Nqp0+Rt*Nqp0/(2*V)]])
        B = np.array([[Gt*(Ntmax-Nt0)*Nqp0+Gd*Nt0+(4*Rstar*(Nqp0**2+NT**2)+5*Rt*Nt0*Nqp0)/(2*V),
                       -Gt*(Ntmax-Nt0)*Nqp0-Gd*Nt0-Rt*Nqp0*Nt0/(2*V)],
                      [-Gt*(Ntmax-Nt0)*Nqp0-Gd*Nt0-Rt*Nqp0*Nt0/(2*V),
                       Gt*(Ntmax-Nt0)*Nqp0+Gd*Nt0+Rt*Nqp0*Nt0/(2*V)]]) 

#         M = np.array([[Gt*(Ntmax-Nt0)+2*Rstar*Nqp0/V+Rt*Nt0/(2*V),-Gd-Gt*Nqp0+Rt*Nqp0/(2*V)],
#                       [-Gt*(Ntmax-Nt0)-Rt*Nt0/(2*V),Gd+Gt*Nqp0-Rt*Nqp0/(2*V)]])
#         B = np.array([[Gt*(Ntmax-Nt0)*Nqp0+Gd*Nt0+4*Rstar*Nqp0**2/V+Rt*Nqp0*Nt0/(2*V),
#                        -Gt*(Ntmax-Nt0)*Nqp0-Gd*Nt0-Rt*Nqp0*Nt0/(2*V)],
#                       [-Gt*(Ntmax-Nt0)*Nqp0-Gd*Nt0-Rt*Nqp0*Nt0/(2*V),
#                        Gt*(Ntmax-Nt0)*Nqp0+Gd*Nt0+5*Rt*Nqp0*Nt0/(2*V)]]) 
        
#         M = np.array([[Gt*(Ntmax-Nt0)+2*Rstar*Nqp0/V+Rt*Nt0/(2*V),-Gd-Gt*Nqp0+Rt*Nqp0/(2*V)],
#                       [-Gt*(Ntmax-Nt0)+Rt*Nt0/(2*V),Gd+Gt*Nt0+Rt*Nqp0/(2*V)]])
#         B = np.array([[Gt*(Ntmax-Nt0)*Nqp0+Gd*Nt0+4*Rstar*Nqp0**2/V+Rt*Nqp0*Nt0/V,
#                        -Gt*(Ntmax-Nt0)*Nqp0-Gd*Nt0],
#                       [-Gt*(Ntmax-Nt0)*Nqp0-Gd*Nt0,
#                        Gt*(Ntmax-Nt0)*Nqp0+Gd*Nt0+Rt*Nqp0*Nt0/V]]) 
        if retnumrates:
            return M,B,{'NqpT':NT,
                        'Nqp0':Nqp0,
                        'Nt0':Nt0,
                        'Nw0':Nw0},{'R*NqpT/2V':R*NT/(2*V),
                                    'R*Nqp0/2V':R*Nqp0/(2*V),
                                    'Rt*Nt0/2V':Rt*Nt0/(2*V),
                                    'T*(Ntmax-Nt0)':Gt*(Ntmax-Nt0),
                                    'Gd':Gd}
        else:
            return M,B
####################################Trap Detrap Ntmax Rt GBsg #################################
class TrapDetrapRtGBsg(Model):
    def __init__(self,V,kbTc,D0,Vsc,tesc):
        self.nrRateEqs = 4
        super().__init__(V,kbTc,D0,Vsc,tesc)
    #Model equations
    def rateeq(self,Ns,t,params):
        N,Nt,Nw,Nwsg = Ns
        R,V,GB,GBsg,Gt,Gd,Rt,Ges,Gessg,NwT,NwsgT,eta,P,D = params
        return [-R*N**2/V -Rt*N*Nt/(2*V) + 2*GB*Nw - Gt*N + Gd*Nt  + GBsg*Nwsg,
                Gt*N -Gd*Nt -Rt*N*Nt/(2*V) + GBsg*Nwsg,
                R*N**2/(2*V) - GB*Nw - Ges*(Nw-NwT),
                Rt*N*Nt/(2*V) - GBsg*Nwsg - Ges*(Nwsg-NwsgT)]
    
    def jac(self,Ns,t,params):
        N,Nt,Nw,Nwsg = Ns
        R,V,GB,GBsg,Gt,Gd,Rt,Ges,Gessg,NwT,NwsgT,eta,P,D = params
        return [[-2*R*N/V-Rt*Nt/(2*V)-Gt, -Rt*N/(2*V) + Gd, 2*GB,GBsg],
                [Gt-Rt*Nt/(2*V),-Rt*N/(2*V)-Gd,0,GBsg],
                [R*N/V ,0,-GB-Ges,0],
                [Rt*Nt/(2*V),Rt*N/(2*V),0,-GBsg-Ges]]

    def calc_params(self,Gt,Gd,Rt,GBsg,e,eta,P,T):
        kbT = self.kb*T
        D = kidcalc.D(kbT,self.N0,self.Vsc,self.kbTD)
        NT = kidcalc.nqp(kbT,D,self.N0)*self.V
        R = (2*D/self.kbTc)**3/(4*D*self.N0*self.t0)
        Rstar = R/(1+self.tesc/self.tpb)
        Ges = 1/self.tesc
        Gessg = Ges
        GB = 1/self.tpb
        NwT = R*NT**2/(2*self.V*GB)
        NwsgT = kidcalc.calc_Nwsg(kbT,self.V,D,e)
        return [R,self.V,GB,GBsg,Gt,Gd,Rt,Ges,Gessg,NwT,NwsgT,eta,P,D],NT,Rstar
    
    def calc_MB(self,*args,retnumrates = False):
        params,NT,Rstar = self.calc_params(*args)
        R,V,GB,GBsg,Gt,Gd,Rt,Ges,Gessg,NwT,NwsgT,eta,P,D = params
        #Steady state values
        t=0 #dummy var. for rate eq.
        Nqp0,Nt0,Nw0,Nwsg0 = root(
            self.rateeq,[NT,NT/2,NwT,NwsgT],args=(t,params),
            tol=1e-12,jac=self.jac,method='hybr').x

        #M and B matrices
        M = np.array([[Gt+2*Rstar*Nqp0/V+Rt*Nt0/(2*V),Rt*Nqp0/(2*V)-Gd],
                      [Rt*Nt0/(2*V)-Gt,Gd+Rt*Nqp0/(2*V)]])
        B = np.array([[Gt*Nqp0+Gd*Nt0+4*Rstar*Nqp0**2/V+Rt*Nqp0*Nt0/V,
                       -Gt*Nqp0-Gd*Nt0-Rt*Nqp0*Nt0/V],
                      [-Gt*Nqp0-Gd*Nt0-Rt*Nqp0*Nt0/V,
                       Gt*Nqp0+Gd*Nt0+Rt*Nqp0*Nt0/V]])

        if retnumrates:
            return M,B,{'NqpT':NT,
                        'Nqp0':Nqp0,
                        'Nt0':Nt0,
                        'Nw0':Nw0,
                        'NwT':NwT,
                        'Nwsg0':Nwsg0},{'R*NqpT/2V':R*NT/(2*V),
                                        'R*Nqp0/2V':R*Nqp0/(2*V),
                                        'Rt*Nt0/2V':Rt*Nt0/(2*V),
                                        'Gt':Gt,
                                        'Gd':Gd,
                                        'GB':GB,
                                        'GBsg':GBsg}
        else:
            return M,B
####################################Kozorzov############################################
class Kozorezov(TrapDetrapRtGBsg):
    def calc_params(self,e,nrTraps,t1,t2,xi,GBsg,eta,P,T):
        kbT = self.kb*T
        D = kidcalc.D(kbT,self.N0,self.Vsc,self.kbTD)
        NT = kidcalc.nqp(kbT,D,self.N0)*self.V
        R = (2*D/self.kbTc)**3/(4*D*self.N0*self.t0)
        Rstar = R/(1+self.tesc/self.tpb)
        Rt = xi/(t1*self.N0*D)*(1+e/D)
        Ges = 1/self.tesc
        Gessg = Ges
        GB = 1/self.tpb
        kbTsat = 86.17*0.
        kbTeff = max(kbT,kbTsat)
        Gd = np.sqrt(np.pi)/4*(kbTeff/D)**(3/2)*np.exp(-(D-e)/kbTeff)*\
              (1/t1*(3+2*((D-e)/kbTeff))*kbTeff/D+4/t2*(1+(D-e)/kbTeff))
        Gt = 2*(nrTraps/self.V/(2*self.N0*D))/t2*(1-e/D)*\
            (1/(np.exp((D-e)/kbT)-1)+1)*(1-1/(np.exp(e/kbT)+1))
        NwT = R*NT**2/(2*self.V*GB)
        NwsgT = kidcalc.calc_Nwsg(kbT,self.V,D,e)
        return [R,self.V,GB,GBsg,Gt,Gd,Rt,Ges,Gessg,NwT,NwsgT,eta,P,D],NT,Rstar

    #Params for TrapDetrapRt:
#     def calc_params(self,e,nrTraps,t1,t2,xi,eta,P,T):
#         kbT = self.kb*T
#         D = kidcalc.D(kbT,self.N0,self.Vsc,self.kbTD)
#         NT = kidcalc.nqp(kbT,D,self.N0)*self.V
#         R = (2*D/self.kbTc)**3/(4*D*self.N0*self.t0)
#         Rstar = R/(1+self.tesc/self.tpb)
#         Rt = xi/(t1*self.N0*D)*(1+e/D)
#         Ges = 1/self.tesc
#         GB = 1/self.tpb
#         kbTsat = 86.17*0.
#         kbTeff = max(kbT,kbTsat)
#         Gd = (np.sqrt(np.pi)/4*(kbTeff/D)**(3/2)*np.exp(-(D-e)/kbTeff)*\
#               (1/t1*(3+2*((D-e)/kbTeff))*kbTeff/D+4/t2*(1+(D-e)/kbTeff)))
#         Gt = 2*(nrTraps/self.V/(2*self.N0*D))/t2*(1-e/D)*\
#             (1/(np.exp((D-e)/kbT)-1)+1)*(1-1/(np.exp(e/kbT)+1))
#         NTw = R*NT**2/(2*self.V*GB)
#         return [R,self.V,GB,Gt,Gd,Rt,Ges,NTw,eta,P,D],NT,Rstar