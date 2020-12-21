import kidcalc
import numpy as np
import scipy.integrate as integrate
from scipy import interpolate
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar as minisc
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class KID(object):
    """A class to make KID objects"""
    def __init__(self, 
                 Qc=2e4, 
                 hw0=5*.6582*2*np.pi,
                 kbT0=.2*86.17, 
                 kbT=.2*86.17, 
                 V=1e3,
                 ak=.0268, 
                 d=.05, 
                 tesc = .14e-3,
                 kbTc = 1.2*86.17):
        self.Qc = Qc  # -
        self.hw0 = hw0  # µeV
        self.kbT0 = kbT0  # µeV
        self.kbT = kbT  # µeV
        self.V = V  # µm^3
        self.ak = ak  # -
        self.d = d  # µm
        self.tesc = tesc  # µs - interpolate for 50 nm from table 8.1 deVisser2014

        # All for Al:
        self.kbTc = kbTc
        self.kbTD = 37312.  # µeV
        self.t0 = 440e-3  # µs
        self.tpb = .28e-3  # µs
        self.lbd0 = .092  # µm
        self.N0 = 1.72e4  # µeV^-1 µm^-3
        # arb. see Guruswamy2014
        self.epb = .6 - .4*np.exp(-self.tesc/self.tpb)

# Calculated attributes
    @property
    def D0(self):
        return 1.76*self.kbTc

    @property
    def D_0(self):
        return kidcalc.D(self.kbT, self.N0, self.kbTc, self.kbTD)

    @property  # takes 1.2s to calculate
    def hwread(self):
        return kidcalc.hwread(self.hw0, self.kbT0, self.ak,
                              self.lbd0, self.d, self.D_0,
                              self.D0, self.kbT, self.N0,
                              self.kbTc, self.kbTD)

    @property
    def Nqp_0(self):
        return self.V*kidcalc.nqp(self.kbT, self.D_0, self.N0)

    @property
    def tqp_0(self):
        return (self.V*self.t0*self.N0*self.kbTc**3 /
                (2*self.D0**2*self.Nqp_0))

    @property
    def tqp_1(self):
        return .5*self.tqp_0*(1+self.tesc/self.tpb)
    @property
    def Qi_0(self):
        hwread = self.hwread
        s_0 = kidcalc.cinduct(hwread, self.D_0, self.kbT)
        Qi_0 = kidcalc.Qi(s_0[0], s_0[1], self.ak,
                          self.lbd0, self.d, self.D_0, self.D0,
                          self.kbT)
        return Qi_0
    @property
    def Q_0(self):
        return self.Qc*self.Qi_0/(self.Qc+self.Qi_0)

    @property
    def tres(self):
        return 2*self.Q_0/(self.hwread/.6582e-3)

    @property
    def Vsc(self):
        return kidcalc.Vsc(self.kbTc,self.N0,self.kbTD)

    @property
    def s20(self):
        D_0 = kidcalc.D(self.kbT0, self.N0, self.kbTc, self.kbTD)
        return kidcalc.cinduct(self.hw0, D_0, self.kbT0)[1]
    
    def fit_epb(self,peakdata,wvl,*args,var='phase'):
        peakheight = peakdata.max()
        hwrad = 6.528e-4*2*np.pi*3e8/(wvl*1e-3)
        tres = self.tres
        def minfunc(epb,hwrad,tres,var,peakheight):
            self.epb = epb
            _,_,dAtheta = self.calc_respt(hwrad,*args,tStop=3*tres,points=10) 
            if var == 'phase':
                return np.abs(dAtheta[1,:].max()-peakheight)
            elif var == 'amp':
                return np.abs(dAtheta[0,:].max()-peakheight)
        res = minisc(minfunc,args=(hwrad,tres,var,peakheight),
                     bounds=(0,1),method='bounded',
                     options={'maxiter':5,'xatol':1e-3})
        self.epb = res.x
    
    def set_Teff(self,eta,P):
        R,V,G_B,G_es,N_w0 = self.calc_params()
        Nqp0 = np.sqrt(V*((1+G_B/G_es)*eta*P/self.D_0+2*G_B*N_w0)/R)
        self.kbT = kidcalc.kbTeff(Nqp0,self.N0, V, self.kbTc, self.kbTD)

# Calculation functions
    def rateeq(self,N,t,params):
        N_qp, N_w = N
        R, V, G_B, G_es, N_w0 = params
        derivs = [-R*N_qp**2/V + 2*G_B*N_w,
                  R*N_qp**2/(2*V) - G_B*N_w - G_es*(N_w - N_w0)]
        return derivs
    
    def calc_params(self):
        R = ((2*self.D0/self.kbTc)**3 /
        (2*self.D0*2*self.N0*self.t0))  # µs^-1*um^3 (From Wilson2004 or 2.29)
        G_B = 1/self.tpb  # µs^-1 (From chap8)
        G_es = 1/self.tesc  # µs^-1 (From chap8)
        N_w0 = R*self.Nqp_0**2*self.tpb/(2*self.V)  # arb.
        return [R, self.V, G_B, G_es, N_w0]

    def calc_Nqpevol(self, dNqp, tStop=None, tInc=None):
        if tStop is None:
            tStop = 2*self.tqp_1
        if tInc is None:
            tInc = tStop/1000
        params = self.calc_params()

        # Initial values
        Nqp_ini = self.Nqp_0 + dNqp
        N_0 = [Nqp_ini, params[-1]]
        
        # Time array
        t = np.arange(0., tStop, tInc)
        return t, odeint(self.rateeq, N_0, t, args=(params,))

    def calc_linNqpevol(self, Nqp_ini, tStop=None, tInc=None):
        if tStop is None:
            tStop = 2*self.tqp_1
        if tInc is None:
            tInc = tStop/1000
        t = np.arange(0., tStop, tInc)
        return (Nqp_ini-self.Nqp_0)*np.exp(-t/self.tqp_1)

    def calc_resNqpevol(self, t, Nqpt, hwread):
        tres = self.tres
        X = np.exp(-t/tres)/np.sum(np.exp(-t/tres))
        dNqpt = np.convolve(Nqpt-self.Nqp_0, X)[:len(t)]
        return dNqpt+self.Nqp_0

    def calc_S21(self, Nqp, hwread, s20, dhw=0):
        kbTeff = kidcalc.kbTeff(Nqp, self.N0, self.V, self.kbTc,
                                self.kbTD)
        D = kidcalc.D(kbTeff, self.N0, self.kbTc, self.kbTD)
        
        s1, s2 = kidcalc.cinduct(hwread + dhw, D, kbTeff)

        Qi = kidcalc.Qi(s1, s2, self.ak, self.lbd0, self.d, D,
                        self.D0, self.kbT)
        hwres = kidcalc.hwres(s2, self.hw0, s20, self.ak,
                              self.lbd0, self.d, D,self.D0, self.kbT)
        return kidcalc.S21(Qi, self.Qc, hwread, dhw, hwres)

    def calc_resp(self, Nqp, hwread, s20, D_0, dhw=0):
        #Calculate S21 
        S21 = self.calc_S21(Nqp, hwread, s20, dhw)
        #Define circle at this temperature:
        s_0 = kidcalc.cinduct(hwread, D_0, self.kbT)
        Qi_0 = kidcalc.Qi(s_0[0], s_0[1], self.ak,
                          self.lbd0, self.d, D_0, self.D0,
                          self.kbT)
        S21min = self.Qc/(self.Qc+Qi_0) #Q/Qi
        xc = (1+S21min)/2
        #translate S21 into this circle:
        dA = 1 - np.sqrt((np.real(S21)-xc)**2 +
                         np.imag(S21)**2)/(1-xc)
        theta = np.arctan2(np.imag(S21), (xc-np.real(S21)))
        return S21, dA, theta

    def calc_linresp(self, Nqp, hwread, D_0):
        s_0 = kidcalc.cinduct(hwread, D_0, self.kbT)
        Qi_0 = kidcalc.Qi(s_0[0], s_0[1], self.ak,
                          self.lbd0, self.d, D_0, self.D0,
                          self.kbT)
        Q = Qi_0*self.Qc/(Qi_0+self.Qc)
        beta = kidcalc.beta(self.lbd0, self.d, D_0, self.D0, self.kbT)
        
        kbTeff = kidcalc.kbTeff(Nqp, self.N0, self.V, self.kbTc,
                                self.kbTD)
        D = kidcalc.D(kbTeff, self.N0, self.kbTc, self.kbTD)
        s1, s2 = kidcalc.cinduct(hwread, D, kbTeff)

        lindA = self.ak*beta*Q*(s1 - s_0[0])/s_0[1]
        lintheta = -self.ak*beta*Q*(s2 - s_0[1])/s_0[1]
        return lindA, lintheta

    def calc_dNqp(self, hwrad):
        return hwrad/self.D_0*self.epb

    def calc_respt(self, hwrad,*args,tStop=None, tInc=None,
                   points=50):
        if tStop is None:
            tStop = 3*self.tqp_1
        if tInc is None:
            tInc = tStop/1000
        hwread = self.hwread
        s20 = self.s20
        D_0 = self.D_0

        dNqp = self.calc_dNqp(hwrad)

        t, Nqpwt = self.calc_Nqpevol(dNqp, tStop, tInc,*args)
        resNqpt = self.calc_resNqpevol(t, Nqpwt[:,0], hwread)
#         mask = np.rint(np.linspace(0,len(t)-1,points)).astype(int)        
        mask = np.rint(np.logspace(-1, np.log10(len(t)-1), points)).astype(int)
        Nqpts = resNqpt[mask]
        ts = t[mask]

        dAtheta = np.zeros((2, len(Nqpts)))
        S21 = np.zeros((len(Nqpts)), dtype='complex')

        for i in range(len(Nqpts)):
            S21[i], dAtheta[0,i], dAtheta[1,i] = \
                self.calc_resp(Nqpts[i], hwread, s20, D_0)
        return ts, S21, dAtheta
    
    ##Noise calculation functions
    def calc_respsv(self,plot=False):
        hwread = self.hwread
        s20 = self.s20
        D_0 = self.D_0
        Nqp_0 = self.Nqp_0
        
        dNqp = Nqp_0*1e-2
        Nqparr = np.arange(Nqp_0-10*dNqp,Nqp_0+10*dNqp,dNqp) 
        S21 = np.zeros(len(Nqparr),dtype=np.complex64)
        dA = np.zeros(len(Nqparr))
        theta = np.zeros(len(Nqparr))
        for i in range(len(Nqparr)):
            dA[i],theta[i] = \
            self.calc_linresp(Nqparr[i],hwread,D_0)
        dAspl = interpolate.splrep(Nqparr,dA,s=20)
        thetaspl = interpolate.splrep(Nqparr,theta,s=20)
        dAdNqp = interpolate.splev(Nqp_0,dAspl,der=1)
        dThetadNqp = interpolate.splev(Nqp_0,thetaspl,der=1)
        if plot:
            Nqpspl = np.linspace(Nqparr.min(),Nqparr.max(),100)
            plt.figure()
            plt.plot(Nqparr,dA,'bo')
            plt.plot(Nqpspl,interpolate.splev(Nqpspl,dAspl),'b-')
            plt.xlabel('$N_{qp}$')
            plt.ylabel('$dA$',color='b')
            plt.twinx()
            plt.plot(Nqparr,theta,'ro')
            plt.plot(Nqpspl,interpolate.splev(Nqpspl,thetaspl),'r-')
            plt.ylabel('$\\theta$',color='r')
            plt.tight_layout()
        return dAdNqp,dThetadNqp
    
    def calc_SATheta(self,fstart=1e0,fstop=1e6,points = 200):
        dAdNqp,dThetadNqp = self.calc_respsv()
        
        f = np.logspace(np.log10(fstart),np.log10(fstop),points)
        Sn = 4*self.Nqp_0*self.tqp_1*1e-6/(1+(2*np.pi*f*self.tqp_1*1e-6)**2)
        Sat = Sn * dAdNqp*dThetadNqp / (1 + (2*np.pi*f*self.tqp_1*1e-6)**2)
        return f,Sat
    
# Plot functions
    def plot_freqsweep(self, start=None, stop=None, points=200):
        hwread = self.hwread
        D_0 = self.D_0
        s20 = self.s20
        
        s_0 = kidcalc.cinduct(hwread, D_0, self.kbT)
        Qi_0 = kidcalc.Qi(s_0[0], s_0[1], self.ak,
                          self.lbd0, self.d, D_0, self.D0,
                          self.kbT)
        Q = Qi_0*self.Qc/(Qi_0+self.Qc)
        S21min = self.Qc/(self.Qc+Qi_0) #Q/Qi
        xc = (1+S21min)/2
        if start is None:
            start = -self.hw0/Q*2
        if stop is None:
            stop = self.hw0/Q*2

        for dhw in np.linspace(start, stop, points):
            S21_0 = self.calc_S21(self.Nqp_0, hwread, s20, dhw = dhw)
            plt.plot(np.real(S21_0), np.imag(S21_0), 'r.')
        plt.plot(xc,0,'kx')
        plt.plot(S21min,0,'gx')

    def plot_S21resp(self, hwrad, tStop=None, tInc=None,
                     points=10):
        plt.figure(figsize=(5,5))
        self.plot_freqsweep()
        ts, S21, dAtheta = self.calc_respt(hwrad, tStop=tStop, tInc=tInc, points=points)
        
        plt.plot(np.real(S21), np.imag(S21), '.b')

    def plot_dAthetaresp(self, hwrad, tStop=None, tInc=None,
                         points=50, plot='both'):

        ts, S21, dAtheta = self.calc_respt(hwrad, tStop=tStop,tInc=tInc,points=points)

        plt.yscale('log')
        if plot == 'both':
            plt.plot(ts, dAtheta[0,:])
            plt.figure()
            plt.plot(ts, dAtheta[1,:])
        if plot == 'dA':
            plt.plot(ts, dAtheta[0,:])
        if plot == 'theta':
            plt.plot(ts, dAtheta[1,:])

    def plot_Nqpt(self, hwrad, tStop=None, tInc=None,
                  plot_phonon=False, fit_secondhalf=False,
                  plot_lin=True):

        if tStop is None:
            tStop = 2*self.tqp_1
        if tInc is None:
            tInc = tStop/1000

        Nqp_ini = self.Nqp_0 + self.calc_dNqp(hwrad)

        t, Nqpevol = self.calc_Nqpevol(Nqp_ini, tStop, tInc)
        Nqpt = Nqpevol[:, 0]
        Nwt = Nqpevol[:, 1]

        plt.plot(t, Nqpt - self.Nqp_0)
        plt.yscale('log')

        if plot_lin:
            Nqptlin = self.calc_linNqpevol(Nqp_ini, tStop, tInc)
            plt.plot(t, Nqptlin)

        if fit_secondhalf:
            fit = curve_fit(lambda x, a, b: b*np.exp(-x/a),
                            t[np.round(len(t)/2).astype(int):],
                            Nqpt[np.round(len(t)/2).astype(int):]-self.Nqp_0,
                            p0=(self.tqp_0, Nqp_ini-self.Nqp_0))
            print(fit[0][0])
            plt.plot(t, fit[0][1]*np.exp(-t/fit[0][0]))

        if plot_phonon:
            plt.figure()
            plt.plot(t, Nwt)
            plt.yscale('log')

    def plot_resp(self, hwrad, tStop=None, tInc=None,
                  points=50,plot='all'):
        
        ts, S21, dAtheta = self.calc_respt(hwrad, 
                                           tStop=tStop, tInc=tInc, points=points)
        if plot == 'all' or 'S21' in plot:
            plt.figure(1, figsize=(5, 5))
            self.plot_freqsweep()
            plt.plot(np.real(S21), np.imag(S21), '.b') 
            plt.xlabel(r'$Re(S_{21})$')
            plt.ylabel(r'$Im(S_{21})$')
        if plot == 'all' or 'Amp' in plot:
            plt.figure(2)
            plt.plot(ts, dAtheta[0, :])
            plt.xlabel('t (µs)')
            plt.ylabel(r'$\delta A$')
            plt.yscale('log')
        if plot == 'all' or 'Phase' in plot:
            plt.figure(3)
            plt.plot(ts, dAtheta[1, :])
            plt.xlabel('t (µs)')
            plt.ylabel(r'$\theta$')
            plt.yscale('log')
        if plot == 'all' or 'Nqp' in plot:
            plt.figure(4)
            self.plot_Nqpt(hwrad, tStop, tInc)
            plt.ylabel(r'$\delta N_{qp}$')
            plt.xlabel('t (µs)')
            plt.yscale('log')
    
    def print_params(self):
        #Print Latex table for all the parameters
        units = ['','\micro\electronvolt','\micro\electronvolt','\micro\electronvolt',
                 '\cubic\micro\meter','','\\nano\meter','\\nano\second',
                 '\micro\electronvolt','\milli\electronvolt','\\nano\second',
                 '\\nano\second','\\nano\meter',
                 '\per\micro\electronvolt\per\cubic\micro\meter','']
        scalefactors = [1,1,1,1,1,1,1e3,1e3,1,1e-3,1e3,1e3,1e3,1,1]
        params = ['$Q_c$','$\hbar\omega_0$','$k_BT_0$','$k_BT$','$V$','$\\alpha_k$',
                 '$d$','$\\tau_{esc}$','$k_BT_c$', '$k_bT_D$', '$\\tau_0$', '$\\tau_{pb}$',
                  '$\lambda(0)$', '$N_0$', '$\eta_{pb}$']
        print('\\begin{tabular}{ll}')
        print("{:<12}&{:<8}\\\\".format('Parameter','Value'))
        print('\\hline')
        for param,value,unit,scalefactor in zip(params,
                                               self.__dict__.values(),
                                               units,scalefactors):
            print("{:<12} (\si{{{:s}}})&\SI{{{:.3g}}}{{}}\\\\".format(param,unit,value*scalefactor))
        print('\\end{tabular}')

########################################################################
class S21KID(KID):
    def __init__(self,
                 S21data,
                 Qc=2e4, 
                 hw0=5*.6582*2*np.pi,
                 kbT0=.2*86.17, 
                 kbT=.2*86.17, 
                 V=1e3,
                 ak=.0268, 
                 d=.05, 
                 tesc = .14e-3,
                 kbTc = 1.2*86.17):
        super().__init__(Qc,hw0,kbT0,kbT,V,ak,d,tesc,kbTc)
        self.Qispl = interpolate.splrep(S21data[:,1]*86.17,S21data[:,4],s=0)

    @property
    def Qi_0(self):
        return interpolate.splev(self.kbT,self.Qispl,ext=3)
    
    def calc_S21(self, Nqp, hwread, s20, dhw=0):
        kbTeff = kidcalc.kbTeff(Nqp, self.N0, self.V, self.kbTc,
                                self.kbTD)
        D = kidcalc.D(kbTeff, self.N0, self.kbTc, self.kbTD)
        
        s1, s2 = kidcalc.cinduct(hwread + dhw, D, kbTeff)

        Qi = interpolate.splev(kbTeff,self.Qispl)

        hwres = kidcalc.hwres(s2, self.hw0, s20, self.ak,
                              self.lbd0, self.d, D,self.D0, self.kbT)
        return kidcalc.S21(Qi, self.Qc, hwread, dhw, hwres)

    def calc_resp(self, Nqp, hwread, s20, D_0, dhw=0):
        #Calculate S21 
        S21 = self.calc_S21(Nqp, hwread, s20, dhw)
        #Define circle at this temperature:
        S21min = self.Qc/(self.Qc+self.Qi_0) #Q/Qi
        xc = (1+S21min)/2
        #translate S21 into this circle:
        dA = 1 - np.sqrt((np.real(S21)-xc)**2 +
                         np.imag(S21)**2)/(1-xc)
        theta = np.arctan2(np.imag(S21), (xc-np.real(S21)))
        return S21, dA, theta