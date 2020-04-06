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
        return kidcalc.D(self.kbT, self.N0, self.Vsc, self.kbTD)

    @property  # takes 1.2s to calculate
    def hwread(self):
        return kidcalc.hwread(self.hw0, self.kbT0, self.ak,
                              self.lbd0, self.d, self.D_0,
                              self.D0, self.kbT, self.N0,
                              self.Vsc, self.kbTD)

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
    def tres(self):
        Qi_0 = self.Qi_0
        Q_0 = self.Qc*Qi_0/(self.Qc+Qi_0)
        return 2*Q_0/(self.hwread/.6582e-3)

    @property
    def Vsc(self):
        def integrand1(E, D):
            return 1/np.sqrt(E**2-D**2)
        return 1/(integrate.quad(integrand1, self.D0, self.kbTD,
                                 args=(self.D0,))[0]*self.N0)

    @property
    def s20(self):
        D_0 = kidcalc.D(self.kbT0, self.N0, self.Vsc, self.kbTD)
        return kidcalc.cinduct(self.hw0, D_0, self.kbT0)[1]

# Calculation functions
    def calc_Nqpevol(self, Nqp_ini, tStop=None, tInc=None):
        if tStop is None:
            tStop = 2*self.tqp_1
        if tInc is None:
            tInc = tStop/1000

        # Calculate N_qp evolution
        def rateeq(N, t, params):
            N_qp, N_w = N
            R, V, G_B, G_es, N_w0 = params
            derivs = [-R*N_qp**2/V + 2*G_B*N_w,
                      R*N_qp**2/(2*V) - G_B*N_w - G_es*(N_w - N_w0)]
            return derivs
        # Parameters
        R = ((2*self.D0/self.kbTc)**3 /
             (2*self.D0*2*self.N0*self.t0))  # µs^-1*um^3 (From Wilson2004 or 2.29)
        G_B = 1/self.tpb  # µs^-1 (From chap8)
        G_es = 1/self.tesc  # µs^-1 (From chap8)
        N_w0 = R*self.Nqp_0**2*self.tpb/(2*self.V)  # arb.
        params = [R, self.V, G_B, G_es, N_w0]

        # Initial values
        N_0 = [Nqp_ini, N_w0]

        # Time array
        t = np.arange(0., tStop, tInc)
        return t, odeint(rateeq, N_0, t, args=(params,))

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
        kbTeff = kidcalc.kbTeff(Nqp, self.N0, self.V, self.Vsc,
                                self.kbTD, self.kbTc)
        D = kidcalc.D(kbTeff, self.N0, self.Vsc, self.kbTD)
        
        s1, s2 = kidcalc.cinduct(hwread + dhw, D, kbTeff)

        Qi = kidcalc.Qi(s1, s2, self.ak, self.lbd0, self.d, D,
                        self.D0, self.kbT)
        hwres = kidcalc.hwres(s2, self.hw0, s20, self.ak,
                              self.lbd0, self.d, D,
                              self.D0, self.kbT)
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
        kbTeff = kidcalc.kbTeff(Nqp, self.N0, self.V, self.Vsc,
                                self.kbTD, self.kbTc)
        D = kidcalc.D(kbTeff, self.N0, self.Vsc, self.kbTD)
        s1, s2 = kidcalc.cinduct(hwread, D, kbTeff)
        s_0 = kidcalc.cinduct(hwread, D_0, self.kbT)
        Qi_0 = kidcalc.Qi(s_0[0], s_0[1], self.ak,
                          self.lbd0, self.d, D_0, self.D0,
                          self.kbT)
        Q = Qi_0*self.Qc/(Qi_0+self.Qc)
        beta = kidcalc.beta(self.lbd0, self.d, D_0, self.D0, self.kbT)
        lindA = -1*-self.ak*beta*Q*(s1 - s_0[0])/s2
        lintheta = -self.ak*beta*Q*(s2 - s_0[1])/s2
        return lindA, lintheta

    def calc_dNqp(self, hwrad):
        return hwrad/self.D_0*self.epb

    def calc_respt(self, hwrad, tStop=None, tInc=None,
                   points=50):
        if tStop is None:
            tStop = 3*self.tqp_1
        if tInc is None:
            tInc = tStop/1000
        hwread = self.hwread
        s20 = self.s20
        D_0 = self.D_0

        Nqp_ini = self.Nqp_0 + self.calc_dNqp(hwrad)

        t, Nqpwt = self.calc_Nqpevol(Nqp_ini, tStop, tInc)
        resNqpt = self.calc_resNqpevol(t, Nqpwt[:, 0], hwread)
#         mask = np.rint(np.linspace(0,len(t)-1,points)).astype(int)
        mask = np.rint(np.logspace(-1, np.log10(len(t)-1), points)).astype(int)
        Nqpts = resNqpt[mask]
        ts = t[mask]

        dAtheta = np.zeros((2, len(Nqpts)))
        S21 = np.zeros((len(Nqpts)), dtype='complex')

        for i in range(len(Nqpts)):
            S21[i], dAtheta[0, i], dAtheta[1, i] = \
                self.calc_resp(Nqpts[i], hwread, s20, D_0)
        return ts, S21, dAtheta
    
    ##Here come the calc funtions for the noise
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
            S21[i],dA[i],theta[i] = \
            self.calc_resp(Nqparr[i],hwread,s20,D_0)
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
        self.plot_freqsweep()
        ts, S21, dAtheta = self.calc_respt(hwrad, tStop, tInc, points)
        plt.plot(np.real(S21), np.imag(S21), '.b')

    def plot_dAthetaresp(self, hwrad, tStop=None, tInc=None,
                         points=50, plot='both'):

        ts, S21, dAtheta = self.calc_respt(hwrad, tStop, tInc, points)

        plt.yscale('log')
        if plot is 'both':
            plt.plot(ts, dAtheta[0, :])
            plt.figure()
            plt.plot(ts, dAtheta[1, :])
        if plot is 'dA':
            plt.plot(ts, dAtheta[0, :])
        if plot is 'theta':
            plt.plot(ts, dAtheta[1, :])

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
        
        ts, S21, dAtheta = self.calc_respt(hwrad, tStop, tInc, points)
        if plot is 'all' or 'S21' in plot:
            plt.figure(1, figsize=(5, 5))
            self.plot_freqsweep()
            plt.plot(np.real(S21), np.imag(S21), '.b') 
            plt.xlabel(r'$Re(S_{21})$')
            plt.ylabel(r'$Im(S_{21})$')
        if plot is 'all' or 'Amp' in plot:
            plt.figure(2)
            plt.plot(ts, dAtheta[0, :])
            plt.xlabel('t (µs)')
            plt.ylabel(r'$\delta A$')
            plt.yscale('log')
        if plot is 'all' or 'Phase' in plot:
            plt.figure(3)
            plt.plot(ts, dAtheta[1, :])
            plt.xlabel('t (µs)')
            plt.ylabel(r'$\theta$')
            plt.yscale('log')
        if plot is 'all' or 'Nqp' in plot:
            plt.figure(4)
            self.plot_Nqpt(hwrad, tStop, tInc)
            plt.ylabel(r'$\delta N_{qp}$')
            plt.xlabel('t (µs)')
            plt.yscale('log')
    def print_params(self):
        #Print Latex table for all the parameters
        units = ['-','$\mu eV$', '$\mu eV$', '$\mu eV$','$\mu m^{-3}$',
                '-','$\mu m$', '$\mu s$','$\mu eV$', '$\mu eV$','$\mu s$',
                '$\mu s$','$\mu m$','$\mu eV^{-1}\mu m^{-3}$','-']
        params = ['$Q_c$','$\hbar\omega_0$','$k_BT_0$','$k_BT$','$V$','$\\alpha_k$',
                 '$d$','$\\tau_{esc}$','$k_BT_c$', '$k_bT_D$', '$\\tau_0$', '$\\tau_{pb}$',
                  '$\lambda(0)$', '$N_0$', '$\eta_{pb}$']
        print('\\begin{tabular}{lll}')
        print("{:<12}&{:<8}\t&{}\\\\".format('Parameter','Value','Unit'))
        print('\\hline')
        for param,value,unit in zip(params,self.__dict__.values(),units):
            print("{:<12}&{:.3g}\t&{:s}\\\\".format(param,value,unit))
        print('\\end{tabular}')

########################################################################
class trKID(KID):
    def __init__(self, 
                 Qc=2e4, 
                 hw0=5*.6582*2*np.pi,
                 kbT0=.2*86.17, 
                 kbT=.2*86.17, 
                 V=1e3,
                 ak=.0268, 
                 d=.05, 
                 tesc = .14e-3,
                 kbTc = 1.2*86.17,
                 G_t = 1e-3/2,
                 G_d = 1e-3/2,
                 G_rt = 2.4e-7,
                 genlocrat = 0,
                 allqps = False):
        KID.__init__(self,Qc,hw0,kbT0,kbT,V,ak,d,tesc,kbTc)
        self.G_t = G_t
        self.G_d = G_d
        self.G_rt = G_rt
        self.genlocrat = genlocrat
        self.allqps = allqps
        
    def calc_Nqpevol(self, Nqp_ini, tStop=None, tInc=None):
        #yet to be done
        pass
        
    def calc_SATheta(self,fstart=1e0,fstop=1e6,points = 200):
        dAdNqp,dThetadNqp = self.calc_respsv()
        f = np.logspace(np.log10(fstart),np.log10(fstop),points)
        
        G_es = 1/self.tesc
        G_d = self.G_d
        G_t = self.G_t
        G_rt = self.G_rt
        V = self.V
        G_B = (1-self.genlocrat)/self.tpb
        G_Bt = self.genlocrat/self.tpb
        R = ((2*self.D0/self.kbTc)**3 /
             (2*self.D0*2*self.N0*self.t0))  # µs^-1*um^3 (From Wilson2004 or 2.29)
        Nqp_0 = self.Nqp_0
        Nw0 = R*Nqp_0**2/(2*V*G_B)
        if self.G_rt != 0:
            Nqp0arr = np.roots([1,G_d + 2*V*G_t/R,-Nqp_0**2,-G_d/G_rt*Nqp_0**2])
            Nqp0 = Nqp0arr[np.logical_and(np.isreal(Nqp0arr),Nqp0arr > 0)]
            if Nqp0.size is 1:
                Nqp0 = Nqp0[0]
            else:
                raise ValueError('Multiple steady state solutions')
        else:
            Nqp0 = Nqp_0
        Nt0 = (G_t*Nqp0+G_Bt*Nw0)/(G_d+G_rt*Nqp0)
        
        M = np.array([[G_t + G_rt*Nt0 + 2*R*Nqp0/V, G_rt*Nqp0-G_d, -2*G_B],
                      [G_rt*Nt0 - G_t, G_d + G_rt*Nqp0,-2*G_Bt],
                      [-R*Nqp0/V-G_rt*Nt0 ,-G_rt*Nt0 ,G_B + G_Bt + G_es]])
        B = np.array([[(G_t + G_rt*Nt0)*Nqp0 + G_d*Nt0 + 2*R*Nqp0**2/V + 4*G_B*Nw0,
                       -G_d*Nt0 - (G_t + G_rt*Nt0)*Nqp0, 
                       -R*Nqp0**2/V-2*G_B*Nw0],
                      [-G_d*Nt0 - (G_t + G_rt*Nt0)*Nqp0, 
                       (G_t + Nt0*G_rt)*Nqp0 + G_d*Nt0 + 4*G_rt*Nqp0*Nt0 + 4*G_Bt*Nw0,
                       -2*G_rt*Nqp0*Nt0],
                      [-R*Nqp0**2/V-2*G_B*Nw0, 
                       -2*G_rt*Nqp0*Nt0 - 2*G_Bt*Nw0, 
                       G_B*Nw0 + R*Nqp0**2/(2*V) + 2*G_es*Nw0 + G_Bt*Nw0 + G_rt*Nqp0*Nt0]])
        warr = 2*np.pi*f*1e-6 #rad/µs
        sw = np.zeros(len(warr))
        for j in range(len(warr)):
            Gw = 2*np.real(np.linalg.multi_dot([
                np.linalg.inv(M+1j*warr[j]*np.eye(3)),
                B,
                np.linalg.inv(M-1j*warr[j]*np.eye(3))]))
            sw[j] = Gw[0,0] 
            if self.allqps:
                sw[j] += Gw[0,1] + Gw[1,0] + Gw[1,1]
        return f,sw*dAdNqp*dThetadNqp
        
    
    