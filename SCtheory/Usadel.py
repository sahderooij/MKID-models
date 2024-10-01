import SCtheory as SCth
import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar
import scipy.integrate as integrate
from scipy.interpolate import splev, splrep
import warnings
from multiprocess import Pool


class Usadel(object):
    '''class for the Usadel equation and methods to solve it'''
    def __init__(self, Delta, alpha):
        self.Delta = Delta
        self.alpha = alpha

    def eq(self, theta, E):
        return (self.Delta * np.cos(theta) 
            + 1j * E * np.sin(theta) 
            - self.alpha * np.sin(theta) * np.cos(theta))

    def dtheta(self, theta, E):
        return ( - self.Delta * np.sin(theta)
            + 1j * E * np.cos(theta)
            - self.alpha * (np.cos(theta)**2 - np.sin(theta)**2))

    def _solve(self, E):
        if E == 0:
            x0 = 0
        else:
            x0 = np.arctan(self.Delta/np.abs(E)*np.sign(E).real)
        result = root_scalar(self.eq, args=(E,), method='newton', x0=x0, fprime=self.dtheta)

        if not result.converged:
            warnings.warn(f'\nSolving Usadel eq. did not converge:\n' + 
                          f' flag={result.flag}\n' + 
                          f'E={E}, Delta={self.Delta}, alpha={self.alpha}\n')
        return result.root

    def solve(self, E):
        if isinstance(E, (int, float, complex)):
            return self._solve(E)
        else:
            theta = np.zeros(len(E), dtype='complex')
            for i, Eit in enumerate(E):
                theta[i] = self._solve(Eit)
            return theta
    

class selfcons(object):
    '''class for calculating Delta self-consistently'''
    def __init__(self, SC, alpha):
        self.kbTc = SC.kbTc # meausured Tc
        self.kbTD = SC.kbTD
        self.alpha = alpha
        self.set_kbTc0() # Tc if alpha was 0
    
    def eq(self, D, wn, kbT):
        theta = np.zeros(len(wn))
        Ueq = Usadel(D, self.alpha)
        for i, w in enumerate(wn):
            th = Ueq.solve(1j*w)
            if th.imag != 0:
                warnings.warn('\nComplex theta found in Delta consistency calculation:\n'
                             + f'theta_imag={th.imag}')
            theta[i] = th.real
        return D * np.log(self.kbTc0/kbT) - 2*np.pi*kbT * np.sum(D/wn - np.sin(theta))

    def _solve(self, kbT):
        nmax = int(np.floor((self.kbTD / (kbT * np.pi) - 1)/2))
        wn = np.array([np.pi * kbT * (2 * n + 1) for n in range(nmax)])
        return root_scalar(self.eq, args=(wn, kbT), 
                           x0=1.764*self.kbTc0 + self.alpha*.6,
                           method='secant').root


    def solve(self, kbT):
        if isinstance(kbT, (int, float, complex)):
            return self._solve(kbT)
        else:
            D = np.zeros(len(kbT))
            for i, kbTit in enumerate(kbT):
                D[i] = self._solve(kbTit)
            return D

    def find_kbTc(self):
        res = root_scalar(lambda kbT: self.solve(kbT) - self.kbTc0*1e-6,
                           bracket=[self.kbTc0 - self.alpha, self.kbTc0], method='bisect', rtol=1e-3)
        return res.root 

    def set_kbTc0(self):
        def var_kbTc0(kbTc0):
            self.kbTc0 = kbTc0
            return self.find_kbTc() - self.kbTc
        res = root_scalar(var_kbTc0, bracket=[self.kbTc, self.kbTc + self.alpha], 
                                 method='bisect', rtol=1e-4)
        self.kbTc0 = res.root
        return res

class Nam(object):
    ''' Nam equations and solving them '''
    def __init__(self, SC, alpha):
        self.selfcons = selfcons(SC, alpha)
        self.Usadel = Usadel(0, alpha)
        self.kbTD = SC.kbTD

    def update_thetaspl(self, hw, D, nfine=1000, ncoarse=100):
        Ear = np.append(np.linspace(-hw, 2 * D, nfine), 
                        np.linspace(2 * D + (2 * D + self.kbTD)/ncoarse, self.kbTD, ncoarse))
        self.Usadel.Delta = D
        theta = self.Usadel.solve(Ear)
        self.rethetaspl = splrep(Ear, theta.real)
        self.imthetaspl = splrep(Ear, theta.imag)

    def g1(self, theta1, theta2):
        return (np.cos(theta1).real * np.cos(theta2).real 
                + (-1j * np.sin(theta1)).real * (-1j * np.sin(theta2)).real)
    
    def g2(self, theta1, theta2):
        return (- np.cos(theta1).imag * np.cos(theta2).real 
                - (-1j * np.sin(theta1)).imag * (-1j * np.sin(theta2)).real)

    def s1integrand1(self, E, hw, kbT):
        theta1, theta2 = (splev([E, E+hw], self.rethetaspl, ext=1) 
                          + 1j*splev([E, E + hw], self.imthetaspl, ext=1))
        return 2/hw * self.g1(theta1, theta2) * (SCth.f(E, kbT) - SCth.f(E + hw, kbT))

    def s1integrand2(self, E, hw, kbT):
        theta1, theta2 = (splev([E, E+hw], self.rethetaspl, ext=1) 
                          + 1j*splev([E, E + hw], self.imthetaspl, ext=1))
        return 1/hw * self.g1(theta1, theta2) * (1 - 2 * SCth.f(E + hw, kbT))

    def s2integrand1(self, E, hw, kbT):
        theta1, theta2 = (splev([E, E+hw], self.rethetaspl, ext=1) 
                          + 1j*splev([E, E + hw], self.imthetaspl, ext=1))
        return 1/hw * self.g2(theta1, theta2) * (1 - 2 * SCth.f(E + hw, kbT))

    def s2integrand2(self, E, hw, kbT):
        theta1, theta2 = (splev([E, E+hw], self.rethetaspl, ext=1) 
                          + 1j*splev([E, E + hw], self.imthetaspl, ext=1))
        return 1/hw * (self.g2(theta2, theta1) * (1 - 2 * SCth.f(E, kbT)))

    # NOTE: checked the integration bounds for consistency with Nam1967. 
    # the g1 and g2 functions ensure that we can start at 0 (both when hw > 2 Egap and hw < 2 Egap)
    # note that D is not the onset of DOS, but coherence peak energy. 
    # So we take D instead of 2D as threshold (note: E=0 only becomes sharp when hw > D)
    def _s1(self, kbT, hw):
        D = self.selfcons.solve(kbT)
        self.update_thetaspl(hw, D)
        spoints = (D - hw, D) # E points with sharp features
        s1 = integrate.quad(self.s1integrand1, 0, spoints[0], args=(hw, kbT),
                           points=spoints)[0]
        s1 += integrate.quad(self.s1integrand1, spoints[0], spoints[1], args=(hw, kbT),
                       points=spoints)[0]
        s1 += integrate.quad(self.s1integrand1, spoints[1], 2*D, args=(hw, kbT),
                       points=spoints)[0]
        s1 += integrate.quad(self.s1integrand1, 2*D, np.inf, args=(hw, kbT))[0]
        if hw > D:
            s1 += integrate.quad(self.s1integrand2, -hw, 0, args=(hw, kbT),
                                 points=(-hw, D-hw, -D, 0))[0]
        return s1
        
    def s1(self, kbT, hw):
        if isinstance(kbT, (int, float, complex)):
            return self._s1(kbT)
        else:
            s1 = np.zeros(len(kbT))
            if isinstance(hw, (int, float, complex)):
                hw = np.ones(len(kbT)) * hw
            with Pool() as p:
                for i, res in enumerate(p.imap(
                    lambda x: self._s1(*x), zip(kbT, hw))):
                    s1[i] = res
            return s1

    def _s2(self, kbT, hw):
        D = self.selfcons.solve(kbT)
        self.update_thetaspl(hw, D)
        spoints = (D - hw, D) # E points with sharp features
        s2 = integrate.quad(self.s2integrand1, -hw, spoints[0], args=(hw, kbT),
                   points=spoints)[0]
        s2 += integrate.quad(self.s2integrand1, spoints[0], spoints[1], args=(hw, kbT),
                           points=spoints)[0]
        s2 += integrate.quad(self.s2integrand1, spoints[1], 2*D, args=(hw, kbT),
                           points=spoints)[0]
        s2 += integrate.quad(self.s2integrand1, 2*D, np.inf, args=(hw, kbT))[0]
        if hw > D:
            s2 += integrate.quad(self.s2integrand2, 0, 2*D, args=(hw, kbT),
                            points=(0, D))[0]
            s2 += integrate.quad(self.s2integrand2, 2*D, np.inf, args=(hw, kbT))[0]
        return s2
        
    def s2(self, kbT, hw):
        if isinstance(kbT, (int, float, complex)):
            return self._s2(kbT, hw)
        else:
            s2 = np.zeros(len(kbT))
            if isinstance(hw, (int, float, complex)):
                hw = np.ones(len(kbT)) * hw
            with Pool() as p:
                for i, res in enumerate(p.imap(
                    lambda x: self._s2(*x), zip(kbT, hw))):
                    s2[i] = res
            return s2


class NamDriessen(Nam):
    def g2(self, theta1, theta2):
        return (- np.cos(theta1).real * np.cos(theta2).imag 
                - (-1j * np.sin(theta1)).imag * (-1j * np.sin(theta2)).real)