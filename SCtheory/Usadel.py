import SCtheory as SCth
import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar
import scipy.integrate as integrate
from scipy.interpolate import splev, splrep
import warnings
from multiprocess import Pool
from scipy.special import digamma


class Usadel(object):
    '''class for the Usadel equation and methods to solve it'''
    def __init__(self, Delta, alpha):
        self.Delta = Delta
        self.alpha = alpha

    @property
    def Eg(self):
        '''Gap energy'''
        if self.alpha < self.Delta:
            return (1-(self.alpha/self.Delta)**(2/3))**(3/2)*self.Delta
        else: 
            return 0
    
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
            x0 = np.pi/2
        else:
            x0 = np.arctan2(self.Eg, np.abs(E)*np.sign(E).real) #alpha=0 solution, with Delta->Eg
        result = root_scalar(self.eq, args=(E,), method='newton', x0=x0, fprime=self.dtheta)

        if not result.converged:
            warnings.warn(f'\nSolving Usadel eq. did not converge:\n' + 
                          f' flag={result.flag}\n' + 
                          f'E={E}, Delta={self.Delta:.2f}, alpha={self.alpha:.2f}, Eg={self.Eg:.2f}\n')
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
        self.kbTc0 = self.kbTc * np.exp(
            digamma(.5 + self.alpha/(2*np.pi*self.kbTc)) 
            - digamma(.5)
        ) # Tc for alpha=0
        
    def eq(self, D, wn, kbT):
        Ueq = Usadel(D, self.alpha)
        th = Ueq.solve(1j*wn)
        if any(th.imag != 0):
                warnings.warn('\nComplex theta found in Delta consistency calculation:\n'
                             + f'theta_imag={th.imag}')
        theta = th.real
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

class Nam(object):
    ''' Nam equations and solving them '''
    def __init__(self, SC, alpha):
        self.selfcons = selfcons(SC, alpha)
        self.Usadel = Usadel(0, alpha)
        self.kbTD = SC.kbTD

    def g1(self, theta1, theta2):
        return np.abs(np.cos(theta1).real * np.cos(theta2).real 
                + (-1j * np.sin(theta1)).real * (-1j * np.sin(theta2)).real)
    
    def g2(self, theta1, theta2):
        return np.abs(np.cos(theta1).imag * np.cos(theta2).real 
                + (-1j * np.sin(theta1)).imag * (-1j * np.sin(theta2)).real)

    def s1integrand1(self, E, hw, kbT):
        theta1 = self.Usadel.solve(E)
        theta2 = self.Usadel.solve(E + hw)
        return 2/hw * self.g1(theta1, theta2) * (SCth.f(E, kbT) - SCth.f(E + hw, kbT))

    def s1integrand2(self, E, hw, kbT):
        theta1 = self.Usadel.solve(E)
        theta2 = self.Usadel.solve(E + hw)
        return 1/hw * self.g1(theta1, theta2) * (1 - 2 * SCth.f(E + hw, kbT))

    def s2integrand1(self, E, hw, kbT):
        theta1 = self.Usadel.solve(E)
        theta2 = self.Usadel.solve(E + hw)
        return 1/hw * self.g2(theta1, theta2) * (1 - 2 * SCth.f(E + hw, kbT))

    def s2integrand2(self, E, hw, kbT):
        theta1 = self.Usadel.solve(E)
        theta2 = self.Usadel.solve(E + hw)
        return 1/hw * (self.g2(theta2, theta1) * (1 - 2 * SCth.f(E, kbT)))

    # NOTE: checked the integration bounds for consistency with Nam1967. 
    # added absolute value to g1 and g2, to converge to MB for alpha=0
    def _s1(self, kbT, hw):
        D = self.selfcons.solve(kbT)
        self.Usadel.Delta = D
        s1 = integrate.quad(self.s1integrand1, self.Usadel.Eg, np.inf, args=(hw, kbT))[0]
        s1 += integrate.quad(self.s1integrand2, self.Usadel.Eg-hw, -self.Usadel.Eg, args=(hw, kbT))[0]
        return s1
        
    def s1(self, kbT, hw):
        if isinstance(kbT, (int, float, complex)) and isinstance(hw, (int, float, complex)):
            return self._s1(kbT)
        else:
            if isinstance(hw, (int, float, complex)):
                hw = np.ones(len(kbT)) * hw
                s1 = np.zeros(len(kbT))
            elif isinstance(kbT, (int, float, complex)):
                kbT = np.ones(len(hw)) * kbT
                s1 = np.zeros(len(hw))
            elif len(kbT) == len(hw):
                s2 = np.zeros(len(kbT))
            else:
                raise ValueError('kbT and hw are arrays of different length')
            with Pool() as p:
                for i, res in enumerate(p.imap(
                    lambda x: self._s1(*x), zip(kbT, hw))):
                    s1[i] = res
            return s1

    def _s2(self, kbT, hw):
        D = self.selfcons.solve(kbT)
        self.Usadel.Delta = D
        # break-up integration into two parts at discontinuities
        if hw > 2*self.Usadel.Eg:
            s2 = integrate.quad(self.s2integrand1, 
                                self.Usadel.Eg-hw, -self.Usadel.Eg, args=(hw, kbT))[0]
            s2 += integrate.quad(self.s2integrand1, 
                                 -self.Usadel.Eg, self.Usadel.Eg, args=(hw, kbT))[0]
        else:
            s2 = integrate.quad(self.s2integrand1, 
                                self.Usadel.Eg-hw, self.Usadel.Eg, args=(hw, kbT))[0]
        s2 += integrate.quad(self.s2integrand1, self.Usadel.Eg, np.inf, args=(hw, kbT))[0]
        s2 += integrate.quad(self.s2integrand2, self.Usadel.Eg, np.inf, args=(hw, kbT))[0]
        return s2
        
    def s2(self, kbT, hw):
        if isinstance(kbT, (int, float, complex)) and isinstance(hw, (int, float, complex)):
            return self._s2(kbT, hw)
        else:
            if isinstance(hw, (int, float, complex)):
                hw = np.ones(len(kbT)) * hw
                s2 = np.zeros(len(kbT))
            elif isinstance(kbT, (int, float, complex)):
                kbT = np.ones(len(hw)) * kbT
                s2 = np.zeros(len(hw))
            elif len(kbT) == len(hw):
                s2 = np.zeros(len(kbT))
            else:
                raise ValueError('kbT and hw are arrays of different length')
            with Pool() as p:
                for i, res in enumerate(p.imap(
                    lambda x: self._s2(*x), zip(kbT, hw))):
                    s2[i] = res
            return s2


class NamDriessen(Nam):
    def g2(self, theta1, theta2):
        return (- np.cos(theta1).real * np.cos(theta2).imag 
                - (-1j * np.sin(theta1)).imag * (-1j * np.sin(theta2)).real)