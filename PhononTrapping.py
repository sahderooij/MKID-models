import numpy as np
import cmath
import warnings
from scipy.integrate import quad
import matplotlib.pyplot as plt

def cot(angle):
    '''the cotangens defined via numpy'''
    return np.cos(angle)/np.sin(angle)


class Solid(object):
    '''Object to store the density and longitudinal 
    and transverse sound velocities in a solid.
    Same attributes as the SC objects.'''
    def __init__(self, rho, cL, cT):
        self.rho = rho
        self.cL = cL
        self.cT = cT
        
class Interface(object):
    '''Defines an interface between two solids, 
    the methods are used for calculating the transparancies for different 
    polarizations as set out by Kaplan in 1979.'''
    def __init__(self, sol1, sol2):
        self.sol1 = sol1
        self.sol2 = sol2
    
    @property
    def critLL(self):
        return cmath.asin(self.sol1.cL/self.sol2.cL).real

    @property
    def critLT(self):
        return cmath.asin(self.sol1.cL/self.sol2.cT).real
    
    @property
    def critTL(self):
        return cmath.asin(self.sol1.cT/self.sol2.cL).real
    
    @property
    def critTT(self):
        return cmath.asin(self.sol1.cT/self.sol2.cT).real
    
    @property
    def critTL1(self):
        return cmath.asin(self.sol1.cT/self.sol1.cL).real
    
    def get_angles(self, theta1=None, gamma1=None):
        ''' Uses Snell's law to calculate the angles of the resulting waves
        note that this should be done complex to account for 
        phase shifts when one or more critical angles are surpassed.'''
        if theta1 is not None:
            gamma1 = cmath.asin(self.sol1.cT/self.sol1.cL
                                *np.sin(theta1))
        elif gamma1 is not None:
            theta1 = cmath.asin(self.sol1.cL/self.sol1.cT
                                *np.sin(gamma1))
        theta2 = cmath.asin(self.sol2.cL/self.sol1.cL*np.sin(theta1))
        gamma2 = cmath.asin(self.sol2.cT/self.sol1.cL*np.sin(theta1))
        return theta1, theta2, gamma1, gamma2
    
    def get_BCmatrix(self, angles):
        '''returns the system of boundary condition equations for both longitudinal
        incoming waves (first column) and SV incoming waves (second column)'''
        theta1, theta2, gamma1, gamma2 = angles
        sol1 = self.sol1
        sol2 = self.sol2
        
        #calculate wave numbers per direction (divided by frequency)
        a1 = np.cos(theta1)/sol1.cL
        a2 = np.cos(theta2)/sol2.cL
        b1 = np.cos(gamma1)/sol1.cT
        b2 = np.cos(gamma2)/sol2.cT
        s = np.sin(theta1)/sol1.cL

        #calculate wave amplitudes, normalized to incoming ampitude
        matrix = np.array(
            [[1, 1, 0, 0, 0, 0],
             [-a1, s, a1, s, a2, -s],
             [s, b1, s, -b1, -s, -b2],
             [sol1.rho*sol1.cT**2*(cot(gamma1)**2-1),
              -2*sol1.rho*sol1.cT**2*cot(gamma1),
              sol1.rho*sol1.cT**2*(cot(gamma1)**2-1),
              2*sol1.rho*sol1.cT**2*cot(gamma1),
              -sol2.rho*sol2.cT**2*(cot(gamma2)**2-1),
              2*sol2.rho*sol2.cT**2*cot(gamma2)],
             [-2*sol1.rho*sol1.cT**2*cot(theta1),
              sol1.rho*sol1.cT**2*(1-cot(gamma1)**2),
              2*sol1.rho*sol1.cT**2*cot(theta1), 
              sol1.rho*sol1.cT**2*(1-cot(gamma1)**2),
              2*sol2.rho*sol2.cT**2*cot(theta2),
              -sol2.rho*sol2.cT**2*(1-cot(gamma2)**2)]], 
            dtype='cfloat'
        )
        return matrix

    
    def eta_l_angle(self, theta1, rettottrans=True):
        '''Longitudinal phonon transparency from solid1 to solid2 
        with incoming phonon at angle theta1'''
        #mask 0's by a really small number to catch nans
        if theta1 == 0: 
            theta1= 1e-18
        
        angles = self.get_angles(theta1=theta1)
        matrix = self.get_BCmatrix(angles)
        
        A, C, D, E, F = np.linalg.solve(matrix[:, (0, 2, 3, 4, 5)], [1, 0, 0, 0, 0])
        if theta1 >= self.critLL:
            E=0
        if theta1 >= self.critLT:
            F=0

        sol1 = self.sol1
        sol2 = self.sol2
        theta1, theta2, gamma1, gamma2 = angles
        trans_l = (np.abs(E)**2*np.cos(theta2) * sol2.rho * sol1.cL)  / (np.cos(theta1) * sol1.rho * sol2.cL)
        trans_t = (np.abs(F)**2*np.cos(gamma2) * sol2.rho * sol1.cL) / (np.cos(theta1) * sol1.rho * sol2.cT) 
        trans = trans_l + trans_t
        refl_l = np.abs(C)**2 
        refl_t = np.abs(D)**2 * sol1.cL * np.cos(gamma1) / (sol1.cT * np.cos(theta1))
        refl = refl_l + refl_t
        if (trans + refl).round(10) != 1:
            tot = trans + refl
            warnings.warn(f'\nT + R != 1, but {tot[tot.round(10) != 1.]}. \nangle={theta1}')

        if rettottrans:
            return trans.real
        else:
            return trans_l, trans_t, refl_l, refl_t
        
    def eta_SV_angle(self, gamma1, rettottrans=True):
        '''Transverse, Shear Vertical polarization, phonon transparency
        with incoming phonon at angle gamma1.
        If rettottrans is True, the total transmission is returned. '''
        #mask 0's by a really small number to catch nans
        if gamma1 == 0: 
            gamma1= 1e-18
        
        angles = self.get_angles(gamma1=gamma1)
        matrix = self.get_BCmatrix(angles)
        
        B, C, D, E, F = np.linalg.solve(matrix[:, (1, 2, 3, 4, 5)], [1, 0, 0, 0, 0])
        if gamma1 >= self.critTL:
            E = 0
        if gamma1 >= self.critTT:
            F = 0
        if gamma1 >= self.critTL1: 
            # note: as generally cT < cL, there is another critical angle at which you don't
            # get longitudinal waves
            C = 0

        sol1 = self.sol1
        sol2 = self.sol2
        theta1, theta2, gamma1, gamma2 = angles
        trans_l = (np.abs(E)**2*np.cos(theta2) * sol2.rho * sol1.cT)  / (np.cos(gamma1) * sol1.rho * sol2.cL)
        trans_t = (np.abs(F)**2*np.cos(gamma2) * sol2.rho * sol1.cT) / (np.cos(gamma1) * sol1.rho * sol2.cT) 
        trans = trans_l + trans_t
        refl_l = np.abs(C)**2 * sol1.cT * np.cos(theta1) / (sol1.cL * np.cos(gamma1))
        refl_t = np.abs(D)**2 
        refl = refl_l + refl_t
        if (trans + refl).round(10) != 1:
            tot = trans + refl
            warnings.warn(f'\nT + R != 1, but {tot[tot.round(10) != 1.]}. \nangle={gamma1}')

        if rettottrans:
            return trans.real
        else:
            return trans_l, trans_t, refl_l, refl_t
        
    def eta_SH_angle(self, gamma1):
        '''Transverse, Shear Horizontal polarization, phonon transparency
        with incoming phonon at gamma1'''
        if gamma1 >= self.critTT:
            return 0
        else:
            sol1 = self.sol1
            sol2 = self.sol2

            gamma2 = cmath.asin(sol2.cT/sol1.cT*np.sin(gamma1))
            x = sol2.rho * sol2.cT * np.cos(gamma2) / (sol1.rho * sol1.cT * np.cos(gamma1))
            return 4 * x / (1 + x)**2
    
    @property
    def eta_l(self):
        '''Angle averaged longitudinal phonon transparency'''
        def integrand(theta1):
            return 2 * np.sin(theta1) * np.cos(theta1) * self.eta_l_angle(theta1)
        return quad(integrand, 0, np.pi/2)[0]
    
    @property
    def eta_SH(self):
        def integrand(gamma1):
            '''Angle averaged transverse, shear horizontal phonon transparency'''
            return (2 * np.sin(gamma1) * np.cos(gamma1) 
                    * self.eta_SH_angle(gamma1) )
        return quad(integrand, 0, np.pi/2)[0]
    
    @property
    def eta_SV(self):
        '''Angle averaged transverse, shear vertical phonon transparency'''
        def integrand(gamma1):
            return (2 * np.sin(gamma1) * np.cos(gamma1) 
                    * self.eta_SV_angle(gamma1) )
        return quad(integrand, 0, np.pi/2)[0]
    
    @property
    def eta_t(self):
        '''Average transverse phonon transparency for all polarizations.'''
        return (self.eta_SV + self.eta_SH) / 2
        
    @property
    def eta(self):
        '''Averaged phonon transparency with sound velocties taken into account. 
        Note: this does not work for very high transparencies 
        (as eta will be come greater than 1)'''
        return ((2*self.eta_t/self.sol1.cT**2 + self.eta_l/self.sol1.cL**2)
                * (2/self.sol1.cT**3 + 1/self.sol1.cL**3)**(-2/3))
    
    def plot_eta_angles(self, nrpoints=100):
        plt.figure()
        angles = np.linspace(0, np.pi/2, nrpoints)
        eta_ls = np.array([self.eta_l_angle(angles[i]) for i in range(len(angles))])
        eta_SHs = np.array([self.eta_SH_angle(angles[i]) for i in range(len(angles))])
        eta_SVs = np.array([self.eta_SV_angle(angles[i]) for i in range(len(angles))])
        eta_ts = (eta_SHs + eta_SVs)/2
        plt.plot(angles, eta_ls, label='L')
        plt.plot(angles, eta_SHs, label='SH')
        plt.plot(angles, eta_SVs, label='SV')
        # plt.plot(angles, eta_ts, label='T')
        plt.plot(angles, 2*np.sin(angles)*np.cos(angles), label='angle dist.', 
                color='k')
        plt.axvline(self.critLL, color='k', linestyle='--')
        plt.axvline(self.critLT, color='k')
        plt.axvline(self.critTL, color='.5')
        plt.axvline(self.critTT, color='.5', linestyle='--')
        plt.xlabel('Angle (rad.)')
        plt.ylabel('$\eta$')
        plt.legend()