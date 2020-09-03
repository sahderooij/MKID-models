import numpy as np
import scipy.special as sp

class CPW(object):
    """Class to calculate CPW properties, such as vph, Z0 and ak. 
    Give S,W,d and l in µm, and Lk in pH/sq"""
    def __init__(self,S,W,d,Lk,eeff,l=1):
        self.S = S*1e-6
        self.W = W*1e-6
        self.d = d*1e-6
        self.Lk = Lk*1e-12
        self.eeff = eeff
        self.l = l*1e-6
        
        self.e0 = 8.85e-12
        self.m0 = 4*np.pi*1e-7
        
    ##### CALCULATED ATTRIBUTES #####
    @property
    def k(self):
        return self.S/(self.S+2*self.W)
    
    @property
    def gc(self):
        return (np.pi+np.log(4*np.pi*self.S/self.d)-self.k*np.log((1+self.k)/(1-self.k)))\
                /(4*self.S*(1-self.k**2)*sp.ellipk(self.k**2)**2)
    
    @property
    def gg(self):
        return self.k*(np.pi\
                       + np.log(4*np.pi*(self.S+2*self.W)/self.d)\
                       - 1/self.k*np.log((1+self.k)/(1-self.k)))\
            /(4*self.S*(1-self.k**2)*sp.ellipk(self.k**2)**2)
    
    @property
    def xi(self):
        return sp.ellipk(1-self.k**2)/sp.ellipk(self.k**2)

    @property
    def Lgl(self):
        return self.m0*self.xi/4
    
    @property
    def Lkl(self):
        return self.Lk*(self.gc+self.gg)
    
    @property
    def Cl(self):
        return 4*self.e0*self.eeff/self.xi
    
    @property
    def Z0(self):
        return np.sqrt((self.Lgl+self.Lkl)/self.Cl)
    
    @property
    def vph(self):
        return 1/np.sqrt((self.Lgl+self.Lkl)*self.Cl)
    
    @property
    def ak(self):
        return self.Lkl/(self.Lkl+self.Lgl)
    
    @property
    def fres(self):
        return self.vph/self.l
    
    ##### METHODS #####
    def calc_beta(self,f):
        return 2*np.pi*f/self.vph
    
    def calc_ABCD(self,f):
        beta = self.calc_beta(f)
        Z0 = self.Z0
        return np.array([[np.cos(beta*self.l),1j*Z0*np.sin(beta*self.l)],
                         [1j/Z0*np.sin(beta*self.l),np.cos(beta*self.l)]])
class KID(object):
    '''Object to calculate the input impedance of a KID (coupler + transmission line). 
    f0 in Hz, Z0 in Ohm.'''
    def __init__(self,fres,Qc,Qi,Z0):
        self.fres = fres
        self.Qc = Qc
        self.Qi = Qi
        self.Z0 = Z0
        
    def calc_Zin(self,f):
        #from pag.76, eq. (3.18) of Pieter's thesis
        dx = (f-self.fres)/self.fres
        return self.Z0*((2*self.Qi*np.sqrt(self.Qc/np.pi)*dx-1j*np.sqrt(1/(np.pi*self.Qc)))\
                        /(1+2j*self.Qi*(dx-np.sqrt(1/(np.pi*self.Qc)))))
    
    def calc_ABCD(self,f):
        Zin = self.calc_Zin(f)
        return np.array([[1,0],[1/Zin,1]])