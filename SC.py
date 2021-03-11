'''This module includes the physical constants needed for the calculations. Instances of the SC class containt superconductor specific constants, such as the Debye Temperature (TD), Density of States at Fermi surface (N0) etc. '''

import scipy.constants as const
import os
import numpy as np

#Parent class:
class Superconductor(object):
    def __init__(self,name,Tc,TD,N0,t0,tpb,tesc,V,lbd0,d):
        self.name = name
        self.kbTc = Tc*const.Boltzmann/const.e*1e6 #critical Temperature in µeV
        self.kbTD = TD*const.Boltzmann/const.e*1e6 #Debye Energy in µeV
        self.N0 = N0 #Electronic DoS at Fermi Surface [µeV^-1 µm^-3]
        self.t0 = t0 #electron-phonon interaction time [µs]
        self.tpb = tpb #phonon pair-breaking time [µs]
        self.tesc = tesc #phonon escape time [µs]
        self.V = V #Volume [µm^3]
        self.lbd0 = lbd0 #magnetic penetration depth at T=0 [µm]
        self.d = d #thickness [µm]
        
    @property
    def Vsc(self):
        '''Calculates the superconducting coupling strength in BSC-theory 
        from the BSC relation 2D=3.52kbTc.'''
        D0 = 1.76*self.kbTc # BSC-relation
        def integrand1(E, D):
            return 1/np.sqrt(E**2-D**2)
        return 1/(integrate.quad(integrand1, D0, self.kbTD,
                                     args=(D0,))[0]*self.N0)
    
    @property
    def Ddata(self):
        '''To speed up the calculation for Delta, an interpolation of generated values is used.
        Ddata_{SC}_{Tc}.npy contains this data, where a new one is needed 
        for each superconductor (SC) and crictical temperature (Tc).
        This function returns the Ddata array.'''

        Ddataloc = os.path.dirname(__file__)+ '/Ddata/'
        if self.name != '':
            Tc = str(np.around(self.kbTc/(const.Boltzmann/const.e*1e6)
                               ,3)).replace('0','').replace('.','_')
            try:
                Ddata = np.load(Ddataloc + 
                    f'Ddata_{self.name}_{Tc}.npy')
            except FileNotFoundError:
                Ddata = None
        else: 
            Ddata = None
        return Ddata
    
    @property
    def D0(self):
        return 1.76*self.kbTc
    
    def set_tesc(self,Chipnum,KIDnum,**kwargs):
        import kidata
        tesc = kidata.calc.tesc(Chipnum,KIDnum,self,**kwargs)
        self.tesc = tesc
        
#Sub-classes, which are the actual superconductors
class Al(Superconductor):
    def __init__(self,Tc=1.2,tesc=.1e-3,V=1e3,lbd0=.092,d=.05):
        '''The default values should be adjusted per device, 
        as they vary from device to device. The hardcoded arguments for the
        __init__() method, are standard constants for Al'''
        super().__init__('Al',Tc,433,1.72e4,.44,.28e-3,tesc,V,lbd0,d)     

###################################################################################
def init_SC(Chipnum,KIDnum,SC_class=Al,set_tesc=True,**tesckwargs):
    '''This function returns an SC instance (which is defined with the SC_class argument), 
    with the parameters initialized from data. 
    Tc, V, and thickness d are from the S21 measurement, and tesc is set with the 
    set_tesc method which uses kidata.calc.tesc() to estimate the phonon escape time.'''
    import kidata
    S21data = kidata.io.get_S21data(Chipnum,KIDnum)
    SC_inst = SC_class(Tc=S21data[0,21],V=S21data[0,14],d=S21data[0,25])
    if set_tesc:
        SC_inst.set_tesc(Chipnum,KIDnum,**tesckwargs)
    return SC_inst
    
        
        