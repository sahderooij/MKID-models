import numpy as np
import scipy.integrate as integrate
from scipy import interpolate
from scipy.optimize import minimize_scalar as minisc
import warnings


# Fermi-Dirac distribution
def f(E, kbT):
    with np.errstate(over='raise',under='ignore'):
        try:
            return 1 / (1 + np.exp(E/kbT))
        except FloatingPointError:
            return np.exp(-E/kbT)

# Calculation for complex conductivity
def cinduct(hw, D, kbT):
    def integrand11(E, hw, D, kbT):
        nume = 2 * (f(E, kbT) - f(E + hw, kbT)) * (E ** 2 + D ** 2 + hw * E)
        deno = hw * ((E ** 2 - D ** 2) * ((E + hw) ** 2 - D ** 2)) ** 0.5
        return nume / deno
        
    def integrand12(E, hw, D, kbT):
        nume = (1 - 2 * f(E + hw, kbT)) * (E ** 2 + D ** 2 + hw * E)
        deno = hw * ((E ** 2 - D ** 2) * ((E + hw) ** 2 - D ** 2)) ** 0.5
        return nume / deno

    def integrand2(E, hw, D, kbT):
        nume = (1 - 2 * f(E + hw, kbT)) * (E ** 2 + D ** 2 + hw * E)
        deno = hw * ((D ** 2 - E ** 2) * ((E + hw) ** 2 - D ** 2)) ** 0.5
        return nume / deno

    s1 = integrate.quad(integrand11, D, np.inf, args=(hw, D, kbT))[0]
    if hw > 2 * D:
        s1 -= integrate.quad(integrand12, D - hw, -D, args=(hw, D, kbT))[0]
    s2 = integrate.quad(integrand2, np.max(
        [D - hw, -D]), D,args=(hw, D, kbT))[0]
    return s1,s2

def Vsc(kbTc,N0,kbTD):
    D0 = 1.76*kbTc
    def integrand1(E, D):
        return 1/np.sqrt(E**2-D**2)
    return 1/(integrate.quad(integrand1, D0, kbTD,
                                 args=(D0,))[0]*N0)
def load_Ddata(N0,Vsc,kbTD):
    if (
        (N0 == 1.72e4) & (kbTD == 37312.0) & (Vsc == 9.663743323443183e-06)
    ):  # speed-up with interpolate
        Ddata = np.load("C:\\Users\\Steven\\Google Drive\\AP\\Thesis\\Coding\\Ddata_Al_1_2.npy")
    elif (
        (N0 == 1.72e4) & (kbTD == 37312.0) & (Vsc == 9.736267969683833e-06)
    ):
        Ddata = np.load("C:\\Users\\Steven\\Google Drive\\AP\\Thesis\\Coding\\Ddata_Al_1_255.npy")
    elif (
        (N0 == 1.72e4) & (kbTD == 37312.0) & (Vsc == 1.565803618633812e-05)
    ):
        Ddata = np.load("C:\\Users\\Steven\\Google Drive\\AP\\Thesis\\Coding\\Ddata_Al_12.npy")
    elif (
        (N0 == 1.72e4) & (kbTD == 37312.0) & (Vsc == 9.856715467321348e-06)
    ):
        Ddata = np.load("C:\\Users\\Steven\\Google Drive\\AP\\Thesis\\Coding\\Ddata_Al_1_35.npy")
    elif (
        (N0 == 1.72e4) & (kbTD == 37312.0) & (Vsc == 9.716702000311747e-06)
    ):
        Ddata = np.load("C:\\Users\\Steven\\Google Drive\\AP\\Thesis\\Coding\\Ddata_Al_1_24.npy")
    elif (
        (N0 == 1.72e4) & (kbTD == 37312.0) & (Vsc == 9.55417723118179e-06)
    ):
        Ddata = np.load("C:\\Users\\Steven\\Google Drive\\AP\\Thesis\\Coding\\Ddata_Al_1_12.npy")
    else:
        Ddata = None
    return Ddata

# Calculation for energy gap D
def D(kbT, N0, Vsc, kbTD):
    Ddata = load_Ddata(N0,Vsc,kbTD)
    if Ddata is not None:
        Dspl = interpolate.splrep(Ddata[0, :], Ddata[1, :], s=0)
        return np.clip(interpolate.splev(kbT, Dspl),0,None)
    else:
        warnings.warn('D takes long.. \n N0={}\n kbTD={}\n Vsc={}'.format(N0,kbTD,Vsc))
        def integrandD(E, D, kbT, N0, Vsc):
            return N0 * Vsc * (1 - 2 * f(E, kbT)) / np.sqrt(E ** 2 - D ** 2)

        def dint(D, kbT, N0, Vsc, kbTD):
            return np.abs(
                integrate.quad(integrandD, D, kbTD,
                               args=(D, kbT, N0, Vsc))[0] - 1
            )

        res = minisc(dint, args=(kbT, N0, Vsc, kbTD))
        if res.success:
            return np.clip(res.x,0,None)

# Calculation for n_qp
def nqp(kbT, D, N0):
    if (kbT<D/20).any():
        return 2*N0*np.sqrt(2*np.pi*kbT*D)*np.exp(-D/kbT)
    else:
        def integrand(E, kbT, D, N0):
            return 4 * N0 * E / np.sqrt(E ** 2 - D ** 2) * f(E, kbT)
        return integrate.quad(integrand, D, np.inf, args=(kbT, D, N0))[0]
            

# Calculation for effective temperature
def kbTeff(N_qp, N0, V, Vsc, kbTD):
    Ddata = load_Ddata(N0,Vsc,kbTD)
    if Ddata is not None:
        kbTspl = interpolate.splrep(Ddata[2,:],Ddata[0,:])
        return interpolate.splev(N_qp/V,kbTspl)
    else:
        warnings.warn('kbTeff takes long.. \n N0={}\n kbTD={}\n Vsc={}'.format(N0,kbTD,Vsc))
        def minfunc(kbT, N_qp, N0, V, Vsc, kbTD):
            Dt = D(kbT, N0, Vsc, kbTD)
            return np.abs(nqp(kbT, Dt, N0) - N_qp/V)
        res = minisc(
            minfunc,
            bounds = (0,1*86.17),
            args=(N_qp, N0, V, Vsc, kbTD), 
            method="bounded",
            options = {'xatol':1e-15}
        )
        if res.success:
            return res.x
    
# Calculation of S21 and A,theta
def beta(lbd0, d, D, D0, kbT):
    lbd = lbd0 * 1 / np.sqrt(D / D0 * np.tanh(D / (2 * kbT)))
    return 1 + 2 * d / (lbd * np.sinh(2 * d / lbd))


def Qi(s1, s2, ak, lbd0, d, D, D0, kbT):
    b = beta(lbd0, d, D, D0, kbT)
    return 2 * s2 / (ak * b * s1)


def hwres(s2, hw0, s20, ak, lbd0, d, D, D0, kbT):
    with np.errstate(all='raise'):
        b = beta(lbd0, d, D, D0, kbT)
        return hw0 * (
            1 + ak * b / 4 / s20 * (s2 - s20)
        )  # note that is a linearized approach

def S21(Qi, Qc, hwread, dhw, hwres):
    Q = Qi * Qc / (Qi + Qc)
    dhw += hwread - hwres
    return (Q / Qi + 2j * Q * dhw / hwres) / (1 + 2j * Q * dhw / hwres)

# Calculate hwread
def hwread(hw0, kbT0, ak, lbd0, d, D_, D0, kbT, N0, Vsc, kbTD):
    D_0 = D(kbT0, N0, Vsc, kbTD)
    s20 = cinduct(hw0, D_0, kbT0)[1]

    def minfuc(hw, hw0, s20, ak, lbd0, d, D_, D0, kbT):
        s1, s2 = cinduct(hw, D_, kbT)
        return np.abs(hwres(s2, hw0, s20, ak, lbd0, d, D_, D0, kbT) - hw)

    res = minisc(
        minfuc,
        bracket=(.5*hw0,hw0,2*hw0),
        args=(hw0, s20, ak, lbd0, d, D_, D0, kbT),
        method="brent",
        options={"xtol": 1e-21},
    )
    if res.success:
        return res.x
    
#Number of subgap phonons with Debye:
def calc_Nwsg(kbT,V,D,e):
    def integrand(E,kbT,V):
        return 3*V*E**2/(2*np.pi*(6.582e-4)**2*(6.3e3)**3*(np.exp(E/kbT)-1))
    return integrate.quad(integrand,e+D,2*D,args=(kbT,V))[0]


