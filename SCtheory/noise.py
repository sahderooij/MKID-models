import SCtheory as SCth
import numpy as np

def Planck1D(hw, kbT):
    return hw/kbT / (np.exp(hw/kbT) - 1)

def Nyquist(hw, kbT, SCsheet): 
    return 4 * kbT * Planck1D(hw, kbT) * np.real(SCth.Zs(hw, kbT, SCsheet))
    