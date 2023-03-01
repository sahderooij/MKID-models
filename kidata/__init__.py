'''
The kidata package contains modules that handle, process, calculate variables and visualizes data from MKID experiments. It contains the following modules:

io -- to read and handle the data files.
calc -- uses the data to calculate parameters, like lifetimes or kinetic induction fraction.
noise -- postprocesses noise files, and calculates PSDs.
filters -- adjusts the PSDs with a certain filter, like deleting the amplifier noise.
plot -- visualizes the (calculated) data.
S21 -- Fits the resonator dips to get F0 and Q-factors.'''

from . import calc, io, noise, plot, pulse, IQ, S21, pulse, THz