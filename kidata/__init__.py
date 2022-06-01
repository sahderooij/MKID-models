'''
The kidata package contains modules that handle, process, calculate variables and visualizes data from MKID experiments. It contains the following modules:

io -- to read and handle the data files.
calc -- uses the data to calculate parameters, like lifetimes or kinetic induction fraction.
noise -- postprocesses noise files, and calculates PSDs.
filters -- adjusts the PSDs with a certain filter, like deleting the amplifier noise.
plot -- visualizes the (calculated) data.
S21 -- Fits the resonator dips to get F0 and Q-factors.'''

import kidata.calc, kidata.filters, kidata.io, kidata.noise, kidata.plot, kidata.pulse, kidata.IQ, kidata.S21