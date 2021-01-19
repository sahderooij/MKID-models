'''
The kidata package contains modules that handle, process, calculate variables and visualizes data from MKID experiments. It contains the following modules:

io -- to read and handle the data files.
calc -- uses the data to calculate parameters, like lifetimes or kinetic induction fraction.
noise -- postprocesses noise files, and calculates the PSDs.
filters -- adjusts the PSDs with a certain filter, like deleting the amplifier noise.
plot -- visualizes the (calculated) data.'''

import kidata.calc,kidata.filters,kidata.io,kidata.noise,kidata.plot