# MKID-models

This repository contains all code related to the modelling of the response of MKIDs. Both the analysis and theoretically modelling of MKIDs is addressed in these sub packages/modules:

Packages:
- **kidata** is a package that deals with MKID data from experiments
- **kidesign** implements theoretical CPW equations for the design of MKIDs and functions to deal with SONNET simulation data 

Modules:
- **KID** defines an KID object, which is used for to predict the single photon response of MKIDs
- **kidcalc** implements all superconductor theory that governs MKIDs (mostly BSC, Mattis-Bardeen and Kaplan)
- **trapmodels** is used for trapping models, based on modified Rothwarf-Taylor equations, in a attempt to explain the observed reduction in noise level
- **SC** contains physical constants of different superconductors, which is used through-out the repository.

Data:
- **Ddata** contains calculated gap energies over temperature for different $T_c$, to speed up the calculation of $\Delta(T)$.

