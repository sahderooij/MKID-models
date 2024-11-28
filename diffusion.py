import numpy as np
import scipy.constants as consts
from scipy.optimize import fsolve
from scipy.special import lambertw
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
import re

import SCtheory as SCth


# everything in um, us, K and eV  and combinations of them unless stated differently
class model_par:
    def __init__(self, SCvol, Teff, L, K=None):
        self.SCvol = SCvol
        self.Teff = Teff  # K
        self.L = L  # µs^-1
        if K is None:
            self.K = SCvol.Rbar / (SCvol.d * SCvol.w) # µm/µs
        else:
            self.K = K


class IC_par:
    def __init__(self, lmbd=0.402, eta_pb=0.59, sigma=10, trickle_time=False):
        self.lmbd = lmbd  # µm
        self.eta_pb = eta_pb
        self.sigma = sigma  # µm
        self.trickle_time = trickle_time  # µs

defaultIC_par = IC_par()

class settings:
    def __init__(
        self,
        D_const=False,
        approx2D=False,
        usesymm=True,
        dt_init=.001,
        dx_or_fraction=1/5,
        adaptivedx=True,
        adaptivedt=True,
        dt_max=10,
        simtime_approx=100,
    ):
        """2D option is sketchy, don't trust without more checking. 
        If loading after a sim is too slow or requires too much space, increase ringingdtinterp."""
        self.D_const = D_const # Bool 
        self.approx2D = approx2D # Bool
        self.usesymm = usesymm # Bool
        self.dt_init = dt_init # µs
        self.dx_or_fraction = dx_or_fraction # µm or (-)
        self.adaptivedx = adaptivedx # Bool
        self.adaptivedt = adaptivedt # Bool
        self.dt_max = dt_max # µs
        self.simtime_approx = simtime_approx # µs

defaultsettings = settings()

class sim:
    def __init__(
        self,
        model_par,
        IC_par=defaultIC_par,
        settings=defaultsettings,
    ):
        self.model_par = model_par
        self.IC_par = IC_par
        self.settings = settings

        ##  Extract needed parameters from model_par
        # geometry
        self.width = self.model_par.SCvol.w
        self.height = self.model_par.SCvol.d
        self.length = self.model_par.SCvol.l
        # SC properties
        self.Delta = self.model_par.SCvol.SC.D0 * 1e-6 # eV, gap energy
        self.D0 = self.model_par.SCvol.SC.D # µm^2/µs, diffusion const.
        self.N0 = self.model_par.SCvol.SC.N0 * 1e6 # eV^-1 µm^-3
        # Quasiparticle equilibrium rates
        self.Q0 = SCth.nqp(consts.Boltzmann/consts.e*1e6 * self.model_par.Teff, 
                           self.model_par.SCvol.SC.D0, self.model_par.SCvol.SC) * self.width * self.height
        self.L = self.model_par.L
        self.K = self.model_par.K # previous: self.L / (2 * self.Q0)
        # Quasiparticle non-equilibirium
        self.E_ph = consts.h / consts.e * 1e6 * consts.c / self.IC_par.lmbd # eV
        self.Nqp_init = self.IC_par.eta_pb * self.E_ph / self.Delta

        # Initialize time axis and dt
        dt = self.settings.dt_init
        self.t_axis = [0]
        self.dtlist = [self.settings.dt_init]

        # set geometry
        if self.settings.adaptivedx:  # if using adaptive dx option
            dx = self.IC_par.sigma * self.settings.dx_or_fraction  # set dx as fraction
        else:  # otherwise set constant dx value
            dx = self.settings.dx_or_fraction
        if (
            self.settings.usesymm
        ):  # calculate list of possible dx values that divide the domain cleanly, from small to large
            maxdiv = int(np.ceil(self.length / 2 / dx))
            valid_dx_list = self.length / 2 / np.arange(1, maxdiv + 0.5)[::-1]
        else:
            maxdiv = int(np.ceil(self.length / dx))
            valid_dx_list = self.length / np.arange(1, maxdiv + 0.5)[::-1]
        dx = valid_dx_list[0]  # update dx to valid value close to the one set before
        self.dxlist = [
            dx
        ]  # store dx in new output list which will contain dx at each timestep
        
        self.set_geometry(
            dx, self.length
        )  # calculate self.x_centers and self.x_borders
        self.x_centers_list = [self.x_centers]  # store x_centers in new output list

        # initialize state variables
        if self.IC_par.trickle_time:  # if using forcing term instead of simple IC
            self.Qintime = [np.zeros_like(self.x_centers)]  # set IC to zero
        else:
            self.Qintime = [
                np.exp(-0.5 * (self.x_centers / self.IC_par.sigma) ** 2)
                * self.Nqp_init
                / (self.IC_par.sigma * np.sqrt(2 * np.pi))
            ]  # set IC to Nqp_init
            self.Qintime[0] = (
                self.Qintime[0] * self.Nqp_init / self.integrate(self.Qintime[0], dx)
            )  # correct total Nqp if boundary clips off tails due to large sigma

        self.Nqpintime = [
            self.integrate(self.Qintime[0], dx)
        ]  # calculate integral of density, store in new list

        # calc thermal density of quasiparticles
        Teff_thermal = self.nqp_to_T(
            self.Q0,
            self.N0,
            self.Delta,
            self.height,
            self.width,
        )  # calculate effective temperature at each volume
        Dfinal = self.D0 * np.sqrt(
            2 * consts.Boltzmann / consts.e * Teff_thermal / (np.pi * self.Delta)
        )  # calculate D array for steady state
        
        # run simulation
        i = 0  # keeps track of simulation step
        self.t_elapsed = 0  # keeps track of elapsed time (us)
        t_elapsed_D = 0  # keeps track of elapsed time but specifically for adapting dx with time
        sqrtMSD2D = self.IC_par.sigma
        if (
            self.IC_par.trickle_time
        ):  # pause adaptive dx as long as the forcing term is still large
            dxAdaptPause = True
        else:
            dxAdaptPause = False

        while True:  # kind of a do-while loop
            if (
                self.t_elapsed > 3 * self.IC_par.trickle_time
            ) and dxAdaptPause:  # the integral from 0 to 3*tau already contains >95% of the surface under the exponential.
                dxAdaptPause = False  # resume the adaptive dx optimization

            # handle adaptive dx
            if (
                self.settings.adaptivedx
                and (i != 0)
                and (dx != valid_dx_list[-1])
                and (dxAdaptPause == False)
            ):
                sqrtMSD = (
                    np.sqrt(2 * Dfinal * t_elapsed_D) + self.IC_par.sigma
                )  # mean squared distance expected from diffusion only (after forcing is negligible)
                dx = valid_dx_list[valid_dx_list <= sqrtMSD * self.settings.dx_or_fraction][
                    -1
                ]  # update to the largest allowed dx value below the requested fraction of mean distance
                self.set_geometry(
                    dx, self.length
                )  # calculate new geometry of simulation
                Qprev = np.interp(
                    self.x_centers, x_centersprev, self.Qintime[i]
                )  # update the qp distribution to new geometry
                Qprev *= self.integrate(
                    self.Qintime[i], self.dxlist[i]
                ) / self.integrate(
                    Qprev, dx
                )  # correct for any potential loss of total Nqp by normalizing
                t_elapsed_D += dt  # update elapsed diffusion time
            else:
                Qprev = self.Qintime[i]

            # update diffusion
            if self.settings.D_const:
                D = self.D0
            else:  # calculate local effective temperature
                Teff_x = self.nqp_to_T(
                    Qprev + self.Q0,
                    self.N0,
                    self.Delta,
                    self.height,
                    self.width,
                )
                D = self.calc_D(
                    self.D0, Teff_x, self.Delta
                )  # calculate new location dependent diffusion
            # 2D approximation
            sqrtMSD2D += 4 * self.D0 * dt
            if self.settings.approx2D and (sqrtMSD2D < self.width / 2):
                D = (
                    8 * D**2 * (self.t_elapsed + dt)
                )  # use higher diffusion rate to simulate 2D diffusion if required

            # do simulation step
            self.dxlist.append(dx)
            self.dtlist.append(dt)
            self.Qintime.append(self.CN_step(dt, dx, D, self.L, self.K, Qprev))
            self.Nqpintime.append(self.integrate(self.Qintime[i + 1], dx))
            self.x_centers_list.append(self.x_centers)

            x_centersprev = self.x_centers
            self.t_elapsed += dt
            self.t_axis.append(self.t_elapsed)
            print(
                f"\rIteration: {i}\tSimtime (us): {self.t_elapsed}", end=""
            )  # print and update progress counter

            if (
                self.t_elapsed > self.settings.simtime_approx
            ):  # check whether simulation has reached required time
                break

            # handle adaptive dt
            if self.settings.adaptivedt and (dt <= self.settings.dt_max):
                dN = np.abs(
                    self.Nqpintime[i] - self.Nqpintime[i + 1]
                )  # calculate difference in Nqp from previous step.
                if i == 0:
                    dNdt = dN * dt  # set at beginning of simulation
                if dN != 0:
                    dt = (
                        dNdt / dN + dt
                    ) / 2  # update dt, taking the mean stabilizes oscillations due to trickle and adaptive dt both depending on dt value.
            i += 1

        self.t_axis = np.array(self.t_axis)  # save time array
        self.Nqpintime = np.array(self.Nqpintime)  # and data array

    def set_geometry(self, dx, length):
        if self.settings.usesymm:  # set geometry for half the MKID
            self.x_borders = np.arange(0, length / 2 + dx / 2, dx)
            self.x_centers = np.arange(dx / 2, length / 2, dx)
        else:  # set geometry for a full MKID
            self.x_borders = np.arange(-length / 2, length / 2 + dx / 2, dx)
            self.x_centers = np.arange(-length / 2 + dx / 2, length / 2, dx)
        return

    def nqp_to_T(self, nqp, N0, Delta, height, width):  # inverts n_qp(T)
        a = (
            2
            * N0
            * height
            * width
            * np.sqrt(2 * np.pi * consts.Boltzmann / consts.e * Delta)
        )
        b = Delta / (consts.Boltzmann / consts.e)
        return np.real(2 * b / lambertw(2 * a**2 * b / (nqp**2)))

    def calc_D(
        self, D0, Teff_x, Delta
    ):  # calculates energy dependent D at elements, interpolates to borders
        return np.interp(
            self.x_borders,
            self.x_centers,
            D0 * np.sqrt(2 * consts.Boltzmann / consts.e * Teff_x
                         / (np.pi * Delta)),
        )

    def diffuse(self, dx, D, Q_prev):  # apply diffusion
        Q_temp = np.pad(
            Q_prev, (1, 1), "edge"
        )  # Assumes von Neumann BCs, for Dirichlet use np.pad(Q_prev,(1,1),'constant', constant_values=(0, 0)), disable 'usesymmetry' for this
        gradient = D * np.diff(Q_temp) / dx
        return (-gradient[:-1] + gradient[1:]) / dx

    def source(self, dt, dx):  # apply quadratically decaying source term
        S_next = np.exp(-0.5 * (self.x_centers / self.IC_par.sigma) ** 2) / (
            self.IC_par.sigma * np.sqrt(2 * np.pi)
        )
        S_prev = (
            (self.Nqp_init / self.IC_par.trickle_time)
            * np.exp(-(self.t_elapsed) / self.IC_par.trickle_time)
            * np.exp(-0.5 * (self.x_centers / self.IC_par.sigma) ** 2)
            / (self.IC_par.sigma * np.sqrt(2 * np.pi))
        )
        integ_next = self.integrate(S_next, dx)
        integ_prev = self.integrate(S_prev, dx)
        if (integ_next > 1e-6) and (integ_prev > 1e-6):
            S_next = (
                S_next
                * (self.Nqp_init / self.IC_par.trickle_time)
                * np.exp(-(self.t_elapsed + dt) / self.IC_par.trickle_time)
                / integ_next
            )
            S_prev = (
                S_prev
                * (self.Nqp_init / self.IC_par.trickle_time)
                * np.exp(-(self.t_elapsed) / self.IC_par.trickle_time)
                / integ_prev
            )
        return S_next, S_prev

    def CN_eqs_source(
        self, dt, dx, D, L, K, Q_prev, Q_next
    ):  # the Crank-Nicolson update equations in case of a source term
        S_next, S_prev = self.source(dt, dx)
        return (
            Q_prev
            - Q_next
            + 0.5
            * dt
            * (
                self.diffuse(dx, D, Q_next)
                - K * Q_next**2
                - L * Q_next
                + S_next
                + self.diffuse(dx, D, Q_prev)
                - K * Q_prev**2
                - L * Q_prev
                + S_prev
            )
        )

    def CN_eqs(
        self, dt, dx, D, L, K, Q_prev, Q_next
    ):  # the Crank-Nicolson update equations without source term
        return (
            Q_prev
            - Q_next
            + 0.5
            * dt
            * (
                self.diffuse(dx, D, Q_next)
                - K * Q_next**2
                - L * Q_next
                + self.diffuse(dx, D, Q_prev)
                - K * Q_prev**2
                - L * Q_prev
            )
        )

    def CN_step(
        self, dt, dx, D, L, K, Q_prev
    ):  # fsolve the CN equations, with the previous step/2 as initial guess
        if self.IC_par.trickle_time:  # if we have a source term:
            return fsolve(
                lambda Q_next: self.CN_eqs_source(dt, dx, D, L, K, Q_prev, Q_next),
                Q_prev / 2,
            )
        else:
            return fsolve(
                lambda Q_next: self.CN_eqs(dt, dx, D, L, K, Q_prev, Q_next), Q_prev / 2
            )

    def integrate(self, Q, dx):  # integrate nqp to find Nqp
        if self.settings.usesymm:
            Nqp = np.sum(Q, axis=-1) * dx * 2
        else:
            Nqp = np.sum(Q, axis=-1) * dx
        return Nqp
