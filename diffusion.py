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

    def __init__(self,
                 SCvol,
                 Teff,
                 L,
                 Lmax=None,
                 K=None,
                 B=None,
                 S=None,
                 D=None):
        self.SCvol = SCvol
        self.Teff = Teff  # K
        self.L = L  # µs^-1
        self.Lmax = Lmax
        self.D = SCvol.SC.D
        if K is None:
            self.K = SCvol.Rbar / (SCvol.d * SCvol.w)  # µm/µs
        else:
            self.K = K

        if B is None:
            self.B = 1 / SCvol.SC.tpb
        else:
            self.B = B

        if S is None:
            self.S = 1 / SCvol.tesc
        else:
            self.S = S


class IC_par:

    def __init__(self, lmbd=0.402, eta_pb=0.59, sigma=10):
        self.lmbd = lmbd  # µm
        self.eta_pb = eta_pb
        self.sigma = sigma  # µm


defaultIC_par = IC_par()


class settings:

    def __init__(
        self,
        D_const=False,
        usesymm=True,
        dt_init=0.001,
        dx_or_fraction=1 / 5,
        adaptivedx=True,
        adaptivedt=True,
        dt_max=10,
        simtime_approx=100,
    ):

        self.D_const = D_const  # Bool
        self.usesymm = usesymm  # Bool
        self.dt_init = dt_init  # µs
        self.dx_or_fraction = dx_or_fraction  # µm or (-)
        self.adaptivedx = adaptivedx  # Bool
        self.adaptivedt = adaptivedt  # Bool
        self.dt_max = dt_max  # µs
        self.simtime_approx = simtime_approx  # µs


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

        # Extract needed parameters from model_par
        # geometry
        self.width = self.model_par.SCvol.w
        self.height = self.model_par.SCvol.d
        self.length = self.model_par.SCvol.l
        # SC properties
        self.Delta = self.model_par.SCvol.SC.D0 * 1e-6  # eV, gap energy
        self.D0 = self.model_par.D  # µm^2/µs, diffusion const.
        self.N0 = self.model_par.SCvol.SC.N0 * 1e6  # eV^-1 µm^-3
        # Quasiparticle equilibrium rates
        self.Q0 = (SCth.nqp(
            consts.Boltzmann / consts.e * 1e6 * self.model_par.Teff,
            self.model_par.SCvol.SC.D0,
            self.model_par.SCvol.SC,
        ) * self.width * self.height)
        self.L = self.model_par.L
        self.K = self.model_par.K
        self.B = self.model_par.B
        self.S = self.model_par.S

        # Quasiparticle non-equilibirium
        self.E_ph = consts.h / consts.e * 1e6 * consts.c / self.IC_par.lmbd  # eV
        self.Nqp_init = self.IC_par.eta_pb * self.E_ph / self.Delta

        # Initialize time axis and dt
        dt = self.settings.dt_init
        self.t_axis = np.zeros(1)
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
        # update dx to valid value close to the one set before
        dx = valid_dx_list[0]
        # store dx in new output list which will contain dx at each timestep
        self.dxlist = [dx]

        self.set_geometry(dx)  # calculate self.x_centers and self.x_borders
        # store x_centers in new output list
        self.x_centers_list = [self.x_centers]

        # initialize state variables

        # set IC to Nqp_init
        self.Qintime = [
            np.exp(-0.5 * (self.x_centers / self.IC_par.sigma)**2) *
            self.Nqp_init / (self.IC_par.sigma * np.sqrt(2 * np.pi))
        ]

        # correct total Nqp if boundary clips off tails due to large sigma
        self.Qintime[0] *= self.Nqp_init / self.integrate(self.Qintime[0], dx)

        self.Nqpintime = np.array([self.integrate(self.Qintime[0], dx)])

        # intialize phonons as 0
        self.Wintime = [np.zeros(len(self.x_centers))]
        self.Nwintime = np.array([self.integrate(self.Wintime[0], dx)])

        # calc thermal density of quasiparticles
        kBTeff_thermal = self.nqp_to_kBT(self.Q0)
        Dfinal = self.D0 * np.sqrt(2 * kBTeff_thermal / (np.pi * self.Delta))

        # run simulation
        i = 0  # keeps track of simulation step
        self.t_elapsed = 0  # keeps track of elapsed time (us)
        t_elapsed_D = (
            0  # keeps track of elapsed time but specifically for adapting dx with time
        )
        sqrtMSD2D = self.IC_par.sigma

        while True:
            # adaptive dx
            if (self.settings.adaptivedx and (i != 0) and
                (dx != valid_dx_list[-1])):
                sqrtMSD = (
                    np.sqrt(2 * Dfinal * t_elapsed_D) + self.IC_par.sigma
                )  # mean squared distance expected from diffusion only
                dx = valid_dx_list[
                    valid_dx_list <= sqrtMSD * self.settings.dx_or_fraction][
                        -1]  # update to the largest allowed dx value below the requested fraction of mean distance
                self.set_geometry(dx)

                # update distributions to new geometry
                Qprev = np.interp(self.x_centers, x_centersprev,
                                  self.Qintime[i])
                Wprev = np.interp(self.x_centers, x_centersprev,
                                  self.Qintime[i])

                #correct for any potential loss of total Nqp, Nw
                Qprev *= self.integrate(self.Qintime[i],
                                        self.dxlist[i]) / self.integrate(
                                            Qprev, dx)
                Wprev *= self.integrate(self.Wintime[i],
                                        self.dxlist[i]) / self.integrate(
                                            Wprev, dx)

                # update elasped diffusion time
                t_elapsed_D += dt

            else:
                Qprev = self.Qintime[i]
                Wprev = self.Wintime[i]

            # do simulation step
            self.dxlist.append(dx)
            self.dtlist.append(dt)
            Q, W = self.CN_step(dt, dx, Qprev, Wprev)
            self.Qintime.append(Q)
            self.Wintime.append(W)
            self.Nqpintime = np.append(self.Nqpintime,
                                       self.integrate(self.Qintime[i + 1], dx))
            self.Nwintime = np.append(self.Nwintime,
                                      self.integrate(self.Wintime[i + 1], dx))
            self.x_centers_list.append(self.x_centers)

            x_centersprev = self.x_centers
            self.t_elapsed += dt
            self.t_axis = np.append(self.t_axis, self.t_elapsed)

            print(f"\rIteration: {i}\tSimtime (us): {self.t_elapsed}", end="")

            if (self.t_elapsed > self.settings.simtime_approx):
                break

            # handle adaptive dt
            if self.settings.adaptivedt:
                dN = np.abs(self.Nqpintime[i] - self.Nqpintime[i + 1])
                dNw = np.abs(self.Nwintime[i] - self.Nwintime[i + 1])
                if i == 0:
                    dNdt = dN * dt  # set at beginning of simulation
                    dNwdt = dNw * dt
                dt = np.min([(dNdt / dN + dt) / 2, (dNwdt / dNw + dt) / 2])
                if dt > self.settings.dt_max:
                    dt = self.settings.dt_max
            i += 1

    def set_geometry(self, dx):
        if self.settings.usesymm:  # set geometry for half the MKID
            self.x_borders = np.arange(0, self.length / 2 + dx / 2, dx)
            self.x_centers = np.arange(dx / 2, self.length / 2, dx)
        else:  # set geometry for a full MKID
            self.x_borders = np.arange(-self.length / 2,
                                       self.length / 2 + dx / 2, dx)
            self.x_centers = np.arange(-self.length / 2 + dx / 2,
                                       self.length / 2, dx)
        return

    def nqp_to_kBT(self, nqp):  # inverts n_qp(T)
        xqp = nqp / (2 * self.N0 * self.height * self.width * self.Delta)
        return np.real(2 * self.Delta / lambertw(4 * np.pi / xqp**2))

    def calc_D(self, kBTeff_x):
        '''calculates energy dependent D at elements, interpolates to borders'''
        return np.interp(
            self.x_borders,
            self.x_centers,
            self.D0 * np.sqrt(2 * kBTeff_x / (np.pi * self.Delta)),
        )

    def diffuse(self, dx, D, Q):
        '''apply diffusion.
        Assumes von Neumann BCs, for Dirichlet use 
        np.pad(Q_prev,(1,1),'constant', 
        constant_values=(0, 0)), disable 'usesymmetry' for this'''
        Q_temp = np.pad(Q, (1, 1), "edge")
        gradient = D * np.diff(Q_temp) / dx
        return (-gradient[:-1] + gradient[1:]) / dx

    def CN_eqs(self, dt, dx, Q_prev, W_prev, QW_next):
        '''the Crank-Nicolson update equations without source term'''
        Q_next = QW_next[:len(Q_prev)]
        W_next = QW_next[-len(W_prev):]

        kBTeff_x = self.nqp_to_kBT(Q_prev + self.Q0)
        if self.settings.D_const:
            D = self.D0
        else:
            D = self.calc_D(kBTeff_x)

        if self.L == 't_eph':
            L = np.min(
                np.array([
                    np.ones(len(kBTeff_x)) * self.model_par.Lmax,
                    self.model_par.SCvol.phtrf / SCth.tau.scat_eph_2D_disorder(
                        kBTeff_x * 1e6, self.model_par.SCvol)
                ]), 0)
        else:
            L = self.L

        return np.append(
            (Q_prev - Q_next + 0.5 * dt *
             (self.diffuse(dx, D, Q_next) - self.K * Q_next**2 - L * Q_next +
              2 * self.B * W_next + self.diffuse(dx, D, Q_prev) -
              self.K * Q_prev**2 - L * Q_prev + 2 * self.B * W_prev))**2,
            (W_prev - W_next + 0.5 * dt *
             (L / 2 * Q_next + self.K / 2 * Q_next**2 -
              (self.B + self.S) * W_next + L / 2 * Q_prev +
              self.K / 2 * Q_prev**2 - (self.B + self.S) * W_prev))**2)

    def CN_step(self, dt, dx, Q_prev, W_prev):
        '''fsolve the CN equations, with the previous step as initial guess'''

        kBTeff_x = self.nqp_to_kBT(Q_prev + self.Q0)
        if self.L == 't_eph':
            L = np.min(
                np.array([
                    np.ones(len(kBTeff_x)) * self.model_par.Lmax,
                    self.model_par.SCvol.phtrf / SCth.tau.scat_eph_2D_disorder(
                        kBTeff_x * 1e6, self.model_par.SCvol)
                ]), 0)
        else:
            L = self.L
        Wguess = (L - self.K * Q_prev) * Q_prev
        QW_next = fsolve(
            lambda QW_next: self.CN_eqs(dt, dx, Q_prev, W_prev, QW_next),
            np.append(Q_prev, Wguess))
        return QW_next[:len(Q_prev)], QW_next[-len(W_prev):]

    def integrate(self, Q, dx):
        '''integrate nqp to find Nqp'''
        if self.settings.usesymm:
            Nqp = np.sum(Q, axis=-1) * dx * 2
        else:
            Nqp = np.sum(Q, axis=-1) * dx
        return Nqp
