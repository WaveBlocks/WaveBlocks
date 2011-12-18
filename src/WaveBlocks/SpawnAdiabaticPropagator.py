"""The WaveBlocks Project

This file contains a very simple spawning propagator class
for wavepackets and spawning in the adiabatic case.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from functools import partial
import numpy as np
import scipy.linalg as spla

from Propagator import Propagator
from HagedornWavepacket import HagedornWavepacket
from AdiabaticSpawner import AdiabaticSpawner


class SpawnAdiabaticPropagator(Propagator):
    """This class can numerically propagate given initial values $\Ket{\Psi}$ in
    a potential $V\ofs{x}$. The propagation is done for several given homogeneous
    Hagedorn wavepackets neglecting interaction."""

    def __init__(self, potential, packet, leading_component, parameters):
        """Initialize a new I{SpawnAdiabaticPropagator} instance.
        @param potential: The potential the wavepacket $\Ket{\Psi}$ feels during the time propagation.
        @param packet: The initial homogeneous Hagedorn wavepacket we propagate in time.
        @param leading_component: The leading component index $\chi$.
        @raise ValueError: If the number of components of $\Ket{\Psi}$ does not
        match the number of energy levels $\lambda_i$ of the potential.
        """
        if packet.get_number_components() != potential.get_number_components():
            raise ValueError("Wave packet does not match to the given potential!")

        #: The potential $V\ofs{x}$ the packet feels.
        self.potential = potential

        #: Number $N$ of components the wavepacket $\Ket{\Psi}$ has got.
        self.number_components = self.potential.get_number_components()

        #: The Hagedorn wavepacket.
        self.packets = [ (packet,leading_component) ]

        # Cache some parameter values for efficiency
        self.parameters = parameters
        self.dt = parameters["dt"]
        self.eps = parameters["eps"]
        self.K = parameters["spawn_K0"]
        self.threshold = parameters["spawn_threshold"]

        self.spawn_condition = lambda time, packet, component: (time >= 5.25 and time < 5.25+self.parameters["dt"])
        #self.spawn_condition = lambda time, packet, component: (P.get_norm(component=component) >= self.threshold)

        # Decide about the matrix exponential algorithm to use
        method = parameters["matrix_exponential"]

        if method == "pade":
            from MatrixExponential import matrix_exp_pade
            self.__dict__["matrix_exponential"] = matrix_exp_pade
        elif method == "arnoldi":
            from MatrixExponential import matrix_exp_arnoldi
            arnoldi_steps = min(parameters["basis_size"], parameters["arnoldi_steps"])
            self.__dict__["matrix_exponential"] = partial(matrix_exp_arnoldi, k=arnoldi_steps)
        else:
            raise ValueError("Unknown matrix exponential algorithm")

        # Precalculate the potential splitting
        self.potential.calculate_local_quadratic(diagonal_component=leading_component)
        self.potential.calculate_local_remainder(diagonal_component=leading_component)


    def __str__(self):
        """Prepare a printable string representing the I{SpawnAdiabaticPropagator} instance."""
        return "Prapagation and spawning in the adabatic case."


    def get_number_components(self):
        """@return: The number $N$ of components $\Phi_i$ of $\Ket{\Psi}$."""
        return self.number_components


    def get_potential(self):
        """@return: The I{MatrixPotential} instance used for time propagation."""
        return self.potential


    def get_number_packets(self):
        """@return: The number of active packets currently in the simulation."""
        return len(self.packets)


    def get_wavepackets(self, packet=None):
        """@return: A list of I{HagedornWavepacket} instances that represents the
        current wavepackets."""
        if packet is None:
            return [ p[0] for p in self.packets ]
        else:
            return self.packets[packet][0]


    def propagate(self):
        """Given the wavepacket $\Psi$ at time $t$, calculate a new wavepacket
        at time $t + \tau$. We perform exactly one timestep $\tau$ here.
        """
        dt = self.dt

        # Perform spawning in necessary
        todo = self.should_spwan()
        for info in todo:
            self.spawn(info)

        # Propagate all packets
        for packet, leading_chi in self.packets:
            # Do a kinetic step of dt/2
            packet.q = packet.q + 0.5*dt * packet.p
            packet.Q = packet.Q + 0.5*dt * packet.P
            packet.S = packet.S + 0.25*dt * packet.p**2

            # Do a potential step with the local quadratic part
            V = self.potential.evaluate_local_quadratic_at(packet.q, diagonal_component=leading_chi)

            packet.p = packet.p - dt * V[1]
            packet.P = packet.P - dt * V[2] * packet.Q
            packet.S = packet.S - dt * V[0]

            # Do a potential step with the local non-quadratic taylor remainder
            quadrature = packet.get_quadrature()
            F = quadrature.build_matrix(packet, partial(self.potential.evaluate_local_remainder_at, diagonal_component=leading_chi))

            coefficients = packet.get_coefficient_vector()
            coefficients = self.matrix_exponential(F, coefficients, dt/self.eps**2)
            packet.set_coefficient_vector(coefficients)

            # Do a kinetic step of dt/2
            packet.q = packet.q + 0.5 * dt * packet.p
            packet.Q = packet.Q + 0.5 * dt * packet.P
            packet.S = packet.S + 0.25 * dt * packet.p**2


    def should_spwan(self):
        """Check if there is a reason to spawn a new wavepacket.
        """
        spawn_todo = []

        # What do we have to do now?
        # For each packet in the simulation
        #   Check if we should spawn

        for packet, leading_chi in self.packets:
            # Spawn condition fulfilled?
            #should_spawn = self.spawn_condition(packet)

            c = packet.get_coefficients()
            c = np.squeeze(c[0][:])

            #n_low = spla.norm(c[:self.K])
            n_high = spla.norm(c[self.K:])

            should_spawn = (n_high >= self.threshold)

            if should_spawn:
                spawn_todo.append(packet)

        return spawn_todo


    def spawn(self, info):
        """Really spawn the wavepackets.
        """
        WP = info

        # Initialize an empty wavepacket for spawning
        SWP = HagedornWavepacket(self.parameters)
        SWP.set_quadrature(None)

        # Initialize a Spawner
        AS = AdiabaticSpawner(self.parameters)

        # Estimate the parameter set Pi
        Pi = AS.estimate_parameters(WP, component=0)

        # Spawn a new packet
        if Pi is not None:
            SWP.set_parameters(Pi)

            # Project the coefficients from mother to child
            AS.project_coefficients(WP, SWP)

            # Append the spawned packet to the world
            self.packets.append((SWP,0))
