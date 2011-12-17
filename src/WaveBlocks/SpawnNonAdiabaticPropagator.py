"""The WaveBlocks Project

This file contains a very simple spawning propagator class
for wavepackets and gaussian spawning.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from functools import partial
import numpy as np
import scipy.linalg as spla

from Propagator import Propagator
from HagedornWavepacket import HagedornWavepacket
from NonAdiabaticSpawner import NonAdiabaticSpawner


class SpawnNonAdiabaticPropagator(Propagator):
    """This class can numerically propagate given initial values $\Ket{\Psi}$ in
    a potential $V\ofs{x}$. The propagation is done for a given homogeneous
    Hagedorn wavepacket."""

    def __init__(self, potential, packet, leading_component, parameters):
        """Initialize a new I{HagedornPropagator} instance.
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

        #: The leading component $\chi$ is the index of the eigenvalue of the
        #: potential that is responsible for propagating the Hagedorn parameters.
        self.leading = leading_component

        #: The Hagedorn wavepacket.
        # TODO: magics
        self.packets = [ (packet,0) ]

        # Cache some parameter values for efficiency
        self.parameters = parameters
        self.dt = parameters["dt"]
        self.eps = parameters["eps"]
        self.threshold = parameters["spawn_threshold"]

        # The quadrature instance matching the packet
        self.quadrature = packet.get_quadrature()

        # todo: put this in the ParameterProvider
        self.already_spawned = False

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
        self.potential.calculate_local_quadratic(diagonal_component=self.leading)
        self.potential.calculate_local_remainder(diagonal_component=self.leading)


    def __str__(self):
        """Prepare a printable string representing the I{HagedornPropagator} instance."""
        return "Hagedorn propagator for " + str(self.number_components) + " components.\n Leading component is " + str(self.leading) + "."


    def get_number_components(self):
        """@return: The number $N$ of components $\Phi_i$ of $\Ket{\Psi}$."""
        return self.number_components


    def get_potential(self):
        """@return: The I{MatrixPotential} instance used for time propagation."""
        return self.potential


    def get_number_packets(self):
        """
        """
        return len(self.packets)


    def get_wavepacket(self, packet=None):
        """@return: The I{HagedornWavepacket} instance that represents the
        current wavepacket $\Ket{\Psi}$."""
        if packet is None:
            return [ p[0] for p in self.packets ]
        else:
            return self.packets[packet][0]


    def propagate(self, time):
        """Given the wavepacket $\Psi$ at time $t$, calculate a new wavepacket
        at time $t + \tau$. We perform exactly one timestep $\tau$ here.
        """
        dt = self.dt

        # Ckeck for spawning
        if self.should_spwan(time):
            # TODO: magics
            self.potential.calculate_local_quadratic(diagonal_component=1)
            self.potential.calculate_local_remainder(diagonal_component=1)

            # Initialize an empty wavepacket for spawning
            SWP = HagedornWavepacket(self.parameters)
            SWP.set_quadrature(None)

            # Initialize a Spawner
            NAS = NonAdiabaticSpawner(self.parameters)

            # Estimate parameter set Pi
            WP = self.packets[0][0]
            WP.project_to_eigen(self.potential)
            # TODO: magics
            Pi = NAS.estimate_parameters(WP, component=1)

            # Spawn a new packet
            if Pi is not None:
                SWP.set_parameters(Pi)
                # TODO: magics
                NAS.project_coefficients(WP, SWP, component=1)

                SWP.project_to_canonical(self.potential)
                WP.project_to_canonical(self.potential)

                # TODO: magics
                self.packets.append((SWP,1))


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
            F = self.quadrature.build_matrix(packet, partial(self.potential.evaluate_local_remainder_at, diagonal_component=leading_chi))

            coefficients = packet.get_coefficient_vector()
            coefficients = self.matrix_exponential(F, coefficients, dt/self.eps**2)
            packet.set_coefficient_vector(coefficients)

            # Do a kinetic step of dt/2
            packet.q = packet.q + 0.5 * dt * packet.p
            packet.Q = packet.Q + 0.5 * dt * packet.P
            packet.S = packet.S + 0.25 * dt * packet.p**2


    def should_spwan(self, time):
        """Check if it's time to spawn a new wavepacket.
        """
        if self.already_spawned:
            return False

        # TODO: magics
        P = self.packets[0][0].clone()
        P.project_to_eigen(self.potential)
        n = P.get_norm(component=1)

        print(n)

        #answer = (n >= self.threshold)
        answer = (time >= 5.25)

        if answer == True:
            self.already_spawned = True

        return answer
