"""The WaveBlocks Project

This file contains the Hagedorn propagator class for homogeneous wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from functools import partial

from WaveFunction import WaveFunction
from Propagator import Propagator


class HagedornPropagator(Propagator):
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
        self.packet = packet

        if self.packet.get_number_components() != self.number_components:
            raise ValueError("Wave packet does not match to the potential.")

        # Cache some parameter values for efficiency
        self.parameters = parameters
        self.dt = parameters["dt"]
        self.eps = parameters["eps"]

        # The quadrature instance matching the packet
        self.quadrature = packet.get_quadrature()

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


    def get_wavepacket(self):
        """@return: The I{HagedornWavepacket} instance that represents the
        current wavepacket $\Ket{\Psi}$."""
        return self.packet


    def propagate(self):
        """Given the wavepacket $\Psi$ at time $t$, calculate a new wavepacket
        at time $t + \tau$. We perform exactly one timestep $\tau$ here.
        """
        dt = self.dt

        # Do a kinetic step of dt/2
        self.packet.q = self.packet.q + 0.5*dt * self.packet.p
        self.packet.Q = self.packet.Q + 0.5*dt * self.packet.P
        self.packet.S = self.packet.S + 0.25*dt * self.packet.p**2

        # Do a potential step with the local quadratic part
        V = self.potential.evaluate_local_quadratic_at(self.packet.q, diagonal_component=self.leading)

        self.packet.p = self.packet.p - dt * V[1]
        self.packet.P = self.packet.P - dt * V[2] * self.packet.Q
        self.packet.S = self.packet.S - dt * V[0]

        # Do a potential step with the local non-quadratic taylor remainder
        F = self.quadrature.build_matrix(self.packet, partial(self.potential.evaluate_local_remainder_at, diagonal_component=self.leading))

        coefficients = self.packet.get_coefficient_vector()
        coefficients = self.matrix_exponential(F, coefficients, dt/self.eps**2)
        self.packet.set_coefficient_vector(coefficients)

        # Do a kinetic step of dt/2
        self.packet.q = self.packet.q + 0.5 * dt * self.packet.p
        self.packet.Q = self.packet.Q + 0.5 * dt * self.packet.P
        self.packet.S = self.packet.S + 0.25 * dt * self.packet.p**2
