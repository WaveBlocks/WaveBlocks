"""The WaveBlocks Project

This file contains the Hagedorn propagator class for inhomogeneous wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from functools import partial

from WaveFunction import WaveFunction
from Propagator import Propagator


class HagedornPropagatorInhomogeneous(Propagator):
    """This class can numerically propagate given initial values $\Ket{\Psi}$ in
    a potential $V\ofs{x}$. The propagation is done for a given inhomogeneous
    Hagedorn wavepacket."""

    def __init__(self, potential, packet, parameters):
        """Initialize a new I{HagedornPropagatorInhomogeneous} instance.
        @param potential: The potential the wavepacket $\Ket{\Psi}$ feels during the time propagation.
        @param packet: The initial inhomogeneous Hagedorn wavepacket we propagate in time.
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
        self.packet = packet

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
        self.potential.calculate_local_quadratic_multi()
        self.potential.calculate_local_remainder_multi()


    def __str__(self):
        """Prepare a printable string representing the I{HagedornPropagatorInhomogeneous} instance."""
        return "Hagedorn propagator for " + str(self.number_components) + " components."


    def get_number_components(self):
        """@return: The number $N$ of components $\Phi_i$ of $\Ket{\Psi}$."""
        return self.number_components


    def get_potential(self):
        """@return: The I{MatrixPotential} instance used for time propagation."""
        return self.potential


    def get_wavepacket(self):
        """@return: The I{HagedornWavepacketInhomogeneous} instance that represents the
        current wavepacket $\Ket{\Psi}$."""
        return self.packet


    def get_wavefunction(self, nodes):
        """Construct a I{WaveFunction} object which contains the components $\Phi_i$
        of the Hagedorn wavepacket evaluated at the given nodes $\gamma$.
        @param nodes: The nodes $\gamma$ on which the Hagedorn wavepacket is evaluated.
        @return: A I{WaveFunction} instance representing the values of the current $\Ket{\Psi}$.
        @note: This method is quite expensive.
        """
        values = self.packet.evaluate_at(nodes, prefactor=True)
        WF = WaveFunction(self.parameters)
        WF.set_grid(nodes)
        WF.set_values(values)
        return WF


    def propagate(self):
        """Given the wavepacket $\Psi$ at time $t$, calculate a new wavepacket
        at time $t + \tau$. We perform exactly one timestep $\tau$ here.
        """
        dt = self.dt

        # Do a kinetic step of dt/2
        for component in xrange(self.number_components):
            (P,Q,S,p,q) = self.packet.get_parameters(component=component)

            q = q + 0.5*dt * p
            Q = Q + 0.5*dt * P
            S = S + 0.25*dt * p**2

            self.packet.set_parameters((P,Q,S,p,q), component=component)

        # Do a potential step with the local quadratic part
        for component in xrange(self.number_components):
            (P,Q,S,p,q) = self.packet.get_parameters(component=component)

            V = self.potential.evaluate_local_quadratic_multi_at(q, component=component)

            p = p - dt * V[1]
            P = P - dt * V[2] * Q
            S = S - dt * V[0]

            self.packet.set_parameters((P,Q,S,p,q), component=component)

        # Do a potential step with the local non-quadratic taylor remainder
        F = self.quadrature.build_matrix(self.packet, self.packet, self.potential.evaluate_local_remainder_at)

        coefficients = self.packet.get_coefficient_vector()
        coefficients = self.matrix_exponential(F, coefficients, dt/self.eps**2)
        self.packet.set_coefficient_vector(coefficients)

        # Do a kinetic step of dt/2
        for component in xrange(self.number_components):
            (P,Q,S,p,q) = self.packet.get_parameters(component=component)

            q = q + 0.5 * dt * p
            Q = Q + 0.5 * dt * P
            S = S + 0.25 * dt * p**2

            self.packet.set_parameters((P,Q,S,p,q), component=component)
