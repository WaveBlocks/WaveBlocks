"""The WaveBlocks Project

This file contains the Hagedorn propagator class for inhomogeneous wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from Propagator import Propagator
from MatrixExponentialFactory import MatrixExponentialFactory


class HagedornPropagatorInhomogeneous(Propagator):
    r"""
    This class can numerically propagate given initial values :math:`\Psi` in
    a potential :math:`V(x)`. The propagation is done for a given set of inhomogeneous
    Hagedorn wavepackets neglecting interaction.
    """

    def __init__(self, potential, packet, parameters):
        r"""
        Initialize a new :py:class:`HagedornPropagatorInhomogeneous` instance.

        :param potential: The potential :math:`V(x)` the wavepacket :math:`\Psi` feels during the time propagation.
        :param packet: The initial inhomogeneous Hagedorn wavepacket :math:`\Psi` we propagate in time.
        :param parameters: A :py:class:`ParameterProvider` instance.

        :raises ValueError: If the number of components of :math:`\Psi` does not match
                            the number of energy levels :math:`\lambda_i` of the potential.
        """
        if packet.get_number_components() != potential.get_number_components():
            raise ValueError("Wavepacket does not match to the given potential!")

        #: The potential :math:`V(x)` the packet(s) feel.
        self.potential = potential

        #: Number :math:`N` of components the wavepacket :math:`\Psi` has got.
        self.number_components = self.potential.get_number_components()

        #: A list of Hagedorn wavepackets :math:`\Psi`.
        #: At the moment we do not use any codata here.
        self.packets = [ packet ]

        # Keep a reference to the parameter provider instance
        self.parameters = parameters

        # Decide about the matrix exponential algorithm to use
        self.__dict__["matrix_exponential"] = MatrixExponentialFactory().get_matrixexponential(parameters)

        # Precalculate the potential splitting
        self.potential.calculate_local_quadratic()
        self.potential.calculate_local_remainder()


    def __str__(self):
        r"""
        Prepare a printable string representing the :py:class:`HagedornPropagatorInhomogeneous` instance.
        """
        return "Hagedorn propagator for " + str(self.number_components) + " components."


    def get_number_components(self):
        r"""
        :return: The number :math:`N` of components :math:`\Phi_i` of :math:`\Psi`.
        """
        return self.number_components


    def get_potential(self):
        r"""
        Returns the potential used for time propagation.

        :return: A :py:class:`MatrixPotential` instance.
        """
        return self.potential


    def get_wavepackets(self, packet=0):
        r"""
        Return the wavepackets taking part in the simulation.

        :param packet: The number of a single packet that is to be returned.
        :type packet: Integer
        :return: A list of :py:class:`HagedornWavepacketInhomogeneous`
                 instances that represents the current wavepackets.
        """
        if packet is None:
            return [ p for p in self.packets ]
        else:
            return self.packets[packet]


    def set_wavepackets(self, packetlist):
        r"""
        Set the wavepackets that the propagator will propagate.

        :param packetlist: A list of new wavepackets to propagate.
        """
        self.packets = packetlist


    def propagate(self):
        r"""
        Given the wavepacket :math:`\Psi` at time :math:`t` compute the propagated
        wavepacket at time :math:`t + \tau`. We perform exactly one timestep :math:`\tau` here.
        """
        # Cache some parameter values for efficiency
        dt = self.parameters["dt"]
        eps = self.parameters["eps"]

        # Propagate all packets
        for packet in self.packets:

            # Do a kinetic step of dt/2
            for component in xrange(self.number_components):
                (P,Q,S,p,q) = packet.get_parameters(component=component)

                q = q + 0.5*dt * p
                Q = Q + 0.5*dt * P
                S = S + 0.25*dt * p**2

                packet.set_parameters((P,Q,S,p,q), component=component)

            # Do a potential step with the local quadratic part
            for component in xrange(self.number_components):
                (P,Q,S,p,q) = packet.get_parameters(component=component)

                V = self.potential.evaluate_local_quadratic_at(q, diagonal_component=component)

                p = p - dt * V[1]
                P = P - dt * V[2] * Q
                S = S - dt * V[0]

                packet.set_parameters((P,Q,S,p,q), component=component)

            # Do a potential step with the local non-quadratic taylor remainder
            quadrature = packet.get_quadrature()
            F = quadrature.build_matrix(packet, packet, self.potential.evaluate_local_remainder_at)

            coefficients = packet.get_coefficient_vector()
            coefficients = self.matrix_exponential(F, coefficients, dt/eps**2)
            packet.set_coefficient_vector(coefficients)

            # Do a kinetic step of dt/2
            for component in xrange(self.number_components):
                (P,Q,S,p,q) = packet.get_parameters(component=component)

                q = q + 0.5 * dt * p
                Q = Q + 0.5 * dt * P
                S = S + 0.25 * dt * p**2

                packet.set_parameters((P,Q,S,p,q), component=component)
