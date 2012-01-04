"""The WaveBlocks Project

This file contains the Hagedorn propagator class for homogeneous wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from functools import partial

from WaveFunction import WaveFunction
from Propagator import Propagator
from MatrixExponentialFactory import MatrixExponentialFactory


class HagedornPropagator(Propagator):
    """This class can numerically propagate given initial values :math:`\Psi` in
    a potential :math:`V(x)`. The propagation is done for a given set of homogeneous
    Hagedorn wavepackets neglecting interaction."""

    def __init__(self, potential, packet, leading_component, parameters):
        """Initialize a new :py:class:`HagedornPropagator` instance.

        :param potential: The potential :math:`V(x)` the wavepacket :math:`\Psi` feels during the time propagation.
        :param packet: The initial homogeneous Hagedorn wavepacket :math:`\Psi` we propagate in time.
        :param leading_component: The leading component index :math:`\chi`.
        :param parameters: A :py:class:`ParameterProvider` instance.

        :raises ValueError: If the number of components of :math:`\Psi` does not match
                            the number of energy levels :math:`\lambda_i` of the potential.
        """
        if packet.get_number_components() != potential.get_number_components():
            raise ValueError("Wave packet does not match to the given potential!")

        #: The potential :math:`V(x)` the packet(s) feel.
        self.potential = potential

        #: Number :math:`N` of components the wavepacket :math:`\Psi` has got.
        self.number_components = self.potential.get_number_components()

        #: A list of Hagedorn wavepackets :math:`\Psi` together with some codata
        #: like the leading component :math:`\chi` which is the index of the eigenvalue
        #: :math:`\lambda_\chi` of the potential :math:`V` that is responsible for
        #: propagating the Hagedorn parameters.
        self.packets = [(packet, leading_component)]

        if self.packets[0][0].get_number_components() != self.number_components:
            raise ValueError("Wave packet does not match to the potential.")

        # Cache some parameter values for efficiency
        self.parameters = parameters
        self.dt = parameters["dt"]
        self.eps = parameters["eps"]

        # Decide about the matrix exponential algorithm to use
        self.__dict__["matrix_exponential"] = MatrixExponentialFactory().get_matrixexponential(parameters)

        # Precalculate the potential splittings needed
        for lc in set([ p[1] for p in self.packets ]):
            self.potential.calculate_local_quadratic(diagonal_component=lc)
            self.potential.calculate_local_remainder(diagonal_component=lc)


    def __str__(self):
        """Prepare a printable string representing the :py:class:`HagedornPropagator` instance."""
        return "Hagedorn propagator for " + str(self.number_components) + " components.\n Leading component is " + str(self.leading) + "."


    def get_number_components(self):
        """:return: The number :math:`N` of components :math:`\Phi_i` of :math:`\Psi`.
        """
        return self.number_components


    def get_potential(self):
        """Returns the potential used for time propagation.

        :return: A :py:class:`MatrixPotential` instance.
        """
        return self.potential


    def get_wavepackets(self, packet=0):
        """Return the wavepackets taking part in the simulation.

        :param packet: The number of a single packet that is to be returned.
        :type packet: Integer
        :return: A list of :py:class:`HagedornWavepacket` instances that represents
                 the current wavepackets.
        """
        if packet is None:
            return [ p[0] for p in self.packets ]
        else:
            return self.packets[packet][0]


    def set_wavepackets(self, packetlist):
        """Set the wavepackets that the propagator will propagate.

        :param packetlist: A list of new wavepackets to propagate.
        """
        self.packets = packetlist


    def propagate(self):
        """Given the wavepacket :math:`\Psi` at time :math:`t` compute the propagated
        wavepacket at time :math:`t + \\tau`. We perform exactly one timestep :math:`\\tau` here.
        """
        dt = self.dt

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
