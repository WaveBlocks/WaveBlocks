"""The WaveBlocks Project

This file contains a very simple spawning propagator class
for wavepackets and spawning in the adiabatic case.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from Propagator import Propagator
from HagedornWavepacket import HagedornWavepacket
from MatrixExponentialFactory import MatrixExponentialFactory
from AdiabaticSpawner import AdiabaticSpawner
from SpawnConditionFactory import SpawnConditionFactory as SCF


class SpawnAdiabaticPropagator(Propagator):
    r"""
    This class can numerically propagate given initial values :math:`\Psi` in
    a potential :math:`V(x)`. The propagation is done for several given homogeneous
    Hagedorn wavepackets neglecting interaction.
    """

    def __init__(self, propagator, potential, packet, leading_component, parameters):
        r"""
        Initialize a new :py:class:SpawnAdiabaticPropagator instance.

        :param propagator: The propagator used for time propagation.
        :param potential: The potential the wavepacket :math:`\Psi` feels during the time propagation.
        :param packet: The initial homogeneous Hagedorn wavepacket we propagate in time.
        :param leading_component: The leading component index :math:`\chi`.

        :raises ValueError: If the number of components of :math:`\Psi` does not
                            match the number of energy levels :math:`\lambda_i`
                            of the potential.
        """
        if packet.get_number_components() != potential.get_number_components():
            raise ValueError("Wave packet does not match to the given potential!")

        #: The potential :math:`V\left(x\right)` the packet feels.
        self.potential = potential

        #: Number :math:`N` of components the wavepacket :math:`|\Psi\rangle` has got.
        self.number_components = self.potential.get_number_components()

        #: The Hagedorn wavepackets.
        self.packets = [ (packet,leading_component) ]

        # Cache some parameter values for efficiency
        self.parameters = parameters

        # The propagator used for time propagation
        self.propagator = propagator

        #: The condition which determines when to spawn.
        oracle = SCF().get_condition(parameters)
        # Setup the environment for the spawning condition.
        self.spawn_condition = oracle(self.parameters, self)

        # Decide about the matrix exponential algorithm to use
        self.__dict__["matrix_exponential"] = MatrixExponentialFactory().get_matrixexponential(parameters)

        # Precalculate the potential splitting
        self.potential.calculate_local_quadratic(diagonal_component=leading_component)
        self.potential.calculate_local_remainder(diagonal_component=leading_component)


    def __str__(self):
        r"""
        Prepare a printable string representing the :py:class:`SpawnAdiabaticPropagator` instance.
        """
        return "Prapagation and spawning in the adabatic case."


    def get_number_components(self):
        r"""
        :return: The number :math:`N` of components :math:`\Phi_i` of :math:`\Psi`.
        """
        return self.number_components


    def get_potential(self):
        r"""
        :return: The :py:class:`MatrixPotential` instance used for time propagation.
        """
        return self.potential


    def get_number_packets(self):
        r"""
        Get the number of packets :math:`\Psi_n` taking part in the simulation.

        :return: The number of packets currently taking part in the simulation.
        """
        return len(self.packets)


    def get_wavepackets(self, packet=None):
        r"""
        Retrieve the wavepackets taking part in the simulation.

        :param packet: The number of a single packet that is to be returned.
        :type packet: Integer
        :return: A list of :py:class:`HagedornWavepacket` instances that represents
                 the current wavepackets.
        """
        if packet is None:
            return [ p[0] for p in self.packets ]
        else:
            return self.packets[packet][0]


    def propagate(self, time):
        r"""
        Given the wavepacket :math:`\Psi` at time :math:`t` compute the propagated
        wavepacket at time :math:`t + \tau`. We perform exactly one timestep :math:`\tau`
        here. At every timestep we check the spawning condition.
        """
        # Make time accessible for spawn condition testers
        self.time = time

        # Perform spawning in necessary
        todo = self.should_spwan()
        for info in todo:
            self.spawn(info)

        # Propagate all packets
        self.propagator.set_wavepackets(self.packets)
        self.propagator.propagate()


    def should_spwan(self):
        r"""
        Check if there is a reason to spawn a new wavepacket.
        """
        spawn_todo = []

        # What do we have to do now?
        # For each packet in the simulation
        #   Check if we should spawn

        for packet, leading_chi in self.packets:
            # Spawn condition fulfilled?
            should_spawn = self.spawn_condition.check_condition(packet, 0, self)

            if should_spawn:
                print("  Spawn condition fulfilled for component 0 of packet with ID "+str(packet.get_id())+".")
                spawn_todo.append(packet)

        return spawn_todo


    def spawn(self, info):
        r"""
        Really spawn the new wavepackets :math:`\tilde{\Psi}`. This method
        appends the new :py:class:`HagedornWavepacket` instances to the list
        :py:attr:`packets` of packets.
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

            print("  Spawned a new wavepacket with ID "+str(SWP.get_id())+".")

            # Append the spawned packet to the world
            self.packets.append((SWP,0))
