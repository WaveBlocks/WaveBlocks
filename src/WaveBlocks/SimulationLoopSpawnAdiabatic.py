"""The WaveBlocks Project

This file contains the main loop for simple adiabatic spawning simulations.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np
import scipy as sp

from TimeManager import TimeManager
from SimulationLoop import SimulationLoop
from PotentialFactory import PotentialFactory
from HagedornWavepacket import HagedornWavepacket
from HagedornPropagator import HagedornPropagator
from SpawnAdiabaticPropagator import SpawnAdiabaticPropagator
from IOManager import IOManager


class SimulationLoopSpawnAdiabatic(SimulationLoop):
    """This class acts as the main simulation loop. It owns a propagator that
    propagates a set of initial values during a time evolution. All values are
    read from the I{Parameters.py} file."""

    def __init__(self, parameters):
        """Create a new simulation loop instance."""
        # Keep a reference to the simulation parameters
        self.parameters = parameters

        self.tm = TimeManager(parameters)

        #: The time propagator instance driving the simulation.
        self.propagator = None

        #: A I{IOManager} instance for saving simulation results.
        self.iom = IOManager()
        self.iom.create_file(parameters)
        self.gid = self.iom.create_group()


    def prepare_simulation(self):
        """Set up a Spawning propagator for the simulation loop. Set the
        potential and initial values according to the configuration.
        @raise ValueError: For invalid or missing input data.
        """
        potential = PotentialFactory().create_potential(self.parameters)
        N = potential.get_number_components()

        # Check for enough initial values
        if self.parameters["leading_component"] > N:
            raise ValueError("Leading component index out of range.")

        if len(self.parameters["parameters"]) < N:
            raise ValueError("Too few initial states given. Parameters are missing.")

        if len(self.parameters["coefficients"]) < N:
            raise ValueError("Too few initial states given. Coefficients are missing.")

        # Create a suitable wave packet
        packet = HagedornWavepacket(self.parameters)
        packet.set_parameters(self.parameters["parameters"][self.parameters["leading_component"]])
        packet.set_quadrature(None)

        # Set the initial values
        for component, data in enumerate(self.parameters["coefficients"]):
            for index, value in data:
                packet.set_coefficient(component, index, value)

        # Project the initial values to the canonical basis
        packet.project_to_canonical(potential)

        # Finally create and initialize the propagator instace
        inner = HagedornPropagator(potential, packet, self.parameters["leading_component"], self.parameters)
        self.propagator = SpawnAdiabaticPropagator(inner, potential, packet, self.parameters["leading_component"], self.parameters)

        # Write some initial values to disk
        slots = self.tm.compute_number_saves()
        for packet in self.propagator.get_wavepackets():
            bid = self.iom.create_block(groupid=self.gid)
            self.iom.add_wavepacket(self.parameters, timeslots=slots, blockid=bid)
            self.iom.save_wavepacket_coefficients(packet.get_coefficients(), blockid=bid, timestep=0)
            self.iom.save_wavepacket_parameters(packet.get_parameters(), blockid=bid, timestep=0)


    def run_simulation(self):
        """Run the simulation loop for a number of time steps. The number of steps
        is calculated in the I{initialize} function."""
        tm = self.tm

        # Run the simulation for a given number of timesteps
        for i in xrange(1, tm.get_nsteps()+1):
            print(" doing timestep "+str(i))

            self.propagator.propagate(tm.compute_time(i))

            # Save some simulation data
            if tm.must_save(i):
                # Check if we need more data blocks for newly spawned packets
                if self.iom.get_number_blocks(groupid=self.gid) < self.propagator.get_number_packets():
                    bid = self.iom.create_block(groupid=self.gid)
                    self.iom.add_wavepacket(self.parameters, blockid=bid)

                for index, packet in enumerate(self.propagator.get_wavepackets()):
                    self.iom.save_wavepacket_coefficients(packet.get_coefficients(), timestep=i, blockid=index)
                    self.iom.save_wavepacket_parameters(packet.get_parameters(), timestep=i, blockid=index)


    def end_simulation(self):
        """Do the necessary cleanup after a simulation. For example request the
        IOManager to write the data and close the output files.
        """
        self.iom.finalize()
