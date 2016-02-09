"""The WaveBlocks Project

This file contains the main simulation loop
for the inhomogeneous Hagedorn propagator.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np
import scipy as sp

from PotentialFactory import PotentialFactory as PF
from HagedornWavepacketInhomogeneous import HagedornWavepacketInhomogeneous
from HagedornPropagatorInhomogeneous import HagedornPropagatorInhomogeneous
from SimulationLoop import SimulationLoop
from IOManager import IOManager


class SimulationLoopHagedornInhomogeneous(SimulationLoop):
    r"""
    This class acts as the main simulation loop. It owns a propagator that
    propagates a set of initial values during a time evolution. All values are
    read from the ``Parameters.py`` file.
    """

    def __init__(self, parameters):
        r"""
        Create a new simulation loop instance.
        """
        # Keep a reference to the simulation parameters
        self.parameters = parameters

        #: The time propagator instance driving the simulation.
        self.propagator = None

        #: A ``IOManager`` instance for saving simulation results.
        self.IOManager = None

        #: The number of time steps we will perform.
        self.nsteps = parameters["nsteps"]

        # Set up serializing of simulation data
        self.IOManager = IOManager()
        self.IOManager.create_file(self.parameters)
        self.IOManager.create_block()


    def prepare_simulation(self):
        r"""
        Set up a multi Hagedorn propagator for the simulation loop. Set the
        potential and initial values according to the configuration.

        :raise ValueError: For invalid or missing input data.
        """
        potential = PF().create_potential(self.parameters)
        N = potential.get_number_components()

        # Check for enough initial values
        if len(self.parameters["parameters"]) < N:
            raise ValueError("Too few initial states given. Parameters are missing.")

        if len(self.parameters["coefficients"]) < N:
            raise ValueError("Too few initial states given. Coefficients are missing.")

        # Create a suitable wave packet
        packet = HagedornWavepacketInhomogeneous(self.parameters)
        packet.set_quadrature(None)

        # Set the parameters for each energy level
        for level, item in enumerate(self.parameters["parameters"]):
            packet.set_parameters(item, level)

        # Set the initial values
        for component, data in enumerate(self.parameters["coefficients"]):
            for index, value in data:
                packet.set_coefficient(component, index, value)

        # Project the initial values to the canonical basis
        packet.project_to_canonical(potential)

        # Finally create and initialize the propagator instace
        self.propagator = HagedornPropagatorInhomogeneous(potential, packet, self.parameters)

        # Which data do we want to save
        tm = self.parameters.get_timemanager()
        slots = tm.compute_number_saves()

        self.IOManager.add_grid(self.parameters, blockid="global")
        self.IOManager.add_inhomogwavepacket(self.parameters, timeslots=slots)

        # Write some initial values to disk
        nodes = self.parameters["f"] * sp.pi * sp.arange(-1, 1, 2.0/self.parameters["ngn"], dtype=np.complexfloating)
        # self.nodes = nodes
        self.IOManager.save_grid(nodes, blockid="global")
        self.IOManager.save_inhomogwavepacket_parameters(self.propagator.get_wavepackets().get_parameters(), timestep=0)
        self.IOManager.save_inhomogwavepacket_coefficients(self.propagator.get_wavepackets().get_coefficients(), timestep=0)


    def run_simulation(self):
        r"""
        Run the simulation loop for a number of time steps. The number of steps is calculated in the ``initialize`` function.
        """
        tm = self.parameters.get_timemanager()

        # Run the simulation for a given number of timesteps
        for i in xrange(1, self.nsteps+1):
            print(" doing timestep "+str(i))

            self.propagator.propagate()

            # Save some simulation data
            if tm.must_save(i):
                self.IOManager.save_inhomogwavepacket_parameters(self.propagator.get_wavepackets().get_parameters(), timestep=i)
                self.IOManager.save_inhomogwavepacket_coefficients(self.propagator.get_wavepackets().get_coefficients(), timestep=i)


    def end_simulation(self):
        r"""
        Do the necessary cleanup after a simulation. For example request the
        IOManager to write the data and close the output files.
        """
        self.IOManager.finalize()
