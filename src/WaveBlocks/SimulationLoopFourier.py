"""The WaveBlocks Project

This file contains the main simulation loop
for the Fourier propagator.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np
import scipy as sp

from PotentialFactory import PotentialFactory as PF
from WaveFunction import WaveFunction
from HagedornWavepacket import HagedornWavepacket
from FourierPropagator import FourierPropagator
from SimulationLoop import SimulationLoop
from IOManager import IOManager


class SimulationLoopFourier(SimulationLoop):
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
        Set up a Fourier propagator for the simulation loop. Set the
        potential and initial values according to the configuration.

        :raise ValueError: For invalid or missing input data.
        """
        # Compute the position space grid points
        nodes = self.parameters["f"] * sp.pi * sp.arange(-1, 1, 2.0/self.parameters["ngn"], dtype=np.complexfloating)

        # The potential instance
        potential = PF().create_potential(self.parameters)

        # Check for enough initial values
        if not self.parameters.has_key("initial_values"):
            if len(self.parameters["parameters"]) < potential.get_number_components():
                raise ValueError("Too few initial states given. Parameters are missing.")

            if len(self.parameters["coefficients"]) < potential.get_number_components():
                raise ValueError("Too few initial states given. Coefficients are missing.")

        # Calculate the initial values sampled from a hagedorn wave packet
        d = dict([("ncomponents", 1), ("basis_size", self.parameters["basis_size"]), ("eps", self.parameters["eps"])])

        # Initial values given in the "fourier" specific format
        if self.parameters.has_key("initial_values"):
            initialvalues = [ np.zeros(nodes.shape, dtype=np.complexfloating) for i in xrange(self.parameters["ncomponents"]) ]

            for level, params, coeffs in self.parameters["initial_values"]:
                hwp = HagedornWavepacket(d)
                hwp.set_parameters(params)

                for index, value in coeffs:
                    hwp.set_coefficient(0, index, value)

                iv = hwp.evaluate_at(nodes, component=0, prefactor=True)

                initialvalues[level] = initialvalues[level] + iv

        # Initial value read in compatibility mode to the packet algorithms
        else:
            # See if we have a list of parameter tuples or just a single 5-tuple
            # This is for compatibility with the inhomogeneous case.
            try:
                # We have a list of parameter tuples this is ok for the loop below
                len(self.parameters["parameters"][0])
                parameters = self.parameters["parameters"]
            except TypeError:
                # We have just a single 5-tuple of parameters, we need to replicate for looping
                parameters = [ self.parameters["parameters"] for i in xrange(self.parameters["ncomponents"]) ]

            initialvalues = []

            for level, item in enumerate(parameters):
                hwp = HagedornWavepacket(d)
                hwp.set_parameters(item)

                # Set the coefficients of the basis functions
                for index, value in self.parameters["coefficients"][level]:
                    hwp.set_coefficient(0, index, value)

                iv = hwp.evaluate_at(nodes, component=0, prefactor=True)

                initialvalues.append(iv)

        # Project the initial values to the canonical basis
        initialvalues = potential.project_to_canonical(nodes, initialvalues)

        # Store the initial values in a WaveFunction object
        IV = WaveFunction(self.parameters)
        IV.set_grid(nodes)
        IV.set_values(initialvalues)

        # Finally create and initialize the propagator instace
        self.propagator = FourierPropagator(potential, IV, self.parameters)

        # Which data do we want to save
        tm = self.parameters.get_timemanager()
        slots = tm.compute_number_saves()

        print(tm)

        self.IOManager.add_grid(self.parameters, blockid="global")
        self.IOManager.add_fourieroperators(self.parameters)
        self.IOManager.add_wavefunction(self.parameters, timeslots=slots)

        # Write some initial values to disk
        self.IOManager.save_grid(nodes, blockid="global")
        self.IOManager.save_fourieroperators(self.propagator.get_operators())
        self.IOManager.save_wavefunction(IV.get_values(), timestep=0)


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
                self.IOManager.save_wavefunction(self.propagator.get_wavefunction().get_values(), timestep=i)


    def end_simulation(self):
        r"""
        Do the necessary cleanup after a simulation. For example request the
        IOManager to write the data and close the output files.
        """
        self.IOManager.finalize()
