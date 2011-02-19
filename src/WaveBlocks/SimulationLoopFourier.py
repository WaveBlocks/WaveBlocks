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
    """This class acts as the main simulation loop. It owns a propagator that
    propagates a set of initial values during a time evolution. All values are
    read from the I{Parameters.py} file."""
    
    def __init__(self, parameters):
        """Create a new simulation loop instance."""
        # Keep a reference to the simulation parameters
        self.parameters = parameters
        
        #: The time propagator instance driving the simulation.
        self.propagator = None
        
        #: A I{IOManager} instance for saving simulation results.
        self.IOManager = None

        #: The number of time steps we will perform.
        self.nsteps = parameters.nsteps

        # Set up serializing of simulation data
        self.IOManager = IOManager()
        self.IOManager.create_file(self.parameters)
        
 
    def prepare_simulation(self):
        """Set up a Fourier propagator for the simulation loop. Set the
        potential and initial values according to the configuration.
        @raise ValueError: For invalid or missing input data.
        """        
        # Compute the position space grid points
        nodes = self.parameters.f * sp.pi * sp.arange(-1, 1, 2.0/self.parameters.ngn, dtype=np.complexfloating) 

        # The potential instance
        potential = PF.create_potential(self.parameters)

        # Check for enough initial values
        if len(self.parameters.parameters) < potential.get_number_components():
            raise ValueError("Too few initial states given. Parameters are missing.")
            
        if len(self.parameters.coefficients) < potential.get_number_components():
            raise ValueError("Too few initial states given. Coefficients are missing.")
        
        # Calculate the initial values sampled from a hagedorn wave packet
        initialvalues = []
        d = dict([("ncomponents", 1), ("basis_size", self.parameters["basis_size"]), ("eps", self.parameters["eps"])])
        for index, item in enumerate(self.parameters.parameters):
            hwp = HagedornWavepacket(d)
            hwp.set_parameters(item)
            
            # Set the coefficients of the basis functions
            for i, value in self.parameters.coefficients[index]:
                hwp.set_coefficient(0,i,value)

            iv = hwp.evaluate_at(nodes, component=0, prefactor=True)

            initialvalues.append(iv)
            
        # Project the initial values to the canonical base
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

        self.IOManager.add_grid(self.parameters)
        self.IOManager.add_fourieroperators(self.parameters)
        self.IOManager.add_wavefunction(self.parameters, timeslots=slots)

        # Write some initial values to disk
        self.IOManager.save_grid(nodes)
        self.IOManager.save_fourieroperators(self.propagator.get_operators())
        self.IOManager.save_wavefunction(IV, timestep=0)
                             

    def run_simulation(self):
        """Run the simulation loop for a number of time steps. The number of steps
        is calculated in the I{initialize} function."""
        tm = self.parameters.get_timemanager()
        
        # Run the simulation for a given number of timesteps
        for i in xrange(1, self.nsteps+1):
            print(" doing timestep "+str(i))

            self.propagator.propagate()
            
            # Save some simulation data
            if tm.must_save(i):
                self.IOManager.save_wavefunction(self.propagator.get_wavefunction(), timestep=i)


    def end_simulation(self):
        """Do the necessary cleanup after a simulation. For example request the
        IOManager to write the data and close the output files.
        """
        self.IOManager.finalize()
