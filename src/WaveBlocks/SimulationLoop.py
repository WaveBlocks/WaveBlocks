"""The WaveBlocks Project

This file contains the main simulation loop interface.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

class SimulationLoop:
    """This class acts as the main simulation loop. It owns a propagator that
    propagates a set of initial values during a time evolution.
    """
    
    def __init__(self, parameters):
        """Create a new simulation loop instance.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'SimulationLoop' is an abstract base class.")


    def prepare_simulation(self):
        """Set up a Fourier propagator for the simulation loop. Set the
        potential and initial values according to the configuration.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'SimulationLoop' is an abstract base class.")


    def run_simulation(self):
        """
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'SimulationLoop' is an abstract base class.")


    def end_simulation(self):
        """Do the necessary cleanup after a simulation. For example request the
        IOManager to write the data and close the output files.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'SimulationLoop' is an abstract base class.")
