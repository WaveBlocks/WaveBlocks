"""The WaveBlocks Project

Reads configuration files containing the simulation parameters and
provides these values to the simulation as global singleton.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import types

from PotentialFactory import PotentialFactory as PF
from TimeManager import TimeManager


class ParameterProvider:

    def __init__(self):
        #: Dict for storing the configuration parameters.
        self.params = {}


    def __getattr__(self, key):
        # todo: This is slow, speed up necessary?
        return self.params[key]
    

    def __getitem__(self, key):
        return self.params[key]


    def __setitem__(self, key, value):
        self.params[key] = value


    def __iter__(self):
        # For itertion over the parameter key-value pairs
        for item in self.params.iteritems():
            yield item


    def __repr__(self):
        return "A ParameterProvider instance."


    def has_key(self, key):
        return self.params.has_key(key)
    

    def get_configuration_variables(self, _scriptcode):
        """Clean environment for reading in local parameters.
        @param _scriptcode: String with the configuration code to execute.
        """
        # Execute the configuration file, they are plain python files
        exec(_scriptcode)

        # Filter out private variables (the ones prefixed by "_")
        # instances like "self" and imported modules.
        parameters = locals().items()
        
        parameters = [ item for item in parameters if not type(item[1]) == types.ModuleType ]
        parameters = [ item for item in parameters if not type(item[1]) == types.InstanceType ]
        parameters = [ item for item in parameters if not item[0].startswith("_") ]

        return dict(parameters)


    def incompletion_tolerance(self):
        """Be fault tolerant for some incomplete configurations.
        """
        # todo: Improve or remove
        if not "write_nth" in self.params:
            self.params["write_nth"] = 0

        if not "save_at" in self.params:
            self.params["save_at"] = []

        # Number of space dimensions
        self.params["dimension"] = 1


    def compute_parameters(self):
        """Compute some further parameters from the given ones.
        """
        self._tm = TimeManager()
        self._tm.set_T(self.params["T"])
        self._tm.set_dt(self.params["dt"])

        # todo: Fix and improve, decide what to do if not all data are given.

        # Set the interval for saving data
        self._tm.set_interval(self.params["write_nth"])

        # Set the fixed times for saving data
        self._tm.add_to_savelist(self.params["save_at"])

        # The number of time steps we will perform.
        self.params["nsteps"] = self._tm.compute_number_timesteps()

        # Ugly hack. Should improve handling of potential libraries
        Potential = PF.create_potential(self)
        # Number of components of $\Psi$
        self.params["ncomponents"] = Potential.get_number_components()


    def read_parameters(self, filepath):
        """Read the parameters from a configuration file.
        @param filepath: Path to the configuration file.
        """
        # Read the configuration file
        cf = open(filepath)
        content = cf.read()
        cf.close()

        # All the parameters as dict
        params = self.get_configuration_variables(content)

        # Put the values into the local storage
        for key, value in params.iteritems():
            self.params[key] = value

        # Compensate for some missing values
        self.incompletion_tolerance()

        # Compute some values on top of the given input parameters
        self.compute_parameters()


    def get_timemanager(self):
        """Return the embedded I{TimeManager} instance.
        """
        return self._tm


    def __str__(self):
        s =  "====================================\n"
        s += "Parameters of the current simulation\n"
        s += "------------------------------------\n"
        s += " Propagation algorithm: " + str(self.params["algorithm"]) + "\n"
        s += " Potential: " + str(self.params["potential"]) + "\n"
        s += "  Number components: " + str(self.params["ncomponents"]) + "\n"
        s += "\n"
        s += " Timestepping:\n"
        s += "  Final simulation time: " + str(self.params["T"]) + "\n"
        s += "  Time step size: " + str(self.params["dt"]) + "\n"
        s += "  Number of timesteps: " + str(self.params["nsteps"]) + "\n"
        s += "\n"
        s += " I/O related:\n"
        s += "  Write results every step (0 = never): " + str(self.params["write_nth"]) + "\n"
        s += "  Write results at time/timesteps (additionally): " + str(self.params["save_at"]) + "\n"

        s += "------------------------------------\n"
        s += "All parameters provided\n"
        s += "------------------------------------\n"

        keys = self.params.keys()
        keys.sort()
        
        for key in keys:
            s += "  " + str(key) + ": " + str(self.params[key]) + "\n"

        s += "====================================\n"

        return s
