"""The WaveBlocks Project

Reads configuration files containing the simulation parameters and
provides these values to the simulation as global singleton.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import types
from copy import deepcopy

import GlobalDefaults
from PotentialFactory import PotentialFactory as PF
from TimeManager import TimeManager


class ParameterProvider:

    def __init__(self):
        #: Dict for storing the configuration parameters.
        self.params = {}


    def __getattr__(self, key):
        print(" Depreceated __getattr__ for key "+str(key)+" at ParameterProvider instance!")
        return self.params[key]


    def __getitem__(self, key):
        # See if we have a parameter with specified name
        if self.params.has_key(key):
            return self.params[key]
        else:
            # If not, try to find a global default value for it and copy over this value
            print("Warning: parameter '"+str(key)+"' not found, now trying global defaults!")
            if GlobalDefaults.__dict__.has_key(key):
                self.__setitem__(key, deepcopy(GlobalDefaults.__dict__[key]))
                return self.params[key]
            else:
                raise KeyError("Could not find a default value for parameter "+str(key)+"!")


    def __setitem__(self, key, value):
        self.params[key] = deepcopy(value)


    def __contains__(self, key):
        return self.params.has_key(key)


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


    def compute_parameters(self):
        """Compute some further parameters from the given ones.
        """
        # Perform the computation only if the basic values are available.
        # This is necessary to add flexibility and essentially read in *any*
        # parameter file with heavily incomplete value sets. (F.e. spawn configs)
        if self.params.has_key("T") and self.params.has_key("dt"):
            self._tm = TimeManager()
            self._tm.set_T(self["T"])
            self._tm.set_dt(self["dt"])

            # Set the interval for saving data
            self._tm.set_interval(self["write_nth"])

            # Set the fixed times for saving data
            if self.has_key("save_at"):
                self._tm.add_to_savelist(self["save_at"])

            # The number of time steps we will perform.
            self.params["nsteps"] = self._tm.compute_number_timesteps()

        if self.params.has_key("potential"):
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
            self.params[key] = deepcopy(value)

        # Compute some values on top of the given input parameters
        self.compute_parameters()


    def set_parameters(self, params):
        """Overwrite the dict containing all parameters with a
        newly provided dict with (possibly) changed parameters.
        @param params: A I{ParameterProvider} instance or a dict
        with new parameters. The values will be deep-copied. No
        old values will remain.
        """
        if not isinstance(params, dict):
            try:
                params = params.get_parameters()
            except:
                raise TypeError("Wrong data type for set_parameters.")

        assert type(params) == dict

        self.params = deepcopy(params)
        # Compute some values on top of the given input parameters
        self.compute_parameters()


    def update_parameters(self, params):
        """Overwrite the dict containing all parameters with a
        newly provided dict with (possibly) changed parameters.
        @param params: A I{ParameterProvider} instance or a dict
        with new parameters. The values will be deep-copied. Old
        values are only overwritten if we have got new values.
        """
        if isinstance(params, ParameterProvider):
            params = params.get_parameters()

        for key, value in params.iteritems():
            self.__setitem__(key, value)
        # Compute some values on top of the given input parameters
        self.compute_parameters()


    def get_timemanager(self):
        """Return the embedded I{TimeManager} instance.
        """
        return self._tm


    def get_parameters(self):
        """Return a copy of the dict containing all parameters.
        @return: A copy of the dict containing all parameters. The dict will be copied.
        """
        return deepcopy(self.params)


    def __str__(self):
        try:
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

        except KeyError:
            pass

        s += "------------------------------------\n"
        s += "All parameters provided\n"
        s += "------------------------------------\n"

        keys = self.params.keys()
        keys.sort()

        for key in keys:
            s += "  " + str(key) + ": " + str(self.params[key]) + "\n"

        s += "====================================\n"

        return s
