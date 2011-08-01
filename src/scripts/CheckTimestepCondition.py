"""The WaveBlocks Project

This file contains a simple script to check if the
simulation parameters violate the timestep condition.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys

from WaveBlocks import ParameterProvider

# Read the path for the configuration file we use for this simulation.
try:
    parametersfile = sys.argv[1]
except IndexError:
    raise ValueError("No configuration file given")
    
print("Testing configuration from file: " + parametersfile)

# Set up the parameter singleton and read the parameters
PA = ParameterProvider()
PA.read_parameters(parametersfile)

if not PA["dt"] <= PA["eps"] ** 2:
    print(" The parameters violate the timestep constraint!")
else:
    print(" Ok")
