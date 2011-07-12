"""The WaveBlocks Project

This file contains some global defaults, for example file names for output files.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

# Some global parameters related to file naming
path_to_autogen_configs = "autogen_configurations"
path_to_configs = "configurations"
path_to_results = "results"

file_metaconfiguration = "metaconfiguration.py"
file_resultdatafile = "simulation_results.hdf5"
file_batchconfiguration = "batchconfiguration.py"

# Left, middle and right delimiter for key->value pairs
# encoded into filenames (as used by the FileTools)
kvp_ldel = "["
kvp_mdel = "="
kvp_rdel = "]"

# Matrix exponential algorithm
matrix_exponential = "pade"
arnoldi_steps = 20
