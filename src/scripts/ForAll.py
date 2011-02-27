"""The WaveBlocks Project

This file contains code some simple code to call a given
python script for a bunch of simulation result files.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
import os
import subprocess as sp

from WaveBlocks.FileTools import get_result_dirs, get_results_file
from WaveBlocks import GlobalDefaults


def execute_for_all(resultspath, scriptcode):
    """Call a given python script with the simulation results data file as first
    command line argument. The script and the data file are specified by (relative)
    file system paths.
    @param resultspath: The path where to look for simulation data.
    @param scriptcode: The python script that gets called for all simulations.
    """
    for simulationpath in get_result_dirs(resultspath):
        print(" Executing code for datafile in " + simulationpath)

        # The file with the simulation data
        afile = get_results_file(simulationpath)

        # Call the given script
        sp.call(["python", scriptcode, afile])    

        # Move plots away if any
        proc = sp.Popen("mv *.png " + simulationpath, shell=True)
        os.waitpid(proc.pid, 0)



if __name__ == "__main__":
    # The scripts to call for all results
    if len(sys.argv) >= 2:
        scriptcode = sys.argv[1]
    else:
        raise ValueError("No other code given")

    # The path where we find the results
    if len(sys.argv) >= 3:
        resultspath = sys.argv[2]
    else:
        resultspath = GlobalDefaults.path_to_results

    print("Will execute the code in '" + scriptcode + "' for all files in '" + resultspath + "'")

    execute_for_all(resultspath, scriptcode)

    print("Done")
