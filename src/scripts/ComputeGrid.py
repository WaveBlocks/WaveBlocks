"""The WaveBlocks Project

A script to compute and save a position space grid.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
import numpy as np

from WaveBlocks import IOManager


def compute_grid(iom, blockid):
    # Load the parameter from the global block
    parameters = iom.load_parameters(blockid="global")

    # Compute the position space grid points
    nodes = parameters["f"] * np.pi * np.arange(-1, 1, 2.0/parameters["ngn"], dtype=np.complexfloating)

    iom.add_grid(parameters, blockid=blockid)
    iom.save_grid(nodes, blockid=blockid)




if __name__ == "__main__":

    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()
    
    # Blocks where we store the grid, per default
    # this is only the global data block.
    blockids = ["global"]

    for blockid in blockids:
        if iom.has_grid(blockid=blockid):
            print("Datablock '"+str(blockid)+"' already contains a grid, silent skip.")
            continue

        compute_grid(iom, blockid=blockid)

    iom.finalize()
