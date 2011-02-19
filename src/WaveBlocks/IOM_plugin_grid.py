"""The WaveBlocks Project

IOM plugin providing functions for handling grid data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_grid(self, parameters, block=0):
    """Add storage for a grid
    """
    self.srf["datablock_"+str(block)].create_dataset("grid", (parameters.dimension, parameters.ngn), np.floating)


def add_grid_reference(self, blockfrom=1, blockto=0):
    self.srf["datablock_"+str(blockfrom)]["grid"] = hdf.SoftLink("/datablock_"+str(blockto)+"/grid")


def save_grid(self, grid, block=0):
    """Save the grid nodes.
    """
    path = "/datablock_"+str(block)+"/grid"
    self.srf[path][:] = np.real(grid)


def load_grid(self, block=0):
    """Load the grid nodes.
    """
    path = "/datablock_"+str(block)+"/grid"
    return np.squeeze(self.srf[path])
