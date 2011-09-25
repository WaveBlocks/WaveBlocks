"""The WaveBlocks Project

IOM plugin providing functions for handling grid data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np
import h5py as hdf


def add_grid(self, parameters, blockid=0):
    """Add storage for a grid.
    """
    self._srf[self._prefixb+str(blockid)].create_dataset("grid", (parameters["dimension"], parameters["ngn"]), np.floating)


def delete_grid(self, blockid=0):
    """Remove the stored grid.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/grid"]
    except KeyError:
        pass


def add_grid_reference(self, blockfrom=1, blockto=0):
    self._srf[self._prefixb+str(blockfrom)]["grid"] = hdf.SoftLink("/"+self._prefixb+str(blockto)+"/grid")


def save_grid(self, grid, blockid=0):
    """Save the grid nodes.
    """
    path = "/"+self._prefixb+str(blockid)+"/grid"
    self._srf[path][:] = np.real(grid)


def load_grid(self, blockid=0):
    """Load the grid nodes.
    """
    path = "/"+self._prefixb+str(blockid)+"/grid"
    return np.squeeze(self._srf[path])
