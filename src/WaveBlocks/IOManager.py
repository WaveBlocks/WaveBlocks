"""The WaveBlocks Project

This file contains code for serializing simulation data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import os
import types
import h5py as hdf
import numpy as np

import GlobalDefaults
import ParameterProvider as ParameterProvider


class IOManager:
    """An IOManager class that can save various simulation results into data
    files. The output files can be processed further for producing e.g. plots.
    """

    def __init__(self):
        self.parameters = None
        self.srf = None
        self.block_ids= []
        self.block_autonumber = 0


    def __str__(self):
        if self.srf is None:
            s = "IOManager instance without an open file."
        else:
            s = "IOManager instance with open file " + str(self.srf.filename)
        return s


    def __getattr__(self, key):
        """Try to load a plugin if a member function is not available.
        """
        parts = key.split("_")

        # Plugin name convention, we only trigger plugin loading
        # for requests starting with "add", "load" or "save".
        # However, IF we load a plugin, we load ALL functions it defines.
        if parts[0] not in ("add", "delete", "has", "load", "save", "update"):
            return
        else:
            print("Requested function: "+key)
            name = "IOM_plugin_" + parts[1]

        # Load the necessary plugin
        print("Plugin to load: "+name)
        try:
            plugin = __import__(name)
        except ImportError:
            raise ImportError("IOM plugin '"+name+"' not found!")

        # Filter out functions we want to add to IOM and
        # bind the methods to the current IOM instance
        for k, v in plugin.__dict__.iteritems():
            if type(v) == types.FunctionType:
                self.__dict__[k] = types.MethodType(v, self)
        
        # Now return the new function to complete it's call
        return self.__dict__[key]
    
    
    def create_file(self, parameters, filename=GlobalDefaults.file_resultdatafile):
        """Set up a new I{IOManager} instance. The output files are created and opened.
        """
        self.block_ids= []
        self.block_autonumber = 0

        #: Keep a reference to the parameters
        self.parameters = parameters

        # Create the file if it does not yet exist.
        # Otherwise raise an exception and avoid overwriting data.
        if os.path.lexists(filename):
            raise IOError("Output file already exists!")        
        else:
            f = self.srf = hdf.File(filename)
            f.attrs["number_blocks"] = 0

        # Save the simulation parameters
        self.create_block(blockid="global")
        self.add_parameters(block="global")
        self.save_parameters(parameters, block="global")


    def open_file(self, filename=GlobalDefaults.file_resultdatafile):
        """Load a given file that contains the results from a former simulation.
        @keyword filename: The filename/path of the file we try to load.
        """
        self.block_ids= []
        self.block_autonumber = 0

        if os.path.lexists(filename):
            self.srf = hdf.File(filename)
        else:
            raise ValueError("Output file does not exist!")

        # Load the simulation parameters from data block 0.
        self.parameters = self.load_parameters(block="global")


    def finalize(self):
        """Close the open output files."""
        self.srf.close()


    def get_number_blocks(self):
        """Return the number of data blocks the data file currently consists of.
        """
        return self.srf.attrs["number_blocks"]


    def create_block(self, blockid=None):
        # Create a data block. Each data block can store several chunks
        # of information, and there may be multiple blocks per file.
        if blockid is None:
            blockid = self.block_autonumber
            self.block_autonumber += 1

        self.block_ids.append(blockid)
        self.srf.create_group("datablock_" + str(blockid))
        self.srf.attrs["number_blocks"] += 1


    def must_resize(self, path, slot, axis=0):
        """Check if we must resize a given dataset and if yes, resize it.
        """
        # Ok, it's inefficient but sufficient for now.
        # todo: Consider resizing in bigger chunks and shrinking at the end if necessary.
        
        # Current size of the array
        cur_len = self.srf[path].shape[axis]

        # Is it smaller than what we need to store at slot "slot"?
        # If yes, then resize the array along the given axis.
        if cur_len-1 < slot:
            self.srf[path].resize(slot+1, axis=axis)


    def find_timestep_index(self, timegridpath, timestep):
        """Lookup the index for a given timestep.
        @note: Assumes the timegrid array is strictly monotone.
        """
        # todo: Make this more efficient
        # todo: allow for slicing etc
        timegrid = self.srf[timegridpath]
        index = np.squeeze(np.where(timegrid[:] == timestep))

        if index.shape == (0,):
            raise ValueError("No data for given timestep!")
        
        return index


    def split_data(self, data, axis):
        """Split a multi-dimensional data block into slabs along a given axis.
        @param data: The data tensor given.
        @param axis: The axis along which to split the data.
        @return: A list of slices.
        """
        parts = data.shape[axis]
        return np.split(data, parts, axis=axis)


    # Shortcut functions to IOM_plugin_parameters
    # Just for backward compatibility
    def get_parameters(self):
        print(" Depreceated get_parameters call at an IOManager instance!")        
        return self.load_parameters(block="global")
