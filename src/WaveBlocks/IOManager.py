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
        self.prefixb = "datablock_"
        self.prefixg = "group_"

        # The current open data file
        self.srf = None

        # The current global simulation parameters
        self.parameters = None

        # Book keeping data
        # todo: consider storing these values inside the data files
        self.block_ids= None
        self.block_count = None
        self.group_ids = None
        self.group_count = None


    def __str__(self):
        if self.srf is None:
            s = "IOManager instance without an open file."
        else:
            s = "IOManager instance with open file " + str(self.srf.filename) + "\n"
            s += " containing " + str(self.block_count) + " data blocks in "
            s += str(self.group_count) + " data groups."
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
        @param parameters: A I{ParameterProvider} instance containing the current simulation
        parameters. This is only used for determining the size of new data sets.
        @keyword filename: The filename (optionally with filepath) of the file we try
        to create. If not given the default value from I{GlobalDefaults} is used.
        """
        # Create the file if it does not yet exist.
        # Otherwise raise an exception to avoid overwriting data.
        if os.path.lexists(filename):
            raise IOError("Output file '"+str(filename)+"' already exists!")
        else:
            self.srf = hdf.File(filename)

        # Initialize the internal book keeping data
        self.block_ids = []
        self.block_count = 0
        self.group_ids = []
        self.group_count = 0

        # Keep a reference to the parameters
        self.parameters = parameters

        # Save the simulation parameters
        self.create_group(groupid="global")
        self.create_block(blockid="global", groupid="global")
        self.add_parameters(block="global")
        self.save_parameters(parameters, block="global")


    def open_file(self, filename=GlobalDefaults.file_resultdatafile):
        """Load a given file that contains the results from a former simulation.
        @keyword filename: The filename (optionally with filepath) of the file we try
        to load. If not given the default value from I{GlobalDefaults} is used.
        """
        # Try to open the file or raise an exception if it does not exist.
        if os.path.lexists(filename):
            if hdf.is_hdf5(filename):
                self.srf = hdf.File(filename)
            else:
                raise IOError("File '"+str(filename)+"' is not a hdf5 file")
        else:
            raise IOError("File '"+str(filename)+"' does not exist!")

        # Initialize the internal book keeping data
        self.block_ids = [ s[len(self.prefixb):] for s in self.srf.keys() if s.startswith(self.prefixb) ]
        self.block_count = len(self.block_ids)

        self.group_ids = [ s[len(self.prefixg):] for s in self.srf.keys() if s.startswith(self.prefixg) ]
        self.group_count = len(self.group_ids)

        # Load the simulation parameters from data block 0.
        self.parameters = self.load_parameters(block="global")


    def finalize(self):
        """Close the open output files."""
        if self.srf is None:
            return

        # Close the file
        self.srf.close()
        self.srf = None
        # Reset book keeping data
        self.parameters = None
        self.block_ids= None
        self.block_count = None
        self.group_ids = None
        self.group_count = None


    def get_number_blocks(self):
        """Return the number of data blocks the data file currently consists of.
        """
        return self.block_count


    def get_number_groups(self):
        """Return the number of groups in the data file.
        """
        return self.group_count


    def get_block_ids(self):
        """Return a copy of the list containing all block ids.
        """
        return self.block_ids[:]


    def get_group_ids(self):
        """Return a copy of the list containing all group ids.
        """
        return self.group_ids[:]


    def create_block(self, blockid=None, groupid="global"):
        """Create a data block with the specified block id. Each data block can
        store several chunks of information, and there can be an arbitrary number
        of data blocks per file.
        @param blockid: The id for the new data block. If not given the blockid
        will be choosen automatically. The block id has to be unique.
        """
        if self.srf is None:
            return

        if blockid is not None and (not str(blockid).isalnum() or str(blockid)[0].isdigit()):
            raise ValueError("Block ID allows only characters A-Z, a-z and 0-9 and no leading digit.")

        if blockid is not None and str(blockid) in self.block_ids:
            raise ValueError("Invalid or already used block ID: " + str(blockid))

        if blockid is None:
            # Try to find a valid autonumber
            autonumber = 0
            while str(autonumber) in self.block_ids:
                autonumber += 1
            blockid = str(autonumber)

        self.block_ids.append(str(blockid))
        self.block_count += 1

        # Create the data block
        self.srf.create_group("/"+self.prefixb + str(blockid))

        # Does the group already exist?
        if not str(groupid) in self.group_ids:
            self.create_group(groupid=groupid)

        # Put the data block into the group
        self.srf["/"+self.prefixb+str(blockid)].attrs["group"] = str(groupid)
        self.srf["/"+self.prefixg+str(groupid)+"/"+str(blockid)] = hdf.SoftLink("/"+self.prefixb+str(blockid))


    def create_group(self, groupid=None):
        """Create a data block with the specified block id. Each data block can
        store several chunks of information, and there can be an arbitrary number
        of data blocks per file.
        @param blockid: The id for the new data block. If not given the blockid
        will be choosen automatically. The block id has to be unique.
        """
        if self.srf is None:
            return

        if groupid is not None and (not str(groupid).isalnum() or str(groupid)[0].isdigit()):
            raise ValueError("Group ID allows only characters A-Z, a-z and 0-9 and no leading digit.")

        if groupid is not None and str(groupid) in self.group_ids:
            raise ValueError("Invalid or already used group ID: " + str(groupid))

        if groupid is None:
            # Try to find a valid autonumber
            autonumber = 0
            while str(autonumber) in self.group_ids:
                autonumber += 1
            groupid = str(autonumber)

        self.group_ids.append(str(groupid))
        self.group_count += 1
        # Create the group
        self.srf.create_group("/"+self.prefixg + str(groupid))


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
