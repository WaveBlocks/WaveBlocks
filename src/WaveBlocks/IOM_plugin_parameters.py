"""The WaveBlocks Project

IOM plugin providing functions for handling simulation parameter data.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import pickle

import ParameterProvider as ParameterProvider


def add_parameters(self, block="global"):
    """Add storage for the simulation parameters.
    """
    # Store the simulation parameters
    # We are only interested in the attributes of this data set
    # as they are used to store the simulation parameters.
    paset = self.srf["datablock_"+str(block)].create_dataset("simulation_parameters", (1,1))


def delete_parameters(self, block="global"):
    """Remove the stored simulation parameters.
    """
    try:
        del self.srf["datablock_"+str(block)+"/simulation_parameters"]
    except KeyError:
        pass


def has_parameters(self, block="global"):
    """Ask if the specified data block has the desired data tensor.
    """
    return "simulation_parameters" in self.srf["datablock_"+str(block)].keys()


def save_parameters(self, parameters, block="global"):
    """Save the norm of wavefunctions or wavepackets.
    """
    paset = self.srf["datablock_"+str(block)+"/simulation_parameters"]

    for param, value in parameters:
        # Store all the values as pickled strings because hdf can
        # only store strings or ndarrays as attributes.
        paset.attrs[param] = pickle.dumps(value)


def load_parameters(self, block="global"):
    """Load the simulation parameters.
    """
    p = self.srf["datablock_"+str(block)+"/simulation_parameters"].attrs
    PP = ParameterProvider.ParameterProvider()

    for key, value in p.iteritems():
        PP[key] = pickle.loads(value)
        # Compute some values on top of the given input parameters
        PP.compute_parameters()

    return PP


def update_parameters(self, parameters, block="global"):
    params = self.load_parameters(block=block)
    self.delete_parameters(block=block)
    params.update_parameters(parameters)
    self.add_parameters(block=block)
    self.save_parameters(params, block=block)
