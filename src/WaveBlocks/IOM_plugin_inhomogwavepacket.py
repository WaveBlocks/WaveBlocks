"""The WaveBlocks Project

IOM plugin providing functions for handling
inhomogeneous Hagedorn wavepacket data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_inhomogwavepacket(self, parameters, timeslots=None, blockid=0):
    r"""
    Add storage for the inhomogeneous wavepackets.

    :param parameters: An ``ParameterProvider`` instance with at least the keys ``basis_size`` and ``ncomponents``.
    """
    grp_wp = self._srf[self._prefixb+str(blockid)].require_group("wavepacket_inhomog")

    # If we run with an adaptive basis size, then we must make the data tensor size maximal
    if parameters.has_key("max_basis_size"):
        bs = parameters["max_basis_size"]
    else:
        bs = np.max(parameters["basis_size"])

    # Create the dataset with appropriate parameters
    if timeslots is None:
        # This case is event based storing
        daset_tg = grp_wp.create_dataset("timegrid", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
        daset_bs = grp_wp.create_dataset("basis_size", (0, parameters["ncomponents"]), dtype=np.integer, chunks=True, maxshape=(None, parameters["ncomponents"]))
        daset_pi = grp_wp.create_dataset("Pi", (0, parameters["ncomponents"], 5), dtype=np.complexfloating, chunks=True, maxshape=(None, parameters["ncomponents"], 5))
        daset_c = grp_wp.create_dataset("coefficients", (0, parameters["ncomponents"], bs), dtype=np.complexfloating, chunks=True, maxshape=(None, parameters["ncomponents"], bs))
    else:
        # User specified how much space is necessary.
        daset_tg = grp_wp.create_dataset("timegrid", (timeslots,), dtype=np.integer)
        daset_bs = grp_wp.create_dataset("basis_size", (timeslots, parameters["ncomponents"]), dtype=np.integer)
        daset_pi = grp_wp.create_dataset("Pi", (timeslots, parameters["ncomponents"], 5), dtype=np.complexfloating)
        daset_c = grp_wp.create_dataset("coefficients", (timeslots, parameters["ncomponents"], bs), dtype=np.complexfloating)

    # Attach pointer to data instead timegrid
    # Reason is that we have have two save functions but one timegrid
    #daset_bs.attrs["pointer"] = 0
    daset_pi.attrs["pointer"] = 0
    daset_c.attrs["pointer"] = 0


def delete_inhomogwavepacket(self, blockid=0):
    r"""
    Remove the stored wavepackets.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/wavepacket_inhomog"]
    except KeyError:
        pass


def has_inhomogwavepacket(self, blockid=0):
    r"""
    Ask if the specified data block has the desired data tensor.
    """
    return "wavepacket_inhomog" in self._srf[self._prefixb+str(blockid)].keys()


def save_inhomogwavepacket_parameters(self, parameters, timestep=None, blockid=0):
    r"""
    Save the parameters of the Hagedorn wavepacket to a file.

    :param parameters: The parameters of the Hagedorn wavepacket.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/Pi"
    timeslot = self._srf[pathd].attrs["pointer"]

    # Write the data
    self.must_resize(pathd, timeslot)
    for index, item in enumerate(parameters):
        self._srf[pathd][timeslot,index,:] = np.squeeze(np.array(item))

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathd].attrs["pointer"] += 1


def save_inhomogwavepacket_coefficients(self, coefficients, timestep=None, blockid=0):
    r"""
    Save the coefficients of the Hagedorn wavepacket to a file.

    :param coefficients: The coefficients of the Hagedorn wavepacket.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/timegrid"
    pathbs = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/basis_size"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/coefficients"
    timeslot = self._srf[pathd].attrs["pointer"]

    # Write the data
    self.must_resize(pathd, timeslot)
    self.must_resize(pathbs, timeslot)
    for index, item in enumerate(coefficients):
        bs = item.shape[0]
        self._srf[pathbs][timeslot,index] = bs
        self._srf[pathd][timeslot,index,:bs] = np.squeeze(np.array(item))

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    #self._srf[pathbs].attrs["pointer"] += 1
    self._srf[pathd].attrs["pointer"] += 1


# The basis size already gets stored when saving the coefficients!
# def save_inhomogwavepacket_basissize(self, basissize, timestep=None, blockid=0):
#     r"""
#     Save the basis size of the Hagedorn wavepacket to a file.
#
#     :param basissize: The basis size of the Hagedorn wavepacket.
#     """
#     pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/timegrid"
#     pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/basis_size"
#     timeslot = self._srf[pathd].attrs["pointer"]
#
#     # Write the data
#     self.must_resize(pathd, timeslot)
#     self._srf[pathd][timeslot,:] = np.squeeze(np.array(basissize))
#
#     # Write the timestep to which the stored values belong into the timegrid
#     self.must_resize(pathtg, timeslot)
#     self._srf[pathtg][timeslot] = timestep
#
#     # Update the pointer
#     self._srf[pathd].attrs["pointer"] += 1


def load_inhomogwavepacket_timegrid(self, blockid=0):
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/timegrid"
    return self._srf[pathtg][:]


def load_inhomogwavepacket_parameters(self, timestep=None, blockid=0):
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/Pi"
    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        params = [ self._srf[pathd][index,i,:] for i in xrange(self._parameters["ncomponents"]) ]
    else:
        params = [ self._srf[pathd][...,i,:] for i in xrange(self._parameters["ncomponents"]) ]

    return params


def load_inhomogwavepacket_coefficients(self, timestep=None, blockid=0):
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/coefficients"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        return self._srf[pathd][index,...]
    else:
        return self._srf[pathd][...]


def load_inhomogwavepacket_basissize(self, timestep=None, blockid=0):
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/basis_size"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        size = self._srf[pathd][index,:]
    else:
        size = self._srf[pathd][...,:]

    return size
