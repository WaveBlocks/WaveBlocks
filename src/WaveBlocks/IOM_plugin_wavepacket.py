"""The WaveBlocks Project

IOM plugin providing functions for handling
homogeneous Hagedorn wavepacket data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_wavepacket(self, parameters, timeslots=None, blockid=0):
    """Add storage for the homogeneous wavepackets.
    :param parameters: An I{ParameterProvider} instance with at least the keys I{basis_size} and I{ncomponents}.
    """
    grp_wp = self._srf[self._prefixb+str(blockid)].require_group("wavepacket")

    # If we run with an adaptive basis size, then we must make the data tensor size maximal
    if parameters.has_key("max_basis_size"):
        bs = parameters["max_basis_size"]
    else:
        bs = np.max(parameters["basis_size"])

    # Create the dataset with appropriate parameters
    if timeslots is None:
        # This case is event based storing
        daset_tg = grp_wp.create_dataset("timegrid", (0,), dtype=np.integer, chunks=True, maxshape=(None,))
        daset_bs = grp_wp.create_dataset("basis_size", (0, parameters["ncomponents"]), dtype=np.integer, chunks=True, maxshape=(None,parameters["ncomponents"]))
        daset_pi = grp_wp.create_dataset("Pi", (0, 1, 5), dtype=np.complexfloating, chunks=True, maxshape=(None,1,5))
        daset_c = grp_wp.create_dataset("coefficients", (0, parameters["ncomponents"], bs), dtype=np.complexfloating, chunks=True, maxshape=(None,parameters["ncomponents"],bs))
    else:
        # User specified how much space is necessary.
        daset_tg = grp_wp.create_dataset("timegrid", (timeslots,), dtype=np.integer)
        daset_bs = grp_wp.create_dataset("basis_size", (timeslots, parameters["ncomponents"]), dtype=np.integer)
        daset_pi = grp_wp.create_dataset("Pi", (timeslots, 1, 5), dtype=np.complexfloating)
        daset_c = grp_wp.create_dataset("coefficients", (timeslots, parameters["ncomponents"], bs), dtype=np.complexfloating)

    # Attach pointer to data instead timegrid
    # Reason is that we have have two save functions but one timegrid
    #daset_bs.attrs["pointer"] = 0
    daset_pi.attrs["pointer"] = 0
    daset_c.attrs["pointer"] = 0


def delete_wavepacket(self, blockid=0):
    """Remove the stored wavepackets.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/wavepacket"]
    except KeyError:
        pass


def has_wavepacket(self, blockid=0):
    """Ask if the specified data block has the desired data tensor.
    """
    return "wavepacket" in self._srf[self._prefixb+str(blockid)].keys()


def save_wavepacket_parameters(self, parameters, timestep=None, blockid=0):
    """Save the parameters of the Hagedorn wavepacket to a file.
    :param parameters: The parameters of the Hagedorn wavepacket.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/Pi"
    timeslot = self._srf[pathd].attrs["pointer"]

    # Write the data
    self.must_resize(pathd, timeslot)
    self._srf[pathd][timeslot,0,:] = np.squeeze(np.array(parameters))

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathd].attrs["pointer"] += 1


def save_wavepacket_coefficients(self, coefficients, timestep=None, blockid=0):
    """Save the coefficients of the Hagedorn wavepacket to a file.
    :param coefficients: The coefficients of the Hagedorn wavepacket.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    pathbs = "/"+self._prefixb+str(blockid)+"/wavepacket/basis_size"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/coefficients"
    timeslot = self._srf[pathd].attrs["pointer"]

    # Write the data
    self.must_resize(pathd, timeslot)
    self.must_resize(pathbs, timeslot)
    for index, item in enumerate(coefficients):
        bs = item.shape[0]
        self._srf[pathbs][timeslot,index] = bs
        self._srf[pathd][timeslot,index,:bs] = np.squeeze(item)

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    #self._srf[pathbs].attrs["pointer"] += 1
    self._srf[pathd].attrs["pointer"] += 1


# The basis size already gets stored when saving the coefficients!
# def save_wavepacket_basissize(self, basissize, timestep=None, blockid=0):
#     """Save the basis size of the Hagedorn wavepacket to a file.
#     :param basissize: The basis size of the Hagedorn wavepacket.
#     """
#     pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
#     pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/basis_size"
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


def load_wavepacket_timegrid(self, blockid=0):
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    return self._srf[pathtg][:]


def load_wavepacket_parameters(self, timestep=None, blockid=0):
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/Pi"
    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        params = self._srf[pathd][index,0,:]
    else:
        params = self._srf[pathd][...,0,:]

    return params


def load_wavepacket_coefficients(self, timestep=None, blockid=0):
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/coefficients"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        return self._srf[pathd][index,...]
    else:
        return self._srf[pathd][...]


def load_wavepacket_basissize(self, timestep=None, blockid=0):
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket/basis_size"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        size = self._srf[pathd][index,:]
    else:
        size = self._srf[pathd][...,:]

    return size
