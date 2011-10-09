"""The WaveBlocks Project

IOM plugin providing functions for handling
inhomogeneous Hagedorn wavepacket data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_inhomogwavepacket(self, parameters, timeslots=None, blockid=0):
    """Add storage for the inhomogeneous wavepackets.
    """
    grp_wp = self._srf[self._prefixb+str(blockid)].require_group("wavepacket_inhomog")

    # Create the dataset with appropriate parameters
    if timeslots is None:
        # This case is event based storing
        daset_pi = grp_wp.create_dataset("Pi", (1, parameters["ncomponents"], 5), dtype=np.complexfloating, chunks=(1, parameters["ncomponents"], 5))
        daset_c = grp_wp.create_dataset("coefficients", (1, parameters["ncomponents"], parameters["basis_size"]),
                                        dtype=np.complexfloating, chunks=(1, parameters["ncomponents"], parameters["basis_size"]))
        daset_tg = grp_wp.create_dataset("timegrid", (1,), dtype=np.integer, chunks=(1,))

        daset_pi.resize(0, axis=0)
        daset_c.resize(0, axis=0)
        daset_tg.resize(0, axis=0)
    else:
        # User specified how much space is necessary.
        daset_pi = grp_wp.create_dataset("Pi", (timeslots, parameters["ncomponents"], 5), dtype=np.complexfloating)
        daset_c = grp_wp.create_dataset("coefficients", (timeslots, parameters["ncomponents"], parameters["basis_size"]), dtype=np.complexfloating)
        daset_tg = grp_wp.create_dataset("timegrid", (timeslots,), dtype=np.integer)

    # Attach pointer to data instead timegrid
    # Reason is that we have have two save functions but one timegrid
    daset_pi.attrs["pointer"] = 0
    daset_c.attrs["pointer"] = 0


def delete_inhomogwavepacket(self, blockid=0):
    """Remove the stored wavepackets.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/wavepacket_inhomog"]
    except KeyError:
        pass


def has_inhomogwavepacket(self, blockid=0):
    """Ask if the specified data block has the desired data tensor.
    """
    return "wavepacket_inhomog" in self._srf[self._prefixb+str(blockid)].keys()


def save_inhomogwavepacket_parameters(self, parameters, timestep=None, blockid=0):
    """Save the parameters of the Hagedorn wavepacket to a file.
    @param parameters: The parameters of the Hagedorn wavepacket.
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
    """Save the coefficients of the Hagedorn wavepacket to a file.
    @param coefficients: The coefficients of the Hagedorn wavepacket.
    """
    pathtg = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/timegrid"
    pathd = "/"+self._prefixb+str(blockid)+"/wavepacket_inhomog/coefficients"
    timeslot = self._srf[pathd].attrs["pointer"]

    # Write the data
    self.must_resize(pathd, timeslot)
    for index, item in enumerate(coefficients):
        self._srf[pathd][timeslot,index,:] = np.squeeze(np.array(item))

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self._srf[pathtg][timeslot] = timestep

    # Update the pointer
    self._srf[pathd].attrs["pointer"] += 1


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
