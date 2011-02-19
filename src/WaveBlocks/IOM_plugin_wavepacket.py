"""The WaveBlocks Project

IOM plugin providing functions for handling
homogeneous Hagedorn wavepacket data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_wavepacket(self, parameters, timeslots=None, block=0):
    # Store the wavepackets
    grp_wp = self.srf["datablock_"+str(block)].require_group("wavepacket")

    # Create the dataset with appropriate parameters
    if timeslots is None:
        # This case is event based storing
        daset_pi = grp_wp.create_dataset("Pi", (1, 1, 5), dtype=np.complexfloating, chunks=(1, 1, 5))
        daset_c = grp_wp.create_dataset("coefficients", (1, parameters.ncomponents, parameters.basis_size), dtype=np.complexfloating, chunks=(1, parameters.ncomponents, parameters.basis_size))
        daset_tg = grp_wp.create_dataset("timegrid", (1,), dtype=np.integer, chunks=(1,))

        daset_pi.resize(0, axis=0)
        daset_c.resize(0, axis=0)
        daset_tg.resize(0, axis=0)
    else:
        # User specified how much space is necessary.
        daset_pi = grp_wp.create_dataset("Pi", (timeslots, 1, 5), np.complexfloating)        
        daset_c = grp_wp.create_dataset("coefficients", (timeslots, parameters.ncomponents, parameters.basis_size), np.complexfloating)
        daset_tg = grp_wp.create_dataset("timegrid", (timeslots,), np.integer)

    # Attach pointer to data instead timegrid
    # Reason is that we have have two save functions but one timegrid
    daset_pi.attrs["pointer"] = 0
    daset_c.attrs["pointer"] = 0


def save_wavepacket_parameters(self, parameters, timestep=None, block=0):
    """Save the parameters of the Hagedorn wavepacket to a file.
    @param parameters: The parameters of the Hagedorn wavepacket.
    """
    pathtg = "/datablock_"+str(block)+"/wavepacket/timegrid"
    pathd = "/datablock_"+str(block)+"/wavepacket/Pi"
    timeslot = self.srf[pathd].attrs["pointer"]

    # Write the data
    self.must_resize(pathd, timeslot)
    self.srf[pathd][timeslot,0,:] = parameters

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self.srf[pathtg][timeslot] = timestep

    # Update the pointer
    self.srf[pathd].attrs["pointer"] += 1


def save_wavepacket_coefficients(self, coefficients, timestep=None, block=0):
    """Save the coefficients of the Hagedorn wavepacket to a file.
    @param coefficients: The coefficients of the Hagedorn wavepacket.
    """
    pathtg = "/datablock_"+str(block)+"/wavepacket/timegrid"
    pathd = "/datablock_"+str(block)+"/wavepacket/coefficients"
    timeslot = self.srf[pathd].attrs["pointer"]

    # Write the data
    self.must_resize(pathd, timeslot)
    for index, item in enumerate(coefficients):
        self.srf[pathd][timeslot,index,:] = np.squeeze(item)

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self.srf[pathtg][timeslot] = timestep

    # Update the pointer
    self.srf[pathd].attrs["pointer"] += 1


def load_wavepacket_timegrid(self, block=0):
    pathtg = "/datablock_"+str(block)+"/wavepacket/timegrid"
    return self.srf[pathtg][:]


def load_wavepacket_parameters(self, timestep=None, block=0):
    pathtg = "/datablock_"+str(block)+"/wavepacket/timegrid"
    pathd = "/datablock_"+str(block)+"/wavepacket/Pi"
    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        params = self.srf[pathd][index,0,:]
    else:
        params = self.srf[pathd][...,0,:]

    return params

    
def load_wavepacket_coefficients(self, timestep=None, block=0):
    pathtg = "/datablock_"+str(block)+"/wavepacket/timegrid"
    pathd = "/datablock_"+str(block)+"/wavepacket/coefficients"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        return self.srf[pathd][index,...]
    else:
        return self.srf[pathd][...]
