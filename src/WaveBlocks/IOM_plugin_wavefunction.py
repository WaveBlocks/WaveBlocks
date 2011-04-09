"""The WaveBlocks Project

IOM plugin providing functions for handling wavefunction data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_wavefunction(self, parameters, timeslots=None, block=0):
    """Add storage for the sampled wavefunction.
    """
    grp_wf = self.srf["datablock_"+str(block)].require_group("wavefunction")

    # Create the dataset with appropriate parameters
    if timeslots is None:
        # This case is event based storing
        daset_psi = grp_wf.create_dataset("Psi", (1, parameters.ncomponents, parameters.ngn), dtype=np.complexfloating, chunks=(1, parameters.ncomponents, parameters.ngn))
        daset_psi_tg = grp_wf.create_dataset("timegrid", (timeslots,), dtype=np.integer, chunks=(1,))

        daset_psi.resize(0, axis=0)
        daset_psi_tg.resize(0, axis=0)
    else:
        # User specified how much space is necessary.
        daset_psi = grp_wf.create_dataset("Psi", (timeslots, parameters.ncomponents, parameters.ngn), dtype=np.complexfloating)
        daset_psi_tg = grp_wf.create_dataset("timegrid", (timeslots,), dtype=np.integer)
        
    daset_psi_tg.attrs["pointer"] = 0


def delete_wavefunction(self, block=0):
    """Remove the stored wavefunction.
    """
    try:
        del self.srf["datablock_"+str(block)+"/wavefunction"]
    except KeyError:
        pass


def save_wavefunction(self, wavefunctionvalues, block=0, timestep=None):
    """Save a I{WaveFunction} instance. The output is suitable for the plotting routines.
    @param wavefunctionvalues: The I{WaveFunction} instance to save.
    @keyword block: The data block where to store the wavefunction.
    """
    #@refactor: take wavefunction or wavefunction.get_values() as input?
    pathtg = "/datablock_"+str(block)+"/wavefunction/timegrid"
    pathd = "/datablock_"+str(block)+"/wavefunction/Psi"
    timeslot = self.srf[pathtg].attrs["pointer"]

    # Store the values given
    self.must_resize(pathd, timeslot)
    
    for index, item in enumerate(wavefunctionvalues):
        self.srf[pathd][timeslot,index,:] = item
        
    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self.srf[pathtg][timeslot] = timestep
    
    # Update the pointer
    self.srf[pathtg].attrs["pointer"] += 1


def load_wavefunction_timegrid(self, block=0):
    pathtg = "/datablock_"+str(block)+"/wavefunction/timegrid"
    return self.srf[pathtg][:]


def load_wavefunction(self, timestep=None, block=0):
    pathtg = "/datablock_"+str(block)+"/wavefunction/timegrid"
    pathd = "/datablock_"+str(block)+"/wavefunction/Psi"
    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        return self.srf[pathd][index,...]
    else:
        return self.srf[pathd][...]
