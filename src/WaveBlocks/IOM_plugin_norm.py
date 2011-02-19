"""The WaveBlocks Project

IOM plugin providing functions for handling norm data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_norm(self, parameters, timeslots=None, block=0):
    """Store the norms
    """
    grp_ob = self.srf["datablock_"+str(block)].require_group("observables")
    
    # Create the dataset with appropriate parameters
    grp_no = grp_ob.create_group("norm")
    
    if timeslots is None:
        # This case is event based storing
        daset_n = grp_no.create_dataset("norm", (1, parameters.ncomponents), dtype=np.floating, chunks=(1, parameters.ncomponents))
        daset_tg = grp_no.create_dataset("timegrid", (1,), dtype=np.integer, chunks=(1,))
        
        daset_n.resize(0, axis=0)
        daset_tg.resize(0, axis=0)
    else:
        # User specified how much space is necessary.
        daset_n = grp_no.create_dataset("norm", (timeslots, parameters.ncomponents), dtype=np.floating)
        daset_tg = grp_no.create_dataset("timegrid", (timeslots,), dtype=np.integer)
        
    daset_tg.attrs["pointer"] = 0


def save_norm(self, norm, timestep=None, block=0):
    """Save the norm of wavefunctions or wavepackets.
    """
    pathtg = "/datablock_"+str(block)+"/observables/norm/timegrid"
    pathd = "/datablock_"+str(block)+"/observables/norm/norm"
    timeslot = self.srf[pathtg].attrs["pointer"]
    
    # Refactor: remove np.array
    norms = np.real(np.array(norm))
    
    # Write the data
    self.must_resize(pathd, timeslot)
    self.srf[pathd][timeslot,:] = norms
    
    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self.srf[pathtg][timeslot] = timestep
    
    # Update the pointer
    self.srf[pathtg].attrs["pointer"] += 1

    
def load_norm_timegrid(self, block=0):
    """Load the timegrid corresponding to the norm data.
    """
    pathtg = "/datablock_"+str(block)+"/observables/norm/timegrid"
    return self.srf[pathtg][:]


def load_norm(self, timestep=None, block=0):
    """Load the norm data.
    """
    pathtg = "/datablock_"+str(block)+"/observables/norm/timegrid"
    pathd = "/datablock_"+str(block)+"/observables/norm/norm"
    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        return self.srf[pathd][index,...]
    else:
        return self.srf[pathd][...]
