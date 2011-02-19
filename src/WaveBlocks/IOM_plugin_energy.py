"""The WaveBlocks Project

IOM plugin providing functions for handling energy data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_energy(self, parameters, timeslots=None, block=0, total=False):
    # Store the potential and kinetic energies
    grp_ob = self.srf["datablock_"+str(block)].require_group("observables")
    
    # Create the dataset with appropriate parameters
    grp_en = grp_ob.create_group("energies")

    if timeslots is None:
        # This case is event based storing
        daset_ek = grp_en.create_dataset("kinetic", (1, parameters.ncomponents), dtype=np.floating, chunks=(1, parameters.ncomponents))
        daset_ep = grp_en.create_dataset("potential", (1, parameters.ncomponents), dtype=np.floating, chunks=(1, parameters.ncomponents))
        daset_tg = grp_en.create_dataset("timegrid", (1,), dtype=np.integer, chunks=(1,))

        daset_ek.resize(0, axis=0)
        daset_ep.resize(0, axis=0)

        daset_tg.resize(0, axis=0)

        if total is True:
            daset_to = grp_en.create_dataset("total", (1, 1), dtype=np.floating, chunks=(1, 1))
            daset_to.resize(0, axis=0)
            daset_to.attrs["pointer"] = 0
    else:
        # User specified how much space is necessary.
        daset_ek = grp_en.create_dataset("kinetic", (timeslots, parameters.ncomponents), dtype=np.floating)
        daset_ep = grp_en.create_dataset("potential", (timeslots, parameters.ncomponents), dtype=np.floating)
        daset_tg = grp_en.create_dataset("timegrid", (timeslots,), dtype=np.integer)

        if total is True:
            daset_to = grp_en.create_dataset("total", (timeslots, 1), dtype=np.floating)
            daset_to.attrs["pointer"] = 0

    daset_tg.attrs["pointer"] = 0


def save_energy(self, energies, timestep=None, block=0):
    """Save the kinetic and potential energies to a file.
    @param energies: A tuple \texttt{(ekin, epot)} containing the energies.
    """
    pathtg = "/datablock_"+str(block)+"/observables/energies/timegrid"
    pathd1 = "/datablock_"+str(block)+"/observables/energies/kinetic"
    pathd2 = "/datablock_"+str(block)+"/observables/energies/potential"
    timeslot = self.srf[pathtg].attrs["pointer"]

    #todo: remove np,array
    ekin = np.real(np.array(energies[0]))
    epot = np.real(np.array(energies[1]))

    # Write the data
    self.must_resize(pathd1, timeslot)
    self.must_resize(pathd2, timeslot)
    self.srf[pathd1][timeslot,:] = ekin
    self.srf[pathd2][timeslot,:] = epot

    # Write the timestep to which the stored values belong into the timegrid
    self.must_resize(pathtg, timeslot)
    self.srf[pathtg][timeslot] = timestep

    # Update the pointer
    self.srf[pathtg].attrs["pointer"] += 1


def save_energy_total(self, total_energy, timestep=None, block=0):
    """Save the total to a file.
    @param total_energy: An array containing a time series of the total energy.
    """
    pathd = "/datablock_"+str(block)+"/observables/energies/total"

    timeslot = self.srf[pathd].attrs["pointer"]

    #todo: remove np,array
    etot = np.real(np.array(total_energy))

    # Write the data
    self.must_resize(pathd, timeslot)
    self.srf[pathd][timeslot,0] = etot

    # Update the pointer
    self.srf[pathd].attrs["pointer"] += 1


def load_energy_timegrid(self, block=0):
    pathtg = "/datablock_"+str(block)+"/observables/energies/timegrid"
    return self.srf[pathtg][:]


def load_energy(self, timestep=None, block=0):
    pathtg = "/datablock_"+str(block)+"/observables/energies/timegrid"
    pathd1 = "/datablock_"+str(block)+"/observables/energies/kinetic"
    pathd2 = "/datablock_"+str(block)+"/observables/energies/potential"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        ekin = self.srf[pathd1][index,...]
        epot = self.srf[pathd2][index,...]
    else:
        ekin = self.srf[pathd1][...]
        epot = self.srf[pathd2][...]
    
    return (ekin, epot)


def load_energy_total(self, timestep=None, block=0):
    pathtg = "/datablock_"+str(block)+"/observables/energies/timegrid"
    pathd = "/datablock_"+str(block)+"/observables/energies/total"

    if timestep is not None:
        index = self.find_timestep_index(pathtg, timestep)
        etot = self.srf[pathd][index,...]
    else:
        etot = self.srf[pathd][...]

    return etot
