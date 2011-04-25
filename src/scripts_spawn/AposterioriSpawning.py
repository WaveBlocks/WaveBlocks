"""The WaveBlocks Project

Script to spawn new wavepackets aposteriori to an already completed simulation.
This can be used to evaluate spawning errors and test criteria for finding the
best spawning time.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys

import numpy as np
from scipy import linalg as spla

from WaveBlocks import AdiabaticSpawner
from WaveBlocks import IOManager
from WaveBlocks import PotentialFactory
from WaveBlocks import HagedornWavepacket


def aposteriori_spawning(f, p, datablock=0):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    @keyword datablock: The data block where the results are.    
    """    
    # Number of time steps we saved
    timesteps = f.load_wavepacket_timegrid(block=datablock)
    nrtimesteps = timesteps.shape[0]
        
    params = f.load_wavepacket_parameters(block=datablock)
    coeffs = f.load_wavepacket_coefficients(block=datablock)

    # A data transformation needed by API specification
    coeffs = [ [ coeffs[i,j,:] for j in xrange(p.ncomponents) ] for i in xrange(nrtimesteps) ]

    # Initialize a mother Hagedorn wavepacket with the data from another simulation
    HAWP = HagedornWavepacket(p)
    HAWP.set_quadrator(None)
    
    # Initialize an empty wavepacket for spawning
    SWP = HagedornWavepacket(p)
    SWP.set_quadrator(None)

    # Initialize a Spawner
    AS = AdiabaticSpawner(p)

    # Iterate over all timesteps and spawn
    for i, step in enumerate(timesteps):
        print(" Try spawning at timestep "+str(step))

        # Configure the wave packet and project to the eigenbasis.
        HAWP.set_parameters(params[i])
        HAWP.set_coefficients(coeffs[i])
        #HAWP.project_to_eigen(Potential)

        # Try spawning a new packet
        ps = AS.estimate_parameters(HAWP, 0)
        SWP.set_parameters(ps)
        AS.project_coefficients(HAWP, SWP)

        # Save the spawned packet
        f.save_wavepacket_parameters(SWP.get_parameters(), timestep=step, block=1)
        f.save_wavepacket_coefficients(SWP.get_coefficients(), timestep=step, block=1)
        


if __name__ == "__main__":

    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    parameters = iom.get_parameters()

    # Spawning related configurations
    # todo: Ugly, remove and replace with better solution
    parameters["K0"] = 50
    parameters["spawn_threshold"] = 1e-4

    # Second data block for the spawned packet
    iom.create_block()

    iom.add_grid_reference()
    iom.add_wavepacket(parameters, block=1)
                                
    if parameters["algorithm"] == "hagedorn":
        # Change the simulation algorithm used to allow for
        # specific observable calculators
        parameters["algorithm"] = "spawning_apost"
        iom.update_simulation_parameters(parameters)
        aposteriori_spawning(iom, parameters)
        
    else:
        raise ValueError("Unknown propagator algorithm.")
    
    iom.finalize()
