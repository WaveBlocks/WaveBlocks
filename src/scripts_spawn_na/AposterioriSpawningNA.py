"""The WaveBlocks Project

Script to spawn new wavepackets aposteriori to an already completed simulation.
This can be used to evaluate spawning errors and test criteria for finding the
best spawning time.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import sys

import numpy as np
from scipy import linalg as spla

from WaveBlocks import ParameterProvider
from WaveBlocks import IOManager
from WaveBlocks import PotentialFactory
from WaveBlocks import HagedornWavepacket
from WaveBlocks import NonAdiabaticSpawner


def aposteriori_spawning(fin, fout, pin, pout, components=None, save_canonical=False):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    @keyword datablock: The data block where the results are.    
    """
    # Number of time steps we saved
    timesteps = fin.load_wavepacket_timegrid()
    nrtimesteps = timesteps.shape[0]
    
    params = fin.load_wavepacket_parameters()
    coeffs = fin.load_wavepacket_coefficients()

    # A data transformation needed by API specification
    coeffs = [ [ coeffs[i,j,:] for j in xrange(pin["ncomponents"]) ] for i in xrange(nrtimesteps) ]

    # The potential
    Potential = PotentialFactory.create_potential(pin)

    # Initialize a mother Hagedorn wavepacket with the data from another simulation
    HAWP = HagedornWavepacket(pin)
    HAWP.set_quadrator(None)
    
    # Initialize an empty wavepacket for spawning
    SWP = HagedornWavepacket(pin)
    SWP.set_quadrator(None)

    # Initialize a Spawner
    NAS = NonAdiabaticSpawner(pout)

    # Try spawning for these components, if none is given, try it for all.
    if components is None:
        components = range(pin["ncomponents"])

    # Iterate over all timesteps and spawn
    for i, step in enumerate(timesteps):
        print(" Try spawning at timestep "+str(step))

        # Configure the wave packet and project to the eigenbasis.
        HAWP.set_parameters(params[i])
        HAWP.set_coefficients(coeffs[i])

        # Project to the eigenbasis as the parameter estimation
        # has to happen there because of coupling.
        T = HAWP.clone()
        T.project_to_eigen(Potential)

        # Try spawning a new packet for each component
        estps = [ NAS.estimate_parameters(T, component=acomp) for acomp in components ]

        for index, ps in enumerate(estps):
            if ps is not None:
                U = SWP.clone()
                U.set_parameters(ps)
                
                # Project the coefficients to the spawned packet
                NAS.project_coefficients(T, U, component=components[index])

                # Transform back
                if save_canonical is True:
                    T.project_to_canonical(Potential)
                    U.project_to_canonical(Potential)

                # Save the mother packet rest
                fout.save_wavepacket_parameters(T.get_parameters(), timestep=step, block=2*index)
                fout.save_wavepacket_coefficients(T.get_coefficients(), timestep=step, block=2*index)

                # Save the spawned packet
                fout.save_wavepacket_parameters(U.get_parameters(), timestep=step, block=2*index+1)
                fout.save_wavepacket_coefficients(U.get_coefficients(), timestep=step, block=2*index+1)




if __name__ == "__main__":
    # Input data manager
    iomin = IOManager()

    # Read file with simulation data
    try:
        iomin.open_file(filename=sys.argv[1])
    except IndexError:
        iomin.open_file()

    parametersin = iomin.get_parameters()

    # Check if we can start a spawning simulation
    if parametersin["algorithm"] != "hagedorn":
        iomin.finalize()
        raise ValueError("Unknown propagator algorithm.")
    
    # Parameters for spawning simulation
    parametersout = ParameterProvider()

    # Transfer the simulation parameters
    parametersout.set_parameters(parametersin.get_parameters())

    # And add spawning related configurations variables
    # todo: Ugly, remove and replace with better solution
    # reading values from a configuration file
    parametersout["algorithm"] = "spawning_apost_na"
    parametersout["spawn_threshold"] = 1e-10
    parametersout["spawn_max_order"] = 4
    parametersout["spawn_normed_gaussian"] = False
    parametersout["spawn_components"] = [1]

    # How much time slots do we need
    tm = parametersout.get_timemanager()
    slots = tm.compute_number_saves()
    
    # Second IOM for output data of the spawning simulation
    iomout = IOManager()
    iomout.create_file(parametersout, filename="simulation_results_spawn.hdf5")

    # Allocate all the data blocks
    for i in xrange(2*len(parametersout["spawn_components"])):
        if i == 0:
            iomout.add_grid(parametersout)
            iomout.save_grid(iomin.load_grid())
        else:
            iomout.create_block()
            iomout.add_grid_reference(blockfrom=i, blockto=0)        
        iomout.add_wavepacket(parametersout, block=i)
    
    # Really do the aposteriori spawning simulation
    aposteriori_spawning(iomin, iomout, parametersin, parametersout, components=parametersout["spawn_components"])

    # Close the inpout/output files
    iomin.finalize()
    iomout.finalize()
