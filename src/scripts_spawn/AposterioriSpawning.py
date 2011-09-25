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
from WaveBlocks import ParameterProvider


def aposteriori_spawning(fin, fout, pin, pout):
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

    # Initialize a mother Hagedorn wavepacket with the data from another simulation
    HAWP = HagedornWavepacket(pin)
    HAWP.set_quadrature(None)

    # Initialize an empty wavepacket for spawning
    SWP = HagedornWavepacket(pout)
    SWP.set_quadrature(None)

    # Initialize a Spawner
    AS = AdiabaticSpawner(pout)

    # Iterate over all timesteps and spawn
    for i, step in enumerate(timesteps):
        print(" Try spawning at timestep "+str(step))

        # Configure the wave packet and project to the eigenbasis.
        HAWP.set_parameters(params[i])
        HAWP.set_coefficients(coeffs[i])
        #HAWP.project_to_eigen(Potential)

        # Try spawning a new packet
        ps = AS.estimate_parameters(HAWP, 0)

        if ps is not None:
            SWP.set_parameters(ps)
            AS.project_coefficients(HAWP, SWP)

            # Save the spawned packet
            fout.save_wavepacket_parameters(HAWP.get_parameters(), timestep=step)
            fout.save_wavepacket_coefficients(HAWP.get_coefficients(), timestep=step)

            fout.save_wavepacket_parameters(SWP.get_parameters(), timestep=step, blockid=1)
            fout.save_wavepacket_coefficients(SWP.get_coefficients(), timestep=step, blockid=1)




if __name__ == "__main__":
    # Input data manager
    iomin = IOManager()

    # Read file with simulation data
    try:
        iomin.open_file(filename=sys.argv[1])
    except IndexError:
        iomin.open_file()

    # Read a configuration file with the spawn parameters
    try:
        parametersspawn = ParameterProvider()
        parametersspawn.read_parameters(sys.argv[2])
    except IndexError:
        raise IOError("No spawn configuration given!")

    parametersin = iomin.load_parameters()

    # Check if we can start a spawning simulation
    if parametersin["algorithm"] != "hagedorn":
        iomin.finalize()
        raise ValueError("Unknown propagator algorithm.")

    # Parameters for spawning simulation
    parametersout = ParameterProvider()

    # Transfer the simulation parameters
    parametersout.set_parameters(parametersin)

    # And add spawning related configurations variables
    parametersout.update_parameters(parametersspawn)

    # How much time slots do we need
    tm = parametersout.get_timemanager()
    slots = tm.compute_number_saves()

    # Second IOM for output data of the spawning simulation
    iomout = IOManager()
    iomout.create_file(parametersout, filename="simulation_results_spawn.hdf5")
    iomout.create_block()
    iomout.create_block()

    iomout.add_grid(parametersout)
    iomout.save_grid(iomin.load_grid())
    iomout.add_grid_reference()

    iomout.add_wavepacket(parametersin)
    iomout.add_wavepacket(parametersout, blockid=1)

    # Really do the aposteriori spawning simulation
    aposteriori_spawning(iomin, iomout, parametersin, parametersout)

    # Close the inpout/output files
    iomin.finalize()
    iomout.finalize()
