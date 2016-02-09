"""The WaveBlocks Project

Calculate the norms of the different wavefunctions as well as the sum of all norms.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from WaveBlocks import PotentialFactory
from WaveBlocks import WaveFunction


def compute_norm(iom, blockid=0):
    """Compute the norm of a wavepacket timeseries.
    :param iom: An ``IOManager`` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()

    if iom.has_grid(blockid=blockid):
        grid = iom.load_grid(blockid=blockid)
    else:
        grid = iom.load_grid(blockid="global")

    # Number of time steps we saved
    timesteps = iom.load_wavefunction_timegrid(blockid=blockid)
    nrtimesteps = timesteps.shape[0]

    # We want to save norms, thus add a data slot to the data file
    iom.add_norm(parameters, timeslots=nrtimesteps, blockid=blockid)

    # Precalculate eigenvectors for efficiency
    Potential = PotentialFactory().create_potential(parameters)
    eigenvectors = Potential.evaluate_eigenvectors_at(grid)

    WF = WaveFunction(parameters)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Computing norms of timestep "+str(step))

        values = iom.load_wavefunction(timestep=step, blockid=blockid)
        values = [ values[j,...] for j in xrange(parameters["ncomponents"]) ]

        # Calculate the norm of the wave functions projected into the eigenbasis
        values_e = Potential.project_to_eigen(grid, values, eigenvectors)
        WF.set_values(values_e)
        norms = WF.get_norm()

        iom.save_norm(norms, timestep=step, blockid=blockid)
