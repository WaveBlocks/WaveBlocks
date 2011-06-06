"""The WaveBlocks Project

Calculate the norms of the different wavefunctions as well as the sum of all norms.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from WaveBlocks import PotentialFactory
from WaveBlocks import WaveFunction


def compute_norm(iom, block=0):
    """Compute the norm of a wavepacket timeseries.
    @param iom: An I{IOManager} instance providing the simulation data.
    @keyword block: The data block from which the values are read.
    """
    parameters = iom.get_parameters()

    nodes = iom.load_grid(block=block)

    # Number of time steps we saved
    timesteps = iom.load_wavefunction_timegrid(block=block)
    nrtimesteps = timesteps.shape[0]

    # We want to save norms, thus add a data slot to the data file
    iom.add_norm(parameters, timeslots=nrtimesteps, block=block)

    # Precalculate eigenvectors for efficiency
    Potential = PotentialFactory.create_potential(parameters)
    eigenvectors = Potential.evaluate_eigenvectors_at(nodes)

    WF = WaveFunction(parameters)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Computing norms of timestep "+str(step))

        values = iom.load_wavefunction(timestep=step, block=block)
        values = [ values[j,...] for j in xrange(parameters["ncomponents"]) ]

        # Calculate the norm of the wave functions projected into the eigenbasis
        values_e = Potential.project_to_eigen(nodes, values, eigenvectors)
        WF.set_values(values_e)
        norms = WF.get_norm()

        iom.save_norm(norms, timestep=step, block=block)
