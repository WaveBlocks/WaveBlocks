"""The WaveBlocks Project

Calculate the norms of the different wave packets as well as the sum of all norms.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from WaveBlocks import PotentialFactory
from WaveBlocks import WaveFunction
from WaveBlocks import IOManager


def compute_norm(f, datablock=0):
    """Compute the norm of a wavepacket timeseries.
    @param f: An I{IOManager} instance providing the simulation data.
    @keyword datablock: The data block where the results are.
    """
    p = f.get_parameters()
    
    nodes = f.load_grid(block=datablock)

    # Number of time steps we saved
    timesteps = f.load_wavefunction_timegrid(block=datablock)
    nrtimesteps = timesteps.shape[0]

    # We want to save norms, thus add a data slot to the data file
    f.add_norm(p, timeslots=nrtimesteps, block=datablock)

    # Precalculate eigenvectors for efficiency
    Potential = PotentialFactory.create_potential(p)
    eigenvectors = Potential.evaluate_eigenvectors_at(nodes)

    WF = WaveFunction(p)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Computing norms of timestep "+str(step))
        
        values = f.load_wavefunction(timestep=step, block=datablock)
        values = [ values[j,...] for j in xrange(p.ncomponents) ]
        
        # Calculate the norm of the wave functions projected into the eigenbasis
        values_e = Potential.project_to_eigen(nodes, values, eigenvectors)
        WF.set_values(values_e)
        norms = WF.get_norm()
        
        f.save_norm(norms, timestep=step, block=datablock)
