"""The WaveBlocks Project

Sample wavepackets at the nodes of a given grid and save the results back
to the given simulation data file.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from WaveBlocks import PotentialFactory
from WaveBlocks import WaveFunction
from WaveBlocks import HagedornWavepacket
from WaveBlocks import IOManager


def compute_evaluate_wavepackets(f, basis="eigen", datablock=0):
    """Evaluate a homogeneous Hagdorn wavepacket on a given grid for each timestep.
    @param f: An I{IOManager} instance providing the simulation data.
    @keyword basis: The basis where the evaluation is done. Can be 'eigen' or 'canonical'.
    @keyword datablock: The data block where the results are.
    """
    p = f.get_parameters()
    
    # Get the data
    grid = f.load_grid(block=datablock)
    
    # Number of time steps we saved
    timesteps = f.load_wavepacket_timegrid(block=datablock)
    nrtimesteps = timesteps.shape[0]
    
    params = f.load_wavepacket_parameters(block=datablock)
    coeffs = f.load_wavepacket_coefficients(block=datablock)
    
    # A data transformation needed by API specification
    coeffs = [ [ coeffs[i,j,:] for j in xrange(p.ncomponents) ] for i in xrange(nrtimesteps)]
    
    # We want to save wavefunctions, thus add a data slot to the data file
    f.add_wavefunction(p, timeslots=nrtimesteps, block=datablock)
    
    # Prepare the potential for basis transformations
    Potential = PotentialFactory.create_potential(p)
    
    HAWP = HagedornWavepacket(p)
    HAWP.set_quadrator(None)
    
    WF = WaveFunction(p)
    WF.set_grid(grid)
    
    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Evaluating homogeneous wavepacket at timestep "+str(step))
        
        # Configure the wavepacket
        HAWP.set_parameters(params[i])
        HAWP.set_coefficients(coeffs[i])

        # Project to the eigenbasis if desired
        if basis == "eigen":
            HAWP.project_to_eigen(Potential)
        
        # Evaluate the wavepacket
        values = HAWP.evaluate_at(grid, prefactor=True)
        
        WF.set_values(values)

        # Save the wave function
        f.save_wavefunction(WF, timestep=step, block=datablock)
