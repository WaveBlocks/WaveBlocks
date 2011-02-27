"""The WaveBlocks Project

Compute the kinetic and potential energies of the inhomogeneous wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from WaveBlocks import PotentialFactory
from WaveBlocks import HagedornMultiWavepacket


def compute_energy(f, datablock=0):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    @keyword datablock: The data block where the results are.    
    """
    p = f.get_parameters()
    
    # Number of time steps we saved
    timesteps = f.load_inhomogwavepacket_timegrid(block=datablock)
    nrtimesteps = timesteps.shape[0]
    
    # We want to save energies, thus add a data slot to the data file
    f.add_energy(p, timeslots=nrtimesteps, block=datablock)
    
    Potential = PotentialFactory.create_potential(p)
    
    params = f.load_inhomogwavepacket_parameters(block=datablock)
    coeffs = f.load_inhomogwavepacket_coefficients(block=datablock)

    # A data transformation needed by API specification
    params = [ [ params[j][i,:] for j in xrange(p.ncomponents) ] for i in xrange(nrtimesteps) ]
    coeffs = [ [ coeffs[i,j,:] for j in xrange(p.ncomponents) ] for i in xrange(nrtimesteps) ]
    
    # Initialize a hagedorn wave packet with the data
    HAWP = HagedornMultiWavepacket(p)
    HAWP.set_quadrator(None)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Computing energies of timestep "+str(step))

        # Configure the wave packet and project to the eigenbasis.
        HAWP.set_parameters(params[i])
        HAWP.set_coefficients(coeffs[i])
        HAWP.project_to_eigen(Potential)

        # Compute the energies
        ekin = HAWP.kinetic_energy()
        epot = HAWP.potential_energy(Potential.evaluate_eigenvalues_at)

        f.save_energy((ekin, epot), timestep=step, block=datablock)
