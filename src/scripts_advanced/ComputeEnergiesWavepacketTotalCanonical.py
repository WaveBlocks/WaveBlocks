"""The WaveBlocks Project

Compute the kinetic and potential energies of the homogeneous wavepackets
as well as the sum of all energies. The sum is computed in the canonical
basis.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys

from WaveBlocks import PotentialFactory
from WaveBlocks import HagedornWavepacket
from WaveBlocks import IOManager


def compute_energy(f, datablock=0):
    p = f.get_parameters()
    
    # Number of time steps we saved
    timesteps = f.load_wavepacket_timegrid(block=datablock)
    nrtimesteps = timesteps.shape[0]
    
    # We want to save energies, thus add a data slot to the data file
    f.add_energy(p, timeslots=nrtimesteps, block=datablock, total=True)
    
    Potential = PotentialFactory.create_potential(p)
    
    params = f.load_wavepacket_parameters(block=datablock)
    coeffs = f.load_wavepacket_coefficients(block=datablock)
    
    # A data transformation needed by API specification
    coeffs = [ [ coeffs[i,j,:] for j in xrange(p.ncomponents) ] for i in xrange(nrtimesteps) ]
    
    # Initialize a hagedorn wave packet with the data    
    HAWP = HagedornWavepacket(p)
    HAWP.set_quadrator(None)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Computing energies of timestep "+str(step))
        
        # Configure the wave packet
        HAWP.set_parameters(params[i])
        HAWP.set_coefficients(coeffs[i])
        
        # Compute overall energy in canonical basis
        ekin = HAWP.kinetic_energy(summed=True)
        epot = HAWP.potential_energy(Potential.evaluate_at, summed=True)
        etot = ekin + epot

        f.save_energy_total(etot, timestep=step, block=datablock)
        
        # Transform to eigenbasis
        HAWP.project_to_eigen(Potential)
        
        # Compute the components' energies in the eigenbasis
        ekin = HAWP.kinetic_energy()
        epot = HAWP.potential_energy(Potential.evaluate_eigenvalues_at)
        
        f.save_energy((ekin, epot), timestep=step, block=datablock)


if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    parameters = iom.get_parameters()

    if parameters["algorithm"] == "hagedorn":
        compute_energy(iom)

    else:
        raise ValueError("Unsupported propagator algorithm.")
    
    iom.finalize()
