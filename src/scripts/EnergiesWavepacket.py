"""The WaveBlocks Project

Compute the kinetic and potential energies of the homogeneous wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from WaveBlocks import PotentialFactory
from WaveBlocks import HagedornWavepacket


def compute_energy(iom, blockid=0):
    """
    @param iom: An I{IOManager} instance providing the simulation data.
    @keyword blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()

    # Number of time steps we saved
    timesteps = iom.load_wavepacket_timegrid(blockid=blockid)
    nrtimesteps = timesteps.shape[0]

    Potential = PotentialFactory.create_potential(parameters)

    # Retrieve simulation data
    params = iom.load_wavepacket_parameters(blockid=blockid)
    coeffs = iom.load_wavepacket_coefficients(blockid=blockid)

    # A data transformation needed by API specification
    coeffs = [ [ coeffs[i,j,:] for j in xrange(parameters["ncomponents"]) ] for i in xrange(nrtimesteps) ]

    # We want to save energies, thus add a data slot to the data file
    iom.add_energy(parameters, timeslots=nrtimesteps, blockid=blockid)

    # Hack for allowing data blocks with different basis size than the global one
    # todo: remove when we got local parameter sets
    parameters.update_parameters({"basis_size": coeffs[0][0].shape[0]})

    # Initialize a Hagedorn wave packet with the data
    HAWP = HagedornWavepacket(parameters)
    HAWP.set_quadrature(None)

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

        iom.save_energy((ekin, epot), timestep=step, blockid=blockid)
