"""The WaveBlocks Project

Compute the kinetic and potential energies of the inhomogeneous wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from WaveBlocks import PotentialFactory
from WaveBlocks import HagedornMultiWavepacket


def compute_energy(iom, block=0):
    """
    @param iom: An I{IOManager} instance providing the simulation data.
    @keyword block: The data block from which the values are read.
    """
    parameters = iom.get_parameters()

    # Number of time steps we saved
    timesteps = iom.load_inhomogwavepacket_timegrid(block=block)
    nrtimesteps = timesteps.shape[0]

    Potential = PotentialFactory.create_potential(parameters)

    # Retrieve simulation data
    params = iom.load_inhomogwavepacket_parameters(block=block)
    coeffs = iom.load_inhomogwavepacket_coefficients(block=block)

    # A data transformation needed by API specification
    params = [ [ params[j][i,:] for j in xrange(parameters["ncomponents"]) ] for i in xrange(nrtimesteps) ]
    coeffs = [ [ coeffs[i,j,:] for j in xrange(parameters["ncomponents"]) ] for i in xrange(nrtimesteps) ]

    # We want to save energies, thus add a data slot to the data file
    iom.add_energy(parameters, timeslots=nrtimesteps, block=block)

    # Hack for allowing data blocks with different basis size than the global one
    # todo: remove when we got local parameter sets
    parameters.update_parameters({"basis_size": coeffs[0][0].shape[0]})

    # Initialize a hagedorn wave packet with the data
    HAWP = HagedornMultiWavepacket(parameters)
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

        iom.save_energy((ekin, epot), timestep=step, block=block)
