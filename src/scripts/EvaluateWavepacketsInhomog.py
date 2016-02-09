"""The WaveBlocks Project

Sample wavepackets at the nodes of a given grid and save the results back
to the given simulation data file.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from WaveBlocks import PotentialFactory
from WaveBlocks import WaveFunction
from WaveBlocks import HagedornWavepacketInhomogeneous


def compute_evaluate_wavepackets(iom, basis="eigen", blockid=0):
    """Evaluate an in homogeneous Hagdorn wavepacket on a given grid for each timestep.
    :param iom: An I{IOManager} instance providing the simulation data.
    @keyword basis: The basis where the evaluation is done. Can be 'eigen' or 'canonical'.
    @keyword blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()

    # Number of time steps we saved
    timesteps = iom.load_inhomogwavepacket_timegrid(blockid=blockid)
    nrtimesteps = timesteps.shape[0]

    # Prepare the potential for basis transformations
    Potential = PotentialFactory().create_potential(parameters)

    # Retrieve simulation data
    if iom.has_grid(blockid=blockid):
        grid = iom.load_grid(blockid=blockid)
    else:
        grid = iom.load_grid(blockid="global")

    params = iom.load_inhomogwavepacket_parameters(blockid=blockid)
    coeffs = iom.load_inhomogwavepacket_coefficients(blockid=blockid)

    # A data transformation needed by API specification
    params = [ [ params[j][i,:] for j in xrange(parameters["ncomponents"]) ] for i in xrange(nrtimesteps) ]
    coeffs = [ [ coeffs[i,j,:] for j in xrange(parameters["ncomponents"]) ] for i in xrange(nrtimesteps) ]

    # We want to save wavefunctions, thus add a data slot to the data file
    iom.add_wavefunction(parameters, timeslots=nrtimesteps, blockid=blockid)

    # Hack for allowing data blocks with different basis size than the global one
    # todo: remove when we got local parameter sets
    parameters.update_parameters({"basis_size": coeffs[0][0].shape[0]})

    HAWP = HagedornWavepacketInhomogeneous(parameters)
    HAWP.set_quadrature(None)

    WF = WaveFunction(parameters)
    WF.set_grid(grid)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Evaluating inhomogeneous wavepacket at timestep "+str(step))

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
        iom.save_wavefunction(WF.get_values(), timestep=step, blockid=blockid)
