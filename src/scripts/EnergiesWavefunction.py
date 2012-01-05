"""The WaveBlocks Project

Compute the kinetic and potential energies of a wave function.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros

from WaveBlocks import PotentialFactory
from WaveBlocks import WaveFunction


def compute_energy(iom, blockid=0):
    """
    @param iom: An I{IOManager} instance providing the simulation data.
    @keyword blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()

    # Number of time steps we saved
    timesteps = iom.load_wavefunction_timegrid(blockid=blockid)
    nrtimesteps = timesteps.shape[0]

    # Retrieve simulation data
    if iom.has_grid(blockid=blockid):
        grid = iom.load_grid(blockid=blockid)
    else:
        grid = iom.load_grid(blockid="global")

    opT, opV = iom.load_fourieroperators(blockid=blockid)

    # We want to save norms, thus add a data slot to the data file
    iom.add_energy(parameters, timeslots=nrtimesteps, blockid=blockid)

    # Precalculate eigenvectors for efficiency
    Potential = PotentialFactory().create_potential(parameters)
    eigenvectors = Potential.evaluate_eigenvectors_at(grid)
    nst = Potential.get_number_components()

    WF = WaveFunction(parameters)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Computing energies of timestep # " + str(step))

        values = iom.load_wavefunction(timestep=step, blockid=blockid)
        values = [ values[j,...] for j in xrange(parameters["ncomponents"]) ]

        # Project wavefunction values to eigenbasis
        values = Potential.project_to_eigen(grid, values, eigenvectors)
        WF.set_values(values)

        ekinlist = []
        epotlist = []

        # For each component of |Psi>
        values = WF.get_values()
        for index, item in enumerate(values):
            # tmp is the Vector (0, 0, 0, \psi_i, 0, 0, ...)
            tmp = [ zeros(item.shape) for z in xrange(nst) ]
            tmp[index] = item

            # Project this vector to the canonical basis
            tmp = Potential.project_to_canonical(grid, tmp, eigenvectors)
            WF.set_values(tmp)

            # And calculate the energies of these components
            ekinlist.append(WF.kinetic_energy(opT, summed=True))
            epotlist.append(WF.potential_energy(opV, summed=True))

        iom.save_energy((ekinlist, epotlist), timestep=step, blockid=blockid)
