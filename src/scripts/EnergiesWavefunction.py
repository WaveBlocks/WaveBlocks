"""The WaveBlocks Project

Compute the kinetic and potential energies of a wave function.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros

from WaveBlocks import PotentialFactory
from WaveBlocks import WaveFunction
from WaveBlocks import IOManager


def compute_energies(f, datablock=0):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    @keyword datablock: The data block where the results are.    
    """
    p = f.get_parameters()

    # Number of time steps we saved
    timesteps = f.load_wavefunction_timegrid(block=datablock)
    nrtimesteps = timesteps.shape[0]

    # We want to save norms, thus add a data slot to the data file
    f.add_energies(p, timeslots=nrtimesteps, block=datablock)

    nodes = f.load_grid(block=datablock)
    opT, opV = f.load_operators(block=datablock)

    # Precalculate eigenvectors for efficiency
    Potential = PotentialFactory.create_potential(p)
    eigenvectors = Potential.evaluate_eigenvectors_at(nodes)
    nst = Potential.get_number_components()

    WF = WaveFunction(p)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Computing energies of timestep # " + str(step))
        
        values = f.load_wavefunction(timestep=step, block=datablock)
        values = [ values[j,...] for j in xrange(p.ncomponents) ]
        
        # Project wavefunction values to eigenbase
        values = Potential.project_to_eigen(nodes, values, eigenvectors)
        WF.set_values(values)

        ekinlist = []
        epotlist = []

        # For each component of |Psi>
        values = WF.get_values()
        for index, item in enumerate(values):
            # tmp is the Vector (0, 0, 0, \psi_i, 0, 0, ...)
            tmp = [ zeros(item.shape) for z in xrange(nst) ]
            tmp[index] = item
            
            # Project this vector to the canonical base
            tmp = Potential.project_to_canonical(nodes, tmp, eigenvectors)
            WF.set_values(tmp)
            
            # And calculate the energies of these components
            ekinlist.append(WF.kinetic_energy(opT, summed=True))
            epotlist.append(WF.potential_energy(opV, summed=True))
            
        f.save_energies((ekinlist, epotlist), timestep=step, block=datablock)
