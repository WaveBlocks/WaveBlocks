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


def compute_energy(iom, blockid=0):
    p = iom.load_parameters()

    # Number of time steps we saved
    timesteps = iom.load_wavepacket_timegrid(blockid=blockid)
    nrtimesteps = timesteps.shape[0]

    # We want to save energies, thus add a data slot to the data file
    iom.add_energy(p, timeslots=nrtimesteps, blockid=blockid, total=True)

    Potential = PotentialFactory.create_potential(p)

    params = iom.load_wavepacket_parameters(blockid=blockid)
    coeffs = iom.load_wavepacket_coefficients(blockid=blockid)

    # A data transformation needed by API specification
    coeffs = [ [ coeffs[i,j,:] for j in xrange(p.ncomponents) ] for i in xrange(nrtimesteps) ]

    # Initialize a hagedorn wave packet with the data
    HAWP = HagedornWavepacket(p)
    HAWP.set_quadrature(None)

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

        iom.save_energy_total(etot, timestep=step, blockid=blockid)

        # Transform to eigenbasis
        HAWP.project_to_eigen(Potential)

        # Compute the components' energies in the eigenbasis
        ekin = HAWP.kinetic_energy()
        epot = HAWP.potential_energy(Potential.evaluate_eigenvalues_at)

        iom.save_energy((ekin, epot), timestep=step, blockid=blockid)


if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    # Iterate over all blocks
    for blockid in iom.get_block_ids():
        print("Computing the energies in data block '"+str(blockid)+"'")

        if iom.has_energy(blockid=blockid):
            print("Datablock '"+str(blockid)+"' already contains energy data, silent skip.")
            continue

        # See if we have an inhomogeneous wavepacket in the current data block
        if iom.has_inhomogwavepacket(blockid=blockid):
            import EnergiesWavepacketInhomog
            EnergiesWavepacketInhomog.compute_energy(iom, blockid=blockid)
        # If not, we test for a homogeneous wavepacket next
        elif iom.has_wavepacket(blockid=blockid):
            import EnergiesWavepacket
            EnergiesWavepacket.compute_energy(iom, blockid=blockid)
        # If we have no wavepacket, then we try for a wavefunction
        elif iom.has_wavefunction(blockid=blockid):
            import EnergiesWavefunction
            EnergiesWavefunction.compute_energy(iom, blockid=blockid)
        # If there is also no wavefunction, then there is nothing to compute the energies
        else:
            print("Warning: Not computing any energies in block '"+str(blockid)+"'!")

    iom.finalize()
