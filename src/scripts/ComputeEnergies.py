"""The WaveBlocks Project

Calculate the energies of the different wavepackets or wavefunctions.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys

from WaveBlocks import IOManager


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
