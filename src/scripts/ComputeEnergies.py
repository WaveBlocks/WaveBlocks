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
    for block in xrange(iom.get_number_blocks()):
        print("Computing the energies in data block "+str(block))

        # See if we have an inhomogeneous wavepacket in the current data block
        if iom.has_inhomogwavepacket(block=block):
            import EnergiesWavepacketInhomog
            EnergiesWavepacketInhomog.compute_energy(iom, block=block)
        # If not, we test for a homogeneous wavepacket next
        elif iom.has_wavepacket(block=block):
            import EnergiesWavepacket
            EnergiesWavepacket.compute_energy(iom, block=block)
        # If we have no wavepacket, then we try for a wavefunction
        elif iom.has_wavefunction(block=block):
            import EnergiesWavefunction
            EnergiesWavefunction.compute_energy(iom, block=block)
        # If there is also no wavefunction, then there is nothing to compute the energies
        else:
            print("Warning: Not computing any energies in block "+str(block)+"!")

    iom.finalize()
