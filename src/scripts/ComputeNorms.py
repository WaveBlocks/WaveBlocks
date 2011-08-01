"""The WaveBlocks Project

Calculate the norms of the different wave packets as well as the sum of all norms.

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
        print("Computing the norms in data block "+str(block))

        # See if we have an inhomogeneous wavepacket in the current data block
        if iom.has_inhomogwavepacket(block=block):
            import NormWavepacketInhomog
            NormWavepacketInhomog.compute_norm(iom, block=block)
        # If not, we test for a homogeneous wavepacket next
        elif iom.has_wavepacket(block=block):
            import NormWavepacket
            NormWavepacket.compute_norm(iom, block=block)
        # If we have no wavepacket, then we try for a wavefunction
        elif iom.has_wavefunction(block=block):
            import NormWavefunction
            NormWavefunction.compute_norm(iom, block=block)
        # If there is also no wavefunction, then there is nothing to compute the norm
        else:
            print("Warning: Not computing any norm in block "+str(block)+"!")

    iom.finalize()
