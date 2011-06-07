"""The WaveBlocks Project

Sample wavepackets at the nodes of a given grid and save the results back
to the given simulation data file.

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
        print("Evaluating wavepackets in data block "+str(block))
        # See if we have an inhomogeneous wavepacket in the current data block
        if iom.has_inhomogwavepacket(block=block):
            import EvaluateWavepacketsInhomog
            EvaluateWavepacketsInhomog.compute_evaluate_wavepackets(iom, block=block, basis="canonical")
        # If not, we test for a homogeneous wavepacket next
        elif iom.has_wavepacket(block=block):
            import EvaluateWavepackets
            EvaluateWavepackets.compute_evaluate_wavepackets(iom, block=block, basis="canonical")
        # If there is also no wavefunction, then there is nothing to compute the norm
        else:
            print("Warning: Not evaluating any wavepackets in block "+str(block)+"!")

    iom.finalize()
