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
    for blockid in iom.get_block_ids():
        print("Evaluating wavepackets in data block '"+str(blockid)+"'")

        if iom.has_wavefunction(blockid=blockid):
            print("Datablock '"+str(blockid)+"' already contains wavefunction data, silent skip.")
            continue

        # See if we have an inhomogeneous wavepacket in the current data block
        if iom.has_inhomogwavepacket(blockid=blockid):
            import EvaluateWavepacketsInhomog
            EvaluateWavepacketsInhomog.compute_evaluate_wavepackets(iom, blockid=blockid)
        # If not, we test for a homogeneous wavepacket next
        elif iom.has_wavepacket(blockid=blockid):
            import EvaluateWavepackets
            EvaluateWavepackets.compute_evaluate_wavepackets(iom, blockid=blockid)
        # If there is also no wavefunction, then there is nothing to compute the norm
        else:
            print("Warning: Not evaluating any wavepackets in block '"+str(blockid)+"'!")

    iom.finalize()
