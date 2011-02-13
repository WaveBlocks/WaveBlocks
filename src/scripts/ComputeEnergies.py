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
        iom.load_file(filename=sys.argv[1])
    except IndexError:
        iom.load_file()

    parameters = iom.get_parameters()

    if parameters.algorithm == "fourier":
        import EnergiesWavefunction
        EnergiesWavefunction.compute_energies(iom)

    elif parameters.algorithm == "hagedorn":
        import EnergiesWavepacket
        EnergiesWavepacket.compute_energies(iom)

    elif parameters.algorithm == "multihagedorn":
        import EnergiesWavepacketInhomog
        EnergiesWavepacketInhomog.compute_energies(iom)

    else:
        raise ValueError("Unknown propagator algorithm.")
    
    iom.finalize()
