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
        iom.load_file(filename=sys.argv[1])
    except IndexError:
        iom.load_file()
        
    parameters = iom.get_parameters()

    if parameters.algorithm == "fourier":
        # Nothing to do for Fourier propagator
        pass

    elif parameters.algorithm == "hagedorn":
        import EvaluateWavepackets
        EvaluateWavepackets.compute_evaluate_wavepackets(iom, basis="canonical")

    elif parameters.algorithm == "multihagedorn":
        import EvaluateWavepacketsInhomog
        EvaluateWavepacketsInhomog.compute_evaluate_wavepackets(iom, basis="canonical")
        
    else:
        raise ValueError("Unknown propagator algorithm.")
    
    iom.finalize()
