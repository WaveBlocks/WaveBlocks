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
        
    parameters = iom.get_parameters()

    if parameters["algorithm"] == "fourier":
        # Nothing to do for Fourier propagator
        pass

    elif parameters["algorithm"] == "hagedorn":
        import EvaluateWavepackets
        EvaluateWavepackets.compute_evaluate_wavepackets(iom, basis="canonical")

    elif parameters["algorithm"] == "multihagedorn":
        import EvaluateWavepacketsInhomog
        EvaluateWavepacketsInhomog.compute_evaluate_wavepackets(iom, basis="canonical")

    elif (parameters["algorithm"] == "spawning_apost" or
          parameters["algorithm"] == "spawning_adiabatic"):
        import EvaluateWavepackets
        EvaluateWavepackets.compute_evaluate_wavepackets(iom, basis="canonical")
        EvaluateWavepackets.compute_evaluate_wavepackets(iom, basis="canonical", datablock=1)
        
    else:
        raise ValueError("Unknown propagator algorithm.")
    
    iom.finalize()
