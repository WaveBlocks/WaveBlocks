"""The WaveBlocks Project

This file contains a the spawning interface common
to all adiabatic and non-adiabatic spawners.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

class Spawner:
    def __init__(self):
        pass


    def estimate_parameters(self, packet, mother_component):
        r"""
        Compute the parameters for a new wavepacket.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'Spawner' is an abstract base class.")


    def project_coefficients(self, mother, child):
        r"""
        Update the superposition coefficients of mother and spawned wavepacket.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'Spawner' is an abstract base class.")
