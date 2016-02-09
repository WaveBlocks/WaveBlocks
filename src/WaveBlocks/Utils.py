"""The WaveBlocks Project

Various small utility functions.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy.lib.arraysetops import in1d

#TODO: Consider merging this into the TimeManager
def common_timesteps(timegridA, timegridB):
    r"""
    Find the indices (wrt to A and B) of the timesteps common to both timegrids.
    """
    IA = in1d(timegridA, timegridB)
    IB = in1d(timegridB, timegridA)

    return (IA, IB)
