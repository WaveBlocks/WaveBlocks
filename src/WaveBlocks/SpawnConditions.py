"""The WaveBlocks Project

This file contains some spawning conditions
which can be used to trigger spawning of wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np
import scipy.linalg as spla


def adiabatic_K_threshold(parameters, time, packet, component):
    c = np.squeeze(packet.get_coefficients(component=component))
    #n_low = spla.norm(c[:parameters["spawn_K0"]])
    n_high = spla.norm(c[parameters["spawn_K0"]:])
    return (n_high >= parameters["spawn_threshold"])


def nonadiabatic_component_timestep(parameters, time, packet, component):
    return (time >= parameters["spawn_time"] and time < parameters["spawn_time"]+parameters["dt"])


def nonadiabatic_component_threshold(parameters, time, packet, component):
    return (packet.get_norm(component=component) >= parameters["spawn_threshold"])
