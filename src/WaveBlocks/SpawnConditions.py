"""The WaveBlocks Project

This file contains some spawning conditions
which can be used to trigger spawning of wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import numpy as np
import scipy.linalg as spla


def adiabatic_K_threshold(packet, component, env):
    c = np.squeeze(packet.get_coefficients(component=component))
    #n_low = spla.norm(c[:parameters["spawn_K0"]])
    n_high = spla.norm(c[env.parameters["spawn_K0"]:])
    return (n_high >= env.parameters["spawn_threshold"])


def nonadiabatic_component_timestep(packet, component, env):
    return (env.time >= env.parameters["spawn_time"] and env.time < env.parameters["spawn_time"]+env.parameters["dt"])


def nonadiabatic_component_threshold(packet, component, env):
    return (packet.get_norm(component=component) >= env.parameters["spawn_threshold"])
