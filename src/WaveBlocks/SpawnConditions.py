"""The WaveBlocks Project

This file contains some spawning conditions ('oracles')
which can be used to trigger spawning of wavepackets.

All oracles O get called with 3 parameters:

  O(packet, component, environment)

where 'packet' is the packet we currectly focus on
and 'component' is the component we investigate
for spawning. The third parameter 'environment'
is the 'Spawn*Propagator' instance calling the oracle.
This allows access to further data, f.e. the current
parameter provider.

For each oracle there may be a setup method which
is allowed to build (arbitrary) data structures in
the 'environment' instance. The function gets called
once when preparing a 'Spawn*Propagator' with this
instance as the only argument.

The name must be the same as the oracle but with
an appended '_setup' string.

To use a certain oracle just write

  spawn_condition = 'oracle_function_name'

in the simulation configuration file.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import collections as co
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


def derivative_threshold_setup(env):
    # Create a data structure which keeps old values for each wavepacket and component.
    env._spawndata = {}

    ml = env.parameters["spawn_hist_len"]

    for p in env.get_wavepackets():
        env._spawndata[p.get_id()] = [ co.deque(maxlen=ml) for i in xrange(p.get_number_components()) ]

def derivative_threshold_l2_setup(env):
    derivative_threshold_setup(env)

def derivative_threshold_max_setup(env):
    derivative_threshold_setup(env)


def derivative_threshold_l2(packet, component, env):
    pid = packet.get_id()

    # If there is no data yet we have a new packet
    if not env._spawndata.has_key(pid):
        ml = env.parameters["spawn_hist_len"]
        env._spawndata[pid] = [ co.deque(maxlen=ml) for i in xrange(packet.get_number_components()) ]

    # Get the datastructure
    da = env._spawndata[pid][component]

    # Compute current norm and append to data
    no = packet.get_norm(component=component)
    da.append(no)

    # Compute L2-norm of derivative
    devno = spla.norm(np.diff(np.array(da)))

    return (devno < env.parameters["spawn_deriv_threshold"] and no >= env.parameters["spawn_threshold"])


def derivative_threshold_max(packet, component, env):
    # The packet id
    pid = packet.get_id()

    # If there is no data yet we have a new packet
    if not env._spawndata.has_key(pid):
        ml = env.parameters["spawn_hist_len"]
        env._spawndata[pid] = [ co.deque(maxlen=ml) for i in xrange(packet.get_number_components()) ]

    # Get the datastructure
    da = env._spawndata[pid][component]

    # Compute current norm and append to data
    no = packet.get_norm(component=component)
    da.append(no)

    # Compute max-norm of derivative
    devno = np.diff(np.array(da)).max()

    return (devno < env.parameters["spawn_deriv_threshold"] and no >= env.parameters["spawn_threshold"])
