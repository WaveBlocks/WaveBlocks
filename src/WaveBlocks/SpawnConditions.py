"""The WaveBlocks Project

This file contains some spawning conditions ('oracles')
which can be used to trigger spawning of wavepackets.

All oracles O get called with 3 parameters:

  O(packet, component, environment)

where 'packet' is the packet we currectly focus on
and 'component' is the component we investigate
for spawning. The third parameter 'environment'
is the 'Spawn*Propagator' instance calling the
oracle. This allows access to further data.

To use a certain oracle just write

  spawn_condition = 'oracle_class_name'

in the simulation configuration file.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import collections as co
import numpy as np
import numpy.linalg as la


class SpawnCondition:
    """This class in the base class for all spawn conditions.
    """

    def __init__(self, parameters):
        """Initialize the new spawn condition. Most subclasses
        do not have to change this method. But some may need
        to set up persistent data structures e.g. for keeping
        histories of some values. This should be done here.

        :param parameters: The simulation parameters.
        :type parameters: A :py:class:`ParameterProvider` instance.
        """
        self._parameters = parameters

    def check_condition(self, packet, component, env):
        """The condition to check. Subclasses have to
        implement the condition in this method.

        :param packet: The wavepacket containing the data.
        :type packet: A :py:class:`Wavepacket` instance.
        :param component: The component :math:`\Phi_i` we analyze.
        :type component: An integer.
        :param env: The caller environment.
        :type env: A :py:class:`Propagator` subclass instance.
        """
        pass

################################################################################
# Orcales ######################################################################
################################################################################

class spawn_at_time(SpawnCondition):

    def check_condition(self, packet, component, env):
        return (env.time >= self._parameters["spawn_time"] and env.time < self._parameters["spawn_time"]+self._parameters["dt"])

################################################################################

class norm_threshold(SpawnCondition):

    def check_condition(self, packet, component, env):
        return (packet.get_norm(component=component) >= self._parameters["spawn_threshold"])

################################################################################

class high_k_norm_threshold(SpawnCondition):

    def check_condition(self, packet, component, env):
        c = np.squeeze(packet.get_coefficients(component=component))
        n_high = la.norm(c[self._parameters["spawn_K0"]:])
        return (n_high >= self._parameters["spawn_threshold"])

################################################################################

class high_k_norm_derivative_threshold(SpawnCondition):

    def __init__(self, parameters, env):
        self._parameters = parameters
        # Create a data structure which keeps old values for each wavepacket and component.
        self._spawndata = {}
        for p in env.get_wavepackets():
            env._spawndata[p.get_id()] = [ co.deque(maxlen=parameters["spawn_hist_len"]) for i in xrange(p.get_number_components()) ]


    def check_condition(self, packet, component, env):
        pid = packet.get_id()

        # If there is no data yet we have a new packet
        if not env._spawndata.has_key(pid):
            env._spawndata[pid] = [ co.deque(maxlen=parameters["spawn_hist_len"]) for i in xrange(packet.get_number_components()) ]

        # Get the datastructure
        da = self._spawndata[pid][component]

        # Compute current norm and append to data
        c = np.squeeze(packet.get_coefficients(component=component))
        #no_low = la.norm(c[:self._parameters["spawn_K0"]])
        no_high = la.norm(c[self._parameters["spawn_K0"]:])
        da.append(no_high)

        # Compute L2-norm of derivative
        devno = la.norm(np.diff(np.array(da)))

        return (devno < self._parameters["spawn_deriv_threshold"] and no_high >= self._parameters["spawn_threshold"])

################################################################################

class norm_derivative_threshold_l2(SpawnCondition):

    def __init__(self, parameters, env):
        self._parameters = parameters
        # Create a data structure which keeps old values for each wavepacket and component.
        self._spawndata = {}
        for p in env.get_wavepackets():
            env._spawndata[p.get_id()] = [ co.deque(maxlen=parameters["spawn_hist_len"]) for i in xrange(p.get_number_components()) ]


    def check_condition(self, packet, component, env):

        pid = packet.get_id()

        # If there is no data yet we have a new packet
        if not env._spawndata.has_key(pid):
            ml = self._parameters["spawn_hist_len"]
            env._spawndata[pid] = [ co.deque(maxlen=ml) for i in xrange(packet.get_number_components()) ]

        # Get the datastructure
        da = env._spawndata[pid][component]

        # Compute current norm and append to data
        no = packet.get_norm(component=component)
        da.append(no)

        # Compute L2-norm of derivative
        devno = la.norm(np.diff(np.array(da)))

        return (devno < self._parameters["spawn_deriv_threshold"] and no >= self._parameters["spawn_threshold"])

################################################################################

class norm_derivative_threshold_max(SpawnCondition):

    def __init__(self, parameters, env):
        self._parameters = parameters
        # Create a data structure which keeps old values for each wavepacket and component.
        self._spawndata = {}
        for p in env.get_wavepackets():
            env._spawndata[p.get_id()] = [ co.deque(maxlen=parameters["spawn_hist_len"]) for i in xrange(p.get_number_components()) ]


    def check_condition(self, packet, component, env):
        # The packet id
        pid = packet.get_id()

        # If there is no data yet we have a new packet
        if not env._spawndata.has_key(pid):
            ml = self._parameters["spawn_hist_len"]
            env._spawndata[pid] = [ co.deque(maxlen=ml) for i in xrange(packet.get_number_components()) ]

        # Get the datastructure
        da = env._spawndata[pid][component]

        # Compute current norm and append to data
        no = packet.get_norm(component=component)
        da.append(no)

        # Compute max-norm of derivative
        devno = np.diff(np.array(da)).max()

        return (devno < self._parameters["spawn_deriv_threshold"] and no >= self._parameters["spawn_threshold"])

################################################################################
