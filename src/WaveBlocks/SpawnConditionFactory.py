"""The WaveBlocks Project

This file contains a simple function that selects the desired
spawn condition from a library of ready made conditions.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

import SpawnConditions as SC


class SpawnConditionFactory:

    def __init__(self):
        pass


    def get_condition(self, parameters):
        condition_name = parameters["spawn_condition"]

        if SC.__dict__.has_key(condition_name):
            return SC.__dict__[condition_name]
        else:
            raise ValueError("Unknown spawn condition " + condition_name + " requested from library.")
