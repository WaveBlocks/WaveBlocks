"""The WaveBlocks Project

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from QuadratureRule import QuadratureRule
from GaussHermiteQR import GaussHermiteQR

from HagedornWavepacket import HagedornWavepacket
from HagedornMultiWavepacket import HagedornMultiWavepacket

from WaveFunction import WaveFunction

from ParameterProvider import ParameterProvider
from TimeManager import TimeManager
from IOManager import IOManager

from PotentialFactory import PotentialFactory
from MatrixPotential import MatrixPotential
from MatrixPotential1S import MatrixPotential1S
from MatrixPotential2S import MatrixPotential2S
from MatrixPotentialMS import MatrixPotentialMS

from Propagator import Propagator
from FourierPropagator import FourierPropagator
from HagedornPropagator import HagedornPropagator
from HagedornMultiPropagator import HagedornMultiPropagator

from SimulationLoop import SimulationLoop
from SimulationLoopFourier import SimulationLoopFourier
from SimulationLoopHagedorn import SimulationLoopHagedorn
from SimulationLoopMultiHagedorn import SimulationLoopMultiHagedorn

from AdiabaticSpawner import AdiabaticSpawner

# Just functions inside this modules.
#from GlobalDefaults import GlobalDefaults
#from FileTools import FileTools

# Enable dynamic plugin loading for IOManager
import sys
import os

plugin_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(plugin_dir)
