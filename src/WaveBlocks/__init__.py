"""The WaveBlocks Project

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from QuadratureRule import QuadratureRule
from GaussHermiteQR import GaussHermiteQR
from TrapezoidalQR import TrapezoidalQR

from Quadrature import Quadrature
from HomogeneousQuadrature import HomogeneousQuadrature
from InhomogeneousQuadrature import InhomogeneousQuadrature

from HagedornWavepacket import HagedornWavepacket
from HagedornWavepacketInhomogeneous import HagedornWavepacketInhomogeneous

from WaveFunction import WaveFunction

from ParameterProvider import ParameterProvider
from ParameterLoader import ParameterLoader
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
from HagedornPropagatorInhomogeneous import HagedornPropagatorInhomogeneous

from SimulationLoop import SimulationLoop
from SimulationLoopFourier import SimulationLoopFourier
from SimulationLoopHagedorn import SimulationLoopHagedorn
from SimulationLoopHagedornInhomogeneous import SimulationLoopHagedornInhomogeneous

from Spawner import Spawner
from AdiabaticSpawner import AdiabaticSpawner
from NonAdiabaticSpawner import NonAdiabaticSpawner
from SpawnAdiabaticPropagator import SpawnAdiabaticPropagator
from SpawnNonAdiabaticPropagator import SpawnNonAdiabaticPropagator
from SimulationLoopSpawnAdiabatic import SimulationLoopSpawnAdiabatic
from SimulationLoopSpawnNonAdiabatic import SimulationLoopSpawnNonAdiabatic

# Just functions inside these modules.
#from GlobalDefaults import GlobalDefaults
#from FileTools import FileTools
#from SpawnConditionFactory import SpawnConditionFactory

# Enable dynamic plugin loading for IOManager
import sys
import os

plugin_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(plugin_dir)
