.. WaveBlocks documentation master file, created by
   sphinx-quickstart on Wed Jan  4 17:32:14 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to WaveBlocks's documentation!
======================================

.. image:: _static/waveblocks.png

Reusable building blocks for simulations with semiclassical wavepackets for
solving the time-dependent Schrödinger equation.


Source code documentation
=========================

WaveBlocks Classes
-------------------

Basic numerics
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   waveblocks_classes/ComplexMath

   waveblocks_classes/MatrixExponential
   waveblocks_classes/MatrixExponentialFactory

   waveblocks_classes/QuadratureRule
   waveblocks_classes/GaussHermiteQR
   waveblocks_classes/TrapezoidalQR

Basic quantum mechanics
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   waveblocks_classes/WaveFunction

   waveblocks_classes/MatrixPotential
   waveblocks_classes/MatrixPotential1S
   waveblocks_classes/MatrixPotential2S
   waveblocks_classes/MatrixPotentialMS

   waveblocks_classes/PotentialLibrary
   waveblocks_classes/PotentialFactory

Wavepackets and related
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   waveblocks_classes/Wavepacket
   waveblocks_classes/HagedornWavepacket
   waveblocks_classes/HagedornWavepacketInhomogeneous

   waveblocks_classes/Quadrature
   waveblocks_classes/HomogeneousQuadrature
   waveblocks_classes/InhomogeneousQuadrature

Time propagation
^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   waveblocks_classes/Propagator
   waveblocks_classes/FourierPropagator
   waveblocks_classes/HagedornPropagator
   waveblocks_classes/HagedornPropagatorInhomogeneous

Toplevel simulation loops
^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   waveblocks_classes/SimulationLoop
   waveblocks_classes/SimulationLoopFourier
   waveblocks_classes/SimulationLoopHagedorn
   waveblocks_classes/SimulationLoopHagedornInhomogeneous

   waveblocks_classes/TimeManager

Simulation result storage I/O
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   waveblocks_classes/IOManager

   waveblocks_classes/IOM_plugin_energy
   waveblocks_classes/IOM_plugin_grid
   waveblocks_classes/IOM_plugin_fourieroperators
   waveblocks_classes/IOM_plugin_inhomogwavepacket
   waveblocks_classes/IOM_plugin_norm
   waveblocks_classes/IOM_plugin_parameters
   waveblocks_classes/IOM_plugin_wavefunction
   waveblocks_classes/IOM_plugin_wavepacket

   waveblocks_classes/FileTools

Related to ``Spawning``
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   waveblocks_classes/Spawner
   waveblocks_classes/AdiabaticSpawner
   waveblocks_classes/NonAdiabaticSpawner

   waveblocks_classes/SpawnConditions
   waveblocks_classes/SpawnConditionFactory

   waveblocks_classes/SpawnAdiabaticPropagator
   waveblocks_classes/SpawnNonAdiabaticPropagator

   waveblocks_classes/SimulationLoopSpawnAdiabatic
   waveblocks_classes/SimulationLoopSpawnNonAdiabatic

Other classes
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   waveblocks_classes/GlobalDefaults
   waveblocks_classes/ParameterLoader
   waveblocks_classes/ParameterProvider
   waveblocks_classes/Utils

Etc
===

.. toctree::
   :maxdepth: 2

   citation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
