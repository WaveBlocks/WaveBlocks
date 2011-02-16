# Default configuration of which scripts are run in the
# batch loop. Change the content of the lists as you like
# but never rename the variables.

# All scripts in this list are called for each simulation
# configuration and with the configuration file as first
# command line argument
call_simulation = ["Main.py"]

# All scripts in this list are called for each simulation
# configuration but without additional arguments. They can
# assume that the simulation results data file is available
# at the default location ('simulation_results.hdf5').
call_for_each = ["ComputeNorms.py",
                 "ComputeEnergies.py",
                 #"PlotPotential.py",
                 "PlotNorms.py",
                 "PlotEnergies.py",
                 #"PlotWavepacketParameters.py",
                 #"PlotWavepacketCoefficients.py",
                 #"EvaluateWavepacketsEigen.py",
                 #"PlotWavefunction.py",
                 #"PlotWavepackets.py",
                 ]

# The scripts in this list are called once after all
# simulations are finished and the results were moved
# to the final location (default './results/*'). Put
# all scripts that do comparisons between different
# simulations in here.
call_once = [
             ]
