"""The WaveBlocks Project

Compute and plot the norm of the difference
of the wavefunctions resulting from different
simulation algorithms.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import array
from scipy.linalg import norm
from matplotlib.pyplot import *

from WaveBlocks import FileTools as FT
from WaveBlocks import PotentialFactory
from WaveBlocks import IOManager
from WaveBlocks import WaveFunction
from WaveBlocks import GlobalDefaults
from WaveBlocks.Plot import legend

import GraphicsDefaults as GD


def load_data(resultspath, which_norm="wf"):
    # Group the data from different simulations
    ids = FT.get_result_dirs(resultspath)
    ids = FT.group_by(ids, "eps")

    nsims = FT.get_number_simulations(resultspath)

    groupdata = []
    axisdata = [ [] for i in xrange(nsims) ]
    normdata = [ [] for i in xrange(nsims) ]

    iom_f = IOManager()
    iom_h = IOManager()

    for index, sims in enumerate(ids):
        # Sorting based on file names
        dirs_f = FT.gather_all(sims, "fourier")
        dirs_h = FT.gather_all(sims, "hagedorn")

        if len(dirs_f) != len(dirs_h):
            raise ValueError("Found different number of fourier and hagedorn simulations!")

        dirs_f = FT.sort_by(dirs_f, "eps", as_string=True)
        dirs_h = FT.sort_by(dirs_h, "eps", as_string=True)

        # Loop over all simulations
        for dir_f, dir_h in zip(dirs_f, dirs_h):

            print("Comparing simulation " + dir_h + " with " + dir_f)

            resultsfile_f = FT.get_results_file(dir_f)
            iom_f.open_file(filename=resultsfile_f)

            resultsfile_h = FT.get_results_file(dir_h)
            iom_h.open_file(filename=resultsfile_h)

            # Read the parameters
            parameters_f = iom_f.load_parameters()
            parameters_h = iom_h.load_parameters()

            grid = iom_f.load_grid(blockid="global")

            # Precalculate eigenvectors for efficiency
            Potential = PotentialFactory().create_potential(parameters_f)
            eigenvectors = Potential.evaluate_eigenvectors_at(grid)

            # Get the data
            # Number of time steps we saved
            timesteps = iom_f.load_wavefunction_timegrid()

            # Scalar parameter that discriminates the simulations
            axisdata[index].append((parameters_f, timesteps))

            WF = WaveFunction(parameters_f)
            WF.set_grid(grid)

            norms = []

            for i, step in enumerate(timesteps):
                # Load the data that belong to the current timestep
                data_f = iom_f.load_wavefunction(timestep=step)
                data_h = iom_h.load_wavefunction(timestep=step)

                data_f = Potential.project_to_eigen(grid, data_f, eigenvectors)
                data_f = array(data_f)

                data_diff = data_f - data_h

                # Compute the norm  || u_f - u_h ||
                if which_norm == "wf":
                    # Rearrange data to fit the input of WF and handle over
                    WF.set_values([ data_diff[n,:] for n in xrange(parameters_f.ncomponents) ])
                    curnorm = WF.get_norm()

                    # More than one component? If yes, compute also the overall norm
                    if parameters_f.ncomponents > 1:
                        nosum = WF.get_norm(summed=True)
                        curnorm = list(curnorm) + [nosum]

                elif which_norm == "max":
                    curnorm = [ max( abs(data_diff[n,:]) ) for n in xrange(parameters_f.ncomponents) ]

                    # More than one component? If yes, compute also the overall norm
                    if parameters_f.ncomponents > 1:
                        nosum = max(curnorm)
                        curnorm = list(curnorm) + [nosum]

                print(" at time " + str(step*parameters_f.dt) + " the error norm is " + str(curnorm))
                norms.append(curnorm)

            # Append norm values to global data structure
            norms = array(norms)
            normdata[index].append(norms)

        # Scalar parameter of the different curves
        # We add this here because the simulation parameters are
        # already loaded but not overwritten yet be the next iteration
        # Remember: we need only a single epsilon out of each eps_group.
        groupdata.append(parameters_f.dt)

    iom_f.finalize()
    iom_h.finalize()

    return (groupdata, axisdata, normdata)


def plot_data(groupdata, axisdata, normdata, which_norm="L2"):

    nona = "L^2" if which_norm == "L2" else "\infty"

    for groupd, axisd, normd in zip(groupdata, axisdata, normdata):
        # Plot the error time series for each simulation
        fig = figure()
        ax = fig.gca()

        for cur_axisd, cur_norm in zip(axisd, normd):

            # One single component in |\Psi>
            if cur_norm.shape[1] == 1:
                ax.plot(cur_axisd[1]*cur_axisd[0].dt, cur_norm, label=r"$\varepsilon = $" + str(cur_axisd[0].eps))

            # More than one component in |\Psi>
            else:
                # Plot all the error norms for all components individually
                for i in xrange(cur_norm.shape[1]-1):
                    ax.semilogy(cur_axisd[1]*cur_axisd[0].dt, cur_norm[:,i], label=r"$\varepsilon = $" + str(cur_axisd[0].eps) + r", $c_"+str(i)+r"$")

                # Plot the overall summed error norm
                ax.semilogy(cur_axisd[1]*cur_axisd[0].dt, cur_norm[:,-1], label=r"$\varepsilon = $" + str(cur_axisd[0].eps) +r", $\sum_i c_i$")

        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\| \phi^F - \phi^H \|_{"+nona+r"}$")
        ax.set_title(r"Timeseries of $\| \phi_f - \phi_h \|_{"+nona+r"}$ ")
        legend()
        fig.savefig("error_timevolution_"+which_norm+"_all"+GD.output_format)
        close(fig)


if __name__ == "__main__":
    # Read simulation data path
    try:
        path_to_results = sys.argv[1]
    except IndexError:
        path_to_results = GlobalDefaults.path_to_results

    data = load_data(path_to_results)
    plot_data(*data)

    data = load_data(path_to_results, which_norm="max")
    plot_data(*data, which_norm="max")
