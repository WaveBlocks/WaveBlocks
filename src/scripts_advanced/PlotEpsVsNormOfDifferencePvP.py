"""The WaveBlocks Project

Plot the timestep versus the norm for
many different simulation setups. This
scripts compares packet to packet data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import array, diff, log
from scipy.linalg import norm
from matplotlib.pyplot import *

from WaveBlocks import GlobalDefaults
from WaveBlocks.FileTools import *
from WaveBlocks import IOManager
from WaveBlocks import WaveFunction
from WaveBlocks.Plot import legend


def load_data(resultsdir, evaluation_times, which_norm="wf"):
    """This script assumes filename specification: something_eps=..._dt=..._[h|f]_other_things.
    We group the simulations first by eps and then by dt.
    """
    iom_min = IOManager()
    iom_h = IOManager()

    # Group the data from different simulations according to epsilon
    ids = get_result_dirs(resultsdir)
    eps_groups = group_by(ids, "eps")

    # Data structures for results
    epsdata = [ None for i in xrange(len(eps_groups)) ]
    axisdata = [ [] for i in xrange(len(eps_groups)) ]
    normdata = [ [ [] for i in xrange(len(eps_groups)) ] for t in xrange(len(evaluation_times)) ]

    # Loop over all simulations, grouped by same eps value
    for index, eps_group in enumerate(eps_groups):

        # Partition into fourier and hagedorn simulations
        dirs_h = gather_all(eps_group, "algorithm=hagedorn")
        
        # And sort by dt value
        dirs_h = sort_by(dirs_h, "dt")

        # Get simulation with smallest dt
        dir_min = get_min_by(dirs_h, "dt")

        # Loop over all simulations with same eps values sorted by size of dt
        for dir_h in dirs_h:

            print("Comparing simulation " + dir_h + " with " + dir_min)
        
            # Why it works when we put the minimal data here instead outside the loop?
            # Open the minimal data
            resultsfile_min = get_results_file(dir_min)
            iom_min.load_file(filename=resultsfile_min)
        
            resultsfile_h = get_results_file(dir_h)
            iom_h.load_file(filename=resultsfile_h)
            
            # Read the parameters
            parameters_min = iom_min.get_parameters()
            parameters_h = iom_h.get_parameters()
            
            # Scalar parameter of the x axis
            axisdata[index].append(parameters_h.dt)

            # Get the data
            grid = iom_h.load_grid()

            WF = WaveFunction(parameters_h)
            WF.set_grid(grid)

            # Convert times to timesteps using the time manager
            tmmin = parameters_min.get_timemanager()
            tmh = parameters_h.get_timemanager()
            
            # Loop over all times
            for t, time in enumerate(evaluation_times):
                print(" at time T: " + str(time))
                
                stepmin = tmmin.compute_timestep(time)
                steph = tmh.compute_timestep(time)
                
                data_min = iom_min.load_wavefunction(timestep=stepmin)
                data_h = iom_h.load_wavefunction(timestep=steph)

                # Compute the norm  || u_f - u_h || for all timesteps
                data_diff = data_min - data_h

                # Adapt for multiple levels here if desired

                if which_norm == "wf":
                    WF.set_values( [ data_diff[0,...] ] )
                    no = WF.get_norm(summed=True)
                elif which_norm == "2":
                    no = norm( data_diff[0,...] )
                elif which_norm == "max":
                    no = max( data_diff[0,...] )
            
                # Append norm values to global data structure
                normdata[t][index].append(no)

        # Scalar parameter of the different curves
        # We add this here because the simulation parameters are
        # already loaded but not overwritten yet be the next iteration
        # Remember: we need only a single epsilon out of each eps_group.
        epsdata[index] = parameters_h.eps
    
    iom_h.finalize()
    iom_min.finalize()
    
    # Convert lists to arrays
    epsdata = array(epsdata)
    axisdata = [ array(item) for item in axisdata ]

    return (times, epsdata, axisdata, normdata)
 

def plot_data(times, epsdata, axisdata, normdata, which_norm="wf"):

    if which_norm == "wf":
        nona = "wf"
    elif which_norm == "2":
        nona = "L^2"
    elif which_norm == "max":
        nona = "max"

    def guessor(x, y):
        u = log(diff(x))
        v = log(diff(y))
        return v / u

    for t, time in enumerate(times):
        # Plot the convergence for all epsilon and fixed times
        fig = figure()
        ax = fig.gca()
        
        for eps, ad, nd in  zip(epsdata, axisdata, normdata[t]):
            ax.loglog(ad, nd, "-o", label=r"$\varepsilon = "+str(eps)+"$")

        # Plot a convergence indicator
        ax.loglog(axisdata[0], axisdata[0]**2, "-k", label=r"$y = x^2$")

        ax.set_xlabel(r"Timestep size $dt$")
        ax.set_ylabel(r"$$\| \phi_h - \phi_h \|_{"+nona+r"}$$")
        ax.set_title(r"Error norm $\| \phi_h - \phi_h \|_{"+nona+r"}$ for time $T=" + str(time) + r"$")
        legend(loc="outer right")
        fig.savefig("convergence_PvP_time="+str(time)+"_"+nona+".png")
        close(fig)


        fig = figure()
        ax = fig.gca()

        for eps, ad, nd in  zip(epsdata, axisdata, normdata[t]):
            values = guessor(ad, nd)
            ax.plot(values, "-o", label=r"$\varepsilon = "+str(eps)+"$")
        
        ax.set_title(r"guessor at time $T=" + str(time) + r"$")
        legend(loc="outer right")
        fig.savefig("guessor_PvP_time="+str(time)+"_"+nona+".png")
        close(fig)


if __name__ == "__main__":
    # Read file with simulation data
    try:
        path_to_results = sys.argv[1]
    except IndexError:
        path_to_results = GlobalDefaults.path_to_results

    # Times for the pointwise comparisons
    times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    data = load_data(path_to_results, times, which_norm="wf")
    plot_data(*data, which_norm="wf")

    data = load_data(path_to_results, times, which_norm="2")
    plot_data(*data, which_norm="2")

    data = load_data(path_to_results, times, which_norm="max")
    plot_data(*data, which_norm="max")
