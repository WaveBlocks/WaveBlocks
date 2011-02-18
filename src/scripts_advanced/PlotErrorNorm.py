"""The WaveBlocks Project

Compute and plot the norm of the difference
of the wavefunctions resulting from different
algorithms.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import array
from scipy.linalg import norm
from matplotlib.pyplot import *

from WaveBlocks import FileTools as FT
from WaveBlocks import IOManager
from WaveBlocks import WaveFunction
from WaveBlocks.Plot import legend


def load_data(resultspath, which_norm="wf"):
    # Sort the data from different simulations
    ids = FT.get_result_dirs(resultspath)
    dirs_f = FT.gather_all(ids, "fourier")
    dirs_h = FT.gather_all(ids, "hagedorn")

    dirs_f = FT.sort_by(dirs_f, "eps")
    dirs_h = FT.sort_by(dirs_h, "eps")

    if len(dirs_f) != len(dirs_h):
        raise ValueError("Found different number of fourier and hagedorn simulations!")

    number_simulations = len(dirs_f)

    normdata = []
    axisdata = []
    
    iom_f = IOManager()
    iom_h = IOManager()
    
    # Loop over all simulations
    for dir_f, dir_h in zip(dirs_f, dirs_h):

        print("Comparing simulation " + dir_h + " with " + dir_f)

        # Load the simulation data files
        resultsfile_f = FT.get_results_file(dir_f)
        iom_f.load_file(filename=resultsfile_f)

        resultsfile_h = FT.get_results_file(dir_h)
        iom_h.load_file(filename=resultsfile_h)
        
        # Read the parameters
        parameters_f = iom_f.get_parameters()
        parameters_h = iom_h.get_parameters()
        
        number_components = parameters_f.ncomponents

        # Scalar parameter that discriminates the simulations
        axisdata.append(parameters_f.eps)

        # Get the data
        grid = iom_f.load_grid()
        timesteps = iom_f.load_wavefunction_timegrid()            
        data_f = iom_f.load_wavefunction()
        data_h = iom_h.load_wavefunction()

        # Compute the norm  || u_f - u_h ||_L2 for all timesteps
        data_diff = data_f - data_h

        WF = WaveFunction(parameters_f)
        WF.set_grid(grid)

        norms = []
        
        for i, step in enumerate(timesteps):
            if which_norm == "wf":
                WF.set_values([ data_diff[i,0,:] ])
                no = WF.get_norm()
            elif which_norm == "2":
                no = norm(data_diff[i,0,:])
            elif which_norm == "max":
                no = max(data_diff[i,0,:])
            
            norms.append(no)
        
        # Append norm values to global data structure
        norms = array(norms)
        normdata.append(norms)

    iom_f.finalize()
    iom_h.finalize()

    return (axisdata, normdata, number_simulations, number_components)
 

def plot_data(axisdata, normdata, number_simulations, number_components, timeindices=None, which_norm="wf"):

    if which_norm == "wf":
        nona = "L^2"
    elif which_norm == "2":
        nona = "L^2"
    elif which_norm == "max":
        nona = "max"

    # Plot the time series for each simulation
    fig = figure()
    ax = fig.gca()
    
    for index in xrange(number_simulations):
        ax.plot(normdata[index], label=r"$\varepsilon = $" + str(axisdata[index]))
    
    ax.set_xlabel(r"Timestep $t$")
    ax.set_ylabel(r"$$\| \phi_f - \phi_h \|_{"+nona+r"}$$")
    ax.set_title(r"Timeseries of $$\| \phi_f - \phi_h \|_{"+nona+r"}$$ ")
    legend(loc="outer right")
    fig.savefig("convergence_"+nona+"_all.png")
    close(fig)
    

    # Plot the comparison over versus axisdata
    for t in timeindices:
        # This version works too for array of different lengths
        tmp = []
        for nd in normdata:
            tmp.append(nd[t])
        tmp = array(tmp)

        fig = figure()
        ax = gca()
        ax.plot(axisdata, tmp, "-o", label=r"$t = " + str(t) + r"$")
        
        ax.grid(True)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(r"$\| \phi_f - \phi_h \|_{"+nona+r"}$")
        ax.set_title(r"$\| \phi_f - \phi_h \|_{"+nona+r"}$ for different $\varepsilon$ ")
        legend(loc="outer right")
        fig.savefig("convergence"+str(t)+"_"+nona+"_comparison.png")
        close(fig)


if __name__ == "__main__":
    # Read file with simulation data
    try:
        path_to_results = sys.argv[1]
    except IndexError:
        path_to_results = GlobalDefaults.path_to_results

    # Times (timeseries indices!) for the pointwise comparisons
    times = [0,1,10,15,20,50,-1]

    data = load_data(path_to_results)
    plot_data(*data, timeindices=times)

    data = load_data(path_to_results, which_norm="2")
    plot_data(*data, timeindices=times, which_norm="2")

    data = load_data(path_to_results, which_norm="max")
    plot_data(*data, timeindices=times, which_norm="max")
