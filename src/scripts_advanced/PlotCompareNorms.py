"""The WaveBlocks Project

Plot the norms of the different wavepackets for several simulation runs.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import array
from matplotlib.pyplot import *

from WaveBlocks import GlobalDefaults
from WaveBlocks import FileTools as FT
from WaveBlocks import IOManager
from WaveBlocks.Plot import legend


def load_data(resultspath):
    # Sort the data from different simulations according to the filenames
    dirs = FT.get_result_dirs(resultspath)
    resultsdir = FT.sort_by(dirs, "eps")
    
    number_simulations = FT.get_number_simulations(resultspath)

    normdata = []
    axisdata = []

    iom = IOManager()

    for resultdir in resultsdir:
        resultsfile = FT.get_results_file(resultdir)

        print(" Reading " + resultsfile)
        
        iom.load_file(filename=resultsfile)
        parameters = iom.get_parameters()
        number_components = parameters.ncomponents
        axisdata.append(parameters.eps)

        norms = iom.load_norm()
        normdata.append(norms)

    iom.finalize()

    return (axisdata, normdata, number_simulations, number_components)
 

def plot_data(axisdata, normdata, number_simulations, number_components):
    colormap = ["b", "g", "r", "c", "m", "y", "b"]

    # Plot the time series for each simulation
    fig = figure()
    ax = fig.gca()

    for index in xrange(number_simulations):
        for jndex in xrange(number_components):
            ax.plot(normdata[index][:,jndex], label=r"$"+str(index)+r":\| \psi_"+str(jndex)+r"\|$", color=colormap[jndex%len(colormap)])
            
    ax.set_ylim(0,1.1)
    ax.grid(True)
    ax.set_xlabel(r"Timestep")
    ax.set_ylabel(r"$\|\cdot\|$")
    ax.set_title(r"Norms timeseries comparison")
    legend(loc="outer right")
    fig.savefig("norms_comparison_all.png")
    close(fig)


    enddata = [ array([ normdata[j][-1,i] for j in xrange(number_simulations) ]) for i in xrange(number_components) ]

    # Plot the comparison over versus axisdata
    fig = figure()
    ax = fig.gca()

    for jndex in xrange(number_components):
        plot(axisdata, enddata[jndex], label=r"$\| \psi_"+str(jndex)+r"\|$", marker="o", color=colormap[jndex%len(colormap)])

    ax.grid(True)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\|\cdot\|$")
    ax.set_title(r"Norms end of time comparison")
    legend(loc="outer right")
    fig.savefig("norms_comparison.png")
    close(fig)


if __name__ == "__main__":
    # Read file with simulation data
    try:
        path_to_results = sys.argv[1]
    except IndexError:
        path_to_results = GlobalDefaults.path_to_results

    data = load_data(path_to_results)
    plot_data(*data)
