"""The WaveBlocks Project

Plot the energies of the different wavepackets for several simulation runs.

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

import GraphicsDefaults as GD


def load_data(resultspath):
    # Sort the data from different simulations
    dirs = FT.get_result_dirs(resultspath)
    resultsdir = FT.sort_by(dirs, "eps")

    number_simulations = FT.get_number_simulations(resultspath)

    ekindata = []
    epotdata = []
    axisdata = []

    iom = IOManager()

    for resultdir in resultsdir:
        resultsfile = FT.get_results_file(resultdir)

        print(" Reading " + resultsfile)

        iom.open_file(filename=resultsfile)
        parameters = iom.get_parameters()
        number_components = parameters["ncomponents"]
        axisdata.append(parameters["eps"])

        ekin, epot = iom.load_energy()
        ekindata.append(ekin)
        epotdata.append(epot)

    iom.finalize()

    return (axisdata, ekindata, epotdata, number_simulations, number_components)


def plot_data(axisdata, ekindata, epotdata, number_simulations, number_components):
    colormap = ["b", "g", "r", "c", "m", "y", "b"]

    # Plot the time series for each simulation
    fig = figure()
    ax = fig.gca()

    for index in xrange(number_simulations):
        for jndex in xrange(number_components):
            ax.plot(ekindata[index][:,jndex], label=r"$"+str(index)+r": E^{kin}_"+str(jndex)+r"$", color=colormap[2*jndex%len(colormap)])
            ax.plot(epotdata[index][:,jndex], label=r"$"+str(index)+r": E^{pot}_"+str(jndex)+r"$", color=colormap[(2*jndex+1)%len(colormap)])

    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_xlabel(r"Timestep")
    ax.set_ylabel(r"$E$")
    ax.set_title(r"Energies timeseries comparison")
    legend(loc="outer right")
    fig.savefig("energies_all"+GD.output_format)
    close(fig)


    endkindata = [ array([ ekindata[j][-1,i] for j in xrange(number_simulations) ]) for i in xrange(number_components) ]
    endpotdata = [ array([ epotdata[j][-1,i] for j in xrange(number_simulations) ]) for i in xrange(number_components) ]

    # Plot the comparison over versus axisdata
    fig = figure()
    ax = fig.gca()

    # Kinetic and potential energies per component
    for index in xrange(number_components):
        ax.plot(axisdata, endkindata[index], label=r"$E^{kin}_"+str(index)+r"$", marker="o")
        ax.plot(axisdata, endpotdata[index], label=r"$E^{pot}_"+str(index)+r"$", marker="o")

    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$E$")
    ax.set_title(r"Energies end of time comparison")
    legend(loc="outer right")
    fig.savefig("energies_comparison"+GD.output_format)
    close(fig)


if __name__ == "__main__":
    # Read file with simulation data
    try:
        path_to_results = sys.argv[1]
    except IndexError:
        path_to_results = GlobalDefaults.path_to_results

    data = load_data(path_to_results)
    plot_data(*data)
