"""The WaveBlocks Project

Plot the norms of the different wavepackets as well as the sum of all norms.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import sqrt, max
from matplotlib.pyplot import *

from WaveBlocks import IOManager
from WaveBlocks.Plot import legend


def read_all_datablocks(iom):
    """Read the data from all blocks that contain any usable data.
    @param iom: An I{IOManager} instance providing the simulation data.
    """
    # Iterate over all blocks and plot their data
    for block in xrange(iom.get_number_blocks()):
        plot_norms(read_data(iom, block=block), index=block)


def read_data(iom, block=0):
    """
    @param iom: An I{IOManager} instance providing the simulation data.
    @keyword block: The data block from which the values are read.
    """
    parameters = iom.get_parameters()
    timegrid = iom.load_norm_timegrid(block=block)
    time = timegrid * parameters["dt"]

    norms = iom.load_norm(block=block, split=True)

    normsum = [ item**2 for item in norms ]
    normsum = reduce(lambda x,y: x+y, normsum)
    norms.append(sqrt(normsum))

    return (time, norms)


def plot_norms(data, index=0):
    print("Plotting the norms of data block "+str(index))

    timegrid, norms = data

    # Plot the norms
    fig = figure()
    ax = fig.gca()

    # Plot the norms of the individual wavepackets
    for i, datum in enumerate(norms[:-1]):
        label_i = r"$\| \Phi_"+str(i)+r" \|$"
        ax.plot(timegrid, datum, label=label_i)

    # Plot the sum of all norms
    ax.plot(timegrid, norms[-1], color=(1,0,0), label=r"${\sqrt{\sum_i {\| \Phi_i \|^2}}}$")

    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_title(r"Norms of $\Psi$")
    legend(loc="outer right")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylim([0,1.1*max(norms[:-1])])
    fig.savefig("norms_block"+str(index)+".png")
    close(fig)


    fig = figure()
    ax = fig.gca()

    # Plot the squared norms of the individual wavepackets
    for i, datum in enumerate(norms[:-1]):
        label_i = r"$\| \Phi_"+str(i)+r" \|^2$"
        ax.plot(timegrid, datum**2, label=label_i)

    # Plot the squared sum of all norms
    ax.plot(timegrid, norms[-1]**2, color=(1,0,0), label=r"${\sum_i {\| \Phi_i \|^2}}$")

    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_title(r"Squared norms of $\Psi$")
    legend(loc="outer right")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylim([0,1.1*max(norms[-1]**2)])
    fig.savefig("norms_sqr_block"+str(index)+".png")
    close(fig)


    # Plot the difference from the theoretical norm
    fig = figure()
    ax = fig.gca()

    ax.plot(timegrid, abs(norms[-1][0] - norms[-1]), label=r"$\|\Psi\|_0 - \|\Psi\|_t$")

    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_title(r"Drift of $\| \Psi \|$")
    legend(loc="outer right")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$\|\Psi\|_0 - \|\Psi\|_t$")
    fig.savefig("norms_drift_block"+str(index)+".png")
    close(fig)


if __name__ == "__main__":
    iom = IOManager()

    # Read the file with the simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    read_all_datablocks(iom)

    iom.finalize()
