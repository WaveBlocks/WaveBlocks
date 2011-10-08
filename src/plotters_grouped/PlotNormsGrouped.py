"""The WaveBlocks Project

Plot the norms of the different wavepackets as well as the sum
of all norms and norm drifts. All data from blocks in the same
group end up in the same plot.


@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import *
from matplotlib.pyplot import *

from WaveBlocks import IOManager
from WaveBlocks.Plot import legend

import GraphicsDefaults as GD


def read_data(iom, gid):
    parameters = iom.load_parameters()

    data = []

    bids = iom.get_block_ids(groupid=gid)

    for bid in bids:
        if not iom.has_norm(blockid=bid):
            continue

        timegrid = iom.load_norm_timegrid(blockid=bid)
        time = timegrid * parameters["dt"]

        norms = iom.load_norm(blockid=bid, split=True)

        normsum = [ item**2 for item in norms ]
        normsum = reduce(lambda x,y: x+y, normsum)
        normsum = sqrt(normsum)

        data.append((timegrid, time, norms, normsum))

    return (parameters, data)


def plot_norms(gid, parameters, data):
    print("Plotting the norms of group '"+str(gid)+"'")

    if len(data) == 0:
        return

    N = parameters["ncomponents"]

    # Plot the norms for all components
    fig = figure()

    for index, datum in enumerate(data):
        timegrid, time, norms, normsum = datum

        for c in xrange(N):
            ax = subplot(N,1,c+1)

            # Plot the norms of the individual wavepackets
            ax.plot(time, normsum, label=r"$\sqrt{\sum_i \| \Phi_i\|^2}$")
            ax.plot(time, norms[c], label=r"$\| \Phi_"+str(c)+r"\|$")

            ax.grid(True)
            ax.set_ylim(0, 1.1*normsum.max())
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            ax.set_xlabel(r"Time $t$")
            ax.set_ylabel(r"$\Phi_"+str(c)+r"$")
            ax.legend(loc="upper left")

    fig.suptitle("Per-component norms")
    fig.savefig("norms_components_group"+str(gid)+GD.output_format)
    close(fig)


    # Plot the drift of the norms component-wise
    fig = figure()

    for c in xrange(N):
        # Compue data
        normsum = [ datum[2][c]**2 for datum in data ]
        normsum = reduce(lambda x,y: x+y, normsum)
        normsum = sqrt(normsum)

        # Plot the norm dirft of the individual components
        ax = subplot(N,1,c+1)

        ax.plot(time, normsum, label=r"$\| \Phi_"+str(c)+r"\|$")

        ax.grid(True)
        ax.set_ylim(0, 1.1*normsum.max())
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$\|\Phi_"+str(c)+r"(t=0)\| - \|\Phi_"+str(c)+r"(t)\|$")
        ax.legend(loc="upper left")
        ax.set_title(r"Drift of the norms of $\|\Phi_i\|$")

    fig.suptitle("Per-component norm dirft")
    fig.savefig("norms_components_drift_group"+str(gid)+GD.output_format)
    close(fig)


    # Plot the drift of the sum of all norms
    fig = figure()
    ax = fig.gca()

    for index, datum in enumerate(data):
        timegrid, time, norms, normsum = datum

        ax.plot(time, normsum[0] - normsum)

    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$\|\Psi(t=0)\| - \|\Psi(t)\|$")
    ax.set_title(r"Drift of the total norm $\|\Psi\|$")
    fig.savefig("norms_sumall_drift_group"+str(gid)+GD.output_format)
    close(fig)




if __name__ == "__main__":
    iom = IOManager()

    # Read the file with the simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    gids = iom.get_group_ids()

    for gid in gids:
        params, data = read_data(iom, gid)
        plot_norms(gid, params, data)

    iom.finalize()
