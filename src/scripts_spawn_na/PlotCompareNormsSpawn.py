"""The WaveBlocks Project

Plot the norms of the different wavepackets as well as the sum
of all norms for the original and spawned wavepacket in a
single graph.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import *
from matplotlib.pyplot import *

from WaveBlocks import IOManager
from WaveBlocks.Plot import legend
from WaveBlocks.Utils import common_timesteps

import GraphicsDefaults as GD


def read_data(iom_o, iom_s, gid, bid_ref=0):
    """
    @param iom_s: An I{IOManager} instance providing the spawning simulation data.
    @param iom_o: An I{IOManager} instance providing the reference simulation data.
    """
    parameters_o = iom_o.load_parameters()
    parameters_s = iom_s.load_parameters()

    # Retrieve reference data
    timegrido = iom_o.load_norm_timegrid(blockid=bid_ref)
    timeo = timegrido * parameters_o["dt"]

    normso = iom_o.load_norm(split=True, blockid=bid_ref)
    normsumo = reduce(lambda x,y: x+y, [ item**2 for item in normso ])
    normso.append(sqrt(normsumo))

    data_ref = (timegrido, timeo, normso)

    # For each mother-child spawn try pair
    bidm, bidc = iom_s.get_block_ids(groupid=gid)

    # Retrieve data of spawned packets
    data = []

    timegrid0 = iom_s.load_norm_timegrid(blockid=bidm)
    time0 = timegrid0 * parameters_s["dt"]

    # Load data of original packet
    norms0m = iom_s.load_norm(blockid=bidm, split=True)

    normsum0m = [ item**2 for item in norms0m ]
    normsum0m = reduce(lambda x,y: x+y, normsum0m)
    norms0m.append(sqrt(normsum0m))

    # Load data of spawned packet
    norms0c = iom_s.load_norm(split=True, blockid=bidc)

    normsum0c = [ item**2 for item in norms0c ]
    normsum0c = reduce(lambda x,y: x+y, normsum0c)
    norms0c.append(sqrt(normsum0c))

    data.append((timegrid0, time0, norms0m, norms0c))

    return (parameters_o, parameters_s, data_ref, data)


def plot_norms(gid, parameters_o, parameters_s, data_ref, data):
    print("Plotting the norms of group '"+str(gid)+"'")

    N = parameters_s["ncomponents"]

    timegrido = data_ref[0]
    timeo = data_ref[1]
    normso = data_ref[2]

    for datum in data:

        timegrid = datum[0]
        time = datum[1]
        norms_m = datum[2]
        norms_c = datum[3]

        overall_norm = sqrt(norms_m[-1]**2 + norms_c[-1]**2)

        indo, inds = common_timesteps(timegrido, timegrid)

        # Plot the norms for all components
        fig = figure()

        for c in xrange(N):
            ax = subplot(N,1,c+1)

            # Plot original norm
            ax.plot(timeo, normso[c], color="orange", label=r"$\| \Phi_"+str(c)+r"^O \|$")

            # Plot the norms of the individual wavepackets
            ax.plot(time, overall_norm, color="black", label=r"$\sqrt{\| \Psi^M \|^2 + \| \Psi^C \|^2}$")
            ax.plot(time, sqrt(norms_m[c]**2 + norms_c[c]**2), color="gray", label=r"$\sqrt{\| \Phi_"+str(c)+r"^M \|^2 + \| \Phi_"+str(c)+r"^C \|^2}$")
            ax.plot(time, norms_m[c], color="blue", label=r"$\| \Phi_"+str(c)+r"^M \|$")
            ax.plot(time, norms_c[c], color="cyan", label=r"$\| \Phi_"+str(c)+r"^C \|$")

            ax.grid(True)
            ax.set_ylim(0, 1.1*overall_norm.max())
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            ax.set_xlabel(r"Time $t$")
            ax.set_ylabel(r"$\Phi_"+str(c)+r"$")
            ax.legend(loc="upper left")

        fig.suptitle("Per-component norms of $\Psi^M$ and $\Psi^C$")
        fig.savefig("norms_compare_components_group"+str(gid)+GD.output_format)
        close(fig)


        # Plot the sum of the norms of mother and child
        fig = figure()
        ax = subplot(2,1,1)
        ax.plot(timeo, normso[-1], color="orange", label=r"$\| \Psi^O \|$")
        ax.plot(time, overall_norm, color="black", label=r"$\sqrt{\| \Psi^M \|^2 + \| \Psi^C \|^2}$")
        ax.plot(time, norms_m[-1], color="blue", label=r"$\| \Psi^M \|$")

        ax.grid(True)
        ax.set_ylim(0, 1.1*overall_norm.max())
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$\Psi^M$")
        ax.legend(loc="upper left")

        ax = subplot(2,1,2)
        ax.plot(timeo, normso[-1], color="orange", label=r"$\| \Psi^O \|$")
        ax.plot(time, overall_norm, color="black", label=r"$\sqrt{\| \Psi^M \|^2 + \| \Psi^C \|^2}$")
        ax.plot(time, norms_c[-1], color="cyan", label=r"$\| \Psi^C \|$")

        ax.grid(True)
        ax.set_ylim(0, 1.1*overall_norm.max())
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$\Psi^C$")
        ax.legend(loc="upper left")

        fig.suptitle(r"Norms of $\Psi^M$ (top) and $\Psi^C$ (bottom)")
        fig.savefig("norms_compare_sumpacket_group"+str(gid)+GD.output_format)
        close(fig)


        # Plot the overall sum of norms of mother and child
        fig = figure()
        ax = fig.gca()

        ax.plot(timeo, normso[-1], color="orange", label=r"$\| \Psi^O \|$")
        ax.plot(time, overall_norm, color="black", label=r"$\sqrt{\| \Psi^M \|^2 + \| \Psi^C \|^2}$")
        ax.plot(time, norms_m[-1], color="blue", label=r"$\| \Psi^M \|$")
        ax.plot(time, norms_c[-1], color="cyan", label=r"$\| \Psi^C \|$")

        ax.grid(True)
        ax.set_ylim(0, 1.1*overall_norm.max())
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        legend(loc="outer right")
        ax.set_xlabel(r"Time $t$")
        ax.set_title(r"Norm of $\Psi^M$ and $\Psi^C$ and $\Psi^M + \Psi^C$")
        fig.savefig("norms_compare_sumall_group"+str(gid)+GD.output_format)
        close(fig)


        # Plot the drift of the sum of all norms
        fig = figure()
        ax = fig.gca()

        ax.plot(timeo, normso[-1][0] - normso[-1], color="orange")
        ax.plot(time, overall_norm[0] - overall_norm, color="blue")

        ax.grid(True)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$\|\Psi(t=0)\| - \|\Psi(t)\|$")
        ax.set_title(r"Drift of the total norm $\|\Psi\| = \sqrt{\| \Psi^M \|^2 + \| \Psi^C \|^2}$")
        fig.savefig("norms_compare_sumall_drift_group"+str(gid)+GD.output_format)
        close(fig)


        # Plot difference of norms between original and spawned

        # Plot the norms for all components
        fig = figure()

        for c in xrange(N):
            ax = subplot(N,1,c+1)

            # Plot original norm
            ax.plot(timeo[indo], normso[c][indo] - sqrt(norms_m[c][inds]**2 + norms_c[c][inds]**2),
                    color="blue", label=r"$\|\Phi_"+str(c)+r"^O\| - \sqrt{\| \Phi_"+str(c)+r"^M \|^2 + \| \Phi_"+str(c)+r"^C \|^2}$")

            ax.grid(True)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            ax.set_xlabel(r"Time $t$")
            ax.set_ylabel(r"$\Phi_"+str(c)+r"$")
            ax.legend(loc="upper left")

        fig.suptitle("Per-component norm difference $\|\Psi^O\| - \|\Psi^S\|$")
        fig.savefig("norms_compare_components_diff_group"+str(gid)+GD.output_format)
        close(fig)


        # Plot the overall sum of norms of mother and child
        fig = figure()
        ax = fig.gca()

        ax.plot(timeo[indo], normso[-1][indo] - overall_norm[inds], color="blue", label=r"$\|\Psi^O\| - \|\Psi^M\| - \|\Psi^C\|$")

        ax.grid(True)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        legend(loc="outer right")
        ax.set_xlabel(r"Time $t$")
        ax.set_title(r"Norm difference $\|\Psi^O\| - \|\Psi^S\|$")
        fig.savefig("norms_compare_sumall_diff_group"+str(gid)+GD.output_format)
        close(fig)




if __name__ == "__main__":
    iom_o = IOManager()
    iom_s = IOManager()

    # NOTE
    #
    # first cmd-line data file is spawning data
    # second cmd-line data file is reference data

    # Read file with new simulation data
    try:
        iom_s.open_file(filename=sys.argv[1])
    except IndexError:
        iom_s.open_file()

    # Read file with original reference simulation data
    try:
        iom_o.open_file(filename=sys.argv[2])
    except IndexError:
        iom_o.open_file()

    gids = iom_s.get_group_ids(exclude=["global"])

    for gid in gids:
        data = read_data(iom_o, iom_s, gid)
        plot_norms(gid, *data)

    iom_o.finalize()
    iom_s.finalize()
