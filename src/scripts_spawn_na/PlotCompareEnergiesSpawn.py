"""The WaveBlocks Project

Plot the energies of the different wavepackets as well as the sum
of all energies for the original and spawned wavepacket in a
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
    :param iom_s: An ``IOManager`` instance providing the spawning simulation data.
    :param iom_o: An ``IOManager`` instance providing the reference simulation data.
    """
    parameters_o = iom_o.load_parameters()
    parameters_s = iom_s.load_parameters()

    # Retrieve reference data
    timegrido = iom_o.load_energy_timegrid(blockid=bid_ref)
    timeo = timegrido * parameters_o["dt"]
    energieso = iom_o.load_energy(split=True, blockid=bid_ref)
    data_ref = (timegrido, timeo, energieso)

    # For each mother-child spawn try pair
    bidm, bidc = iom_s.get_block_ids(groupid=gid)

    data = []

    timegrid0 = iom_s.load_energy_timegrid(blockid=bidm)
    time0 = timegrid0 * parameters_s["dt"]

    # Load data of original packet
    energies0m = iom_s.load_energy(blockid=bidm, split=True)
    # Load data of spawned packet
    energies0c = iom_s.load_energy(blockid=bidc, split=True)

    data.append((timegrid0, time0, energies0m, energies0c))

    return (parameters_o, parameters_s, data_ref, data)


def plot_energies(gid, parameters_o, parameters_s, data_ref, data):
    print("Plotting the energies of group '"+str(gid)+"'")

    N = parameters_o["ncomponents"]

    timegrido = data_ref[0]
    timeo = data_ref[1]
    energies_o = data_ref[2]

    ekin_o = reduce(lambda x,y: x+y, [ energies_o[0][i] for i in xrange(N) ])
    epot_o = reduce(lambda x,y: x+y, [ energies_o[1][i] for i in xrange(N) ])
    esum_o = ekin_o + epot_o

    for datum in data:

        timegrid = datum[0]
        time = datum[1]
        energies_m = datum[2]
        energies_c = datum[3]

        ekin_m = reduce(lambda x,y: x+y, [ energies_m[0][i] for i in xrange(N) ])
        epot_m = reduce(lambda x,y: x+y, [ energies_m[1][i] for i in xrange(N) ])
        ekin_c = reduce(lambda x,y: x+y, [ energies_c[0][i] for i in xrange(N) ])
        epot_c = reduce(lambda x,y: x+y, [ energies_c[1][i] for i in xrange(N) ])

        esum = ekin_m + epot_m + ekin_c + epot_c

        indo, inds = common_timesteps(timegrido, timegrid)

        # Plot the energies for all components
        fig = figure()

        for c in xrange(N):
            ax = subplot(N,1,c+1)

            # Plot the original energies
            ax.plot(timeo, energies_o[0][c], color="red", label=r"$E^O_{kin}$")
            ax.plot(timeo, energies_o[1][c], color="orange", label=r"$E^O_{pot}$")
            ax.plot(timeo, energies_o[0][c] + energies_o[1][c], color="gray", label=r"$E^O$")
            # Plot the energies of the individual wavepackets
            ax.plot(time, energies_m[0][c], color="darkgreen", label=r"$E^M_{kin}$")
            ax.plot(time, energies_c[0][c], color="lightgreen", label=r"$E^C_{kin}$")
            ax.plot(time, energies_m[1][c], color="blue", label=r"$E^M_{pot}$")
            ax.plot(time, energies_c[1][c], color="cyan", label=r"$E^C_{pot}$")
            # Overall energy on component c
            ax.plot(time, energies_m[0][c] + energies_c[0][c] + energies_m[1][c] + energies_c[1][c], color="black", label=r"$\sum E$")

            ax.grid(True)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            ax.set_xlabel(r"Time $t$")
            ax.set_ylabel(r"$\Phi_"+str(c)+r"$")
            ax.legend(loc="upper left")

        fig.suptitle("Per-component energies of $\Psi^M$ and $\Psi^C$")
        fig.savefig("energies_compare_components_group"+str(gid)+GD.output_format)
        close(fig)


        # Plot the sum of the energies of mother and child
        fig = figure()
        ax = subplot(2,1,1)
        ax.plot(timeo, ekin_o, color="red", label=r"$E_{kin}^O$")
        ax.plot(timeo, epot_o, color="orange", label=r"$E_{pot}^O$")
        ax.plot(timeo, ekin_o + epot_o, color="magenta", label=r"$E^O$")
        ax.plot(time, ekin_m, color="green", label=r"$E_{kin}^M$")
        ax.plot(time, epot_m, color="blue", label=r"$E_{pot}^M$")
        ax.plot(time, ekin_m + epot_m, color="black", label=r"$E_{kin}^M + E_{pot}^M$")

        ax.grid(True)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$\Psi^M$")
        ax.legend(loc="upper left")

        ax = subplot(2,1,2)
        ax.plot(timeo, ekin_o, color="red", label=r"$E_{kin}^O$")
        ax.plot(timeo, epot_o, color="orange", label=r"$E_{pot}^O$")
        ax.plot(timeo, ekin_o + epot_o, color="magenta", label=r"$E^O$")
        ax.plot(time, ekin_c, color="green", label=r"$E_{kin}^C$")
        ax.plot(time, epot_c, color="blue", label=r"$E_{pot}^C$")
        ax.plot(time, ekin_c + epot_c, color="black", label=r"$E_{kin}^C + E_{pot}^C$")

        ax.grid(True)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$\Psi^C$")
        ax.legend(loc="upper left")

        fig.suptitle(r"Energies of $\Psi^M$ (top) and $\Psi^C$ (bottom)")
        fig.savefig("energies_compare_packetsum_group"+str(gid)+GD.output_format)
        close(fig)


        # Plot the overall sum of energies of mother and child
        fig = figure()
        ax = fig.gca()

        ax.plot(timeo, ekin_o + epot_o, color="orange", label=r"$E_{kin}^O + E_{pot}^O$")
        ax.plot(time, ekin_m + epot_m, color="blue", label=r"$E_{kin}^M + E_{pot}^M$")
        ax.plot(time, ekin_c + epot_c, color="cyan", label=r"$E_{kin}^C + E_{pot}^C$")
        ax.plot(time, ekin_m + epot_m + ekin_c + epot_c, color="black", label=r"$E^M + E^C$")

        ax.grid(True)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        legend(loc="outer right")
        ax.set_xlabel(r"Time $t$")
        ax.set_title(r"Energies of $\Psi^M$ and $\Psi^C$ and $\Psi^M + \Psi^C$")
        fig.savefig("energies_compare_sumall_group"+str(gid)+GD.output_format)
        close(fig)


        # Plot the drift of the sum of all energies
        fig = figure()
        ax = fig.gca()

        ax.plot(timeo, esum_o[0] - esum_o, color="orange")
        ax.plot(time, esum[0] - esum, color="blue")

        ax.grid(True)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$E(t=0) - E(t)$")
        ax.set_title(r"Drift of the total energy $E = E^M + E^C$")
        fig.savefig("energies_compare_sumall_drift_group"+str(gid)+GD.output_format)
        close(fig)


        # Plot difference of energies between original and spawned

        # Plot the energies for all components
        fig = figure()

        for c in xrange(N):
            ax = subplot(N,1,c+1)

            # Plot the original energies
            ax.plot(timeo[indo], energies_o[0][c][indo] - (energies_m[0][c][inds] + energies_c[0][c][inds]), color="green", label=r"$E^O_{kin} - E^S_{kin}$")
            ax.plot(timeo[indo], energies_o[1][c][indo] - (energies_m[1][c][inds] + energies_c[1][c][inds]), color="blue", label=r"$E^O_{pot} - E^S_{pot}$")
            ax.plot(timeo[indo], energies_o[0][c][indo] + energies_o[1][c][indo] -
                    (energies_m[0][c][inds] + energies_c[0][c][inds] + energies_m[1][c][inds] + energies_c[1][c][inds]), color="red", label=r"$E^O - E^S$")

            ax.grid(True)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            ax.set_xlabel(r"Time $t$")
            ax.set_ylabel(r"$\Phi_"+str(c)+r"$")
            ax.legend(loc="upper left")

        fig.suptitle("Per-component energy difference $E^O - E^S$")
        fig.savefig("energies_compare_components_diff_group"+str(gid)+GD.output_format)
        close(fig)


        # Plot the overall sum of energies of mother and child
        fig = figure()
        ax = fig.gca()

        ax.plot(timeo[indo], ekin_o[indo] - (ekin_m[inds] + ekin_c[inds]), color="green", label=r"$E_{kin}^O + E_{kin}^S$")
        ax.plot(timeo[indo], epot_o[indo] - (epot_m[inds] + epot_c[inds]), color="blue", label=r"$E_{pot}^O + E_{pot}^S$")
        ax.plot(timeo[indo], ekin_o[indo] + epot_o[indo] - (ekin_m[inds] + epot_m[inds] + ekin_c[inds] + epot_c[inds]), color="red", label=r"$E^O + E^S$")

        ax.grid(True)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        legend(loc="outer right")
        ax.set_xlabel(r"Time $t$")
        ax.set_title(r"Energy difference $E^O - E^S$")
        fig.savefig("energies_compare_sumall_diff_group"+str(gid)+GD.output_format)
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
        plot_energies(gid, *data)

    iom_o.finalize()
    iom_s.finalize()
