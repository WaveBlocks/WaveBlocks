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

import GraphicsDefaults as GD


def read_data(iom):
    """
    @param iom: An I{IOManager} instance providing the simulation data.
    """
    parameters = iom.load_parameters()
    NB = iom.get_number_blocks()

    data = []

    # For each mother-child spawn try pair
    # TODO: Generalize for mother-child groups
    for b in xrange(0,NB,2):
        timegrid0 = iom.load_energy_timegrid(blockid=b)
        time0 = timegrid0 * parameters["dt"]

        # Load data of original packet
        energies0m = iom.load_energy(blockid=b, split=True)

        # Load data of spawned packet
        energies0c = iom.load_energy(blockid=b+1, split=True)

        data.append((time0, energies0m, energies0c))

    return (parameters, data)


def plot_energies(parameters, data):
    print("Plotting the energies")

    N = parameters["ncomponents"]

    for index, datum in enumerate(data):

        time = datum[0]
        energies_m = datum[1]
        energies_c = datum[2]

        ekin_m = reduce(lambda x,y: x+y, [ energies_m[0][i] for i in xrange(N) ])
        epot_m = reduce(lambda x,y: x+y, [ energies_m[1][i] for i in xrange(N) ])
        ekin_c = reduce(lambda x,y: x+y, [ energies_c[0][i] for i in xrange(N) ])
        epot_c = reduce(lambda x,y: x+y, [ energies_c[1][i] for i in xrange(N) ])

        esum = ekin_m + epot_m + ekin_c + epot_c

        # Plot the energies for all components
        fig = figure()

        for c in xrange(N):
            ax = subplot(N,1,c+1)

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
        fig.savefig("energies_spawn_components_group"+str(index)+GD.output_format)
        close(fig)


        # Plot the sum of the energies of mother and child
        fig = figure()
        ax = subplot(2,1,1)
        ax.plot(time, ekin_m, color="green", label=r"$E_{kin}^M$")
        ax.plot(time, epot_m, color="blue", label=r"$E_{pot}^M$")
        ax.plot(time, ekin_m + epot_m, color="black", label=r"$E_{kin}^M + E_{pot}^M$")

        ax.grid(True)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$\Psi^M$")
        ax.legend(loc="upper left")

        ax = subplot(2,1,2)
        ax.plot(time, ekin_c, color="green", label=r"$E_{kin}^C$")
        ax.plot(time, epot_c, color="blue", label=r"$E_{pot}^C$")
        ax.plot(time, ekin_c + epot_c, color="black", label=r"$E_{kin}^C + E_{pot}^C$")

        ax.grid(True)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$\Psi^C$")
        ax.legend(loc="upper left")

        fig.suptitle(r"Energies of $\Psi^M$ (top) and $\Psi^C$ (bottom)")
        fig.savefig("energies_spawn_packetsum_group"+str(index)+GD.output_format)
        close(fig)


        # Plot the overall sum of energies of mother and child
        fig = figure()
        ax = fig.gca()

        ax.plot(time, ekin_m + epot_m, color="blue", label=r"$E_{kin}^M + E_{pot}^M$")
        ax.plot(time, ekin_c + epot_c, color="cyan", label=r"$E_{kin}^C + E_{pot}^C$")
        ax.plot(time, ekin_m + epot_m + ekin_c + epot_c, label=r"$E^M + E^C$")

        ax.grid(True)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        legend(loc="outer right")
        ax.set_xlabel(r"Time $t$")
        ax.set_title(r"Energies of $\Psi^M$ and $\Psi^C$ and $\Psi^M + \Psi^C$")
        fig.savefig("energies_spawn_sumall_group"+str(index)+GD.output_format)
        close(fig)


        # Plot the drift of the sum of all energies
        fig = figure()
        ax = fig.gca()

        ax.plot(time, esum[0] - esum)

        ax.grid(True)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$E(t=0) - E(t)$")
        ax.set_title(r"Drift of the total energy $E = E^M + E^C$")
        fig.savefig("energies_spawn_sumall_drift_group"+str(index)+GD.output_format)
        close(fig)




if __name__ == "__main__":
    iom = IOManager()

    # Read the file with the simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    data = read_data(iom)
    plot_energies(*data)

    iom.finalize()
