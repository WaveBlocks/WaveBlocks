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

import GraphicsDefaults as GD


def read_data(iom):
    """
    @param iom: An I{IOManager} instance providing the simulation data.
    """
    parameters = iom.load_parameters()

    data = []

    # For each mother-child spawn try pair
    # TODO: Generalize for mother-child groups
    for b in xrange(0,iom.get_number_blocks(),2):
        timegrid0 = iom.load_norm_timegrid(block=b)
        time0 = timegrid0 * parameters["dt"]

        # Load data of original packet
        norms0m = iom.load_norm(block=b, split=True)

        normsum0m = [ item**2 for item in norms0m ]
        normsum0m = reduce(lambda x,y: x+y, normsum0m)
        norms0m.append(sqrt(normsum0m))

        # Load data of spawned packet
        norms0c = iom.load_norm(split=True, block=b+1)

        normsum0c = [ item**2 for item in norms0c ]
        normsum0c = reduce(lambda x,y: x+y, normsum0c)
        norms0c.append(sqrt(normsum0c))

        data.append((time0, norms0m, norms0c))

    return (parameters, data)


def plot_norms(parameters, data):
    print("Plotting the norms")

    N = parameters["ncomponents"]

    for index, datum in enumerate(data):

        time = datum[0]
        norms_m = datum[1]
        norms_c = datum[2]

        overall_norm = sqrt(norms_m[-1]**2 + norms_c[-1]**2)

        # Plot the norms for all components
        fig = figure()

        for c in xrange(N):
            ax = subplot(N,1,c+1)

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
        fig.savefig("norms_spawn_components_group"+str(index)+GD.output_format)
        close(fig)


        # Plot the sum of the norms of mother and child
        fig = figure()
        ax = subplot(2,1,1)
        ax.plot(time, overall_norm, color="black", label=r"$\sqrt{\| \Psi^M \|^2 + \| \Psi^C \|^2}$")
        ax.plot(time, norms_m[-1], color="blue", label=r"$\| \Psi^M \|$")

        ax.grid(True)
        ax.set_ylim(0, 1.1*overall_norm.max())
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$\Psi^M$")
        ax.legend(loc="upper left")

        ax = subplot(2,1,2)
        ax.plot(time, overall_norm, color="black", label=r"$\sqrt{\| \Psi^M \|^2 + \| \Psi^C \|^2}$")
        ax.plot(time, norms_c[-1], color="cyan", label=r"$\| \Psi^C \|$")

        ax.grid(True)
        ax.set_ylim(0, 1.1*overall_norm.max())
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$\Psi^C$")
        ax.legend(loc="upper left")

        fig.suptitle(r"Norms of $\Psi^M$ (top) and $\Psi^C$ (bottom)")
        fig.savefig("norms_spawn_sumpacket_group"+str(index)+GD.output_format)
        close(fig)


        # Plot the overall sum of norms of mother and child
        fig = figure()
        ax = fig.gca()

        ax.plot(time, overall_norm, color="black", label=r"$\sqrt{\| \Psi^M \|^2 + \| \Psi^C \|^2}$")
        ax.plot(time, norms_m[-1], color="blue", label=r"$\| \Psi^M \|$")
        ax.plot(time, norms_c[-1], color="cyan", label=r"$\| \Psi^C \|$")

        ax.grid(True)
        ax.set_ylim(0, 1.1*overall_norm.max())
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        legend(loc="outer right")
        ax.set_xlabel(r"Time $t$")
        ax.set_title(r"Norm of $\Psi^M$ and $\Psi^C$ and $\Psi^M + \Psi^C$")
        fig.savefig("norms_spawn_sumall_group"+str(index)+GD.output_format)
        close(fig)


        # Plot the drift of the sum of all norms
        fig = figure()
        ax = fig.gca()

        ax.plot(time, overall_norm[0] - overall_norm)

        ax.grid(True)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"$\|\Psi(t=0)\| - \|\Psi(t)\|$")
        ax.set_title(r"Drift of the total norm $\|\Psi\| = \sqrt{\| \Psi^M \|^2 + \| \Psi^C \|^2}$")
        fig.savefig("norms_spawn_sumall_drift_group"+str(index)+GD.output_format)
        close(fig)




if __name__ == "__main__":
    iom = IOManager()

    # Read the file with the simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    data = read_data(iom)
    plot_norms(*data)

    iom.finalize()
