"""The WaveBlocks Project

Plot the norms of the different wavepackets as well as the sum
of all norms for the original and spawned wavepacket.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import *
from matplotlib.pyplot import *

from WaveBlocks import IOManager
from WaveBlocks.Plot import legend

import GraphicsDefaults as GD


def read_data(f):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    parameters = f.get_parameters()

    timegrid0 = f.load_norm_timegrid()
    time0 = timegrid0 * parameters["dt"]
    timegrid1 = f.load_norm_timegrid(block=1)
    time1 = timegrid1 * parameters["dt"]

    # Load data of original packet
    norms0 = f.load_norm(split=True)

    normsum0 = [ item**2 for item in norms0 ]
    normsum0 = reduce(lambda x,y: x+y, normsum0)
    norms0.append(sqrt(normsum0))

    # Load data of spawned packet
    norms1 = f.load_norm(split=True, block=1)

    normsum1 = [ item**2 for item in norms1 ]
    normsum1 = reduce(lambda x,y: x+y, normsum1)
    norms1.append(sqrt(normsum1))

    return (time0, time1, norms0, norms1)


def plot_norms(time0, time1, norms0, norms1):
    print("Plotting the norms")

    # Plot the norms
    fig = figure()
    ax = subplot(2,1,1)

    # Plot the norms of the individual wavepackets
    for i, datum in enumerate(norms0[:-1]):
        ax.plot(time0, datum, label=r"$\| \Phi_"+str(i)+r" \|$")

    # Plot the sum of all norms
    ax.plot(time0, norms0[-1], color=(1,0,0), label=r"$\sqrt{\sum_i {\| \Phi_i \|^2}}$")

    xlims = ax.get_xlim()
    ax.grid(True)
    ax.set_ylim([0.9*min(norms0[-1])[0],1.1*max(norms0[-1])[0]])
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_xlabel(r"Time $t$")
    ax.set_title(r"Norms of the mother packet $\Psi_m$")

    ax = subplot(2,1,2)

    # Plot the norms of the individual wavepackets
    for i, datum in enumerate(norms1[:-1]):
        ax.plot(time1, datum, label=r"$\| \Phi_"+str(i)+r" \|$")

    # Plot the sum of all norms
    ax.plot(time1, norms1[-1], color=(1,0,0), label=r"$\sqrt{\sum_i {\| \Phi_i \|^2}}$")

    ax.set_xlim(xlims)
    ax.grid(True)
    ax.set_ylim([0.9*min(norms1[-1])[0],1.1*max(norms1[-1])[0]])
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_xlabel(r"Time $t$")
    ax.set_title(r"Norms of the spawned packet $\Psi_s$")

    fig.savefig("norms_spawn"+GD.output_format)
    close(fig)


    # Data transformation necessary for plotting different parts
    x0 = time0.shape[0]
    x1 = time1.shape[0]

    time_sum_pre = time0[:x0-x1]
    time_sum_post = time1

    n0_pre = norms0[-1][:x0-x1]
    n0_post = norms0[-1][x0-x1:]

    n1_post = norms1[-1]

    norms_sum_pre = n0_pre
    norms_sum_post = sqrt(n0_post**2 + n1_post**2)

    time_sum = time0
    norms_sum = hstack([squeeze(norms_sum_pre), squeeze(norms_sum_post)])


    # Plot the sum of all norms
    fig = figure()
    ax = fig.gca()

    ax.plot(time0, norms0[-1], label=r"$\| \Phi_m \|$")
    ax.plot(time1, norms1[-1], label=r"$\| \Phi_s \|$")
    ax.plot(time_sum_pre, norms_sum_pre, color=(0,0,0), label=r"$\sqrt{\sum {\| \Phi \|^2}}$")
    ax.plot(time_sum_post, norms_sum_post, color=(0,0,0), label=r"$\sqrt{\sum {\| \Phi \|^2}}$")

    ax.grid(True)
    ax.set_ylim([0,1.1*max(norms0[-1])])
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    legend(loc="outer right")
    ax.set_xlabel(r"Time $t$")
    ax.set_title(r"Sum $\sqrt{\sum {\| \Phi \|^2}}$ for all spawned packets $\Phi$")
    fig.savefig("norms_spawn_sum"+GD.output_format)
    close(fig)


    # Plot the drift of the sum of all norms
    fig = figure()
    ax = fig.gca()

    ax.plot(time_sum, abs(norms_sum[0]-norms_sum))

    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_xlabel(r"Time $t$")
    ax.set_title(r"Drift of the sum of the norms of all spawned packets")
    fig.savefig("norms_spawn_sum_drift"+GD.output_format)
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
