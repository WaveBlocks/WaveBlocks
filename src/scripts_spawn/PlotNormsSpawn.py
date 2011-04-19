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


def read_data(f):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    parameters = f.get_parameters()

    timegrid0 = f.load_norm_timegrid()
    time0 = timegrid0 * parameters.dt
    timegrid1 = f.load_norm_timegrid(block=1)
    time1 = timegrid1 * parameters.dt
    
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
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_xlabel(r"Time $t$")
    ax.set_title(r"Norms of the original wave packet $\Psi$")


    ax = subplot(2,1,2)
    
    # Plot the norms of the individual wavepackets
    for i, datum in enumerate(norms1[:-1]):
        ax.plot(time1, datum, label=r"$\| \Phi_"+str(i)+r" \|$")

    # Plot the sum of all norms
    ax.plot(time1, norms1[-1], color=(1,0,0), label=r"$\sqrt{\sum_i {\| \Phi_i \|^2}}$")

    ax.set_xlim(xlims)
    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_xlabel(r"Time $t$")
    ax.set_title(r"Spawned packet")
    
    fig.savefig("norms_spawn.png")
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
