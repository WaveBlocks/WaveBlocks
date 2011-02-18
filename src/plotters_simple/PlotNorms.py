"""The WaveBlocks Project

Plot the norms of the different wavepackets as well as the sum of all norms.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from matplotlib.pyplot import *

from WaveBlocks import IOManager
from WaveBlocks.Plot import legend


def read_data(f):
    para = f.get_parameters()

    timegrid = f.load_norm_timegrid()
    
    norms = f.load_norm()
    norms = [ norms[:,c] for c in xrange(para.ncomponents) ]

    normsum = [ item**2 for item in norms ]
    normsum = reduce(lambda x,y: x+y, normsum)
    norms.append(normsum)
    
    return (timegrid, norms)


def plot_norms(timegrid, data):
    print("Plotting the norms")
    
    fig = figure()
    ax = fig.gca()
    
    # Plot the norms of the individual wavepackets
    for i, datum in enumerate(data[:-1]):
        label_i = r"$\| \Phi_"+str(i)+r" \|$"
        ax.plot(timegrid, datum, label=label_i)

    # Plot the sum of all norms
    ax.plot(timegrid, data[-1], color=(1,0,0), label=r"${\sqrt{\sum_i {\| \Phi_i \|^2}}}$")

    ax.set_title(r"Norms of $\Psi$")
    legend(loc="outer right")
    ax.set_xlabel(r"Timesteps")
    ax.grid(True)

    fig.savefig("norms.png")
    close(fig)


    # Plot the difference from the theoretical norm
    fig = figure()
    ax = fig.gca()

    ax.plot(timegrid, abs(data[-1][0] - data[-1]), label=r"$\|\Psi\|_0 - \|\Psi\|_t$")
    ax.grid(True)
    ax.set_title(r"Drift of $\| \Psi \|$")
    legend(loc="outer right")
    ax.set_xlabel(r"Timesteps")
    ax.set_ylabel(r"$$\|\Psi\|_0 - \|\Psi\|_t$$")

    fig.savefig("norms_drift.png")
    close(fig)


if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.load_file(filename=sys.argv[1])
    except IndexError:
        iom.load_file()      
    
    data = read_data(iom)
    plot_norms(*data)
    
    iom.finalize()
