"""The WaveBlocks Project

Plot the evolution of the relations between the parameters P and Q
homogeneous or inhomogeneous Hagedorn wavepacket during the
time propagation.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import conj, abs
from matplotlib.pyplot import *

from WaveBlocks import IOManager

import GraphicsDefaults as GD


def read_data_homogeneous(iom, blockid=0):
    """
    :param iom: An ``IOManager`` instance providing the simulation data.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_wavepacket_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    Pi = iom.load_wavepacket_parameters(blockid=blockid)

    Phist = [ Pi[:,0] ]
    Qhist = [ Pi[:,1] ]

    return (time, Phist, Qhist)


def read_data_inhomogeneous(iom, blockid=0):
    """
    :param iom: An ``IOManager`` instance providing the simulation data.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_inhomogwavepacket_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    Pi = iom.load_inhomogwavepacket_parameters(blockid=blockid)

    Phist = [ Pi[i][:,0] for i in xrange(parameters["ncomponents"]) ]
    Qhist = [ Pi[i][:,1] for i in xrange(parameters["ncomponents"]) ]

    return (time, Phist, Qhist)


def plot_parameters(blockid, timegrid, Phist, Qhist):
    # Plot the time evolution of the parameters P, Q, S, p and q
    fig = figure(figsize=(12,12))
    ax = fig.gca()

    for ptem, qtem in zip(Phist, Qhist):
        ax.plot(timegrid, abs(conj(qtem)*ptem - conj(ptem)*qtem - 2.0j))

    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$| \overline{Q} P - \overline{P} Q - 2i |$")
    ax.set_title(r"Compatibility condition $\overline{Q} P - \overline{P} Q = 2i$")
    fig.savefig("conjQP-conjPQ_block"+str(blockid)+GD.output_format)
    close(fig)


if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    # Iterate over all blocks
    for blockid in iom.get_block_ids():
        print("Plotting PQ relation of data block '"+str(blockid)+"'")

        # See if we have an inhomogeneous wavepacket in the current data block
        if iom.has_inhomogwavepacket(blockid=blockid):
            data = read_data_inhomogeneous(iom, blockid=blockid)
            plot_parameters(blockid, *data)
        # If not, we test for a homogeneous wavepacket next
        elif iom.has_wavepacket(blockid=blockid):
            data = read_data_homogeneous(iom, blockid=blockid)
            plot_parameters(blockid, *data)
        # There is no wavepacket in the current block
        else:
            print("Warning: No wavepacket found in block '"+str(blockid)+"'!")

    iom.finalize()
