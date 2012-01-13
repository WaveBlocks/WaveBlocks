"""The WaveBlocks Project

Script to plot the support of the transformed quadrature
nodes during the time evolution.

@author: R. Bourquin
@copyright: Copyright (C) 2012 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import unique, array, squeeze
from matplotlib.pyplot import *

from WaveBlocks import IOManager
from WaveBlocks import HomogeneousQuadrature

import GraphicsDefaults as GD


def read_all_datablocks(iom):
    """Read the data from all blocks that contain any usable data.
    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    """
    # Iterate over all blocks and plot their data
    for blockid in iom.get_block_ids():
        if iom.has_wavepacket(blockid=blockid):
            plot_qr_support(read_data(iom, blockid=blockid), index=blockid)
        else:
            print("Warning: No wavepackets found in block '"+str(blockid)+"'!")


def read_data(iom, blockid=0):
    """
    :param iom: An :py:class:`IOManager` instance providing the simulation data.
    :keyword blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_wavepacket_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    bs = iom.load_wavepacket_basissize(blockid=blockid)

    QRules = {}
    for order in unique(bs):
        QRules[order] = HomogeneousQuadrature(order=order)

    data = [ [] for i in xrange(parameters["ncomponents"]) ]
    xpos = []

    for i in timegrid:
        Pi = iom.load_wavepacket_parameters(timestep=i)
        bs = iom.load_wavepacket_basissize(timestep=i)

        # Store the position
        xpos.append(Pi[4])

        # Store the transformed QR nodes
        for n in xrange(parameters["ncomponents"]):
            tn = squeeze(QRules[bs[n]].transform_nodes(Pi, parameters["eps"]))
            data[n].append(tn)

    return (time, array(xpos), array(data))


def plot_qr_support(data, index=0):
    # Unpack
    tg = data[0]
    xp = data[1]
    data = data[-1]
    N = data.shape[0]

    # Plot the QR nodes support versus time
    fig = figure()

    # Subplots for each component
    for n in xrange(0,N):
        ax = fig.add_subplot(N, 1, n+1)
        # Nodes
        ax.plot(tg, data[n,:,:], "b-")
        # Center pos
        ax.plot(tg, xp, "r-")
        ax.grid(True)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\gamma_i$")

    fig.suptitle(r"Support region of quadrature nodes")
    fig.savefig("support_qr_block"+str(index) +GD.output_format)
    close(fig)


    # Plot the QR nodes support versus position
    fig = figure()

    # Subplots for each component
    for n in xrange(0,N):
        ax = fig.add_subplot(1, N, n+1)
        # Nodes
        ax.plot(data[n,:,:], xp, "b-")
        # Center pos
        ax.plot(xp, xp, "r-")
        ax.grid(True)
        ax.set_xlabel(r"$\gamma_i$")
        ax.set_ylabel(r"$x$")
        ax.set_ylim(xp.min(), xp.max())

    fig.suptitle(r"Support region of quadrature nodes")
    fig.savefig("support_qr_block_2"+str(index) +GD.output_format)
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
