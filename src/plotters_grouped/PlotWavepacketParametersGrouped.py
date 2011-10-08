"""The WaveBlocks Project

Plot the evolution of the parameters Pi_i = (P,Q,S,p,q) of a
any number of homogeneous Hagedorn wavepackets during the
time propagation.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import real, imag, abs, angle
from matplotlib.pyplot import *

from WaveBlocks import ComplexMath
from WaveBlocks import IOManager

import GraphicsDefaults as GD


def read_data(iom, gid):
    parameters = iom.load_parameters()

    data = []

    bids = iom.get_block_ids(groupid=gid)

    # Load the data from each block
    for bid in bids:
        if not iom.has_wavepacket(blockid=bid):
            continue

        timegrid = iom.load_wavepacket_timegrid(blockid=bid)
        time = timegrid * parameters["dt"]

        Pi = iom.load_wavepacket_parameters(blockid=bid)
        Phist = Pi[:,0]
        Qhist = Pi[:,1]
        Shist = Pi[:,2]
        phist = Pi[:,3]
        qhist = Pi[:,4]

        data.append((time, (Phist, Qhist, Shist, phist, qhist)))

    return data


def plot_parameters(gid, data):
    print("Plotting the wavepacket parameters of group '"+str(gid)+"'")

    if len(data) == 0:
        return

    # Plot the time evolution of the parameters P, Q, S, p and q
    fig = figure(figsize=(12,12))
    ax = [ fig.add_subplot(4,2,i) for i in xrange(1,8) ]

    for datum in data:
        grid, PI = datum
        Phist, Qhist, Shist, phist, qhist = PI

        ax[0].plot(grid, real(Phist), label=r"$\Re P$")
        ax[0].grid(True)
        ax[0].set_title(r"$\Re P$")

        ax[1].plot(grid, imag(Phist), label=r"$\Im P$")
        ax[1].grid(True)
        ax[1].set_title(r"$\Im P$")

        ax[2].plot(grid, real(Qhist), label=r"$\Re Q$")
        ax[2].grid(True)
        ax[2].set_title(r"$\Re Q$")

        ax[3].plot(grid, imag(Qhist), label=r"$\Im Q$")
        ax[3].grid(True)
        ax[3].set_title(r"$\Im Q$")

        ax[4].plot(grid, real(qhist), label=r"$q$")
        ax[4].grid(True)
        ax[4].set_title(r"$q$")

        ax[5].plot(grid, real(phist), label=r"$p$")
        ax[5].grid(True)
        ax[5].set_title(r"$p$")

        ax[6].plot(grid, real(Shist), label=r"$S$")
        ax[6].grid(True)
        ax[6].set_title(r"$S$")

    fig.suptitle("Wavepacket parameters")
    fig.savefig("wavepacket_parameters_group"+str(gid)+GD.output_format)
    close(fig)


    # Plot the time evolution of the parameters P, Q, S, p and q
    # This time plot abs/angle instead of real/imag
    fig = figure(figsize=(12,12))
    ax = [ fig.add_subplot(4,2,i) for i in xrange(1,8) ]

    for datum in data:
        grid, PI = datum
        Phist, Qhist, Shist, phist, qhist = PI

        ax[0].plot(grid, abs(Phist), label=r"$|P|$")
        ax[0].grid(True)
        ax[0].set_title(r"$|P|$")

        ax[1].plot(grid, ComplexMath.cont_angle(Phist), label=r"$\arg P$")
        ax[1].grid(True)
        ax[1].set_title(r"$\arg P$")

        ax[2].plot(grid, abs(Qhist), label=r"$|Q|$")
        ax[2].grid(True)
        ax[2].set_title(r"$|Q|$")

        ax[3].plot(grid, ComplexMath.cont_angle(Qhist), label=r"$\arg Q$")
        ax[3].grid(True)
        ax[3].set_title(r"$\arg Q$")

        ax[4].plot(grid, real(qhist), label=r"$q$")
        ax[4].grid(True)
        ax[4].set_title(r"$q$")

        ax[5].plot(grid, real(phist), label=r"$p$")
        ax[5].grid(True)
        ax[5].set_title(r"$p$")

        ax[6].plot(grid, real(Shist), label=r"$S$")
        ax[6].grid(True)
        ax[6].set_title(r"$S$")

    fig.suptitle("Wavepacket parameters")
    fig.savefig("wavepacket_parameters_absang_group"+str(gid)+GD.output_format)
    close(fig)




if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    gids = iom.get_group_ids(exclude=["global"])

    for gid in gids:
        data = read_data(iom, gid)
        plot_parameters(gid, data)

    iom.finalize()
