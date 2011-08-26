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


def read_data_spawn(f, assume_duplicate_mother=False):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    @keyword assume_duplicate_mother: Parameter to tell the code to leave out
    every second data block and only take blocks [0, 1, 3, 5, 7, ...]. This
    is usefull because in aposteriori spawning we have to store clones of
    the mother packet.
    """
    parameters = f.get_parameters()
    ndb = f.get_number_blocks()

    timegrids = []
    AllPA = []

    if assume_duplicate_mother is True:
        blocks = [0] + range(1, ndb, 2)
    else:
        blocks = range(ndb)

    # Load the data from each block
    for block in blocks:
        timegrids.append(parameters["dt"] * f.load_wavepacket_timegrid(block=block))

        Pi = f.load_wavepacket_parameters(block=block)
        Phist = Pi[:,0]
        Qhist = Pi[:,1]
        Shist = Pi[:,2]
        phist = Pi[:,3]
        qhist = Pi[:,4]

        AllPA.append([Phist, Qhist, Shist, phist, qhist])

    return timegrids, AllPA


def plot_parameters_spawn(timegrids, AllPA):
    # Plot the time evolution of the parameters P, Q, S, p and q
    fig = figure(figsize=(12,12))
    ax = [ fig.add_subplot(4,2,i) for i in xrange(1,8) ]

    for index in xrange(len(timegrids)):
        grid = timegrids[index]
        Phist, Qhist, Shist, phist, qhist = AllPA[index]

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

    fig.suptitle("Wavepacket (spawned) parameters")
    fig.savefig("wavepacket_parameters_spawned"+GD.output_format)
    close(fig)


    # Plot the time evolution of the parameters P, Q, S, p and q
    # This time plot abs/angle instead of real/imag
    fig = figure(figsize=(12,12))
    ax = [ fig.add_subplot(4,2,i) for i in xrange(1,8) ]

    for index in xrange(len(timegrids)):
        grid = timegrids[index]
        Phist, Qhist, Shist, phist, qhist = AllPA[index]

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

    fig.suptitle("Wavepacket (spawned) parameters")
    fig.savefig("wavepacket_parameters_abs_ang_spawned"+GD.output_format)
    close(fig)




if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    parameters = iom.get_parameters()

    if parameters["algorithm"] == "spawning_apost_na":
        plot_parameters_spawn(*read_data_spawn(iom, assume_duplicate_mother=True))
    else:
        plot_parameters_spawn(*read_data_spawn(iom))

    iom.finalize()
