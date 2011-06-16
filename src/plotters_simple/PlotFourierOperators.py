"""The WaveBlocks Project

Plot the operators used by the Fourier propagation.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import real, imag, arange, hstack
from matplotlib.pyplot import *

from WaveBlocks import PotentialFactory
from WaveBlocks import IOManager
from WaveBlocks.Plot import legend


def read_all_datablocks(iom):
    """Read the data from all blocks that contain any usable data.
    @param iom: An I{IOManager} instance providing the simulation data.
    """
    # Iterate over all blocks and plot their data
    for block in xrange(iom.get_number_blocks()):
        plot_operators(*read_data(iom, block=block), index=block)


def read_data(iom, block=0):
    """
    @param iom: An I{IOManager} instance providing the simulation data.
    @keyword block: The data block from which the values are read.
    """
    parameters = iom.get_parameters()
    # The real space grid
    grid = iom.load_grid(block=block)
    # The Fourier space grid
    omega_1 = arange(0, parameters["ngn"]/2.0)
    omega_2 = arange(-parameters["ngn"]/2.0, 0, 1)
    omega = hstack([omega_2, omega_1])
    # The operators
    opT, opV = iom.load_fourieroperators(block=block)
    # Shift negative frequencies
    opT_1 = opT[:omega_1.shape[0]]
    opT_2 = opT[omega_1.shape[0]:]
    opT = hstack([opT_2, opT_1])

    return (parameters, omega, grid, opT, opV)


def plot_operators(parameters, omega, grid, opT, opV, index=0):
    """Plot the propagation operators T and V
    """
    # Plot the kinetic operator T
    fig = figure()
    ax = fig.gca()

    ax.plot(omega, real(opT), label=r"$\Re T$")
    ax.plot(omega, imag(opT), label=r"$\Im T$")

    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.grid(True)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$T\left(\omega\right)$")
    legend(loc="outer right")
    ax.set_title(r"The kinetic operator T")
    fig.savefig("kinetic_operator_block"+str(index)+".png")
    close(fig)

    # Plot the potential operator V
    fig = figure()
    N = parameters["ncomponents"]
    k = 1

    for i in xrange(N):
        for j in xrange(N):
            ax = fig.add_subplot(N, N, k)
            ax.plot(grid, real(opV[i*N+j]), label=r"$\Re V_{"+str(i)+","+str(j)+r"}$")
            ax.plot(grid, imag(opV[i*N+j]), label=r"$\Im V_{"+str(i)+","+str(j)+r"}$")

            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            ax.grid(True)
            #ax.set_xlim([0, parameters["ngn"]])
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$V_{"+str(i)+","+str(j)+r"}\left(x\right)$")

            k += 1

    fig.savefig("potential_operator_block"+str(index)+".png")
    close(fig)



if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    read_all_datablocks(iom)

    iom.finalize()
