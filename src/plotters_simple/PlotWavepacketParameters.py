"""The WaveBlocks Project

Plot the evolution of the parameters Pi_i = (P,Q,S,p,q) of a
homogeneous or inhomogeneous Hagedorn wavepacket during the
time propagation.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import real, imag, abs
from matplotlib.pyplot import *

from WaveBlocks import ComplexMath
from WaveBlocks import IOManager

import GraphicsDefaults as GD


def read_all_datablocks(iom):
    """Read the data from all blocks that contains any usable data.
    :param iom: An ``IOManager`` instance providing the simulation data.
    """
    # Iterate over all blocks and plot their data
    for blockid in iom.get_block_ids():
        if iom.has_wavepacket(blockid=blockid):
            plot_parameters(read_data_homogeneous(iom, blockid=blockid), index=blockid)
        elif iom.has_inhomogwavepacket(blockid=blockid):
            plot_parameters(read_data_inhomogeneous(iom, blockid=blockid), index=blockid)
        else:
            print("Warning: Not plotting wavepacket parameters in block '"+str(blockid)+"'!")


def read_data_homogeneous(iom, blockid=0):
    """
    :param iom: An ``IOManager`` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_wavepacket_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    Pi = iom.load_wavepacket_parameters(blockid=blockid)

    Phist = [ Pi[:,0] ]
    Qhist = [ Pi[:,1] ]
    Shist = [ Pi[:,2] ]
    phist = [ Pi[:,3] ]
    qhist = [ Pi[:,4] ]

    return (time, Phist, Qhist, Shist, phist, qhist)


def read_data_inhomogeneous(iom, blockid=0):
    """
    :param iom: An ``IOManager`` instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_inhomogwavepacket_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    Pi = iom.load_inhomogwavepacket_parameters(blockid=blockid)

    # Number of components
    N = parameters["ncomponents"]

    Phist = [ Pi[i][:,0] for i in xrange(N) ]
    Qhist = [ Pi[i][:,1] for i in xrange(N) ]
    Shist = [ Pi[i][:,2] for i in xrange(N) ]
    phist = [ Pi[i][:,3] for i in xrange(N) ]
    qhist = [ Pi[i][:,4] for i in xrange(N) ]

    return (time, Phist, Qhist, Shist, phist, qhist)


def plot_parameters(data, index=0):
    """Plot the data parameters (P, Q, S, p, q) over time.
    For each new I{index} we start a new figure.
    """
    print("Plotting the parameters of data block '"+str(index)+"'")

    timegrid, Phist, Qhist, Shist, phist, qhist = data

    # Plot the time evolution of the parameters P, Q, S, p and q
    fig = figure(figsize=(12,12))

    ax = fig.add_subplot(4,2,1)
    for item in Phist:
        ax.plot(timegrid, real(item), label=r"$\Re P$")
    ax.grid(True)
    ax.set_title(r"$\Re P$")

    ax = fig.add_subplot(4,2,2)
    for item in Phist:
        ax.plot(timegrid, imag(item), label=r"$\Im P$")
    ax.grid(True)
    ax.set_title(r"$\Im P$")

    ax = fig.add_subplot(4,2,3)
    for item in Qhist:
        ax.plot(timegrid, real(item), label=r"$\Re Q$")
    ax.grid(True)
    ax.set_title(r"$\Re Q$")

    ax = fig.add_subplot(4,2,4)
    for item in Qhist:
        ax.plot(timegrid, imag(item), label=r"$\Im Q$")
    ax.grid(True)
    ax.set_title(r"$\Im Q$")

    ax = fig.add_subplot(4,2,5)
    for item in qhist:
        ax.plot(timegrid, real(item), label=r"$q$")
    ax.grid(True)
    ax.set_title(r"$q$")

    ax = fig.add_subplot(4,2,6)
    for item in phist:
        ax.plot(timegrid, real(item), label=r"$p$")
    ax.grid(True)
    ax.set_title(r"$p$")

    ax = fig.add_subplot(4,2,7)
    for item in Shist:
        ax.plot(timegrid, real(item), label=r"$S$")
    ax.grid(True)
    ax.set_title(r"$S$")

    fig.suptitle("Wavepacket parameters")
    fig.savefig("wavepacket_parameters_block"+str(index)+GD.output_format)
    close(fig)


    # Plot the time evolution of the parameters P, Q, S, p and q
    # This time plot abs/angle instead of real/imag
    fig = figure(figsize=(12,12))

    ax = fig.add_subplot(4,2,1)
    for item in Phist:
        ax.plot(timegrid, abs(item), label=r"$|P|$")
    ax.grid(True)
    ax.set_title(r"$|P|$")

    ax = fig.add_subplot(4,2,2)
    for item in Phist:
        ax.plot(timegrid, ComplexMath.cont_angle(item), label=r"$\arg P$")
    ax.grid(True)
    ax.set_title(r"$\arg P$")

    ax = fig.add_subplot(4,2,3)
    for item in Qhist:
        ax.plot(timegrid, abs(item), label=r"$|Q|$")
    ax.grid(True)
    ax.set_title(r"$|Q|$")

    ax = fig.add_subplot(4,2,4)
    for item in Qhist:
        ax.plot(timegrid, ComplexMath.cont_angle(item), label=r"$\arg Q$")
    ax.grid(True)
    ax.set_title(r"$\arg Q$")

    ax = fig.add_subplot(4,2,5)
    for item in qhist:
        ax.plot(timegrid, real(item), label=r"$q$")
    ax.grid(True)
    ax.set_title(r"$q$")

    ax = fig.add_subplot(4,2,6)
    for item in phist:
        ax.plot(timegrid, real(item), label=r"$p$")
    ax.grid(True)
    ax.set_title(r"$p$")

    ax = fig.add_subplot(4,2,7)
    for item in Shist:
        ax.plot(timegrid, abs(item), label=r"$S$")
    ax.grid(True)
    ax.set_title(r"$S$")

    fig.suptitle("Wavepacket parameters")
    fig.savefig("wavepacket_parameters_abs_ang_block"+str(index)+GD.output_format)
    close(fig)


    # Plot the complex trajectory of the parameters P
    fig = figure()
    ax = fig.gca()
    for item in Phist:
        ax.plot(real(item), imag(item), "-o", label=r"Trajectory $P$")
    ax.grid(True)
    ax.set_title(r"Trajectory of $P$")
    fig.savefig("wavepacket_parameters_trajectoryP_block"+str(index)+GD.output_format)
    close(fig)


    # Plot the complex trajectory of the parameters Q
    fig = figure()
    ax = fig.gca()
    for item in Qhist:
        ax.plot(real(item), imag(item), "-o", label=r"Trajectory $Q$")
    ax.grid(True)
    ax.set_title(r"Trajectory of $Q$")
    fig.savefig("wavepacket_parameters_trajectoryQ_block"+str(index)+GD.output_format)
    close(fig)




if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    # Read the data and plot it, one plot for each data block.
    read_all_datablocks(iom)

    iom.finalize()
