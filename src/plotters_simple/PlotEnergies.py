"""The WaveBlocks Project

Plot the energies of the different wavepackets as well as the sum of all energies.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import abs
from matplotlib.pyplot import *

from WaveBlocks import IOManager
from WaveBlocks.Plot import legend


def read_all_datablocks(iom):
    """Read the data from all blocks that contain any usable data.
    @param iom: An I{IOManager} instance providing the simulation data.
    """
    # Iterate over all blocks and plot their data
    for block in xrange(iom.get_number_blocks()):
        plot_energies(read_data(iom, block=block), index=block)


def read_data(iom, block=0):
    """
    @param iom: An I{IOManager} instance providing the simulation data.
    @keyword block: The data block from which the values are read.
    """
    parameters = iom.get_parameters()
    timegrid = iom.load_energy_timegrid(block=block)
    time = timegrid * parameters["dt"]

    ekin, epot = iom.load_energy(block=block, split=True)

    # Compute the sum of all energies
    ekinsum = reduce(lambda x,y: x+y, ekin)
    epotsum = reduce(lambda x,y: x+y, epot)

    ekin.append(ekinsum)
    epot.append(epotsum)

    return (time, ekin, epot)


def plot_energies(data, index=0):
    print("Plotting the energies of data block "+str(index))

    timegrid, ekin, epot = data

    # Plot the energies
    fig = figure()
    ax = fig.gca()

    # Plot the kinetic energy of the individual wave packets
    for i, kin in enumerate(ekin[:-1]):
        ax.plot(timegrid, kin, label=r"$E^{kin}_"+str(i)+r"$")

    # Plot the potential energy of the individual wave packets
    for i, pot in enumerate(epot[:-1]):
        ax.plot(timegrid, pot, label=r"$E^{pot}_"+str(i)+r"$")

    # Plot the sum of kinetic and potential energy for all wave packets
    for i, (kin, pot) in enumerate(zip(ekin, epot)[:-1]):
        ax.plot(timegrid, kin + pot, label=r"$E^{kin}_"+str(i)+r"+E^{pot}_"+str(i)+r"$")

    # Plot sum of kinetic and sum of potential energy
    ax.plot(timegrid, ekin[-1], label=r"$\sum_i E^{kin}_i$")
    ax.plot(timegrid, epot[-1], label=r"$\sum_i E^{pot}_i$")

    # Plot the overall energy of all wave packets
    ax.plot(timegrid, ekin[-1] + epot[-1], label=r"$\sum_i E^{kin}_i + \sum_i E^{pot}_i$")

    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.grid(True)
    ax.set_xlabel(r"Time $t$")
    legend(loc="outer right")
    ax.set_title(r"Energies of the wavepacket $\Psi$")

    fig.savefig("energies_block"+str(index)+".png")
    close(fig)


    # Plot the energy drift
    e_orig = (ekin[-1]+epot[-1])[0]
    data = abs(e_orig-(ekin[-1]+epot[-1]))

    fig = figure()
    ax = fig.gca()

    ax.plot(timegrid, data, label=r"$|E_O^0 - \left( E_k^0 + E_p^0 \right) |$")

    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.grid(True)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$|E_O^0 - \left( E_k^0 + E_p^0 \right) |$")
    ax.set_title(r"Energy drift of the wavepacket $\Psi$")

    fig.savefig("energy_drift_block"+str(index)+".png")
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
