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


def read_data(f):
    para = f.get_parameters()

    timegrid = f.load_energy_timegrid()
    ekin, epot = f.load_energy()

    ekin = [ ekin[:,c] for c in xrange(para.ncomponents) ]
    epot = [ epot[:,c] for c in xrange(para.ncomponents) ]    

    # Compute the sum of all energies
    ekinsum = reduce(lambda x,y: x+y, ekin)
    epotsum = reduce(lambda x,y: x+y, epot)
    
    ekin.append(ekinsum)
    epot.append(epotsum)

    return (timegrid, ekin, epot)


def plot_energy(timegrid, ekin, epot):
    print("Plotting the energies")

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

    ax.grid(True)
    ax.set_xlabel(r"Timesteps")
    legend(loc="outer right")
    ax.set_title(r"Energies of the wavepacket $\Psi$")

    fig.savefig("energies.png")
    close(fig)

    # Plot the energy drift
    e_orig = (ekin[-1]+epot[-1])[0]
    data = abs(e_orig-(ekin[-1]+epot[-1]))

    fig = figure()
    ax = fig.gca()

    ax.plot(timegrid, data, label=r"$|E_O^0 - \left( E_k^0 + E_p^0 \right) |$")
    ax.grid(True)
    ax.set_xlabel(r"Timesteps")
    ax.set_ylabel(r"$|E_O^0 - \left( E_k^0 + E_p^0 \right) |$")
    ax.set_title(r"Energy drift of the wavepacket $\Psi$")

    fig.savefig("energy_drift.png")
    close(fig)


if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.load_file(filename=sys.argv[1])
    except IndexError:
        iom.load_file()      

    data = read_data(iom)
    plot_energy(*data)

    iom.finalize()
