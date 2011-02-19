"""The WaveBlocks Project

Plot the energies of the different wave packets as well as the
overall energy. Load the overall energy from the simulation
data file.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import *
from matplotlib.pyplot import *

from WaveBlocks import GlobalDefaults
from WaveBlocks import IOManager
from WaveBlocks.Plot import legend


def load_data(iom):
    # Load the data
    parameters = iom.get_parameters()
    
    timegrid = iom.load_energy_timegrid()
    times = timegrid * parameters.dt

    # Load kinetic, potential and total energies
    ekin, epot = iom.load_energy()
    etot = iom.load_energy_total()

    # Some data transformation
    ekin = [ ekin[:,c] for c in xrange(parameters.ncomponents) ]
    epot = [ epot[:,c] for c in xrange(parameters.ncomponents) ]

    # Calculate the sum of all energies
    ekinsum = reduce(lambda x,y: x+y, ekin)
    epotsum = reduce(lambda x,y: x+y, epot)
    
    ekin.append(ekinsum)
    epot.append(epotsum)
    
    return times, ekin, epot, etot


def plot_energy(times, ekin, epot, etot):
    print("Plotting the energies")
    
    # Plot the energies
    fig = figure()
    ax = fig.gca()

    # Plot the kinetic energy of the individual wave packets
    for i, kin in enumerate(ekin[:-1]):
        ax.plot(times, kin,  label=r"$E^{kin}_"+str(i)+r"$")

    # Plot the potential energy of the individual wave packets
    for i, pot in enumerate(epot[:-1]):
        ax.plot(times, pot, label=r"$E^{pot}_"+str(i)+r"$")

    # Plot the sum of kinetic and potential energy for all wave packets
    for i, (kin, pot) in enumerate(zip(ekin, epot)[:-1]):
        ax.plot(times, kin + pot, label=r"$E^{kin}_"+str(i)+r"+E^{pot}_"+str(i)+r"$")
    
    # Plot sum of kinetic and sum of potential energy
    ax.plot(times, ekin[-1], label=r"$\sum_i E^{kin}_i$")
    ax.plot(times, epot[-1], label=r"$\sum_i E^{pot}_i$")
    
    # Plot the overall energy of all wave packets
    ax.plot(times, etot, label=r"$\sum_i E^{kin}_i + \sum_i E^{pot}_i$")

    ax.grid(True)
    ax.set_xlabel(r"Time")
    ax.set_title(r"Energies of the wavepacket $\Psi$")
    legend(loc="outer right")
    fig.savefig("energies_etot_canonical.png")
    close(fig)


    # Plot the difference between the computation of the overall energy
    # done in the eigenbasis and the canonical basis.
    etot_eig = ekin[-1] + epot[-1]    

    fig = figure()
    ax = fig.gca()
    ax.plot(times, squeeze(etot) - etot_eig, label=r"etot eig - etot can")
    fig.savefig("etot_canonical_diff.png")
    close(fig)


    # Plot the energy drift over time
    e_orig = etot[0]

    fig = figure()
    ax = fig.gca()
    ax.plot(times, abs(e_orig-(ekin[-1]+epot[-1])), label=r"$|E_O^0 - \left( E_k^0 + E_p^0 \right) |$")
    ax.grid(True)
    ax.set_xlabel(r"Timesteps")
    ax.set_ylabel(r"$|E_O^0 - \left( E_k^0 + E_p^0 \right) |$")
    ax.set_title(r"Energy drift of the wave packet $\Psi$")
    fig.savefig("energies_etot_canonical_drift.png")
    close(fig)


if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    data = load_data(iom)
    plot_energy(*data)

    iom.finalize()
