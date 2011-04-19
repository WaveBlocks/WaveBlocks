"""The WaveBlocks Project

Plot the energies of the different wavepackets as well as the
sum of all energies for the original and spawned wavepacket.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import *
from scipy import *
from matplotlib.pyplot import *

from WaveBlocks import IOManager
from WaveBlocks.Plot import legend


def read_data(f):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    params = f.get_parameters()    
    timegrid0 = f.load_energy_timegrid()
    time0 = timegrid0 * params.dt
    timegrid1 = f.load_energy_timegrid(block=1)
    time1 = timegrid1 * params.dt

    # Load data of original packet
    ekin0, epot0 = f.load_energy(split=True)

    # Compute the sum of all energies
    ekinsum0 = reduce(lambda x,y: x+y, ekin0)
    epotsum0 = reduce(lambda x,y: x+y, epot0)
    
    ekin0.append(ekinsum0)
    epot0.append(epotsum0)

    # Load data of spawned packet
    ekin1, epot1 = f.load_energy(split=True, block=1)

    # Compute the sum of all energies
    ekinsum1 = reduce(lambda x,y: x+y, ekin1)
    epotsum1 = reduce(lambda x,y: x+y, epot1)
    
    ekin1.append(ekinsum1)
    epot1.append(epotsum1)
    
    return (time0, time1, ekin0, epot0, ekin1, epot1)


def plot_energy(timegrid0, timegrid1, ekin0, epot0, ekin1, epot1):
    print("Plotting the energies")
    
    # Plot the energies
    fig = figure()

    ax = subplot(2,1,1)

    # Plot the kinetic energy of the individual wave packets
    for i, kin in enumerate(ekin0[:-1]):
        ax.plot(timegrid0, kin, label=r"$E^{kin}_"+str(i)+r"$")

    # Plot the potential energy of the individual wave packets
    for i, pot in enumerate(epot0[:-1]):
        ax.plot(timegrid0, pot, label=r"$E^{pot}_"+str(i)+r"$")

    # Plot the overall energy of all wave packets
    ax.plot(timegrid0, ekin0[-1] + epot0[-1], label=r"$\sum_i E^{kin}_i + \sum_i E^{pot}_i$")

    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y") 
    ax.grid(True)
    ax.set_xlabel(r"Time $t$")
    ax.set_title(r"Energies of the original wave packet $\Psi$")
    #legend(loc="outer right")

    xlims = ax.get_xlim()

    ax = subplot(2,1,2)

    # Plot the kinetic energy of the individual wave packets
    for i, kin in enumerate(ekin1[:-1]):
        ax.plot(timegrid1, kin, label=r"$E^{kin}_"+str(i)+r"$")

    # Plot the potential energy of the individual wave packets
    for i, pot in enumerate(epot1[:-1]):
        ax.plot(timegrid1, pot, label=r"$E^{pot}_"+str(i)+r"$")

    # Plot the overall energy of all wave packets
    ax.plot(timegrid1, ekin1[-1] + epot1[-1], label=r"$\sum_i E^{kin}_i + \sum_i E^{pot}_i$")

    ax.set_xlim(xlims)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y") 
    ax.grid(True)
    ax.set_xlabel(r"Time $t$")
    ax.set_title(r"Spawned packet")

    fig.savefig("energies_spawn.png")
    close(fig)



if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()      

    data = read_data(iom)
    plot_energy(*data)

    iom.finalize()
