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


def plot_energy(time0, time1, ekin0, epot0, ekin1, epot1):
    print("Plotting the energies")
    
    # Plot the energies
    fig = figure()

    ax = subplot(2,1,1)

    # Plot the kinetic energy of the individual wave packets
    for i, kin in enumerate(ekin0[:-1]):
        ax.plot(time0, kin, label=r"$E^{kin}_"+str(i)+r"$")

    # Plot the potential energy of the individual wave packets
    for i, pot in enumerate(epot0[:-1]):
        ax.plot(time0, pot, label=r"$E^{pot}_"+str(i)+r"$")

    # Plot the overall energy of all wave packets
    ax.plot(time0, ekin0[-1] + epot0[-1], label=r"$\sum_i E^{kin}_i + \sum_i E^{pot}_i$")

    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y") 
    ax.grid(True)
    ax.set_xlabel(r"Time $t$")
    ax.set_title(r"Energies of the mother packet $\Psi_m$")
    #legend(loc="outer right")

    xlims = ax.get_xlim()

    ax = subplot(2,1,2)

    # Plot the kinetic energy of the individual wave packets
    for i, kin in enumerate(ekin1[:-1]):
        ax.plot(time1, kin, label=r"$E^{kin}_"+str(i)+r"$")

    # Plot the potential energy of the individual wave packets
    for i, pot in enumerate(epot1[:-1]):
        ax.plot(time1, pot, label=r"$E^{pot}_"+str(i)+r"$")

    # Plot the overall energy of all wave packets
    ax.plot(time1, ekin1[-1] + epot1[-1], label=r"$\sum_i E^{kin}_i + \sum_i E^{pot}_i$")

    ax.set_xlim(xlims)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y") 
    ax.grid(True)
    ax.set_xlabel(r"Time $t$")
    ax.set_title(r"Energies of the spawned packet $\Psi_s$")

    fig.savefig("energies_spawn.png")
    close(fig)


    # Data transformation necessary for plotting different parts
    x0 = time0.shape[0]
    x1 = time1.shape[0]
        
    time_sum_pre = time0[:x0-x1]
    time_sum_post = time1

    es0 = ekin0[-1] + epot0[-1]
    es1 = ekin1[-1] + epot1[-1]
    
    e0_pre = es0[:x0-x1]
    e0_post = es0[x0-x1:]

    e1_post = es1

    energies_sum_pre = e0_pre
    energies_sum_post = e0_post + e1_post

    time_sum = time0
    energies_sum = hstack([squeeze(energies_sum_pre), squeeze(energies_sum_post)])


    # Plot the sum of all energies
    fig = figure()
    ax = fig.gca()

    ax.plot(time0, es0, label=r"$E_{\Phi_m}$")
    ax.plot(time1, es1, label=r"$E_{\Phi_s}$")
    ax.plot(time_sum_pre, energies_sum_pre, color=(0,0,0), label=r"$E_{\Phi_m} + E_{\Phi_s}$")
    ax.plot(time_sum_post, energies_sum_post, color=(0,0,0), label=r"$E_{\Phi_m} + E_{\Phi_s}$")

    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    legend(loc="outer right")
    ax.set_xlabel(r"Time $t$")
    ax.set_title(r"Sum $\sum { E_{\Phi} }$ for all spawned packets $\Phi$")
    fig.savefig("energies_spawn_sum.png")
    close(fig)


    # Plot the drift of the sum of all energies
    fig = figure()
    ax = fig.gca()

    ax.plot(time_sum, abs(energies_sum[0]-energies_sum))

    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_xlabel(r"Time $t$")
    ax.set_title(r"Drift of the sum of the energies of all spawned packets")
    fig.savefig("energies_spawn_sum_drift.png")
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
