"""The WaveBlocks Project

Plot the spawn error given by $|\Psi_{original}(x)|^2 -\sqrt{\sum_i |\Psi_{{spawn},i}(x)|^2 }$
for each timestep. This is valid for both aposteriori spawn and spawn propagation results.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import conj, sqrt, zeros, array
from matplotlib.pyplot import *

from WaveBlocks import IOManager
from WaveBlocks import WaveFunction

import GraphicsDefaults as GD


def read_data(iom_o, iom_s, gid, bid_ref=0):
    """Plot the wave function for a series of timesteps.
    @param iom_s: An I{IOManager} instance providing the spawning simulation data.
    @param iom_o: An I{IOManager} instance providing the reference simulation data.
    @keyword bid_ref: The block ID of the reference data. Default is data block '0'.
    """
    parameters_o = iom_o.load_parameters()
    parameters_s = iom_s.load_parameters()

    # For each mother-child spawn try pair
    bidm, bidc = iom_s.get_block_ids(groupid=gid)

    # Read original data from first block
    grid_o = iom_o.load_grid(blockid="global")
    timegrid_o = iom_o.load_wavefunction_timegrid(blockid=bid_ref)

    WF = WaveFunction(parameters_o)
    WF.set_grid(grid_o)

    norms_L2 = []
    norms_max = []

    for step in timegrid_o:
        print(" Timestep # " + str(step))

        # Retrieve original reference data
        wave_o = iom_o.load_wavefunction(timestep=step, blockid=bid_ref)
        values_o = [ wave_o[j,...] for j in xrange(parameters_o["ncomponents"]) ]

        # Compute absolute values, assume the data were stored in the eigenbasis
        values_o = [ sqrt(conj(item)*item) for item in values_o ]

        # Retrieve spawn data for mother and child packets
        values_s = []
        try:
            # Load data of original packet
            wave = iom_s.load_wavefunction(timestep=step, blockid=bidm)
            values_s.append([ wave[j,...] for j in xrange(parameters_s["ncomponents"]) ])

            # Load data of spawned packet
            wave = iom_s.load_wavefunction(timestep=step, blockid=bidc)
            values_s.append([ wave[j,...] for j in xrange(parameters_s["ncomponents"]) ])

            have_spawn_data = True
        except ValueError:
            have_spawn_data = False

        if have_spawn_data is True:
            # Sum up the spawned parts
            values_sum = []
            for i in xrange(parameters_o["ncomponents"]):
                values_sum.append(sqrt(reduce(lambda x,y: x+y, [ conj(item[i])*item[i] for item in values_s ])))

            # Compute the difference to the original
            values_diff = [ item_o - item_s for item_o, item_s in zip(values_o, values_sum) ]
        else:
            # Return zeros if we did not spawn yet in this timestep
            values_diff = [ zeros(values_o[0].shape) for i in xrange(parameters_o["ncomponents"]) ]

        # Compute the L^2 norm
        WF.set_values(values_diff)
        curnorm_L2 = list(WF.get_norm())
        curnorm_L2.append(WF.get_norm(summed=True))

        # Compute the max norm
        curnorm_max = [ max(abs(item)) for item in values_diff ]
        curnorm_max.append(max(curnorm_max))

        print(" at time " + str(step*parameters_o["dt"]) + " the error in L^2 norm is " + str(curnorm_L2))
        norms_L2.append(curnorm_L2)
        norms_max.append(curnorm_max)

    return (timegrid_o*parameters_o["dt"], array(norms_L2), array(norms_max))


def plot_data(gid, data):
    # Unpack data
    timegrid, norms_L2, norms_max = data

    # Plot the L^2 norm of the spawning error component wise
    fig = figure()

    for i in xrange(norms_L2.shape[1]-1):
        ax = fig.add_subplot(norms_L2.shape[1]-1,1,i+1)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

        ax.semilogy(timegrid, norms_L2[:,i], label=r"$\|\Phi^O_"+str(i)+r" - \Phi^S_"+str(i)+r"\|_{L^2}$")

        # Set the axis properties
        ax.grid(True)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\Phi_"+str(i)+r"$")

    fig.suptitle(r"$\| |\Psi_{original}(x)|^2 -\sqrt{\sum_i |\Psi_{{spawn},i}(x)|^2 } \|_{L^2}$")
    fig.savefig("spawn_error_component_L2norm_group"+str(gid)+GD.output_format)
    close(fig)


    # Plot the L^2 norm of the spawning error
    fig = figure()
    ax = fig.gca()
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

    for i in xrange(norms_L2.shape[1]-1):
        ax.semilogy(timegrid, norms_L2[:,i], label=r"$\|\Phi^O_"+str(i)+r" - \Phi^S_"+str(i)+r"\|_{L^2}$")
    ax.semilogy(timegrid, norms_L2[:,-1], label=r"$\|\Psi^O - \Psi^S\|_{L^2}$")

    # Set the axis properties
    ax.grid(True)
    ax.set_xlabel(r"$t$")
    ax.legend(loc="upper left")

    fig.suptitle(r"$\| |\Psi_{original}(x)|^2 -\sqrt{\sum_i |\Psi_{{spawn},i}(x)|^2 } \|_{L^2}$")
    fig.savefig("spawn_error_sum_L2norm"+str(gid)+GD.output_format)
    close(fig)


    # Plot the max norm of the spawning error component wise
    fig = figure()

    for i in xrange(norms_max.shape[1]-1):
        ax = fig.add_subplot(norms_max.shape[1]-1,1,i+1)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

        ax.semilogy(timegrid, norms_max[:,i], label=r"$\|\Phi^O_"+str(i)+r" - \Phi^S_"+str(i)+r"\|_{max}$")

        # Set the axis properties
        ax.grid(True)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\Phi_"+str(i)+r"$")

    fig.suptitle(r"$\| |\Psi_{original}(x)|^2 -\sqrt{\sum_i |\Psi_{{spawn},i}(x)|^2 } \|_{max}$")
    fig.savefig("spawn_error_component_maxnorm"+str(gid)+GD.output_format)
    close(fig)


    # Plot the max norm of the spawning error
    fig = figure()
    ax = fig.gca()
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

    for i in xrange(norms_max.shape[1]-1):
        ax.semilogy(timegrid, norms_max[:,i], label=r"$\|\Phi^O_"+str(i)+r" - \Phi^S_"+str(i)+r"\|_{max}$")
    ax.semilogy(timegrid, norms_max[:,-1], label=r"$\|\Psi^O - \Psi^S\|_{max}$")

    # Set the axis properties
    ax.grid(True)
    ax.set_xlabel(r"$t$")
    ax.legend(loc="upper left")

    fig.suptitle(r"$\| |\Psi_{original}(x)|^2 -\sqrt{\sum_i |\Psi_{{spawn},i}(x)|^2 } \|_{max}$")
    fig.savefig("spawn_error_sum_maxnorm"+str(gid)+GD.output_format)
    close(fig)


    # Plot the difference between the original and spawned Wavefunctions in both norms
    # Plot the max norm of the spawning error component wise
    fig = figure()

    for i in xrange(norms_max.shape[1]-1):
        ax = fig.add_subplot(norms_max.shape[1]-1,1,i+1)
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

        ax.semilogy(timegrid, norms_L2[:,i], label=r"$\|\Phi^O_"+str(i)+r" - \Phi^S_"+str(i)+r"\|_{L^2}$")
        ax.semilogy(timegrid, norms_max[:,i], label=r"$\|\Phi^O_"+str(i)+r" - \Phi^S_"+str(i)+r"\|_{max}$")

        # Set the axis properties
        ax.grid(True)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\Phi_"+str(i)+r"$")

    ax.legend(loc="upper left")

    fig.suptitle(r"$\| |\Psi_{original}(x)|^2 -\sqrt{\sum_i |\Psi_{{spawn},i}(x)|^2 } \|$")
    fig.savefig("spawn_error_component_norms"+str(gid)+GD.output_format)
    close(fig)


    # Plot the max norm of the spawning error
    fig = figure()
    ax = fig.gca()
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

    for i in xrange(norms_max.shape[1]-1):
        ax.semilogy(timegrid, norms_L2[:,i], label=r"$\|\Phi^O_"+str(i)+r" - \Phi^S_"+str(i)+r"\|_{L^2}$")
        ax.semilogy(timegrid, norms_max[:,i], label=r"$\|\Phi^O_"+str(i)+r" - \Phi^S_"+str(i)+r"\|_{max}$")
    ax.semilogy(timegrid, norms_L2[:,-1], label=r"$\|\Psi^O - \Psi^S\|_{L^2}$")
    ax.semilogy(timegrid, norms_max[:,-1], label=r"$\|\Psi^O - \Psi^S\|_{max}$")

    # Set the axis properties
    ax.grid(True)
    ax.set_xlabel(r"$t$")
    ax.legend(loc="upper left")

    fig.suptitle(r"$\| |\Psi_{original}(x)|^2 -\sqrt{\sum_i |\Psi_{{spawn},i}(x)|^2 } \|_{max}$")
    fig.savefig("spawn_error_sum_norms"+str(gid)+GD.output_format)
    close(fig)




if __name__ == "__main__":
    iom_o = IOManager()
    iom_s = IOManager()

    # NOTE
    #
    # first cmd-line data file is spawning data
    # second cmd-line data file is reference data

    # Read file with new simulation data
    try:
        iom_s.open_file(filename=sys.argv[1])
    except IndexError:
        iom_s.open_file()

    # Read file with original reference simulation data
    try:
        iom_o.open_file(filename=sys.argv[2])
    except IndexError:
        iom_o.open_file()

    gids = iom_s.get_group_ids(exclude=["global"])

    for gid in gids:
        data = read_data(iom_o, iom_s, gid)
        plot_data(gid, data)

    iom_o.finalize()
    iom_s.finalize()
