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


def compute_data(data_s, data_o):
    """Plot the wave function for a series of timesteps.
    @param data_s: An I{IOManager} instance providing the spawning simulation data.
    @param data_o: An I{IOManager} instance providing the reference simulation data.
    @param which_norm: Decide which norm is used to quantify the spawn error, can be 'L2' or 'max'.
    """
    parameters_o = data_o.get_parameters()
    parameters_s = data_s.get_parameters()

    grid_o = data_o.load_grid()

    timegrid_o = data_o.load_wavefunction_timegrid()

    WF = WaveFunction(parameters_o)
    WF.set_grid(grid_o)

    norms_L2 = []
    norms_max = []

    for step in timegrid_o:
        print(" Timestep # " + str(step))

        # Retrieve reference data
        wave_o = data_o.load_wavefunction(timestep=step)
        values_o = [ wave_o[j,...] for j in xrange(parameters_o["ncomponents"]) ]

        # Compute absolute values
        values_o = [ sqrt(conj(item)*item) for item in values_o ]

        # Retrieve spawn data for both packets
        values_s = []
        try:
            for blocknr in xrange(data_s.get_number_blocks()):
                wave = data_s.load_wavefunction(timestep=step, block=blocknr)
                values_s.append( [ wave[j,...] for j in xrange(parameters_s["ncomponents"]) ] )

            have_spawn_data = True
        except ValueError:
            have_spawn_data = False

        if have_spawn_data is True:
            # Sum up the spawned parts
            values_sum = []
            for i in xrange(parameters_o["ncomponents"]):
                values_sum.append( sqrt(reduce(lambda x,y: x+y, [ conj(item[i])*item[i] for item in values_s ])) )

            # Compute the difference to the original
            values_diff = [ item_o - item_s for item_o, item_s in zip(values_o, values_sum) ]
        else:
            # Return zeros if we did not spawn yet in this timestep
            values_diff = [ zeros(values_o[0].shape) for i in xrange(parameters_o["ncomponents"]) ]

        # Compute the L^2 norm
        WF.set_values(values_diff)
        curnorm_L2 = WF.get_norm()

        # Compute the max norm
        curnorm_max = [ max(abs(item)) for item in values_diff ]

        print(" at time " + str(step*parameters_o["dt"]) + " the error in L^2 norm is " + str(curnorm_L2))
        norms_L2.append(curnorm_L2)
        norms_max.append(curnorm_max)

    return (timegrid_o*parameters_o["dt"], array(norms_L2), array(norms_max))


def plot_data(timegrid, norms_L2, norms_max):

    # Plot the L^2 norm of the spawning error
    fig = figure()
    ax = fig.gca()
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

    # Plot the difference between the original and spawned Wavefunctions in L^2 norm
    for i in xrange(norms_L2.shape[1]):
        ax.semilogy(timegrid, norms_L2[:,i])

    # Set the axis properties
    ax.grid(True)
    ax.set_ylim([10**-6, 10**0])
    ax.set_xlabel(r"$x$")

    fig.suptitle(r"$\| |\Psi_{original}(x)|^2 -\sqrt{\sum_i |\Psi_{{spawn},i}(x)|^2 } \|_{L^2}$")
    fig.savefig("spawn_error_L2-norm"+GD.output_format)
    close(fig)


    # Plot the max norm of the spawning error
    fig = figure()
    ax = fig.gca()
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

    # Plot the difference between the original and spawned Wavefunctions in max norm
    for i in xrange(norms_max.shape[1]):
        ax.semilogy(timegrid, norms_max[:,i])

    # Set the axis properties
    ax.grid(True)
    ax.set_ylim([10**-6, 10**0])
    ax.set_xlabel(r"$x$")

    fig.suptitle(r"$\| |\Psi_{original}(x)|^2 -\sqrt{\sum_i |\Psi_{{spawn},i}(x)|^2 } \|_{max}$")
    fig.savefig("spawn_error_max-norm"+GD.output_format)
    close(fig)


    # Plot the L^2 and max norm of the spawning error
    fig = figure()
    ax = fig.gca()
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

    # Plot the difference between the original and spawned Wavefunctions in both norms
    for i in xrange(norms_L2.shape[1]):
        ax.semilogy(timegrid, norms_L2[:,i], label=r"$L^2$ norm")
        ax.semilogy(timegrid, norms_max[:,i], label=r"$L^\infty$ norm")

    # Set the axis properties
    ax.grid(True)
    ax.set_ylim([10**-6, 10**0])
    ax.set_xlabel(r"$x$")
    ax.legend()
    fig.suptitle(r"$\| |\Psi_{original}(x)|^2 -\sqrt{\sum_i |\Psi_{{spawn},i}(x)|^2 } \|$")
    fig.savefig("spawn_error_norm"+GD.output_format)
    close(fig)




if __name__ == "__main__":
    iom_s = IOManager()
    iom_o = IOManager()

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

    plot_data(*compute_data(iom_s, iom_o))

    iom_s.finalize()
    iom_o.finalize()
