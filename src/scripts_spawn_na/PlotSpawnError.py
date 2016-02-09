"""The WaveBlocks Project

Plot the spawn error given by $|\Psi_{original}(x)|^2 -\sqrt{\sum_i |\Psi_{{spawn},i}(x)|^2 }$
for each timestep. This is valid for both aposteriori spawn and spawn propagation results.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import conj, sqrt, zeros
from matplotlib.pyplot import *

from WaveBlocks import IOManager

import GraphicsDefaults as GD


def plot_frames(iom_o, iom_s, gid, bid_ref=0, view=None):
    """Plot the wave function for a series of timesteps.
    :param iom_o: An ``IOManager`` instance providing the reference simulation data.
    :param iom_s: An ``IOManager`` instance providing the spawning simulation data.
    :param bid_ref: The block ID of the reference data. Default is data block '0'.
    :param view: The aspect ratio.
    """
    parameters_o = iom_o.load_parameters()
    parameters_s = iom_s.load_parameters()
    grid_o = iom_o.load_grid(blockid="global")

    # For each mother-child spawn try pair
    bidm, bidc = iom_s.get_block_ids(groupid=gid)

    timegrid_o = iom_o.load_wavefunction_timegrid(blockid=bid_ref)

    for step in timegrid_o:
        print(" Timestep # " + str(step))

        # Retrieve reference data
        wave_o = iom_o.load_wavefunction(timestep=step, blockid=bid_ref)
        values_o = [ wave_o[j,...] for j in xrange(parameters_o["ncomponents"]) ]

        # Compute absolute values
        values_o = [ sqrt(conj(item)*item) for item in values_o ]

        # Retrieve spawn data for both packets
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
                values_sum.append( sqrt(reduce(lambda x,y: x+y, [ conj(item[i])*item[i] for item in values_s ])) )

            # Compute the difference to the original
            values_diff = [ abs(item_o - item_s) for item_o, item_s in zip(values_o, values_sum) ]
        else:
            # Return zeros if we did not spawn yet in this timestep
            values_diff = [ zeros(values_o[0].shape) for i in xrange(parameters_o["ncomponents"]) ]

        values_sum = reduce(lambda x,y: x+y, [ item**2 for item in values_diff ])

        # Plot the spawn error for each component
        fig = figure()

        # Create a bunch of subplots, one for each component
        axes = []

        for index, component in enumerate(values_diff):
            ax = fig.add_subplot(parameters_o["ncomponents"],1,index+1)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            axes.append(ax)

        # Plot the difference between the original and spawned Wavefunctions
        for index, values in enumerate(values_diff):
            axes[index].plot(grid_o, values)
            axes[index].set_ylabel(r"Error on $\Phi_"+str(index)+r"$")

        # Set the axis properties
        for index in xrange(len(axes)):
            axes[index].grid(True)
            axes[index].set_xlabel(r"$x$")

            # Set the aspect window
            if view is not None:
                axes[index].set_xlim(view)

        fig.suptitle(r"$|\Psi_{original}(x)|^2 -\sqrt{\sum_i |\Psi_{{spawn},i}(x)|^2 }$ at time $"+str(step*parameters_o["dt"])+r"$")
        fig.savefig("wavefunction_spawn_error_components_group"+str(gid)+ (5-len(str(step)))*"0"+str(step) +GD.output_format)
        close(fig)

        # Plot the overall spawn error
        fig = figure()
        ax = fig.gca()
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

        ax.plot(grid_o, values_sum)

        # Set the aspect window
        if view is not None:
            ax.set_xlim(view)
        ax.grid(True)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"Error on $\Psi$")
        fig.savefig("wavefunction_spawn_error_sum_group"+str(gid)+ (5-len(str(step)))*"0"+str(step) +GD.output_format)
        close(fig)

    print(" Plotting frames finished")




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

    # The axes rectangle that is plotted
    view = [-8.5, 8.5]

    gids = iom_s.get_group_ids(exclude=["global"])

    for gid in gids:
        plot_frames(iom_o, iom_s, gid, view=view)

    iom_o.finalize()
    iom_s.finalize()
