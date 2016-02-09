"""The WaveBlocks Project

Plot the wavefunctions probability densities of the
spawned wavepackets.


@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import angle, conj, real, imag
from matplotlib.pyplot import *

from WaveBlocks import PotentialFactory
from WaveBlocks import IOManager
from WaveBlocks.Plot import plotcf

import GraphicsDefaults as GD


def plot_frames(iom, gid, view=None, plotphase=False, plotcomponents=False, plotabssqr=True, imgsize=(12,9)):
    """Plot the wave function for a series of timesteps.
    :param iom: An I{IOManager} instance providing the spawning simulation data.
    :param gid: The group ID of the data group we plot the frames.
    @keyword view: The aspect ratio.
    @keyword plotphase: Whether to plot the complex phase. (slow)
    @keyword plotcomponents: Whether to plot the real/imaginary parts..
    @keyword plotabssqr: Whether to plot the absolute value squared.
    """
    parameters_s = iom.load_parameters()
    grid = iom.load_grid(blockid="global")

    # For each mother-child spawn try pair
    bidm, bidc = iom.get_block_ids(groupid=gid)

    timegrid = iom.load_wavefunction_timegrid(blockid=bidm)

    for step in timegrid:
        print(" Timestep # " + str(step))

        # Retrieve spawn data for both packets
        values = []
        try:
            # Load data of original packet
            wave = iom.load_wavefunction(timestep=step, blockid=bidm)
            values.append([ wave[j,...] for j in xrange(parameters_s["ncomponents"]) ])

            # Load data of spawned packet
            wave = iom.load_wavefunction(timestep=step, blockid=bidc)
            values.append([ wave[j,...] for j in xrange(parameters_s["ncomponents"]) ])

            have_spawn_data = True
        except ValueError:
            have_spawn_data = False

        # Plot the probability densities projected to the eigenbasis
        fig = figure(figsize=imgsize)

        # Create a bunch of subplots
        axes = []

        for index in xrange(parameters_s["ncomponents"]):
            ax = fig.add_subplot(parameters_s["ncomponents"],1,index+1)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            axes.append(ax)

        # Plot spawned Wavefunctions
        if have_spawn_data is True:
            # For all data blocks
            for colind, values in enumerate(values):
                # For all components of a packet
                for index, component in enumerate(values):
                    # Plot the packet
                    if plotcomponents is True:
                        axes[index].plot(grid, real(component))
                        axes[index].plot(grid, imag(component))
                        axes[index].set_ylabel(r"$\Re \varphi_"+str(index)+r", \Im \varphi_"+str(index)+r"$")
                    if plotabssqr is True:
                        axes[index].plot(grid, component*conj(component), color=colors_mc[colind])
                        axes[index].set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")
                    if plotphase is True:
                        plotcf(grid, angle(component), component*conj(component))
                        axes[index].set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")

        # Set the axis properties
        for ax in axes:
            ax.set_xlabel(r"$x$")

            # Set the aspect window
            if view is not None:
                ax.set_xlim(view[:2])
                ax.set_ylim(view[2:])

        fig.suptitle(r"$\Psi$ at time $"+str(step*parameters_s["dt"])+r"$")
        fig.savefig("wavefunction_spawned_group"+str(gid)+"_"+ (5-len(str(step)))*"0"+str(step) +GD.output_format)
        close(fig)

    print(" Plotting frames finished")


if __name__ == "__main__":
    iom = IOManager()

    # Read file with new simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    # The axes rectangle that is plotted
    view = [-8.5, 8.5, -0.01, 0.6]

    # Colors foth mother and child packet
    colors_mc = ["red", "orange"]

    gids = iom.get_group_ids(exclude=["global"])

    for gid in gids:
        plot_frames(iom, gid, view=view)

    iom.finalize()
