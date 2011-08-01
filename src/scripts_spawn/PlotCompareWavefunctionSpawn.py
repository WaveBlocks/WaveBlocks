"""The WaveBlocks Project

Plot the wavefunctions probability densities of the spawned wavepackets
and compare the some reference data (usually a simulation performed without
spawning).

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import angle, conj, real, imag
from matplotlib.pyplot import *

from WaveBlocks import PotentialFactory
from WaveBlocks import IOManager
from WaveBlocks.Plot import plotcf

import GraphicsDefaults as GD


def plot_frames(data_s, data_o, view=None, plotphase=False, plotcomponents=False, plotabssqr=True, imgsize=(12,9)):
    """Plot the wave function for a series of timesteps.
    @param data_s: An I{IOManager} instance providing the spawning simulation data.
    @param data_o: An I{IOManager} instance providing the reference simulation data.
    @keyword view: The aspect ratio.
    @keyword plotphase: Whether to plot the complex phase. (slow)
    @keyword plotcomponents: Whether to plot the real/imaginary parts..
    @keyword plotabssqr: Whether to plot the absolute value squared.
    """
    parameters_o = data_o.get_parameters()
    parameters_s = data_s.get_parameters()

    grid_o = data_o.load_grid()
    grid_s = data_s.load_grid()

    timegrid_o = data_o.load_wavefunction_timegrid()

    for step in timegrid_o:
        print(" Timestep # " + str(step))

        # Retrieve reference data
        wave_o = data_o.load_wavefunction(timestep=step)
        values_o = [ wave_o[j,...] for j in xrange(parameters_o["ncomponents"]) ]

        # Retrieve spawn data for both packets
        values_s = []
        try:
            for blocknr in xrange(data_s.get_number_blocks()):
                wave = data_s.load_wavefunction(timestep=step, block=blocknr)
                values_s.append( [ wave[j,...] for j in xrange(parameters_s["ncomponents"]) ] )

            have_spawn_data = True
        except ValueError:
            have_spawn_data = False

        # Plot the probability densities projected to the eigenbasis
        fig = figure(figsize=imgsize)

        # Create a bunch of subplots
        axes = []

        for index, component in enumerate(values_o):
            ax = fig.add_subplot(parameters_o["ncomponents"],1,index+1)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            axes.append(ax)

        # Plot reference Wavefunction
        for index, component in enumerate(values_o):
            plotcf(grid_o, angle(component), component*conj(component))
            axes[index].set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")

        # Plot spawned Wavefunctions
        if have_spawn_data is True:
            # For all spawned packets a.k.a data blocks
            for values in values_s:
                # For all components of a packet
                for index, component in enumerate(values):
                    # Plot the packet
                    if plotcomponents is True:
                        axes[index].plot(grid_s, real(component))
                        axes[index].plot(grid_s, imag(component))
                        axes[index].set_ylabel(r"$\Re \varphi_"+str(index)+r", \Im \varphi_"+str(index)+r"$")
                    if plotabssqr is True:
                        axes[index].plot(grid_s, component*conj(component))
                        axes[index].set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")
                    if plotphase is True:
                        plotcf(grid_s, angle(component), component*conj(component))
                        axes[index].set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")

        # Set the axis properties
        for index in xrange(len(values_o)):
            axes[index].set_xlabel(r"$x$")

            # Set the aspect window
            if view is not None:
                axes[index].set_xlim(view[:2])
                axes[index].set_ylim(view[2:])

        fig.suptitle(r"$\Psi$ at time $"+str(step*parameters_o["dt"])+r"$")
        fig.savefig("wavefunction_compare_spawned_"+ (5-len(str(step)))*"0"+str(step) +GD.output_format)
        close(fig)

    print(" Plotting frames finished")


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

    # The axes rectangle that is plotted
    view = [-8.5, 8.5, -0.01, 0.6]

    plot_frames(iom_s, iom_o, view=view)

    iom_s.finalize()
    iom_o.finalize()
