"""The WaveBlocks Project

Plot the wavefunctions probability densities in the eigenbase.
Additionally plot the spawned wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import angle, conj, real, imag, squeeze
from matplotlib.pyplot import *

from WaveBlocks import PotentialFactory
from WaveBlocks import IOManager
from WaveBlocks.Plot import plotcf


def plot_frames(f, view=None, plotphase=False, plotcomponents=False, plotabssqr=True, imgsize=(12,9)):
    """Plot the wave function for a series of timesteps.
    @param f: An I{IOManager} instance providing the simulation data.
    @keyword view: The aspect ratio.
    @keyword plotphase: Whether to plot the complex phase. (slow)
    @keyword plotcomponents: Whether to plot the real/imaginary parts..
    @keyword plotabssqr: Whether to plot the absolute value squared.
    """
    parameters = f.get_parameters()
    
    grid = f.load_grid()

    # Precompute eigenvectors for efficiency
    Potential = PotentialFactory.create_potential(parameters)

    timegrid_m = f.load_wavefunction_timegrid(block=0)
    timegrid_s = f.load_wavefunction_timegrid(block=1)

    for step in timegrid_m:
        print(" Timestep # " + str(step))

        # Retrieve spawn data for both packets
        try:
            wave_m = f.load_wavefunction(timestep=step, block=0)
            values_m = [ squeeze(wave_m[j,...]) for j in xrange(parameters["ncomponents"]) ]
            have_mother_data = True
        except ValueError:
            have_mother_data = False

        # Retrieve spawn data
        try:
            wave_s = f.load_wavefunction(timestep=step, block=1)
            values_s = [ squeeze(wave_s[j,...]) for j in xrange(parameters["ncomponents"]) ]
            have_spawn_data = True
        except ValueError:
            have_spawn_data = False

        # Plot the probability densities projected to the eigenbasis
        fig = figure(figsize=imgsize)

        # Create a bunch of subplots
        axes = []

        for index, component in enumerate(values_m):
            ax = fig.add_subplot(parameters["ncomponents"],1,index+1)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            axes.append(ax)

        # Plot original Wavefunction
        if have_mother_data is True:
            for index, component in enumerate(values_m):
                if plotcomponents is True:
                    axes[index].plot(grid, real(component))
                    axes[index].plot(grid, imag(component))
                    axes[index].set_ylabel(r"$\Re \varphi_"+str(index)+r", \Im \varphi_"+str(index)+r"$")
                if plotabssqr is True:
                    axes[index].plot(grid, component*conj(component), color="black")
                    axes[index].set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")
                if plotphase is True:
                    plotcf(grid, angle(component), component*conj(component))
                    axes[index].set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")

        # Overlay spawned parts
        if have_spawn_data is True:
            for index, component in enumerate(values_s):
                if plotcomponents is True:
                    axes[index].plot(grid, real(component))
                    axes[index].plot(grid, imag(component))
                    axes[index].set_ylabel(r"$\Re \varphi_"+str(index)+r", \Im \varphi_"+str(index)+r"$")
                if plotabssqr is True:
                    axes[index].plot(grid, component*conj(component), color="red")
                    axes[index].set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")
                if plotphase is True:
                    plotcf(grid, angle(component), component*conj(component))
                    axes[index].set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")

        # Set the axis properties
        for index in xrange(len(values_m)):
            axes[index].set_xlabel(r"$x$")

            # Set the aspect window
            if view is not None:
                axes[index].set_xlim(view[:2])
                axes[index].set_ylim(view[2:])

        fig.suptitle(r"$\Psi$ at time $"+str(step*parameters["dt"])+r"$")
        fig.savefig("wavefunction_"+ (5-len(str(step)))*"0"+str(step) +".png")
        close(fig)
        
    print(" Plotting frames finished")


if __name__ == "__main__":
    iom = IOManager()
    
    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()
    
    # The axes rectangle that is plotted
    view = [-8.5, 8.5, -0.01, 0.6]
    
    plot_frames(iom, view=view)
    
    iom.finalize()
