"""The WaveBlocks Project

Plot the wavefunctions probability densities in the eigenbase.
Additionally plot the spawned wavepackets.

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


def plot_frames(f, view=None, plotphase=True, plotcomponents=False, plotabssqr=False, imgsize=(12,9)):
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
    eigenvectors = Potential.evaluate_eigenvectors_at(grid)

    timegrid = f.load_wavefunction_timegrid()

    for step in timegrid:
        print(" Timestep # " + str(step))

        wave = f.load_wavefunction(timestep=step)
        values = [ wave[j,...] for j in xrange(parameters.ncomponents) ]

        # Transform the values to the eigenbasis
        # TODO: improve this:
        if parameters.algorithm == "fourier":
            ve = Potential.project_to_eigen(grid, values, eigenvectors)
        else:
            ve = values

        # Retrieve spawn data
        waves = f.load_wavefunction(timestep=step, block=1)
        valuess = [ waves[j,...] for j in xrange(parameters.ncomponents) ]

        # Transform the values to the eigenbasis
        # TODO: improve this:
        if parameters.algorithm == "fourier":
            ves = Potential.project_to_eigen(grid, valuess, eigenvectors)
        else:
            ves = valuess


        # Plot the probability densities projected to the eigenbasis
        fig = figure(figsize=imgsize)

        # Create a bunch of subplots
        axes = []

        for index, component in enumerate(ve):
            ax = fig.add_subplot(parameters.ncomponents,1,index+1)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            axes.append(ax)

        # Plot original Wavefunction
        for index, component in enumerate(ve):
            if plotcomponents is True:
                axes[index].plot(grid, real(component))
                axes[index].plot(grid, imag(component))
                axes[index].set_ylabel(r"$\Re \varphi_"+str(index)+r", \Im \varphi_"+str(index)+r"$")
            if plotabssqr is True:
                axes[index].plot(grid, component*conj(component))
                axes[index].set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")
            if plotphase is True:
                plotcf(grid, angle(component), component*conj(component))
                axes[index].set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")

        # Overlay spawned parts
        for index, component in enumerate(ves):
            axes[index].plot(grid, component*conj(component), "-r")
            axes[index].set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")

        # Set the axis properties
        for index in xrange(len(ve)):
            axes[index].set_xlabel(r"$x$")

            # Set the aspect window
            if view is not None:
                axes[index].set_xlim(view[:2])
                axes[index].set_ylim(view[2:])

        fig.suptitle(r"$\Psi$ at time $"+str(step*parameters.dt)+r"$")
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
    view = [-10.5, 10.5, -0.1, 3.5]
    
    plot_frames(iom, view=view, plotphase=True, plotcomponents=False, plotabssqr=False)
    
    iom.finalize()
