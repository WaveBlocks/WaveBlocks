"""The WaveBlocks Project

Plot the wavefunctions probability densities in the eigenbase.
Additionally plot the spawned wavepackets. The plot is splitted
into 4 subplots corresponding to the left and the right of the barrier.


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
    n = parameters["ncomponents"]
    
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

        # Split the data as necessary
        gl = grid[grid<=X0]
        gr = grid[grid>X0]
        
        yl = ve[0][grid<=X0]
        yr = ve[0][grid>X0]
        
        ysl = ves[0][grid<=X0]
        ysr = ves[0][grid>X0]

        # Plot the probability densities projected to the eigenbasis
        fig = figure(figsize=imgsize)

        # Plot the probability density, left to X0
        ax1 = fig.add_subplot(1,2,1)
        ax1.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        # mother
        plotcf(gl, angle(yl), conj(yl)*yl)
        # spawned
        plot(gl, conj(ysl)*ysl, "-r")

        if view is not None:
            ax1.set_xlim(view[0],0)
            ax1.set_ylim(view[2:4])

        ax1.set_xlabel(r"$x \le 0$")
        ax1.set_ylabel(r"$\langle\varphi |\varphi \rangle$")

        # Plot the probability density, right to X0
        ax2 = fig.add_subplot(1,2,2)
        ax2.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        # mother
        plotcf(gr, angle(yr), conj(yr)*yr)
        # spawned
        plot(gr, conj(ysr)*ysr, "-r")

        if view is not None:
            ax2.set_xlim(0, view[1])
            ax2.set_ylim(view[4:])

        ax2.set_xlabel(r"$x > 0$")
        ax2.set_ylabel(r"$\langle\varphi |\varphi \rangle$")

        fig.suptitle(r"Time $"+str(step*parameters.dt)+r"$")
        fig.savefig("wavepackets_"+ (5-len(str(step)))*"0"+str(step) +".png")
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
    view = [-15, 15, 0.0, 1.5, 0.0, 0.05]

    X0 = 0.0
    
    plot_frames(iom, view=view, plotphase=True, plotcomponents=False, plotabssqr=False)
    
    iom.finalize()
