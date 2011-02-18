"""The WaveBlocks Project

Plot the wavefunctions probability densities in the eigenbase.

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
        # todo: improve this:
        if parameters.algorithm == "fourier":
            ve = Potential.project_to_eigen(grid, values, eigenvectors)
        else:
            ve = values

        # plot the probability densities projected to the eigenbase
        fig = figure(figsize=imgsize)
        
        for index, component in enumerate(ve):
            ax = fig.add_subplot(parameters.ncomponents,1,index+1)

            if plotcomponents is True:
                ax.plot(grid, real(component))
                ax.plot(grid, imag(component))
                ax.set_ylabel(r"$\Re \varphi_"+str(index)+r", \Im \varphi_"+str(index)+r"$")
            if plotabssqr is True:
                ax.plot(grid, component*conj(component))
                ax.set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")
            if plotphase is True:
                plotcf(grid, angle(component), component*conj(component))
                ax.set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")
            
            ax.set_xlabel(r"$x$")

            # Set the aspect window
            if view is not None:
                ax.set_xlim(view[:2])
                ax.set_ylim(view[2:])
            
        fig.suptitle(r"$\Psi$ at time $"+str(step*parameters.dt)+r"$")
        fig.savefig("wavefunction_"+ (5-len(str(step)))*"0"+str(step) +".png")
        close(fig)
        
    print(" Plotting frames finished")


if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.load_file(filename=sys.argv[1])
    except IndexError:
        iom.load_file()      

    # The axes rectangle that is plotted
    view = [-1.5, 1.5, -0.1, 3.5]

    plot_frames(iom, view=view, plotphase=True, plotcomponents=False, plotabssqr=False)

    iom.finalize()
