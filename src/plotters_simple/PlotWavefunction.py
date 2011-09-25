"""The WaveBlocks Project

Plot the wavefunctions probability densities in the eigenbasis.

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


def plot_frames(iom, blockid=0, view=None, plotphase=True, plotcomponents=False, plotabssqr=False, imgsize=(12,9)):
    """Plot the wave function for a series of timesteps.
    @param iom: An I{IOManager} instance providing the simulation data.
    @keyword view: The aspect ratio.
    @keyword plotphase: Whether to plot the complex phase. (slow)
    @keyword plotcomponents: Whether to plot the real/imaginary parts..
    @keyword plotabssqr: Whether to plot the absolute value squared.
    """
    parameters = iom.load_parameters()

    grid = iom.load_grid(blockid=blockid)
    timegrid = iom.load_wavefunction_timegrid(blockid=blockid)

    # Precompute eigenvectors for efficiency
    Potential = PotentialFactory.create_potential(parameters)
    eigenvectors = Potential.evaluate_eigenvectors_at(grid)

    for step in timegrid:
        print(" Plotting frame of timestep # " + str(step))

        wave = iom.load_wavefunction(blockid=blockid, timestep=step)
        values = [ wave[j,...] for j in xrange(parameters["ncomponents"]) ]

        # Transform the values to the eigenbasis
        # TODO: improve this:
        if parameters["algorithm"] == "fourier":
            ve = Potential.project_to_eigen(grid, values, eigenvectors)
        else:
            ve = values

        # Plot the probability densities projected to the eigenbasis
        fig = figure(figsize=imgsize)

        for index, component in enumerate(ve):
            ax = fig.add_subplot(parameters["ncomponents"],1,index+1)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

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

        fig.suptitle(r"$\Psi$ at time $"+str(step*parameters["dt"])+r"$")
        fig.savefig("wavefunction_block"+str(blockid)+"_"+ (7-len(str(step)))*"0"+str(step) +GD.output_format)
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
    view = [-3.5, 3.5, -0.1, 3.5]

    # Iterate over all blocks and plot their data
    for blockid in iom.get_block_ids():
        print("Plotting frames of data block '"+str(blockid)+"'")
        # See if we have wavefunction values
        if iom.has_wavefunction(blockid=blockid):
            plot_frames(iom, blockid=blockid, view=view, plotphase=True, plotcomponents=False, plotabssqr=False)
        else:
            print("Warning: Not plotting any wavefunctions in block '"+str(blockid)+"'!")

    iom.finalize()
