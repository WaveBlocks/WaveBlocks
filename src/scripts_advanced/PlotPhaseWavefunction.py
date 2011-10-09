"""The WaveBlocks Project

Plot the location dependent phase of a wavefunction.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import angle, conj
from matplotlib.pyplot import *

from WaveBlocks import PotentialFactory
from WaveBlocks import IOManager
from WaveBlocks import ComplexMath
from WaveBlocks.Plot import plotcf

import GraphicsDefaults as GD


def plot_frames(iom, blockid=0, view=None, imgsize=(12,9)):
    """Plot the phase of a wavefunction for a series of timesteps.
    @param iom: An I{IOManager} instance providing the simulation data.
    @keyword view: The aspect ratio.
    """
    parameters = iom.load_parameters()

    grid = iom.load_grid(blockid="global")
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

        fig = figure(figsize=imgsize)

        for index, component in enumerate(ve):
            ax = fig.add_subplot(parameters["ncomponents"],1,index+1)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

            # Plot the wavefunction
            ax.plot(grid, component*conj(component), color="gray")
            ax.set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")
            ax.set_xlabel(r"$x$")

            # Compute the phase from the wavefunction restricted to "important" regions
            restr_grid = grid[component*conj(component) > 10e-8]
            restr_comp = component[component*conj(component) > 10e-8]

            # Plot the phase
            ax.plot(restr_grid, angle(restr_comp), "-", color="green")
            ax.plot(restr_grid, ComplexMath.continuate(angle(restr_comp)), ".", color="green")

            # Set the aspect window
            if view is not None:
                ax.set_xlim(view[:2])
                #ax.set_ylim(view[2:])

        fig.suptitle(r"$\arg \Psi$ at time $"+str(step*parameters["dt"])+r"$")
        fig.savefig("wavefunction_phase_block"+str(blockid)+"_"+ (7-len(str(step)))*"0"+str(step) +GD.output_format)
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
        print("Plotting frames of data block "+str(blockid))
        # See if we have wavefunction values
        if iom.has_wavefunction(blockid=blockid):
            plot_frames(iom, blockid=blockid, view=view)
        else:
            print("Warning: Not plotting any wavefunctions in block "+str(blockid)+"!")

    iom.finalize()
