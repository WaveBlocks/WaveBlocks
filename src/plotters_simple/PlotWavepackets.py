"""The WaveBlocks Project

Plot homogeneous and inhomogeneous wavepackets and their coefficients
during the time evolution.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import array, conj, abs, angle, squeeze
from matplotlib.pyplot import *

from WaveBlocks import PotentialFactory
from WaveBlocks import HagedornWavepacket
from WaveBlocks import HagedornWavepacketInhomogeneous
from WaveBlocks import IOManager
from WaveBlocks.Plot import plotcf, stemcf

import GraphicsDefaults as GD


def plot_frames_homogeneous(iom, block=0, view=None):
    """
    @param iom: An I{IOManager} instance providing the simulation data.
    """
    parameters = iom.get_parameters()

    # Number of time steps we saved
    timesteps = iom.load_wavepacket_timegrid(block=block)
    nrtimesteps = timesteps.shape[0]

    # Initialize a Hagedorn wavepacket with the data
    Potential = PotentialFactory.create_potential(parameters)

    # Retrieve simulation data
    grid = iom.load_grid(block=block)
    params = iom.load_wavepacket_parameters(block=block)
    coeffs = iom.load_wavepacket_coefficients(block=block)

    # A data transformation needed by API specification
    coeffs = [ [ coeffs[i,j,:] for j in xrange(parameters["ncomponents"]) ] for i in xrange(nrtimesteps)]

    # Hack for allowing data blocks with different basis size than the global one
    # todo: remove when we got local parameter sets
    parameters.update_parameters({"basis_size": coeffs[0][0].shape[0]})

    HAWP = HagedornWavepacket(parameters)
    HAWP.set_quadrature(None)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Plotting frame of timestep "+str(step))

        # Configure the wavepacket and project to the eigenbasis.
        HAWP.set_parameters(params[i])
        HAWP.set_coefficients(coeffs[i])
        HAWP.project_to_eigen(Potential)

        values = HAWP.evaluate_at(grid, prefactor=True)
        coeffi = HAWP.get_coefficients()

        plot_frame(step, parameters, grid, values, coeffi, index=block, view=view)

    print(" Plotting frames finished")


def plot_frames_inhomogeneous(iom, block=0, view=None):
    """
    @param iom: An I{IOManager} instance providing the simulation data.
    """
    parameters = iom.get_parameters()

    # Number of time steps we saved
    timesteps = iom.load_inhomogwavepacket_timegrid(block=block)
    nrtimesteps = timesteps.shape[0]

    # Initialize a Hagedorn wavepacket with the data
    Potential = PotentialFactory.create_potential(parameters)

    # Retrieve simulation data
    grid = iom.load_grid(block=block)
    params = iom.load_inhomogwavepacket_parameters(block=block)
    coeffs = iom.load_inhomogwavepacket_coefficients(block=block)

    # A data transformation needed by API specification
    params = [ [ params[j][i,:] for j in xrange(parameters["ncomponents"]) ] for i in xrange(nrtimesteps)]
    coeffs = [ [ coeffs[i,j,:] for j in xrange(parameters["ncomponents"]) ] for i in xrange(nrtimesteps)]

    # Hack for allowing data blocks with different basis size than the global one
    # todo: remove when we got local parameter sets
    parameters.update_parameters({"basis_size": coeffs[0][0].shape[0]})

    HAWP = HagedornWavepacketInhomogeneous(parameters)
    HAWP.set_quadrature(None)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Plotting frame of timestep "+str(step))

        # Configure the wavepacket and project to the eigenbasis.
        HAWP.set_parameters(params[i])
        HAWP.set_coefficients(coeffs[i])
        HAWP.project_to_eigen(Potential)

        values = HAWP.evaluate_at(grid, prefactor=True)
        coeffi = HAWP.get_coefficients()

        plot_frame(step, parameters, grid, values, coeffi, index=block, view=view)

    print(" Plotting frames finished")


def plot_frame(step, parameters, grid, values, coeffs, index=0, view=None, imgsize=(12,9)):
    n = parameters["ncomponents"]
    k = array(range(parameters["basis_size"]))

    # Start new plot
    fig = figure(figsize=imgsize)

    for s in xrange(n):
        y = values[s]
        c = squeeze(coeffs[s])

        # Plot the probability densities
        ax1 = fig.add_subplot(n,2,2*s+1)
        ax1.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        plotcf(grid, angle(y), conj(y)*y)

        if view is not None:
            ax1.set_xlim(view[:2])
            ax1.set_ylim(view[2:])

        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$\langle\varphi_"+str(s)+r"|\varphi_"+str(s)+r"\rangle$")

        # Plot the coefficients of the Hagedorn wavepacket
        ax2 = fig.add_subplot(n,2,2*s+2)
        ax2.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
        stemcf(k, angle(c), abs(c))

        # axis formatting:
        m = max(abs(c))
        ax2.set_xlim(-1,parameters["basis_size"])
        ax2.set_ylim(-0.1*m, 1.1*m)

        ax2.set_xlabel(r"$k$")
        ax2.set_ylabel(r"$|c|$")

    fig.suptitle(r"Time $"+str(step*parameters["dt"])+r"$")
    fig.savefig("wavepackets_block"+str(index)+"_"+ (7-len(str(step)))*"0"+str(step) +GD.output_format)
    close(fig)


if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    # The axes rectangle that is plotted
    view = [-2.5, 2.5, -0.1, 2.5]

    # Iterate over all blocks and plot their data
    for block in xrange(iom.get_number_blocks()):
        print("Plotting frames of data block "+str(block))
        # See if we have an inhomogeneous wavepacket in the current data block
        if iom.has_inhomogwavepacket(block=block):
            plot_frames_inhomogeneous(iom, block=block, view=view)
        # If not, we test for a homogeneous wavepacket next
        elif iom.has_wavepacket(block=block):
            plot_frames_homogeneous(iom, block=block, view=view)
        # There is nothing to plot
        else:
            print("Warning: Not plotting any wavepackets in block "+str(block)+"!")

    iom.finalize()
