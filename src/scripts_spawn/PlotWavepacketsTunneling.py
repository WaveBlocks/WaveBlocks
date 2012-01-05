"""The WaveBlocks Project

Plot homogeneous and inhomogeneous wavepackets and their coefficients
during the time evolution for a tunneling dynamics situation. The
plot is splitted into 4 subplots corresponding to the left and the right
of the barrier.

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


def plot_frames_homogeneous(iom, bid, view=None):
    """
    @param iom: An I{IOManager} instance providing the simulation data.
    """
    p = iom.load_parameters()

    # Get the data
    grid = iom.load_grid(blockid="global")
    timesteps = iom.load_wavepacket_timegrid(blockid=bid)
    nrtimesteps = timesteps.shape[0]

    params = iom.load_wavepacket_parameters(blockid=bid)
    coeffs = iom.load_wavepacket_coefficients(blockid=bid)

    coeffs = [ [ coeffs[i,j,:] for j in xrange(p["ncomponents"]) ] for i in xrange(nrtimesteps)]

    # Initialize a Hagedorn wavepacket with the data
    Potential = PotentialFactory().create_potential(p)

    HAWP = HagedornWavepacket(p)
    HAWP.set_quadrature(None)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Plotting timestep "+str(step))

        # Configure the wavepacket and project to the eigenbasis.
        HAWP.set_parameters(params[i])
        HAWP.set_coefficients(coeffs[i])
        HAWP.project_to_eigen(Potential)

        values = HAWP.evaluate_at(grid, prefactor=True)
        coeffi = HAWP.get_coefficients()

        plot_frame(bid, step, p, grid, values, coeffi, view=view)

    print(" Plotting frames finished")


def plot_frames_inhomogeneous(iom, bid, view=None):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    p = iom.load_parameters()

    # Get the data
    grid = iom.load_grid(blockid="global")
    timesteps = iom.load_inhomogwavepacket_timegrid(blockid=bid)
    nrtimesteps = timesteps.shape[0]

    params = iom.load_inhomogwavepacket_parameters(blockid=bid)
    coeffs = iom.load_inhomogwavepacket_coefficients(blockid=bid)

    params = [ [ params[j][i,:] for j in xrange(p["ncomponents"]) ] for i in xrange(nrtimesteps)]
    coeffs = [ [ coeffs[i,j,:] for j in xrange(p["ncomponents"]) ] for i in xrange(nrtimesteps)]

    # Initialize a Hagedorn wavepacket with the data
    Potential = PotentialFactory().create_potential(p)

    HAWP = HagedornWavepacketInhomogeneous(p)
    HAWP.set_quadrature(None)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Plotting timestep "+str(step))

        # Configure the wavepacket and project to the eigenbasis.
        HAWP.set_parameters(params[i])
        HAWP.set_coefficients(coeffs[i])
        HAWP.project_to_eigen(Potential)

        values = HAWP.evaluate_at(grid, prefactor=True)
        coeffi = HAWP.get_coefficients()

        plot_frame(bid, step, p, grid, values, coeffi, view=view)

    print(" Plotting frames finished")


def plot_frame(bid, step, parameters, grid, values, coeffs, view=None, imgsize=(16,12)):
    n = parameters["ncomponents"]
    k = array(range(parameters["basis_size"]))

    # This is only for a single level
    y = values[0]
    c = squeeze(coeffs[0])

    # Split the data as necessary
    kl = k[k<K0]
    kr = k[k>=K0]

    gl = grid[grid<=X0]
    gr = grid[grid>X0]

    yl = y[grid<=X0]
    yr = y[grid>X0]

    cs = c[k<K0]
    cb = c[k>=K0]

    # Start new plot
    fig = figure(figsize=imgsize)

    # Plot the probability density, left to X0
    ax1 = fig.add_subplot(2,2,1)
    ax1.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    plotcf(gl, angle(yl), conj(yl)*yl)

    if view is not None:
        ax1.set_xlim(view[0],0)
        ax1.set_ylim(view[2:4])

    ax1.set_xlabel(r"$x \le 0$")
    ax1.set_ylabel(r"$\langle\varphi |\varphi \rangle$")

    # Plot the probability density, right to X0
    ax2 = fig.add_subplot(2,2,2)
    ax2.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    plotcf(gr, angle(yr), conj(yr)*yr)

    if view is not None:
        ax2.set_xlim(0, view[1])
        ax2.set_ylim(view[4:])

    ax2.set_xlabel(r"$x > 0$")
    ax2.set_ylabel(r"$\langle\varphi |\varphi \rangle$")

    # Plot the coefficients smaller than K0 of the Hagedorn wavepacket
    ax3 = fig.add_subplot(2,2,3)
    ax3.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    stemcf(kl, angle(cs), abs(cs))

    # axis formatting:
    m = max(abs(cs))
    ax3.set_xlim(-1, K0+1)
    ax3.set_ylim(0, 1.1*m)
    ax3.set_xlabel(r"$k < K_0$")
    ax3.set_ylabel(r"$|c|$")

    # Plot the coefficients bigger than K0 of the Hagedorn wavepacket
    ax4 = fig.add_subplot(2,2,4)
    ax4.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    stemcf(kr, angle(cb), abs(cb))

    # axis formatting:
    m = max(abs(cb))
    ax4.set_xlim(K0-1,parameters["basis_size"])
    ax4.set_ylim(0, 1.1*m)
    ax4.set_xlabel(r"$k \ge K_0$")
    ax4.set_ylabel(r"$|c|$")

    fig.suptitle(r"Time $"+str(step*parameters["dt"])+r"$")
    fig.savefig("wavepackets_block"+str(bid)+"_"+ (5-len(str(step)))*"0"+str(step) +GD.output_format)
    close(fig)


if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    parameters = iom.load_parameters()

    # The values where to split the position axis and the coefficients
    X0 = 0
    K0 = 50

    # The axes rectangle that is plotted
    view = [-15, 15, 0.0, 1.5, 0.0, 0.05]

    # Iterate over all blocks and plot their data
    for blockid in iom.get_block_ids():
        print("Plotting frames of data block '"+str(blockid)+"'")
        # See if we have an inhomogeneous wavepacket in the current data block
        if iom.has_inhomogwavepacket(blockid=blockid):
            plot_frames_inhomogeneous(iom, blockid, view=view)
        # If not, we test for a homogeneous wavepacket next
        elif iom.has_wavepacket(blockid=blockid):
            plot_frames_homogeneous(iom, blockid, view=view)
        # There is nothing to plot
        else:
            print("Warning: Not plotting any wavepackets in block '"+str(blockid)+"'!")

    iom.finalize()
