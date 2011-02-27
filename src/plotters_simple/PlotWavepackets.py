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
from WaveBlocks import HagedornMultiWavepacket
from WaveBlocks import IOManager
from WaveBlocks.Plot import plotcf, stemcf


def plot_frames_homogeneous(f, view=None):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    p = f.get_parameters()
    
    # Get the data
    grid = f.load_grid()
    timesteps = f.load_wavepacket_timegrid()
    nrtimesteps = timesteps.shape[0]
    
    params = f.load_wavepacket_parameters()
    coeffs = f.load_wavepacket_coefficients()

    coeffs = [ [ coeffs[i,j,:] for j in xrange(p.ncomponents) ] for i in xrange(nrtimesteps)]

    # Initialize a Hagedorn wavepacket with the data
    Potential = PotentialFactory.create_potential(p)

    HAWP = HagedornWavepacket(p)
    HAWP.set_quadrator(None)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Plotting timestep "+str(step))

        # Configure the wavepacket and project to the eigenbasis.
        HAWP.set_parameters(params[i])
        HAWP.set_coefficients(coeffs[i])
        HAWP.project_to_eigen(Potential)

        values = HAWP.evaluate_at(grid, prefactor=True)
        coeffi = HAWP.get_coefficients()
        
        plot_frame(step, p, grid, values, coeffi, view=view)
    
    print(" Plotting frames finished")


def plot_frames_inhomogeneous(f, view=None):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    p = f.get_parameters()
    
    # Get the data
    grid = f.load_grid()
    timesteps = f.load_inhomogwavepacket_timegrid()
    nrtimesteps = timesteps.shape[0]
    
    params = f.load_inhomogwavepacket_parameters()
    coeffs = f.load_inhomogwavepacket_coefficients()

    params = [ [ params[j][i,:] for j in xrange(p.ncomponents) ] for i in xrange(nrtimesteps)]
    coeffs = [ [ coeffs[i,j,:] for j in xrange(p.ncomponents) ] for i in xrange(nrtimesteps)]

    # Initialize a Hagedorn wavepacket with the data
    Potential = PotentialFactory.create_potential(p)

    HAWP = HagedornMultiWavepacket(p)
    HAWP.set_quadrator(None)

    # Iterate over all timesteps
    for i, step in enumerate(timesteps):
        print(" Plotting timestep "+str(step))

        # Configure the wavepacket and project to the eigenbasis.
        HAWP.set_parameters(params[i])
        HAWP.set_coefficients(coeffs[i])
        HAWP.project_to_eigen(Potential)

        values = HAWP.evaluate_at(grid, prefactor=True)
        coeffi = HAWP.get_coefficients()

        plot_frame(step, p, grid, values, coeffi, view=view)
        
    print(" Plotting frames finished")


def plot_frame(step, parameters, grid, values, coeffs, view=None, imgsize=(12,9)):
    n = parameters.ncomponents
    k = array(range(parameters.basis_size))
    
    # Start new plot
    fig = figure(figsize=imgsize)

    for s in xrange(n):
        y = values[s]
        c = squeeze(coeffs[s])
            
        # Plot the probability densities
        ax1 = fig.add_subplot(n,2,2*s+1)
        ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
        plotcf(grid, angle(y), conj(y)*y)

        if view is not None:
            ax1.set_xlim(view[:2])
            ax1.set_ylim(view[2:])

        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$\langle\varphi_"+str(s)+r"|\varphi_"+str(s)+r"\rangle$")

        # Plot the coefficients of the Hagedorn wavepacket
        ax2 = fig.add_subplot(n,2,2*s+2)
        ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
        stemcf(k, angle(c), abs(c))

        # axis formatting:
        m = max(abs(c))
        ax2.set_xlim(-1,parameters.basis_size)
        ax2.set_ylim(-1.1*m, 1.1*m)
        
        ax2.set_xlabel(r"$k$")
        ax2.set_ylabel(r"$|c|$")

    fig.suptitle(r"Time $"+str(step*parameters.dt)+r"$")
    fig.savefig("wavepackets_"+ (5-len(str(step)))*"0"+str(step) +".png")
    close(fig)


if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    parameters = iom.get_parameters()

    # The axes rectangle that is plotted
    view = [-5.5, 5.5, -1.5, 1.5]

    if parameters.algorithm == "hagedorn":
        plot_frames_homogeneous(iom, view=view)
    elif parameters.algorithm == "multihagedorn":
        plot_frames_inhomogeneous(iom, view=view)
    else:
        iom.finalize()
        sys.exit("Can only postprocess (multi)hagedorn algorithm data. Silent return ...")
        
    iom.finalize()
