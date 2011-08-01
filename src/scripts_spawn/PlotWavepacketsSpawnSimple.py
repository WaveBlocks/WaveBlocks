"""The WaveBlocks Project

Plot homogeneous wavepackets and their coefficients during the
time evolution of spawning.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
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


def plot_frames_homogeneous(f, plotphase=False, plotcomponents=False, plotabssqr=True, view=None, imgsize=(12,9)):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    parameters = f.get_parameters()
    
    grid = f.load_grid()
    k = array(range(parameters["basis_size"]))

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
            coeffs_m = squeeze(f.load_wavepacket_coefficients(timestep=step, block=0))
            have_mother_data = True
        except ValueError:
            have_mother_data = False

        # Retrieve spawn data
        try:
            wave_s = f.load_wavefunction(timestep=step, block=1)
            values_s = [ squeeze(wave_s[j,...]) for j in xrange(parameters["ncomponents"]) ]
            coeffs_s = squeeze(f.load_wavepacket_coefficients(timestep=step, block=1))
            have_spawn_data = True
        except ValueError:
            have_spawn_data = False

        # Start new plot
        fig = figure(figsize=imgsize)

        ax1 = subplot2grid((2,2), (0,0), colspan=2)
        ax1.ticklabel_format(style="sci", scilimits=(0,0), axis="y")

        # Plot original Wavefunction
        if have_mother_data is True:
            for index, component in enumerate(values_m):
                if plotcomponents is True:
                    ax1.plot(grid, real(component))
                    ax1.plot(grid, imag(component))
                    ax1.set_ylabel(r"$\Re \varphi_"+str(index)+r", \Im \varphi_"+str(index)+r"$")
                if plotabssqr is True:
                    ax1.plot(grid, component*conj(component))
                    ax1.set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")
                if plotphase is True:
                    plotcf(grid, angle(component), component*conj(component))
                    ax1.set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")

        # Overlay spawned parts
        if have_spawn_data is True:
            for index, component in enumerate(values_s):
                if plotcomponents is True:
                    ax1.plot(grid, real(component))
                    ax1.plot(grid, imag(component))
                    ax1.set_ylabel(r"$\Re \varphi_"+str(index)+r", \Im \varphi_"+str(index)+r"$")
                if plotabssqr is True:
                    ax1.plot(grid, component*conj(component))
                    ax1.set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")
                if plotphase is True:
                    plotcf(grid, angle(component), component*conj(component))
                    ax1.set_ylabel(r"$\langle \varphi_"+str(index)+r"| \varphi_"+str(index)+r"\rangle$")

        ax1.set_xlim(view[0:2])
        ax1.set_ylim(view[2:4])
        ax1.set_xlabel(r"$x$")
        ax1.set_title(r"Densities of mother and spawned packets")

        # Plot coefficients for mother packet
        if have_mother_data is True:
            ax2 = subplot2grid((2,2), (1,0))
            ax2.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            stemcf(k, angle(coeffs_m), abs(coeffs_m))
            ax2.set_xlim((-1, parameters["basis_size"]))
            ax2.set_xlabel(r"$k$")
            ax2.set_ylabel(r"$|c|$")
            ax2.set_title(r"Mother packet $| \Psi^m \rangle$")

        # Plot coefficients for spawned packet
        if have_spawn_data is True:
            ax3 = subplot2grid((2,2), (1,1))
            ax3.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            stemcf(k, angle(coeffs_s), abs(coeffs_s))
            ax3.set_xlim((-1, parameters["basis_size"]))
            ax3.set_xlabel(r"$k$")
            ax3.set_ylabel(r"$|c|$")
            ax3.set_title(r"Spawned packet $| \Psi^s \rangle$")

        fig.suptitle(r"Time $"+str(step*parameters["dt"])+r"$")
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
    view = [-8, 8, 0.0, 0.6]

    plot_frames_homogeneous(iom, view=view)
            
    iom.finalize()
