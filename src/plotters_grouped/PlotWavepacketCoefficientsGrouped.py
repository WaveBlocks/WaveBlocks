"""The WaveBlocks Project

Plot the evolution of the coefficients $c_i$ of each component
of a homogeneous Hagedorn wavepacket during time propagation.
This script plots several wavepeackets simultaneously.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import squeeze, real, imag, abs
from matplotlib.pyplot import *

from WaveBlocks import IOManager

import GraphicsDefaults as GD


def read_data(iom, gid):
    parameters = iom.load_parameters()

    data = []

    bids = iom.get_block_ids(groupid=gid)

    # Load the data from each block
    for bid in bids:
        if not iom.has_wavepacket(blockid=bid):
            continue

        timegrid = iom.load_wavepacket_timegrid(blockid=bid)
        time = timegrid * parameters["dt"]

        coeffs = iom.load_wavepacket_coefficients(blockid=bid)
        coeffs = [ squeeze(coeffs[:,i,:]) for i in xrange(parameters["ncomponents"]) ]

        data.append((time, coeffs))

    return parameters, data


def plot_coefficients(gid, parameters, data, amount=5, imgsize=(14,14)):
    """
    :param amount: The number of coefficients we want to plot.
    @keyword imgsize: The size of the plot. For a large number of
    plotted coefficients, we might have to increase this value.
    """
    # First ones
    fig = figure(figsize=imgsize)

    i = 1
    for coeff in xrange(amount):
        for component in xrange(parameters["ncomponents"]):
            print(" plotting coefficient " + str(coeff) + " of component " + str(component))
            ax = fig.add_subplot(amount, parameters["ncomponents"], i)

            for time, coeffs in data:
                ax.plot(time, real(coeffs[component][:,coeff]))
                ax.plot(time, imag(coeffs[component][:,coeff]))
                ax.plot(time, abs(coeffs[component][:,coeff]))

            ax.grid(True)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            ax.set_title(r"$\Re c^{"+str(component)+"}_{"+str(coeff)+r"}$ and $\Im c^{"+str(component)+"}_{"+str(coeff)+r"}$")
            i += 1

    fig.savefig("wavepacket_coefficients_grouped_first_group"+str(gid)+GD.output_format)
    close(fig)


    # And last ones
    fig = figure(figsize=imgsize)

    i = 1
    for coeff in reversed(xrange(parameters["basis_size"]-amount,parameters["basis_size"])):
        for component in xrange(parameters["ncomponents"]):
            print(" plotting coefficient " + str(coeff) + " of component " + str(component))
            ax = fig.add_subplot(amount, parameters["ncomponents"], i)

            for time, coeffs in data:
                ax.plot(time, real(coeffs[component][:,coeff]))
                ax.plot(time, imag(coeffs[component][:,coeff]))
                ax.plot(time, abs(coeffs[component][:,coeff]))

            ax.grid(True)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            ax.set_title(r"$\Re c^{"+str(component)+"}_{"+str(coeff)+r"}$ and $\Im c^{"+str(component)+"}_{"+str(coeff)+r"}$")
            i += 1

    fig.savefig("wavepacket_coefficients_grouped_last_group"+str(gid)+GD.output_format)
    close(fig)




if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    gids = iom.get_group_ids(exclude=["global"])

    for gid in gids:
        params, data = read_data(iom, gid)
        plot_coefficients(gid, params, data, amount=2)

    iom.finalize()
