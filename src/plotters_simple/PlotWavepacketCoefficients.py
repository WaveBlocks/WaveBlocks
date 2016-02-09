"""The WaveBlocks Project

Plot the evolution of the coefficients $c_i$ of each component
of a homogeneous or inhomogeneous Hagedorn wavepacket during the
time propagation.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import squeeze, real, imag, abs
from matplotlib.pyplot import *

from WaveBlocks import IOManager

import GraphicsDefaults as GD


def read_all_datablocks(iom):
    """Read the data from all blocks that contains any usable data.
    :param iom: An I{IOManager} instance providing the simulation data.
    """
    parameters = iom.load_parameters()

    # Iterate over all blocks and plot their data
    for blockid in iom.get_block_ids():
        if iom.has_wavepacket(blockid=blockid):
            plot_coefficients(parameters, read_data_homogeneous(iom, blockid=blockid), index=blockid)
        elif iom.has_inhomogwavepacket(blockid=blockid):
            plot_coefficients(parameters, read_data_inhomogeneous(iom, blockid=blockid), index=blockid)
        else:
            print("Warning: Not plotting wavepacket coefficients in block '"+str(blockid)+"'!")


def read_data_homogeneous(iom, blockid=0):
    """
    :param iom: An I{IOManager} instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_wavepacket_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    C = iom.load_wavepacket_coefficients(blockid=blockid)

    coeffs = []
    for i in xrange(parameters["ncomponents"]):
        coeffs.append(squeeze(C[:,i,:]))

    return time, coeffs


def read_data_inhomogeneous(iom, blockid=0):
    """
    :param iom: An I{IOManager} instance providing the simulation data.
    :param blockid: The data block from which the values are read.
    """
    parameters = iom.load_parameters()
    timegrid = iom.load_inhomogwavepacket_timegrid(blockid=blockid)
    time = timegrid * parameters["dt"]

    C = iom.load_inhomogwavepacket_coefficients(blockid=blockid)

    coeffs = []
    for i in xrange(parameters["ncomponents"]):
        coeffs.append(squeeze(C[:,i,:]))

    return time, coeffs


def plot_coefficients(parameters, data, amount=5, index=0, imgsize=(14,14)):
    """
    :param parameters: A I{ParameterProvider} instance.
    :param timegrid: The timegrid that belongs to the coefficient values.
    :param: coeffs: The coefficient values.
    :param amount: The number of coefficients we want to plot.
    :param imgsize: The size of the plot. For a large number of
    plotted coefficients, we might have to increase this value.
    """
    print("Plotting the coefficients of data block '"+str(index)+"'")

    # Check if we have enough coefficients to plot
    timegrid, coeffs = data

    # Hack for allowing data blocks with different basis size than the global one
    # todo: improve second arg when we got local parameter sets
    amount = min(amount, coeffs[0].shape[-1])

    # First ones
    fig = figure(figsize=imgsize)

    i = 1
    for coeff in xrange(amount):
        for component in xrange(len(coeffs)):
            print(" plotting coefficient " + str(coeff) + " of component " + str(component))
            ax = fig.add_subplot(amount, len(coeffs), i)

            ax.plot(timegrid, real(coeffs[component][:,coeff]))
            ax.plot(timegrid, imag(coeffs[component][:,coeff]))
            ax.plot(timegrid, abs(coeffs[component][:,coeff]))

            ax.grid(True)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            ax.set_title(r"$\Re c^{"+str(component)+"}_{"+str(coeff)+r"}$ and $\Im c^{"+str(component)+"}_{"+str(coeff)+r"}$")
            i += 1

    fig.savefig("wavepacket_coefficients_first_block"+str(index)+GD.output_format)
    close(fig)

    # And last ones
    fig = figure(figsize=imgsize)

    # Hack for allowing data blocks with different basis size than the global one
    # todo: improve second arg when we got local parameter sets
    bs = coeffs[0][0,:].shape[0]

    i = 1
    for coeff in reversed(xrange(bs-amount,bs)):
        for component in xrange(parameters["ncomponents"]):
            print(" plotting coefficient " + str(coeff) + " of component " + str(component))
            ax = fig.add_subplot(amount, parameters["ncomponents"], i)

            ax.plot(timegrid, real( coeffs[component][:,coeff] ) )
            ax.plot(timegrid, imag( coeffs[component][:,coeff] ) )
            ax.plot(timegrid, abs( coeffs[component][:,coeff] ) )

            ax.grid(True)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            ax.set_title(r"$\Re c^{"+str(component)+"}_{"+str(coeff)+r"}$ and $\Im c^{"+str(component)+"}_{"+str(coeff)+r"}$")
            i += 1

    fig.savefig("wavepacket_coefficients_last_block"+str(index)+GD.output_format)
    close(fig)


if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    # Read the data and plot it, one plot for each data block.
    read_all_datablocks(iom)

    iom.finalize()
