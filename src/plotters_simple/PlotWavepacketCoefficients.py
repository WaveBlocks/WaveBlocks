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


def read_all_datablocks(iom):
    """Read the data from all blocks that contains any usable data.
    """
    parameters = iom.get_parameters()
    ndb = iom.get_number_blocks()

    # Iterate over all blocks and plot their data
    for block in xrange(ndb):
        if iom.has_wavepacket(block=block):
            plot_coefficients(parameters, read_data_homogeneous(iom, block=block), index=block)
        elif  iom.has_inhomogwavepacket():
            plot_coefficients(parameters, read_data_inhomogeneous(iom, block=block), index=block)


def read_data_homogeneous(f, block=0):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    parameters = f.get_parameters()
    timegrid = f.load_wavepacket_timegrid(block=block)
    time = timegrid * parameters["dt"]
    
    C = f.load_wavepacket_coefficients(block=block)
    
    coeffs = []
    for i in xrange(parameters["ncomponents"]):
        coeffs.append(squeeze(C[:,i,:]))

    return time, coeffs


def read_data_inhomogeneous(f, block=0):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    parameters = f.get_parameters()
    timegrid = f.load_inhomogwavepacket_timegrid(block=block)
    time = timegrid * parameters["dt"]
    
    C = f.load_inhomogwavepacket_coefficients(block=block)

    coeffs = []
    for i in xrange(parameters["ncomponents"]):
        coeffs.append(squeeze(C[:,i,:]))

    return time, coeffs


def plot_coefficients(parameters, data, amount=5, index=0, imgsize=(14,14)):
    """
    @param parameters: A I{ParameterProvider} instance.
    @param timegrid: The timegrid that belongs to the coefficient values.
    @param: coeffs: The coefficient values.
    @param amount: The number of coefficients we want to plot.
    @keyword imgsize: The size of the plot. For a large number of
    plotted coefficients, we might have to increase this value.
    """    
    # Check if we have enough coefficients to plot
    timegrid, coeffs = data
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

    fig.savefig("wavepacket_coefficients_first_block"+str(index)+".png")
    close(fig)
    
    # And last ones
    fig = figure(figsize=imgsize)
    
    i = 1
    for coeff in reversed(xrange(parameters["basis_size"]-amount,parameters["basis_size"])):
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

    fig.savefig("wavepacket_coefficients_last_block"+str(index)+".png")
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
