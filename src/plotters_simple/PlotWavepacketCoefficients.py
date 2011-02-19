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


def read_data_homogeneous(f):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    parameters = f.get_parameters()
    
    timegrid = f.load_wavepacket_timegrid()
    C = f.load_wavepacket_coefficients()

    coeffs = []
    for i in xrange(parameters.ncomponents):
        coeffs.append(squeeze(C[:,i,:]))

    return (parameters, timegrid, coeffs)


def read_data_inhomogeneous(f):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    parameters = f.get_parameters()
    
    timegrid = f.load_inhomogwavepacket_timegrid()
    C = f.load_inhomogwavepacket_coefficients()

    coeffs = []
    for i in xrange(parameters.ncomponents):
        coeffs.append(squeeze(C[:,i,:]))

    return (parameters, timegrid, coeffs)


def plot_coefficients(parameters, timegrid, coeffs, amount=5, imgsize=(14,14)):
    """
    @param parameters: A I{ParameterProvider} instance.
    @param timegrid: The timegrid that belongs to the coefficient values.
    @param: coeffs: The coefficient values.
    @param amount: The number of coefficients we want to plot.
    @keyword imgsize: The size of the plot. For a large number of
    plotted coefficients, we might have to increase this value.
    """
    # First ones
    fig = figure(figsize=imgsize)
    
    i = 1
    for coeff in xrange(amount):
        for component in xrange(parameters.ncomponents):
            print(" plotting coefficient " + str(coeff) + " of component " + str(component))
            ax = fig.add_subplot(amount, parameters.ncomponents, i)
            
            ax.plot(timegrid, real(coeffs[component][:,coeff]))
            ax.plot(timegrid, imag(coeffs[component][:,coeff]))
            ax.plot(timegrid, abs(coeffs[component][:,coeff]))

            ax.set_title(r"$\Re c^{"+str(component)+"}_{"+str(coeff)+r"}$ and $\Im c^{"+str(component)+"}_{"+str(coeff)+r"}$")
            i += 1

    fig.savefig("wavepacket_coefficients_first.png")
    close(fig)
    
    # And last ones
    fig = figure(figsize=imgsize)
    
    i = 1
    for coeff in reversed(xrange(parameters.basis_size-amount,parameters.basis_size)):
        for component in xrange(parameters.ncomponents):
            print(" plotting coefficient " + str(coeff) + " of component " + str(component))
            ax = fig.add_subplot(amount, parameters.ncomponents, i)

            ax.plot(timegrid, real( coeffs[component][:,coeff] ) )
            ax.plot(timegrid, imag( coeffs[component][:,coeff] ) )
            ax.plot(timegrid, abs( coeffs[component][:,coeff] ) )

            ax.set_title(r"$\Re c^{"+str(component)+"}_{"+str(coeff)+r"}$ and $\Im c^{"+str(component)+"}_{"+str(coeff)+r"}$")
            i += 1

    fig.savefig("wavepacket_coefficients_last.png")
    close(fig)

    
if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    parameters = iom.get_parameters()

    if parameters.algorithm == "hagedorn":
        data = read_data_homogeneous(iom)
    elif parameters.algorithm == "multihagedorn":
        data = read_data_inhomogeneous(iom)
    else:
        iom.finalize()
        sys.exit("Can only postprocess (multi)hagedorn algorithm data. Silent return ...")

    plot_coefficients(*data, amount=3)

    iom.finalize()
