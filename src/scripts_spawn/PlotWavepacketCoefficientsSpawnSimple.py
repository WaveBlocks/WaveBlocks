"""The WaveBlocks Project

Plot the evolution of the coefficients $c_i$ of each component
of a homogeneous Hagedorn wavepacket and a spawned wavepacket during
time propagation.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import squeeze, real, imag, abs
from matplotlib.pyplot import *

from WaveBlocks import IOManager


def read_data_spawn(f):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    parameters = f.get_parameters()

    timegrid0 = f.load_wavepacket_timegrid()
    timegrid1 = f.load_wavepacket_timegrid(block=1)
    time0 = timegrid0 * parameters.dt
    time1 = timegrid1 * parameters.dt
    
    C0 = f.load_wavepacket_coefficients()
    coeffs0 = []
    for i in xrange(parameters.ncomponents):
        coeffs0.append(squeeze(C0[:,i,:]))

    C1 = f.load_wavepacket_coefficients(block=1)
    coeffs1 = []
    for i in xrange(parameters.ncomponents):
        coeffs1.append(squeeze(C1[:,i,:]))

    return (parameters, time0, time1, coeffs0, coeffs1)


def plot_coefficients_spawn(parameters, timegrid0, timegrid1, coeffs0, coeffs1, amount=5, imgsize=(14,14)):
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
            
            ax.plot(timegrid0, real(coeffs0[component][:,coeff]))
            ax.plot(timegrid0, imag(coeffs0[component][:,coeff]))
            ax.plot(timegrid0, abs(coeffs0[component][:,coeff]))

            ax.plot(timegrid1, real(coeffs1[component][:,coeff]), "c")
            ax.plot(timegrid1, imag(coeffs1[component][:,coeff]), "m")
            ax.plot(timegrid1, abs(coeffs1[component][:,coeff]), "k")

            ax.grid(True)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
            ax.set_title(r"$\Re c^{"+str(component)+"}_{"+str(coeff)+r"}$ and $\Im c^{"+str(component)+"}_{"+str(coeff)+r"}$")
            i += 1

    fig.savefig("wavepacket_coefficients_spawn_first.png")
    close(fig)
    
    # And last ones
    fig = figure(figsize=imgsize)
    
    i = 1
    for coeff in reversed(xrange(parameters.basis_size-amount,parameters.basis_size)):
        for component in xrange(parameters.ncomponents):
            print(" plotting coefficient " + str(coeff) + " of component " + str(component))
            ax = fig.add_subplot(amount, parameters.ncomponents, i)

            ax.plot(timegrid0, real(coeffs0[component][:,coeff]))
            ax.plot(timegrid0, imag(coeffs0[component][:,coeff]))
            ax.plot(timegrid0, abs(coeffs0[component][:,coeff]))

            ax.plot(timegrid1, real(coeffs1[component][:,coeff]), "c")
            ax.plot(timegrid1, imag(coeffs1[component][:,coeff]), "m")
            ax.plot(timegrid1, abs(coeffs1[component][:,coeff]), "k")

            ax.grid(True)
            ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y") 
            ax.set_title(r"$\Re c^{"+str(component)+"}_{"+str(coeff)+r"}$ and $\Im c^{"+str(component)+"}_{"+str(coeff)+r"}$")
            i += 1

    fig.savefig("wavepacket_coefficients_spawn_last.png")
    close(fig)

    
if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    plot_coefficients_spawn(*read_data_spawn(iom), amount=5)

    iom.finalize()
