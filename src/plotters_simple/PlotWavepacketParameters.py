"""The WaveBlocks Project

Plot the evolution of the parameters Pi_i = (P,Q,S,p,q) of a
homogeneous or inhomogeneous Hagedorn wavepacket during the
time propagation.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import real, imag, abs
from matplotlib.pyplot import *

from WaveBlocks import ComplexMath 
from WaveBlocks import IOManager


def read_data_homogeneous(f):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    parameters = f.get_parameters()
    timegrid = f.load_wavepacket_timegrid()
    time = timegrid * parameters.dt

    Pi = f.load_wavepacket_parameters()

    Phist = [ Pi[:,0] ]
    Qhist = [ Pi[:,1] ]
    Shist = [ Pi[:,2] ]
    phist = [ Pi[:,3] ]
    qhist = [ Pi[:,4] ]

    return (time, Phist, Qhist, Shist, phist, qhist)


def read_data_inhomogeneous(f):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    parameters = f.get_parameters()
    timegrid = f.load_inhomogwavepacket_timegrid()
    time = timegrid * params.dt
    
    timegrid = f.load_inhomogwavepacket_timegrid()
    Pi = f.load_inhomogwavepacket_parameters()

    # Number of components
    N = parameters.ncomponents

    Phist = [ Pi[i][:,0] for i in xrange(N) ]
    Qhist = [ Pi[i][:,1] for i in xrange(N) ]
    Shist = [ Pi[i][:,2] for i in xrange(N) ]
    phist = [ Pi[i][:,3] for i in xrange(N) ]
    qhist = [ Pi[i][:,4] for i in xrange(N) ]

    return (time, Phist, Qhist, Shist, phist, qhist)


def plot_parameters(timegrid, Phist, Qhist, Shist, phist, qhist):
    # Plot the time evolution of the parameters P, Q, S, p and q
    fig = figure(figsize=(12,12))

    ax = fig.add_subplot(4,2,1)
    for item in Phist:
        ax.plot(timegrid, real(item), label=r"$\Re P$")
    ax.grid(True)
    ax.set_title(r"$\Re P$")
    
    ax = fig.add_subplot(4,2,2)
    for item in Phist:
        ax.plot(timegrid, imag(item), label=r"$\Im P$")
    ax.grid(True)
    ax.set_title(r"$\Im P$")
    
    ax = fig.add_subplot(4,2,3)
    for item in Qhist:
        ax.plot(timegrid, real(item), label=r"$\Re Q$")
    ax.grid(True)
    ax.set_title(r"$\Re Q$")
    
    ax = fig.add_subplot(4,2,4)
    for item in Qhist:
        ax.plot(timegrid, imag(item), label=r"$\Im Q$")
    ax.grid(True)
    ax.set_title(r"$\Im Q$")
    
    ax = fig.add_subplot(4,2,5)
    for item in qhist:
        ax.plot(timegrid, real(item), label=r"$q$")
    ax.grid(True)
    ax.set_title(r"$q$")
    
    ax = fig.add_subplot(4,2,6)
    for item in phist:
        ax.plot(timegrid, real(item), label=r"$p$")
    ax.grid(True)
    ax.set_title(r"$p$")
    
    ax = fig.add_subplot(4,2,7)
    for item in Shist:
        ax.plot(timegrid, real(item), label=r"$S$")
    ax.grid(True)
    ax.set_title(r"$S$")

    fig.suptitle("Wavepacket parameters")
    fig.savefig("wavepacket_parameters.png")
    close(fig)


    # Plot the time evolution of the parameters P, Q, S, p and q
    # This time plot abs/angle instead of real/imag
    fig = figure(figsize=(12,12))

    ax = fig.add_subplot(4,2,1)
    for item in Phist:
        ax.plot(timegrid, abs(item), label=r"$|P|$")
    ax.grid(True)
    ax.set_title(r"$|P|$")
    
    ax = fig.add_subplot(4,2,2)
    for item in Phist:
        ax.plot(timegrid, ComplexMath.cont_angle(item), label=r"$\arg P$")
    ax.grid(True)
    ax.set_title(r"$\arg P$")
    
    ax = fig.add_subplot(4,2,3)
    for item in Qhist:
        ax.plot(timegrid, abs(item), label=r"$|Q|$")
    ax.grid(True)
    ax.set_title(r"$|Q|$")
    
    ax = fig.add_subplot(4,2,4)
    for item in Qhist:
        ax.plot(timegrid, ComplexMath.cont_angle(item), label=r"$\arg Q$")
    ax.grid(True)
    ax.set_title(r"$\arg Q$")
    
    ax = fig.add_subplot(4,2,5)
    for item in qhist:
        ax.plot(timegrid, real(item), label=r"$q$")
    ax.grid(True)
    ax.set_title(r"$q$")
    
    ax = fig.add_subplot(4,2,6)
    for item in phist:
        ax.plot(timegrid, real(item), label=r"$p$")
    ax.grid(True)
    ax.set_title(r"$p$")
    
    ax = fig.add_subplot(4,2,7)
    for item in Shist:
        ax.plot(timegrid, abs(item), label=r"$S$")
    ax.grid(True)
    ax.set_title(r"$S$")

    fig.suptitle("Wavepacket parameters")
    fig.savefig("wavepacket_parameters_abs_ang.png")
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
        
    plot_parameters(*data)

    iom.finalize()
