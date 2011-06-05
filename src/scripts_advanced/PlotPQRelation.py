"""The WaveBlocks Project

Plot the evolution of the relations between the parameters P and Q
homogeneous or inhomogeneous Hagedorn wavepacket during the
time propagation.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import conj, abs
from matplotlib.pyplot import *

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

    return (time, Phist, Qhist)


def read_data_inhomogeneous(f):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    parameters = f.get_parameters()
    timegrid = f.load_inhomogwavepacket_timegrid()
    time = timegrid * params["dt"]
    
    timegrid = f.load_inhomogwavepacket_timegrid()
    Pi = f.load_inhomogwavepacket_parameters()

    # Number of components
    N = parameters.ncomponents

    Phist = [ Pi[i][:,0] for i in xrange(N) ]
    Qhist = [ Pi[i][:,1] for i in xrange(N) ]
    
    return (time, Phist, Qhist)


def plot_parameters(timegrid, Phist, Qhist):
    # Plot the time evolution of the parameters P, Q, S, p and q
    fig = figure(figsize=(12,12))
    ax = fig.gca()

    for ptem, qtem in zip(Phist, Qhist):
        ax.plot(timegrid, abs(conj(qtem)*ptem - conj(ptem)*qtem - 2.0j))
    
    ax.grid(True)
    ax.ticklabel_format(style="sci", scilimits=(0,0), axis="y")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$| \overline{Q} P - \overline{P} Q - 2i |$")
    ax.set_title(r"Compatibility condition $\overline{Q} P - \overline{P} Q = 2i$")
    fig.savefig("conjQP-conjPQ.png")
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
