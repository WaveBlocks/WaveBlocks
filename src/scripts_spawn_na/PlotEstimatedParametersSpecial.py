"""The WaveBlocks Project

Plot some interesting values of the original and estimated
parameters sets Pi_m=(P,Q,S,p,q) and Pi_s=(B,A,S,b,a).

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import real, imag, abs, angle
from matplotlib.pyplot import *

from WaveBlocks import ComplexMath
from WaveBlocks import IOManager


def read_data_spawn(f, assume_duplicate_mother=False):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    @keyword assume_duplicate_mother: Parameter to tell the code to leave out
    every second data block and only take blocks [0, 1, 3, 5, 7, ...]. This
    is usefull because in aposteriori spawning we have to store clones of
    the mother packet.
    """
    parameters = f.get_parameters()
    ndb = f.get_number_blocks()

    timegrids = []
    AllPA = []

    if assume_duplicate_mother is True:
        blocks = [0] + range(1, ndb, 2)
    else:
        blocks = range(ndb)

    # Load the data from each block
    for block in blocks:
        timegrids.append( parameters["dt"] * f.load_wavepacket_timegrid(block=block) )

        Pi = f.load_wavepacket_parameters(block=block)
        Phist = Pi[:,0]
        Qhist = Pi[:,1]
        Shist = Pi[:,2]
        phist = Pi[:,3]
        qhist = Pi[:,4]

        AllPA.append( [Phist, Qhist, Shist, phist, qhist] )

    return timegrids, AllPA


def plot_parameters_spawn(timegrids, AllPA):
    """Plot some interesting values of the original and estimated
    parameters sets Pi_m=(P,Q,S,p,q) and Pi_s=(B,A,S,b,a).
    """

    # Grid of mother and first spawned packet
    grid_m = timegrids[0]
    grid_s = timegrids[1]

    # Parameters of mother and first spawned packet
    P, Q, S, p, q = AllPA[0]
    B, A, S, b, a = AllPA[1]

    X = P*abs(Q)/Q

    # Various interesting figures

    fig = figure()
    ax = fig.gca()

    ax.plot(grid_m, real(X), "*", label=r"$\Re \frac{P |Q|}{Q}$")
    ax.plot(grid_s, real(B), "o", label=r"$\Re B$")

    ax.legend()
    ax.grid(True)
    fig.savefig("test_spawned_PI_realparts.png")



    fig = figure()
    ax = fig.gca()

    ax.plot(grid_m, imag(X), "*", label=r"$\Im \frac{P |Q|}{Q}$")
    ax.plot(grid_s, imag(B), "o", label=r"$\Im B$")

    ax.legend()
    ax.grid(True)
    fig.savefig("test_spawned_PI_imagparts.png")



    fig = figure()
    ax = fig.gca()

    ax.plot(real(X), imag(X), "-*", label=r"traject $\frac{P |Q|}{Q}$")
    ax.plot(real(B), imag(B), "-*", label=r"traject $B$")

    ax.legend()
    ax.grid(True)
    fig.savefig("test_spawned_PI_complex_trajectories.png")



    fig = figure()
    ax = fig.gca()

    ax.plot(grid_m, angle(X), label=r"$\arg \frac{P |Q|}{Q}$")
    ax.plot(grid_s, angle(B), label=r"$\arg B$")

    ax.legend()
    ax.grid(True)
    fig.savefig("test_spawned_PI_angles.png")



if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    parameters = iom.get_parameters()

    if parameters["algorithm"] == "spawning_apost_na":
        plot_parameters_spawn(*read_data_spawn(iom, assume_duplicate_mother=True))
    else:
        raise NotImplementedError

    iom.finalize()
