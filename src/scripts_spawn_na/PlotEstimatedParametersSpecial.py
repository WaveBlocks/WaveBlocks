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

        Pi = iom.load_wavepacket_parameters(blockid=bid)
        Phist = Pi[:,0]
        Qhist = Pi[:,1]
        Shist = Pi[:,2]
        phist = Pi[:,3]
        qhist = Pi[:,4]

        data.append((time, (Phist, Qhist, Shist, phist, qhist)))

    return data


def plot_parameters(gid, data):
    print("Plotting transformed wavepacket parameters of group '"+str(gid)+"'")

    # Grid of mother and first spawned packet
    grid_m = data[0][0]
    grid_s = data[1][0]

    # Parameters of mother and first spawned packet
    P, Q, S, p, q = data[0][1]
    B, A, S, b, a = data[1][1]

    X = P*abs(Q)/Q

    # Various interesting figures

    fig = figure()
    ax = fig.gca()

    ax.plot(grid_m, real(X), "*", label=r"$\Re \frac{P |Q|}{Q}$")
    ax.plot(grid_s, real(B), "o", label=r"$\Re B$")

    ax.legend()
    ax.grid(True)
    fig.savefig("test_spawned_PI_realparts_group"+str(gid)+GD.output_format)



    fig = figure()
    ax = fig.gca()

    ax.plot(grid_m, imag(X), "*", label=r"$\Im \frac{P |Q|}{Q}$")
    ax.plot(grid_s, imag(B), "o", label=r"$\Im B$")

    ax.legend()
    ax.grid(True)
    fig.savefig("test_spawned_PI_imagparts_group"+str(gid)+GD.output_format)



    fig = figure()
    ax = fig.gca()

    ax.plot(real(X), imag(X), "-*", label=r"traject $\frac{P |Q|}{Q}$")
    ax.plot(real(B), imag(B), "-*", label=r"traject $B$")

    ax.legend()
    ax.grid(True)
    fig.savefig("test_spawned_PI_complex_trajectories_group"+str(gid)+GD.output_format)



    fig = figure()
    ax = fig.gca()

    ax.plot(grid_m, angle(X), label=r"$\arg \frac{P |Q|}{Q}$")
    ax.plot(grid_s, angle(B), label=r"$\arg B$")

    ax.legend()
    ax.grid(True)
    fig.savefig("test_spawned_PI_angles_group"+str(gid)+GD.output_format)



if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    gids = iom.get_group_ids(exclude=["global"])

    for gid in gids:
        data = read_data(iom, gid)
        plot_parameters(gid, data)

    iom.finalize()
