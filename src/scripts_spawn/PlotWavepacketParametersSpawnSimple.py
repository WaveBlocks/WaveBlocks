"""The WaveBlocks Project

Plot the evolution of the parameters Pi_i = (P,Q,S,p,q) of a
homogeneous Hagedorn wavepacket and a spawned wavepacket during
the time propagation.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import real, imag, abs, angle
from matplotlib.pyplot import *

from WaveBlocks import ComplexMath
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

    Pi0 = f.load_wavepacket_parameters()

    Phist0 = [ Pi0[:,0] ]
    Qhist0 = [ Pi0[:,1] ]
    Shist0 = [ Pi0[:,2] ]
    phist0 = [ Pi0[:,3] ]
    qhist0 = [ Pi0[:,4] ]

    AllPA0 = [ Phist0, Qhist0, Shist0, phist0, qhist0 ]

    Pi1 = f.load_wavepacket_parameters(block=1)

    Phist1 = [ Pi1[:,0] ]
    Qhist1 = [ Pi1[:,1] ]
    Shist1 = [ Pi1[:,2] ]
    phist1 = [ Pi1[:,3] ]
    qhist1 = [ Pi1[:,4] ]

    AllPA1 = [ Phist1, Qhist1, Shist1, phist1, qhist1 ]

    return (time0, time1, AllPA0, AllPA1)



def plot_parameters_spawn(timegrid0, timegrid1, AllPA0, AllPA1):

    Phist0, Qhist0, Shist0, phist0, qhist0 = AllPA0
    Phist1, Qhist1, Shist1, phist1, qhist1 = AllPA1

    # Plot the time evolution of the parameters P, Q, S, p and q
    fig = figure(figsize=(12,12))

    ax = fig.add_subplot(4,2,1)
    for item in Phist0:
        ax.plot(timegrid0, real(item), label=r"$\Re P$")
    for item in Phist1:
        ax.plot(timegrid1, real(item), "c", label=r"$\Re P^s$")
    ax.grid(True)
    ax.set_title(r"$\Re P$")
    
    ax = fig.add_subplot(4,2,2)
    for item in Phist0:
        ax.plot(timegrid0, imag(item), label=r"$\Im P$")
    for item in Phist1:
        ax.plot(timegrid1, imag(item), "c", label=r"$\Im P^s$")
    ax.grid(True)
    ax.set_title(r"$\Im P$")
    
    ax = fig.add_subplot(4,2,3)
    for item in Qhist0:
        ax.plot(timegrid0, real(item), label=r"$\Re Q$")
    for item in Qhist1:
        ax.plot(timegrid1, real(item), "c", label=r"$\Re Q^s$")
    ax.grid(True)
    ax.set_title(r"$\Re Q$")
    
    ax = fig.add_subplot(4,2,4)
    for item in Qhist0:
        ax.plot(timegrid0, imag(item), label=r"$\Im Q$")
    for item in Qhist1:
        ax.plot(timegrid1, imag(item), "c", label=r"$\Im Q^s$")
    ax.grid(True)
    ax.set_title(r"$\Im Q$")
    
    ax = fig.add_subplot(4,2,5)
    for item in qhist0:
        ax.plot(timegrid0, real(item), label=r"$q$")
    for item in qhist1:
        ax.plot(timegrid1, real(item), "c", label=r"$q^s$")
    ax.grid(True)
    ax.set_title(r"$q$")
    
    ax = fig.add_subplot(4,2,6)
    for item in phist0:
        ax.plot(timegrid0, real(item), label=r"$p$")
    for item in phist1:
        ax.plot(timegrid1, real(item), "c", label=r"$p^s$")
    ax.grid(True)
    ax.set_title(r"$p$")
    
    ax = fig.add_subplot(4,2,7)
    for item in Shist0:
        ax.plot(timegrid0, real(item), label=r"$S$")
    for item in Shist1:
        ax.plot(timegrid1, real(item), "c", label=r"$S^s$")
    ax.grid(True)
    ax.set_title(r"$S$")

    fig.suptitle("Wavepacket (spawned) parameters")
    fig.savefig("wavepacket_parameters_spawned.png")
    close(fig)


    # Plot the time evolution of the parameters P, Q, S, p and q
    # This time plot abs/angle instead of real/imag
    fig = figure(figsize=(12,12))

    ax = fig.add_subplot(4,2,1)
    for item in Phist0:
        ax.plot(timegrid0, abs(item), label=r"$|P|$")
    for item in Phist1:
        ax.plot(timegrid1, abs(item), "c", label=r"$|P^s|$")
    ax.grid(True)
    ax.set_title(r"$|P|$")
    
    ax = fig.add_subplot(4,2,2)
    for item in Phist0:
        ax.plot(timegrid0, ComplexMath.cont_angle(item), label=r"$\arg P$")
    for item in Phist1:
        ax.plot(timegrid1, ComplexMath.cont_angle(item), "c", label=r"$\arg P^s$")
    ax.grid(True)
    ax.set_title(r"$\arg P$")
    
    ax = fig.add_subplot(4,2,3)
    for item in Qhist0:
        ax.plot(timegrid0, abs(item), label=r"$|Q|$")
    for item in Qhist1:
        ax.plot(timegrid1, abs(item), "c", label=r"$|Q^s|$")
    ax.grid(True)
    ax.set_title(r"$|Q|$")
    
    ax = fig.add_subplot(4,2,4)
    for item in Qhist0:
        ax.plot(timegrid0, ComplexMath.cont_angle(item), label=r"$\arg Q$")
    for item in Qhist1:
        ax.plot(timegrid1, ComplexMath.cont_angle(item), "c", label=r"$\arg Q^s$")
    ax.grid(True)
    ax.set_title(r"$\arg Q$")
    
    ax = fig.add_subplot(4,2,5)
    for item in qhist0:
        ax.plot(timegrid0, real(item), label=r"$q$")
    for item in qhist1:
        ax.plot(timegrid1, real(item), "c", label=r"$q^s$")
    ax.grid(True)
    ax.set_title(r"$q$")
    
    ax = fig.add_subplot(4,2,6)
    for item in phist0:
        ax.plot(timegrid0, real(item), label=r"$p$")
    for item in phist1:
        ax.plot(timegrid1, real(item), "c", label=r"$p^s$")
    ax.grid(True)
    ax.set_title(r"$p$")
    
    ax = fig.add_subplot(4,2,7)
    for item in Shist0:
        ax.plot(timegrid0, real(item), label=r"$S$")
    for item in Shist1:
        ax.plot(timegrid1, real(item), "c", label=r"$S^s$")
    ax.grid(True)
    ax.set_title(r"$S$")

    fig.suptitle("Wavepacket (spawned) parameters")
    fig.savefig("wavepacket_parameters_abs_ang_spawned.png")
    close(fig)




if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    parameters = iom.get_parameters()

    if parameters["algorithm"] == "spawning_apost":
        data = read_data_spawn(iom)
    else:
        iom.finalize()
        sys.exit("Can only postprocess hagedorn algorithm data. Silent return ...")
        
    plot_parameters_spawn(*data)

    iom.finalize()
