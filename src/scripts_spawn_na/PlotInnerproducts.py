"""The WaveBlocks Project

Plot some interesting values of the original and estimated
parameters sets Pi_m=(P,Q,S,p,q) and Pi_s=(B,A,S,b,a).
Plot the inner products of spawned and original packets.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import real, imag, abs, angle
from matplotlib.pyplot import *

from WaveBlocks import ComplexMath
from WaveBlocks import IOManager
from WaveBlocks import TrapezoidalQR
from WaveBlocks import HagedornWavepacket
from WaveBlocks import InhomogeneousQuadrature

import GraphicsDefaults as GD


def read_data_spawn(fo, fs):
    parameters_fo = fo.load_parameters()
    parameters_fs = fs.load_parameters()

    timegrids = []
    AllPA = []
    AllC = []


    timegrids.append(parameters_fo["dt"] * fo.load_wavepacket_timegrid(blockid=0))

    Pi = fo.load_wavepacket_parameters(blockid=0)
    Phist = Pi[:,0]
    Qhist = Pi[:,1]
    Shist = Pi[:,2]
    phist = Pi[:,3]
    qhist = Pi[:,4]
    AllPA.append([Phist, Qhist, Shist, phist, qhist])

    Ci = fo.load_wavepacket_coefficients(blockid=0)
    AllC.append(Ci)


    timegrids.append(parameters_fs["dt"] * fs.load_wavepacket_timegrid(blockid=1))

    Pi = fs.load_wavepacket_parameters(blockid=1)
    Phist = Pi[:,0]
    Qhist = Pi[:,1]
    Shist = Pi[:,2]
    phist = Pi[:,3]
    qhist = Pi[:,4]
    AllPA.append([Phist, Qhist, Shist, phist, qhist])

    Ci = fs.load_wavepacket_coefficients(blockid=1)
    AllC.append(Ci)

    return parameters_fo, parameters_fs, timegrids, AllPA, AllC


def compute(parameters_fo, parameters_fs, timegrids, AllPA, AllC):
    # Grid of mother and first spawned packet
    grid_m = timegrids[0]
    grid_s = timegrids[1]

    # Parameters of the original packet
    P, Q, S, p, q = AllPA[0]

    # Parameter of the spawned packet, first try
    B, A, S, b, a = AllPA[1]

    # Parameter of the spawned packet, second try
    A2, S2, b2, a2 = A, S, b, a
    B2 = -real(B)+1.0j*imag(B)

    C0 = AllC[0]
    C1 = AllC[1]

    # Construct the packets from the data
    OWP = HagedornWavepacket(parameters_fo)
    OWP.set_quadrature(None)

    S1WP = HagedornWavepacket(parameters_fs)
    S1WP.set_quadrature(None)

    S2WP = HagedornWavepacket(parameters_fs)
    S2WP.set_quadrature(None)

    nrtimesteps = grid_m.shape[0]

    # The quadrature
    quadrature = InhomogeneousQuadrature()

    # Quadrature, assume same quadrature order for both packets
    # Assure the "right" quadrature is choosen if OWP and S*WP have
    # different basis sizes
    if OWP.get_basis_size() > S1WP.get_basis_size():
        quadrature.set_qr(OWP.get_quadrature().get_qr())
    else:
        quadrature.set_qr(S1WP.get_quadrature().get_qr())

    ip_oo = []
    ip_os1 = []
    ip_os2 = []
    ip_s1s1 = []
    ip_s2s2 = []

    # Inner products
    for step in xrange(nrtimesteps):
        print("Timestep "+str(step))

        # Put the data from the current timestep into the packets
        OWP.set_parameters((P[step], Q[step], S[step], p[step], q[step]))
        OWP.set_coefficients(C0[step,...])

        S1WP.set_parameters((B[step], A[step], S[step], b[step], a[step]))
        S1WP.set_coefficients(C1[step,...])

        S2WP.set_parameters((B2[step], A2[step], S2[step], b2[step], a2[step]))
        S2WP.set_coefficients(C1[step,...])

        # Compute the inner products
        ip_oo.append(quadrature.quadrature(OWP, OWP, summed=True))
        ip_os1.append(quadrature.quadrature(OWP, S1WP, summed=True))
        ip_os2.append(quadrature.quadrature(OWP, S2WP, summed=True))
        ip_s1s1.append(quadrature.quadrature(S1WP, S1WP, summed=True))
        ip_s2s2.append(quadrature.quadrature(S2WP, S2WP, summed=True))

    # Plot
    figure()
    plot(grid_m, abs(ip_oo), label=r"$\langle O|O\rangle $")
    plot(grid_m, abs(ip_os1), "-*", label=r"$\langle O|S1\rangle $")
    plot(grid_m, abs(ip_os2), "-", label=r"$\langle O|S2\rangle $")
    plot(grid_m, abs(ip_s1s1), label=r"$\langle S1|S1\rangle $")
    plot(grid_m, abs(ip_s2s2), label=r"$\langle S2|S2\rangle $")
    legend()
    grid(True)
    savefig("inner_products"+GD.output_format)
    close()




if __name__ == "__main__":
    iom_s = IOManager()
    iom_o = IOManager()

    # NOTE
    #
    # first cmd-line data file is spawning data
    # second cmd-line data file is reference data

    # Read file with new simulation data
    try:
        iom_s.open_file(filename=sys.argv[1])
    except IndexError:
        iom_s.open_file()

    # Read file with original reference simulation data
    try:
        iom_o.open_file(filename=sys.argv[2])
    except IndexError:
        iom_o.open_file()

    compute(*read_data_spawn(iom_o, iom_s))

    iom_s.finalize()
    iom_o.finalize()
