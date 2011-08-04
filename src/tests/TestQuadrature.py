"""The WaveBlocks Project

Compute mixing quadrature for two wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import *
from matplotlib.pyplot import *

from WaveBlocks import HagedornWavepacket
from WaveBlocks import HomogeneousQuadrature
from WaveBlocks import InhomogeneousQuadrature

nmax = 6

params = {}
params["eps"] = 0.2
params["basis_size"] = 4
params["ncomponents"] = 1


WP1 = HagedornWavepacket(params)
WP1.set_parameters((1.0j, 1.0, 0.0, 0.0, 0.85))
WP1.set_coefficient(0, 0, 1)
WP1.set_quadrature(None)

params["basis_size"] = 6

WP2 = HagedornWavepacket(params)
WP2.set_coefficient(0, 0, 1)
WP2.set_quadrature(None)


HQ1 = HomogeneousQuadrature()
HQ1.build_qr(WP1.get_basis_size())
HQ2 = HomogeneousQuadrature()
HQ2.build_qr(WP2.get_basis_size())

IHQ = InhomogeneousQuadrature()
IHQ.build_qr(nmax)


Pibra = WP1.get_parameters()

x = linspace(-4,4,4000)
positions = linspace(-0.5,2.5,61)

quads1 = []
quads2 = []
quads12 = []

for index, pos in enumerate(positions):
    print(pos)
    # Moving Gaussian
    WP2.set_parameters((1.0j, 1.0, 0.0, 0.0, pos))

    # Transform the nodes
    nodes1 =  squeeze(HQ1.transform_nodes(WP1.get_parameters(), WP1.eps))
    nodes2 =  squeeze(HQ2.transform_nodes(WP2.get_parameters(), WP2.eps))
    nodes12 = squeeze(IHQ.transform_nodes(Pibra, WP2.get_parameters(), WP1.eps))

    # Compute inner products
    Q1 = IHQ.quadrature(WP1, WP1, summed=True)
    Q2 = IHQ.quadrature(WP2, WP2, summed=True)
    Q12 = IHQ.quadrature(WP1, WP2, summed=True)

    quads1.append(Q1)
    quads2.append(Q2)
    quads12.append(Q12)

    # Evaluate the packets
    y = WP1.evaluate_at(x, prefactor=True, component=0)
    z = WP2.evaluate_at(x, prefactor=True, component=0)

    figure()
    plot(x, conj(y)*y, "b")
    plot(x, conj(z)*z, "g")
    plot(x, conj(y)*z, "r")
    plot(nodes1, zeros(nodes1.shape), "ob")
    plot(nodes2, zeros(nodes2.shape), "og")
    plot(nodes12, zeros(nodes12.shape), "or")
    ylim([-0.2, 3])
    title(r"$\langle \Psi_1 | \Psi_2 \rangle$ = "+str(Q12))
    savefig("mixed_quadrature_"+(5-len(str(index)))*"0"+str(index)+".png")
    close()


figure()
plot(positions, quads1, "b", label=r"$\langle \Psi_1 | \Psi_1 \rangle$")
plot(positions, quads2, "g", label=r"$\langle \Psi_2 | \Psi_2 \rangle$")
plot(positions, quads12, "r", label=r"$\langle \Psi_1 | \Psi_2 \rangle$")
legend()
xlabel(r"$x$")
ylabel(r"$\langle \Psi | \Psi \rangle$")
savefig("mixing_quadrature_values.png")
close()
