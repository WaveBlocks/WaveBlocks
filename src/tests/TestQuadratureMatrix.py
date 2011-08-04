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


params = {}
params["eps"] = 0.2
params["basis_size"] = 7
params["ncomponents"] = 2

WP1 = HagedornWavepacket(params)
WP1.set_parameters((1.0j, 1.0, 0.0, 0.0, 0.85))
WP1.set_quadrature(None)

params["eps"] = 0.2
params["basis_size"] = 6
params["ncomponents"] = 3

WP2 = HagedornWavepacket(params)
WP2.set_parameters((1.0j, 1.0, 0.0, 0.0, 0.125))
WP2.set_quadrature(None)


HQ1 = HomogeneousQuadrature()
HQ1.build_qr(WP1.get_basis_size())
HQ2 = HomogeneousQuadrature()
HQ2.build_qr(WP2.get_basis_size())

IHQ = InhomogeneousQuadrature()
IHQ.build_qr(10)


M1 = HQ1.build_matrix(WP1)
M2 = HQ2.build_matrix(WP2)

figure()
matshow(real(M1))
savefig("M1r.png")

figure()
matshow(imag(M1))
savefig("M1i.png")


figure()
matshow(real(M2))
savefig("M2r.png")

figure()
matshow(imag(M2))
savefig("M2i.png")


M12 = IHQ.build_matrix(WP2,WP1)

figure()
matshow(real(M12))
savefig("M12r.png")

figure()
matshow(imag(M12))
savefig("M12i.png")
