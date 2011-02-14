"""The WaveBlocks Project

Build an artificial wavepacket.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import *
from matplotlib.pyplot import *

from WaveBlocks import HagedornWavepacket
from WaveBlocks.plot import plotcf

nmax = 10
amp0 = 0.5

params = {}
params["eps"] = 0.2
params["basis_size"] = nmax
params["ncomponents"] = 1

HAWP = HagedornWavepacket(params)
HAWP.set_parameters((1.0j, 1.0, 0.0, -1.0, 0.0))

for i in range(0,nmax):
    HAWP.set_coefficient(0, i, amp0/ 2**i)

x = linspace(-4,4,4000)
y = HAWP.evaluate_at(x, prefactor=True, component=0)

figure()
plotcf(x, angle(y), conj(y)*y)
show()
