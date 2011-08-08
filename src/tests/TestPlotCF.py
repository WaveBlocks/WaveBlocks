"""The WaveBlocks Project

Test the complex phase plot function.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np
from matplotlib.pyplot import *

from WaveBlocks.Plot import plotcf

x = np.r_[0.0:2.0*np.pi:1j*2**9]
u = np.exp(1.0j*x)

rvals = np.real(u)
ivals = np.imag(u)
cvals = np.conjugate(u)*u
angles = np.angle(u)

figure(figsize=(20,20))

subplot(2,2,1)
plotcf(x, angles, rvals)
xlim([0,2*np.pi])
ylim([-1.5, 1.5])
xlabel(r"$\Re \psi$")

subplot(2,2,2)
plotcf(x, angles, ivals, color="k")
xlim([0,2*np.pi])
ylim([-1.5, 1.5])
xlabel(r"$\Im \psi$")

subplot(2,2,3)
plotcf(x, angles, cvals, darken=True)
xlim([0,2*np.pi])
ylim([0, 1.5])
xlabel(r"$|\psi|^2$")

savefig("planewave_plot.png")

show()
