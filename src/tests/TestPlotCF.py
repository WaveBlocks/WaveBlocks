"""The WaveBlocks Project

Test the complex phase plot function.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np
from matplotlib.pyplot import *

from WaveBlocks.plot import plotcf

x = np.r_[-1.:1.:1j*2**12]
u = np.exp(-x**2)*(np.cos(10*x) + 1j *np.sin(10*x))

rvals = np.real(u)
ivals = np.imag(u)
cvals = np.conjugate(u)*u
angles = np.angle(u)

figure(figsize=(20,20))

subplot(2,2,1)
plotcf(x, angles, rvals)
xlabel(r"$\Re \psi$")

subplot(2,2,2)
plotcf(x, angles, ivals)
xlabel(r"$\Im \psi$")

subplot(2,2,3)
plotcf(x, angles, cvals)
xlabel(r"$|\psi|^2$")

savefig("phaseplot.png")

show()
