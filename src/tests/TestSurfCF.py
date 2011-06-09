"""The WaveBlocks Project

Test the complex phase surface plot function.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np
from enthought.mayavi import mlab

from WaveBlocks.Plot import surfcf


x, y = np.mgrid[-2*np.pi:2*np.pi:0.05, -2*np.pi:2*np.pi:0.05]

# Plane waves:
k = 5
l = 3
z = np.exp(-1.0j*k*x)*np.exp(-1.0j*l*y)

surfcf(x, y, np.angle(z), np.real(z))
mlab.savefig("plane_wave_real.png")
mlab.close()

surfcf(x, y, np.angle(z), np.imag(z))
mlab.savefig("plane_wave_imag.png")
mlab.close()

surfcf(x, y, np.angle(z), np.conj(z)*z)
mlab.savefig("plane_wave_abs.png")
mlab.close()
