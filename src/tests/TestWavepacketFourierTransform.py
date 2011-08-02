from numpy import *
from scipy import fft
from numpy.fft import fftshift
from matplotlib.pyplot import *

from WaveBlocks import HagedornWavepacket
from WaveBlocks.Plot import plotcf


params = {"eps":1,
          "basis_size":10,
          "ncomponents":1,
          "f":3,
          "ngn":500}

# Realspace grid
x = params["f"] * pi * arange(-1, 1, 2.0/params["ngn"], dtype=np.complexfloating)

# Fourierspace grid
omega_1 = arange(-params["ngn"]/2.0, 0, 1)
omega_2 = arange(0, params["ngn"]/2.0)
omega = hstack([omega_1, omega_2])

k = omega / (2*params["f"] * pi)
#k = omega / params["ngn"]


# A HAWP
HAWP = HagedornWavepacket(params)
HAWP.set_quadrature(None)

HAWP.set_parameters((1j, 1, 0, 3, -2))
HAWP.set_coefficient(0,2,1)

# Evaluate in real space
y = HAWP.evaluate_at(x)[0]

# Transform to Fourier space
HAWP.to_fourier_space()
# Evaluate in Fourier space
w = HAWP.evaluate_at(k)[0]

HAWP.to_real_space()
# Evaluate in real space again
z = HAWP.evaluate_at(x)[0]


figure()
plotcf(x, angle(y), abs(y))

figure()
plotcf(k, angle(w), abs(w))

figure()
plotcf(x, angle(z), abs(z))

show()
