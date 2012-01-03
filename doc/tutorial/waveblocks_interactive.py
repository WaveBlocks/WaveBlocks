# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

from numpy import *
from matplotlib.pyplot import *

%load_ext sympyprt

# <markdowncell>

# This is an example session showing the interactive use of the WaveBlocks simulation packet.

# <codecell>

from WaveBlocks import *

# <markdowncell>

# Some simulation parameters:

# <codecell>

params = {"eps":0.1, "ncomponents":1, "potential":"quadratic", "dt":0.1, "matrix_exponential":"pade"}

# <markdowncell>

# Create a Hagedorn wavepacket $\Psi$

# <codecell>

Phi = HagedornWavepacket(params)

# <markdowncell>

# Assign the parameter set $\Pi$ with position 1.0 and momentum 0.5

# <codecell>

Pi = Phi.get_parameters()
Pi

# <codecell>

Pi = list(Pi)
Pi[3] = 0.5
Pi[4] = 1.0
Pi

# <codecell>

Phi.set_parameters(Pi)

# <markdowncell>

#  Set the coefficients such that we start with a $\phi_1$ packet

# <codecell>

Phi.set_coefficient(0,1,1)

# <markdowncell>

# Plot the initial configuration

# <codecell>

x = linspace(0.5,1.5,1000)

# <codecell>

y = Phi.evaluate_at(x, prefactor=True)[0]

# <codecell>

plot(x, abs(y)**2)

# <codecell>

from WaveBlocks.Plot import plotcf, stemcf

# <codecell>

plotcf(x, angle(y), abs(y)**2)

# <codecell>

c = Phi.get_coefficients(component=0)
c = squeeze(c)

# <codecell>

c.shape

# <codecell>

figure(figsize=(6,4))
stemcf(arange(c.shape[0]), angle(c), abs(c)**2)

# <markdowncell>

# Set up the potential $V(x)$ for our simulation. We use a simple harmonic oscillator.

# <codecell>

V = PotentialFactory.create_potential(params)

# <codecell>

V.potential

# <codecell>

u = linspace(-2,2,1000)
v = V.evaluate_at(u)[0]

# <codecell>

plot(u,v)
xlabel(r"$x$")
ylabel(r"$V(x)$")

# <markdowncell>

# Don't forget to set up the quadratur rule $(\gamma, \omega)$

# <codecell>

Phi.set_quadrature(None)

# <markdowncell>

# Now construct the time propagator

# <codecell>

P = HagedornPropagator(V, Phi, 0, params)

# <markdowncell>

# Propagate for 16 timesteps and plot each state

# <codecell>

fig = figure(figsize=(14,14))

for i in xrange(16):
    P.propagate()
    ynew = P.get_wavepackets().evaluate_at(x, prefactor=True)[0]
    ax = subplot(4,4,i+1)
    plotcf(x, angle(ynew), abs(ynew)**2, axes=ax)
    ax.set_ylim((-0.5, 5))

# <markdowncell>

# Look at the coefficients $c$ again

# <codecell>

cnew = Phi.get_coefficients(component=0)
cnew = squeeze(cnew)

# <codecell>

figure(figsize=(6,4))
stemcf(arange(8), angle(cnew), abs(c)**2)

# <markdowncell>

# We see that the packet $\Psi$ is still a $\phi_1$

# <markdowncell>

# Now we go back in time ...

# <codecell>

params["dt"] *= -1

# <codecell>

Pinv = HagedornPropagator(V, Phi, 0, params)

# <codecell>

fig = figure(figsize=(14,14))

for i in xrange(16):
    Pinv.propagate()
    ynew = Pinv.get_wavepackets().evaluate_at(x, prefactor=True)[0]
    ax = subplot(4,4,i+1)
    plotcf(x, angle(ynew), abs(ynew)**2, axes=ax)
    ax.set_ylim((-0.5,5))

# <codecell>

Phi.get_parameters()

# <markdowncell>

# We see that the propagation is reversible up to machine precision!

