"""The WaveBlocks Project

Test the complex math functions.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import *
from matplotlib.pyplot import *

from WaveBlocks.ComplexMath import *
from WaveBlocks.Plot import plotcf


# Continuous complex angle function
a = linspace(0,5,1000)
b = linspace(5,10,1000)

c = hstack([a, b])
y = hstack([ exp(-1.0j*a**2), exp(+1.0j*b**1.6) ])

figure()

plotcf(c, angle(y), abs(y))
plot(c, real(y), "b-", label=r"$\Re y$")
plot(c, imag(y), "g-", label=r"$\Im y$")
plot(c, angle(y), "c-", label=r"$\arg y$")
plot(c, cont_angle(y), "m-", label=r"$\arg_c y$")
plot(c, pi*ones(c.shape), "y--", label=r"$\pi$")
plot(c, -pi*ones(c.shape), "y--", label=r"$-\pi$")

legend(loc="lower left")
savefig("complex_angle_continuous.png")


# Continuous complex sqrt
x = linspace(0, 6*pi, 5000)
y = 2*exp(1.0j*x)


figure()
polar(x, abs(y))
savefig("complex_numbers.png")


# Do it "wrong"
z = sqrt(y)

figure()

subplot(2,1,1)
plotcf(x, angle(y), abs(y))
plot(x, real(y), "-b")
plot(x, imag(y), "-g")
grid(True)
ylim([-2.1,2.1])
#xlabel(r"$\phi$")
ylabel(r"$2 \cdot \exp(i \cdot \phi)$")
title(r"$z = r \cdot \exp(i \cdot \phi)$")

subplot(2,1,2)
plotcf(x, angle(z), abs(z))
plot(x, real(z), "-b")
plot(x, imag(z), "-g")
grid(True)
ylim([-2.1,2.1])
xlabel(r"$\phi$")
ylabel(r"$\sqrt{2 \cdot \exp(i \cdot \phi)}$")
title(r"$\sqrt{z} = \sqrt{r} \cdot \exp \left( i \cdot \frac{\phi}{2} \right)$")

savefig("complex_sqrt_non-continuous.png")


# Do it "right"
z = cont_sqrt(y)

figure()

subplot(2,1,1)
plotcf(x, angle(y), abs(y))
plot(x, real(y), "-b")
plot(x, imag(y), "-g")
grid(True)
ylim([-2.1,2.1])
#xlabel(r"$\phi$")
ylabel(r"$2 \cdot \exp(i \cdot \phi)$")
title(r"$z = r \cdot \exp(i \cdot \phi)$")

subplot(2,1,2)
plotcf(x, angle(z), abs(z))
plot(x, real(z), "-b")
plot(x, imag(z), "-g")
grid(True)
ylim([-2.1,2.1])
xlabel(r"$\phi$")
ylabel(r"$\sqrt{2 \cdot \exp(i \cdot \phi)}$")
title(r"$\sqrt{z} = \sqrt{r} \cdot \exp \left( i \cdot \frac{\phi}{2} \right)$")

savefig("complex_sqrt_continuous.png")


# Another example
x = linspace(0, 6*pi, 5000)
y = 0.4*x*exp(1.0j*x)


figure()
polar(x,abs(y))
savefig("complex_numbers2.png")


# Do it "right"
z = cont_sqrt(y)

figure()

subplot(2,1,1)
plotcf(x, angle(y), abs(y))
plot(x, real(y), "-b")
plot(x, imag(y), "-g")
grid(True)
#xlabel(r"$\phi$")
ylabel(r"$2 \cdot \exp(i \cdot \phi)$")
title(r"$z = r \cdot \exp(i \cdot \phi)$")

subplot(2,1,2)
plotcf(x, angle(z), abs(z), darken=True)
plot(x, real(z), "-b")
plot(x, imag(z), "-g")
grid(True)
xlabel(r"$\phi$")
ylabel(r"$\sqrt{2 \cdot \exp(i \cdot \phi)}$")
title(r"$\sqrt{z} = \sqrt{r} \cdot \exp \left( i \cdot \frac{\phi}{2} \right)$")

savefig("complex_sqrt_continuous2.png")
