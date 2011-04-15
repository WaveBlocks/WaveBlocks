from numpy import *
from matplotlib.pyplot import *
from WaveBlocks.Plot import plotcf

a = linspace(0,2*pi,6000)
y = exp(1.0j*a)

fig = figure()
ax = fig.gca()

plotcf(a, angle(y), abs(y))
ax.plot(a, real(y), "b-", label=r"$\Re y$")
ax.plot(a, imag(y), "g-", label=r"$\Im y$")
ax.plot(a, angle(y), "c-", label=r"$\arg y$")

ax.set_xlim(0,2*pi)

ax.set_xticks((0,
               pi/4,
               pi/2,
               3*pi/4,
               pi,
               5*pi/4,
               3*pi/2,
               7*pi/4,
               2*pi))

ax.set_xticklabels((r"$0$",
                    r"$\frac{\pi}{4}$",
                    r"$\frac{\pi}{2}$",
                    r"$\frac{3\pi}{4}$",
                    r"$\pi$",
                    r"$\frac{5\pi}{4}$",
                    r"$\frac{3\pi}{2}$",
                    r"$\frac{7\pi}{4}$",
                    r"$2\pi$"))

ax.set_yticks((-pi,
               -1,
               0,
               1,
               pi))

ax.set_yticklabels((r"$-\pi$",
                    r"$-1$",
                    r"$0$",
                    r"$1$",
                    r"$\pi$"))

ax.grid(True)

fig.savefig("color_legend.png")

close(fig)
