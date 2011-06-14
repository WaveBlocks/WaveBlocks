"""The WaveBlocks Project

Plot function for plotting functions of the type f:R -> C
with abs(f) as y-value and phase(f) as color code.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import pi, empty, array, sign
from matplotlib.colors import hsv_to_rgb
from matplotlib.collections import LineCollection
from matplotlib.pyplot import gca


def plotcf(grid, phase, modulus, axes=None, linestylep="solid", linewidthp=1, color="k", **kwargs):
    """Plot the modulus of a complex valued function $f:R -> C$ together with its phase in a color coded fashion.
    @param grid: The grid nodes of the real domain R
    @param phase: The phase of the complex domain result f(grid)
    @param modulus: The modulus of the complex domain result f(grid)
    @keyword axes: The axes instance used for plotting.
    @keyword linestylep: The line style of the phase curve.
    @keyword linewidthp: The line width of the phase curve.
    @keyword color: The color of the phase curve.
    @note: Additional keyword arguments are passe to the plot function.
    """
    # Color mapping
    hsv_colors = empty((1, len(grid), 3))
    hsv_colors[:, :, 0] = 0.5*(phase+(1-sign(phase))*pi)/pi
    hsv_colors[:, :, 1] = 1.0
    hsv_colors[:, :, 2] = 1.0
    rgb_colors = hsv_to_rgb(hsv_colors)

    # Put all the vertical line into a collection
    segments = [ array([[node,0], [node,value]]) for node, value in zip(grid, modulus) ]
    line_segments = LineCollection(segments)

    # Set some properties of the lines
    rgb_colors = line_segments.to_rgba(rgb_colors)
    line_segments.set_color(rgb_colors[0])
    line_segments.set_linestyle(linestylep)
    line_segments.set_linewidth(linewidthp)

    # Plot to the given axis instance or retrieve the current one
    if axes is None:
        axes = gca()

    # Plot the phase
    axes.add_collection(line_segments)
    # Plot the modulus
    axes.plot(grid, modulus, color=color, **kwargs)
