"""The WaveBlocks Project

This file contains some global defaults
all related to advanced plotting.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import itertools
import matplotlib.pyplot

# Possible line colors, styles and markers
styles = ['--', ':', '-.', '_']
colors = ['r', 'g', 'b', 'c', 'm', 'k']
markers = ['<' , '>' , 'D' , 's' , '^'  , 'd' , 'v' , 'x']

# Cyclic lists of all line colors, styles and markers
styles = itertools.cycle(styles)
colors = itertools.cycle(colors)
markers = itertools.cycle(markers)

# Put a marker every n-th item
marker_every = 50

# Output plot file format
output_format = ".png"

# Default matplotlib parameters
gfx_params = {
    'backend': 'png',
    'axes.labelsize': 20,
    'text.fontsize': 20,
    'legend.fontsize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'axes.titlesize': 30,
    'text.usetex': True,
    'lines.markeredgewidth' : 2,
    'lines.markersize': 8, 
    'lines.linewidth': 2.0
    #'figure.figsize': fig_size
    }

# And really set the parameters upon import of this file
matplotlib.pyplot.rcParams.update(gfx_params)
