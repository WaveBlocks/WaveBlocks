"""The WaveBlocks Project

Plot the eigenvalues (energy levels) of the potential.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from matplotlib.pyplot import *

from WaveBlocks import PotentialFactory
from WaveBlocks import IOManager
from WaveBlocks.plot import legend


def plot_potential(grid, potential, imgsize=(8,6)):
    # Create potential and evaluate eigenvalues
    potew = potential.evaluate_eigenvalues_at(grid)

    # Plot the energy surfaces of the potential
    fig = figure()
    ax = fig.gca()
    
    for index, ew in enumerate(potew):
        ax.plot(grid, ew, label=r"$\lambda_"+str(index)+r"$")
    
    ax.grid(True)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\lambda_i\left(x\right)$")
    legend(loc="outer right")
    ax.set_title(r"The eigenvalues $\lambda_i$ of the potential $V\left(x\right)$")
    fig.savefig("potential.png")


if __name__ == "__main__":
    iom = IOManager()
    
    # Read file with simulation data
    try:
        iom.load_file(filename=sys.argv[1])
    except IndexError:
        iom.load_file()
    
    parameters = iom.get_parameters()
    potential = PotentialFactory.create_potential(parameters)
    grid = iom.load_grid()

    plot_potential(grid, potential)

    iom.finalize()
