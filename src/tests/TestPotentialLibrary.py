"""The WaveBlocks Project

Test the potential library, plot all the potential which
are available in the PotentialLibrary.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import *
from matplotlib.pyplot import *

from WaveBlocks import PotentialFactory
import WaveBlocks.PotentialLibrary as PL

pots =  [ i for i in dir(PL) if not i.startswith("_") ]

params = {"eps":0.2, "delta":0.2, "delta1":0.2, "delta2":0.2}

x = linspace(-5,5,5000)

for pot in pots:
    print("Potential is: " + pot)
    
    params["potential"] = pot
    P = PotentialFactory.create_potential(params)
    y = P.evaluate_eigenvalues_at(x)

    figure()
    for yvals in y:
        plot(x, yvals)
    xlabel(r"$x$")
    ylabel(r"$\lambda_i\left(x\right)$")
    title(pot.replace("_","-"))
    savefig(pot + ".png")
