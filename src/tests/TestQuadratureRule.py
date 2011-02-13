"""The WaveBlocks Project

Plot some quadrature rules.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import squeeze
from matplotlib.pyplot import *

from WaveBlocks import GaussHermiteQR


tests = (2, 3, 4, 7, 32, 64, 128)
    
for I in tests:

    Q = GaussHermiteQR(I)
    
    print(Q)

    N = Q.get_nodes()
    N = squeeze(N)
            
    W = Q.get_weights()
    W = squeeze(W)
            
    fig = figure()
    ax = fig.gca()
    
    ax.stem(N, W)

    ax.set_xlabel(r"$\gamma_i$")
    ax.set_ylabel(r"$\omega_i$")
    ax.set_title(r"Gauss-Hermite quadrature with $"+str(Q.get_number_nodes())+r"$ nodes")

    fig.savefig("qr_order_"+str(Q.get_order())+".png")
