"""The WaveBlocks Project

This file contains the class for Gauss-Hermite quadrature.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating
from scipy import pi, exp, sqrt
from scipy.special.orthogonal import h_roots

from QuadratureRule import QuadratureRule


class GaussHermiteQR(QuadratureRule):
    """This class implements a Gauss-Hermite quadrature rule
    tailored at the needs of Hagedorn wavepackets.
    """

    def __init__(self, order):
        """Initialize a new quadrature rule.
        @param order: The order $R$ of the Gauss-Hermite quadrature.
        @raise ValueError: If the order is less then 2.
        """
        #: The order $R$ of the Gauss-Hermite quadrature.
        self.order = order

        # The minimal order of 2 has no mathematical reasons but is rather of
        # technical nature. The recursive evaluation of Hermite functions
        # needs two base cases for the induction steps.
        if self.order < 2:
            raise ValueError("Quadrature rule has to be of order 2 at least.")

        nodes, weights = h_roots(self.order)

        self.number_nodes = nodes.size

        h = self.hermite_recursion(nodes)
        weights = 1.0/((h**2) * self.order)

        #: The quadrature nodes $\gamma_i$.
        self.nodes = nodes.reshape((1,self.number_nodes))
        #: The quadrature weights $\omega_i$.
        self.weights = weights[-1,:]
        self.weights = self.weights.reshape((1,self.number_nodes))


    def __str__(self):
        return "Gauss-Hermite quadrature rule of order " + str(self.order) + "."


    def get_order(self):
        """@return: The order $R$ of the quadrature.
        """
        return self.order


    def get_number_nodes(self):
        """@return: The number of quadrature nodes.
        """
        return self.number_nodes


    def get_nodes(self):
        """@return: An array containing the quadrature nodes $\gamma_i$.
        """
        return self.nodes.copy()


    def get_weights(self):
        """@return: An array containing the quadrature weights $\omega_i$.
        """
        return self.weights.copy()


    def hermite_recursion(self, nodes):
        """Evaluate the Hermite functions recursively up to the order $R$ on the given nodes.
        @param nodes: The points at which the Hermite functions are evaluated.
        @return: Returns a twodimensional array $H$ where the entry $H[k,i]$ is the value
        of the $k$-th Hermite function evaluated at the node $i$.
        """
        H = zeros((self.order, nodes.size), dtype=complexfloating)

        H[0] = pi**(-0.25) * exp(-0.5*nodes**2)
        H[1] = sqrt(2.0) * nodes * H[0]

        for k in xrange(2, self.order):
            H[k] = sqrt(2.0/k) * nodes * H[k-1] - sqrt((k-1.0)/k) * H[k-2]

        return H
