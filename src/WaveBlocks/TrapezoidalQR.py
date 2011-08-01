"""The WaveBlocks Project

This file contains the class for trapezoidal quadrature rules.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import linspace, abs, ones

from QuadratureRule import QuadratureRule


class TrapezoidalQR(QuadratureRule):
    """This class implements a Trapezoidal quadrature rule.
    """

    def __init__(self, left, right, order):
        """Initialize a new quadrature rule.
        @param order: The order $R$ of the trapezoidal quadrature rule.
        @raise ValueError: If the order is less then 2.
        """
        #: The order $R$ of the trapezoidal quadrature rule.
        self.order = order

        # The minimal order of 2 has no mathematical reasons but is rather of
        # technical nature.
        if self.order < 2:
            raise ValueError("Quadrature rule has to be of order 2 at least.")

        self.left = left
        self.right = right
        self.compute_qr()


    def __str__(self):
        return "Trapezoidal quadrature rule of order " + str(self.order) + "."


    def compute_qr(self):
        nodes = linspace(self.left, self.right, self.order)
        self.number_nodes = nodes.size
        dx = abs(self.right-self.left) / self.number_nodes
        weights = ones(nodes.shape)
        weights[0] = 0.5
        weights[-1] = 0.5
        weights = dx * weights
        self.nodes = nodes.reshape((1, nodes.size))
        self.weights = weights.reshape((1, weights.size))


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
