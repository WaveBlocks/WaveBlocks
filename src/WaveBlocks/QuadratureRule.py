"""The WaveBlocks Project

This file contains the interface for general quadrature rules.
Do not confuse quadratures with quadrature rules! Quadrature rules
are structs containing just nodes and weights and some convenience
methods. Quadratures are classes that really can compute things
like inner products (brakets) etc.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

class QuadratureRule:
    """This class is an abstract interface to quadrature rules in general.
    """

    def __init__(self):
        """General interface for quadrature rules.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'QuadratureRule' is an abstract interface.")


    def __str__(self):
        raise NotImplementedError("'QuadratureRule' is an abstract interface.")


    def get_order(self):
        """:return: The order :math:`R` of the quadrature.
        """
        raise NotImplementedError("'QuadratureRule' is an abstract interface.")


    def get_number_nodes(self):
        """:return: The number of quadrature nodes.
        """
        raise NotImplementedError("'QuadratureRule' is an abstract interface.")


    def get_nodes(self):
        """:return: An array containing the quadrature nodes :math:`\gamma_i`.
        """
        raise NotImplementedError("'QuadratureRule' is an abstract interface.")


    def get_weights(self):
        """:return: An array containing the quadrature weights :math:`\omega_i`.
        """
        raise NotImplementedError("'QuadratureRule' is an abstract interface.")
