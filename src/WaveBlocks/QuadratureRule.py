"""The WaveBlocks Project

This file contains the interface for general quadrature rules.

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
        """@return: The order $R$ of the quadrature.
        """
        raise NotImplementedError("'QuadratureRule' is an abstract interface.")
        
        
    def get_number_nodes(self):
        """@return: The number of quadrature nodes.
        """
        raise NotImplementedError("'QuadratureRule' is an abstract interface.")
    
    
    def get_nodes(self):
        """@return: An array containing the quadrature nodes $\gamma_i$.
        """
        raise NotImplementedError("'QuadratureRule' is an abstract interface.")
    
    
    def get_weights(self):
        """@return: An array containing the quadrature weights $\omega_i$.
        """
        raise NotImplementedError("'QuadratureRule' is an abstract interface.")
