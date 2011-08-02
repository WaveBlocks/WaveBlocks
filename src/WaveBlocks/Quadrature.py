"""The WaveBlocks Project

This file contains the interface for general quadratures.
Do not confuse quadratures with quadrature rules! Quadrature rules
are structs containing just nodes and weights and some convenience
methods. Quadratures are classes that really can compute things
like inner products (brakets) etc.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from GaussHermiteQR import GaussHermiteQR


class Quadrature:
    """This class is an abstract interface to quadratures in general.
    """

    def __init__(self):
        """General interface for quadratures.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def __str__(self):
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def set_qr(self, QR):
        """Set the I{GaussHermiteQR} instance used for quadrature.
        @param QR: The new I{GaussHermiteQR} instance.
        """
        self.QR = QR


    def get_qr(self):
        """Return the I{GaussHermiteQR} instance used for quadrature.
        @return: The current instance of the quadrature rule.
        """
        return self.QR


    def build_qr(self, qorder):
        """Create a quadrature rule of the given order.
        @param qorder: The order of the quadrature rule.
        """
        self.QR = GaussHermiteQR(qorder)


    def transform_nodes(self):
        """Transform the quadrature nodes such that they fit the given wavepacket.
        @note: Arguments may vary through subclasses!
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def quadrature(self):
        """Performs the quadrature of $\Braket{\Psi|f|\Psi}$ for a general $f$.
        @note: Arguments may vary through subclasses!
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def build_matrix(self):
        """Calculate the matrix representation of $\Braket{\Psi|f|\Psi}$.
        @note: Arguments may vary through subclasses!
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Quadrature' is an abstract interface.")
