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
    r"""
    This class is an abstract interface to quadratures in general.
    """

    def __init__(self):
        r"""
        General interface for quadratures.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def __str__(self):
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def set_qr(self, QR):
        r"""
        Set the ``GaussHermiteQR`` instance used for quadrature.

        :param QR: The new ``GaussHermiteQR`` instance.
        """
        self.QR = QR


    def get_qr(self):
        r"""
        Return the ``GaussHermiteQR`` instance used for quadrature.

        :return: The current instance of the quadrature rule.
        """
        return self.QR


    def build_qr(self, qorder):
        r"""
        Create a quadrature rule of the given order.

        :param qorder: The order of the quadrature rule.
        """
        self.QR = GaussHermiteQR(qorder)


    def transform_nodes(self):
        r"""
        Transform the quadrature nodes such that they fit the given wavepacket.

        :raise NotImplementedError: Abstract interface.

        .. note:: Arguments may vary through subclasses!
        """
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def quadrature(self):
        r"""
        Performs the quadrature of :math:`\langle\Psi|f|\Psi\rangle` for a general :math:`f`.

        :raise NotImplementedError: Abstract interface.

        .. note:: Arguments may vary through subclasses!
        """
        raise NotImplementedError("'Quadrature' is an abstract interface.")


    def build_matrix(self):
        r"""
        Calculate the matrix representation of :math:`\langle\Psi|f|\Psi\rangle`.

        :raise NotImplementedError: Abstract interface.

        .. note:: Arguments may vary through subclasses!
        """
        raise NotImplementedError("'Quadrature' is an abstract interface.")
