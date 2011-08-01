"""The WaveBlocks Project

This file contains code for the homogeneous quadrature of wavepackets.
The class defined here can compute brakets, inner products and expectation
values and compute the $F$ matrix.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating, sum, matrix, squeeze
from scipy import conj, dot

from GaussHermiteQR import GaussHermiteQR


class HomogeneousQuadrature:

    def __init__(self, QR=None, order=None):
        if QR is not None:
            self.set_qr(QR)
        elif order is not None:
            self.build_qr(order)
        else:
            self.QR = None


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


    def transform_nodes(self, packet, QR=None):
        """Transform the quadrature nodes such that they fit the given wavepacket.
        """
        if QR is None:
            QR = self.QR

        nodes = packet.q + packet.eps * abs(packet.Q) * QR.get_nodes()
        return nodes.copy()


    def quadrature(self, packet, operator=None, summed=False):
        """Performs the quadrature of $\Braket{\Psi|f|\Psi}$ for a general $f$.
        @param function: A real-valued function $f(x):R \rightarrow R^{N \times N}.$
        @param summed: Whether to sum up the individual integrals $\Braket{\Phi_i|f_{i,j}|\Phi_j}$.
        @return: The value of $\Braket{\Psi|f|\Psi}$. This is either a scalar
        value or a list of $N^2$ scalar elements.
        """
        nodes = self.transform_nodes(packet)
        weights = self.QR.get_weights()
        basis = packet.evaluate_base_at(nodes)

        # Operator is None is interpreted as identity transformation
        if operator is None:
            values = nodes
        else:
            values = operator(nodes)

        N = packet.get_number_components()
        K = packet.get_basis_size()

        coeffs = packet.get_coefficients()

        result = []
        for i in xrange(N):
            for j in xrange(N):
                M = zeros((K,K), dtype=complexfloating)

                vals = values[i*N + j]
                factor = squeeze(packet.eps * weights * vals)

                # Summing up matrices over all quadrature nodes
                for k in xrange(self.QR.get_number_nodes()):
                    tmp = matrix(basis[:,k])
                    M += factor[k] * tmp.H * tmp

                # And include the coefficients as conj(c)*M*c
                result.append( dot(conj(coeffs[i]).T, dot(M,coeffs[j])) )

        if summed is True:
            result = sum(result)

        return result


    def build_matrix(self, packet, operator):
        """Calculate the matrix representation of $\Braket{\Psi|f|\Psi}$.
        @param function: A function with two arguments $f:(q, x) -> \mathbb{R}$.
        @return: A square matrix of size $NK \times NK$.
        """
        nodes = self.transform_nodes(packet)
        weights = self.QR.get_weights()
        basis = packet.evaluate_base_at(nodes)

        # Operator is None is interpreted as identity transformation
        if operator is None:
            values = nodes
        else:
            # Todo: operator should be only f(nodes)
            values = operator(packet.q, nodes)

        N = packet.get_number_components()
        K = packet.get_basis_size()

        result = zeros((N*K,N*K), dtype=complexfloating)

        for i in xrange(N):
            for j in xrange(N):
                M = zeros((K,K), dtype=complexfloating)
                vals = values[i*N + j]
                factor = squeeze(packet.eps * weights * vals)

                # Summing up matrices over all quadrature nodes
                for k in xrange(self.QR.get_number_nodes()):
                    tmp = matrix(basis[:,k])
                    M += factor[k] * tmp.H * tmp

                result[i*K:(i+1)*K, j*K:(j+1)*K] = M

        return result
