"""The WaveBlocks Project

This file contains code for the homogeneous quadrature of wavepackets.
The class defined here can compute brakets, inner products and expectation
values and compute the $F$ matrix.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating, sum, matrix, squeeze, ones
from scipy import conj, dot

from Quadrature import Quadrature


class HomogeneousQuadrature(Quadrature):

    def __init__(self, QR=None, order=None):
        if QR is not None:
            self.set_qr(QR)
        elif order is not None:
            self.build_qr(order)
        else:
            self.QR = None


    def __str__(self):
        return "Homogeneous quadrature using a " + str(self.QR)


    def transform_nodes(self, Pi, eps, QR=None):
        """Transform the quadrature nodes such that they fit the given wavepacket.
        @param Pi: The parameter set of the wavepacket.
        @param eps: The epsilon of the wavepacket.
        @keyword QR: An optional quadrature rule providing the nodes.
        """
        if QR is None:
            QR = self.QR

        P, Q, S, p, q = Pi

        nodes = q + eps * abs(Q) * QR.get_nodes()
        return nodes.copy()


    def quadrature(self, packet, operator=None, summed=False, component=None):
        """Performs the quadrature of $\Braket{\Psi|f|\Psi}$ for a general $f$.
        @param packet: The wavepacket $|\Psi>$.
        @param operator: A real-valued function $f(x):R \rightarrow R^{N \times N}.$
        @param summed: Whether to sum up the individual integrals $\Braket{\Phi_i|f_{i,j}|\Phi_j}$.
        @return: The value of $\Braket{\Psi|f|\Psi}$. This is either a scalar
        value or a list of $N^2$ scalar elements.
        """
        nodes = self.transform_nodes(packet.get_parameters(), packet.eps)
        weights = self.QR.get_weights()
        basis = packet.evaluate_basis_at(nodes)

        # Operator is None is interpreted as identity transformation
        if operator is None:
            values = []
            for row in xrange(N):
                for col in xrange(N):
                    if row == col:
                        values.append(ones(nodes.shape))
                    else:
                        values.append(zeros(nodes.shape))
        else:
            values = operator(nodes)

        N = packet.get_number_components()
        K = packet.get_basis_size()

        coeffs = packet.get_coefficients()

        result = []
        for i in xrange(N):
            for j in xrange(N):
                M = zeros((K,K), dtype=complexfloating)
                factor = squeeze(packet.eps * weights * values[i*N + j])

                # Summing up matrices over all quadrature nodes
                for k in xrange(self.QR.get_number_nodes()):
                    tmp = matrix(basis[:,k])
                    M += factor[k] * tmp.H * tmp

                # And include the coefficients as conj(c)*M*c
                result.append(dot(conj(coeffs[i]).T, dot(M,coeffs[j])))

        # Todo: improve to avoid unnecessary computations of other components
        if component is not None:
            result = result[component]
        elif summed is True:
            result = sum(result)

        return result


    def build_matrix(self, packet, operator=None):
        """Calculate the matrix representation of $\Braket{\Psi|f|\Psi}$.
        @param packet: The wavepacket $|\Psi>$.
        @param operator: A function with two arguments $f:(q, x) -> \mathbb{R}$.
        @return: A square matrix of size $N*K \times N*K$.
        """
        nodes = self.transform_nodes(packet.get_parameters(), packet.eps)
        weights = self.QR.get_weights()
        basis = packet.evaluate_basis_at(nodes)

        N = packet.get_number_components()
        K = packet.get_basis_size()

        # Operator is None is interpreted as identity transformation
        if operator is None:
            values = []
            for row in xrange(N):
                for col in xrange(N):
                    if row == col:
                        values.append(ones(nodes.shape))
                    else:
                        values.append(zeros(nodes.shape))
        else:
            # Todo: operator should be only f(nodes)
            values = operator(packet.q, nodes)

        result = zeros((N*K,N*K), dtype=complexfloating)

        for i in xrange(N):
            for j in xrange(N):
                M = zeros((K,K), dtype=complexfloating)
                factor = squeeze(packet.eps * weights * values[i*N + j])

                # Summing up matrices over all quadrature nodes
                for k in xrange(self.QR.get_number_nodes()):
                    tmp = matrix(basis[:,k])
                    M += factor[k] * tmp.H * tmp

                result[i*K:(i+1)*K, j*K:(j+1)*K] = M

        return result
