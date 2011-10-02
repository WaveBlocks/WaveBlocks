"""The WaveBlocks Project

This file contains code for the homogeneous quadrature of wavepackets.
The class defined here can compute brakets, inner products and expectation
values and compute the $F$ matrix.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, ones, complexfloating, sum, cumsum, squeeze, conjugate, dot, outer

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


    def quadrature(self, packet, operator=None, summed=False, component=None, diag_component=None):
        """Performs the quadrature of $\Braket{\Psi|f|\Psi}$ for a general $f$.
        @param packet: The wavepacket $|\Psi>$.
        @keyword operator: A real-valued function $f(x):R \rightarrow R^{N \times N}.$
        @keyword summed: Whether to sum up the individual integrals $\Braket{\Phi_i|f_{i,j}|\Phi_j}$.
        @keyword component: Request only the i-th component of the result. Remember that $i \in [0, N^2-1]$.
        @keyword diag_component: Request only the i-th component from the diagonal entries, here $i \in [0, N-1]$
        @return: The value of $\Braket{\Psi|f|\Psi}$. This is either a scalar value or a list of $N^2$ scalar elements.
        @note: 'component' takes precedence over 'diag_component' if both are supplied. (Which is discouraged)
        """
        nodes = self.transform_nodes(packet.get_parameters(), packet.eps)
        weights = self.QR.get_weights()
        basis = packet.evaluate_basis_at(nodes)

        N = packet.get_number_components()
        K = packet.get_basis_size()

        if operator is None:
            # Operator is None is interpreted as identity transformation
            operator = lambda nodes, component=None: ones(nodes.shape) if component[0] == component[1] else zeros(nodes.shape)
            values = [ operator(nodes, component=(r,c)) for r in xrange(N) for c in xrange(N) ]
        else:
            values = operator(nodes)

        coeffs = packet.get_coefficients()

        result = []
        for row in xrange(N):
            for col in xrange(N):
                M = zeros((K[row],K[col]), dtype=complexfloating)
                factor = squeeze(packet.eps * weights * values[row*N + col])

                # Summing up matrices over all quadrature nodes
                for r in xrange(self.QR.get_number_nodes()):
                    M += factor[r] * outer(conjugate(basis[:K[row],r]), basis[:K[col],r])

                # And include the coefficients as conj(c)*M*c
                result.append(dot(conjugate(coeffs[row]).T, dot(M,coeffs[col])))

        # TODO: improve to avoid unnecessary computations of other components
        if component is not None:
            result = result[component]
        elif diag_component is not None:
            result = result[diag_component*N + diag_component]
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
        # The partition scheme of the block vectors and block matrix
        partition = [0] + list(cumsum(K))

        if operator is None:
            # Operator is None is interpreted as identity transformation
            operator = lambda nodes, component=None: ones(nodes.shape) if component[0] == component[1] else zeros(nodes.shape)
            values = [ operator(nodes, component=(r,c)) for r in xrange(N) for c in xrange(N) ]
        else:
            # TODO: operator should be only f(nodes)
            values = operator(packet.q, nodes)

        result = zeros((sum(K),sum(K)), dtype=complexfloating)

        for row in xrange(N):
            for col in xrange(N):
                M = zeros((K[row],K[col]), dtype=complexfloating)
                factor = squeeze(packet.eps * weights * values[row*N + col])

                # Sum up matrices over all quadrature nodes and
                # remember to slice the evaluated basis appropriately
                for r in xrange(self.QR.get_number_nodes()):
                    M += factor[r] * outer(conjugate(basis[:K[row],r]), basis[:K[col],r])

                result[partition[row]:partition[row+1], partition[col]:partition[col+1]] = M

        return result
