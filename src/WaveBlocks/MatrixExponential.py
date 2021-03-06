"""The WaveBlocks Project

This file contains several different algorithms to compute the
matrix exponential. Currently we have an exponential based on
Pade approximations and an Arnoldi iteration method.

@author: R. Bourquin
@copyright: Copyright (C) 2007 V. Gradinaru
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, hstack, mat, dot, complexfloating, asarray
from scipy.linalg import norm, expm


def matrix_exp_pade(A, coefficients, factor):
    r"""
    Compute the solution of :math:`v' = A v` with a full matrix exponential via Pade approximation.

    :param A: The matrix.
    :param coefficients: The vector with the coefficients.
    :param factor: An additional factor, usually contains at least the timestep.
    """
    return dot(expm(-1.0j*A*factor), coefficients)


def arnoldi(A, v0, k):
    r"""
    Arnoldi algorithm (Krylov approximation of a matrix)

    :param A: The matrix to approximate.
    :param v0: The initial vector (should be in matrix form)
    :param k: The number of Krylov steps.
    :return: A tupel (V, H) where V is the matrix (large, :math:`N \times k`) containing the orthogonal vectors and
             H is the matrix (small, :math:`k \times k`) containing the Krylov approximation of A.
    """
    V = mat(v0.copy() / norm(v0))
    H = mat(zeros((k+1,k)), dtype=complexfloating)
    for m in xrange(k):
        vt = A * V[:,m]
        for j in xrange(m+1):
            H[j,m] = (V[:,j].H*vt)[0,0]
            vt -= H[j,m] * V[:,j]
        H[m+1,m] = norm(vt)
        V = hstack((V, vt.copy()/H[m+1,m]))
    return (V, H)


def matrix_exp_arnoldi(A, v, factor, k):
    r"""
    Compute the solution of :math:`v' = A v` via k steps of a the Arnoldi krylov method.

    :param A: The matrix.
    :param v: The vector.
    :param factor: An additional factor, usually contains at least the timestep.
    :param k: The number of Krylov steps.
    """
    V, H = arnoldi(A, v, min(min(A.shape), k))
    eH = mat(expm(-1.0j*factor*H[:-1,:]))
    r = V[:,:-1] * eH[:,0]
    return asarray(r * norm(v))
