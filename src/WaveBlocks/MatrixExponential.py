"""The WaveBlocks Project

This file contains several different algorithms to compute the
matrix exponential. Currently we have an exponential based on
Pade approximations and an Arnoldi iteration method.

@author: R. Bourquin
@copyright: Copyright (C) 2007 V. Gradinaru
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from scipy import zeros, dot, mat, hstack
from scipy.linalg import norm, expm


def matrix_exp_pade(A, coefficients, factor):
    """Compute the solution of v' = A v with a full matrix exponential via Pade approximation.
    @param A: The matrix.
    @param v: The vector.
    @param factor: An additional factor, usually contains at least the timestep.
    """
    return dot(expm(-1.0j*A*factor), coefficients)


def arnoldi(A, v0, k):
    """Arnoldi algorithm (Krylov approximation of a matrix)
    @param A: The matrix to approximate.
    @param v0: The initial vector (should be in matrix form) 
    @param k: The number of Krylov steps.
    @return: A tupel (V, H) where V is the matrix (large, N*k) containing the orthogonal vectors and
    H is the matrix (small, k*k) containing the Krylov approximation of A.
    """
    V = mat( v0.copy() / norm(v0) )
    H = mat( zeros((k+1,k)) )
    for m in xrange(k):
        vt = A * V[:,m]
        for j in xrange(m+1):
            H[j,m] = (V[:,j].H*vt)[0,0]
            vt -= H[j,m] * V[:,j]        
        H[m+1,m] = norm(vt)
        V = hstack( (V, vt.copy()/H[m+1,m]) ) 
    return (V, H)


def matrix_exp_arnoldi(A, v, factor, k):
    """Compute the solution of v' = A v via k steps of a the Arnoldi krylov method.
    @param A: The matrix.
    @param v: The vector.
    @param factor: An additional factor, usually contains at least the timestep.
    @param k: The number of Krylov steps.
    """
    V, H = arnoldi(A, v, k)
    eH = mat( expm(-1.0j*factor*H[:-1,:]) )
    r = V[:,:-1] * eH[:,0] 
    return r * norm(v)
