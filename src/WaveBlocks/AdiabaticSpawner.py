"""The WaveBlocks Project

This file contains the class for Gauss-Hermite quadrature.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np
from scipy import linalg as spla

from Spawner import Spawner


class AdiabaticSpawner(Spawner):
    """This class implements parameter estimation and basis
    projection for spawning of Hagedorn wavepackets.
    """

    def __init__(self, parameters):
        # Cache some parameter values for efficiency
        self.parameters = parameters

        # Configuration parameters related to spawning
        self.eps = parameters["eps"]
        self.basis_size = parameters["basis_size"]
        self.K = parameters["K0"]
        self.threshold = parameters["spawn_threshold"]
        if parameters.has_key("spawn_max_order"):
            self.max_order = parameters["spawn_max_order"]
        else:
            self.max_order = 1


    def estimate_parameters(self, packet, mother_component):
        """Compute the parameters for a new wavepacket.
        """
        P, Q, S, p, q = packet.get_parameters()
        c = packet.get_coefficients(component=mother_component)
        c = np.squeeze(c)

        # Higher coefficients
        ck = c[self.K:]
        w = spla.norm(ck)**2

        if w < self.threshold:
            print(" Warning: really small w! Nothing to spawn!")
            return None

        # Some temporary values
        k = np.arange(self.K+1, self.basis_size)
        ck   = c[self.K+1:]
        ckm1 = c[self.K:-1]

        tmp = np.sum( np.conj(ck) * ckm1 * np.sqrt(k) )

        # Compute spawning position and impulse
        a = q + np.sqrt(2)*self.eps/w * np.real( Q * tmp )
        b = p + np.sqrt(2)*self.eps/w * np.real( P * tmp )

        # theta_1
        k = np.arange(self.K, self.basis_size)
        ck = c[self.K:]
        theta1 = np.sum( np.abs(ck)**2 * (2.0*k + 1.0) )

        # theta_2
        k = np.arange(self.K, self.basis_size-2)
        ck   = c[self.K:-2]
        ckp2 = c[self.K+2:]
        theta2 = np.sum( np.conj(ckp2) * ck * np.sqrt((k+1)*(k+2)) )

        # Compute other parameters
        A = -2.0/self.eps**2 * (q-a)**2 + 1.0/w * ( abs(Q)**2 * theta1 + 2.0*np.real(Q**2 * theta2) )
        B = -2.0/self.eps**2 * (p-b)**2 + 1.0/w * ( abs(P)**2 * theta1 + 2.0*np.real(P**2 * theta2) )

        # Normalize
        A = np.sqrt(A)
        B = (np.sqrt(A**2 * B - 1.0) + 1.0j) / A

        return (B, A, S, b, a)


    # def project_coefficients(self, mother, child):
    #     """Update the superposition coefficients of mother and
    #     spawned wavepacket.
    #     """
    #     c_old = mother.get_coefficients(component=0)        
    #     w = spla.norm( np.squeeze(c_old[self.K:,:]) )

    #     # Mother packet
    #     c_new = np.zeros(c_old.shape, dtype=np.complexfloating)
    #     c_new[:self.K,:] = c_old[:self.K,:]

    #     # Spawned packet
    #     c_new2 = np.zeros(c_old.shape, dtype=np.complexfloating)
    #     # pure Gaussian
    #     c_new2[0,0] = 1.0
    #     # But normalized
    #     c_new2 = w * c_new2

    #     mother.set_coefficient_vector(c_new)
    #     child.set_coefficient_vector(c_new2)

    #     return (mother, child)


    def project_coefficients(self, mother, child):
        """Update the superposition coefficients of mother and
        spawned wavepacket. We do a full basis projection to the
        basis of the spawned wavepacket here.
        """
        c_old = mother.get_coefficients(component=0)        
        #w = spla.norm( np.squeeze(c_old[self.K:,0]) )

        # Mother packet
        c_new_m = np.zeros(c_old.shape, dtype=np.complexfloating)
        c_new_m[:self.K,:] = c_old[:self.K,:]

        # Spawned packet
        c_new_s = np.zeros(c_old.shape, dtype=np.complexfloating)

        # Quadrature rule, assume same quadrature order for both packets
        QR = mother.get_quadrator()
        R = QR.get_order()

        # Mix the parameters for quadrature
        (Pm, Qm, Sm, pm, qm) = mother.get_parameters()
        (Ps, Qs, Ss, ps, qs) = child.get_parameters()
        
        rm = Pm/Qm
        rs = Ps/Qs
        
        r = np.conj(rm)-rs
        s = np.conj(rm)*qm - rs*qs
        
        q0 = np.imag(s) / np.imag(r)
        Q0 = -0.5 * np.imag(r)
        QS = 1 / np.sqrt(Q0)

        # The quadrature nodes and weights
        nodes = q0 + self.eps * QS * QR.get_nodes()
        weights = np.squeeze(QR.get_weights())

        # Basis sets for both packets
        basis_m = mother.evaluate_base_at(nodes, prefactor=True)
        basis_s = child.evaluate_base_at(nodes, prefactor=True)

        # Project to the basis of the spawned wavepacket
        # Original, inefficient code for projection
        # for i in xrange(self.max_order):
        #     # Loop over all quadrature points
        #     tmp = 0.0j
        #     for r in xrange(R):
        #         tmp += np.conj(np.dot( c_old[self.K:,0], basis_m[self.K:,r] )) * basis_s[i,r] * weights[r]   
        #
        #     c_new_s[i,0] = self.eps * QS * tmp

        # Optimised code (maybe can optimise further and removing the outer loop too?)
        tmp = np.zeros((self.max_order,), dtype=np.complexfloating)
        for r in xrange(R):
            tmp += np.conj(np.dot( c_old[self.K:,0], basis_m[self.K:,r] )) * basis_s[:self.max_order,r] * weights[r]
        
        c_new_s[:self.max_order,0] = self.eps * QS * tmp

        # Reassign the new coefficients
        mother.set_coefficient_vector(c_new_m)
        child.set_coefficient_vector(c_new_s)

        return (mother, child)
