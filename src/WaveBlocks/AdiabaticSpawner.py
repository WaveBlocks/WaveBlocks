"""The WaveBlocks Project

This file conatins the code for spawning new wavepackets depending
on some criterion. 

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
        self.K = parameters["spawn_K0"]
        self.threshold = parameters["spawn_threshold"]
        if parameters.has_key("spawn_normed_gaussian"):
            self.spawn_normed_gaussian = parameters["spawn_normed_gaussian"]
        else:
            self.spawn_normed_gaussian = True
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

        if w < self.threshold**2:
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


    def project_coefficients(self, mother, child):
        """Update the superposition coefficients of mother and
        spawned wavepacket. Here we decide which method to use
        and call the corresponding method.
        """
        if self.spawn_normed_gaussian is True:
            return self.normed_gaussian(mother, child)
        else:
            return self.full_basis_projection(mother, child)


    def normed_gaussian(self, mother, child):
        """Update the superposition coefficients of mother and
        spawned wavepacket. We produce just a gaussian which
        takes the full norm <w|w> of w.
        """
        c_old = mother.get_coefficients(component=0)        
        w = spla.norm( np.squeeze(c_old[self.K:,:]) )

        # Mother packet
        c_new_m = np.zeros(c_old.shape, dtype=np.complexfloating)
        c_new_m[:self.K,:] = c_old[:self.K,:]

        # Spawned packet
        c_new_s = np.zeros(c_old.shape, dtype=np.complexfloating)
        # pure Gaussian
        c_new_s[0,0] = 1.0
        # But normalized
        c_new_s = w * c_new_s

        mother.set_coefficient_vector(c_new_m)
        child.set_coefficient_vector(c_new_s)

        return (mother, child)


    def full_basis_projection(self, mother, child):
        """Update the superposition coefficients of mother and
        spawned wavepacket. We do a full basis projection to the
        basis of the spawned wavepacket here.
        """
        c_old = mother.get_coefficients(component=0)

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
        weights = QR.get_weights()

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

        # Optimised and vectorised code (in ugly formatting)
        c_new_s[:self.max_order,:] = self.eps * QS * (
            np.reshape(
                np.sum(
                    np.transpose(
                        np.reshape(
                            np.conj(
                                np.sum(c_old[self.K:,:] * basis_m[self.K:,:],axis=0)
                            ) ,(-1,1)
                        )
                    ) * (basis_s[:self.max_order,:] * weights[:,:]), axis=1
                ) ,(-1,1)
            ))

        # Reassign the new coefficients
        mother.set_coefficient_vector(c_new_m)
        child.set_coefficient_vector(c_new_s)

        return (mother, child)