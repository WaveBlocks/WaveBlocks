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
from InhomogeneousQuadrature import InhomogeneousQuadrature


class AdiabaticSpawner(Spawner):
    r"""
    This class implements parameter estimation and basis
    projection for spawning of Hagedorn wavepackets.
    """

    def __init__(self, parameters):
        # Cache some parameter values for efficiency
        self.parameters = parameters

        # Configuration parameters related to spawning
        self.eps = parameters["eps"]
        self.K = parameters["spawn_K0"]
        self.threshold = parameters["spawn_threshold"]

        # The spawning method used, default is "lumping"
        self.spawn_method = parameters["spawn_method"] if "spawn_method" in parameters else "lumping"

        # Only used for spawning method "projection"
        self.max_order = parameters["spawn_max_order"] if "spawn_max_order" in  parameters else 1


    def estimate_parameters(self, packet, component):
        r"""
        Compute the parameters for a new wavepacket.
        """
        P, Q, S, p, q = packet.get_parameters()
        c = packet.get_coefficients(component=component)
        c = np.squeeze(c)

        # Higher coefficients
        ck = c[self.K:]
        w = spla.norm(ck)**2

        if w < self.threshold**2:
            # This is to prevent malfunctions like division by 0 in the code below
            print("  Warning: really small component! Nothing to spawn!")
            return None

        # Compute spawning position and impulse
        k = np.arange(self.K+1, packet.get_basis_size(component=component))
        ck   = c[self.K+1:]
        ckm1 = c[self.K:-1]
        tmp = np.sum(np.conj(ck) * ckm1 * np.sqrt(k))

        a = q + np.sqrt(2)*self.eps/w * np.real(Q * tmp)
        b = p + np.sqrt(2)*self.eps/w * np.real(P * tmp)

        # theta_1
        k = np.arange(self.K, packet.get_basis_size(component=component))
        ck = c[self.K:]
        theta1 = np.sum(np.abs(ck)**2 * (2.0*k + 1.0))

        # theta_2
        k = np.arange(self.K, packet.get_basis_size(component=component)-2)
        ck   = c[self.K:-2]
        ckp2 = c[self.K+2:]
        theta2 = np.sum(np.conj(ckp2) * ck * np.sqrt((k+1)*(k+2)))

        # Compute other parameters
        A = -2.0/self.eps**2 * (q-a)**2 + 1.0/w * (abs(Q)**2 * theta1 + 2.0*np.real(Q**2 * theta2))
        B = -2.0/self.eps**2 * (p-b)**2 + 1.0/w * (abs(P)**2 * theta1 + 2.0*np.real(P**2 * theta2))

        # Transform
        A = np.sqrt(A)
        B = (np.sqrt(A**2 * B - 1.0) + 1.0j) / A

        return (B, A, S, b, a)


    def project_coefficients(self, mother, child, component=0):
        r"""
        Update the superposition coefficients of mother and
        spawned wavepacket. Here we decide which method to use
        and call the corresponding method.
        """
        if self.spawn_method == "lumping":
            return self.spawn_lumping(mother, child, component)
        elif self.spawn_method == "projection":
            return self.spawn_basis_projection(mother, child, component)
        else:
            raise ValueError("Unknown spawning method '" + self.spawn_method + "'!")


    def spawn_lumping(self, mother, child, component):
        r"""
        Update the superposition coefficients of mother and
        spawned wavepacket. We produce just a gaussian which
        takes the full norm :math:`\langle w | w \rangle` of :math:`w`.
        """
        c_old = mother.get_coefficients(component=component)
        w = spla.norm(np.squeeze(c_old[self.K:,:]))

        # Mother packet
        c_new_m = np.zeros(c_old.shape, dtype=np.complexfloating)
        c_new_m[:self.K,:] = c_old[:self.K,:]

        # Spawned packet
        c_new_s = np.zeros(c_old.shape, dtype=np.complexfloating)
        # pure Gaussian
        c_new_s[0,0] = 1.0
        # But normalized
        c_new_s = w * c_new_s

        mother.set_coefficients(c_new_m, component=component)
        child.set_coefficients(c_new_s, component=component)

        return (mother, child)


    def spawn_basis_projection(self, mother, child, component):
        r"""
        Update the superposition coefficients of mother and
        spawned wavepacket. We do a full basis projection to the
        basis of the spawned wavepacket here.
        """
        c_old = mother.get_coefficients(component=component)

        # Mother packet
        c_new_m = np.zeros(c_old.shape, dtype=np.complexfloating)
        c_new_m[:self.K,:] = c_old[:self.K,:]

        # Spawned packet
        c_new_s = np.zeros((child.get_basis_size(component=component),1), dtype=np.complexfloating)

        # The quadrature
        quadrature = InhomogeneousQuadrature()

        # Quadrature rule. Assure the "right" quadrature is choosen if
        # mother and child have different basis sizes
        if mother.get_basis_size(component=component) > child.get_basis_size(component=component):
            quadrature.set_qr(mother.get_quadrature().get_qr())
        else:
            quadrature.set_qr(child.get_quadrature().get_qr())

        # The quadrature nodes and weights
        q0, QS = quadrature.mix_parameters(mother.get_parameters(), child.get_parameters())
        nodes = quadrature.transform_nodes(mother.get_parameters(), child.get_parameters(), mother.eps)
        weights = quadrature.get_qr().get_weights()

        # Basis sets for both packets
        basis_m = mother.evaluate_basis_at(nodes, prefactor=True)
        basis_s = child.evaluate_basis_at(nodes, prefactor=True)

        max_order = min(child.get_basis_size(component=component), self.max_order)

        # Project to the basis of the spawned wavepacket
        # Original, inefficient code for projection
        # R = QR.get_order()
        # for i in xrange(max_order):
        #     # Loop over all quadrature points
        #     tmp = 0.0j
        #     for r in xrange(R):
        #         tmp += np.conj(np.dot(c_old[self.K:,0], basis_m[self.K:,r])) * basis_s[i,r] * weights[r]
        #
        #     c_new_s[i,0] = self.eps * QS * tmp

        # Optimised and vectorised code (in ugly formatting)
        c_new_s[:max_order,:] = self.eps * QS * (
            np.reshape(
                np.sum(
                    np.transpose(
                        np.reshape(
                            np.conj(
                                np.sum(c_old[self.K:,:] * basis_m[self.K:,:],axis=0)
                            ) ,(-1,1)
                        )
                    ) * (basis_s[:max_order,:] * weights[:,:]), axis=1
                ) ,(-1,1)
            ))

        # Reassign the new coefficients
        mother.set_coefficients(c_new_m, component=component)
        child.set_coefficients(c_new_s, component=component)

        return (mother, child)
