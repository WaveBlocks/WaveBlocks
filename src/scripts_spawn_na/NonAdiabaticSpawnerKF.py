"""The WaveBlocks Project

This file conatins the code for spawning new wavepackets depending
on some criterion.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np
from scipy import linalg as spla

from WaveBlocks import Spawner


class NonAdiabaticSpawnerKF(Spawner):
    """This class implements parameter estimation and basis
    projection for spawning of Hagedorn wavepackets in the
    non-adiabatic case.
    """

    def __init__(self, parameters):
        # Cache some parameter values for efficiency
        self.parameters = parameters

        # Configuration parameters related to spawning
        self.eps = parameters["eps"]
        self.threshold = parameters["spawn_threshold"]

        # The spawning method used, default is "lumping"
        self.spawn_method = parameters["spawn_method"] if "spawn_method" in parameters else "lumping"

        # The value of k for the (2*k+1) in the parameter estimation formula
        # Do not confuse this with the K in the adiabatic spawning!
        # The default 0 Corresponds to a Gaussian.
        self.order = parameters["spawn_order"] if "spawn_order" in parameters else 0

        # Only used for spawning method "projection"
        self.max_order = parameters["spawn_max_order"] if "spawn_max_order" in  parameters else 1


    def estimate_parameters(self, packet, component=0, order=None):
        """Compute the parameters for a new wavepacket.
        """
        if order is None:
            order = self.order

        P, Q, S, p, q = packet.get_parameters(component=component)

        c = packet.get_coefficients(component=component)
        c = np.squeeze(c)

        w = spla.norm(c)**2

        if w < self.threshold**2:
            print(" Warning: really small w! Nothing to spawn!")
            return None

        # Compute spawning position and impulse
        k = np.arange(1, packet.get_basis_size())
        ck   = c[1:]
        ckm1 = c[:-1]
        tmp = np.sum( np.conj(ck) * ckm1 * np.sqrt(k) )

        a = q + np.sqrt(2)*self.eps/w * np.real( Q * tmp )
        b = p + np.sqrt(2)*self.eps/w * np.real( P * tmp )


        # # Diagonal parts
        # k = np.arange(0, packet.get_basis_size())
        # ck = c[:]
        # theta1 = np.sum( np.abs(ck)**2 * (2.0*k + 1.0) )

        # # First off-diagonal parts
        # k = np.arange(0, packet.get_basis_size()-2)
        # ck   = c[:-2]
        # ckp2 = c[2:]
        # theta2 = np.sum( np.conj(ckp2) * ck * np.sqrt((k+1)*(k+2)) )

        # # Compute other parameters
        # tmpA = self.eps**2/2 * ( abs(Q)**2 * theta1 + 2.0*np.real(Q**2 * theta2) )
        # tmpB = self.eps**2/2 * ( abs(P)**2 * theta1 + 2.0*np.real(P**2 * theta2) )


        # # Diagonal parts
        # k = np.arange(0, packet.get_basis_size())
        # ck = c[:]
        # theta0 = np.sum( np.abs(ck)**2 * (6.0*k**2 + 6.0*k + 3.0) )

        # # Second off-diagonal parts
        # k = np.arange(0, packet.get_basis_size()-2)
        # ck   = c[:-2]
        # ckp2 = c[2:]
        # theta2 = np.sum( np.conj(ckp2) * ck * (4.0*k + 6.0) * np.sqrt((k+1)*(k+2)) )

        # # Fourth off-diagonal parts
        # k = np.arange(0, packet.get_basis_size()-4)
        # ck   = c[:-4]
        # ckp4 = c[4:]
        # theta4 = np.sum( np.conj(ckp4) * ck * np.sqrt((k+1)*(k+2)*(k+3)*(k+4)) )

        # tmpA2 = self.eps**4/4 * ( abs(Q)**4 * theta0 + 2.0*np.real( Q**3*np.conj(Q) * theta2 + Q**4 * theta4 ) )
        # tmpB2 = self.eps**4/4 * ( abs(P)**4 * theta0 + 2.0*np.real( P**3*np.conj(P) * theta2 + P**4 * theta4 ) )

        #alpha0 = 1.0
        #alpha2 = tmpA
        #alpha4 = tmpA2

        # Compute other parameters
        #Aabs2 = -2.0/self.eps**2 * (q-a)**2 + 1.0/w * ( abs(Q)**4 * theta0 + 2.0*np.real( Q**3*np.conj(Q) * theta2 + Q**4 * theta4 ) )
        #Babs2 = -2.0/self.eps**2 * (p-b)**2 + 1.0/w * ( abs(P)**4 * theta0 + 2.0*np.real( P**3*np.conj(P) * theta2 + P**4 * theta4 ) )
        #Aabs2 = -2.0/self.eps**2 * (q-a)**2 + 1.0/w * ( abs(Q)**4 * theta0 + 2.0*np.real( Q**3*np.conj(Q) * theta2 + Q**4 * theta4 ) )
        #Babs2 = -2.0/self.eps**2 * (p-b)**2 + 1.0/w * ( abs(P)**4 * theta0 + 2.0*np.real( P**3*np.conj(P) * theta2 + P**4 * theta4 ) )

        # Try it with numerical quadrature
        f0 = lambda x: np.ones(x.shape)
        f2 = lambda x: (x-q)**2
        f4 = lambda x: (x-q)**4

        quadrature = packet.get_quadrature()

        alpha0 = quadrature.quadrature(packet, f0, summed=True)
        alpha2 = 2/self.eps**2 * quadrature.quadrature(packet, f2, summed=True)
        alpha4 = 4/self.eps**4 * quadrature.quadrature(packet, f4, summed=True)

        Aabs2 = (alpha2 + np.sqrt(0.0 + alpha2**2 - alpha0*alpha4)) / (alpha0)


        fpacket = packet.to_fourier_space(assign=False)

        f0 = lambda x: np.ones(x.shape)
        f2 = lambda x: (x-p)**2
        f4 = lambda x: (x-p)**4

        beta0 = quadrature.quadrature(fpacket, f0, summed=True)
        beta2 = 2/self.eps**2 * quadrature.quadrature(fpacket, f2, summed=True)
        beta4 = 4/self.eps**4 * quadrature.quadrature(fpacket, f4, summed=True)

        Babs2 = (beta2 + np.sqrt(0.0 + beta2**2 - beta0*beta4)) / (beta0)

        print((alpha0, alpha2, alpha4))
        print((beta0, beta2, beta4))

        # Transform
        #A = np.sqrt(Aabs2)
        #B = (np.sqrt(Aabs2 * Babs2 - 1.0 + 0.0j) + 1.0j) / A

        A = Aabs2
        B = Babs2


        return (B, A, S, b, a)


    def project_coefficients(self, mother, child, component=0, order=None):
        """Update the superposition coefficients of mother and
        spawned wavepacket. Here we decide which method to use
        and call the corresponding method.
        """
        if self.spawn_method == "lumping":
            return self.spawn_lumping(mother, child, component, order=order)
        elif self.spawn_method == "projection":
            return self.spawn_basis_projection(mother, child, component, order=order)
        else:
            raise ValueError("Unknown spawning method '" + self.spawn_method + "'!")


    def spawn_lumping(self, mother, child, component, order=None):
        """Update the superposition coefficients of mother and
        spawned wavepacket. We produce just a gaussian which
        takes the full norm <w|w> of w.
        """
        if order is None:
            order = self.order

        c_old = mother.get_coefficients(component=component)
        w = spla.norm(np.squeeze(c_old))

        # Mother packet
        c_new_m = np.zeros(c_old.shape, dtype=np.complexfloating)

        # Spawned packet
        c_new_s = np.zeros((child.get_basis_size(),1), dtype=np.complexfloating)
        # Pure $\phi_order$ function
        c_new_s[order,0] = 1.0
        # But normalized
        c_new_s = w * c_new_s

        mother.set_coefficients(c_new_m, component=component)
        child.set_coefficients(c_new_s, component=component)

        return (mother, child)


    def spawn_basis_projection(self, mother, child, component, order=None):
        """Update the superposition coefficients of mother and
        spawned wavepacket. We do a full basis projection to the
        basis of the spawned wavepacket here.
        """
        c_old = mother.get_coefficients(component=component)

        # Mother packet
        c_new_m = np.zeros(c_old.shape, dtype=np.complexfloating)

        # Spawned packet
        c_new_s = np.zeros((child.get_basis_size(),1), dtype=np.complexfloating)

        # Quadrature rule, assume same quadrature order for both packets
        QR = mother.get_quadrature().get_qr()

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
        basis_m = mother.evaluate_basis_at(nodes, prefactor=True)
        basis_s = child.evaluate_basis_at(nodes, prefactor=True)

        max_order = min(child.get_basis_size(), self.max_order)

        # Project to the basis of the spawned wavepacket
        # Original, inefficient code for projection
        # R = QR.get_order()
        # for i in xrange(max_order):
        #     # Loop over all quadrature points
        #     tmp = 0.0j
        #     for r in xrange(R):
        #         tmp += np.conj(np.dot( c_old[:,0], basis_m[:,r] )) * basis_s[i,r] * weights[0,r]
        #     c_new_s[i,0] = self.eps * QS * tmp

        # Optimised and vectorised code (in ugly formatting)
        c_new_s[:max_order,:] = self.eps * QS * (
            np.reshape(
                np.sum(
                    np.transpose(
                        np.reshape(
                            np.conj(
                                np.sum(c_old[:,:] * basis_m[:,:],axis=0)
                            ) ,(-1,1)
                        )
                    ) * (basis_s[:max_order,:] * weights[:,:]), axis=1
                ) ,(-1,1)
            ))

        # Reassign the new coefficients
        mother.set_coefficients(c_new_m, component=component)
        child.set_coefficients(c_new_s, component=component)

        return (mother, child)
