"""The WaveBlocks Project

This file contains a very simple spawning propagator class
for wavepackets and gaussian spawning.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from functools import partial
import numpy as np
import scipy as sp
import scipy.linalg as spla

from Propagator import Propagator
from HagedornWavepacket import HagedornWavepacket
from AdiabaticSpawner import AdiabaticSpawner


class SpawnAdiabaticPropagator(Propagator):
    """This class can numerically propagate given initial values $\Ket{\Psi}$ in
    a potential $V\ofs{x}$. The propagation is done for a given homogeneous
    Hagedorn wavepacket."""

    def __init__(self, potential, packet, leading_component, parameters):
        """Initialize a new I{HagedornPropagator} instance. 
        @param potential: The potential the wavepacket $\Ket{\Psi}$ feels during the time propagation.
        @param packet: The initial homogeneous Hagedorn wavepacket we propagate in time.
        @param leading_component: The leading component index $\chi$.
        @raise ValueError: If the number of components of $\Ket{\Psi}$ does not
        match the number of energy levels $\lambda_i$ of the potential.
        """
        if packet.get_number_components() != potential.get_number_components():
            raise ValueError("Wave packet does not match to the given potential!")
        
        #: The potential $V\ofs{x}$ the packet feels.
        self.potential = potential

        #: Number $N$ of components the wavepacket $\Ket{\Psi}$ has got.
        self.number_components = self.potential.get_number_components()

        #: The leading component $\chi$ is the index of the eigenvalue of the
        #: potential that is responsible for propagating the Hagedorn parameters.
        self.leading = leading_component

        #: The Hagedorn wavepacket.
        self.packets = [ packet ]

        #: Number of wavepackets the propagator currently handles
        self.number_packets = 1

        # Cache some parameter values for efficiency
        self.parameters = parameters
        self.dt = parameters["dt"]
        self.eps = parameters["eps"]
        self.K = parameters["spawn_K0"]
        self.threshold = parameters["spawn_threshold"]

        # todo: put this in the ParameterProvider
        self.already_spawned = False

        # Decide about the matrix exponential algorithm to use
        if parameters.has_key("matrix_exponential"):
            method = parameters["matrix_exponential"]
        else:
            method = GlobalDefaults.matrix_exponential

        if method == "pade":
            from MatrixExponential import matrix_exp_pade
            self.__dict__["matrix_exponential"] = matrix_exp_pade
        elif method == "arnoldi":
            from MatrixExponential import matrix_exp_arnoldi

            if parameters.has_key("arnoldi_steps"):
                arnoldi_steps = parameters["arnoldi_steps"]
            else:
                arnoldi_steps = min(parameters["basis_size"], GlobalDefaults.arnoldi_steps)

            self.__dict__["matrix_exponential"] = partial(matrix_exp_arnoldi, k=arnoldi_steps)
        else:
            raise ValueError("Unknown matrix exponential algorithm")

        # Precalculate the potential splitting
        self.potential.calculate_local_quadratic(diagonal_component=self.leading)
        self.potential.calculate_local_remainder(diagonal_component=self.leading)
        
        
    def __str__(self):
        """Prepare a printable string representing the I{HagedornPropagator} instance."""
        return "Hagedorn propagator for " + str(self.number_components) + " components.\n Leading component is " + str(self.leading) + "."
        
        
    def get_number_components(self):
        """@return: The number $N$ of components $\Phi_i$ of $\Ket{\Psi}$."""
        return self.number_components


    def get_potential(self):
        """@return: The I{MatrixPotential} instance used for time propagation."""
        return self.potential


    def get_number_packets(self):
        """
        """
        return self.number_packets


    def get_wavepacket(self, packet=None):
        """@return: The I{HagedornWavepacket} instance that represents the
        current wavepacket $\Ket{\Psi}$."""
        if packet is None:
            return self.packets
        else:
            return self.packets[packet]


    def propagate(self):
        """Given the wavepacket $\Psi$ at time $t$, calculate a new wavepacket
        at time $t + \tau$. We perform exactly one timestep $\tau$ here.
        """
        dt = self.dt

        # Ckeck for spawning
        if self.should_spwan():
            # Initialize an empty wavepacket for spawning
            SWP = HagedornWavepacket(self.parameters)
            SWP.set_quadrator(None)

            # Initialize a Spawner
            AS = AdiabaticSpawner(self.parameters)
            
            # Spawn a new packet
            ps = AS.estimate_parameters(self.packets[-1], 0)

            if ps is not None:
                SWP.set_parameters(ps)
                AS.project_coefficients(self.packets[-1], SWP)

                self.number_packets += 1
                self.packets.append(SWP)

        
        # Propagate all packets
        for packet in self.packets:            
            # Do a kinetic step of dt/2
            packet.q = packet.q + 0.5*dt * packet.p
            packet.Q = packet.Q + 0.5*dt * packet.P
            packet.S = packet.S + 0.25*dt * packet.p**2
            
            # Do a potential step with the local quadratic part            
            V = self.potential.evaluate_local_quadratic_at(packet.q)
            
            packet.p = packet.p - dt * V[1]
            packet.P = packet.P - dt * V[2] * packet.Q
            packet.S = packet.S - dt * V[0]
            
            # Do a potential step with the local non-quadratic taylor remainder
            F = packet.matrix(self.potential.evaluate_local_remainder_at)
            coefficients = packet.get_coefficient_vector()            
            coefficients = self.matrix_exponential(F, coefficients, dt/self.eps**2)
            packet.set_coefficient_vector(coefficients)
            
            # Do a kinetic step of dt/2
            packet.q = packet.q + 0.5 * dt * packet.p
            packet.Q = packet.Q + 0.5 * dt * packet.P
            packet.S = packet.S + 0.25 * dt * packet.p**2


    def should_spwan(self):
        """Check if it's time to spawn a new wavepacket.
        """
        if self.already_spawned:
            return False

        c = self.packets[0].get_coefficients()
        c = np.squeeze(c[0][:])
        
        c_low = c[:self.K]
        c_high = c[self.K:]

        n_low = spla.norm(c_low)
        n_high = spla.norm(c_high)

        print((n_low, n_high))

        answer = (n_high >= self.threshold)

        if answer == True:
            self.already_spawned = True

        return answer
