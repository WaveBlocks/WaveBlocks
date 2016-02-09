"""The WaveBlocks Project

This file contains the class which represents an inhomogeneous Hagedorn wavepacket.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from functools import partial
from numpy import zeros, complexfloating, array, sum, transpose, arange
from scipy import pi, sqrt, exp, conj, dot
from scipy.linalg import norm

from ComplexMath import cont_sqrt
from Wavepacket import Wavepacket
from InhomogeneousQuadrature import InhomogeneousQuadrature
import GlobalDefaults as GD


class HagedornWavepacketInhomogeneous(Wavepacket):
    """This class represents inhomogeneous vector valued wavepackets :math:`\Ket{\Psi}`.
    """

    def __init__(self, parameters):
        """Initialize the I{HagedornWavepacketInhomogeneous} object that represents :math:`\Ket{\Psi}`.
        :param parameters: A I{ParameterProvider} instance or a dict containing simulation parameters.
        @raise ValueError: For :math:`N < 1` or :math:`K < 2`.
        """
        #: Number of components :math:`\Phi_i` the wavepacket :math:`\Ket{\Psi}` has got.
        self.number_components = parameters["ncomponents"]

        if self.number_components < 1:
            raise ValueError("Number of components of the Hagedorn wavepacket has to be >= 1.")

        # Size of the basis from which we construct the wavepacket.
        # If there is a key "basis_size" in the input parameters, the corresponding
        # value can be either a single int or a list of ints. If there is no such key
        # we use the values from the global defaults.
        if parameters.has_key("basis_size"):
            bs = parameters["basis_size"]
            if type(bs) is list or type(bs) is tuple:
                if not len(bs) == self.number_components:
                    raise ValueError("Number of value(s) for basis size(s) does not match.")

                self.basis_size = bs[:]
            else:
                self.basis_size = self.number_components * [ bs ]
        else:
            self.basis_size = self.number_components * [ GD.default_basis_size ]

        if any([bs < 2 for bs in self.basis_size]):
            raise ValueError("Number of basis fucntions for Hagedorn wavepacket has to be >= 2.")

        # Cache the parameter values epsilon we will use over and over again.
        self.eps = parameters["eps"]

        #: Data structure that contains the Hagedorn parameter sets :math:`\Pi_i` of each component :math:`\Phi_i`.
        #: The parameter values are initialized to the Harmonic Oscillator Eigenfunctions
        self.parameters = [ GD.default_Pi for i in xrange(self.number_components) ]

        #: The coefficients :math:`c^i` of the linear combination for each component :math:`\Phi_i`.
        self.coefficients = [ zeros((self.basis_size[index],1), dtype=complexfloating) for index in xrange(self.number_components) ]

        #: An object that can compute brakets via quadrature.
        self.quadrature = None

        self._cont_sqrt_cache = [ 0.0 for i in xrange(self.number_components) ]


    def __str__(self):
        """:return: A string describing the Hagedorn wavepacket.
        """
        s =  "Inhomogeneous Hagedorn wavepacket with "+str(self.number_components)+" components\n"
        return s


    def clone(self, keepid=False):
        # Parameters of this packet
        params = {"ncomponents": self.number_components,
                  "eps":         self.eps}

        # Create a new Packet
        other = HagedornWavepacketInhomogeneous(params)
        # If we wish to keep the packet ID
        if keepid is True:
            other.set_id(self.get_id())
        # And copy over all (private) data
        other.set_basis_size(self.get_basis_size())
        other.set_quadrature(self.get_quadrature())
        other.set_parameters(self.get_parameters())
        other.set_coefficients(self.get_coefficients())
        other._cont_sqrt_cache = [ cache for cache in self._cont_sqrt_cache ]

        return other


    def get_parameters(self, component=None, aslist=False):
        """Get the Hagedorn parameters :math:`\Pi_i` of each component :math:`\Phi_i` of the wavepacket :math:`\Psi`.
        :param component: The index :math:`i` of the component whose parameters :math:`\Pi_i` we want to get.
        :param aslist: Dummy parameter for API compatibility with the homogeneous packets.
        :return: A list with all the sets :math:`\Pi_i` or a single set.
        """
        if component is None:
            result = [ tuple(item) for item in self.parameters ]
        else:
            result = self.parameters[component]
        return tuple(result)


    def set_parameters(self, parameters, component=None):
        """Set the Hagedorn parameters :math:`\Pi_i` of each component :math:`\Phi_i` of the wavepacket :math:`\Psi`.
        :param parameters: A list or a single set of Hagedorn parameters.
        :param component: The index :math:`i` of the component whose parameters :math:`\Pi_i` we want to update.
        """
        if component is None:
            for index, item in enumerate(parameters):
                self.parameters[index] = item[:]
        else:
            self.parameters[component] = parameters[:]


    def set_quadrature(self, quadrature):
        """Set the I{InhomogeneousQuadrature} instance used for evaluating brakets.
        :param quadrature: The new I{InhomogeneousQuadrature} instance. May be I{None}
        to use a dafault one with a quadrature rule of order :math:`K+4`.
        """
        # TODO: Put an "extra accuracy" parameter into global defaults with value of 4.
        # TODO: Improve on the max(basis_size) later
        # TODO: Rethink if wavepackets should contain a QR
        if quadrature is None:
            self.quadrature = InhomogeneousQuadrature(order=max(self.basis_size) + 4)
        else:
            self.quadrature = quadrature


    def get_quadrature(self):
        """Return the I{InhomogeneousQuadrature} instance used for evaluating brakets.
        :return: The current instance I{InhomogeneousQuadrature}.
        """
        return self.quadrature


    def evaluate_basis_at(self, nodes, component, prefactor=False):
        """Evaluate the Hagedorn functions :math:`\phi_k` recursively at the given nodes :math:`\gamma`.
        :param nodes: The nodes :math:`\gamma` at which the Hagedorn functions are evaluated.
        :param component: The index :math:`i` of the component whose basis functions :math:`\phi^i_k` we want to evaluate.
        :param prefactor: Whether to include a factor of :math:`\left(\det\ofs{Q_i}\right)^{-\frac{1}{2}}`.
        :return: Returns a twodimensional array :math:`H` where the entry :math:`H[k,i]` is the value
        of the :math:`k`-th Hagedorn function evaluated at the node :math:`i`.
        """
        H = zeros((self.basis_size[component], nodes.size), dtype=complexfloating)

        (P, Q, S, p, q) = self.parameters[component]
        Qinv = Q**(-1.0)
        Qbar = conj(Q)
        nodes = nodes.reshape((1,nodes.size))

        H[0] = pi**(-0.25)*self.eps**(-0.5) * exp(1.0j/self.eps**2 * (0.5*P*Qinv*(nodes-q)**2 + p*(nodes-q)))
        H[1] = Qinv*sqrt(2.0/self.eps**2) * (nodes-q) * H[0]

        for k in xrange(2, self.basis_size[component]):
            H[k] = Qinv*sqrt(2.0/self.eps**2)*1.0/sqrt(k) * (nodes-q) * H[k-1] - Qinv*Qbar*sqrt((k-1.0)/k) * H[k-2]

        if prefactor is True:
            sqrtQ, self._cont_sqrt_cache[component] = cont_sqrt(Q, reference=self._cont_sqrt_cache[component])
            H = 1.0/sqrtQ*H

        return H


    def evaluate_at(self, nodes, component=None, prefactor=False):
        """Evaluete the Hagedorn wavepacket :math:`\Psi` at the given nodes :math:`\gamma`.
        :param nodes: The nodes :math:`\gamma` at which the Hagedorn wavepacket gets evaluated.
        :param component: The index :math:`i` of a single component :math:`\Phi_i` to evaluate.
        (Defaults to 'None' for evaluating all components.)
        :param prefactor: Whether to include a factor of :math:`\left(\det\ofs{Q_i}\right)^{-\frac{1}{2}}`.
        :return: A list of arrays or a single array containing the values of the :math:`\Phi_i` at the nodes :math:`\gamma`.
        """
        nodes = nodes.reshape((1,nodes.size))

        if component is not None:
            # Avoid the expensive evaluation of unused other bases
            (P, Q, S, p, q) = self.parameters[component]

            basis = self.evaluate_basis_at(nodes, component, prefactor=prefactor)
            phase = exp(1.0j*S/self.eps**2)
            values = phase * sum(self.coefficients[component] * basis, axis=0)
        else:
            values = [ zeros(nodes.shape) for index in xrange(self.number_components) ]

            for index in xrange(self.number_components):
                (P, Q, S, p, q) = self.parameters[index]

                basis = self.evaluate_basis_at(nodes, index, prefactor=prefactor)
                phase = exp(1.0j*S/self.eps**2)
                values[index] = phase * sum(self.coefficients[index] * basis, axis=0)

        return values


    def get_norm(self, component=None, summed=False):
        """Calculate the :math:`L^2` norm of the wavepacket :math:`\Ket{\Psi}`.
        :param component: The component :math:`\Phi_i` of which the norm is calculated.
        :param summed: Whether to sum up the norms of the individual components :math:`\Phi_i`.
        :return: A list containing the norms of all components :math:`\Phi_i` or the overall norm of :math:`\Psi`.
        """
        if component is None:
            result = [ norm(item) for item in self.coefficients ]

            if summed is True:
                result = reduce(lambda x,y: x+conj(y)*y, result, 0)
                result = sqrt(result)
        else:
            result = norm(self.coefficients[component])

        return result


    def potential_energy(self, potential, summed=False):
        """Calculate the potential energy :math:`\Braket{\Psi|V|\Psi}` of the wavepacket componentwise.
        :param potential: The potential energy operator :math:`V` as function.
        :param summed: Wheter to sum up the individual integrals :math:`\Braket{\Phi_i|V_{i,j}|\Phi_j}`.
        :return: The potential energy of the wavepacket's components :math:`\Phi_i` or the overall potential energy of :math:`\Psi`.
        """
        f = partial(potential, as_matrix=True)
        Q = self.quadrature.quadrature(self, self, f)

        N = self.number_components
        epot = [ sum(Q[i*N:(i+1)*N]) for i in xrange(N) ]

        if summed is True:
            epot = sum(epot)
        return epot


    def kinetic_energy(self, summed=False):
        """Calculate the kinetic energy :math:`\Braket{\Psi|T|\Psi}` of the wavepacket componentwise.
        :param summed: Wheter to sum up the individual integrals :math:`\Braket{\Phi_i|T_{i,j}|\Phi_j}`.
        :return: The kinetic energy of the wavepacket's components :math:`\Phi_i` or the overall kinetic energy of :math:`\Psi`.
        """
        tmp = [ self.grady(component) for component in xrange(self.number_components) ]
        # TODO: Check 0.25 vs orig 0.5!
        ekin = [ 0.25*norm(item)**2 for item in tmp ]

        if summed is True:
            ekin = sum(ekin)

        return ekin


    def grady(self, component):
        """Calculate the effect of :math:`-i \epsilon^2 \frac{\partial}{\partial x}`
        on a component :math:`\Phi_i` of the Hagedorn wavepacket :math:`\Psi`.
        :param component: The index :math:`i` of the component :math:`\Phi_i` on which we apply the above operator.
        :return: The modified coefficients.
        """
        (P,Q,S,p,q) = self.parameters[component]
        sh = array(self.coefficients[component].shape)
        sh = sh+1
        c = zeros(sh, dtype=complexfloating)
        k = 0
        c[k] = c[k] + p*self.coefficients[component][k]
        c[k+1] = c[k+1] + sqrt(k+1)*P*sqrt(self.eps**2*0.5)*self.coefficients[component][k]

        for k in xrange(1,self.basis_size[component]):
            c[k] = c[k] + p*self.coefficients[component][k]
            c[k+1] = c[k+1] + sqrt(k+1)*P*sqrt(self.eps**2*0.5)*self.coefficients[component][k]
            c[k-1] = c[k-1] + sqrt(k)*conj(P)*sqrt(self.eps**2*0.5)*self.coefficients[component][k]

        return c


    def project_to_canonical(self, potential):
        """Project the Hagedorn wavepacket into the canonical basis.
        :param potential: The potential :math:`V` whose eigenvectors :math:`nu_l` are used for the transformation.
        @note: This function is expensive and destructive! It modifies the coefficients
        of the I{self} instance.
        """
        # No projection for potentials with a single energy level.
        # The canonical and eigenbasis are identical here.
        if potential.get_number_components() == 1:
            return

        potential.calculate_eigenvectors()

        # Basically an ugly hack to overcome some shortcomings of the matrix function
        # and of the data layout.
        def f(q, x, component):
            x = x.reshape((self.quadrature.get_qr().get_number_nodes(),))
            z = potential.evaluate_eigenvectors_at(x)
            (row, col) = component
            return z[col][row,:]

        F = transpose(conj(self.quadrature.build_matrix(self,self,f)))
        c = self.get_coefficient_vector()
        d = dot(F, c)
        self.set_coefficient_vector(d)


    def project_to_eigen(self, potential):
        """Project the Hagedorn wavepacket into the eigenbasis of a given potential :math:`V`.
        :param potential: The potential :math:`V` whose eigenvectors :math:`nu_l` are used for the transformation.
        @note: This function is expensive and destructive! It modifies the coefficients
        of the I{self} instance.
        """
        # No projection for potentials with a single energy level.
        # The canonical and eigenbasis are identical here.
        if potential.get_number_components() == 1:
            return

        potential.calculate_eigenvectors()

        # Basically an ugly hack to overcome some shortcomings of the matrix function
        # and of the data layout.
        def f(q, x, component):
            x = x.reshape((self.quadrature.get_qr().get_number_nodes(),))
            z = potential.evaluate_eigenvectors_at(x)
            (row, col) = component
            return z[col][row,:]

        F = self.quadrature.build_matrix(self,self,f)
        c = self.get_coefficient_vector()
        d = dot(F, c)
        self.set_coefficient_vector(d)


    def to_fourier_space(self, assign=True):
        """Transform the wavepacket to Fourier space.
        :param assign: Whether to assign the transformation to
        this packet or return a cloned packet.
        @note: This is the inverse of the method I{to_real_space()}.
        """
        # The Fourier transformed parameters
        Pihats = [ (1.0j*Q, -1.0j*P, S, -q, p) for P,Q,S,p,q in self.parameters ]

        # The Fourier transformed coefficients
        coeffshat = []
        for index in xrange(self.number_components):
            k = arange(0, self.basis_size[index]).reshape((self.basis_size[index], 1))
            # Compute phase arising from the transformation
            phase = (-1.0j)**k * exp(-1.0j*self.parameters[index][3]*self.parameters[index][4] / self.eps**2)
            # Absorb phase into the coefficients
            coeffshat.append(phase * self.get_coefficients(component=index))

        if assign is True:
            self.set_parameters(Pihats)
            self.set_coefficients(coeffshat)
        else:
            FWP = self.clone()
            FWP.set_parameters(Pihats)
            FWP.set_coefficients(coeffshat)
            return FWP


    def to_real_space(self, assign=True):
        """Transform the wavepacket to real space.
        :param assign: Whether to assign the transformation to
        this packet or return a cloned packet.
        @note: This is the inverse of the method I{to_fourier_space()}.
        """
        # The inverse Fourier transformed parameters
        Pi = [ (1.0j*Q, -1.0j*P, S, q, -p) for P,Q,S,p,q in self.parameters ]

        # The inverse Fourier transformed coefficients
        coeffs = []
        for index in xrange(self.number_components):
            k = arange(0, self.basis_size[index]).reshape((self.basis_size[index], 1))
            # Compute phase arising from the transformation
            phase = (-1.0j)**k * exp(-1.0j*self.parameters[index][3]*self.parameters[index][4] / self.eps**2)
            # Absorb phase into the coefficients
            coeffs.append(phase * self.get_coefficients(component=index))

        if assign is True:
            self.set_parameters(Pi)
            self.set_coefficients(coeffs)
        else:
            RWP = self.clone()
            RWP.set_parameters(Pi)
            RWP.set_coefficients(coeffs)
            return RWP
