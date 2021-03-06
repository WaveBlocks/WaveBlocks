"""The WaveBlocks Project

This file contains the class which represents a homogeneous Hagedorn wavepacket.

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
from HomogeneousQuadrature import HomogeneousQuadrature
import GlobalDefaults as GD


class HagedornWavepacket(Wavepacket):
    r"""
    This class represents homogeneous vector valued wavepackets :math:`|\Psi\rangle`.
    """

    def __init__(self, parameters):
        r"""
        Initialize the ``HagedornWavepacket`` object that represents :math:`|\Psi\rangle`.

        :param parameters: A ``ParameterProvider`` instance or a dict containing simulation parameters.
        :raise ValueError: For :math:`N < 1` or :math:`K < 2`.
        """
        #: Number of components :math:`\Phi_i` the wavepacket :math:`|\Psi\rangle` has got.
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
            raise ValueError("Number of basis functions for Hagedorn wavepacket has to be >= 2.")

        # Cache the parameter values epsilon we will use over and over again.
        self.eps = parameters["eps"]

        #: The parameter set Pi initialized to the Harmonic Oscillator Eigenfunctions
        self.P, self.Q, self.S, self.p, self.q = GD.default_Pi

        #: The coefficients :math:`c^i` of the linear combination for each component :math:`\Phi_k`.
        self.coefficients = [ zeros((self.basis_size[index],1), dtype=complexfloating) for index in xrange(self.number_components) ]

        #: An object that can compute brakets via quadrature.
        self.quadrature = None

        self._cont_sqrt_cache = 0.0


    def __str__(self):
        r"""
        :return: A string describing the Hagedorn wavepacket.
        """
        s =  "Homogeneous Hagedorn wavepacket with "+str(self.number_components)+" components\n"
        return s


    def clone(self, keepid=False):
        # Parameters of this packet
        params = {"ncomponents": self.number_components,
                  "eps":         self.eps}

        # Create a new Packet
        other = HagedornWavepacket(params)
        # If we wish to keep the packet ID
        if keepid is True:
            other.set_id(self.get_id())
        # And copy over all (private) data
        other.set_basis_size(self.get_basis_size())
        other.set_quadrature(self.get_quadrature())
        other.set_parameters(self.get_parameters())
        other.set_coefficients(self.get_coefficients())
        other._cont_sqrt_cache = self._cont_sqrt_cache

        return other


    def get_parameters(self, component=None, aslist=False):
        r"""
        Get the Hagedorn parameters :math:`\Pi` of the wavepacket :math:`\Psi`.

        :param component: Dummy parameter for API compatibility with the inhomogeneous packets.
        :param aslist: Return a list of :math:`N` parameter tuples. This is for API compatibility with inhomogeneous packets.
        :return: The Hagedorn parameters :math:`P`, :math:`Q`, :math:`S`, :math:`p`, :math:`q` of :math:`\Psi` in this order.
        """
        if aslist is True:
            return self.number_components * [(self.P, self.Q, self.S, self.p, self.q)]
        return (self.P, self.Q, self.S, self.p, self.q)


    def set_parameters(self, parameters, component=None):
        r"""
        Set the Hagedorn parameters :math:`\Pi` of the wavepacket :math:`\Psi`.

        :param parameters: The Hagedorn parameters :math:`P`, :math:`Q`, :math:`S`, :math:`p`, :math:`q` of :math:`\Psi` in this order.
        :param component: Dummy parameter for API compatibility with the inhomogeneous packets.
        """
        (self.P, self.Q, self.S, self.p, self.q) = parameters


    def set_quadrature(self, quadrature):
        r"""
        Set the ``HomogeneousQuadrature`` instance used for evaluating brakets.

        :param quadrature: The new ``HomogeneousQuadrature`` instance. May be ``None``
                           to use a dafault one with a quadrature rule of order :math:`K+4`.
        """
        # TODO: Put an "extra accuracy" parameter into global defaults with value of 4.
        # TODO: Improve on the max(basis_size) later
        # TODO: Rethink if wavepackets should contain a QR
        if quadrature is None:
            self.quadrature = HomogeneousQuadrature(order=max(self.basis_size) + 4)
        else:
            self.quadrature = quadrature


    def get_quadrature(self):
        r"""
        Return the ``HomogeneousQuadrature`` instance used for evaluating brakets.

        :return: The current instance ``HomogeneousQuadrature``.
        """
        return self.quadrature


    def evaluate_basis_at(self, nodes, component=None, prefactor=False):
        r"""
        Evaluate the Hagedorn functions :math:`\phi_k` recursively at the given nodes :math:`\gamma`.

        :param nodes: The nodes :math:`\gamma` at which the Hagedorn functions are evaluated.
        :param component: Takes the basis size :math:`K_i` of this component :math:`i` as upper bound for :math:`K`.
        :param prefactor: Whether to include a factor of :math:`\left(\det\left(Q\right)\right)^{-\frac{1}{2}}`.
        :return: Returns a twodimensional :math:`K` times #nodes array :math:`H` where the entry :math:`H[k,i]` is
                 the value of the :math:`k`-th Hagedorn function evaluated at the node :math:`i`.
        """
        if component is not None:
            basis_size = self.basis_size[component]
        else:
            # Evaluate up to maximal :math:`K_i` and slice later if necessary
            basis_size = max(self.basis_size)

        H = zeros((basis_size, nodes.size), dtype=complexfloating)

        Qinv = self.Q**(-1.0)
        Qbar = conj(self.Q)
        nodes = nodes.reshape((1,nodes.size))

        H[0] = pi**(-0.25)*self.eps**(-0.5) * exp(1.0j/self.eps**2 * (0.5*self.P*Qinv*(nodes-self.q)**2 + self.p*(nodes-self.q)))
        H[1] = Qinv*sqrt(2.0/self.eps**2) * (nodes-self.q) * H[0]

        for k in xrange(2, basis_size):
            H[k] = Qinv*sqrt(2.0/self.eps**2)*1.0/sqrt(k) * (nodes-self.q) * H[k-1] - Qinv*Qbar*sqrt((k-1.0)/k) * H[k-2]

        if prefactor is True:
            sqrtQ, self._cont_sqrt_cache = cont_sqrt(self.Q, reference=self._cont_sqrt_cache)
            H = 1.0/sqrtQ*H

        return H


    def evaluate_at(self, nodes, component=None, prefactor=False):
        r"""
        Evaluete the Hagedorn wavepacket :math:`\Psi` at the given nodes :math:`\gamma`.

        :param nodes: The nodes :math:`\gamma` at which the Hagedorn wavepacket gets evaluated.
        :param component: The index :math:`i` of a single component :math:`\Phi_i` to evaluate. (Defaults to 'None' for evaluating all components.)
        :param prefactor: Whether to include a factor of :math:`\left(\det\left(Q\right)\right)^{-\frac{1}{2}}`.
        :return: A list of arrays or a single array containing the values of the :math:`\Phi_i` at the nodes :math:`\gamma`.
        """
        nodes = nodes.reshape((1,nodes.size))
        basis = self.evaluate_basis_at(nodes, component=component, prefactor=prefactor)
        phase = exp(1.0j*self.S/self.eps**2)

        if component is not None:
            values = phase * sum(self.coefficients[component] * basis, axis=0)
        else:
            # Remember to slice the basis to the correct basis size for each component
            values = [ phase * sum(self.coefficients[index] * basis[:self.basis_size[index],:], axis=0) for index in xrange(self.number_components) ]

        return values


    def get_norm(self, component=None, summed=False):
        r"""
        Calculate the :math:`L^2` norm of the wavepacket :math:`|\Psi\rangle`.

        :param component: The component :math:`\Phi_i` of which the norm is calculated.
        :param summed: Whether to sum up the norms of the individual components :math:`\Phi_i`.
        :return: A list containing the norms of all components :math:`\Phi_i` or the overall norm of :math:`\Psi`.
        """
        if component is not None:
            result = norm(self.coefficients[component])
        else:
            result = [ norm(item) for item in self.coefficients ]

            if summed is True:
                result = reduce(lambda x,y: x+conj(y)*y, result, 0)
                result = sqrt(result)

        return result


    def potential_energy(self, potential, summed=False):
        r"""
        Calculate the potential energy :math:`\langle\Psi|V|\Psi\rangle` of the wavepacket componentwise.

        :param potential: The potential energy operator :math:`V` as function.
        :param summed: Wheter to sum up the individual integrals :math:`\langle\Phi_i|V_{i,j}|\Phi_j\rangle`.
        :return: The potential energy of the wavepacket's components :math:`\Phi_i` or the overall potential energy of :math:`\Psi`.
        """
        f = partial(potential, as_matrix=True)
        Q = self.quadrature.quadrature(self, f)
        tmp = [ item[0,0] for item in Q ]

        N = self.number_components
        epot = [ sum(tmp[i*N:(i+1)*N]) for i in xrange(N) ]

        if summed is True:
            epot = sum(epot)

        return epot


    def kinetic_energy(self, summed=False):
        r"""
        Calculate the kinetic energy :math:`\langle\Psi|T|\Psi\rangle` of the wavepacket componentwise.

        :param summed: Wheter to sum up the individual integrals :math:`\langle\Phi_i|T_{i,j}|\Phi_j\rangle`.
        :return: The kinetic energy of the wavepacket's components :math:`\Phi_i` or the overall kinetic energy of :math:`\Psi`.
        """
        tmp = [ self.grady(component) for component in xrange(self.number_components) ]
        # TODO: Check 0.25 vs orig 0.5!
        ekin = [ 0.25*norm(item)**2 for item in tmp ]

        if summed is True:
            ekin = sum(ekin)

        return ekin


    def grady(self, component):
        r"""
        Compute the effect of the operator :math:`-i \varepsilon^2 \frac{\partial}{\partial x}` on the basis
        functions of a component :math:`\Phi_i` of the Hagedorn wavepacket :math:`\Psi`.

        :param component: The index :math:`i` of the component :math:`\Phi_i` on which we apply the above operator.
        :return: The modified coefficients.
        """
        sh = array(self.coefficients[component].shape)
        c = zeros(sh+1, dtype=complexfloating)
        k = 0
        c[k] = c[k] + self.p*self.coefficients[component][k]
        c[k+1] = c[k+1] + sqrt(k+1)*self.P*sqrt(self.eps**2*0.5)*self.coefficients[component][k]

        for k in xrange(1,self.basis_size[component]):
            c[k] = c[k] + self.p*self.coefficients[component][k]
            c[k+1] = c[k+1] + sqrt(k+1)*self.P*sqrt(self.eps**2*0.5)*self.coefficients[component][k]
            c[k-1] = c[k-1] + sqrt(k)*conj(self.P)*sqrt(self.eps**2*0.5)*self.coefficients[component][k]

        return c


    def project_to_canonical(self, potential, assign=True):
        r"""
        Project the Hagedorn wavepacket into the canonical basis.

        :param potential: The potential :math:`V` whose eigenvectors :math:`nu_l` are used for the transformation.
        :param assign: Whether to assign the new coefficient values to the wavepacket. Default true.

        .. note:: This function is expensive and destructive! It modifies the coefficients
                  of the ``self`` instance if the ``assign`` parameter is True (default).
        """
        # No projection for potentials with a single energy level.
        # The canonical and eigenbasis are identical here.
        if potential.get_number_components() == 1:
            return

        potential.calculate_eigenvectors()

        # Basically an ugly hack to overcome some shortcomings of the matrix function
        # and of the data layout.
        def f(q, x):
            x = x.reshape((self.quadrature.get_qr().get_number_nodes(),))
            z = potential.evaluate_eigenvectors_at(x)

            result = []

            for col in xrange(self.number_components):
                for row in xrange(self.number_components):
                    result.append( z[col][row,:] )

            return result

        F = transpose(conj(self.quadrature.build_matrix(self, f)))
        c = self.get_coefficient_vector()
        d = dot(F, c)
        if assign is True:
            self.set_coefficient_vector(d)
        else:
            return d


    def project_to_eigen(self, potential, assign=True):
        r"""
        Project the Hagedorn wavepacket into the eigenbasis of a given potential :math:`V`.

        :param potential: The potential :math:`V` whose eigenvectors :math:`nu_l` are used for the transformation.
        :param assign: Whether to assign the new coefficient values to the wavepacket. Default true.

        .. note:: This function is expensive and destructive! It modifies the coefficients
                  of the ``self`` instance if the ``assign`` parameter is True (default).
        """
        # No projection for potentials with a single energy level.
        # The canonical and eigenbasis are identical here.
        if potential.get_number_components() == 1:
            return

        potential.calculate_eigenvectors()

        # Basically an ugly hack to overcome some shortcomings of the matrix function
        # and of the data layout.
        def f(q, x):
            x = x.reshape((self.quadrature.get_qr().get_number_nodes(),))
            z = potential.evaluate_eigenvectors_at(x)

            result = []

            for col in xrange(self.number_components):
                for row in xrange(self.number_components):
                    result.append( z[col][row,:] )

            return result

        F = self.quadrature.build_matrix(self, f)
        c = self.get_coefficient_vector()
        d = dot(F, c)
        if assign:
            self.set_coefficient_vector(d)
        else:
            return d


    def to_fourier_space(self, assign=True):
        r"""
        Transform the wavepacket to Fourier space.

        :param assign: Whether to assign the transformation to this packet or return a cloned packet.

        .. note:: This is the inverse of the method ``to_real_space()``.
        """
        # The Fourier transformed parameters
        Pihat = (1.0j*self.Q, -1.0j*self.P, self.S, -self.q, self.p)

        # The Fourier transformed coefficients
        coeffshat = []
        for index in xrange(self.number_components):
            k = arange(0, self.basis_size[index]).reshape((self.basis_size[index], 1))
            # Compute phase arising from the transformation
            phase = (-1.0j)**k * exp(-1.0j*self.p*self.q / self.eps**2)
            # Absorb phase into the coefficients
            coeffshat.append(phase * self.get_coefficients(component=index))

        if assign is True:
            self.set_parameters(Pihat)
            self.set_coefficients(coeffshat)
        else:
            FWP = self.clone()
            FWP.set_parameters(Pihat)
            FWP.set_coefficients(coeffshat)
            return FWP


    def to_real_space(self, assign=True):
        r"""
        Transform the wavepacket to real space.

        :param assign: Whether to assign the transformation to this packet or return a cloned packet.

        .. note:: This is the inverse of the method ``to_fourier_space()``.
        """
        # The inverse Fourier transformed parameters
        Pi = (1.0j*self.Q, -1.0j*self.P, self.S, self.q, -self.p)

        # The inverse Fourier transformed coefficients
        coeffs = []
        for index in xrange(self.number_components):
            k = arange(0, self.basis_size[index]).reshape((self.basis_size[index], 1))
            # Compute phase arising from the transformation
            phase = (1.0j)**k * exp(-1.0j*self.p*self.q / self.eps**2)
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
