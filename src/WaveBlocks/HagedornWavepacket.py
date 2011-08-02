"""The WaveBlocks Project

This file contains the class which represents a homogeneous Hagedorn wavepacket.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from functools import partial
from numpy import zeros, complexfloating, array, sum, vstack, vsplit, transpose, arange
from scipy import pi, sqrt, exp, conj, dot
from scipy.linalg import norm

from ComplexMath import cont_sqrt
from HomogeneousQuadrature import HomogeneousQuadrature


class HagedornWavepacket:
    """This class represents homogeneous vector valued wavepackets $\Ket{\Psi}$.
    """

    def __init__(self, parameters):
        """Initialize the I{HagedornWavepacket} object that represents $\Ket{\Psi}$.
        @param parameters: A I{ParameterProvider} instance or a dict containing simulation parameters.
        @raise ValueError: For $N < 1$ or $K < 2$.
        """
        #: Number of components $\Phi_i$ the wavepacket $\Ket{\Psi}$ has got.
        self.number_components = parameters["ncomponents"]

        #: Size of the basis from which we construct the wavepacket.
        self.basis_size = parameters["basis_size"]

        if self.number_components < 1:
            raise ValueError("Number of components of the hagedorn wavepacket has to be >= 1.")

        if self.basis_size < 2:
            raise ValueError("Number of basis fucntions for hagedorn wavepacket has to be >= 2.")

        # Cache the parameter values epsilon we will use over and over again.
        self.eps = parameters["eps"]

        #: The parameters initialized to zero
        self.P, self.Q, self.S, self.p, self.q = 0.0, 0.0, 0.0, 0.0, 0.0

        #: The coefficients $c^i$ of the linear combination for each component $\Phi_k$.
        self.coefficients = [ zeros((self.basis_size,1), dtype=complexfloating) for index in xrange(self.number_components) ]

        #: An object that can compute brakets via quadrature.
        self.quadrature = None

        self._cont_sqrt_cache = 0.0


    def __str__(self):
        """@return: A string describing the Hagedorn wavepacket.
        """
        s =  "Homogeneous Hagedorn wavepacket for "+str(self.number_components)+" energy level(s)\n"
        return s


    def clone(self):
        # Parameters of this packet
        params = {"ncomponents": self.number_components,
                  "basis_size":  self.basis_size,
                  "eps":         self.eps}

        # Create a new Packet
        other = HagedornWavepacket(params)
        # And copy over all (private) data
        other.set_quadrature(self.get_quadrature())
        other.set_parameters(self.get_parameters())
        other.set_coefficients(self.get_coefficients())
        other._cont_sqrt_cache = self._cont_sqrt_cache

        return other


    def get_number_components(self):
        """@return: The number $N$ of components the wavepacket $\Psi$ has."""
        return self.number_components


    def get_basis_size(self):
        """@return: The size of the basis, i.e. the number $K$ of ${\phi_k}_{k=1}^K$."""
        return self.basis_size


    def set_coefficients(self, values, component=None):
        """Update the coefficients $c$ of $\Psi$.
        @param values: The new values of the coefficients $c^i$ of $\Phi_i$.
        @param component: The index $i$ of the component we want to update with new coefficients.
        @note: This function can either set new coefficients for a single component
        $\Phi_i$ only if the I{component} attribute is set or for all components
        simultaneously if I{values} is a list of arrays. This function *DOES* copy the input data!
        @raise ValueError: For invalid indices $i$.
        """
        if component is None:
            for index, value in enumerate(values):
                if index > self.number_components-1:
                    raise ValueError("There is no component with index "+str(index)+".")

                self.coefficients[index] = value.copy().reshape((self.basis_size,1))
        else:
            if component > self.number_components-1:
                raise ValueError("There is no component with index "+str(component)+".")

            self.coefficients[component] = values.copy().reshape((self.basis_size,1))


    def set_coefficient(self, component, index, value):
        """Set a single coefficient $c^i_k$ of the specified component $\Phi_i$ of $\Ket{\Psi}$.
        @param component: The index $i$ of the component $\Phi_i$ we want to update.
        @param index: The index $k$ of the coefficient $c^i_k$ we want to update.
        @param value: The new value of the coefficient $c^i_k$.
        @raise ValueError: For invalid indices $i$ or $k$.
        """
        if component > self.number_components-1:
            raise ValueError("There is no component with index "+str(component)+".")
        if index > self.basis_size-1:
            raise ValueError("There is no basis function with index "+str(index)+".")

        self.coefficients[component][index] = value


    def get_coefficients(self, component=None):
        """Returns the coefficients $c^i$ for some components $\Phi_i$ of $\Ket{\Psi}$.
        @keyword component: The index $i$ of the coefficients $c^i$ we want to get.
        @return: The coefficients $c^i$ either for all components $\Phi_i$
        or for a specified one.
        @note: This function *DOES* copy the output data!
        """
        if component is None:
            return [ item.copy() for item in self.coefficients ]
        else:
            return self.coefficients[component].copy()


    def get_coefficient_vector(self):
        """@return: The coefficients $c^i$ of all components $\Phi_i$ as a single long column vector.
        @note: This function does *NOT* copy the output data! This is for efficiency as this
        routine is used in the innermost loops.
        """
        vec = vstack(self.coefficients)
        return vec


    def set_coefficient_vector(self, vector):
        """Set the coefficients for all components $\Phi_i$ simultaneously.
        @param vector: The coefficients of all components as a single long column vector.
        @note: This function does *NOT* copy the input data! This is for efficiency as this
        routine is used in the innermost loops.
        """
        cl = vsplit(vector, self.number_components)
        for index in xrange(self.number_components):
            self.coefficients[index] = cl[index]


    def get_parameters(self, component=None):
        """Get the Hagedorn parameters $\Pi$ of the wavepacket $\Psi$.
        @param component: Dummy parameter for API compatibility with the inhomogeneous packets.
        @return: The Hagedorn parameters $P$, $Q$, $S$, $p$, $q$ of $\Psi$ in this order.
        """
        return (self.P, self.Q, self.S, self.p, self.q)


    def set_parameters(self, parameters):
        """Set the Hagedorn parameters $\Pi$ of the wavepacket $\Psi$.
        @param parameters: The Hagedorn parameters $P$, $Q$, $S$, $p$, $q$ of $\Psi$ in this order.
        """
        (self.P, self.Q, self.S, self.p, self.q) = parameters


    def set_quadrature(self, quadrature):
        """Set the I{HomogeneousQuadrature} instance used for evaluating brakets.
        @param quadrature: The new I{HomogeneousQuadrature} instance. May be I{None}
        to use a dafault one with a quadrature rule of order $K+4$.
        """
        if quadrature is None:
            self.quadrature = HomogeneousQuadrature(order=self.basis_size + 4)
        else:
            self.quadrature = quadrature


    def get_quadrature(self):
        """Return the I{HomogeneousQuadrature} instance used for evaluating brakets.
        @return: The current instance I{HomogeneousQuadrature}.
        """
        return self.quadrature


    def evaluate_basis_at(self, nodes, prefactor=False):
        """Evaluate the Hagedorn functions $\phi_k$ recursively at the given nodes $\gamma$.
        @param nodes: The nodes $\gamma$ at which the Hagedorn functions are evaluated.
        @keyword prefactor: Whether to include a factor of $\left(\det\ofs{Q}\right)^{-\frac{1}{2}}$.
        @return: Returns a twodimensional array $H$ where the entry $H[k,i]$ is the value
        of the $k$-th Hagedorn function evaluated at the node $i$.
        """
        H = zeros((self.basis_size, nodes.size), dtype=complexfloating)

        Qinv = self.Q**(-1.0)
        Qbar = conj(self.Q)
        nodes = nodes.reshape((1,nodes.size))

        H[0] = pi**(-0.25)*self.eps**(-0.5) * exp(1.0j/self.eps**2 * (0.5*self.P*Qinv*(nodes-self.q)**2 + self.p*(nodes-self.q)))
        H[1] = Qinv*sqrt(2.0/self.eps**2) * (nodes-self.q) * H[0]

        for k in xrange(2, self.basis_size):
            H[k] = Qinv*sqrt(2.0/self.eps**2)*1.0/sqrt(k) * (nodes-self.q) * H[k-1] - Qinv*Qbar*sqrt((k-1.0)/k) * H[k-2]

        if prefactor is True:
            sqrtQ, self._cont_sqrt_cache = cont_sqrt(self.Q, reference=self._cont_sqrt_cache)
            H = 1.0/sqrtQ*H

        return H


    def evaluate_at(self, nodes, component=None, prefactor=False):
        """Evaluete the Hagedorn wavepacket $\Psi$ at the given nodes $\gamma$.
        @param nodes: The nodes $\gamma$ at which the Hagedorn wavepacket gets evaluated.
        @keyword component: The index $i$ of a single component $\Phi_i$ to evaluate.
        (Defaults to 'None' for evaluating all components.)
        @keyword prefactor: Whether to include a factor of $\left(\det\ofs{Q}\right)^{-\frac{1}{2}}$.
        @return: A list of arrays or a single array containing the values of the $\Phi_i$ at the nodes $\gamma$.
        """
        nodes = nodes.reshape((1,nodes.size))
        basis = self.evaluate_basis_at(nodes, prefactor=prefactor)
        values = [ self.coefficients[index] * basis for index in xrange(self.number_components) ]
        phase = exp(1.0j*self.S/self.eps**2)
        values = [ phase * sum(values[index], axis=0) for index in xrange(self.number_components) ]

        if not component is None:
            values = values[component]

        return values


    def get_norm(self, component=None, summed=False):
        """Calculate the $L^2$ norm of the wavepacket $\Ket{\Psi}$.
        @keyword component: The component $\Phi_i$ of which the norm is calculated.
        @keyword summed: Whether to sum up the norms of the individual components $\Phi_i$.
        @return: A list containing the norms of all components $\Phi_i$ or the overall norm of $\Psi$.
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
        """Calculate the potential energy $\Braket{\Psi|V|\Psi}$ of the wavepacket componentwise.
        @param potential: The potential energy operator $V$ as function.
        @keyword summed: Wheter to sum up the individual integrals $\Braket{\Phi_i|V_{i,j}|\Phi_j}$.
        @return: The potential energy of the wavepacket's components $\Phi_i$ or the overall potential energy of $\Psi$.
        """
        f = partial(potential, as_matrix=True)
        Q = self.quadrature.quadrature(self, f)
        tmp = [ item[0,0] for item in Q ]

        N = self.number_components

        #epot = [ 0 for i in xrange(N) ]
        #for i in range(N):
        #    epot[i] = sum(tmp[i*N:(i+1)*N])

        epot = [ sum(tmp[i*N:(i+1)*N]) for i in xrange(N) ]

        if summed is True:
            epot = sum(epot)

        return epot


    def kinetic_energy(self, summed=False):
        """Calculate the kinetic energy $\Braket{\Psi|T|\Psi}$ of the wavepacket componentwise.
        @keyword summed: Wheter to sum up the individual integrals $\Braket{\Phi_i|T_{i,j}|\Phi_j}$.
        @return: The kinetic energy of the wavepacket's components $\Phi_i$ or the overall kinetic energy of $\Psi$.
        """
        tmp = [ self.grady(component) for component in xrange(self.number_components) ]
        # TODO: Check 0.25 vs orig 0.5!
        ekin = [ 0.25*norm(item)**2 for item in tmp ]

        if summed is True:
            ekin = sum(ekin)

        return ekin


    def grady(self, component):
        """Calculate the effect of $ -i \epsilon^2 \frac{\partial}{\partial x}$
        on a component $\Phi_i$ of the Hagedorn wavepacket $\Psi$.
        @keyword component: The index $i$ of the component $\Phi_i$ on which we apply the above operator.
        @return: The modified coefficients.
        """
        sh = array(self.coefficients[component].shape)
        c = zeros(sh+1, dtype=complexfloating)
        k = 0
        c[k] = c[k] + self.p*self.coefficients[component][k]
        c[k+1] = c[k+1] + sqrt(k+1)*self.P*sqrt(self.eps**2*0.5)*self.coefficients[component][k]

        for k in xrange(1,self.basis_size):
            c[k] = c[k] + self.p*self.coefficients[component][k]
            c[k+1] = c[k+1] + sqrt(k+1)*self.P*sqrt(self.eps**2*0.5)*self.coefficients[component][k]
            c[k-1] = c[k-1] + sqrt(k)*conj(self.P)*sqrt(self.eps**2*0.5)*self.coefficients[component][k]

        return c


    def project_to_canonical(self, potential, assign=True):
        """Project the Hagedorn wavepacket into the canonical basis.
        @param potential: The potential $V$ whose eigenvectors $nu_l$ are used for the transformation.
        @keyword assign: Whether to assign the new coefficient values to the wavepacket. Default true.
        @note: This function is expensive and destructive! It modifies the coefficients
        of the I{self} instance if the I{assign} parameter is True (default).
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
        """Project the Hagedorn wavepacket into the eigenbasis of a given potential $V$.
        @param potential: The potential $V$ whose eigenvectors $nu_l$ are used for the transformation.
        @keyword assign: Whether to assign the new coefficient values to the wavepacket. Default true.
        @note: This function is expensive and destructive! It modifies the coefficients
        of the I{self} instance if the I{assign} parameter is True (default).
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
        """Transform the wavepacket to Fourier space.
        @keyword assign: Whether to assign the transformation to
        this packet or return a cloned packet.
        @note: This is the inverse of the method I{to_real_space()}.
        """
        # The Fourier transformed parameters
        Pihat = (1.0j*self.Q, -1.0j*self.P, self.S, -self.q, self.p)
        # Compute phase coming from the transformation
        k = arange(0, self.basis_size).reshape((self.basis_size, 1))
        phase = (-1.0j)**k * exp(-1.0j*self.p*self.q / self.eps**2)
        # Absorb phase into the coefficients
        coeffshat = [ phase * coeff for coeff in self.get_coefficients() ]

        if assign is True:
            self.set_parameters(Pihat)
            self.set_coefficients(coeffshat)
        else:
            FWP = self.clone()
            FWP.set_parameters(Pihat)
            FWP.set_coefficients(coeffshat)
            return FWP


    def to_real_space(self, assign=True):
        """Transform the wavepacket to real space.
        @keyword assign: Whether to assign the transformation to
        this packet or return a cloned packet.
        @note: This is the inverse of the method I{to_fourier_space()}.
        """
        # The inverse Fourier transformed parameters
        Pi = (1.0j*self.Q, -1.0j*self.P, self.S, self.q, -self.p)
        # Compute phase coming from the transformation
        k = arange(0, self.basis_size).reshape((self.basis_size, 1))
        phase = (1.0j)**k * exp(-1.0j*self.p*self.q / self.eps**2)
        # Absorb phase into the coefficients
        coeffs = [ phase * coeff for coeff in self.get_coefficients() ]

        if assign is True:
            self.set_parameters(Pi)
            self.set_coefficients(coeffs)
        else:
            RWP = self.clone()
            RWP.set_parameters(Pi)
            RWP.set_coefficients(coeffs)
            return RWP
