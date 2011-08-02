"""The WaveBlocks Project

This file contains the class which represents an inhomogeneous Hagedorn wavepacket.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from functools import partial
from numpy import zeros, complexfloating, array, sum, vstack, vsplit, transpose, arange
from scipy import pi, sqrt, exp, conj, dot
from scipy.linalg import norm

from ComplexMath import cont_sqrt
from InhomogeneousQuadrature import InhomogeneousQuadrature


class HagedornWavepacketInhomogeneous:
    """This class represents inhomogeneous vector valued wavepackets $\Ket{\Psi}$.
    """

    def __init__(self, parameters):
        """Initialize the I{HagedornWavepacketInhomogeneous} object that represents $\Ket{\Psi}$.
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

        #: Data structure that contains the Hagedorn parameters $\Pi_i$ of each component $\Phi_i$.
        self.parameters = [ [] for i in xrange(self.number_components) ]

        #: The coefficients $c^i$ of the linear combination for each component $\Phi_i$.
        self.coefficients = [ zeros((self.basis_size,1), dtype=complexfloating) for index in xrange(self.number_components) ]

        #: An object that can compute brakets via quadrature.
        self.quadrature = None

        self._cont_sqrt_cache = [ 0.0 for i in xrange(self.number_components) ]


    def __str__(self):
        """@return: A string describing the Hagedorn wavepacket.
        """
        s =  "Hagedorn multi wave packet for "+str(self.number_components)+" states\n"
        return s


    def clone(self):
        # Parameters of this packet
        params = {"ncomponents": self.number_components,
                  "basis_size":  self.basis_size,
                  "eps":         self.eps}

        # Create a new Packet
        other = HagedornWavepacketInhomogeneous(params)
        # And copy over all (private) data
        other.set_quadrature(self.get_quadrature())
        other.set_parameters(self.get_parameters())
        other.set_coefficients(self.get_coefficients())
        other._cont_sqrt_cache = [ cache for cache in self._cont_sqrt_cache ]

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
        simultaneously if I{values} is a list of arrays.
        @raise ValueError: For invalid indices $i$.
        """
        if component is None:
            for index, value in enumerate(values):
                if index > self.number_components-1:
                    raise ValueError("There is no component with index "+str(index)+".")

                self.coefficients[index] = value[:].reshape((self.basis_size,1))
        else:
            if component > self.number_components-1:
                raise ValueError("There is no component with index "+str(component)+".")

            self.coefficients[component] = values[:].reshape((self.basis_size,1))


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
        """
        if component is None:
            return self.coefficients[:]
        else:
            return self.coefficients[component]


    def get_coefficient_vector(self):
        """@return: The coefficients $c^i$ of all components $\Phi_i$ as a single long column vector.
        """
        vec = vstack(self.coefficients)
        return vec

    def set_coefficient_vector(self, vector):
        """Set the coefficients for all components $\Phi_i$ simultaneously.
        @param vector: The coefficients of all components as a single long column vector.
        """
        cl = vsplit(vector, self.number_components)
        for index in xrange(self.number_components):
            self.coefficients[index] = cl[index]


    def get_parameters(self, component=None):
        """Get the Hagedorn parameters $\Pi_i$ of each component $\Phi_i$ of the wavepacket $\Psi$.
        @keyword component: The index $i$ of the component whose parameters $\Pi_i$ we want to get.
        @return: A list with all the sets $\Pi_i$ or a single set.
        """
        if component is None:
            result = [ tuple(item) for item in self.parameters ]
        else:
            result = self.parameters[component]
        return tuple(result)


    def set_parameters(self, parameters, component=None):
        """Set the Hagedorn parameters $\Pi_i$ of each component $\Phi_i$ of the wavepacket $\Psi$.
        @param parameters: A list or a single set of Hagedorn parameters.
        @keyword component: The index $i$ of the component whose parameters $\Pi_i$ we want to update.
        """
        if component is None:
            for index, item in enumerate(parameters):
                self.parameters[index] = item[:]
        else:
            self.parameters[component] = parameters[:]


    def set_quadrature(self, quadrature):
        """Set the I{InhomogeneousQuadrature} instance used for evaluating brakets.
        @param quadrature: The new I{InhomogeneousQuadrature} instance. May be I{None}
        to use a dafault one with a quadrature rule of order $K+4$.
        """
        if quadrature is None:
            self.quadrature = InhomogeneousQuadrature(order=self.basis_size + 4)
        else:
            self.quadrature = quadrature


    def get_quadrature(self):
        """Return the I{InhomogeneousQuadrature} instance used for evaluating brakets.
        @return: The current instance I{InhomogeneousQuadrature}.
        """
        return self.quadrature


    def evaluate_basis_at(self, nodes, component, prefactor=False):
        """Evaluate the Hagedorn functions $\phi_k$ recursively at the given nodes $\gamma$.
        @param nodes: The nodes $\gamma$ at which the Hagedorn functions are evaluated.
        @param component: The index $i$ of the component whose basis functions $\phi^i_k$ we want to evaluate.
        @keyword prefactor: Whether to include a factor of $\left(\det\ofs{Q_i}\right)^{-\frac{1}{2}}$.
        @return: Returns a twodimensional array $H$ where the entry $H[k,i]$ is the value
        of the $k$-th Hagedorn function evaluated at the node $i$.
        """
        H = zeros((self.basis_size, nodes.size), dtype=complexfloating)

        (P, Q, S, p, q) = self.parameters[component]
        Qinv = Q**(-1.0)
        Qbar = conj(Q)
        nodes = nodes.reshape((1,nodes.size))

        H[0] = pi**(-0.25)*self.eps**(-0.5) * exp(1.0j/self.eps**2 * (0.5*P*Qinv*(nodes-q)**2 + p*(nodes-q)))
        H[1] = Qinv*sqrt(2.0/self.eps**2) * (nodes-q) * H[0]

        for k in xrange(2, self.basis_size):
            H[k] = Qinv*sqrt(2.0/self.eps**2)*1.0/sqrt(k) * (nodes-q) * H[k-1] - Qinv*Qbar*sqrt((k-1.0)/k) * H[k-2]

        if prefactor is True:
            sqrtQ, self._cont_sqrt_cache[component] = cont_sqrt(Q, reference=self._cont_sqrt_cache[component])
            H = 1.0/sqrtQ*H

        return H


    def evaluate_at(self, nodes, component=None, prefactor=False):
        """Evaluete the Hagedorn wavepacket $\Psi$ at the given nodes $\gamma$.
        @param nodes: The nodes $\gamma$ at which the Hagedorn wavepacket gets evaluated.
        @keyword component: The index $i$ of a single component $\Phi_i$ to evaluate.
        (Defaults to 'None' for evaluating all components.)
        @keyword prefactor: Whether to include a factor of $\left(\det\ofs{Q_i}\right)^{-\frac{1}{2}}$.
        @return: A list of arrays or a single array containing the values of the $\Phi_i$ at the nodes $\gamma$.
        """
        nodes = nodes.reshape((1,nodes.size))

        if component is None:
            values = [ zeros(nodes.shape) for index in xrange(self.number_components) ]

            for index in xrange(self.number_components):
                (P, Q, S, p, q) = self.parameters[index]

                basis = self.evaluate_basis_at(nodes, index, prefactor=prefactor)
                vals = self.coefficients[index] * basis
                phase = exp(1.0j*S/self.eps**2)
                values[index] = phase * sum(vals, axis=0)
        else:
            # Avoid the expensive evaluation of unused other bases
            (P, Q, S, p, q) = self.parameters[component]

            basis = self.evaluate_basis_at(nodes, component, prefactor=prefactor)
            vals = self.coefficients[component] * basis
            phase = exp(1.0j*S/self.eps**2)
            values = phase * sum(vals, axis=0)

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
        Q = self.quadrature.quadrature(self, self, f)

        N = self.number_components
        epot = [ 0 for i in xrange(N) ]
        for i in range(N):
            epot[i] = sum(Q[i*N:(i+1)*N])

        # Works only for eigenbasis
        #epot = [ Q[row*N][0,0] for row in xrange(N) ]

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
        """Calculate the effect of $-i \epsilon^2 \frac{\partial}{\partial x}$
        on a component $\Phi_i$ of the Hagedorn wavepacket $\Psi$.
        @keyword component: The index $i$ of the component $\Phi_i$ on which we apply the above operator.
        @return: The modified coefficients.
        """
        (P,Q,S,p,q) = self.parameters[component]
        sh = array(self.coefficients[component].shape)
        sh = sh+1
        c = zeros(sh, dtype=complexfloating)
        k = 0
        c[k] = c[k] + p*self.coefficients[component][k]
        c[k+1] = c[k+1] + sqrt(k+1)*P*sqrt(self.eps**2*0.5)*self.coefficients[component][k]

        for k in xrange(1,self.basis_size):
            c[k] = c[k] + p*self.coefficients[component][k]
            c[k+1] = c[k+1] + sqrt(k+1)*P*sqrt(self.eps**2*0.5)*self.coefficients[component][k]
            c[k-1] = c[k-1] + sqrt(k)*conj(P)*sqrt(self.eps**2*0.5)*self.coefficients[component][k]

        return c


    def project_to_canonical(self, potential):
        """Project the Hagedorn wavepacket into the canonical basis.
        @param potential: The potential $V$ whose eigenvectors $nu_l$ are used for the transformation.
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
        """Project the Hagedorn wavepacket into the eigenbasis of a given potential $V$.
        @param potential: The potential $V$ whose eigenvectors $nu_l$ are used for the transformation.
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
        @keyword assign: Whether to assign the transformation to
        this packet or return a cloned packet.
        @note: This is the inverse of the method I{to_real_space()}.
        """
        # The Fourier transformed parameters
        Pihats = [ (1.0j*Q, -1.0j*P, S, -q, p) for P,Q,S,p,q in self.parameters ]
        # Compute phase coming from the transformation
        k = arange(0, self.basis_size).reshape((self.basis_size, 1))
        phases = [ (-1.0j)**k * exp(-1.0j*p*q / self.eps**2) for P,Q,S,p,q in self.parameters ]
        # Absorb phase into the coefficients
        coeffshat = [ phase * coeffs for phase, coeffs in zip(phases, self.get_coefficients()) ]

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
        @keyword assign: Whether to assign the transformation to
        this packet or return a cloned packet.
        @note: This is the inverse of the method I{to_fourier_space()}.
        """
        # The inverse Fourier transformed parameters
        Pi = [ (1.0j*Q, -1.0j*P, S, q, -p) for P,Q,S,p,q in self.parameters ]
        # Compute phase coming from the transformation
        k = arange(0, self.basis_size).reshape((self.basis_size, 1))
        k = arange(0, self.basis_size).reshape((self.basis_size, 1))
        phases = [ (1.0j)**k * exp(-1.0j*p*q / self.eps**2) for P,Q,S,p,q in self.parameters ]
        # Absorb phase into the coefficients
        coeffs = [ phase * coeffs for phase, coeffs in zip(phases, self.get_coefficients()) ]

        if assign is True:
            self.set_parameters(Pi)
            self.set_coefficients(coeffs)
        else:
            RWP = self.clone()
            RWP.set_parameters(Pi)
            RWP.set_coefficients(coeffs)
            return RWP
