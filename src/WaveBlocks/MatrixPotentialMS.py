"""The WaveBlocks Project

This file contains code for the representation of potentials for three and more components.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from functools import partial
import sympy
import numpy
from scipy import linalg
import numdifftools as ndt

from MatrixPotential import MatrixPotential


class MatrixPotentialMS(MatrixPotential):
    """This class represents a matrix potential $V\ofs{x}$. The potential is given as an analytical
    expression with a matrix of size bigger than $2 \times 2$. Some calculations
    with the potential are supported. For example calculation of eigenvalues and
    exponentials and numerical evaluation. Further, there are methods for
    splitting the potential into a Taylor expansion and for basis transformations
    between canonical and eigenbasis. All methods use numerical techniques because
    symbolical calculations are unfeasible.
    """

    def __init__(self, expression, variables):
        """Create a new I{MatrixPotentialMS} instance for a given potential matrix $V\ofs{x}$.
        @param expression: An expression representing the potential.
        """
        #: The variable $x$ that represents position space.
        self.x = variables[0]
        #: The matrix of the potential $V\ofs{x}$.
        self.potential = expression

        self.number_components = self.potential.shape[0]

        # prepare the function in every potential matrix cell for numerical evaluation
        self.potential_n = tuple([ sympy.vectorize(0)(sympy.lambdify(self.x, item, "numpy")) for item in self.potential ])

        # {}[chi] -> [(order, function),...]
        self.taylor_eigen_n = {}

        # {}[chi] -> [remainder]
        self.remainder_eigen_s = {}
        self.remainder_eigen_n = {}

        # [] -> [remainder]
        self.remainder_eigen_ih_s = None
        self.remainder_eigen_ih_n = None


    def __str__(self):
        """Put the number of components and the analytical expression (the matrix) into a printable string.
        """
        return """Matrix potential with """ + str(self.number_components) + """ states given by matrix: V(x) = \n""" + str(self.potential)


    def get_number_components(self):
        """@return: The number $N$ of components the potential supports. This is
        also the size of the matrix.
        """
        return self.number_components


    def evaluate_at(self, nodes, component=None, as_matrix=True):
        """Evaluate the potential matrix elementwise at some given grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the potential at.
        @keyword component: The component $V_{i,j}$ that gets evaluated or 'None' to evaluate all.
        @keyword as_matrix: Returns the whole matrix $\Lambda$ instead of only a list with the eigenvalues $\lambda_i$.
        @return: A list with the $N^2$ entries evaluated at the nodes.
        """
        result = tuple([ numpy.array(f(nodes), dtype=numpy.floating) for f in self.potential_n ])

        if not component is None:
            result = result[component * self.number_components + component]

        return result


    def calculate_eigenvalues(self):
        """Calculate the eigenvalues $\lambda_i\ofs{x}$ of the potential $V\ofs{x}$.
        We do the calculations with numerical tools. The multiplicities are taken
        into account.
        @note: Note: the eigenvalues are memoized for later reuse.
        """
        # We have to use numercial techniques here, the eigenvalues are
        # calculated while evaluating them in 'evaluate_eigenvalues_at'.
        pass


    def evaluate_eigenvalues_at(self, nodes, component=None, as_matrix=False):
        """Evaluate the eigenvalues $\lambda_i\ofs{x}$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the eigenvalues at.
        @keyword component: The index $i$ of the eigenvalue $\lambda_i$ that gets evaluated.
        @keyword as_matrix: Returns the whole matrix $\Lambda$ instead of only a list with the eigenvalues $\lambda_i$.
        @return: A sorted list with $N$ entries for all the eigenvalues evaluated at the nodes. Or a
        single value if a component was specified.
        """
        result = []

        # Hack to see if we evaluate at a single value
        if type(nodes) == numpy.ndarray:
            # Max to get rid of singular dimensions
            n = max(nodes.shape)
        else:
            try:
                n = len(nodes)
            except TypeError:
                n = len([nodes])

        # Memory for storing temporary values
        tmppot = numpy.ndarray((n,self.number_components,self.number_components), dtype=numpy.floating)
        tmpew = numpy.ndarray((n,self.number_components), dtype=numpy.floating)

        # evaluate potential
        values = self.evaluate_at(nodes)

        # fill in values
        for row in xrange(0, self.number_components):
            for col in xrange(0, self.number_components):
                tmppot[:, row, col] = values[row*self.number_components + col]

        # calculate eigenvalues assuming hermitian matrix (eigvalsh for stability!)
        for i in xrange(0, n):
            ew = linalg.eigvalsh(tmppot[i,:,:])
            # Sorting the eigenvalues biggest first.
            ew.sort()
            tmpew[i,:] = ew[::-1]

        tmp = [ tmpew[:,index] for index in xrange(0, self.number_components) ]

        if component is not None:
            (row, col) = component
            if row == col:
                result = tmp[row]
            else:
                result = numpy.zeros(tmp[row].shape, dtype=numpy.floating)
        elif as_matrix is True:
            result = []
            for row in xrange(self.number_components):
                for col in xrange(self.number_components):
                    if row == col:
                        result.append(tmp[row])
                    else:
                        result.append( numpy.zeros(tmp[row].shape, dtype=numpy.floating) )
        else:
            result = tmp

        return result


    def calculate_eigenvectors(self):
        """Calculate the two eigenvectors $nu_i\ofs{x}$ of the potential $V\ofs{x}$.
        We do the calculations with numerical tools.
        @note: The eigenvectors are memoized for later reuse.
        """
        # We have to use numercial techniques here, the eigenvectors are
        # calculated while evaluating them in 'evaluate_eigenvects_at'.
        pass


    def evaluate_eigenvectors_at(self, nodes):
        """Evaluate the eigenvectors $nu_i\ofs{x}$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the eigenvectors at.
        @return: A list with the $N$ eigenvectors evaluated at the given nodes.
        """
        result = []

        n = len(nodes)
        tmppot = numpy.ndarray((n,self.number_components,self.number_components), dtype=numpy.complexfloating)
        tmpev = numpy.ndarray((n,self.number_components,self.number_components), dtype=numpy.complexfloating)

        # evaluate potential
        values = self.evaluate_at(nodes)

        # fill in values
        for row in xrange(0, self.number_components):
            for col in xrange(0, self.number_components):
                tmppot[:, row, col] = values[row*self.number_components + col]

        # calculate eigenvalues assuming hermitian matrix (eigvalsh for stability!)
        for i in xrange(0, n):
            ew, ev = linalg.eigh(tmppot[i,:,:])
            # Sorting the eigenvectors in the same order as the eigenvalues.
            ind = numpy.argsort(ew)
            ind = ind[::-1]
            evs = ev[:,ind]
            tmpev[i,:,:] = evs

        # A trick due to G. Hagedorn to get continuous eigenvectors
        for i in xrange(1, n):
            for ev in xrange(0,self.number_components):
                if numpy.dot(tmpev[i,:,ev],tmpev[i-1,:,ev]) < 0:
                    tmpev[i,:,ev] *= -1

        result = tuple([ numpy.transpose(tmpev[:,:,index]) for index in xrange(0, self.number_components) ])
        return result


    def project_to_eigen(self, nodes, values, basis=None):
        """Project a given vector from the canonical basis to the eigenbasis of the potential.
        @param nodes: The grid nodes $\gamma$ for the pointwise transformation.
        @param values: The list of vectors $\phi_i$ containing the values we want to transform.
        @keyword basis: A list of basisvectors $nu_i$. Allows to use this function for external data, similar to a static function.
        @return: Returned is another list containing the projection of the values into the eigenbasis.
        """
        if basis is None:
            self.calculate_eigenvectors()
            eigv = self.evaluate_eigenvectors_at(nodes)
        else:
            eigv = basis

        result = []
        for i in xrange(0, self.number_components):
            tmp = numpy.zeros(values[0].shape, dtype=numpy.complexfloating)
            for j in xrange(0, self.number_components):
                tmp += eigv[i][j,:] * values[j]
            result.append(tmp)

        return tuple(result)


    def project_to_canonical(self, nodes, values, basis=None):
        """Project a given vector from the potential's eigenbasis to the canonical basis.
        @param nodes: The grid nodes $\gamma$ for the pointwise transformation.
        @param values: The list of vectors $\varphi_i$ containing the values we want to transform.
        @keyword basis: A list of basis vectors $nu_i$. Allows to use this function for external data, similar to a static function.
        @return: Returned is another list containing the projection of the values into the eigenbasis.
        """
        if basis is None:
            self.calculate_eigenvectors()
            eigv = self.evaluate_eigenvectors_at(nodes)
        else:
            eigv = basis

        result = []
        for i in xrange(0, self.number_components):
            tmp = numpy.zeros(values[0].shape, dtype=numpy.complexfloating)
            for j in xrange(0, self.number_components):
                tmp += eigv[j][i,:] * values[j]
            result.append(tmp)

        return tuple(result)


    def calculate_exponential(self, factor=1):
        """Calculate the matrix exponential $E = \exp\ofs{\alpha M}$. In the case where
        the matrix is of size bigger than $2 \times 2$ symbolical calculations become
        unfeasible. We use numerical approximations to determine the matrix exponential.
        @keyword factor: A prefactor $\alpha$ in the exponential.
        """
        # Store the factor for later numerical computations.
        self.factor = factor


    def evaluate_exponential_at(self, nodes):
        """Evaluate the exponential of the potential matrix $V$ at some grid nodes $\gamma$.
        For matrices of size $> 2$ we do completely numerical exponentation.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the exponential at.
        @return: The numerical approximation of the matrix exponential at the given grid nodes.
        """
        n = len(nodes)
        tmp = numpy.ndarray((n,self.number_components,self.number_components), dtype=numpy.complexfloating)

        # evaluate potential
        values = self.evaluate_at(nodes)

        # fill in values
        for row in xrange(0, self.number_components):
            for col in xrange(0, self.number_components):
                tmp[:, row, col] = self.factor * values[row*self.number_components + col]

        # calculate exponential
        for i in xrange(0, n):
            tmp[i,:,:] = linalg.expm(tmp[i,:,:], 10)

        result = tuple([ tmp[:,row,col] for row in xrange(self.number_components) for col in xrange(self.number_components) ])
        return result


    def calculate_jacobian(self):
        """Calculate the jacobian matrix for each component $V_{i,j}$ of the potential.
        For potentials which depend only one variable $x$, this equals the first derivative.
        """
        self.jacobian_s = tuple([ sympy.diff(item, self.x) for item in self.potential ])
        self.jacobian_n = tuple([ sympy.vectorize(0)(sympy.lambdify(self.x, item, "numpy")) for item in self.jacobian_s ])


    def evaluate_jacobian_at(self, nodes, component=None):
        """Evaluate the jacobian at some grid nodes $\gamma$ for each component
        $V_{i,j}$ of the potential.
        @param nodes: The grid nodes $\gamma$ the jacobian gets evaluated at.
        @keyword component: The index tuple $\left(i,j\right)$ that specifies
        the potential's entry of which the jacobian is evaluated. (Defaults to 'None' to evaluate all)
        @return: Either a list or a single value depending on the optional parameters.
        """
        if not component is None:
            values = self.jacobian_n[component * self.number_components + component](nodes)
        else:
            values = tuple([ f(nodes) for f in self.jacobian_n ])

        return values


    def calculate_hessian(self):
        """Calculate the hessian matrix for each component $V_{i,j}$ of the potential.
        For potentials which depend only one variable $x$, this equals the second derivative.
        """
        self.hessian_s = tuple([ sympy.diff(item, self.x, 2) for item in self.potential ])
        self.hessian_n = tuple([ sympy.vectorize(0)(sympy.lambdify(self.x, item, "numpy")) for item in self.hessian_s ])


    def evaluate_hessian_at(self, nodes, component=None):
        """Evaluate the hessian at some grid nodes $\gamma$ for each component
        $V_{i,j}$ of the potential.
        @param nodes: The grid nodes $\gamma$ the hessian gets evaluated at.
        @keyword component: The index tuple $\left(i,j\right)$ that specifies
        the potential's entry of which the hessian is evaluated. (Or 'None' to evaluate all)
        @return: Either a list or a single value depending on the optional parameters.
        """
        if not component is None:
            values = self.hessian_n[component * self.number_components + component](nodes)
        else:
            values = tuple([ f(nodes) for f in self.hessian_n ])

        return values


    def _calculate_local_quadratic_component(self, diagonal_component):
        """Calculate the local quadratic approximation matrix $U$ of the potential's
        eigenvalues in $\Lambda$. This function is used for the homogeneous case and
        takes into account the leading component $\chi$.
        @param diagonal_component: Specifies the index $i$ of the eigenvalue $\lambda_i$
        that gets expanded into a Taylor series $u_i$.
        Calculate the local quadratic approximation matrix $U$ of all the
        potential's eigenvalues in $\Lambda$. This function is used for the inhomogeneous case.
        """
        if self.taylor_eigen_n.has_key(diagonal_component):
            # Calculation already done at some earlier time
            return
        else:
            self.taylor_eigen_n[diagonal_component] = []

        # Use numerical differentiation for the case of three and more states.
        # We can not solve these problems by symbolical manipulations.
        v = partial(self.evaluate_eigenvalues_at, component=(diagonal_component,diagonal_component))
        self.taylor_eigen_n[diagonal_component].append((0,v))

        vj = ndt.Derivative(v)
        self.taylor_eigen_n[diagonal_component].append((1,vj))

        vh = ndt.Derivative(v, derOrder=2)
        self.taylor_eigen_n[diagonal_component].append((2,vh))


    def calculate_local_quadratic(self, diagonal_component=None):
        """Calculate the local quadratic approximation matrix $U$ of the potential's
        eigenvalues in $\Lambda$. This function can be used for the homogeneous case
        and takes into account the leading component $\chi$.
        If the parameter $\chi$ is not given, calculate the local quadratic approximation
        matrix $U$ of all the potential's eigenvalues in $\Lambda$. This function is used
        for the inhomogeneous case.
        @param diagonal_component: Specifies the index $i$ of the eigenvalue $\lambda_i$
        that gets expanded into a Taylor series $u_i$.
        """
        if diagonal_component is not None:
            self._calculate_local_quadratic_component(diagonal_component)
        else:
            for component in xrange(self.number_components):
                self._calculate_local_quadratic_component(component)


    def evaluate_local_quadratic_at(self, nodes, diagonal_component):
        """Numerically evaluate the local quadratic approximation matrix $U$ of
        the potential's eigenvalues in $\Lambda$ at the given grid nodes $\gamma$.
        This function is used for the homogeneous case and takes into account the
        leading component $\chi$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the quadratic approximation at.
        @return: A list of arrays containing the values of $U_{i,j}$ at the nodes $\gamma$.
        Numerically evaluate the local quadratic approximation matrix $U$ of
        the potential's eigenvalues in $\Lambda$ at the given grid nodes $\gamma$.
        This function is used for the inhomogeneous case.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the quadratic approximation at.
        @keyword component: The component $\left(i,j\right)$ of the quadratic approximation
        matrix $U$ that is evaluated.
        @return: A list of arrays or a single array containing the values of $U_{i,j}$ at the nodes $\gamma$.
        """
        if diagonal_component is not None:
            return tuple([ f(nodes) for order, f in self.taylor_eigen_n[diagonal_component] ])
        else:
            return tuple([
                [ numpy.array(f(nodes), dtype=numpy.floating) for order, f in item ]
                for item in self.taylor_eigen_n.itervalues()
                ])


    def _calculate_local_remainder_component(self, diagonal_component):
        self.calculate_local_quadratic(diagonal_component=diagonal_component)

        v, vj, vh = [ item[1] for item in self.taylor_eigen_n[diagonal_component] ]

        quadratic = lambda q, node: numpy.real(v(q) + vj(q)*(node-q) + 0.5*vh(q)*(node-q)**2)

        self.remainder_eigen_s[diagonal_component] = []
        for row in xrange(self.number_components):
            for col in xrange(self.number_components):
                # Avoid closure issues
                def element(row, col):
                    if row == col:
                        return lambda q, node: self.potential_n[row*self.number_components+col](node) - quadratic(q, node)
                    else:
                        return lambda q, node: self.potential_n[row*self.number_components+col](node)

                self.remainder_eigen_s[diagonal_component].append(element(row, col))

        self.remainder_eigen_n[diagonal_component] = tuple([ numpy.vectorize(item) for item in self.remainder_eigen_s[diagonal_component] ])


    def _calculate_local_remainder_inhomogeneous(self):
        def f(v,vj,vh):
            return lambda q, node: numpy.real(v(q) + vj(q)*(node-q) + 0.5*vh(q)*(node-q)**2)

        quadratic = []
        for item in self.taylor_eigen_n.itervalues():
            quadratic.append(f(item[0][1], item[1][1], item[2][1]))

        self.remainder_eigen_ih_s = []
        for row in xrange(self.number_components):
            for col in xrange(self.number_components):
                # Avoid closure issues
                def element(row, col):
                    if row == col:
                        return lambda q, node: self.potential_n[row*self.number_components+col](node) - quadratic[row](q, node)
                    else:
                        return lambda q, node: self.potential_n[row*self.number_components+col](node)

                self.remainder_eigen_ih_s.append( element(row, col) )

        self.remainder_eigen_ih_n = tuple([ numpy.vectorize(item) for item in self.remainder_eigen_ih_s ])


    def calculate_local_remainder(self, diagonal_component=None):
        """Calculate the non-quadratic remainder matrix $W$ of the quadratic
        approximation matrix $U$ of the potential's eigenvalue matrix $\Lambda$.
        This function is used for the homogeneous case and takes into account the
        leading component $\chi$.
        @param diagonal_component: Specifies the index $\chi$ of the leading component $\lambda_\chi$.
        """
        """Calculate the non-quadratic remainder matrix $W$ of the quadratic
        approximation matrix $U$ of the potential's eigenvalue matrix $\Lambda$.
        This function is used for the inhomogeneous case.
        """
        if diagonal_component is not None:
            self._calculate_local_remainder_component(diagonal_component)
        else:
            self._calculate_local_remainder_inhomogeneous()


    def evaluate_local_remainder_at(self, position, nodes, diagonal_component=None, component=None):
        """Numerically evaluate the non-quadratic remainder matrix $W$ of the quadratic
        approximation matrix $U$ of the potential's eigenvalues in $\Lambda$ at the
        given nodes $\gamma$. This function is used for the homogeneous and the
        inhomogeneous case and just evaluates the remainder matrix $W$.
        @param position: The point $q$ where the Taylor series is computed.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the potential at.
        @keyword component: The component $\left(i,j\right)$ of the remainder matrix $W$
        that is evaluated.
        @return: A list with a single entry consisting of an array containing the
        values of $W$ at the nodes $\gamma$.
        """
        if diagonal_component is not None:
            data = self.remainder_eigen_n[diagonal_component]
        else:
            data = self.remainder_eigen_ih_n

        if component is not None:
            (row, col) = component
            f = data[row*self.number_components+col]
            return f(numpy.real(position), numpy.real(nodes))
        else:
            return tuple([ f(numpy.real(position), numpy.real(nodes)) for f in data ])
