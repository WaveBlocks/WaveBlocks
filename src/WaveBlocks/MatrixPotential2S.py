"""The WaveBlocks Project

This file contains code for the representation of potentials for two components.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sympy
import numpy

from MatrixPotential import MatrixPotential


class MatrixPotential2S(MatrixPotential):
    """This class represents a matrix potential $V\ofs{x}$. The potential is given as an
    analytical $2 \times 2$ matrix expression. Some symbolic calculations with
    the potential are supported. For example calculation of eigenvalues and
    exponentials and numerical evaluation. Further, there are methods for
    splitting the potential into a Taylor expansion and for basis transformations
    between canonical and eigenbasis.
    """

    def __init__(self, expression, variables):
        """Create a new I{MatrixPotential2S} instance for a given potential matrix $V\ofs{x}$.
        @param expression: An expression representing the potential.
        """
        #: The variable $x$ that represents position space.
        self.x = variables[0]
        #: The matrix of the potential $V\ofs{x}$.
        self.potential = expression

        self.number_components = 2

        # Prepare the function in every potential matrix cell for numerical evaluation
        self.potential_n = tuple([ sympy.vectorize(0)(sympy.lambdify(self.x, item, "numpy")) for item in self.potential ])

        # Symbolic and numerical eigenvalues and eigenvectors
        self.eigenvalues_s = None
        self.eigenvalues_n = None
        self.eigenvectors_s = None
        self.eigenvectors_n = None

        # {}[chi] -> [(order, function),...]
        self.taylor_eigen_s = {}
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
        return """Matrix potential for 2 states given by matrix: V(x) = \n""" + str(self.potential)


    def get_number_components(self):
        """@return: The number $N$ of components the potential supports. This is also the size
        of the matrix. In the current case it's 2.
        """
        return 2


    def evaluate_at(self, nodes, component=None, as_matrix=True):
        """Evaluate the potential matrix elementwise at some given grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the potential at.
        @keyword component: The component $V_{i,j}$ that gets evaluated or 'None' to evaluate all.
        @keyword as_matrix: Returns the whole matrix $\Lambda$ instead of only a list with the eigenvalues $\lambda_i$.
        @return: A list with the $4$ entries evaluated at the nodes.
        """
        if component is not None:
            (row, col) = component
            f = self.potential_n[row * self.number_components + col]
            result = numpy.array(f(nodes), dtype=numpy.floating)
        else:
            result = tuple([ numpy.array(f(nodes), dtype=numpy.floating) for f in self.potential_n ])

        return result


    def calculate_eigenvalues(self):
        """Calculate the two eigenvalues $\lambda_i\ofs{x}$ of the potential $V\ofs{x}$.
        We can do this by symbolical calculations. The multiplicities are taken into account.
        @note: Note: the eigenvalues are memoized for later reuse.
        """
        if self.eigenvalues_s is not None:
            return

        a = self.potential[0,0]
        b = self.potential[0,1]
        c = self.potential[1,1]
        # Symbolic formula for the eigenvalues of a symmetric 2x2 matrix
        l1 = (sympy.sqrt(c**2-2*a*c+4*b**2+a**2)+c+a)/2
        l2 = -(sympy.sqrt(c**2-2*a*c+4*b**2+a**2)-c-a)/2

        self.eigenvalues_s = tuple([ sympy.simplify(item) for item in [l1,l2] ])
        self.eigenvalues_n = tuple([ sympy.vectorize(0)(sympy.lambdify(self.x, item, "numpy")) for item in self.eigenvalues_s ])


    def evaluate_eigenvalues_at(self, nodes, component=None, as_matrix=False):
        """Evaluate the eigenvalues $\lambda_i\ofs{x}$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the eigenvalues at.
        @keyword component: The index $i$ of the eigenvalue $\lambda_i$ that gets evaluated.
        @keyword as_matrix: Returns the whole matrix $\Lambda$ instead of only a list with the eigenvalues $\lambda_i$.
        @return: A sorted list with $2$ entries for the two eigenvalues evaluated at the nodes. Or a
        single value if a component was specified.
        """
        self.calculate_eigenvalues()

        tmp = numpy.vstack([ numpy.array(f(nodes)) for f in self.eigenvalues_n ])
        # Sort the eigenvalues
        tmp = numpy.sort(tmp, axis=0)
        tmp = [ tmp[i,:] for i in reversed(xrange(self.number_components)) ]

        if not component is None:
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
        We can do this by symbolical calculations.
        @note: The eigenvectors are memoized for later reuse.
        """
        if self.eigenvectors_s is not None:
            return

        V1 = self.potential[0,0]
        V2 = self.potential[0,1]

        theta = sympy.Rational(1,2) * sympy.atan2(V2,V1)

        # The two eigenvectors
        upper = sympy.Matrix([[sympy.cos(theta)],[sympy.sin(theta)]])
        lower = sympy.Matrix([[-sympy.sin(theta)],[sympy.cos(theta)]])

        # The symbolic expressions for the eigenvectors
        self.eigenvectors_s = (upper, lower)

        # The numerical expressions for the eigenvectors
        self.eigenvectors_n = []

        for vector in self.eigenvectors_s:
            self.eigenvectors_n.append( tuple([ sympy.vectorize(0)(sympy.lambdify(self.x, component, "numpy")) for component in vector ]) )


    def evaluate_eigenvectors_at(self, nodes):
        """Evaluate the eigenvectors $nu_i\ofs{x}$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the eigenvectors at.
        @return: A list with the two eigenvectors evaluated at the given nodes.
        """
        self.calculate_eigenvectors()

        result = []

        for vector in self.eigenvectors_n:
            tmp = numpy.zeros((self.number_components, len(nodes)), dtype=numpy.floating)
            for index in xrange(self.number_components):
                # Assure real values as atan2 is only defined for real values!
                tmp[index,:] = vector[index](numpy.real(nodes))
            result.append(tmp)

        return tuple(result)


    def project_to_eigen(self, nodes, values, basis=None):
        """Project a given vector from the canonical basis to the eigenbasis of the potential.
        @param nodes: The grid nodes $\gamma$ for the pointwise transformation.
        @param values: The list of vectors $\varphi_i$ containing the values we want to transform.
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
        """Calculate the matrix exponential $E = \exp\ofs{\alpha M}$. In this case
        the matrix is of size $2 \times 2$ thus the general exponential can be
        calculated analytically.
        @keyword factor: A prefactor $\alpha$ in the exponential.
        """
        M = factor * self.potential
        a = M[0,0]
        b = M[0,1]
        c = M[1,0]
        d = M[1,1]

        D = sympy.sqrt((a-d)**2 + 4*b*c)/2
        t = sympy.exp((a+d)/2)

        M = sympy.Matrix([[0,0],[0,0]])

        if sympy.Eq(D,0):
            # special case
            M[0,0] = sympy.simplify( t * (1 + (a-d)/2) )
            M[0,1] = sympy.simplify( t * b )
            M[1,0] = sympy.simplify( t * c )
            M[1,1] = sympy.simplify( t * (1 - (a-d)/2) )
        else:
            # general case
            M[0,0] = sympy.simplify( t * (sympy.cosh(D) + (a-d)/2 * sympy.sinh(D)/D) )
            M[0,1] = sympy.simplify( t * (b * sympy.sinh(D)/D) )
            M[1,0] = sympy.simplify( t * (c * sympy.sinh(D)/D) )
            M[1,1] = sympy.simplify( t * (sympy.cosh(D) - (a-d)/2 * sympy.sinh(D)/D) )

        self.exponential = M


    def evaluate_exponential_at(self, nodes):
        """Evaluate the exponential of the potential matrix $V$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the exponential at.
        @return: The numerical approximation of the matrix exponential at the given grid nodes.
        """
        # Hack for older sympy versions, see recent issue:
        # http://www.mail-archive.com/sympy@googlegroups.com/msg05137.html
        lookup = {"I" : 1j}

        # prepare the function of every potential matrix exponential cell for numerical evaluation
        self.expfunctions = tuple([ sympy.vectorize(0)(sympy.lambdify(self.x, item, (lookup, "numpy"))) for item in self.exponential ])

        return tuple([ numpy.array(f(nodes)) for f in self.expfunctions ])


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
        """
        if self.taylor_eigen_s.has_key(diagonal_component):
            # Calculation already done at some earlier time
            return
        else:
            self.taylor_eigen_s[diagonal_component] = []

        v = self.eigenvalues_s[diagonal_component]
        self.taylor_eigen_s[diagonal_component].append((0, v))

        vj = sympy.simplify( sympy.diff(v, self.x, 1) )
        self.taylor_eigen_s[diagonal_component].append((1, vj))

        vh = sympy.simplify( sympy.diff(v, self.x, 2) )
        self.taylor_eigen_s[diagonal_component].append((2, vh))

        # Construct functions to evaluate the approximation at point q at the given nodes
        assert(not self.taylor_eigen_n.has_key(diagonal_component))

        self.taylor_eigen_n[diagonal_component] = [
            (order, sympy.vectorize(0)(sympy.lambdify([self.x], f, "numpy")))
            for order, f in self.taylor_eigen_s[diagonal_component]
            ]


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
        self.calculate_eigenvalues()

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
        @keyword diagonal_component: The eigenvalue to which we evaluate the quadratic approximation.
        @return: A list of arrays containing the values of $U_{i,j}$ at the nodes $\gamma$.
        """
        """Numerically evaluate the local quadratic approximation matrix $U$ of
        the potential's eigenvalues in $\Lambda$ at the given grid nodes $\gamma$.
        This function is used for the inhomogeneous case.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the quadratic approximation at.
        @param diagonal_component: Specifies the index $i$ of the eigenvalue $\lambda_i$
        that gets expanded into a Taylor series $u_i$.
        @return: A list of arrays or a single array containing the values of $U_{i,j}$ at the nodes $\gamma$.
        """
        if diagonal_component is not None:
            return tuple([
                numpy.array(f(nodes), dtype=numpy.floating)
                for order, f in self.taylor_eigen_n[diagonal_component]
                ])
        else:
            return tuple([
                [ numpy.array(f(nodes), dtype=numpy.floating) for order, f in item ]
                for item in self.taylor_eigen_n.itervalues()
                ])


    def _calculate_remainder_component(self, diagonal_component):
        """Calculate the non-quadratic remainder matrix $W$ of the quadratic
        approximation matrix $U$ of the potential's eigenvalue matrix $\Lambda$.
        This function is used for the homogeneous case and takes into account the
        leading component $\chi$.
        @param diagonal_component: Specifies the index $\chi$ of the leading component $\lambda_\chi$.
        """
        if self.remainder_eigen_s.has_key(diagonal_component):
            # Calculation already done at some earlier time
            return
        else:
            self.remainder_eigen_s[diagonal_component] = []

        f = self.eigenvalues_s[diagonal_component]

        # point where the taylor series is computed
        q = sympy.Symbol("q")

        p = f.subs(self.x, q)
        j = sympy.diff(f, self.x)
        j = j.subs(self.x, q)
        h = sympy.diff(f, self.x, 2)
        h = h.subs(self.x, q)

        quadratic =  sympy.simplify(p + j*(self.x-q) + sympy.Rational(1,2)*h*(self.x-q)**2)

        for row in xrange(self.number_components):
            for col in xrange(self.number_components):
                e = self.potential[row,col]
                if col == row:
                    e = sympy.simplify(e - quadratic)
                self.remainder_eigen_s[diagonal_component].append(e)

        # Construct functions to evaluate the approximation at point q at the given nodes
        assert(not self.remainder_eigen_n.has_key(diagonal_component))

        self.remainder_eigen_n[diagonal_component] = [
            sympy.vectorize(1)(sympy.lambdify([q, self.x], item, "numpy"))
            for item in self.remainder_eigen_s[diagonal_component]
            ]


    def _calculate_local_remainder_inhomogeneous(self):
        """Calculate the non-quadratic remainder matrix $W$ of the quadratic
        approximation matrix $U$ of the potential's eigenvalue matrix $\Lambda$.
        This function is used for the inhomogeneous case.
        """
        if self.remainder_eigen_ih_s is not None:
            # Calculation already done at some earlier time
            return
        else:
            self.remainder_eigen_ih_s = []

        # Quadratic taylor series for all eigenvalues
        quadratic = []

        for item in self.eigenvalues_s:
            # point where the taylor series is computed
            q = sympy.Symbol("q")

            p = item.subs(self.x, q)
            j = sympy.diff(item, self.x)
            j = j.subs(self.x, q)
            h = sympy.diff(item, self.x, 2)
            h = h.subs(self.x, q)

            qa =  sympy.simplify(p + j*(self.x-q) + sympy.Rational(1,2)*h*(self.x-q)**2)

            quadratic.append(qa)

        for row in xrange(self.number_components):
            for col in xrange(self.number_components):
                e = self.potential[row,col]
                if col == row:
                    e = sympy.simplify(e - quadratic[row])
                self.remainder_eigen_ih_s.append(e)

        # Construct functions to evaluate the approximation at point q at the given nodes
        assert(self.remainder_eigen_ih_n is None)

        self.remainder_eigen_ih_n = [
            sympy.vectorize(1)(sympy.lambdify([q, self.x], item, "numpy"))
            for item in self.remainder_eigen_ih_s
            ]


    def calculate_local_remainder(self, diagonal_component=None):
        self.calculate_eigenvalues()

        if diagonal_component is not None:
            self._calculate_remainder_component(diagonal_component)
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
            result = numpy.array(f(position, nodes), dtype=numpy.floating)
        else:
            result = tuple([ numpy.array(f(position, nodes), dtype=numpy.floating) for f in data ])

        return result
