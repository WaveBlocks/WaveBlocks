"""The WaveBlocks Project

This file contains code for the representation of potentials for a single component.
These potential are of course scalar ones.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sympy
import numpy

from MatrixPotential import MatrixPotential


class MatrixPotential1S(MatrixPotential):
    """This class represents a scalar potential $V\ofs{x}$. The potential is given as an
    analytical $1 \times 1$ matrix expression. Some symbolic calculations with
    the potential are supported. For example calculation of eigenvalues and
    exponentials and numerical evaluation. Further, there are methods for
    splitting the potential into a Taylor expansion and for basis transformations
    between canonical and eigenbasis.
    """

    def __init__(self, expression, variables):
        """Create a new I{MatrixPotential1S} instance for a given potential matrix $V\ofs{x}$.
        @param expression: An expression representing the potential.
        """
        #: The variable $x$ that represents position space.
        self.x = variables[0]
        #: The matrix of the potential $V\ofs{x}$.
        self.potential = expression
        # Unpack single matrix entry
        self.potential = self.potential[0,0]
        self.exponential = None

        self.number_components = 1

        # prepare the function in every potential matrix cell for numerical evaluation
        self.potential_n = sympy.vectorize(0)(sympy.lambdify(self.x, self.potential, "numpy"))

        # Symbolic and numerical eigenvalues and eigenvectors
        self.eigenvalues_s = None
        self.eigenvalues_n = None
        self.eigenvectors_s = None
        self.eigenvectors_n = None

        self.taylor_eigen_s = None
        self.taylor_eigen_n = None

        self.remainder_eigen_s = None
        self.remainder_eigen_n = None


    def __str__(self):
        """Put the number of components and the analytical expression (the matrix) into a printable string.
        """
        return """Scalar potential given by the expression: V(x) = \n""" + str(self.potential)


    def get_number_components(self):
        """@return: The number $N$ of components the potential supports. In the
        one dimensional case, it's just 1.
        """
        return 1


    def evaluate_at(self, nodes, component=0, as_matrix=False):
        """Evaluate the potential matrix elementwise at some given grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the potential at.
        @keyword component: The component $V_{i,j}$ that gets evaluated or 'None' to evaluate all.
        @keyword as_matrix: Dummy parameter which has no effect here.
        @return: A list with the single entry evaluated at the nodes.
        """
        return tuple([ numpy.array(self.potential_n(nodes), dtype=numpy.floating) ])


    def calculate_eigenvalues(self):
        """Calculate the eigenvalue $\lambda_0\ofs{x}$ of the potential $V\ofs{x}$.
        In the scalar case this is just the matrix entry $V_{0,0}$.
        @note: This function is idempotent and the eigenvalues are memoized for later reuse.
        """
        if self.eigenvalues_s is None:
            self.eigenvalues_s = self.potential
            self.eigenvalues_n = sympy.vectorize(0)(sympy.lambdify(self.x, self.potential, "numpy"))


    def evaluate_eigenvalues_at(self, nodes, component=None, as_matrix=False):
        """Evaluate the eigenvalue $\lambda_0\ofs{x}$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the eigenvalue at.
        @keyword diagonal_component: Dummy parameter that has no effect here.
        @keyword as_matrix: Dummy parameter which has no effect here.
        @return: A list with the single eigenvalue evaluated at the nodes.
        """
        self.calculate_eigenvalues()

        return tuple([ numpy.array(self.eigenvalues_n(nodes)) ])


    def calculate_eigenvectors(self):
        """Calculate the eigenvector $nu_0\ofs{x}$ of the potential $V\ofs{x}$.
        In the scalar case this is just the value $1$.
        @note: This function is idempotent and the eigenvectors are memoized for later reuse.
        """
        if self.eigenvectors_s is None:
            self.eigenvectors_s = sympy.Matrix([[1]])
            self.eigenvectors_n = sympy.vectorize(0)(sympy.lambdify(self.x, 1, "numpy"))


    def evaluate_eigenvectors_at(self, nodes):
        """Evaluate the eigenvector $nu_0\ofs{x}$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the eigenvector at.
        @return: A list with the eigenvector evaluated at the given nodes.
        """
        self.calculate_eigenvectors()

        return tuple([ numpy.ones((1, len(nodes)), dtype=numpy.floating) ])


    def project_to_eigen(self, nodes, values, basis=None):
        """Project a given vector from the canonical basis to the eigenbasis of the potential.
        @param nodes: The grid nodes $\gamma$ for the pointwise transformation.
        @param values: The list of vectors $\varphi_i$ containing the values we want to transform.
        @keyword basis: A list of basisvectors $nu_i$. Allows to use this function for external data, similar to a static function.
        @return: This method does nothing and returns the values.
        """
        return [ values[0].copy() ]


    def project_to_canonical(self, nodes, values, basis=None):
        """Project a given vector from the potential's eigenbasis to the canonical basis.
        @param nodes: The grid nodes $\gamma$ for the pointwise transformation.
        @param values: The list of vectors $\varphi_i$ containing the values we want to transform.
        @keyword basis: A list of basis vectors $nu_i$. Allows to use this function for external data, similar to a static function.
        @return: This method does nothing and returns the values.
        """
        return [ values[0].copy() ]


    def calculate_exponential(self, factor=1):
        """Calculate the matrix exponential $E = \exp\ofs{\alpha M}$. In this case
        the matrix is of size $1 \times 1$ thus the exponential simplifies to
        the scalar exponential function.
        @keyword factor: A prefactor $\alpha$ in the exponential.
        @note: This function is idempotent.
        """
        if self.exponential is None:
            self.exponential = sympy.exp(factor*self.potential)


    def evaluate_exponential_at(self, nodes):
        """Evaluate the exponential of the potential matrix $V$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the exponential at.
        @return: The numerical approximation of the matrix exponential at the given grid nodes.
        """
        # Hack for older sympy versions, see recent issue:
        # http://www.mail-archive.com/sympy@googlegroups.com/msg05137.html
        lookup = {"I" : 1j}

        # prepare the function of every potential matrix exponential cell for numerical evaluation
        self.expfunctions = sympy.vectorize(0)(sympy.lambdify(self.x, self.exponential, (lookup, "numpy")))

        return tuple([ numpy.array(self.expfunctions(nodes)) ])


    def calculate_jacobian(self):
        """Calculate the jacobian matrix for the component $V_{0,0}$ of the potential.
        For potentials which depend only one variable $x$, this equals the first derivative.
        """
        self.jacobian_s = sympy.diff(self.potential, self.x)
        self.jacobian_n = sympy.vectorize(0)(sympy.lambdify(self.x, self.jacobian_s, "numpy"))


    def evaluate_jacobian_at(self, nodes, component=None):
        """Evaluate the potential's jacobian at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ the jacobian gets evaluated at.
        @keyword component: Dummy parameter that has no effect here.
        @return: The value of the potential's jacobian at the given nodes.
        """
        return tuple([ self.jacobian_n(nodes) ])


    def calculate_hessian(self):
        """Calculate the hessian matrix for component $V_{0,0}$ of the potential.
        For potentials which depend only one variable $x$, this equals the second derivative.
        """
        self.hessian_s = sympy.diff(self.potential, self.x, 2)
        self.hessian_n = sympy.vectorize(0)(sympy.lambdify(self.x, self.hessian_s, "numpy"))


    def evaluate_hessian_at(self, nodes, component=None):
        """Evaluate the potential's hessian at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ the hessian gets evaluated at.
        @keyword component: Dummy parameter that has no effect here.
        @return: The value of the potential's hessian at the given nodes.
        """
        return tuple([ self.hessian_n(nodes) ])


    def calculate_local_quadratic(self, diagonal_component=None):
        """Calculate the local quadratic approximation $U$ of the potential's
        eigenvalue $\lambda$.
        @keyword diagonal_component: Dummy parameter that has no effect here.
        @note: This function is idempotent.
        """
        # Calculation already done at some earlier time?
        if self.taylor_eigen_s is not None:
            return

        self.calculate_eigenvalues()
        self.calculate_jacobian()
        self.calculate_hessian()

        self.taylor_eigen_s = [ (0, self.eigenvalues_s), (1, self.jacobian_s), (2, self.hessian_s) ]

        # Construct function to evaluate the approximation at point q at the given nodes
        assert(self.taylor_eigen_n is None)

        self.taylor_eigen_n = [
            (order, sympy.vectorize(0)(sympy.lambdify([self.x], f, "numpy")))
            for order, f in self.taylor_eigen_s
            ]


    def evaluate_local_quadratic_at(self, nodes, diagonal_component=None):
        """Numerically evaluate the local quadratic approximation $U$ of
        the potential's eigenvalue $\lambda$ at the given grid nodes $\gamma$.
        This function is used for the homogeneous case.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the quadratic approximation at.
        @return: An array containing the values of $U$ at the nodes $\gamma$.
        """
        return tuple([ numpy.array(f(nodes), dtype=numpy.floating) for order, f in self.taylor_eigen_n ])


    def calculate_local_remainder(self, diagonal_component=None):
        """Calculate the non-quadratic remainder $W$ of the quadratic
        approximation $U$ of the potential's eigenvalue $\lambda$.
        This function is used for the homogeneous case and takes into account
        the leading component $\chi$.
        @param diagonal_component: Dummy parameter that has no effect here.
        @note: This function is idempotent.
        """
        # Calculation already done at some earlier time?
        if self.remainder_eigen_s is not None:
            return

        self.calculate_eigenvalues()

        f = self.eigenvalues_s

        # point where the taylor series is computed
        q = sympy.Symbol("q")

        p = f.subs(self.x, q)
        j = sympy.diff(f, self.x)
        j = j.subs(self.x, q)
        h = sympy.diff(f, self.x, 2)
        h = h.subs(self.x, q)

        quadratic =  p + j*(self.x-q) + sympy.Rational(1,2)*h*(self.x-q)**2

        # Symbolic expression for the taylor expansion remainder term
        self.remainder_eigen_s = self.potential - quadratic

        # Construct functions to evaluate the approximation at point q at the given nodes
        assert(self.remainder_eigen_n is None)

        self.remainder_eigen_n = sympy.vectorize(1)(sympy.lambdify([q, self.x], self.remainder_eigen_s, "numpy"))


    def evaluate_local_remainder_at(self, position, nodes, diagonal_component=None, component=None):
        """Numerically evaluate the non-quadratic remainder $W$ of the quadratic
        approximation $U$ of the potential's eigenvalue $\lambda$ at the given nodes $\gamma$.
        This function is used for the homogeneous and the inhomogeneous case and
        just evaluates the remainder $W$.
        @param position: The point $q$ where the Taylor series is computed.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the potential at.
        @keyword component: Dummy parameter that has no effect here.
        @return: A list with a single entry consisting of an array containing the
        values of $W$ at the nodes $\gamma$.
        """
        return tuple([ numpy.array(self.remainder_eigen_n(position, nodes), dtype=numpy.floating) ])
