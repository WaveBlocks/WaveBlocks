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

        self.number_components = 1

        # prepare the function in every potential matrix cell for numerical evaluation
        self.functions = sympy.vectorize(0)(sympy.lambdify(self.x, self.potential, "numpy"))

        # Some flags used for memoization
        self.__valid_eigenvalues = False
        self.__valid_eigenvectors = False


    def __str__(self):
        """Put the number of components and the analytical expression (the matrix) into a printable string.
        """
        return """Scalar potential given by the expression:\n""" + str(self.potential)

        
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
        result = tuple([ numpy.array(self.functions(nodes), dtype=numpy.complexfloating) ])
        return result
        
        
    def calculate_eigenvalues(self):        
        """Calculate the eigenvalue $\lambda_0\ofs{x}$ of the potential $V\ofs{x}$.
        In the scalar case this is just the matrix entry $V_{0,0}$.
        @note: Note: the eigenvalues are memoized for later reuse.
        """
        if self.__valid_eigenvalues != True:
            self.eigenvalues_s = self.potential
            self.eigf = sympy.vectorize(0)(sympy.lambdify(self.x, self.potential, "numpy"))
            # Ok, now we have valid eigenvalues at hand
            self.__valid_eigenvalues = True
        
        
    def evaluate_eigenvalues_at(self, nodes, diagonal_component=None, as_matrix=True):
        """Evaluate the eigenvalue $\lambda_0\ofs{x}$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the eigenvalue at.
        @keyword diagonal_component: Dummy parameter that has no effect here.
        @keyword as_matrix: Dummy parameter which has no effect here.
        @return: A list with the single eigenvalue evaluated at the nodes.
        """
        self.calculate_eigenvalues()
        
        result = tuple([ numpy.array(self.eigf(nodes)) ])
        return result
        
        
    def calculate_eigenvectors(self):
        """Calculate the eigenvector $nu_0\ofs{x}$ of the potential $V\ofs{x}$.
        In the scalar case this is just the value $1$.
        @note: The eigenvectors are memoized for later reuse.
        """
        if self.__valid_eigenvectors != True:
            self.eigenvectors_s = sympy.Matrix([[1]])
            self.eigenvectors_n = sympy.vectorize(0)(sympy.lambdify(self.x, 1, "numpy"))
            # Ok, now we have valid eigenvectors at hand
            self.__valid_eigenvectors = True
            
            
    def evaluate_eigenvectors_at(self, nodes):
        """Evaluate the eigenvector $nu_0\ofs{x}$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the eigenvector at.
        @return: A list with the eigenvector evaluated at the given nodes.
        """
        self.calculate_eigenvectors()
        
        result = numpy.ones((1, len(nodes)), dtype=numpy.floating)
        return tuple([ result ])
        
        
    def project_to_eigen(self, nodes, values, basis=None):
        """Project a given vector from the canonical basis to the eigenbasis of the potential.
        @param nodes: The grid nodes $\gamma$ for the pointwise transformation.
        @param values: The list of vectors $\varphi_i$ containing the values we want to transform.
        @keyword basis: A list of basisvectors $nu_i$. Allows to use this function for external data, similar to a static function.
        @return: This method does nothing and return the values.
        """
        return values[:]


    def project_to_canonical(self, nodes, values, basis=None):
        """Project a given vector from the potential's eigenbasis to the canonical basis.
        @param nodes: The grid nodes $\gamma$ for the pointwise transformation.
        @param values: The list of vectors $\varphi_i$ containing the values we want to transform.
        @keyword basis: A list of basis vectors $nu_i$. Allows to use this function for external data, similar to a static function.
        @return: This method does nothing and return the values.
        """
        return values[:]
        
        
    def calculate_exponential(self, factor=1):
        """Calculate the matrix exponential $E = \exp\ofs{\alpha M}$. In this case
        the matrix is of size $1 \times 1$ thus the exponential simplifies to
        the scalar exponential function.
        @keyword factor: A prefactor $\alpha$ in the exponential.
        """
        self.exponential = sympy.simplify( sympy.exp(factor*self.potential) )
        
        
    def evaluate_exponential_at(self, nodes):
        """Evaluate the exponential of the potential matrix $V$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the exponential at.
        @return: The numerical approximation of the matrix exponential at the given grid nodes.
        """
        # Hack for older sympy versions, see recent issue:
        # http://www.mail-archive.com/sympy@googlegroups.com/msg05137.html
        lookup = {"I" : 1j}
        
        # prepare the function of every potential matrix exponential cell for
        # numerical evaluation
        self.expfunctions = sympy.vectorize(0)(sympy.lambdify(self.x, self.exponential, (lookup, "numpy")))

        result = tuple([ numpy.array(self.expfunctions(nodes)) ])
        return result
        
        
    def calculate_jacobian(self):
        """Calculate the jacobian matrix for the component $V_{0,0}$ of the potential.
        For potentials which depend only one variable $x$, this equals the first derivative.
        """
        self.jacobian = sympy.diff(self.potential, self.x)            
        self.jacobian_n = sympy.vectorize(0)(sympy.lambdify(self.x, self.jacobian, "numpy"))
        
        
    def evaluate_jacobian_at(self, nodes, component=None):
        """Evaluate the potential's jacobian at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ the jacobian gets evaluated at.
        @keyword component: Dummy parameter that has no effect here.
        @return: The value of the potential's jacobian at the given nodes.
        """
        values = tuple([ self.jacobian_n(nodes) ])
        return values
        

    def calculate_hessian(self):
        """Calculate the hessian matrix for component $V_{0,0}$ of the potential.
        For potentials which depend only one variable $x$, this equals the second derivative.
        """
        self.hessian = sympy.diff(self.potential, self.x, 2)
        self.hessian_n = sympy.vectorize(0)(sympy.lambdify(self.x, self.hessian, "numpy"))


    def evaluate_hessian_at(self, nodes, component=None):
        """Evaluate the potential's hessian at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ the hessian gets evaluated at.
        @keyword component: Dummy parameter that has no effect here.
        @return: The value of the potential's hessian at the given nodes.
        """
        values = tuple([ self.hessian_n(nodes) ])
        return values


    def calculate_local_quadratic(self, diagonal_component=0):
        """Calculate the local quadratic approximation $U$ of the potential's
        eigenvalue $\lambda$.
        @keyword diagonal_component: Dummy parameter that has no effect here.
        """
        self.calculate_eigenvalues()
        self.calculate_jacobian()
        self.calculate_hessian()
        
        self.quadratic_s = []
            
        self.quadratic_s.append(self.eigenvalues_s)
        self.quadratic_s.append(self.jacobian)
        self.quadratic_s.append(self.hessian)

        # Construct function to evaluate the approximation at point q at the given nodes
        self.quadratic_n = [ sympy.vectorize(0)(sympy.lambdify([self.x], item, "numpy")) for item in self.quadratic_s ]
            
                        
    def evaluate_local_quadratic_at(self, nodes):
        """Numerically evaluate the local quadratic approximation $U$ of
        the potential's eigenvalue $\lambda$ at the given grid nodes $\gamma$.
        This function is used for the homogeneous case.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the quadratic approximation at.
        @return: An array containing the values of $U$ at the nodes $\gamma$.
        """
        result = tuple([ numpy.array(f(nodes), dtype=numpy.complexfloating) for f in self.quadratic_n ])
        return result


    def calculate_local_remainder(self, diagonal_component=0):
        """Calculate the non-quadratic remainder $W$ of the quadratic
        approximation $U$ of the potential's eigenvalue $\lambda$.
        This function is used for the homogeneous case and takes into account
        the leading component $\chi$.
        @param diagonal_component: Dummy parameter that has no effect here.
        """
        self.calculate_eigenvalues()
        f = self.eigenvalues_s

        # point where the taylor series is computed
        q = sympy.Symbol("q")
        
        p = f.subs(self.x, q)
        j = sympy.diff(f, self.x)
        j = j.subs(self.x, q)
        h = sympy.diff(f, self.x, 2)
        h = h.subs(self.x, q)

        quadratic =  sympy.simplify(p + j*(self.x-q) + sympy.Rational(1,2)*h*(self.x-q)**2)

        # Symbolic expression for the taylor expansion remainder term
        self.nonquadratic_s = sympy.simplify(self.potential - quadratic)
        
        # Construct functions to evaluate the approximation at point q at the given nodes
        self.nonquadratic_n = sympy.vectorize(1)(sympy.lambdify([q, self.x], self.nonquadratic_s, "numpy"))
        
        
    def evaluate_local_remainder_at(self, position, nodes, component=None):
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
        result = tuple([ numpy.array(self.nonquadratic_n(position, nodes), dtype=numpy.complexfloating) ])
        return result


    def calculate_local_quadratic_multi(self):
        """Calculate the local quadratic approximation $U$ of the potential's
        eigenvalue $\lambda$. This function is used for the inhomogeneous case.
        @raise ValueError: There are no inhomogeneous wavepackets with a single component.
        """
        raise ValueError("There are no inhomogeneous wavepackets with a single component!")
    
    
    def evaluate_local_quadratic_multi_at(self, nodes, component=None):
        """Numerically evaluate the local quadratic approximation $U$ of
        the potential's eigenvalue $\lambda$ at the given grid nodes $\gamma$.
        This function is used for the inhomogeneous case.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the quadratic approximation at.
        @keyword component: Dummy parameter that has no effect here.
        @raise ValueError: There are no inhomogeneous wavepackets with a single component.
        """
        raise ValueError("There are no inhomogeneous wavepackets with a single component!")


    def calculate_local_remainder_multi(self):
        """Calculate the non-quadratic remainder $W$ of the quadratic
        approximation $U$ of the potential's eigenvalue $\lambda$.
        This function is used for the inhomogeneous case.
        @raise ValueError: There are no inhomogeneous wavepackets with a single component.
        """
        raise ValueError("There are no inhomogeneous wavepackets with a single component!")
