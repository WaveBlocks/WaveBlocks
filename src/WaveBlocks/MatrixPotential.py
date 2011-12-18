"""The WaveBlocks Project

This file contains the abstract base class for representation of potentials
for an arbitrary number of components. It defines the interface every subclass
must support to represent a potential.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

class MatrixPotential:
    """This class represents a potential $V\ofs{x}$. The potential is given as an analytic
    expression. Some calculations with the potential are supported. For example
    calculation of eigenvalues and exponentials and numerical evaluation.
    Further, there are methods for splitting the potential into a Taylor
    expansion and for basis transformations between canonical and eigenbasis.
    """

    def __init__(self):
        """Create a new I{MatrixPotential} instance for a given potential matrix $V\ofs{x}$.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'MatrixPotential' is an abstract base class.")


    def __str__(self):
        """Put the number of components and the analytical expression (the matrix)
        into a printable string.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'MatrixPotential' is an abstract base class.")


    def get_number_components(self):
        """@return: The number $N$ of components the potential supports.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("get_number_components(...)")


    def evaluate_at(self, nodes, component=None):
        """Evaluate the potential matrix elementwise at some given grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the potential at.
        @keyword component: The component $V_{i,j}$ that gets
        evaluated or 'None' to evaluate all.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_at(...)")


    def calculate_eigenvalues(self):
        """Calculate the eigenvalues $\lambda_i\ofs{x}$ of the potential $V\ofs{x}$.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_eigenvalues(...)")


    def evaluate_eigenvalues_at(self, nodes, diagonal_component=None):
        """Evaluate the eigenvalues $\lambda_i\ofs{x}$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the eigenvalues at.
        @keyword diagonal_component: The index $i$ of the eigenvalue $\lambda_i$
        that gets evaluated or 'None' to evaluate all.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_eigenvalues_at(...)")


    def calculate_eigenvectors(self):
        """Calculate the eigenvectors $\nu_i\ofs{x}$ of the potential $V\ofs{x}$.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_eigenvectors(...)")


    def evaluate_eigenvectors_at(self, nodes):
        """Evaluate the eigenvectors $\nu_i\ofs{x}$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the eigenvectors at.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_eigenvectors_at(...)")


    def project_to_eigen(self, nodes, values, basis=None):
        """Project a given vector from the canonical basis to the eigenbasis of the potential.
        @param nodes: The grid nodes $\gamma$ for the pointwise transformation.
        @param values: The list of vectors $\varphi_i$ containing the values we want to transform.
        @keyword basis: A list of basisvectors $nu_i$. Allows to use this function for external data, similar to a static function.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("project_to_eigen(...)")


    def project_to_canonical(self, nodes, values, basis=None):
        """Project a given vector from the potential's eigenbasis to the canonical basis.
        @param nodes: The grid nodes $\gamma$ for the pointwise transformation.
        @param values: The list of vectors $\varphi_i$ containing the values we want to transform.
        @keyword basis: A list of basis vectors $nu_i$. Allows to use this function for external data, similar to a static function.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("project_to_canonical(...)")


    def calculate_exponential(self, factor=1):
        """Calculate the matrix exponential $E = \exp\ofs{\alpha M}$.
        @keyword factor: A prefactor $\alpha$ in the exponential.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_exponential(...)")


    def evaluate_exponential_at(self, nodes):
        """Evaluate the exponential of the potential matrix $V$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the exponential at.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_exponential_at(...)")


    def calculate_jacobian(self):
        """Calculate the jacobian matrix for each component $V_{i,j}$ of the potential.
        For potentials which depend only one variable $x$, this equals the first derivative.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_jacobian(...)")


    def evaluate_jacobian_at(self, nodes, component=None):
        """Evaluate the jacobian at some grid nodes $\gamma$ for each component
        $V_{i,j}$ of the potential.
        @param nodes: The grid nodes $\gamma$ the jacobian gets evaluated at.
        @keyword component: The index tuple $\left(i,j\right)$ that specifies
        the potential's entry of which the jacobian is evaluated. (Defaults to 'None' to evaluate all)
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_jacobian_at(...)")


    def calculate_hessian(self):
        """Calculate the hessian matrix for each component $V_{i,j}$ of the potential.
        For potentials which depend only one variable $x$, this equals the second derivative.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_hessian(...)")


    def evaluate_hessian_at(self, nodes, component=None):
        """Evaluate the hessian at some grid nodes $\gamma$ for each component
        $V_{i,j}$ of the potential.
        @param nodes: The grid nodes $\gamma$ the hessian gets evaluated at.
        @keyword component: The index tuple $\left(i,j\right)$ that specifies
        the potential's entry of which the hessian is evaluated. (Or 'None' to evaluate all)
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_hessian_at(...)")


    def calculate_local_quadratic(self, diagonal_component=None):
        """Calculate the local quadratic approximation matrix $U$ of the potential's
        eigenvalues in $\Lambda$. This function is used for the homogeneous case and
        takes into account the leading component $\chi$.
        @keyword diagonal_component: Specifies the index $i$ of the eigenvalue $\lambda_i$
        that gets expanded into a Taylor series $u_i$.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_local_quadratic(...)")


    def evaluate_local_quadratic_at(self, nodes):
        """Numerically evaluate the local quadratic approximation matrix $U$ of
        the potential's eigenvalues in $\Lambda$ at the given grid nodes $\gamma$.
        This function is used for the homogeneous case and takes into account the
        leading component $\chi$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the quadratic approximation at.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_local_quadratic_at(...)")


    def calculate_local_remainder(self, diagonal_component=0):
        """Calculate the non-quadratic remainder matrix $W$ of the quadratic
        approximation matrix $U$ of the potential's eigenvalue matrix $\Lambda$.
        This function is used for the homogeneous case and takes into account the
        leading component $\chi$.
        @param diagonal_component: Specifies the index $\chi$ of the leading component $\lambda_\chi$.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_local_remainder(...)")


    def evaluate_local_remainder_at(self, position, nodes, component=None):
        """Numerically evaluate the non-quadratic remainder matrix $W$ of the quadratic
        approximation matrix $U$ of the potential's eigenvalues in $\Lambda$ at the
        given nodes $\gamma$. This function is used for the homogeneous and the
        inhomogeneous case and just evaluates the remainder matrix $W$.
        @param position: The point $q$ where the Taylor series is computed.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the potential at.
        @keyword component: The component $\left(i,j\right)$ of the remainder matrix $W$
        that is evaluated.
        @raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_local_remainder_at(...)")
