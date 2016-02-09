"""The WaveBlocks Project

This file contains the abstract base class for representation of potentials
for an arbitrary number of components. It defines the interface every subclass
must support to represent a potential.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

class MatrixPotential:
    r"""
    This class represents a potential :math:`V\left(x\right)`. The potential is given as an analytic
    expression. Some calculations with the potential are supported. For example
    calculation of eigenvalues and exponentials and numerical evaluation.
    Further, there are methods for splitting the potential into a Taylor
    expansion and for basis transformations between canonical and eigenbasis.
    """

    def __init__(self):
        r"""
        Create a new ``MatrixPotential`` instance for a given potential matrix :math:`V\left(x\right)`.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'MatrixPotential' is an abstract base class.")


    def __str__(self):
        r"""
        Put the number of components and the analytical expression (the matrix) into a printable string.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'MatrixPotential' is an abstract base class.")


    def get_number_components(self):
        r"""
        :return: The number :math:`N` of components the potential supports.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("get_number_components(...)")


    def evaluate_at(self, nodes, component=None):
        r"""
        Evaluate the potential matrix elementwise at some given grid nodes :math:`\gamma`.

        :param nodes: The grid nodes :math:`\gamma` we want to evaluate the potential at.
        :param component: The component :math:`V_{i,j}` that gets evaluated or 'None' to evaluate all.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_at(...)")


    def calculate_eigenvalues(self):
        r"""
        Calculate the eigenvalues :math:`\lambda_i\left(x\right)` of the potential :math:`V\left(x\right)`.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_eigenvalues(...)")


    def evaluate_eigenvalues_at(self, nodes, diagonal_component=None):
        r"""
        Evaluate the eigenvalues :math:`\lambda_i\left(x\right)` at some grid nodes :math:`\gamma`.

        :param nodes: The grid nodes :math:`\gamma` we want to evaluate the eigenvalues at.
        :param diagonal_component: The index :math:`i` of the eigenvalue :math:`\lambda_i` that gets evaluated or 'None' to evaluate all.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_eigenvalues_at(...)")


    def calculate_eigenvectors(self):
        r"""
        Calculate the eigenvectors :math:`\nu_i\left(x\right)` of the potential :math:`V\left(x\right)`.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_eigenvectors(...)")


    def evaluate_eigenvectors_at(self, nodes):
        r"""
        Evaluate the eigenvectors :math:`\nu_i\left(x\right)` at some grid nodes :math:`\gamma`.

        :param nodes: The grid nodes :math:`\gamma` we want to evaluate the eigenvectors at.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_eigenvectors_at(...)")


    def project_to_eigen(self, nodes, values, basis=None):
        r"""
        Project a given vector from the canonical basis to the eigenbasis of the potential.

        :param nodes: The grid nodes :math:`\gamma` for the pointwise transformation.
        :param values: The list of vectors :math:`\varphi_i` containing the values we want to transform.
        :param basis: A list of basisvectors :math:`nu_i`. Allows to use this function for external data, similar to a static function.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("project_to_eigen(...)")


    def project_to_canonical(self, nodes, values, basis=None):
        r"""
        Project a given vector from the potential's eigenbasis to the canonical basis.

        :param nodes: The grid nodes :math:`\gamma` for the pointwise transformation.
        :param values: The list of vectors :math:`\varphi_i` containing the values we want to transform.
        :param basis: A list of basis vectors :math:`nu_i`. Allows to use this function for external data, similar to a static function.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("project_to_canonical(...)")


    def calculate_exponential(self, factor=1):
        r"""
        Calculate the matrix exponential :math:`E = \exp\left(\alpha M\right)`.

        :param factor: A prefactor :math:`\alpha` in the exponential.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_exponential(...)")


    def evaluate_exponential_at(self, nodes):
        r"""
        Evaluate the exponential of the potential matrix :math:`V` at some grid nodes :math:`\gamma`.

        :param nodes: The grid nodes :math:`\gamma` we want to evaluate the exponential at.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_exponential_at(...)")


    def calculate_jacobian(self):
        r"""
        Calculate the jacobian matrix for each component :math:`V_{i,j}` of the potential.
        For potentials which depend only one variable :math:`x`, this equals the first derivative.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_jacobian(...)")


    def evaluate_jacobian_at(self, nodes, component=None):
        r"""
        Evaluate the jacobian at some grid nodes :math:`\gamma` for each component :math:`V_{i,j}` of the potential.

        :param nodes: The grid nodes :math:`\gamma` the jacobian gets evaluated at.
        :param component: The index tuple :math:`\left(i,j\right)` that specifies the potential's entry of which the jacobian is evaluated. (Defaults to 'None' to evaluate all)
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_jacobian_at(...)")


    def calculate_hessian(self):
        r"""
        Calculate the hessian matrix for each component :math:`V_{i,j}` of the potential.
        For potentials which depend only one variable :math:`x`, this equals the second derivative.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_hessian(...)")


    def evaluate_hessian_at(self, nodes, component=None):
        r"""
        Evaluate the hessian at some grid nodes :math:`\gamma` for each component :math:`V_{i,j}` of the potential.

        :param nodes: The grid nodes :math:`\gamma` the hessian gets evaluated at.
        :param component: The index tuple :math:`\left(i,j\right)` that specifies the potential's entry of which the hessian is evaluated. (Or 'None' to evaluate all)
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_hessian_at(...)")


    def calculate_local_quadratic(self, diagonal_component=None):
        r"""
        Calculate the local quadratic approximation matrix :math:`U` of the potential's
        eigenvalues in :math:`\Lambda`. This function is used for the homogeneous case and
        takes into account the leading component :math:`\chi`.

        :param diagonal_component: Specifies the index :math:`i` of the eigenvalue :math:`\lambda_i` that gets expanded into a Taylor series :math:`u_i`.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_local_quadratic(...)")


    def evaluate_local_quadratic_at(self, nodes):
        r"""
        Numerically evaluate the local quadratic approximation matrix :math:`U` of
        the potential's eigenvalues in :math:`\Lambda` at the given grid nodes :math:`\gamma`.
        This function is used for the homogeneous case and takes into account the leading component :math:`\chi`.

        :param nodes: The grid nodes :math:`\gamma` we want to evaluate the quadratic approximation at.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_local_quadratic_at(...)")


    def calculate_local_remainder(self, diagonal_component=0):
        r"""
        Calculate the non-quadratic remainder matrix :math:`W` of the quadratic
        approximation matrix :math:`U` of the potential's eigenvalue matrix :math:`\Lambda`.
        This function is used for the homogeneous case and takes into account the leading component :math:`\chi`.

        :param diagonal_component: Specifies the index :math:`\chi` of the leading component :math:`\lambda_\chi`.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("calculate_local_remainder(...)")


    def evaluate_local_remainder_at(self, position, nodes, component=None):
        r"""
        Numerically evaluate the non-quadratic remainder matrix :math:`W` of the quadratic
        approximation matrix :math:`U` of the potential's eigenvalues in :math:`\Lambda` at the
        given nodes :math:`\gamma`. This function is used for the homogeneous and the
        inhomogeneous case and just evaluates the remainder matrix :math:`W`.

        :param position: The point :math:`q` where the Taylor series is computed.
        :param nodes: The grid nodes :math:`\gamma` we want to evaluate the potential at.
        :param component: The component :math:`\left(i,j\right)` of the remainder matrix :math:`W` that is evaluated.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("evaluate_local_remainder_at(...)")
