"""The WaveBlocks Project

This file contains the basic interface for general wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import vstack, vsplit, cumsum, zeros, complexfloating


class Wavepacket:
    r"""
    This class is primarily an abstract interface to wavepackets in general.
    But it implemets some methods for both the homogeneous and inhomogeneous Hagedorn
    wavepackets.
    """

    def __init__(self, parameters):
        r"""
        Initialize the ``Wavepacket`` object that represents :math:`|\Psi\rangle`.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def __str__(self):
        r"""
        :return: A string describing the wavepacket.
        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def _resize_coefficient_vector(self, component):
        r"""
        Adapt the coefficient vector for a given component to a new size.
        """
        oldsize = self.coefficients[component].shape[0]
        newsize = self.basis_size[component]

        if oldsize == newsize:
            return
        elif oldsize < newsize:
            # Append some zeros
            z = zeros((newsize - oldsize, 1), dtype=complexfloating)
            self.coefficients[component] = vstack([self.coefficients[component], z])
        elif oldsize > newsize:
            # Cut off the last part
            self.coefficients[component] = self.coefficients[component][0:newsize]


    def clone(self):
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def gen_id(self):
        r"""
        Generate an (unique) ID per wavepacket instance.
        """
        #TODO: Better id generating function!
        self._id = id(self)


    def get_id(self):
        r"""
        Return the packet ID of this wavepacket instance. The ID may be used for storing packets in associative lists.
        """
        if not hasattr(self, "_id"):
            self.gen_id()

        return self._id


    def set_id(self, anid):
        r"""
        Manually set an ID for the current wavepacket instance.
        """
        assert(type(anid) is int)
        self._id = anid


    def get_number_components(self):
        r"""
        :return: The number :math:`N` of components the wavepacket :math:`\Psi` has.
        """
        return self.number_components


    def get_basis_size(self, component=None):
        r"""
        :return: The size of the basis, i.e. the number :math:`K` of :math:`{\phi_k}_{k=1}^K`.
        """
        if component is not None:
            return self.basis_size[component]
        else:
            return tuple(self.basis_size)


    def set_basis_size(self, basis_size, component=None):
        r"""
        Set the size of the basis of a given component or all components.

        :param basis_size: An single positive integer or a list of :math:`N` positive integers.
        :param component: The component for which we want to set the basis size.
                          Default is ``None`` which means 'all'.
        """
        if component is not None:
            # Check for valid input basis size
            if not component in range(self.number_components):
                raise ValueError("Invalid component index " + str(component))

            if basis_size < 2:
                raise ValueError("Basis size has to be a positive integer >=2.")

            # Set the new basis size for the given component
            self.basis_size[component] = basis_size
            # And adapt the coefficient vectors
            self._resize_coefficient_vector(component)

        else:
            # Check for valid input basis size
            if any([bs < 2 for bs in basis_size]):
                raise ValueError("Basis size has to be a positive integer >=2.")

            if not len(basis_size) == self.number_components:
                raise ValueError("Number of value(s) for basis size(s) does not match.")

            # Set the new basis size for all components
            self.basis_size = [ bs for bs in basis_size ]
            # And adapt the coefficient vectors
            for index in xrange(self.number_components):
                self._resize_coefficient_vector(index)


    def set_coefficients(self, values, component=None):
        r"""
        Update the coefficients :math:`c` of :math:`\Psi`.

        :param values: The new values of the coefficients :math:`c^i` of :math:`\Phi_i`.
        :param component: The index :math:`i` of the component we want to update with new coefficients.
        :raise ValueError: For invalid indices :math:`i`.

        .. note:: This function can either set new coefficients for a single component :math:`\Phi_i`
                  only if the ``component`` attribute is set or for all components simultaneously if
                  ``values`` is a list of arrays.
        """
        if component is None:
            for index, value in enumerate(values):
                if index > self.number_components-1:
                    raise ValueError("There is no component with index "+str(index)+".")

                self.coefficients[index] = value.copy().reshape((self.basis_size[index],1))
        else:
            if component > self.number_components-1:
                raise ValueError("There is no component with index "+str(component)+".")

            self.coefficients[component] = values.copy().reshape((self.basis_size[component],1))


    def set_coefficient(self, component, index, value):
        r"""
        Set a single coefficient :math:`c^i_k` of the specified component :math:`\Phi_i` of :math:`|\Psi\rangle`.

        :param component: The index :math:`i` of the component :math:`\Phi_i` we want to update.
        :param index: The index :math:`k` of the coefficient :math:`c^i_k` we want to update.
        :param value: The new value of the coefficient :math:`c^i_k`.
        :raise ValueError: For invalid indices :math:`i` or :math:`k`.
        """
        if component > self.number_components-1:
            raise ValueError("There is no component with index "+str(component)+".")
        if index > self.basis_size[component]-1:
            raise ValueError("There is no basis function with index "+str(index)+".")

        self.coefficients[component][index] = value


    def get_coefficients(self, component=None):
        r"""
        Returns the coefficients :math:`c^i` for some components :math:`\Phi_i` of :math:`|\Psi\rangle`.

        :param component: The index :math:`i` of the coefficients :math:`c^i` we want to get.
        :return: The coefficients :math:`c^i` either for all components :math:`\Phi_i`
                 or for a specified one.
        """
        if component is None:
            return [ item.copy() for item in self.coefficients ]
        else:
            return self.coefficients[component].copy()


    def get_coefficient_vector(self):
        r"""
        :return: The coefficients :math:`c^i` of all components :math:`\Phi_i` as a single long column vector.
        """
        return vstack(self.coefficients)


    def set_coefficient_vector(self, vector):
        r"""
        Set the coefficients for all components :math:`\Phi_i` simultaneously.

        :param vector: The coefficients of all components as a single long column vector.

        .. note:: This function does *NOT* copy the input data! This is for efficiency as this
                  routine is used in the innermost loops.
        """
        # Compute the partition of the block-vector from the basis sizes
        partition = cumsum(self.basis_size)[:-1]

        # Split the block-vector with the given partition and assign
        self.coefficients = vsplit(vector, partition)


    def get_parameters(self, component=None, aslist=False):
        r"""
        Get the Hagedorn parameters :math:`{\Pi}` of the wavepacket :math:`\Psi`.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def set_parameters(self, parameters, component=None):
        r"""
        Set the Hagedorn parameters :math:`{\Pi}` of the wavepacket :math:`\Psi`.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def evaluate_basis_at(self, nodes, component, prefactor=False):
        r"""
        Evaluate the basis functions :math:`\phi_k` recursively at the given nodes :math:`\gamma`.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def evaluate_at(self, nodes, component=None, prefactor=False):
        r"""
        Evaluete the wavepacket :math:`\Psi` at the given nodes :math:`\gamma`.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def get_norm(self, component=None, summed=False):
        r"""
        Calculate the :math:`L^2` norm of the wavepacket :math:`|\Psi\rangle`.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def potential_energy(self, potential, summed=False):
        r"""
        Calculate the potential energy :math:`\langle\Psi|V|\Psi\rangle ` of the wavepacket componentwise.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def kinetic_energy(self, summed=False):
        r"""
        Calculate the kinetic energy :math:`\langle\Psi|T|\Psi\rangle ` of the wavepacket componentwise.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def project_to_canonical(self, potential):
        r"""
        Project the wavepacket to the canonical basis.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def project_to_eigen(self, potential):
        r"""
        Project the wavepacket to the eigenbasis of a given potential :math:`V`.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def to_fourier_space(self, assign=True):
        r"""
        Transform the wavepacket to Fourier space.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def to_real_space(self, assign=True):
        r"""
        Transform the wavepacket to real space.

        :raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")
