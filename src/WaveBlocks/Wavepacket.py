"""The WaveBlocks Project

This file contains the basic interface for general wavepackets.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import vstack, vsplit, cumsum, zeros, complexfloating


class Wavepacket:
    """This class is primarily an abstract interface to wavepackets in general.
    But it implemets some methods for both the homogeneous and inhomogeneous Hagedorn
    wavepackets.
    """

    def __init__(self, parameters):
        """Initialize the I{HagedornWavepacket} object that represents $\Ket{\Psi}$.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def __str__(self):
        """@return: A string describing the wavepacket.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def _resize_coefficient_vector(self, component):
        """Adapt the coefficient vector for a given component to a new size.
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


    def get_number_components(self):
        """@return: The number $N$ of components the wavepacket $\Psi$ has.
        """
        return self.number_components


    def get_basis_size(self, component=None):
        """@return: The size of the basis, i.e. the number $K$ of ${\phi_k}_{k=1}^K$.
        """
        if component is not None:
            return self.basis_size[component]
        else:
            return tuple(self.basis_size)


    def set_basis_size(self, basis_size, component=None):
        """Set the size of the basis of a given component or all components.
        @param basis_size: An single positive integer or a list of $N$ positive integers.
        @keyword component: The component for which we want to set the basis size.
        Default is I{None} which means 'all'.
        """
        if component is not None:
            # Check for valid input basis size
            if basis_size < 1:
                raise ValueError("Basis size has to be a positive integer.")

            # Set the new basis size for the given component
            self.basis_size[component] = basis_size
            # And adapt the coefficient vectors
            self._resize_coefficient_vector(component)

        else:
            # Check for valid input basis size
            if any([bs < 1 for bs in basis_size]):
                raise ValueError("Basis size has to be a positive integer.")

            if not len(basis_size) == self.number_components:
                raise ValueError("Number of value(s) for basis size(s) does not match.")

            # Set the new basis size for all components
            self.basis_size = [ bs for bs in basis_size ]
            # And adapt the coefficient vectors
            for index in xrange(self.number_components):
                self._resize_coefficient_vector(index)


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

                self.coefficients[index] = value.copy().reshape((self.basis_size[index],1))
        else:
            if component > self.number_components-1:
                raise ValueError("There is no component with index "+str(component)+".")

            self.coefficients[component] = values.copy().reshape((self.basis_size[component],1))


    def set_coefficient(self, component, index, value):
        """Set a single coefficient $c^i_k$ of the specified component $\Phi_i$ of $\Ket{\Psi}$.
        @param component: The index $i$ of the component $\Phi_i$ we want to update.
        @param index: The index $k$ of the coefficient $c^i_k$ we want to update.
        @param value: The new value of the coefficient $c^i_k$.
        @raise ValueError: For invalid indices $i$ or $k$.
        """
        if component > self.number_components-1:
            raise ValueError("There is no component with index "+str(component)+".")
        if index > self.basis_size[component]-1:
            raise ValueError("There is no basis function with index "+str(index)+".")

        self.coefficients[component][index] = value


    def get_coefficients(self, component=None):
        """Returns the coefficients $c^i$ for some components $\Phi_i$ of $\Ket{\Psi}$.
        @keyword component: The index $i$ of the coefficients $c^i$ we want to get.
        @return: The coefficients $c^i$ either for all components $\Phi_i$
        or for a specified one.
        """
        if component is None:
            return [ item.copy() for item in self.coefficients ]
        else:
            return self.coefficients[component].copy()


    def get_coefficient_vector(self):
        """@return: The coefficients $c^i$ of all components $\Phi_i$ as a single long column vector.
        """
        return vstack(self.coefficients)


    def set_coefficient_vector(self, vector):
        """Set the coefficients for all components $\Phi_i$ simultaneously.
        @param vector: The coefficients of all components as a single long column vector.
        @note: This function does *NOT* copy the input data! This is for efficiency as this
        routine is used in the innermost loops.
        """
        # Compute the partition of the block-vector from the basis sizes
        partition = cumsum(self.basis_size)[:-1]

        # Split the block-vector with the given partition and assign
        self.coefficients = vsplit(vector, partition)


    def get_parameters(self, component=None, aslist=False):
        """Get the Hagedorn parameters ${\Pi}$ of the wavepacket $\Psi$.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def set_parameters(self, parameters, component=None):
        """Set the Hagedorn parameters ${\Pi}$ of the wavepacket $\Psi$.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def evaluate_basis_at(self, nodes, component, prefactor=False):
        """Evaluate the basis functions $\phi_k$ recursively at the given nodes $\gamma$.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def evaluate_at(self, nodes, component=None, prefactor=False):
        """Evaluete the wavepacket $\Psi$ at the given nodes $\gamma$.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def get_norm(self, component=None, summed=False):
        """Calculate the $L^2$ norm of the wavepacket $\Ket{\Psi}$.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def potential_energy(self, potential, summed=False):
        """Calculate the potential energy $\Braket{\Psi|V|\Psi}$ of the wavepacket componentwise.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def kinetic_energy(self, summed=False):
        """Calculate the kinetic energy $\Braket{\Psi|T|\Psi}$ of the wavepacket componentwise.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def project_to_canonical(self, potential):
        """Project the wavepacket to the canonical basis.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def project_to_eigen(self, potential):
        """Project the wavepacket to the eigenbasis of a given potential $V$.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def to_fourier_space(self, assign=True):
        """Transform the wavepacket to Fourier space.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")


    def to_real_space(self, assign=True):
        """Transform the wavepacket to real space.
        @raise NotImplementedError: Abstract interface.
        """
        raise NotImplementedError("'Wavepacket' is an abstract interface.")
