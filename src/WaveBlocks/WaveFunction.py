"""The WaveBlocks Project

This file contains code to numerically represent multiple
components of vector valued wave functions (states) together
with the grid nodes the values belong to. In addition there
are some methods for calculating data as for example :math:`L^2`
norms and the energies.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import zeros, complexfloating
from scipy import fft, sqrt, pi, dot, conj
from scipy import linalg as la


class WaveFunction:
    """This class represents a vector valued quantum state :math:`\Ket{\Psi}` as used
    in the vector valued time-dependent Schroedinger equation. The state
    :math:`\Ket{\Psi}` is composed of :math:`\psi_0, \ldots, \psi_{N-1}` where :math:`\psi_i` is
    a single wavefunction component."""
    
    def __init__(self, parameters):
        """Initialize the I{WaveFunction} object that represents the vector of states :math:`\Ket{\Psi}`.
        :param parameters: A I{ParameterProvider} instance with at least the items 'ncomponents', 'f' and 'ngn'.
        """
        self.number_components = parameters["ncomponents"]
        self.support = None
        self.values = None

        # Cache some parameter values
        self.f = parameters["f"]
        self.ngn = parameters["ngn"]


    def __str__(self):
        """
        :return: A string that describes the wavefunction :math:`\Ket{\Psi}`.
        """
        return "Wavefunction vector for " + str(self.number_components) + " states."


    def get_number_components(self):
        """
        :return: The number of components :math:`\psi_i` the vector :math:`\Ket{\Psi}` consists of.
        """
        return self.number_components


    def get_nodes(self):
        """
        :return: The grid nodes :math:`\gamma` the wave function values belong to.
        """
        return self.support


    def get_values(self):
        """Return the wave function values for each component of :math:`\Ket{\Psi}`.
        :return: A list with the values of all components :math:`\psi_i` evaluated on
        the grid nodes :math:`\gamma`.
        """
        return self.values[:]


    def set_grid(self, grid):
        """Assign a new grid to the wavefunction. All values are regarded to belong
        to these grid nodes.
        :param grid: The grid values as an numeric array.
        """
        self.support = grid[:]  


    def set_values(self, values, component=None):
        """Assign new function values for each component of :math:`\Ket{\Psi}`.
        :param values: A list with the new values of all the :math:`\psi_i`.
        @raise ValueError: If the list I{values} has the wrong number of entries.
        """
        if component is not None:
            if component >= self.number_components:
                raise ValueError("Missmatch in number of states.")

            if self.values is None:
                self.values = [ zeros(values.shape) ]
                
            self.values[component] = values
        else:
            if len(values) != self.number_components:
                raise ValueError("Missmatch in number of states.")
        
            self.values = values[:]


    def get_norm(self, values=None, summed=False, component=None):
        """Calculate the :math:`L^2` norm of the whole vector :math:`\Ket{\Psi}` or some
        individual components :math:`\psi_i`. The calculation is done in momentum space.
        :param values: Allows to use this function for external data, similar to a static function.
        :param summed: Whether to sum up the norms of the individual components.
        :param component: The component :math:`\psi_i` of which the norm is calculated.
        :return: The :math:`L^2` norm of :math:`\Ket{\Psi}` or a list of the :math:`L^2` norms of
        all components :math:`\psi_i`. (Depending on the optional arguments.)
        """
        if values is None:
            values = self.values
            
        if not component is None:
            result = sqrt(2.0*pi*self.f) * la.norm(fft(values[component]), ord=2)/self.ngn
        else:
            result = tuple([ sqrt(2.0*pi*self.f) * la.norm(item, ord=2)/self.ngn for item in [ fft(item) for item in values ] ])
            if summed is True:
                result = map(lambda x: x**2, result)
                result = sqrt(sum(result))
        
        return result


    def kinetic_energy(self, kinetic, summed=False):
        """Calculate the kinetic energy :math:`E_{\text{kin}} \assign \Braket{\Psi|T|\Psi}`
        of the different components.
        :param kinetic: The kinetic energy operator :math:`T`.
        :param summed: Whether to sum up the kinetic energies of the individual components.
        :return: A list with the kinetic energies of the individual components
        or the overall kinetic energy of the wavefunction. (Depending on the optional arguments.)
        """
        ekin = tuple([ (2.0*pi*self.f) * dot(conj(item),(kinetic*item)) / self.ngn**2  for item in [ fft(component) for component in self.values ] ])

        if summed is True:
            ekin = sum(ekin)
            
        return ekin


    def potential_energy(self, potential, summed=False):
        """Calculate the potential energy :math:`E_{\text{pot}} \assign \Braket{\Psi|V|\Psi}`
        of the different components.
        :param potential: The potential energy operator :math:`V`.
        :param summed: Whether to sum up the potential energies of the individual components.
        :return: A list with the potential energies of the individual components
        or the overall potential energy of the wavefunction. (Depending on the optional arguments.)
        """
        fvalues = [ fft(item) for item in self.values ]

        tmp = [ zeros(item.shape, dtype=complexfloating) for item in self.values ]
        for row in xrange(0, self.number_components):
            for col in xrange(0, self.number_components):
                tmp[row] = tmp[row] + potential[row*self.number_components+col] * self.values[col]
                
        epot = tuple([ (2.0*pi*self.f) * dot(conj(fitem),fft(item)) / self.ngn**2 for item, fitem in zip(tmp,fvalues) ])

        if summed is True:
            epot = sum(epot)
            
        return epot
