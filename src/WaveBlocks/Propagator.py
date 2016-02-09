"""The WaveBlocks Project

This file contains the abstract base class for general time propagators.
It defines the interface every subclass must support to implement a
time propagation algorithm.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

class Propagator:
    r"""
    Propagators can numerically simulate the time evolution of quantum states
    as described by the time-dependent Schroedinger equation.
    """

    def __init__(self):
        r"""
        Initialize a new :py:class:`Propagator` instance.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'Propagator' is an abstract base class.")


    def __str__(self):
        r"""
        Prepare a printable string representing the ``Propagator`` instance.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("'Propagator' is an abstract base class.")


    def get_number_components(self):
        r"""
        :return: The number of components of :math:`|\Psi\rangle`.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("get_number_components(...)")


    def get_potential(self):
        r"""
        :return: The embedded ``MatrixPotential`` instance.
        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("get_potential(...)")


    def propagate(self):
        r"""
        Given the wavefunction :math:`\Psi` at time :math:`t`, calculate the new :math:`\Psi`
        at time :math:`t + \tau`. We do exactly one timestep :math:`\tau` here.

        :raise NotImplementedError: This is an abstract base class.
        """
        raise NotImplementedError("propagate(...)")
