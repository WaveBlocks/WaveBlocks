"""The WaveBlocks Project

This file contains the Fourier propagator class.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import arange, append, exp, zeros, complexfloating
from scipy import fft, ifft

from Propagator import Propagator


class FourierPropagator(Propagator):
    """This class can numerically propagate given initial values $\Ket{\Psi}$ in
    a potential surface $V\ofs{x}$. The propagation is done with a Strang splitting
    of the time propagation operator."""

    def __init__(self, potential, initial_values, para):
        """Initialize a new I{FourierPropagator} instance. Precalculate also the
        grid and the propagation operators.
        @param potential: The potential the state $\Ket{\Psi}$ feels during the time propagation.
        @param initial_values: The initial values $\Ket{\Psi\ofs{t=0}}$ given in the canonical basis.
        @raise ValueError: If the number of components of $\Ket{\Psi}$ does not
        match the number of energy levels $\lambda_i$ of the potential.
        """
        #: The embedded I{MatrixPotential} instance representing the potential $V$.
        self.potential = potential

        #: The initial values of the components $\psi_i$ sampled at the given nodes.
        self.Psi = initial_values

        if self.potential.get_number_components() != self.Psi.get_number_components():
            raise ValueError("Potential dimension and number of states do not match")

        #: The position space nodes $\gamma$.
        self.nodes = initial_values.get_nodes()

        #: The potential operator $V$ defined in position space.
        self.V = self.potential.evaluate_at(self.nodes)

        #: The momentum space nodes $\omega$.
        self.omega = arange(0, para.ngn/2.0)
        self.omega = append(self.omega, arange(para.ngn/2.0, 0, -1))
        
        #: The kinetic operator $T$ defined in momentum space.
        self.T = 0.5 * para.eps**4 * self.omega**2 / para.f**2
        
        #: Exponential $\exp\ofs{T}$ of $T$ used in the Strang splitting.
        self.TE = exp(-0.5j * para.dt * para.eps**2 * self.omega**2 / para.f**2)

        self.potential.calculate_exponential(-0.5j * para.dt / para.eps**2)
        #: Exponential $\exp\ofs{V}$ of $V$ used in the Strang splitting.
        self.VE = self.potential.evaluate_exponential_at(self.nodes)


    def __str__(self):
        """Prepare a printable string representing the I{FourierPropagator} instance."""
        return "Fourier propagator for " + str(self.potential.get_number_components()) + " components."


    def get_number_components(self):
        """@return: The number of components of $\Ket{\Psi}$."""
        return self.potential.get_number_components()
        

    def get_potential(self):
        """@return: The I{MatrixPotential} instance used for time propagation."""
        return self.potential


    def get_wavefunction(self):
        """@return: The I{WaveFunction} instance that stores the current wavefunction data."""
        return self.Psi


    def get_operators(self):
        """@return: Return the numerical expressions of the propagation
        operators $T$ and $V$.
        """
        return (self.T, self.V)


    def propagate(self):
        """Given the wavefunction values $\Psi$ at time $t$, calculate new
        values at time $t + \tau$. We perform exactly one timestep $\tau$ here.
        """
        # How many states we have
        nst = self.Psi.get_number_components()
        
        # Read values out of current WaveFunction state
        vals = self.Psi.get_values()
        
        # Do the propagation
        tmp = [ zeros(vals[0].shape, dtype=complexfloating) for item in vals ]
        for row in xrange(0, nst):
            for col in xrange(0, nst):
                tmp[row] = tmp[row] + self.VE[row*nst+col] * vals[col]
                
        tmp = tuple([ fft(item) for item in tmp ])
    
        tmp = tuple([ self.TE * item for item in tmp ])
        
        tmp = tuple([ ifft(item) for item in tmp ])
        
        values = [ zeros(tmp[0].shape, dtype=complexfloating) for item in tmp ]
        for row in xrange(0, nst):
            for col in xrange(0, nst):
                values[row] = values[row] + self.VE[row*nst+col] * tmp[col]
                
        # Write values back to WaveFunction object
        self.Psi.set_values(values)


    def kinetic_energy(self, summed=False):
        """This method just delegates the calculation of kinetic energies to the
        embedded I{WaveFunction} object.
        @keyword summed: Whether to sum up the kinetic energies of the individual components.
        @return: The kinetic energies.
        """
        return self.Psi.kinetic_energy(self.T, summed=summed)


    def potential_energy(self, summed=False):
        """This method just delegates the calculation of potential energies to the
        embedded I{WaveFunction} object.
        @keyword summed: Whether to sum up the potential energies of the individual components.
        @return: The potential energies. 
        """
        return self.Psi.potential_energy(self.V, summed=summed)
