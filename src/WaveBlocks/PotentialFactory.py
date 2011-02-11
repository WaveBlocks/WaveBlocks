"""The WaveBlocks Project

This file contains a simple factory for MatrixPotential instances. The exact
subtype of the instance is derived from the potentials' symbolic expression.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sympy


class PotentialFactory:
    """A factory for I{MatrixPotential} instances. We decide which subclass of the
    abstract base class I{MatrixPotential} to instantiate according to the size
    of the potential's matrix. For a $1 \times 1$ matrix we can use the class I{MatrixPotential1S}
    which implements simplified scalar symbolic calculations. In the case of a $2 \times 2$
    matrix we use the class I{MatrixPotential2S} that implements the full symbolic
    calculations for matrices. And for matrices of size bigger than $2 \times 2$ symbolic
    calculations are unfeasible and we have to fall back to pure numerical methods
    implemented in I{MatrixPotentialMS}.
    """

    @staticmethod
    def create_potential(parameters):
        """Static method that creates a I{MatrixPotential} instance and decides
        which subclass to instantiate depending on the given potential expression.
        @param parameters: A I{ParameterProvider} instance with all necessary parameters (at least a 'potential' entry).
        @return: An adequate I{MatrixPotential} instance.
        @raise ValueError: In case of various input error, f.e. if the potential
        can not be found or if the potential matrix is not square etc.
        """
        # The potential reference given in the parameter provider.
        # This may be a string which is the common name of the potential
        # or a full potential description. In the first case we try
        # to find the referenced potential in the potential library
        # while in the second one, we can omit this step.
        potential_reference = parameters["potential"]

        if type(potential_reference) == str:
            # Try to load the potential from the library
            import PotentialLibrary as PL
            if PL.__dict__.has_key(potential_reference):
                potential_description = PL.__dict__[potential_reference]
            else:
                raise ValueError("Unknown potential " + potential_reference + " requested from library.")
        elif type(potential_reference) == dict:
            # The potential reference given in the parameter provider was a full description
            potential_description = potential_reference
        else:
            raise ValueError("Invalid potential reference.")
        
        # The symbolic expression strings of the potential
        pot = potential_description["potential"]

        # Potential is just one level, wrap it into a matrix
        if type(pot) == str:
            pot = [[pot]]
        
        # Sympify the expression strings for each entry of the potential matrix
        potmatrix = [ [ sympy.sympify(item) for item in row ] for row in pot ]

        # Get the default parameters, if any
        if potential_description.has_key("defaults"):
            default_params = potential_description["defaults"]
        else:
            default_params = {}

        # Build the potential matrix with known values substituted for symbolic constants
        final_matrix = []
        free_symbols = []

        for row in potmatrix:
            cur_row = []
            for item in row:
                # Get atoms, but only symbols
                symbols = item.atoms(sympy.Symbol)
                values = {}

                # Search symbols and find a value for each one
                for atom in symbols:
                    if parameters.has_key(atom.name):
                        # A value is given by the parameter provider
                        val = parameters[atom.name]
                    elif default_params.has_key(atom.name):
                        # We do have a default value
                        val = default_params[atom.name]
                    else:
                        # No default value found either! This could be an input
                        # error in the potential definition but we rather interpret this
                        # case as a free parameter, for example the first space coordinate x
                        free_symbols.append(atom)
                        continue

                    # Sympify in case the values was specified as a string
                    values[atom.name] = sympy.sympify(val)

                # Substitute the values for the symbols
                for key, value in values.iteritems():
                    # Remember expressions are immutable
                    item = item.subs(key, value)

                # Simplify and insert into the final potential matrix
                cur_row.append(sympy.simplify(item))
            final_matrix.append(cur_row)

        # Create a real sympy matrix instance
        potential_matrix = sympy.Matrix(final_matrix)
        # The list with all free symbols
        free_symbols = tuple(set(free_symbols))

        if not potential_matrix.is_square:
            raise ValueError("Potential matrix is not square!")

        size = potential_matrix.shape[0]

        # Create instances of MatrixPotential*
        if size == 1:
            from MatrixPotential1S import MatrixPotential1S
            potential = MatrixPotential1S(potential_matrix, free_symbols)
        elif size == 2:
            from MatrixPotential2S import MatrixPotential2S
            potential = MatrixPotential2S(potential_matrix, free_symbols)
        else:
            from MatrixPotentialMS import MatrixPotentialMS
            potential = MatrixPotentialMS(potential_matrix, free_symbols)
        
        return potential
