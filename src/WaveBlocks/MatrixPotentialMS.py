"""The WaveBlocks Project

This file contains code for the representation of potentials for three and more components.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from functools import partial
import sympy
import numpy
from scipy import linalg
import numdifftools as ndt

from MatrixPotential import MatrixPotential


class MatrixPotentialMS(MatrixPotential):
    """This class represents a matrix potential $V\ofs{x}$. The potential is given as an analytical
    expression with a matrix of size bigger than $2 \times 2$. Some calculations
    with the potential are supported. For example calculation of eigenvalues and
    exponentials and numerical evaluation. Further, there are methods for
    splitting the potential into a Taylor expansion and for basis transformations
    between canonical and eigenbasis. All methods use numerical techniques because
    symbolical calculations are unfeasible.
    """

    def __init__(self, expression, variables):
        """Create a new I{MatrixPotentialMS} instance for a given potential matrix $V\ofs{x}$.
        @param expression: An expression representing the potential.
        """
        #: The variable $x$ that represents position space.
        self.x = variables[0]
        #: The matrix of the potential $V\ofs{x}$.
        self.potential = expression

        self.number_components = self.potential.shape[0]

        # prepare the function in every potential matrix cell for numerical evaluation
        self.functions = tuple([ sympy.vectorize(0)(sympy.lambdify(self.x, item, "numpy")) for item in self.potential ])

        # Some flags used for memoization
        self.__valid_eigenvalues = False
        self.__valid_eigenvectors = False
        

    def __str__(self):
        """Put the number of components and the analytical expression (the matrix) into a printable string.
        """
        return """Matrix potential with """ + str(self.number_components) + """ states given by matrix:\n""" + str(self.potential)


    def get_number_components(self):
        """@return: The number $N$ of components the potential supports. This is
        also the size of the matrix.
        """
        return self.number_components
    

    def evaluate_at(self, nodes, component=None, as_matrix=True):
        """Evaluate the potential matrix elementwise at some given grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the potential at.
        @keyword component: The component $V_{i,j}$ that gets evaluated or 'None' to evaluate all.
        @keyword as_matrix: Returns the whole matrix $\Lambda$ instead of only a list with the eigenvalues $\lambda_i$.
        @return: A list with the $N^2$ entries evaluated at the nodes.
        """
        result = tuple([ numpy.array(f(nodes), dtype=numpy.complexfloating) for f in self.functions ])
        
        if not component is None:
            result = result[component * self.number_components + component]
        
        return result


    def calculate_eigenvalues(self):
        """Calculate the eigenvalues $\lambda_i\ofs{x}$ of the potential $V\ofs{x}$.
        We do the calculations with numerical tools. The multiplicities are taken
        into account.
        @note: Note: the eigenvalues are memoized for later reuse.
        """
        if self.__valid_eigenvalues == True:
            return
 
        # We have to use numercial techniques here, the eigenvalues are
        # calculated while evaluating them in 'evaluate_eigenvalues_at'.
        pass
        
        # Ok, now we have valid eigenvalues at hand            
        self.__valid_eigenvalues = True


    def evaluate_eigenvalues_at(self, nodes, component=None, as_matrix=False):
        """Evaluate the eigenvalues $\lambda_i\ofs{x}$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the eigenvalues at.
        @keyword component: The index $i$ of the eigenvalue $\lambda_i$ that gets evaluated.
        @keyword as_matrix: Returns the whole matrix $\Lambda$ instead of only a list with the eigenvalues $\lambda_i$.
        @return: A sorted list with $N$ entries for all the eigenvalues evaluated at the nodes. Or a
        single value if a component was specified.
        """
        result = []
        
        # Hack to see if we evaluate at a single value
        try:
            n = len(nodes)
        except TypeError:
            n = 1
        tmppot = numpy.ndarray((n,self.number_components,self.number_components), dtype=numpy.complexfloating)
        tmpew = numpy.ndarray((n,self.number_components), dtype=numpy.complexfloating)
        
        # evaluate potential
        values = self.evaluate_at(nodes)
        
        # fill in values
        for row in xrange(0, self.number_components):
            for col in xrange(0, self.number_components):
                tmppot[:, row, col] = values[row*self.number_components + col]

        # calculate eigenvalues assuming hermitian matrix (eigvalsh for stability!)
        for i in xrange(0, n):
            ew = linalg.eigvalsh(tmppot[i,:,:])
            # Sorting the eigenvalues biggest first.
            ew.sort()
            tmpew[i,:] = ew[::-1]

        result = [ tmpew[:,index] for index in xrange(0, self.number_components) ]
        
        # Hack to undo vectorization
        if n == 1:
            result = [ item[0] for item in result ]
            
        if not component is None:
            result = result[component]
            
        return result


    def calculate_eigenvectors(self):
        """Calculate the two eigenvectors $nu_i\ofs{x}$ of the potential $V\ofs{x}$.
        We do the calculations with numerical tools.
        @note: The eigenvectors are memoized for later reuse.
        """
        if self.__valid_eigenvectors == True:
            return
            
        self.eigenvectors_s = []
        # We have to use numercial techniques here, the eigenvectors are
        # calculated while evaluating them in 'evaluate_eigenvects_at'.
            
        # The numerical expressions for the eigenvectors
        self.eigenvectors_n = []
            
        for vector in self.eigenvectors_s:
            self.eigenvectors_n.append( tuple([ sympy.vectorize(0)(sympy.lambdify(self.x, component, "numpy")) for component in vector ]) )
        
        # Ok, now we have valid eigenvectors at hand
        self.__valid_eigenvectors = True


    def evaluate_eigenvectors_at(self, nodes):
        """Evaluate the eigenvectors $nu_i\ofs{x}$ at some grid nodes $\gamma$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the eigenvectors at.
        @return: A list with the $N$ eigenvectors evaluated at the given nodes.
        """
        result = []

        n = len(nodes)
        tmppot = numpy.ndarray((n,self.number_components,self.number_components), dtype=numpy.complexfloating)
        tmpev = numpy.ndarray((n,self.number_components,self.number_components), dtype=numpy.complexfloating)
        
        # evaluate potential
        values = self.evaluate_at(nodes)
        
        # fill in values
        for row in xrange(0, self.number_components):
            for col in xrange(0, self.number_components):
                tmppot[:, row, col] = values[row*self.number_components + col]

        # calculate eigenvalues assuming hermitian matrix (eigvalsh for stability!)
        for i in xrange(0, n):
            ew, ev = linalg.eigh(tmppot[i,:,:])
            # Sorting the eigenvectors in the same order as the eigenvalues.
            ind = numpy.argsort(ew)
            ind = ind[::-1]
            evs = ev[:,ind]
            tmpev[i,:,:] = evs
            
        # A trick due to G. Hagedorn to get continuous eigenvectors
        for i in xrange(1, n):
            for ev in xrange(0,self.number_components):
                if numpy.dot(tmpev[i,:,ev],tmpev[i-1,:,ev]) < 0:
                    tmpev[i,:,ev] *= -1

        result = tuple([ numpy.transpose(tmpev[:,:,index]) for index in xrange(0, self.number_components) ])
        return result
        

    def project_to_eigen(self, nodes, values, basis=None):
        """Project a given vector from the canonical basis to the eigenbasis of the potential.
        @param nodes: The grid nodes $\gamma$ for the pointwise transformation.
        @param values: The list of vectors $\phi_i$ containing the values we want to transform.
        @keyword basis: A list of basisvectors $nu_i$. Allows to use this function for external data, similar to a static function.
        @return: Returned is another list containing the projection of the values into the eigenbasis.
        """    
        if basis is None:
            self.calculate_eigenvectors()
            eigv = self.evaluate_eigenvectors_at(nodes)
        else:
            eigv = basis
            
        result = []
        for i in xrange(0, self.number_components):
            tmp = numpy.zeros(values[0].shape, dtype=numpy.complexfloating)
            for j in xrange(0, self.number_components):
                tmp += eigv[i][j,:] * values[j]
            result.append(tmp)
        
        return tuple(result)


    def project_to_canonical(self, nodes, values, basis=None):
        """Project a given vector from the potential's eigenbasis to the canonical basis.
        @param nodes: The grid nodes $\gamma$ for the pointwise transformation.
        @param values: The list of vectors $\varphi_i$ containing the values we want to transform.
        @keyword basis: A list of basis vectors $nu_i$. Allows to use this function for external data, similar to a static function.
        @return: Returned is another list containing the projection of the values into the eigenbasis.
        """
        if basis is None:
            self.calculate_eigenvectors()
            eigv = self.evaluate_eigenvectors_at(nodes)
        else:
            eigv = basis

        result = []
        for i in xrange(0, self.number_components):
            tmp = numpy.zeros(values[0].shape, dtype=numpy.complexfloating)
            for j in xrange(0, self.number_components):
                tmp += eigv[j][i,:] * values[j]
            result.append(tmp)
        
        return tuple(result)


    def calculate_exponential(self, factor=1):
        """Calculate the matrix exponential $E = \exp\ofs{\alpha M}$. In the case where
        the matrix is of size bigger than $2 \times 2$ symbolical calculations become
        unfeasible. We use numerical approximations to determine the matrix exponential.
        @keyword factor: A prefactor $\alpha$ in the exponential.
        """
        # Store the factor for later numerical computations.
        self.factor = factor


    def evaluate_exponential_at(self, nodes):
        """Evaluate the exponential of the potential matrix $V$ at some grid nodes $\gamma$.
        For matrices of size $> 2$ we do completely numerical exponentation.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the exponential at.
        @return: The numerical approximation of the matrix exponential at the given grid nodes.
        """        
        n = len(nodes)
        tmp = numpy.ndarray((n,self.number_components,self.number_components), dtype=numpy.complexfloating)
        
        # evaluate potential
        values = self.evaluate_at(nodes)
        
        # fill in values
        for row in xrange(0, self.number_components):
            for col in xrange(0, self.number_components):
                tmp[:, row, col] = self.factor * values[row*self.number_components + col]

        # calculate exponential
        for i in xrange(0, n):
            tmp[i,:,:] = linalg.expm(tmp[i,:,:], 10)

        result = tuple([ tmp[:,row,col] for row in xrange(self.number_components) for col in xrange(self.number_components) ])
        return result

    
    def calculate_jacobian(self):
        """Calculate the jacobian matrix for each component $V_{i,j}$ of the potential.
        For potentials which depend only one variable $x$, this equals the first derivative.
        """
        self.jacobian = tuple([ sympy.diff(item, self.x) for item in self.potential ])
        self.jacobian_n = tuple([ sympy.vectorize(0)(sympy.lambdify(self.x, item, "numpy")) for item in self.jacobian ])

        
    def evaluate_jacobian_at(self, nodes, component=None):
        """Evaluate the jacobian at some grid nodes $\gamma$ for each component
        $V_{i,j}$ of the potential.
        @param nodes: The grid nodes $\gamma$ the jacobian gets evaluated at.
        @keyword component: The index tuple $\left(i,j\right)$ that specifies
        the potential's entry of which the jacobian is evaluated. (Defaults to 'None' to evaluate all)
        @return: Either a list or a single value depending on the optional parameters.
        """
        if not component is None:
            values = self.jacobian_n[component * self.number_components + component](nodes)
        else:
            values = tuple([ f(nodes) for f in self.jacobian_n ])

        return values
        

    def calculate_hessian(self):
        """Calculate the hessian matrix for each component $V_{i,j}$ of the potential.
        For potentials which depend only one variable $x$, this equals the second derivative.
        """
        self.hessian = tuple([ sympy.diff(item, self.x, 2) for item in self.potential ])
        self.hessian_n = tuple([ sympy.vectorize(0)(sympy.lambdify(self.x, item, "numpy")) for item in self.hessian ])


    def evaluate_hessian_at(self, nodes, component=None):
        """Evaluate the hessian at some grid nodes $\gamma$ for each component
        $V_{i,j}$ of the potential.
        @param nodes: The grid nodes $\gamma$ the hessian gets evaluated at.
        @keyword component: The index tuple $\left(i,j\right)$ that specifies
        the potential's entry of which the hessian is evaluated. (Or 'None' to evaluate all)
        @return: Either a list or a single value depending on the optional parameters.
        """
        if not component is None:
            values = self.hessian_n[component * self.number_components + component](nodes)
        else:
            values = tuple([ f(nodes) for f in self.hessian_n ])
    
        return values


    def calculate_local_quadratic(self, diagonal_component):
        """Calculate the local quadratic approximation matrix $U$ of the potential's
        eigenvalues in $\Lambda$. This function is used for the homogeneous case and
        takes into account the leading component $\chi$.
        @param diagonal_component: Specifies the index $i$ of the eigenvalue $\lambda_i$
        that gets expanded into a Taylor series $u_i$.
        """
        self.calculate_eigenvalues()

        # Use numerical differentiation for the case of three and more states.
        # We can not solve these problems by symbolical manipulations.
        self.quadratic_n = []

        v = partial(self.evaluate_eigenvalues_at, component=diagonal_component)
        self.quadratic_n.append(v)

        vj = ndt.Derivative(v)
        self.quadratic_n.append(vj)

        vh = ndt.Derivative(v, derOrder=2)
        self.quadratic_n.append(vh)
            
                        
    def evaluate_local_quadratic_at(self, nodes):
        """Numerically evaluate the local quadratic approximation matrix $U$ of
        the potential's eigenvalues in $\Lambda$ at the given grid nodes $\gamma$.
        This function is used for the homogeneous case and takes into account the
        leading component $\chi$.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the quadratic approximation at.
        @return: A list of arrays containing the values of $U_{i,j}$ at the nodes $\gamma$.
        """
        result = tuple([ f(nodes) for f in self.quadratic_n ])
        return result


    def calculate_local_remainder(self, diagonal_component):
        """Calculate the non-quadratic remainder matrix $W$ of the quadratic
        approximation matrix $U$ of the potential's eigenvalue matrix $\Lambda$.
        This function is used for the homogeneous case and takes into account the
        leading component $\chi$.
        @param diagonal_component: Specifies the index $\chi$ of the leading component $\lambda_\chi$.
        """
        self.calculate_local_quadratic(diagonal_component=diagonal_component)

        v, vj, vh = self.quadratic_n
        
        quadratic = lambda q, node: numpy.real(v(q) + vj(q)*(node-q) + 0.5*vh(q)*(node-q)**2)
        
        self.nonquadratic_s = []
        for row in xrange(self.number_components):
            for col in xrange(self.number_components):
                # Avoid closure issues
                def element(row, col):
                    if row == col:
                        return lambda q, node: self.functions[row*self.number_components+col](node) - quadratic(q, node)
                    else:
                        return lambda q, node: self.functions[row*self.number_components+col](node)

                self.nonquadratic_s.append( element(row, col) )
        
        self.nonquadratic_n = []
        for i, item in enumerate(self.nonquadratic_s):
            self.nonquadratic_n.append( numpy.vectorize(item) )
        

    def evaluate_local_remainder_at(self, position, nodes, component=None):
        """Numerically evaluate the non-quadratic remainder matrix $W$ of the quadratic
        approximation matrix $U$ of the potential's eigenvalues in $\Lambda$ at the
        given nodes $\gamma$. This function is used for the homogeneous and the 
        inhomogeneous case and just evaluates the remainder matrix $W$.
        @param position: The point $q$ where the Taylor series is computed.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the potential at.
        @keyword component: The component $\left(i,j\right)$ of the remainder matrix $W$
        that is evaluated.
        @return: A list with a single entry consisting of an array containing the
        values of $W$ at the nodes $\gamma$.
        """
        if component is not None:
            (row, col) = component
            f = self.nonquadratic_n[row*self.number_components+col]
            result = f(numpy.real(position), numpy.real(nodes))
        else:
            result = tuple([ f(numpy.real(position), numpy.real(nodes)) for f in self.nonquadratic_n ])
        
        return result


    def calculate_local_quadratic_multi(self):
        """Calculate the local quadratic approximation matrix $U$ of all the
        potential's eigenvalues in $\Lambda$. This function is used for the inhomogeneous case.
        """
        self.calculate_eigenvalues()

        # Use numerical differentiation for the case of three and more states.
        # We can not solve these problems by symbolical manipulations.
        self.quadratic_multi_n = []
        
        for index in xrange(self.number_components):
            tmp = []
            v = partial(self.evaluate_eigenvalues_at, component=index)
            tmp.append(v)
            
            vj = ndt.Derivative(v)
            tmp.append(vj)

            vh = ndt.Derivative(v, derOrder=2)
            tmp.append(vh)
     
            self.quadratic_multi_n.append(tmp)
            

    def evaluate_local_quadratic_multi_at(self, nodes, component=None):
        """Numerically evaluate the local quadratic approximation matrix $U$ of
        the potential's eigenvalues in $\Lambda$ at the given grid nodes $\gamma$.
        This function is used for the inhomogeneous case.
        @param nodes: The grid nodes $\gamma$ we want to evaluate the quadratic approximation at.
        @keyword component: The component $\left(i,j\right)$ of the quadratic approximation
        matrix $U$ that is evaluated.
        @return: A list of arrays or a single array containing the values of $U_{i,j}$ at the nodes $\gamma$.
        """      
        if component is not None:
            result = [ numpy.array(f(nodes), dtype=numpy.complexfloating) for f in self.quadratic_multi_n[component] ]
        else:
            result = []
            for item in self.quadratic_multi_n:
                tmp = []
                for f in item:
                    tmp.append( numpy.array(f(nodes), dtype=numpy.complexfloating) )
                result.append(tmp)

        return result
        
        
    def calculate_local_remainder_multi(self):
        """Calculate the non-quadratic remainder matrix $W$ of the quadratic
        approximation matrix $U$ of the potential's eigenvalue matrix $\Lambda$.
        This function is used for the inhomogeneous case.
        """
        self.calculate_eigenvalues()
        
        def f(v,vj,vh):
            return lambda q, node: numpy.real(v(q) + vj(q)*(node-q) + 0.5*vh(q)*(node-q)**2)
        
        quadratic = []
        for index, item in enumerate(self.quadratic_multi_n):
            quadratic.append(f(item[0], item[1], item[2]))

        self.nonquadratic_s = []
        for row in xrange(self.number_components):
            for col in xrange(self.number_components):
                # Avoid closure issues
                def element(row, col):
                    if row == col:
                        return lambda q, node: self.functions[row*self.number_components+col](node) - quadratic[row](q, node)
                    else:
                        return lambda q, node: self.functions[row*self.number_components+col](node)

                self.nonquadratic_s.append( element(row, col) )
        
        self.nonquadratic_n = []
        for i, item in enumerate(self.nonquadratic_s):
            self.nonquadratic_n.append( numpy.vectorize(item) )
            
