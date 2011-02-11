"""
NUMDIFFTOOLS
============
Suite of tools to solve automatic numerical differentiation
problems in one or more variables. All of these methods also
produce error estimates on the result.
A pdf file is also provided to explain the theory behind these tools.

To test if the toolbox is working paste the following in an interactive
python session::

   import numdifftools as nd
   nd.test(coverage=True)

Derivative:
-----------
A flexible tool for the computation of derivatives of order 1 through 4
on any scalar function. Finite differences are used in an adaptive manner,
coupled with a Romberg extrapolation methodology to provide a maximally
accurate result. The user can configure many of the options, changing
the order of the method or the extrapolation, even allowing the user to
specify whether central, forward or backward differences are used.

Gradient
--------
Computes the gradient vector of a scalar function of one or more variables
at any location.

Jacobian
--------
Computes the Jacobian matrix of a vector (or array) valued function of
one or more variables.

Hessian
-------
Computes the Hessian matrix of all 2nd partial derivatives of a scalar
function of one or more variables.

Hessdiag
--------
The diagonal elements of the Hessian matrix are the pure second order
partial derivatives.

Examples
--------
Compute 1'st and 2'nd derivative of exp(x), at x == 1::

    >>> import numpy as np
    >>> import numdifftools as nd
    >>> fd = nd.Derivative(np.exp)              # 1'st derivative
    >>> fdd = nd.Derivative(np.exp,derOrder=2)  # 2'nd derivative
    >>> fd(1)
    array([ 2.71828183])

Nonlinear least squares::

    >>> xdata = np.reshape(np.arange(0,1,0.1),(-1,1))
    >>> ydata = 1+2*np.exp(0.75*xdata)
    >>> fun = lambda c: (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2
    >>> Jfun = Jacobian(fun)
    >>> Jfun([1,2,0.75]) # should be numerically zero
    array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,  -1.30229526e-17],
           [  0.00000000e+00,  -2.12916532e-17,   6.35877095e-17],
           [  0.00000000e+00,   0.00000000e+00,   6.95367972e-19],
           [  0.00000000e+00,   0.00000000e+00,  -2.13524915e-17],
           [  0.00000000e+00,  -3.08563327e-16,   7.43577440e-16],
           [  0.00000000e+00,   1.16128292e-15,   1.71041646e-15],
           [  0.00000000e+00,   0.00000000e+00,  -5.51592310e-16],
           [  0.00000000e+00,  -4.51138245e-19,   1.90866225e-15],
           [ -2.40861944e-19,  -1.82530534e-15,  -4.02819694e-15]])

Compute gradient of sum(x**2)::

    >>> fun = lambda x: np.sum(x**2)
    >>> dfun = Gradient(fun)
    >>> dfun([1,2,3])
    array([ 2.,  4.,  6.])


See also
--------
scipy.misc.derivative

"""