"""The WaveBlocks Project

Some selected functions for complex math.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import hstack, cumsum, diff, around, sqrt, abs, angle, exp, sqrt, pi


def continuate(data, jump=2.0*pi):
    """Make the given data continuous by removing all jumps of size k*jump
    but not touching jumps of any other size. This can be used to overcome
    issues with the branch cut along the negative axis.
    @param data: An array with the input data.
    @keyword jump: The basic size of jumps which will be removed. Default is 2*pi.
    @Note: There may be issues with jumps that are of size nearly k*jump.
    """
    return data - jump*hstack([0,cumsum(around(diff(data)/jump))])


def cont_angle(data):
    """Compute the angle of a complex number *not* constrained to
    the principal value and avoiding discontinuities at the branch cut.
    @param data: An array with the input data.
    @note: This function just applies 'continuate(.)' to the complex phase.
    """
    return continuate(angle(data))


def cont_sqrt(data):
    """Compute the complex square root (following the Riemann surface)
    yields a result *not* constrained to the principal value and avoiding
    discontinuities at the branch cut.
    @param data: An array with the input data.
    @note: This function applies 'continuate(.)' to the complex phase
    and computes the complex square root according to the formula
    $\sqrt{z} = \sqrt{r} \cdot \exp \left( i \cdot \frac{\phi}{2} \right)$
    """
    return sqrt(abs(data))*exp(1.0j*continuate(angle(data))/2)
