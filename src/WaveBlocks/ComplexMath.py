"""The WaveBlocks Project

Some selected functions for complex math.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from numpy import array, hstack, cumsum, diff, around, abs, angle, exp, sqrt, pi


def continuate(data, jump=2.0*pi, reference=0.0):
    r"""
    Make the given data continuous by removing all jumps of size k*jump
    but not touching jumps of any other size. This can be used to overcome
    issues with the branch cut along the negative axis.

    :param data: An array with the input data.
    :param jump: The basic size of jumps which will be removed. Default is 2*pi.
    :param reference: This value allows the specify the starting point for continuation explicitely.
                      It can be used together with ``data`` beeing a single number.

    .. Note:: There may be issues with jumps that are of size nearly k*jump.
    """
    data = hstack([array(reference), array(data)])
    return (data - jump*hstack([ 0.0, cumsum(around( diff(data)/jump )) ]))[1:]


def cont_angle(data, reference=None):
    r"""
    Compute the angle of a complex number *not* constrained to
    the principal value and avoiding discontinuities at the branch cut.

    :param data: An array with the input data.
    :param reference: This value allows the specify the starting point for continuation explicitely.
                      It can be used together with ``data`` beeing a single number.

    .. note:: This function just applies 'continuate(.)' to the complex phase.
    """
    if reference is None:
        # Return just cont_f(x)
        return continuate(angle(data))
    else:
        # Return a 2-tuple ( cont_f(x), new_reference )
        return 2*( continuate(angle(data), reference=reference) ,)


def cont_sqrt(data, reference=None):
    r"""
    Compute the complex square root (following the Riemann surface)
    yields a result *not* constrained to the principal value and avoiding
    discontinuities at the branch cut.

    :param data: An array with the input data.
    :param reference: This value allows the specify the starting point for continuation explicitely.
                      It can be used together with ``data`` beeing a single number.

    .. note:: This function applies 'continuate(.)' to the complex phase and computes the complex square root according
              to the formula :math:`\sqrt{z} = \sqrt{r} \cdot \exp \left( i \cdot \frac{\phi}{2} \right)`
    """
    if reference is None:
        # Return just cont_f(x)
        return sqrt(abs(data))*exp(1.0j*continuate(angle(data))/2)
    else:
        # Return a 2-tuple ( cont_f(x), new_reference )
        phi = continuate(angle(data), reference=reference)
        return (sqrt(abs(data))*exp(1.0j*phi/2)[0], phi)
