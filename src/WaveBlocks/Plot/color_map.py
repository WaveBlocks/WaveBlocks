"""The WaveBlocks Project

Function for mapping complex numbers to colors specified
by the usual color map used in quantum mechanics.

@author: R. Bourquin
@copyright: Copyright (C) 2011, 2012 R. Bourquin
@license: Modified BSD License
"""

from numpy import angle, empty, pi, fmod, abs, arctan2, real, where
from matplotlib.colors import hsv_to_rgb


def color_map(data, phase=None, modulus=None, darken=1.0):
    """Color mapping according to the QM standard map.

    :param data: The complex numbers.
    :param phase: The phase of the complex numbers, computed if not given.
    :param modulus: The modulus of the complex numbers, computed if not given.
    :param darken: How strong to take into account the modulus of the data to darken colors.
                   Values with :math:`|z| = R` will get fully saturated colors
                   while :math:`|z| = 0` is black and :math:`|z| \rightarrow \infty`
                   get whiter and whiter.
    :type darken: Float or ``None`` to disable darkening of colors. Default is :math:`R = 1.0`.
    :param axes: The axes instance used for plotting.
    """
    if len(data.shape) == 1:
        hsv_colors = empty((1,)+data.shape+(3,))
    else:
        hsv_colors = empty(data.shape+(3,))

    if phase is None:
        phase = angle(data)

    hsv_colors[..., 0] = 0.5*fmod(phase+2*pi,2*pi)/pi
    hsv_colors[..., 1] = 1.0
    hsv_colors[..., 2] = 1.0

    # Darken colors such that 0+0i maps to black
    if darken is not None:
        if modulus is None:
            modulus = abs(data)

        # Lightness
        hsv_colors[...,2] = 2*arctan2(real(modulus),darken)/pi

        # Saturation
        l = hsv_colors[...,2]
        hsv_colors[...,1] = where(l <= 0.5, 2*l, 2*(1-l))

    return hsv_to_rgb(hsv_colors)
