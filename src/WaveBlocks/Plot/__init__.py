"""The WaveBlocks Project

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

from legend import legend
from plotcf import plotcf
from stemcf import stemcf

try:
    from surfcf import surfcf
except ImportError:
    pass
