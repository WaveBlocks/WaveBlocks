"""The WaveBlocks Project

IOM plugin providing functions for handling the
propagation operators that appear in the Fourier
algorithm.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_fourieroperators(self, parameters, blockid=0):
    r"""
    Add storage for the Fourier propagation operators.
    """
    grp_pr = self._srf[self._prefixb+str(blockid)].create_group("propagation")
    grp_op = grp_pr.create_group("operators")
    grp_op.create_dataset("opkinetic", (parameters["ngn"],), np.complexfloating)
    grp_op.create_dataset("oppotential", (parameters["ngn"], parameters["ncomponents"]**2), np.complexfloating)


def delete_fourieroperators(self, blockid=0):
    r"""
    Remove the stored Fourier operators.
    """
    try:
        del self._srf[self._prefixb+str(blockid)+"/propagation/operators"]
        # Check if there are other children, if not remove the whole node.
        if len(self._srf[self._prefixb+str(blockid)+"/propagation"].keys()) == 0:
            del self._srf[self._prefixb+str(blockid)+"/propagation"]
    except KeyError:
        pass


def has_fourieroperators(self, blockid=0):
    r"""
    Ask if the specified data block has the desired data tensor.
    """
    return ("propagation" in self._srf[self._prefixb+str(blockid)].keys() and
            "operators" in self._srf[self._prefixb+str(blockid)]["propagation"].keys())


def save_fourieroperators(self, operators, blockid=0):
    r"""
    Save the kinetic and potential operator to a file.

    :param operators: The operators to save, given as (T, V).
    """
    # Save the kinetic propagation operator
    path = "/"+self._prefixb+str(blockid)+"/propagation/operators/opkinetic"
    self._srf[path][...] = np.squeeze(operators[0].astype(np.complexfloating))
    # Save the potential propagation operator
    path = "/"+self._prefixb+str(blockid)+"/propagation/operators/oppotential"
    for index, item in enumerate(operators[1]):
        self._srf[path][:,index] = item.astype(np.complexfloating)


def load_fourieroperators(self, blockid=0):
    path = "/"+self._prefixb+str(blockid)+"/propagation/operators/"
    opT = self._srf[path+"opkinetic"]
    opV = self._srf[path+"oppotential"]
    opV = [ opV[:,index] for index in xrange(self._parameters["ncomponents"]**2) ]

    return (opT, opV)
