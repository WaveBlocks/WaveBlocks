"""The WaveBlocks Project

IOM plugin providing functions for handling the
propagation operators that appear in the Fourier
algorithm.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import numpy as np


def add_fourieroperators(self, parameters, block=0):
    # Store the propagation operators (if available)
    grp_pr = self.srf["datablock_"+str(block)].create_group("propagation")
    grp_op = grp_pr.create_group("operators")
    grp_op.create_dataset("opkinetic", (parameters.ngn,), np.floating)
    grp_op.create_dataset("oppotential", (parameters.ngn, parameters.ncomponents**2), np.complexfloating)
    
    
def save_fourieroperators(self, operators, block=0):
    """Save the kinetic and potential operator to a file.
    @param operators: The operators to save, given as (T, V).
    """
    # Save the kinetic propagation operator
    path = "/datablock_"+str(block)+"/propagation/operators/opkinetic"
    self.srf[path][...] = np.squeeze(operators[0])
    # Save the potential propagation operator
    path = "/datablock_"+str(block)+"/propagation/operators/oppotential"
    for index, item in enumerate(operators[1]):
        self.srf[path][:,index] = item
        
        
def load_fourieroperators(self, block=0):
    path = "/datablock_"+str(block)+"/propagation/operators/"
    opT = self.srf[path+"opkinetic"]
    opV = self.srf[path+"oppotential"]
    opV = [ opV[:,index] for index in xrange(self.parameters.ncomponents**2) ]
    
    return (opT, opV)
