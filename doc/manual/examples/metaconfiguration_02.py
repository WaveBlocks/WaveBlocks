"""Remarks:

- You can use any valid python statement as value
- All statements are written to a pure python code file
- You can write numbers, lists etc as plain text strings
- All that is not in string form gets evaluated *right now*
- Remember to escape python strings twice
- You can use variable references but with great care!
- The ordering of the statements in the output file is such that
  all statements can be executed w.r.t. local variables. This is 
  some kind of topological sorting. Be warned, it's implemented
  using black magic and may fail now and then!

  That should be all ...
"""

# Global parameters that stay the same for all simulations:
GP = {}

GP["algorithm"] = "\"fourier\""
GP["potential"] = "\"delta_gap\""
GP["T"] = 3
GP["dt"] = 0.02
GP["parameters"] = "[ (1.0j, 1.0-6.0j, 0.0, 1.0, -6.0), (1.0j, 1.0-6.0j, 0.0, 1.0, -6.0) ]"
GP["coefficients"] = [ [(0,1.0)], [(0,0.0)] ]
GP["basis_size"] = 2
GP["ngn"] = 2**12
GP["f"] = 4.0
GP["write_nth"] = 2

# Local parameters that change with each simulation
LP = {}

LP["eps"] = [0.1, 0.5]
LP["delta"] = ["0.5*eps", "1.0*eps", "1.5*eps"]
