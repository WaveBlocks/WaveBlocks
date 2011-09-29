algorithm = "multihagedorn"

potential = "delta_gap"

T = 2.5
dt = 0.02

eps = 0.2

delta = 0.1*eps

P = 1.0j
Q = 1.0-2.0j
S = 0.0

parameters = [ (P, Q, S, 1.0, -2.0), (P, Q, S, 1.0, -2.0) ]
coefficients = [ [(0,1.0)], [(0,0.0)] ]

ngn = 2**12
f = 2.0

basis_size = 64
leading_component = 0

matrix_exponential = "pade"

write_nth = 2
