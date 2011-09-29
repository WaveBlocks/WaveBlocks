algorithm = "multihagedorn"

potential = "two_quadratic"

T = 30
dt = 0.02

eps = 0.1

P = 1.0j
Q = 1.0
S = 0.0

coefficients = [ [(0,1.0)], [(1,1.0)] ]
parameters = [ (P, Q, S, 0.0, -2.0), (P, Q, S, 0.0, 1.0) ]

f = 5.0
ngn = 4096

basis_size = 4

matrix_exponential = "pade"

write_nth = 5
