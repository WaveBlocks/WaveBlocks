algorithm = "hagedorn"

potential = {}
potential["potential"] = "x/4"

T = 6.5
dt = 0.01

eps = 0.1

f = 2.0
ngn = 4096

leading_component = 0
basis_size = 64

P = 1.0j
Q = 1.0
S = 0.0

parameters = [ (P, Q, S, 0.0, 2.0) ]
coefficients = [[(0, 1.0)]]

matrix_exponential = "pade"

write_nth = 2
