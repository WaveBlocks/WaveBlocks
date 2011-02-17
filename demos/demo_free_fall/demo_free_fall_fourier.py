algorithm = "fourier"

potential = {}
potential["potential"] = "x/4"

T = 8.5
dt = 0.01

eps = 0.1

f = 2.0
ngn = 4096

leading_component = 0
basis_size = 4

P = 1.0j
S = 0.0
Q = 1.0

parameters = [ (P, Q, S, 1.0, -2.0) ]
coefficients = [[(0, 1.0)]]

write_nth = 2
