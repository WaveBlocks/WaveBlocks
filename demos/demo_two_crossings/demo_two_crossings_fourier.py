algorithm = "fourier"

potential = "two_crossings"

T = 16
dt = 0.01

eps = 0.2

delta = 5.0*eps

f = 5.0
ngn = 4096

leading_component = 0
basis_size = 128

P = 1.0j
Q = 1.0-6.0j
S = 0.0

parameters = [ (P, Q, S, 1.0, -6.0), (P, Q, S, 1.0, -6.0) ]
coefficients = [[(0, 1.0)], [(0, 0.0)]]

write_nth = 2
