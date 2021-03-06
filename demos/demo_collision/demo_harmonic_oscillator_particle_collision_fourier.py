algorithm = "fourier"

potential = "quadratic"

T = 7.5
dt = 0.01

eps = 0.1

f = 4.0
ngn = 4096

basis_size = 4

P = 1.0j
Q = 1.0
S = 0.0

initial_values = [[ 0, (P, Q, S,  0.0,   2.0), [(0,1.0)]],
                  [ 0, (P, Q, S,  0.0,  -2.0), [(0,1.0)]]]

write_nth = 2
