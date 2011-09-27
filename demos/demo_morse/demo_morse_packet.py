algorithm = "hagedorn"

potential = "morse"
D = 3
a = 0.3

T = 10
dt = 0.005

eps = 0.2

f = 4.0
ngn = 4096

basis_size = 64
leading_component = 0

P = 1.0j
Q = 1
S = 0.0
p = 0.0
q = 1.5

parameters = [ (P, Q, S, p, q) ]
coefficients = [[(0, 1.0)]]

write_nth = 20

matrix_exponential = "arnoldi"
arnoldi_steps = 20
