algorithm = "hagedorn"

potential = "eckart"

sigma = 100 * 3.8008 * 10**(-4.0)
a =  1.0/(2*0.52918)

T = 70
dt = 0.005

eps = 0.0234218**(0.5)

f = 9.0
ngn = 4096

basis_size = 512
leading_component = 0

P = 0.1935842258501978j
Q = 5.1657101481699996
S = 0.0
p = 0.24788547371
q = -7.55890450883

parameters = [ (P, Q, S, p, q) ]
coefficients = [[(0, 1.0)]]

write_nth = 20

matrix_exponential = "arnoldi"
arnoldi_steps = 15
