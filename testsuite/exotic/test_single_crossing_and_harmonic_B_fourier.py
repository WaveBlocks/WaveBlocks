algorithm = "fourier"

potential = {}
potential["potential"] = [["1/2 * sigma * x**2 + 2", 0              , 0               ],
                          [0                       , "1/2 * tanh(x)", "delta"         ],
                          [0                       , "delta"        , "-1/2 * tanh(x)"]]

potential["defaults"] = {"sigma":"1/2"}


T = 10
dt = 0.01

eps = 0.2

delta = 1.0*eps

f = 4.0
ngn = 4096

basis_size = 64

P = 2.0j
Q = 0.5
S = 0.0

parameters = [ (P, Q, S, 0.5, -2.0), (P, Q, S, 0.5, -2.0), (P, Q, S, 0.5, -2.0) ]
coefficients = [[(0,1.0)], [(0,1.0)], [(0,0.0)]]

write_nth = 2
