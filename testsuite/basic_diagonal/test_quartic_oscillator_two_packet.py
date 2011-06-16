algorithm = "hagedorn"

potential = {}
potential["potential"] = [["1/4 * sigma * x**4", 0                   ],
                          [0                   , "1/4 * sigma * x**4"]]
potential["defaults"] = {"sigma":"0.05"}


T = 6
dt = 0.01

eps = 0.1

f = 3.0
ngn = 4096

basis_size = 64

P = 2.0j
Q = 0.5
S = 0.0

parameters = [ (P, Q, S, -0.5, 2.0), (P, Q, S, -0.5, 2.0) ]
coefficients = [[(0, 1.0)], [(0, 1.0)]]

leading_component = 0

write_nth = 2
