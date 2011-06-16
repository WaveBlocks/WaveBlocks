algorithm = "hagedorn"

potential = {}
potential["potential"] = [["1/2 * sigma * x**2", 0                   ],
                          [0                   , "1/2 * sigma * x**2"]]
potential["defaults"] = {"sigma":"1/2"}


T = 12
dt = 0.01

eps = 0.1

f = 3.0
ngn = 4096

basis_size = 4

P = 2.0j
Q = 0.5
S = 0.0

parameters = [ (P, Q, S, -0.5, 2.0), (P, Q, S, -0.5, 2.0) ]
coefficients = [[(0, 1.0)], [(0, 1.0)]]

leading_component = 0

write_nth = 2
