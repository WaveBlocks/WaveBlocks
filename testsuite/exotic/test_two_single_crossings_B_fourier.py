algorithm = "fourier"

potential = {}
potential["potential"] = [["1/2 * tanh(x)+2", "delta"           , 0                , 0                 ],
                          ["delta"          , "-1/2 * tanh(x)+2", 0                , 0                 ],
                          [0                , 0                 , "1/2 * tanh(x)-2", "delta"           ],
                          [0                , 0                 , "delta"          , "-1/2 * tanh(x)-2"]]

potential["defaults"] = {"sigma":"1/2"}


T = 10
dt = 0.01

eps = 0.2

delta = 1.0*eps

f = 4.0
ngn = 4096

basis_size = 64

P = 1.0j
Q = 1.0-5.0j
S = 0.0

parameters = [ (P, Q, S, 1.0, -5.0), (P, Q, S, 1.0, -5.0), (P, Q, S, 1.0, -5.0), (P, Q, S, 1.0, -5.0) ]
coefficients = [[(0,1.0)], [(0,0.0)], [(0,1.0)], [(0,0.0)]]

write_nth = 2
