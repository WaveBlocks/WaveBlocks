algorithm = "spawning_nonadiabatic"
#algorithm = "hagedorn"

potential = "delta_gap"

T = 10
dt = 0.01

eps = 0.1
delta = 0.75*eps

f = 4.0
ngn = 4096

leading_component = 0
basis_size = 80

P = 1.0j
Q = 1.0-5.0j
S = 0.0

parameters = [ (P, Q, S, 1.0, -5.0), (P, Q, S, 1.0, -5.0) ]
coefficients = [[(0,1.0)], [(0,0.0)]]

matrix_exponential = "arnoldi"

write_nth = 5

spawn_method = "projection"
#spawn_method = "lumping"
spawn_order = 0
spawn_max_order = 32

# This is a better oracle
spawn_condition = "norm_derivative_threshold_l2"
spawn_hist_len = 20
spawn_threshold = 0.05
spawn_deriv_threshold = 5e-06
