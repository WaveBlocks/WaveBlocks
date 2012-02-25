algorithm = "spawning_adiabatic"
#algorithm = "hagedorn"

potential = "eckart"

T = 70
dt = 0.005

eps = 0.0234218**0.5

basis_size = 300

parameters = [ (0.1935842258501978j, 5.1657101481699996, 0.0, 0.24788547371, -7.55890450883) ]
coefficients = [[(0, 1.0)]]

leading_component = 0

f = 9.0
ngn = 4096

write_nth = 20

# Better spawning oracle
spawn_method = "projection"
spawn_max_order = 16
spawn_order = 0

spawn_condition = "high_k_norm_derivative_threshold"
spawn_K0 = 100
spawn_hist_len = 20
spawn_threshold = 0.1
spawn_deriv_threshold = 5e-06
