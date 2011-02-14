# Algorithm 
# =========

algorithm = "fourier"


# Time stepping
# =============

# Perform a simulation in the time interval [0, T].
T = 3.0

# Duration of a single time step.
dt = 0.02


# Semi-classical parameter
# ========================

# The epsilon parameter in the semiclassical scaling
eps = 0.2


# Potential
# =========

# The potential used in the simulation
potential = "delta_gap"

# Energy gap, used in the definition of this potential
delta = 0.1*eps


# Initial values
# ==============

# The hagedorn parameters of the initial wavepackets
parameters = [ (1.0j, 1.0-2.0j, 0.0, 1.0, -2.0), (1.0j, 1.0-2.0j, 0.0, 1.0, -2.0) ]

# A list with the lists of (index,value) tuples that set the coefficients
# of the basis functions for the initial wavepackets.
coefficients = [ [(0,1.0)], [(0,0.0)] ]

# Number of basis functions used for Hagedorn packets.
basis_size = 2


# Specific for Fourier
# ====================

# Number of grid nodes
ngn = 2**12

# Scaling factor for the computational domain
# The interval in the position space is [-f*pi, f*pi]
f = 2.0

# I/O configuration
# =================

# Write data to disk only each n-th timestep
write_nth = 2
