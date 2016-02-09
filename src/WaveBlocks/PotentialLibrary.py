"""The WaveBlocks Project

This file contains some ready made potentials with up to five
separate energy levels. This is a pure data file without any
code. To load the potentials, use the ``PotentialFactory``.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

# Potentials with one energy level
##################################

#: Free particle
free_particle = {}
free_particle["potential"] = "c"
free_particle["defaults"] = {"c":"0"}

#: Simple harmonic potential
quadratic = {}
quadratic["potential"] = "1/2 * sigma * x**2"
quadratic["defaults"] = {"sigma":"1/2"}

#: Perturbed harmonic potential
pert_quadratic = {}
pert_quadratic["potential"] = "1/2 * sigma * x**2 + 1/2 * delta**2 * x**2"
pert_quadratic["defaults"] = {"sigma":0.05, "delta":0.2}

#: A simple fourth order anharmonic potential
quartic = {}
quartic["potential"] = "1/4 * sigma * x**4"
quartic["defaults"] = {"sigma":0.05}

#: A potential consisting of a cosine wave
cos_waves = {}
cos_waves["potential"] = "a * (1 - cos(b*x))"
cos_waves["defaults"] = {"a":0.07, "b":1.0}

#: The Morse potential
morse = {}
morse["potential"] = "D * (1 - exp(-a*(x-x0)))**2"
morse["defaults"] = {"D":3.0, "a":0.3, "x0":0.0}

#: A double well potential
double_well = {}
double_well["potential"] = "sigma * (x**2 - 1)**2"
double_well["defaults"] = {"sigma":1.0}

#: The Eckart potential
eckart = {}
eckart["potential"] = "sigma * cosh(x/a)**(-2)"
eckart["defaults"] = {"sigma":100*3.8088*10**(-4), "a":1.0/(2.0*0.52918)}

#: A smooth unitstep like wall
wall = {}
wall["potential"] = "atan(sigma*x) + pi/2"
wall["defaults"] = {"sigma":10.0}

#: A narrow 'V'-like potential
v_shape = {}
v_shape["potential"] = "1/2 * sqrt(tanh(x)**2+4*delta**2)"
v_shape["defaults"] = {"delta":0.2}


# Potentials with two energy levels
###################################

#: Double harmonic potential for two components
two_quadratic = {}
two_quadratic["potential"] = [["1/2*sigma*x**2", "0"             ],
                              ["0",              "1/2*sigma*x**2"]]
two_quadratic["defaults"] = {"sigma":0.05}

#: Double quartic anharmonic potential for two components
two_quartic = {}
two_quartic["potential"] = [["1/4*sigma*x**4", "0"             ],
                              ["0",            "1/8*sigma*x**4"]]
two_quartic["defaults"] = {"sigma":1.0}

#: A potential with a single avoided crossing
delta_gap = {}
delta_gap["potential"] = [["1/2 * tanh(x)", "delta"         ],
                          ["delta",         "-1/2 * tanh(x)"]]

#: Diagonalized single avoided crossing
delta_gap_diag = {}
delta_gap_diag["potential"] = [["sqrt(delta**2 + tanh(x)**2/4)", "0"                             ],
                               ["0",                             "-sqrt(delta**2 + tanh(x)**2/4)"]]

#: A potential with two avoided crossings in series
two_crossings = {}
two_crossings["potential"] = [["tanh(x-rho)*tanh(x+rho)/2", "delta/2"                   ],
                              ["delta/2",                   "-tanh(x-rho)*tanh(x+rho)/2"]]
two_crossings["defaults"] = {"rho":3.0}


# Potentials with three energy levels
#####################################

#: Decoupled harmonic potentials for three components
three_quadratic = {}
three_quadratic["potential"] = [["1/2 * sigma * x**2", "0",                  "0"                 ],
                                ["0",                  "1/2 * sigma * x**2", "0"                 ],
                                ["0",                  "0",                  "1/2 * sigma * x**2"]]
three_quadratic["defaults"] = {"sigma":0.05}

#: A potential with three energy levels and multiple crossings
three_levels = {}
three_levels["potential"] = [["tanh(x+rho) + tanh(x-rho)", "delta1",       "delta2"         ],
                             ["delta1",                    "-tanh(x+rho)", "0"              ],
                             ["delta2",                    "0",            "1 - tanh(x-rho)"]]
three_levels["defaults"] = {"rho":3.0}


# Potentials with four energy levels
####################################

#: Decoupled harmonic potentials for four components
four_quadratic = {}
four_quadratic["potential"] = [["1/2 * sigma * x**2", "0",                  "0",                  "0"                 ],
                               ["0",                  "1/2 * sigma * x**2", "0",                  "0"                 ],
                               ["0",                  "0",                  "1/2 * sigma * x**2", "0"                 ],
                               ["0",                  "0",                  "0",                  "1/2 * sigma * x**2"]]
four_quadratic["defaults"] = {"sigma":0.05}

#: Harmonic and higher order anharmonic potentials for four components
four_powers = {}
four_powers["potential"] = [["1/2 * sigma * x**2", "0",                  "0",                  "0"                 ],
                            ["0",                  "1/4 * sigma * x**4", "0",                  "0"                 ],
                            ["0",                  "0",                  "1/6 * sigma * x**6", "0"                 ],
                            ["0",                  "0",                  "0",                  "1/8 * sigma * x**8"]]
four_powers["defaults"] = {"sigma":0.05}


# Potentials with five energy levels
####################################

#: Decoupled harmonic potential for five components
five_quadratic = {}
five_quadratic["potential"] = [["1/2 * sigma * x**2", "0",                  "0",                  "0",                  "0"                 ],
                               ["0",                  "1/2 * sigma * x**2", "0",                  "0",                  "0"                 ],
                               ["0",                  "0",                  "1/2 * sigma * x**2", "0",                  "0"                 ],
                               ["0",                  "0",                  "0",                  "1/2 * sigma * x**2", "0"                 ],
                               ["0",                  "0",                  "0",                  "0",                  "1/2 * sigma * x**2"]]
five_quadratic["defaults"] = {"sigma":0.05}
