"""The WaveBlocks Project

Plot all potentials and put the definitions into a latex file.
This autogenerated latex file should replace the manually created
section in the manual some day.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

from sympy import *
from numpy import *
from matplotlib.pyplot import *

from WaveBlocks import PotentialFactory
import WaveBlocks.PotentialLibrary as PL


pots =  [ i for i in dir(PL) if not i.startswith("_") ]

params = {"delta":0.2, "delta1":0.2, "delta2":0.2}

x = linspace(-5,5,5000)

file = open("potentials.tex", "wb")


for pot in pots:
    print("Potential is: " + pot)

    # Extract raw information from potential library
    # Never do this like here!
    potdef = PL.__dict__[pot]

    potname = pot.replace("_","\_")

    if type(potdef["potential"]) == str:
        potformula = latex(sympify(potdef["potential"]))
    else:
        potformula = latex(Matrix(sympify(potdef["potential"])))

    if potdef.has_key("defaults"):
        potdefaults = potdef["defaults"]
    else:
        potdefaults = {}

    # Create the potential "the right way"
    params["potential"] = pot
    P = PotentialFactory().create_potential(params)
    y = P.evaluate_eigenvalues_at(x)

    # Plot the potential
    figure()
    for yvals in y:
        plot(x, yvals)
    xlabel(r"$x$")
    ylabel(r"$\lambda_i\left(x\right)$")
    xlim(min(x), max(x))
    savefig(pot + ".pdf")

    # The latex code
    ls = """
\\begin{minipage}{0.5\\linewidth}
  Name:    \\texttt{""" + potname + """}
  \\begin{equation*}
    V\\left(x\right) = """ + potformula + """
  \\end{equation*}"""

    if len(potdefaults) > 0:
        ls += """
  Defaults:
  \\begin{align*}"""
        for key, value, in potdefaults.iteritems():
           ls += latex(sympify(key)) + " & = " + str(value) + "\\\\"
        ls += """
  \\end{align*}"""

    ls += """
\\end{minipage}
\\begin{minipage}{0.5\\linewidth}
  \\begin{center}
    \\includegraphics[scale=0.25]{./""" + pot + ".pdf" + """}
  \\end{center}
\\end{minipage}
      """

    ls.encode("UTF-8")
    file.write(ls)

file.close()
