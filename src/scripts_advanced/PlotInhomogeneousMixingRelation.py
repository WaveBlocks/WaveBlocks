"""The WaveBlocks Project

Plot the evolution of some mixing results of the parameters Pi_i = (P,Q,S,p,q)
of an inhomogeneous hagedorn wavepacket during the time propagation.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import sys
from numpy import real, imag, conj
from matplotlib.pyplot import *

from WaveBlocks import IOManager


def load_data(f):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    """
    parameters = f.get_parameters()
    timegrid = f.load_inhomogwavepacket_timegrid()
    
    Pi = f.load_inhomogwavepacket_parameters()

    # Number of components
    N = parameters.ncomponents

    Phist = [ Pi[i][:,0] for i in xrange(N) ]
    Qhist = [ Pi[i][:,1] for i in xrange(N) ]
    Shist = [ Pi[i][:,2] for i in xrange(N) ]
    phist = [ Pi[i][:,3] for i in xrange(N) ]
    qhist = [ Pi[i][:,4] for i in xrange(N) ]

    return (N, timegrid, Phist, Qhist, Shist, phist, qhist)


def plot_data(N, timegrid, Phist, Qhist, Shist, phist, qhist):
    # Plot Hagedorn relation (Pbar_k / Qbar_k - P_l / Q_l)
    fig = figure(figsize=(14,14))

    for i in xrange(N):
        for j in xrange(N):
            ax = fig.add_subplot(N,N,i*N+j+1)
            
            data = conj(Phist[i]) / conj(Qhist[i]) - Phist[j] / Qhist[j]

            ax.plot(timegrid, real(data)) 
            ax.plot(timegrid, imag(data)) 

            ax.grid(True)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$\Re(\cdot), \Im(\cdot)$")
            ax.set_title(r"$\frac{\overline{P_"+str(i)+r"}}{\overline{Q_"+str(i)+r"}} - \frac{P_"+str(j)+r"}{Q_"+str(j)+r"}$")

    fig.savefig("wavepacket_parameter_mixing_relation.png")
    close(fig)


    # Plot "commutation" relation
    for i in xrange(N):
        for j in xrange(N):
    
            fig = figure(figsize=(10,10))
            ax = fig.gca()

            data1 = conj(Phist[i]) / conj(Qhist[i]) - Phist[j] / Qhist[j]
            data2 = Phist[i] / Qhist[i] - conj(Phist[j]) / conj(Qhist[j])
            
            data3 = -2 * (1 / abs(Qhist[i]**2) + 1/abs(Qhist[j]**2))
            
            ax.plot(timegrid, imag(data1))
            ax.plot(timegrid, imag(data2))
            ax.plot(timegrid, real(data3))
            
            ax.plot(timegrid, imag(data1) - imag(data2))

            ax.grid(True)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$\Im(\cdot), \Im(\cdot), \Im(\cdot)-\Im(\cdot)$")
            ax.set_title(r"$\Im\left(\frac{\overline{P_"+str(i)+r"}}{\overline{Q_"+str(i)+r"}} - \frac{P_"+str(j)+r"}{Q_"+str(j)+r"}\right) - \Im\left(\frac{P_"+str(i)+r"}{Q_"+str(i)+r"} - \frac{\overline{P_"+str(j)+r"}}{\overline{Q_"+str(j)+r"}}\right)$")

            fig.savefig("wavepacket_parameter_mixing_exchange"+str(i)+str(j)+".png")
            close(fig)


    # Plot parameter mixing of q
    fig = figure(figsize=(14,14))

    for i in xrange(N):
        for j in xrange(N):
            ax = fig.add_subplot(N,N,i*N+j+1)

            rk = Phist[i] / Qhist[i]
            rl = Phist[j] / Qhist[j]

            upper = conj(rk) * qhist[i] - rl * qhist[j]
            lower = conj(rk) - rl

            data = imag(upper) / imag(lower)

            ax.plot(timegrid, qhist[i], "b")
            ax.plot(timegrid, qhist[j], "g")

            ax.plot(timegrid, data, "r") 

            ax.grid(True)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$\frac{\Im(\overline{r_k}q_k - r_l q_l)}{\Im(\overline{r_k}-r_l)}$")
            ax.set_title(r"Mixing of $q_0$ from $\Pi_"+str(i)+r"$ and $\Pi_"+str(j)+r"$")

    fig.savefig("wavepacket_parameter_mixing_q.png")
    close(fig)


    # Plot parameter mixing of Q
    fig = figure(figsize=(14,14))

    for i in xrange(N):
        for j in xrange(N):
            ax = fig.add_subplot(N,N,i*N+j+1)

            rk = Phist[i] / Qhist[i]
            rl = Phist[j] / Qhist[j]

            data = -0.5 * imag(conj(rk)-rl)

            ax.plot(timegrid, real(Qhist[i]), "b")
            ax.plot(timegrid, imag(Qhist[i]), "g")

            ax.plot(timegrid, real(Qhist[j]), "c")
            ax.plot(timegrid, imag(Qhist[j]), "m")

            ax.plot(timegrid, data, "r") 

            ax.grid(True)
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$-\frac{\Im(\overline{r_k}-r_l)}{2}$")
            ax.set_title(r"Mixing of $Q_0$ from $\Pi_"+str(i)+r"$ and $\Pi_"+str(j)+r"$")

    fig.savefig("wavepacket_parameter_mixing_Q.png")
    close(fig)


if __name__ == "__main__":
    iom = IOManager()

    # Read file with simulation data
    try:
        iom.open_file(filename=sys.argv[1])
    except IndexError:
        iom.open_file()

    parameters = iom.get_parameters()

    if parameters.algorithm != "multihagedorn":
        sys.exit("Can only postprocess multihagedorn algorithm data. Silent return ...")

    plot_data(*load_data(iom))

    iom.finalize()
