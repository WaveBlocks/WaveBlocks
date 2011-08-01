"""The WaveBlocks Project

Script to spawn new wavepackets aposteriori to an already completed simulation.
This can be used to evaluate spawning errors and test criteria for finding the
best spawning time.

@author: R. Bourquin
@copyright: Copyright (C) 2011 R. Bourquin
@license: Modified BSD License
"""

import sys

import numpy as np
from scipy import linalg as spla

from WaveBlocks import ParameterProvider
from WaveBlocks import IOManager
from WaveBlocks import PotentialFactory
from WaveBlocks import HagedornWavepacket
from WaveBlocks import NonAdiabaticSpawner


def aposteriori_spawning(fin, fout, pin, pout, save_canonical=False):
    """
    @param f: An I{IOManager} instance providing the simulation data.
    @keyword datablock: The data block where the results are.
    """
    # Number of time steps we saved
    timesteps = fin.load_wavepacket_timegrid()
    nrtimesteps = timesteps.shape[0]

    params = fin.load_wavepacket_parameters()
    coeffs = fin.load_wavepacket_coefficients()

    # A data transformation needed by API specification
    coeffs = [ [ coeffs[i,j,:] for j in xrange(pin["ncomponents"]) ] for i in xrange(nrtimesteps) ]

    # The potential
    Potential = PotentialFactory.create_potential(pin)

    # Initialize a mother Hagedorn wavepacket with the data from another simulation
    HAWP = HagedornWavepacket(pin)
    HAWP.set_quadrator(None)

    # Initialize an empty wavepacket for spawning
    SWP = HagedornWavepacket(pout)
    SWP.set_quadrator(None)

    # Initialize a Spawner
    NAS = NonAdiabaticSpawner(pout)

    # Try spawning for these components, if none is given, try it for all.
    if not "spawn_components" in parametersout:
        components = range(pin["ncomponents"])
    else:
        components = parametersout["spawn_components"]

    # Iterate over all timesteps and spawn
    for i, step in enumerate(timesteps):
        print(" Try spawning at timestep "+str(step))

        # Configure the wave packet and project to the eigenbasis.
        HAWP.set_parameters(params[i])
        HAWP.set_coefficients(coeffs[i])

        # Project to the eigenbasis as the parameter estimation
        # has to happen there because of coupling.
        T = HAWP.clone()
        T.project_to_eigen(Potential)

        # Try spawning a new packet for each component
        estps = [ NAS.estimate_parameters(T, component=acomp) for acomp in components ]

        for index, ps in enumerate(estps):
            if ps is not None:
                # One choice of the sign
                U = SWP.clone()
                U.set_parameters(ps)
                # Project the coefficients to the spawned packet
                tmp = T.clone()
                NAS.project_coefficients(tmp, U, component=components[index])

                # Other choice of the sign
                V = SWP.clone()
                # Transform parameters
                psm = list(ps)
                B = ps[0]
                Bm = -np.real(B)+1.0j*np.imag(B)
                psm[0] = Bm
                V.set_parameters(psm)
                # Project the coefficients to the spawned packet
                tmp = T.clone()
                NAS.project_coefficients(tmp, V, component=components[index])

                # Compute some inner products to finally determine which parameter set we use
                parameters = {"eps":SWP.eps}
                ou = abs(inner(parameters, T, U))
                ov = abs(inner(parameters, T, V))

                # Choose the packet which maximizes the inner product.
                # This is the main point!
                if ou >= ov:
                    U = U
                else:
                    U = V

                # Finally do the spawning, this is essentially to get the remainder T right
                # The packet U is already ok by now.
                NAS.project_coefficients(T, U, component=components[index])

                # Transform back
                if save_canonical is True:
                    T.project_to_canonical(Potential)
                    U.project_to_canonical(Potential)

                # Save the mother packet rest
                fout.save_wavepacket_parameters(T.get_parameters(), timestep=step, block=2*index)
                fout.save_wavepacket_coefficients(T.get_coefficients(), timestep=step, block=2*index)

                # Save the spawned packet
                fout.save_wavepacket_parameters(U.get_parameters(), timestep=step, block=2*index+1)
                fout.save_wavepacket_coefficients(U.get_coefficients(), timestep=step, block=2*index+1)



def inner(parameters, P1, P2, QR=None):
    """Compute the inner product between two wavepackets.
    """
    # todo: Refactor this and put it into an own file/class!

    # Quadrature for <orig, s1>
    c1 = P1.get_coefficients(component=0)
    c2 = P2.get_coefficients(component=0)

    # Assuming same quadrature rule for both packets
    # Implies same basis size!
    if QR is None:
        QR = P1.get_quadrator().get_qr()

    # Mix the parameters for quadrature
    (Pm, Qm, Sm, pm, qm) = P1.get_parameters()
    (Ps, Qs, Ss, ps, qs) = P2.get_parameters()

    rm = Pm/Qm
    rs = Ps/Qs

    r = np.conj(rm)-rs
    s = np.conj(rm)*qm - rs*qs

    q0 = np.imag(s) / np.imag(r)
    Q0 = -0.5 * np.imag(r)
    QS = 1 / np.sqrt(Q0)

    # The quadrature nodes and weights
    nodes = q0 + parameters["eps"] * QS * QR.get_nodes()
    weights = QR.get_weights()

    # Basis sets for both packets
    basis_1 = P1.evaluate_base_at(nodes, prefactor=True)
    basis_2 = P2.evaluate_base_at(nodes, prefactor=True)

    R = QR.get_order()

    # Loop over all quadrature points
    tmp = 0.0j
    for r in xrange(R):
        tmp += np.conj(np.dot( c1[:,0], basis_1[:,r] )) * np.dot( c2[:,0], basis_2[:,r]) * weights[0,r]
    result = parameters["eps"] * QS * tmp

    return result




if __name__ == "__main__":
    # Input data manager
    iomin = IOManager()

    # Read file with simulation data
    try:
        iomin.open_file(filename=sys.argv[1])
    except IndexError:
        iomin.open_file()

    # Read a configuration file with the spawn parameters
    try:
        parametersspawn = ParameterProvider()
        parametersspawn.read_parameters(sys.argv[2])
    except IndexError:
        raise IOError("No spawn configuration given!")

    parametersin = iomin.get_parameters()

    # Check if we can start a spawning simulation
    if parametersin["algorithm"] != "hagedorn":
        iomin.finalize()
        raise ValueError("Unknown propagator algorithm.")

    # Parameters for spawning simulation
    parametersout = ParameterProvider()

    # Transfer the simulation parameters
    parametersout.set_parameters(parametersin)

    # And add spawning related configurations variables
    parametersout.update_parameters(parametersspawn)

    # How much time slots do we need
    tm = parametersout.get_timemanager()
    slots = tm.compute_number_saves()

    # Second IOM for output data of the spawning simulation
    iomout = IOManager()
    iomout.create_file(parametersout, filename="simulation_results_spawn.hdf5")

    # Allocate all the data blocks
    iomout.add_grid(parametersout)
    iomout.save_grid(iomin.load_grid())
    iomout.add_wavepacket(parametersin)

    for i in xrange(1,2*len(parametersout["spawn_components"])):
        iomout.create_block()
        iomout.add_grid_reference(blockfrom=i, blockto=0)
        if i % 2 == 0:
            # Block for remainder / mother after spawning
            iomout.add_wavepacket(parametersin, block=i)
        else:
            # Block for spawned packet
            iomout.add_wavepacket(parametersout, block=i)

    # Really do the aposteriori spawning simulation
    aposteriori_spawning(iomin, iomout, parametersin, parametersout)

    # Close the inpout/output files
    iomin.finalize()
    iomout.finalize()
