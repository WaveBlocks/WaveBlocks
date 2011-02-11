"""The WaveBlocks Project

This file contains code for serializing simulation data.

@author: R. Bourquin
@copyright: Copyright (C) 2010, 2011 R. Bourquin
@license: Modified BSD License
"""

import os
import h5py as hdf
import pickle
import numpy as np

import GlobalDefaults
import ParameterProvider as ParameterProvider


class IOManager:
    """An IOManager class that can save various simulation results into data
    files. The output files can be processed further for producing e.g. plots.
    """

    def __init__(self):
        self.parameters = None
        self.srf = None
        
        
    def create_file(self, parameters, filename=GlobalDefaults.file_resultdatafile):
        """Set up a new I{IOManager} instance. The output files are created and opened.
        """
        #: Keep a reference to the parameters
        self.parameters = parameters

        # Create the file if it does not yet exist.
        # Otherwise raise an exception and avoid overwriting data.
        if os.path.lexists(filename):
            raise ValueError("Output file already exists!")        
        else:
            f = self.srf = hdf.File(filename)
            f.attrs["number_blocks"] = 0

        # Save the simulation parameters
        self.save_simulation_parameters(parameters)

        # Build up the hdf data tree, by default we provide one data block
        self.create_block()


    def load_file(self, filename=GlobalDefaults.file_resultdatafile):
        """Load a given file that contains the results from a former simulation.
        @keyword filename: The filename/path of the file we try to load.
        """
        if os.path.lexists(filename):
            self.srf = hdf.File(filename)
        else:
            raise ValueError("Output file does not exist!")

        # Load the simulation parameters
        self.parameters = ParameterProvider.ParameterProvider()
        p = self.srf["/simulation_parameters/parameters"].attrs

        for key, value in p.iteritems():
            self.parameters[key] = pickle.loads(value)

        # Compute some values on top of the given input parameters
        self.parameters.compute_parameters()


    def create_block(self):
        # Create a data block. Each data block can store several chunks
        # of information, and there may be multiple blocks per file.
        number_blocks = self.srf.attrs["number_blocks"]
        self.srf.create_group("datablock_" + str(number_blocks))
        self.srf.attrs["number_blocks"] += 1


    def save_simulation_parameters(self, parameters):
        # Store the simulation parameters
        grp_pa = self.srf.create_group("simulation_parameters")
        # We are only interested in the attributes of this data set
        # as they are used to store the simulation parameters.
        paset = grp_pa.create_dataset("parameters", (1,1))

        for param, value in parameters:
            # Store all the values as pickled strings because hdf can
            # only store strings or ndarrays as attributes.
            paset.attrs[param] = pickle.dumps(value)

       
    def get_parameters(self):
        """Return the reference to the current I{ParameterProvider} instance.
        """
        return self.parameters


    def must_resize(self, path, slot, axis=0):
        """Check if we must resize a given dataset and if yes, resize it.
        """
        #Ok, it's inefficient but sufficient for now.
        # todo: Consider resizing in bigger chunks and shrinking at the end if necessary.
        
        # Current size of the array
        cur_len = self.srf[path].shape[axis]

        # Is it smaller than what we need to store at slot "slot"?
        # If yes, then resize the array along the given axis.
        if cur_len-1 < slot:
            self.srf[path].resize(slot+1, axis=axis)

            
    def finalize(self):
        """Close the open output files."""
        self.srf.close()             


    #
    # Functions for adding data sets to the hdf tree
    #

    def add_grid(self, parameters, block=0):
        # Add storage for a grid
        self.srf["datablock_"+str(block)].create_dataset("grid", (parameters.dimension, parameters.ngn), np.floating)


    def add_grid_reference(self, blockfrom=1, blockto=0):
        self.srf["datablock_"+str(blockfrom)]["grid"] = hdf.SoftLink("/datablock_"+str(blockto)+"/grid")
        

    def add_propagators(self, parameters, block=0):
        # Store the propagation operators (if available)
        grp_pr = self.srf["datablock_"+str(block)].create_group("propagation")
        grp_op = grp_pr.create_group("operators")
        grp_op.create_dataset("opkinetic", (parameters.ngn,), np.floating)
        grp_op.create_dataset("oppotential", (parameters.ngn, parameters.ncomponents**2), np.complexfloating)


    def add_wavefunction(self, parameters, timeslots=None, block=0):
        # Store the sampled wavefunction
        grp_wf = self.srf["datablock_"+str(block)].require_group("wavefunction")

        # Create the dataset with appropriate parameters
        if timeslots is None:
            # This case is event based storing
            daset_psi = grp_wf.create_dataset("Psi", (1, parameters.ncomponents, parameters.ngn), dtype=np.complexfloating, chunks=(1, parameters.ncomponents, parameters.ngn))
            daset_psi_tg = grp_wf.create_dataset("timegrid", (timeslots,), dtype=np.integer, chunks=(1,))

            daset_psi.resize(0, axis=0)
            daset_psi_tg.resize(0, axis=0)
        else:
            # User specified how much space is necessary.
            daset_psi = grp_wf.create_dataset("Psi", (timeslots, parameters.ncomponents, parameters.ngn), dtype=np.complexfloating)
            daset_psi_tg = grp_wf.create_dataset("timegrid", (timeslots,), dtype=np.integer)
            
        daset_psi_tg.attrs["pointer"] = 0


    def add_wavepacket(self, parameters, timeslots=None, block=0):
        # Store the wave packets
        grp_wp = self.srf["datablock_"+str(block)].require_group("wavepacket")

        # Create the dataset with appropriate parameters
        if timeslots is None:
            # This case is event based storing
            daset_pi = grp_wp.create_dataset("Pi", (1, 1, 5), dtype=np.complexfloating, chunks=(1, 1, 5))
            daset_c = grp_wp.create_dataset("coefficients", (1, parameters.ncomponents, parameters.basis_size), dtype=np.complexfloating, chunks=(1, parameters.ncomponents, parameters.basis_size))
            daset_tg = grp_wp.create_dataset("timegrid", (1,), dtype=np.integer, chunks=(1,))

            daset_pi.resize(0, axis=0)
            daset_c.resize(0, axis=0)
            daset_tg.resize(0, axis=0)
        else:
            # User specified how much space is necessary.
            daset_pi = grp_wp.create_dataset("Pi", (timeslots, 1, 5), np.complexfloating)        
            daset_c = grp_wp.create_dataset("coefficients", (timeslots, parameters.ncomponents, parameters.basis_size), np.complexfloating)
            daset_tg = grp_wp.create_dataset("timegrid", (timeslots,), np.integer)

        # Attach pointer to data instead timegrid
        # Reason is that we have have two save functions but one timegrid
        daset_pi.attrs["pointer"] = 0
        daset_c.attrs["pointer"] = 0


    def add_wavepacket_inhomog(self, parameters, timeslots=None, block=0):
        # Store the wave packets
        grp_wp = self.srf["datablock_"+str(block)].require_group("wavepacket_inhomog")

        # Create the dataset with appropriate parameters
        if timeslots is None:
            # This case is event based storing
            daset_pi = grp_wp.create_dataset("Pi", (1, parameters.ncomponents, 5), dtype=np.complexfloating, chunks=(1, parameters.ncomponents, 5))
            daset_c = grp_wp.create_dataset("coefficients", (1, parameters.ncomponents, parameters.basis_size), dtype=np.complexfloating, chunks=(1, parameters.ncomponents, parameters.basis_size))
            daset_tg = grp_wp.create_dataset("timegrid", (1,), dtype=np.integer, chunks=(1,))

            daset_pi.resize(0, axis=0)
            daset_c.resize(0, axis=0)
            daset_tg.resize(0, axis=0)
        else:
            # User specified how much space is necessary.
            daset_pi = grp_wp.create_dataset("Pi", (timeslots, parameters.ncomponents, 5), dtype=np.complexfloating)        
            daset_c = grp_wp.create_dataset("coefficients", (timeslots, parameters.ncomponents, parameters.basis_size), dtype=np.complexfloating)
            daset_tg = grp_wp.create_dataset("timegrid", (timeslots,), dtype=np.integer)

        # Attach pointer to data instead timegrid
        # Reason is that we have have two save functions but one timegrid
        daset_pi.attrs["pointer"] = 0
        daset_c.attrs["pointer"] = 0


    def add_norm(self, parameters, timeslots=None, block=0):
        # Store the norms
        grp_ob = self.srf["datablock_"+str(block)].require_group("observables")

        # Create the dataset with appropriate parameters
        grp_no = grp_ob.create_group("norm")

        if timeslots is None:
            # This case is event based storing
            daset_n = grp_no.create_dataset("norm", (1, parameters.ncomponents), dtype=np.floating, chunks=(1, parameters.ncomponents))
            daset_tg = grp_no.create_dataset("timegrid", (1,), dtype=np.integer, chunks=(1,))

            daset_n.resize(0, axis=0)
            daset_tg.resize(0, axis=0)
        else:
            # User specified how much space is necessary.
            daset_n = grp_no.create_dataset("norm", (timeslots, parameters.ncomponents), dtype=np.floating)
            daset_tg = grp_no.create_dataset("timegrid", (timeslots,), dtype=np.integer)

        daset_tg.attrs["pointer"] = 0


    def add_energies(self, parameters, timeslots=None, block=0, total=False):
        # Store the potential and kinetic energies
        grp_ob = self.srf["datablock_"+str(block)].require_group("observables")
        
        # Create the dataset with appropriate parameters
        grp_en = grp_ob.create_group("energies")

        if timeslots is None:
            # This case is event based storing
            daset_ek = grp_en.create_dataset("kinetic", (1, parameters.ncomponents), dtype=np.floating, chunks=(1, parameters.ncomponents))
            daset_ep = grp_en.create_dataset("potential", (1, parameters.ncomponents), dtype=np.floating, chunks=(1, parameters.ncomponents))
            daset_tg = grp_en.create_dataset("timegrid", (1,), dtype=np.integer, chunks=(1,))

            daset_ek.resize(0, axis=0)
            daset_ep.resize(0, axis=0)

            daset_tg.resize(0, axis=0)

            if total is True:
                daset_to = grp_en.create_dataset("total", (1, 1), dtype=np.floating, chunks=(1, 1))
                daset_to.resize(0, axis=0)
                daset_to.attrs["pointer"] = 0
        else:
            # User specified how much space is necessary.
            daset_ek = grp_en.create_dataset("kinetic", (timeslots, parameters.ncomponents), dtype=np.floating)
            daset_ep = grp_en.create_dataset("potential", (timeslots, parameters.ncomponents), dtype=np.floating)
            daset_tg = grp_en.create_dataset("timegrid", (timeslots,), dtype=np.integer)

            if total is True:
                daset_to = grp_en.create_dataset("total", (timeslots, 1), dtype=np.floating)
                daset_to.attrs["pointer"] = 0

        daset_tg.attrs["pointer"] = 0


    #
    # Functions for actually save simulation data
    #

    def save_grid(self, grid, block=0):
        """Save the grid nodes to a file.
        """
        path = "/datablock_"+str(block)+"/grid"
        self.srf[path][:] = np.real(grid)


    def save_operators(self, operators, block=0):
        """Save the kinetic and potential operator to a file.
        @param operators: The operators to save, given as (T, V).
        """
        # Save the kinetic propagation operator
        path = "/datablock_"+str(block)+"/propagation/operators/opkinetic"
        self.srf[path][...] = np.squeeze(operators[0])
        # Save the potential propagation operator
        path = "/datablock_"+str(block)+"/propagation/operators/oppotential"
        for index, item in enumerate(operators[1]):
            self.srf[path][:,index] = item


    def save_wavefunction(self, wavefunction, block=0, timestep=None):
        """Save a I{WaveFunction} instance. The output is suitable for the plotting routines.
        @param wavefunction: The I{WaveFunction} instance to save.
        @keyword block: The data block where to store the wavefunction.
        """
        #@refactor: take wavefunction or wavefunction.get_values() as input?
        pathtg = "/datablock_"+str(block)+"/wavefunction/timegrid"
        pathd = "/datablock_"+str(block)+"/wavefunction/Psi"
        timeslot = self.srf[pathtg].attrs["pointer"]

        # Store the values given
        self.must_resize(pathd, timeslot)
        for index, item in enumerate(wavefunction.get_values()):
            self.srf[pathd][timeslot,index,:] = item

        # Write the timestep to which the stored values belong into the timegrid
        self.must_resize(pathtg, timeslot)
        self.srf[pathtg][timeslot] = timestep

        # Update the pointer
        self.srf[pathtg].attrs["pointer"] += 1


    def save_parameters(self, parameters, timestep=None, block=0):
        """Save the parameters of the Hagedorn wavepacket to a file.
        @param parameters: The parameters of the Hagedorn wavepacket.
        """
        pathtg = "/datablock_"+str(block)+"/wavepacket/timegrid"
        pathd = "/datablock_"+str(block)+"/wavepacket/Pi"
        timeslot = self.srf[pathd].attrs["pointer"]

        # Write the data
        self.must_resize(pathd, timeslot)
        self.srf[pathd][timeslot,0,:] = parameters

        # Write the timestep to which the stored values belong into the timegrid
        self.must_resize(pathtg, timeslot)
        self.srf[pathtg][timeslot] = timestep

        # Update the pointer
        self.srf[pathd].attrs["pointer"] += 1


    def save_parameters_inhomog(self, parameters, timestep=None, block=0):
        """Save the parameters of the Hagedorn wavepacket to a file.
        @param parameters: The parameters of the Hagedorn wavepacket.
        """
        pathtg = "/datablock_"+str(block)+"/wavepacket_inhomog/timegrid"
        pathd = "/datablock_"+str(block)+"/wavepacket_inhomog/Pi"
        timeslot = self.srf[pathd].attrs["pointer"]

        # Write the data
        self.must_resize(pathd, timeslot)
        for index, item in enumerate(parameters):
            self.srf[pathd][timeslot,index,:] = np.array(item)
        
        # Write the timestep to which the stored values belong into the timegrid
        self.must_resize(pathtg, timeslot)
        self.srf[pathtg][timeslot] = timestep

        # Update the pointer
        self.srf[pathd].attrs["pointer"] += 1


    def save_coefficients(self, coefficients, timestep=None, block=0):
        """Save the coefficients of the Hagedorn wavepacket to a file.
        @param coefficients: The coefficients of the Hagedorn wavepacket.
        """
        pathtg = "/datablock_"+str(block)+"/wavepacket/timegrid"
        pathd = "/datablock_"+str(block)+"/wavepacket/coefficients"
        timeslot = self.srf[pathd].attrs["pointer"]

        # Write the data
        self.must_resize(pathd, timeslot)
        for index, item in enumerate(coefficients):
            self.srf[pathd][timeslot,index,:] = np.squeeze(item)

        # Write the timestep to which the stored values belong into the timegrid
        self.must_resize(pathtg, timeslot)
        self.srf[pathtg][timeslot] = timestep

        # Update the pointer
        self.srf[pathd].attrs["pointer"] += 1


    def save_coefficients_inhomog(self, coefficients, timestep=None, block=0):
        """Save the coefficients of the Hagedorn wavepacket to a file.
        @param coefficients: The coefficients of the Hagedorn wavepacket.
        """
        pathtg = "/datablock_"+str(block)+"/wavepacket_inhomog/timegrid"
        pathd = "/datablock_"+str(block)+"/wavepacket_inhomog/coefficients"
        timeslot = self.srf[pathd].attrs["pointer"]

        # Write the data
        self.must_resize(pathd, timeslot)
        for index, item in enumerate(coefficients):
            self.srf[pathd][timeslot,index,:] = np.squeeze(item)

        # Write the timestep to which the stored values belong into the timegrid
        self.must_resize(pathtg, timeslot)
        self.srf[pathtg][timeslot] = timestep

        # Update the pointer
        self.srf[pathd].attrs["pointer"] += 1

    
    def save_energies(self, energies, timestep=None, block=0):
        """Save the kinetic and potential energies to a file.
        @param energies: A tuple \texttt{(ekin, epot)} containing the energies.
        """
        pathtg = "/datablock_"+str(block)+"/observables/energies/timegrid"
        pathd1 = "/datablock_"+str(block)+"/observables/energies/kinetic"
        pathd2 = "/datablock_"+str(block)+"/observables/energies/potential"
        timeslot = self.srf[pathtg].attrs["pointer"]

        #todo: remove np,array
        ekin = np.real(np.array(energies[0]))
        epot = np.real(np.array(energies[1]))

        # Write the data
        self.must_resize(pathd1, timeslot)
        self.must_resize(pathd2, timeslot)
        self.srf[pathd1][timeslot,:] = ekin
        self.srf[pathd2][timeslot,:] = epot

        # Write the timestep to which the stored values belong into the timegrid
        self.must_resize(pathtg, timeslot)
        self.srf[pathtg][timeslot] = timestep

        # Update the pointer
        self.srf[pathtg].attrs["pointer"] += 1


    def save_energies_total(self, total_energy, timestep=None, block=0):
        """Save the total to a file.
        @param energies: An array containing the energies.
        """
        pathd = "/datablock_"+str(block)+"/observables/energies/total"

        timeslot = self.srf[pathd].attrs["pointer"]

        #todo: remove np,array
        etot = np.real(np.array(total_energy))

        # Write the data
        self.must_resize(pathd, timeslot)
        self.srf[pathd][timeslot,0] = etot

        # Update the pointer
        self.srf[pathd].attrs["pointer"] += 1


    def save_norm(self, norm, timestep=None, block=0):
        """Save the norm of the wave packets"""
        pathtg = "/datablock_"+str(block)+"/observables/norm/timegrid"
        pathd = "/datablock_"+str(block)+"/observables/norm/norm"
        timeslot = self.srf[pathtg].attrs["pointer"]

        #@refactor: remove np,array
        norms = np.real(np.array(norm))

        # Write the data
        self.must_resize(pathd, timeslot)
        self.srf[pathd][timeslot,:] = norms

        # Write the timestep to which the stored values belong into the timegrid
        self.must_resize(pathtg, timeslot)
        self.srf[pathtg][timeslot] = timestep

        # Update the pointer
        self.srf[pathtg].attrs["pointer"] += 1


    #
    # Functions for retrieving simulation data
    #

    def find_timestep_index(self, timegridpath, timestep):
        """Lookup the index for a given timestep.
        @note: Assumes the timegrid array is strictly monotone.
        """
        # todo: Make this more efficient
        # todo: allow for slicing etc
        timegrid = self.srf[timegridpath]
        index = np.squeeze(np.where(timegrid[:] == timestep))

        if index.shape == (0,):
            raise ValueError("No data for given timestep!")
        
        return index
        

    def load_grid(self, block=0):
        path = "/datablock_"+str(block)+"/grid"
        return np.squeeze(self.srf[path])


    def load_operators(self, block=0):
        path = "/datablock_"+str(block)+"/propagation/operators/"
        opT = self.srf[path+"opkinetic"]
        opV = self.srf[path+"oppotential"]
        opV = [ opV[:,index] for index in xrange(self.parameters.ncomponents**2) ]

        return (opT, opV)


    def load_wavefunction_timegrid(self, block=0):
        pathtg = "/datablock_"+str(block)+"/wavefunction/timegrid"
        return self.srf[pathtg][:]


    def load_wavefunction(self, timestep=None, block=0):
        pathtg = "/datablock_"+str(block)+"/wavefunction/timegrid"
        pathd = "/datablock_"+str(block)+"/wavefunction/Psi"
        if timestep is not None:
            index = self.find_timestep_index(pathtg, timestep)
            return self.srf[pathd][index,...]
        else:
            return self.srf[pathd][...]


    def load_norm_timegrid(self, block=0):
        pathtg = "/datablock_"+str(block)+"/observables/norm/timegrid"
        return self.srf[pathtg][:]


    def load_norm(self, timestep=None, block=0):
        pathtg = "/datablock_"+str(block)+"/observables/norm/timegrid"
        pathd = "/datablock_"+str(block)+"/observables/norm/norm"
        if timestep is not None:
            index = self.find_timestep_index(pathtg, timestep)
            return self.srf[pathd][index,...]
        else:
            return self.srf[pathd][...]


    def load_energies_timegrid(self, block=0):
        pathtg = "/datablock_"+str(block)+"/observables/energies/timegrid"
        return self.srf[pathtg][:]


    def load_energies(self, timestep=None, block=0):
        pathtg = "/datablock_"+str(block)+"/observables/energies/timegrid"
        pathd1 = "/datablock_"+str(block)+"/observables/energies/kinetic"
        pathd2 = "/datablock_"+str(block)+"/observables/energies/potential"

        if timestep is not None:
            index = self.find_timestep_index(pathtg, timestep)
            ekin = self.srf[pathd1][index,...]
            epot = self.srf[pathd2][index,...]
        else:
            ekin = self.srf[pathd1][...]
            epot = self.srf[pathd2][...]
        
        return (ekin, epot)


    def load_energies_total(self, timestep=None, block=0):
        pathtg = "/datablock_"+str(block)+"/observables/energies/timegrid"
        pathd = "/datablock_"+str(block)+"/observables/energies/total"

        if timestep is not None:
            index = self.find_timestep_index(pathtg, timestep)
            etot = self.srf[pathd][index,...]
        else:
            etot = self.srf[pathd][...]

        return etot


    def load_wavepacket_timegrid(self, block=0):
        pathtg = "/datablock_"+str(block)+"/wavepacket/timegrid"
        return self.srf[pathtg][:]

    
    def load_parameters(self, timestep=None, block=0):
        pathtg = "/datablock_"+str(block)+"/wavepacket/timegrid"
        pathd = "/datablock_"+str(block)+"/wavepacket/Pi"
        if timestep is not None:
            index = self.find_timestep_index(pathtg, timestep)
            params = self.srf[pathd][index,0,:]
        else:
            params = self.srf[pathd][...,0,:]

        return params

        
    def load_coefficients(self, timestep=None, block=0):
        pathtg = "/datablock_"+str(block)+"/wavepacket/timegrid"
        pathd = "/datablock_"+str(block)+"/wavepacket/coefficients"

        if timestep is not None:
            index = self.find_timestep_index(pathtg, timestep)
            return self.srf[pathd][index,...]
        else:
            return self.srf[pathd][...]


    def load_wavepacket_inhomog_timegrid(self, block=0):
        pathtg = "/datablock_"+str(block)+"/wavepacket_inhomog/timegrid"
        return self.srf[pathtg][:]

    
    def load_parameters_inhomog(self, timestep=None, block=0):
        pathtg = "/datablock_"+str(block)+"/wavepacket_inhomog/timegrid"
        pathd = "/datablock_"+str(block)+"/wavepacket_inhomog/Pi"
        if timestep is not None:
            index = self.find_timestep_index(pathtg, timestep)
            params = [ self.srf[pathd][index,i,:] for i in xrange(self.parameters.ncomponents) ]
        else:
            params = [ self.srf[pathd][...,i,:] for i in xrange(self.parameters.ncomponents) ]

        return params

        
    def load_coefficients_inhomog(self, timestep=None, block=0):
        pathtg = "/datablock_"+str(block)+"/wavepacket_inhomog/timegrid"
        pathd = "/datablock_"+str(block)+"/wavepacket_inhomog/coefficients"

        if timestep is not None:
            index = self.find_timestep_index(pathtg, timestep)
            return self.srf[pathd][index,...]
        else:
            return self.srf[pathd][...]
