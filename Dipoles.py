#
# Python wrapper for Dipoles.cpp
#
""" Python wrapper for the C++ shared library Dipoles.  This is for
anything related to the dipole generation and optical force calculation."""
import os
import sys, platform
import ctypes, ctypes.util
import numpy.ctypeslib
import numpy as np
import Beams

###################################################################################

# Find the library and load it
#dipoles_path = ctypes.util.find_library("./Dipoles")
match(sys.platform):
    case('linux'):
        dipoles_abs_path = os.path.abspath("./libDipoles.so");
    case('darwin'):
        dipoles_abs_path = os.path.abspath("./Dipoles");
    case _:
        print("System not found; Found ",sys.platform);
        sys.exit();
dipoles_path = ctypes.util.find_library(dipoles_abs_path)
if not dipoles_path:
    print("Unable to find the specified Dipoles library.")
    sys.exit()

try:
    Dipoles = ctypes.CDLL(dipoles_path)
except OSError:
    print("Unable to load the Dipoles C++ library")
    sys.exit()
#
#
# Python helper functions
#

    
#
# Python wrapper functions
#
def py_grad_E_cc(position, polarisation, kvec):
    """
    position: x, y, z coordinates of point (double precision);
    polarisation: complex vector;
    kvec: scalar wave vetor (should be corrected for medium)
    gradEE: a complex array to receive the gradients.
    """
    dgradEE = np.zeros(18,dtype=np.float64)
    polarisation_unwrap = polarisation.view(dtype=np.float64).reshape((6,1)).flatten()
    Dipoles.grad_E_cc(position, polarisation_unwrap, kvec, dgradEE)
    gradEE = dgradEE.view(dtype=np.complex128).reshape((3,3))

#    print("position:",position)
#    print("polarisation:",polarisation)
#    print("kvec:",kvec)
#    print("gradEE:",gradEE)

    return gradEE

#==============================================================================
# Wrapper for new optical force array code
#==============================================================================

def py_optical_force_array(array_of_particles,dipole_radius,dipole_primitive,inverse_polarisation, beam_collection):
    """
    Wrapper function for the new optical force code.
    """
    num_particles = len(array_of_particles)
    num_dipoles = len(dipole_primitive)
#    print(array_of_particles.shape)
#    print(dipole_primitive.shape)
    forces = np.zeros((num_particles,3),dtype=np.float64)
    inv_polar_unwrap = inverse_polarisation.view(dtype=np.float64).reshape((num_particles*2,1)).flatten()
    #print(inv_polar_unwrap)
    Dipoles.optical_force_array(array_of_particles, num_particles, dipole_radius, dipole_primitive, num_dipoles, inv_polar_unwrap, beam_collection, forces)
    return forces
#==============================================================================
# Wrapper for new optical force and torque array code
#==============================================================================

def py_optical_force_torque_array(array_of_particles, dipole_primitive_num, dipole_radius, dipole_primitives, inverse_polarisation, beam_collection):
    """
    Wrapper function for the new optical force code, including torques (= r X F), and couples (= p X E).
    """
    #
    # NOTE; dipole_primitives NOT using float64, just float
    #
    num_particles = len(array_of_particles)
    num_dipoles = len(dipole_primitives)
    dipole_forces = np.zeros((num_dipoles,3),dtype=np.float64)
    forces = np.zeros((num_particles,3),dtype=np.float64)
    torques = np.zeros((num_particles,3),dtype=np.float64)
    couples = np.zeros((num_particles,3),dtype=np.float64)
    inv_polar_unwrap = inverse_polarisation.view(dtype=np.float64).reshape((num_particles*2,1)).flatten()
    #print(inv_polar_unwrap)
    Dipoles.optical_force_torque_array(array_of_particles, num_particles, dipole_radius, dipole_primitives, dipole_primitive_num, inv_polar_unwrap, beam_collection, dipole_forces, forces, torques, couples)
    
    return dipole_forces, forces, torques, couples
#
# General dipole function interfaces
#
grad_E_cc = Dipoles.grad_E_cc
grad_E_cc.argtypes = [numpy.ctypeslib.ndpointer(dtype=np.float64, ndim=1, shape=(3), flags='C_CONTIGUOUS'), numpy.ctypeslib.ndpointer(dtype=np.float64, ndim=1, shape=(6), flags='C_CONTIGUOUS'), ctypes.c_double, numpy.ctypeslib.ndpointer(dtype=np.float64, ndim=1, shape=(18), flags='C_CONTIGUOUS'),]

ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags="C")
ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags="C")
ND_POINTER_1_INT = np.ctypeslib.ndpointer(dtype=np.int64,ndim=1,flags="C")
ND_POINTER_2_INT = np.ctypeslib.ndpointer(dtype=np.int64,ndim=2,flags="C")

optical_force_array = Dipoles.optical_force_array

optical_force_array.argtypes = [ND_POINTER_2,ctypes.c_int,ctypes.c_double,ND_POINTER_2,ND_POINTER_1_INT,ND_POINTER_1,ctypes.POINTER(Beams.BEAM_COLLECTION),ND_POINTER_2]
#optical_force_array.argtypes = [ND_POINTER_2,ctypes.c_int,ctypes.c_double,ND_POINTER_2,ctypes.c_int, ND_POINTER_1,ctypes.POINTER(Beams.BEAM_COLLECTION),ND_POINTER_2]
#optical_force_array.restype = ctypes.POINTER(ctypes.c_double)
optical_force_torque_array = Dipoles.optical_force_torque_array

optical_force_torque_array.argtypes = [ND_POINTER_2,ctypes.c_int,ctypes.c_double,ND_POINTER_2, ND_POINTER_1_INT, ND_POINTER_1,ctypes.POINTER(Beams.BEAM_COLLECTION),ND_POINTER_2,ND_POINTER_2,ND_POINTER_2,ND_POINTER_2]
#optical_force_torque_array.argtypes = [ND_POINTER_2,ctypes.c_int,ctypes.c_double,ND_POINTER_2,ctypes.c_int, ND_POINTER_1,ctypes.POINTER(Beams.BEAM_COLLECTION),ND_POINTER_2,ND_POINTER_2,ND_POINTER_2]
#
#
###################################################################################
