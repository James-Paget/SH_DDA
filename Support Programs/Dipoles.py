#
# Python wrapper for Dipoles.cpp
#
""" Python wrapper for the C++ shared library Dipoles.  This is for
anything related to the dipole generation and optical force calculation."""
import sys, platform
import ctypes, ctypes.util
import numpy.ctypeslib
import numpy as np

###################################################################################

# Find the library and load it
dipoles_path = ctypes.util.find_library("./Dipoles")
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


#
# General dipole function interfaces
#
grad_E_cc = Dipoles.grad_E_cc
grad_E_cc.argtypes = [numpy.ctypeslib.ndpointer(dtype=np.float64, ndim=1, shape=(3), flags='C_CONTIGUOUS'), numpy.ctypeslib.ndpointer(dtype=np.float64, ndim=1, shape=(6), flags='C_CONTIGUOUS'), ctypes.c_double, numpy.ctypeslib.ndpointer(dtype=np.float64, ndim=1, shape=(18), flags='C_CONTIGUOUS'),]
#
# Beam function interfaces
#
