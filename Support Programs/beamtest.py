import numpy as np
import ctypes
import Beams

###################################################################################
# Beam type parameters for switch
###################################################################################
# Beams.BEAMTYPE_PLANE = 0
# Beams.BEAMTYPE_GAUSS_BARTON5 = 1
# Beams.BEAMTYPE_GAUSS_CSP = 2
# Beams.BEAMTYPE_BESSEL = 3
###################################################################################


    
wavelength = 1e-6
w0 = wavelength * 0.8  # breaks down when w0 < wavelength
alpha_by_k = 0.5

n1 = 3.9
ep1 = n1 * n1
ep2 = 1.0
radius = 200e-9  # half a micron to one micron
water_permittivity = 80.4
k = 2 * np.pi / wavelength
a0 = (4 * np.pi * 8.85e-12) * (radius ** 3) * ((ep1 - 1) / (ep1 + 2))
a = a0 / (1 - (2 / 3) * 1j * k ** 3 * a0)  # complex form from Chaumet (2000)
# a = a0

jones_vector = np.zeros(2,dtype=complex)
jones_vector[0] = (1 + 0j) / np.sqrt(2)
jones_vector[1] = (0 + 1j) / np.sqrt(2)  # change the polarisation of the beam
E0 = 3e6

###################################################################################
# New code for BEAM class
###################################################################################
n_beams = 2
beam_collection = np.zeros(n_beams,dtype=object)
mybeam = Beams.BEAM()
kk = 2*np.pi / wavelength
kt_by_kz = 0.2  # ratio of transverse to longitudinal wavevector, kz currently set to 2pi/wavelength (in general_bessel_constants)
kz = kk / np.sqrt(1+kt_by_kz**2)
kt = kt_by_kz*kz
order = 0
mybeam.kz = kz
mybeam.kt = kt
mybeam.kt_by_kz = kt_by_kz
mybeam.E0 = E0
mybeam.beamtype = Beams.BEAMTYPE_GAUSS_BARTON5
mybeam.order = order
mybeam.w0 = w0
mybeam.k = kk
#
# Build the Jones matrix
#
jones_matrix = np.zeros((2,2),dtype=np.float64)
jones_matrix[0][0] = 1/np.sqrt(2)  # real part
jones_matrix[0][1] = 0          # imaginary part
jones_matrix[1][0] = 0          # real part
jones_matrix[1][1] = 1/np.sqrt(2)  # imaginary part
mybeam.jones = np.ctypeslib.as_ctypes(jones_matrix.flatten())
#
# Beam orientation matrix
# Beam is by default parallel to z.  Take a rotation about x axis, keeping beam in z-y plane
# with final axis parallel to x.
#
angle = 0.0 # degrees (+ve in anticlockwise sense)
angler = angle * np.pi / 180.0 # radians
rotation_matrix = np.zeros((3,3),dtype=np.float64)
# new x axis
rotation_matrix[0][0] = 1.0
rotation_matrix[0][1] = 0.0
rotation_matrix[0][2] = 0.0
# new y axis
rotation_matrix[1][0] = 0.0
rotation_matrix[1][1] = np.cos(angler)
rotation_matrix[1][2] = np.sin(angler)
# new z axis
rotation_matrix[2][0] = 0.0
rotation_matrix[2][1] = -np.sin(angler)
rotation_matrix[2][2] = np.cos(angler)
#
mybeam.rotation = np.ctypeslib.as_ctypes(rotation_matrix.flatten())
#
# Beam position vector
#
beamposition = np.array((0.0,0.0,0.0),dtype=np.float64) # specify position in metres
mybeam.translation = np.ctypeslib.as_ctypes(beamposition)
#
# Store in collection
#
beam_collection[0] = mybeam
###################################################################################
mybeam = Beams.BEAM()
kk = 2*np.pi / wavelength
kt_by_kz = 0.2  # ratio of transverse to longitudinal wavevector, kz currently set to 2pi/wavelength (in general_bessel_constants)
kz = kk / np.sqrt(1+kt_by_kz**2)
kt = kt_by_kz*kz
order = 0
mybeam.kz = kz
mybeam.kt = kt
mybeam.kt_by_kz = kt_by_kz
mybeam.E0 = E0
mybeam.beamtype = Beams.BEAMTYPE_BESSEL
mybeam.order = order
mybeam.w0 = w0
mybeam.k = kk
#
# Build the Jones matrix
#
jones_matrix = np.zeros((2,2),dtype=np.float64)
jones_matrix[0][0] = 1/np.sqrt(2)  # real part
jones_matrix[0][1] = 0          # imaginary part
jones_matrix[1][0] = 0          # real part
jones_matrix[1][1] = -1/np.sqrt(2)  # imaginary part
mybeam.jones = np.ctypeslib.as_ctypes(jones_matrix.flatten())
#
# Beam orientation matrix
# Beam is by default parallel to z.  Take a rotation about x axis, keeping beam in z-y plane
# with final axis parallel to x.
#
angle = 0.0 # degrees (+ve in anticlockwise sense)
angler = angle * np.pi / 180.0 # radians
rotation_matrix = np.zeros((3,3),dtype=np.float64)
# new x axis
rotation_matrix[0][0] = 1.0
rotation_matrix[0][1] = 0.0
rotation_matrix[0][2] = 0.0
# new y axis
rotation_matrix[1][0] = 0.0
rotation_matrix[1][1] = np.cos(angler)
rotation_matrix[1][2] = np.sin(angler)
# new z axis
rotation_matrix[2][0] = 0.0
rotation_matrix[2][1] = -np.sin(angler)
rotation_matrix[2][2] = np.cos(angler)
#
mybeam.rotation = np.ctypeslib.as_ctypes(rotation_matrix.flatten())
#
# Beam position vector
#
beamposition = np.array((0.0,0.0,0.0),dtype=np.float64) # specify position in metres
mybeam.translation = np.ctypeslib.as_ctypes(beamposition)
#
# Store in collection
#
beam_collection[1] = mybeam
###################################################################################

EE = np.zeros(3,dtype=np.complex128) # initialised to zero
Beams.all_incident_fields((0.0,0.0,0.0), beam_collection, EE)
print(EE[0],EE[1],EE[2])
print(EE)
