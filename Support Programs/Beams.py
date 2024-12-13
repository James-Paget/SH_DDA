#
# Python wrapper for Beams.cpp
#
""" Python wrapper for the C++ shared library Beams"""
import sys, platform
import ctypes, ctypes.util
import numpy.ctypeslib
import numpy as np

###################################################################################
# Beam type parameters for switch
###################################################################################
BEAMTYPE_PLANE = 0
BEAMTYPE_GAUSS_BARTON5 = 1
BEAMTYPE_GAUSS_CSP = 2
BEAMTYPE_BESSEL = 3
BEAMTYPE_LAGUERRE_GAUSSIAN = 4
###################################################################################
# Jones vector names
###################################################################################
POLARISATION_X = 0
POLARISATION_Y = 1
POLARISATION_RCP = 2
POLARISATION_LCP = 3
POLARISATION_RADIAL = 4
POLARISATION_AZIMUTHAL = 5
###################################################################################

# Find the library and load it
beams_path = ctypes.util.find_library("./Beams")
if not beams_path:
    print("Unable to find the specified library.")
    sys.exit()

try:
    Beams = ctypes.CDLL(beams_path)
except OSError:
    print("Unable to load the Beams C++ library")
    sys.exit()
#
# Beam class
#
class BEAM(ctypes.Structure):
    _fields_ = [
        ('beamtype', ctypes.c_int32),
        ('E0', ctypes.c_double),
        ('k', ctypes.c_double),
        ('kz', ctypes.c_double),
        ('kt', ctypes.c_double),
        ('kt_by_kz', ctypes.c_double),
        ('order', ctypes.c_int),
#        ('jones', numpy.ctypeslib.ndpointer(dtype=np.float64, ndim=2, shape=(2,2), flags='C_CONTIGUOUS'))
        ('jones', ctypes.c_double * 4),
        ('translation', ctypes.c_double * 3),
        ('rotation', ctypes.c_double * 9),
        ('w0', ctypes.c_double),
        ('gouy', ctypes.c_int),
        ('numkpoints', ctypes.c_int)
    ]
#
# Helper functions
#
def get_jones_matrix(polarisation_state):
    """
    Makes and returns a flattened Jones matrix (all real coefficients)
    """
    jones_matrix = np.zeros((2,2),dtype=np.float64)
    if polarisation_state == POLARISATION_X:
        jones_matrix[0][0] = 1  # real part
        jones_matrix[0][1] = 0          # imaginary part
        jones_matrix[1][0] = 0          # real part
        jones_matrix[1][1] = 0  # imaginary part
    elif polarisation_state == POLARISATION_Y:
        jones_matrix[0][0] = 0  # real part
        jones_matrix[0][1] = 0          # imaginary part
        jones_matrix[1][0] = 1          # real part
        jones_matrix[1][1] = 0  # imaginary part
    elif polarisation_state == POLARISATION_RCP:
        jones_matrix[0][0] = 1/np.sqrt(2)  # real part
        jones_matrix[0][1] = 0          # imaginary part
        jones_matrix[1][0] = 0          # real part
        jones_matrix[1][1] = 1/np.sqrt(2)  # imaginary part
    elif polarisation_state == POLARISATION_LCP:
        jones_matrix[0][0] = 1/np.sqrt(2)  # real part
        jones_matrix[0][1] = 0          # imaginary part
        jones_matrix[1][0] = 0          # real part
        jones_matrix[1][1] = -1/np.sqrt(2)  # imaginary part
    #elif polarisation_state == POLARISATION_RADIAL:
    #elif polarisation_state == POLARISATION_AZIMUTHAL:
    else:
        print("Error - invalid polarisation state")
    return jones_matrix.flatten()
    
def get_rotation_matrix(angle,zangle):
    """
    Makes and returns a flattened rotation matrix using:
    angle - rotation about x axis (degrees)
    zangle - rotation about z axis (degrees)
    """
#
# Beam orientation matrix
# Beam is by default parallel to z.  Take a rotation about x axis, keeping beam in z-y plane
# with final axis parallel to x.
#
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
# Second rotation about z axis
#
    zangler = zangle * np.pi / 180.0 # radians
    zrotation_matrix = np.zeros((3,3),dtype=np.float64)
    # new x axis
    zrotation_matrix[0][0] = np.cos(zangler)
    zrotation_matrix[0][1] = np.sin(zangler)
    zrotation_matrix[0][2] = 0.0
    # new y axis
    zrotation_matrix[1][0] = -np.sin(zangler)
    zrotation_matrix[1][1] = np.cos(zangler)
    zrotation_matrix[1][2] = 0.0
    # new z axis
    zrotation_matrix[2][0] = 0.0
    zrotation_matrix[2][1] = 0.0
    zrotation_matrix[2][2] = 1.0
#
# Combine rotations
#
    final_rotation = rotation_matrix@zrotation_matrix
    return final_rotation.flatten()
    
    
def make_beam(beam_type, E0, kk, kt_by_kz = 0.3, order = 0, gouy = 0, w0 = 1.2e-6, jones = None, rotation = None, translation = None, numkpoints = 20):
    """
    Function to build a beam and return the structure.
    """
    mybeam = BEAM()
    mybeam.beamtype = beam_type
    mybeam.E0 = E0
    mybeam.k = kk
    kz = kk / np.sqrt(1+kt_by_kz**2)
    kt = kt_by_kz*kz
    mybeam.kz = kz
    mybeam.kt = kt
    mybeam.kt_by_kz = kt_by_kz
    mybeam.order = order
    mybeam.w0 = w0
    mybeam.gouy = gouy
    mybeam.numkpoints = numkpoints # should be even
#
# Build a Jones matrix
#
    if jones is None:
        jones_matrix = get_jones_matrix(POLARISATION_X)
        mybeam.jones = np.ctypeslib.as_ctypes(jones_matrix)
    else:
        mybeam.jones = np.ctypeslib.as_ctypes(jones)
#
# Build a rotation matrix
#
    if rotation is None:
        rotation_matrix = get_rotation_matrix(0.0,0.0)
        print("Rotation: ",rotation_matrix)
        mybeam.rotation = np.ctypeslib.as_ctypes(rotation_matrix)
    else:
        mybeam.rotation = np.ctypeslib.as_ctypes(rotation)
#
# Beam position vector
#
    if translation is None:
        beamposition = np.array((0.0,0.0,0.0),dtype=np.float64) # specify position in metres
        mybeam.translation = np.ctypeslib.as_ctypes(beamposition)
    else:
        mybeam.translation = np.ctypeslib.as_ctypes(translation)

    return mybeam
    
    
#=======================================================================
# Create the whole beam collection from the yaml configuration
#=======================================================================

def create_beam_collection(wavelength,beaminfo):
    """
    Function to create a set of beams from the beaminfo dictionary.
    In absence of information, beam will default to Gaussian CPS,
    X-polarised, with w0=wavelength, no rotation or translation.
    input:
      wavelength (float): needed for computing w0 and k's.
      beaminfo (dict): contains all pertinent information for
        defining a set of beams.  Any missing information will be
        replaced by defaults.
    output: beam_collection (Ctypes struct array): an array of beams.
    """
    BeamTypes = {"BEAMTYPE_PLANE": 0, "BEAMTYPE_GAUSS_BARTON5": 1, "BEAMTYPE_GAUSS_CSP": 2, "BEAMTYPE_BESSEL": 3, "BEAMTYPE_LAGUERRE_GAUSSIAN": 4}
    JonesTypes = {"POLARISATION_X":0, "POLARISATION_Y":1, "POLARISATION_RCP":2, "POLARISATION_LCP":3, "POLARISATION_RADIAL":4, "POLARISATION_AZIMUTHAL":5}

    n_beams = len(beaminfo)
    beam_collection = np.zeros(n_beams,dtype=object)
    
    i=0
    for newbeam in beaminfo:
        beam = beaminfo[newbeam]
        print('Creating beam: ',newbeam)
        beamtypestr = beam.get('beamtype','BEAMTYPE_GAUSS_CSP')
        beamtype = BeamTypes[beamtypestr]
        E0 = float(beam.get('E0',1.0))
        kk = 2*np.pi / wavelength
        kt_by_kz = float(beam.get('kt_by_kz',0.3))
        order = int(beam.get('order',1))
        gouy = int(beam.get('gouy',0))
        w0rel = float(beam.get('w0',1.0))
        w0 = wavelength*w0rel
        jonestypestr = beam.get('jones','POLARISATION_X')
        jones_matrix = get_jones_matrix(JonesTypes[jonestypestr])
        #
        rotation = beam.get('rotation',"0.0 0.0")
        fields = rotation.split(" ")
        if fields[0]=="None":
            rotation_matrix = get_rotation_matrix(0.0,0.0)
        else:
            beamrotation = np.array((0.0,0.0),dtype=np.float64)
            for j in range(min(len(fields),2)):
                beamrotation[j] = float(fields[j])
            rotation_matrix = get_rotation_matrix(beamrotation[0],beamrotation[1])
        #
        translation = beam.get('translation',"0.0 0.0 0.0")
        fields = translation.split(" ")
        if fields[0]=="None":
            beamposition = np.array((0.0,0.0,0.0),dtype=np.float64)
        else:
            beamposition = np.array((0.0,0.0,0.0),dtype=np.float64)
            for j in range(min(len(fields),3)):
                beamposition[j] = float(fields[j])
    
        numkpoints = int(beam.get('numkpoints',20))

        beam_collection[i] = make_beam(beamtype, E0, kk, kt_by_kz=kt_by_kz, order=order, gouy=gouy, w0=w0, jones=jones_matrix, rotation=rotation_matrix, translation=beamposition, numkpoints=numkpoints)

        i+=1
    return beam_collection
#=======================================================================



    
    
#
# Wrapper functions
#
def all_incident_fields(positions, the_beams, EE):
    """
    positions: x, y, z coordinates of point (double precision);
    the_beams: is a beam collection array;
    EE: a complex array to receive the electric fields (Ex, Ey, Ez).
    """
    x = positions[0];
    y = positions[1];
    z = positions[2];
    EE[0] = 0 + 0j
    EE[1] = 0 + 0j
    EE[2] = 0 + 0j
    nn = len(the_beams)
    #print(nn)
    dEE = np.zeros(6,dtype=np.float64)
    for i in range(nn):
        Beams.compute_fields(x, y, z, the_beams[i], dEE)
        for j in range(3):
            EE[j] += complex(dEE[j*2],dEE[j*2+1])
    return

def all_incident_field_gradients(positions, the_beams, gradEE):
    """
    positions: x, y, z coordinates of point (double precision);
    the_beams: is a beam collection array;
    gradEE: a complex array to receive the electric fields (Ex, Ey, Ez).
    """
    x = positions[0];
    y = positions[1];
    z = positions[2];
    for i in range(3):
        for j in range(3):
            gradEE[i][j] = 0 + 0j;
    nn = len(the_beams)
    #print(nn)
    dgradEE = np.zeros(18,dtype=np.float64)
    for i in range(nn):
        Beams.compute_field_gradients(x, y, z, the_beams[i], dgradEE)
        for j in range(3):
            for l in range(3):
                gradEE[j][l] += complex(dgradEE[(j*3+l)*2],-dgradEE[(j*3+l)*2+1]) # conjugate
    return

#
# General field function interfaces
#
compute_fields = Beams.compute_fields
compute_fields.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(BEAM), numpy.ctypeslib.ndpointer(dtype=np.float64, ndim=1, shape=(6), flags='C_CONTIGUOUS'),]
#compute_fields.restype = numpy.ctypeslib.ndpointer(dtype=np.float64, ndim=1, shape=(6), flags='C_CONTIGUOUS')
compute_field_gradients = Beams.compute_field_gradients
compute_field_gradients.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(BEAM), numpy.ctypeslib.ndpointer(dtype=np.float64, ndim=1, shape=(18), flags='C_CONTIGUOUS'),]
#
# Beam function interfaces
#
"""general_bessel = Beams.general_bessel
general_bessel.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(BEAM)]
#general_bessel.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, numpy.ctypeslib.ndpointer(dtype=complex, ndim=1, shape=(2), flags='C_CONTIGUOUS'), ctypes.c_int, ctypes.POINTER(BEAM)]
general_bessel.restype = ctypes.POINTER(ctypes.c_double)

general_bessel_gradient = Beams.general_bessel_gradient
general_bessel_gradient.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(BEAM)]
general_bessel_gradient.restype = ctypes.POINTER(ctypes.c_double)
"""
"""gaussian_xpol = Beams.gaussian_xpol
gaussian_xpol.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(BEAM)]
gaussian_xpol.restype = ctypes.POINTER(ctypes.c_double)

gaussian_xpol_gradient = Beams.gaussian_xpol_gradient
gaussian_xpol_gradient.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(BEAM)]
gaussian_xpol_gradient.restype = ctypes.POINTER(ctypes.c_double)

plane_wave = Beams.plane_wave
plane_wave.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(BEAM)]
plane_wave.restype = ctypes.POINTER(ctypes.c_double)

plane_wave_gradient = Beams.plane_wave_gradient
plane_wave_gradient.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(BEAM)]
plane_wave_gradient.restype = ctypes.POINTER(ctypes.c_double)
"""
