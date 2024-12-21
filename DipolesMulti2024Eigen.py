# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:59:31 2018
This version using multiple dipoles per sphere.
@author: Chaoyi Zhang
"""

import sys
from operator import pos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from cycler import cycler
import cmath
import itertools as it
from numpy.core import numerictypes
from scipy.special import j0, j1, jvp, jv
from numpy import sin, cos, pi, arctan2
import time
import Beams
import Dipoles
import Output
import Particles
import ReadYAML
import Display
import ctypes
import datetime
import os.path
import yaml

import matplotlib.animation as animation



def optical_force(grad, p):
    """Calulates the optical force from the TRANSPOSE of the gradient of the  field"""
    Force = np.zeros(3)
    Force[0] = (1 / 2) * np.real(
        p[0] * grad[0, 0] + p[1] * grad[0, 1] + p[2] * grad[0, 2]
    )
    Force[1] = (1 / 2) * np.real(
        p[0] * grad[1, 0] + p[1] * grad[1, 1] + p[2] * grad[1, 2]
    )
    Force[2] = (1 / 2) * np.real(
        p[0] * grad[2, 0] + p[1] * grad[2, 1] + p[2] * grad[2, 2]
    )
    return Force


def func1(a, b, r):
    """
    Non-diagonal elements of matrix A_jk
    """
    C = ((a * b) / (r ** 2)) * (k ** 2 + (3 * ((1j * k * r) - 1) / (r ** 2)))

    return C


def func2(a, r):
    """
    Diagonal elements of matrix A_jk
    """
    #print("k: ",k," a: ",a," r: ",r)
    return (k ** 2) * (((a ** 2) / (r ** 2)) - 1) + (((1j * k * r) - 1) / (r ** 2)) * (
        ((3 * (a ** 2)) / (r ** 2)) - 1
    )


def Ajj(polarisability):
    """
    A_jj matrix; diagonal elements of big matrix of matrices
    """
    ajj = 1 / polarisability
    #print(ajj)
    A = np.zeros([3, 3], dtype=complex)
    np.fill_diagonal(A, ajj)
    #print(A)
    return A


def Ajk(x, y, z, r):
    """
    A_jk matrix; off-diagonal elements of big matrix
    """
    A = np.zeros([3, 3], dtype=complex)
    A[0][0] = func2(x, r)
    A[1][1] = func2(y, r)
    A[2][2] = func2(z, r)
    A[0][1] = func1(x, y, r)
    A[1][0] = A[0][1]
    A[0][2] = func1(x, z, r)
    A[2][0] = A[0][2]
    A[1][2] = func1(y, z, r)
    A[2][1] = A[1][2]
    # *(1/(4*np.pi*8.85e-12))

    return ((cmath.exp(1j * k * r)) / (r)) * A * (1 / (4 * np.pi * 8.85e-12))


def func3(a, r):

    return ((a ** 2) / (r ** 2)) + 1


def func4(a, b, r):

    return (a * b) / (r ** 2)


def Djj(dipole_radius):  # For Diffusion
    #
    # This is valid for a sphere, but not other shapes e.g. a torus
    # This will need to be changed when considering the dynamics of ither particle shapes
    #
    djj = (k_B * temperature) / (6 * np.pi * viscosity * dipole_radius)
    D = np.zeros([3, 3])
    np.fill_diagonal(D, djj)

    return D


def Djk(x, y, z, r):
    D = np.zeros([3, 3])
    D[0][0] = func3(x, r)
    D[1][1] = func3(y, r)
    D[2][2] = func3(z, r)
    D[0][1] = func4(x, y, r)
    D[1][0] = D[0][1]
    D[0][2] = func4(x, z, r)
    D[2][0] = D[0][2]
    D[1][2] = func4(y, z, r)
    D[2][1] = D[1][2]

    return ((k_B * temperature) / (8 * np.pi * viscosity * r)) * D


"""
def buckingham_force(Hamaker, constant1, constant2, r, radius_i, radius_j):
    r_max = 1.1 * (radius_i +radius_j)
    r_abs = np.linalg.norm(r)
    if r_abs < r_max:
        print("Eeek!! r_abs = ", r_abs)
        r_abs = r_max  # capping the force

    force = np.array(
        [
            -(
                constant1 * constant2 * np.exp(-constant2 * r_abs)
                - (
                    (32 * Hamaker * (dipole_radius ** 6))
                    / (3 * (r_abs ** 3) * (r_abs ** 2 - 4 * (dipole_radius ** 2)) ** 2)
                )
            )
            * (r[i] / r_abs)
            for i in range(3)
        ]
    )
    # force = np.zeros(3)
    # force[0] = -(constant1*constant2*np.exp(-constant2*r_abs) - ((32*Hamaker*(dipole_radius**6))/(3*(r_abs**3)*(r_abs**2 - 4*(dipole_radius**2))**2)))*(r[0]/r_abs)
    # force[1] = -(constant1*constant2*np.exp(-constant2*r_abs) - ((32*Hamaker*(dipole_radius**6))/(3*(r_abs**3)*(r_abs**2 - 4*(dipole_radius**2))**2)))*(r[1]/r_abs)
    # force[2] = -(constant1*constant2*np.exp(-constant2*r_abs) - ((32*Hamaker*(dipole_radius**6))/(3*(r_abs**3)*(r_abs**2 - 4*(dipole_radius**2))**2)))*(r[2]/r_abs)

    return force
"""

def buckingham_force(Hamaker, constant1, constant2, r, radius_i, radius_j):
    r_max = 1.1 * (radius_i +radius_j)
    r_abs = np.linalg.norm(r)
    if r_abs < r_max:
        #
        # NOTE; Temporary while bug fixing
        #
        #print("Eeek!! r_abs = ", r_abs)
        r_abs = r_max  # capping the force

    ###
    ### NOTE; WILL NEED REWORKING FOR BETTER RADIUS MANAGEMENT
    ###
    radius_avg = (radius_i+radius_j)/2.0
    force = np.array(
        [
            -(
                constant1 * constant2 * np.exp(-constant2 * r_abs)
                - (
                    (32 * Hamaker * (radius_avg ** 6))
                    / (3 * (r_abs ** 3) * (r_abs ** 2 - 4 * (radius_avg ** 2)) ** 2)
                )
            )
            * (r[i] / r_abs)
            for i in range(3)
        ]
    )
    # force = np.zeros(3)
    # force[0] = -(constant1*constant2*np.exp(-constant2*r_abs) - ((32*Hamaker*(dipole_radius**6))/(3*(r_abs**3)*(r_abs**2 - 4*(dipole_radius**2))**2)))*(r[0]/r_abs)
    # force[1] = -(constant1*constant2*np.exp(-constant2*r_abs) - ((32*Hamaker*(dipole_radius**6))/(3*(r_abs**3)*(r_abs**2 - 4*(dipole_radius**2))**2)))*(r[1]/r_abs)
    # force[2] = -(constant1*constant2*np.exp(-constant2*r_abs) - ((32*Hamaker*(dipole_radius**6))/(3*(r_abs**3)*(r_abs**2 - 4*(dipole_radius**2))**2)))*(r[2]/r_abs)

    return force


def spring_force(constant1, r, dipole_radius):
    #print("Dipole Radius:",dipole_radius)
    r_abs = np.linalg.norm(r)
    force = [constant1 * (r_abs - 2 * dipole_radius) * (r[i] / r_abs) for i in range(3)]
    # force = np.zeros(3)
    # force[0] = constant1*(r_abs-2*dipole_radius)*(r[0]/r_abs)
    # force[1] = constant1*(r_abs-2*dipole_radius)*(r[1]/r_abs)
    # force[2] = constant1*(r_abs-2*dipole_radius)*(r[2]/r_abs)

    return force


def driving_force(constant1, r):
    #print("Dipole Radius:",dipole_radius)
    r_abs = np.linalg.norm(r)
    force = [constant1 * (r_abs - 2 * dipole_radius) * (r[i] / r_abs) for i in range(3)]
    # force = np.zeros(3)
    # force[0] = constant1*(r_abs-2*dipole_radius)*(r[0]/r_abs)
    # force[1] = constant1*(r_abs-2*dipole_radius)*(r[1]/r_abs)
    # force[2] = constant1*(r_abs-2*dipole_radius)*(r[2]/r_abs)

    return force


def bending_force(bond_stiffness, ri, rj, rk):
    rij = rj - ri
    rik = rk - ri
    rij_abs = np.linalg.norm(rij)
    rik_abs = np.linalg.norm(rik)
    rijrik = rij_abs * rik_abs
    rij2 = rij_abs * rij_abs
    rik2 = rik_abs * rik_abs
    costhetajik = np.dot(rij, rik) / rijrik
    force = np.zeros([3, 3])
    i = 1
    force[i] = bond_stiffness * (
        (rik + rij) / rijrik - costhetajik * (rij / rij2 + rik / rik2)
    )
    force[i - 1] = bond_stiffness * (costhetajik * rij / rij2 - rik / rijrik)
    force[i + 1] = bond_stiffness * (costhetajik * rik / rik2 - rij / rijrik)
    #    print(force)
    return force


def displacement_matrix(array_of_positions):
    number_of_dipoles = len(array_of_positions)
    #print("x",number_of_dipoles)
    list_of_displacements = [u - v for u, v in it.combinations(array_of_positions, 2)]
    array_of_displacements = np.zeros(len(list_of_displacements), dtype=object)
    for i in range(len(list_of_displacements)):
        array_of_displacements[i] = list_of_displacements[i]
    displacement_matrix = np.zeros([number_of_dipoles, number_of_dipoles], dtype=object)
    iu = np.triu_indices(number_of_dipoles, 1)
    displacement_matrix[iu] = array_of_displacements
    displacement_matrix.T[iu] = -array_of_displacements
    return displacement_matrix


def dipole_moment_array(array_of_positions, E0, dipole_radius, number_of_dipoles_in_primitive):

    list_of_displacements = [u - v for u, v in it.combinations(array_of_positions, 2)]
    number_of_displacements = len(list_of_displacements)
    array_of_displacements = np.zeros(number_of_displacements, dtype=object)

    for i in range(number_of_displacements):
        array_of_displacements[i] = list_of_displacements[i]
    array_of_distances = np.array([np.linalg.norm(w) for w in array_of_displacements])

    number_of_dipoles = len(array_of_positions)
    #print("y",number_of_dipoles)

    E_array = np.zeros(number_of_dipoles, dtype=object)

    # initialize array of external electric field vectors
    A_matrix = np.zeros([number_of_dipoles, number_of_dipoles], dtype=object)
    Ajk_array = np.zeros(
        number_of_displacements, dtype=object
    )  # initialize array to store A_jk matrices
    Ajj_array = np.zeros(
        number_of_dipoles, dtype=object
    )  # initialize array to store A_jj matrices
    iu = np.triu_indices(number_of_dipoles, 1)
    di = np.diag_indices(number_of_dipoles)

    for i in range(
        number_of_displacements
    ):  # for loop that goes over every x,y,z value of displacement
        Ajk_array[i] = Ajk(
            array_of_displacements[i][0],
            array_of_displacements[i][1],
            array_of_displacements[i][2],
            array_of_distances[i],
        )
    for i in range(number_of_dipoles):  # creates D_jj matrices
        ii = i//number_of_dipoles_in_primitive
        Ajj_array[i] = Ajj(polarizability[ii])
        # E_array[i] = np.array(
        #     [
        #         0,
        #         -YPOL * 2 * 1j * E0 * np.cos(k * array_of_positions[i][0]),
        #         ZPOL * 2 * E0 * np.cos(k * array_of_positions[i][0]),
        #     ]
        # )
#        E_array[i] = incident_field(beam, array_of_positions[i])
        E = np.zeros(3,dtype=np.complex128)
        Beams.all_incident_fields(array_of_positions[i], beam_collection, E)
        E_array[i] = E


    A_matrix[iu] = Ajk_array
    A_matrix.T[iu] = A_matrix[iu]
    A_matrix[di] = Ajj_array
    temporary_array = np.zeros(number_of_dipoles, dtype=object)
    for i in range(number_of_dipoles):
        temporary_array[i] = np.concatenate(A_matrix[i])
    A = np.hstack(temporary_array)
#    print("A matrix",A)
    E = np.hstack(E_array)  # merges the array of E field vectors to form 3Nx1 vector

    P_list = np.hsplit(np.linalg.solve(A, E), number_of_dipoles)
    P_array = np.zeros(number_of_dipoles, dtype=object)
    for i in range(number_of_dipoles):
        P_array[i] = P_list[i]

    return P_array


def optical_force_array(array_of_particles, E0, dipole_radius, dipole_primitive):
#
# Need to:
# (0) change array_of_positions to array of particle positions
# (1) generate a full list of dipole positions
# (2) do the optical force calculation on the full list
# (3) construct the forces on the individual particles
#
# (0):
    number_of_particles = len(array_of_particles)
    number_of_dipoles_in_primitive = len(dipole_primitive)
    number_of_dipoles = number_of_particles*number_of_dipoles_in_primitive
    #print(number_of_particles,number_of_dipoles_in_primitive,number_of_dipoles)
# (1):
    array_of_positions = np.zeros((number_of_dipoles, 3))
    for i in range(number_of_particles):
        for j in range(number_of_dipoles_in_primitive):
            array_of_positions[i*number_of_dipoles_in_primitive+j] = array_of_particles[i] + dipole_primitive[j]
    #print(array_of_positions)
# (2):
    p_array = dipole_moment_array(array_of_positions, E0, dipole_radius, number_of_dipoles_in_primitive)
    # print(p_array)
    displacements_matrix = displacement_matrix(array_of_positions)

    grad_matrix = np.zeros([number_of_dipoles, number_of_dipoles], dtype=object)
    for i in range(number_of_dipoles):
        for j in range(number_of_dipoles):
            if i == j:
                grad_matrix[i][j] = 0
            else:
                grad_matrix[i][j] = Dipoles.py_grad_E_cc(displacements_matrix[i][j], p_array[i], k)
    grad_matrix_T = np.transpose(grad_matrix)
    #print("Array of particles:",array_of_positions)
    #print("Displacements of particles:",displacements_matrix)
    #print("dipole vectors:",p_array)
    #print("Gradient matrix:",grad_matrix)
    #print("Gradient matrix T:",grad_matrix_T)
    optical_force_matrix = np.zeros(
        [number_of_dipoles, number_of_dipoles], dtype=object
    )
    for i in range(number_of_dipoles):
        for j in range(number_of_dipoles):
            if i == j:
                optical_force_matrix[i][j] = np.zeros(3)
            else:
                optical_force_matrix[i][j] = optical_force(
                    grad_matrix_T[i][j], p_array[i]  # TRANSPOSE INPUT!!!!
                )

    optical_force_array_scat = np.sum(optical_force_matrix, axis=1)

    grad_E_inc = np.zeros(number_of_dipoles, dtype=object)
    optical_force_array_inc = np.zeros(number_of_dipoles, dtype=object)
    for i in range(number_of_dipoles):
        gradE = np.zeros((3,3),dtype=np.complex128)
        Beams.all_incident_field_gradients(array_of_positions[i], beam_collection, gradE)
        grad_E_inc[i] = gradE
#        grad_E_inc[i] = incident_field_gradient(beam, array_of_positions[i])
        optical_force_array_inc[i] = optical_force(
            np.transpose(grad_E_inc[i]), p_array[i]
        )

    optical_force_array_tot = optical_force_array_scat + optical_force_array_inc
# (3):
    final_optical_forces = np.zeros(number_of_particles, dtype=object)
    for i in range(number_of_particles):
        final_optical_forces[i] = np.sum(optical_force_array_tot[i*number_of_dipoles_in_primitive:(i+1)*number_of_dipoles_in_primitive],axis=0)
    if excel_output==True and include_couple==True:
        couples = np.zeros((n_particles,3),dtype=np.double)
        a0conj = a0.conjugate()
        p_arrayconj = p_array.conjugate()
        #print(p_array,p_arrayconj)
        for i in range(number_of_particles):
            for j in range(number_of_dipoles_in_primitive):
                ij = i*number_of_dipoles_in_primitive+j
                #print(i,j,ij,p_array[ij])
                couples[i,0]+=0.5*np.real((p_array[ij][1]*p_arrayconj[ij][2]-p_array[ij][2]*p_arrayconj[ij][1])/a0conj[i])
                couples[i,1]+=0.5*np.real((p_array[ij][2]*p_arrayconj[ij][0]-p_array[ij][0]*p_arrayconj[ij][2])/a0conj[i])
                couples[i,2]+=0.5*np.real((p_array[ij][0]*p_arrayconj[ij][1]-p_array[ij][1]*p_arrayconj[ij][0])/a0conj[i])
        """for i in range(number_of_particles):
            for j in range(number_of_dipoles_in_primitive):
                couples[i,0]+=dipole_primitive[j][1]*optical_force_array_tot[i*number_of_dipoles_in_primitive+j][2]-dipole_primitive[j][2]*optical_force_array_tot[i*number_of_dipoles_in_primitive+j][1]
                couples[i,1]+=dipole_primitive[j][2]*optical_force_array_tot[i*number_of_dipoles_in_primitive+j][0]-dipole_primitive[j][0]*optical_force_array_tot[i*number_of_dipoles_in_primitive+j][2]
                couples[i,2]+=dipole_primitive[j][0]*optical_force_array_tot[i*number_of_dipoles_in_primitive+j][1]-dipole_primitive[j][1]*optical_force_array_tot[i*number_of_dipoles_in_primitive+j][0]
        """
    else:
        couples=None
    #print(optical_force_array_tot)
    #print(final_optical_forces)
    return final_optical_forces,couples

"""
#
# LEGACY -> BEFORE SHAPE,ARGS CHANGE
#
def buckingham_force_array(array_of_positions, dipole_radius):
    number_of_dipoles = len(array_of_positions)
    displacements_matrix = displacement_matrix(array_of_positions)
    displacements_matrix_T = np.transpose(displacements_matrix)
    #    Hamaker = (np.sqrt(30e-20) - np.sqrt(4e-20))**2
    Hamaker = 0
    ConstantA = 1.0e23
    ConstantB = 2.0e8  # 4.8e8
    buckingham_force_matrix = np.zeros(
        [number_of_dipoles, number_of_dipoles], dtype=object
    )
    for i in range(number_of_dipoles):
        for j in range(number_of_dipoles):
            if i == j:
                buckingham_force_matrix[i][j] = [0, 0, 0]
            else:
                buckingham_force_matrix[i][j] = buckingham_force(
                    Hamaker,
                    ConstantA,
                    ConstantB,
                    displacements_matrix_T[i][j],
                    dipole_radius,
                )
    buckingham_force_array = np.zeros((number_of_dipoles,3),dtype=np.float64)
    temp = np.sum(buckingham_force_matrix, axis=1)
    for i in range(number_of_dipoles):
        buckingham_force_array[i] = temp[i]
    return buckingham_force_array
"""


def buckingham_force_array(array_of_positions, effective_radii):
    number_of_dipoles = len(array_of_positions)
    displacements_matrix = displacement_matrix(array_of_positions)
    displacements_matrix_T = np.transpose(displacements_matrix)
    #    Hamaker = (np.sqrt(30e-20) - np.sqrt(4e-20))**2
    Hamaker = 0
    ConstantA = 1.0e23
    ConstantB = 2.0e8  # 4.8e8
    buckingham_force_matrix = np.zeros(
        [number_of_dipoles, number_of_dipoles], dtype=object
    )
    #
    # ?? Maybe rename 'dipoles' to 'particles' in here, as just used for whole particles now
    #


    for i in range(number_of_dipoles):
        for j in range(number_of_dipoles):
            if i == j:
                buckingham_force_matrix[i][j] = [0, 0, 0]
            else:
                buckingham_force_matrix[i][j] = buckingham_force(
                    Hamaker,
                    ConstantA,
                    ConstantB,
                    displacements_matrix_T[i][j],
                    effective_radii[i],
                    effective_radii[j]
                )
    buckingham_force_array = np.zeros((number_of_dipoles,3),dtype=np.float64)
    temp = np.sum(buckingham_force_matrix, axis=1)
    for i in range(number_of_dipoles):
        buckingham_force_array[i] = temp[i]
    return buckingham_force_array


def spring_force_array(array_of_positions, dipole_radius):
    number_of_dipoles = len(array_of_positions)
    displacements_matrix = displacement_matrix(array_of_positions)
    displacements_matrix_T = np.transpose(displacements_matrix)
    stiffness = 1.0e-5
    spring_force_matrix = np.zeros([number_of_dipoles, number_of_dipoles], dtype=object)
    ##
    ## Possible double zeroing?
    ##
    for i in range(number_of_dipoles):
        for j in range(number_of_dipoles):
            spring_force_matrix[i][j] = np.zeros(3)
    for i in range(0,number_of_dipoles,2):
        j = i + 1
        spring_force_matrix[i][j] = spring_force(stiffness, displacements_matrix_T[i][j], dipole_radius)
    for i in range(1,number_of_dipoles,2):
        j = i - 1
        spring_force_matrix[i][j] = spring_force(stiffness, displacements_matrix_T[i][j], dipole_radius)
    #print("Spring force array shape before:",spring_force_matrix.shape)
    spring_force_array = np.sum(spring_force_matrix, axis=1)
    #    print("Springs",spring_force_array)
    #print("Spring force array shape after:",spring_force_array.shape)
    return spring_force_array


def driving_force_array(array_of_positions):
    number_of_dipoles = len(array_of_positions)
    displacements_matrix = displacement_matrix(array_of_positions)
    displacements_matrix_T = np.transpose(displacements_matrix)
    driver = 3.0e-7#6
    driving_force_array = np.zeros(number_of_dipoles, dtype=object)
    for i in range(0,number_of_dipoles,2):
        j = i+1
        driving_force_array[i] = driving_force(driver, displacements_matrix_T[i][j])
        driving_force_array[j] = driving_force_array[i]
    return driving_force_array


def bending_force_array(array_of_positions, dipole_radius):
    number_of_dipoles = len(array_of_positions)
    bond_stiffness = BENDING
    bending_force_matrix = np.zeros([number_of_dipoles], dtype=object)
    bending_force_temp = np.zeros([3], dtype=object)
    for i in range(1, number_of_dipoles - 1):
        bending_force_temp = bending_force(
            bond_stiffness,
            array_of_positions[i],
            array_of_positions[i - 1],
            array_of_positions[i + 1],
        )
        #        print(bending_force_temp)
        bending_force_matrix[i - 1] += bending_force_temp[0]
        bending_force_matrix[i] += bending_force_temp[1]
        bending_force_matrix[i + 1] += bending_force_temp[2]
    #    print("Springs",spring_force_array)
    return bending_force_matrix
    
    
def gravity_force_array(array_of_positions, dipole_radius):
    number_of_beads = len(array_of_positions)
    bond_stiffness = BENDING
    gravity_force_matrix = np.zeros([number_of_beads], dtype=object)
    gravity_force_temp = np.zeros([3], dtype=object)
    gravity_force_temp[2] = -1e-12#-9.81*mass
    for i in range(number_of_beads):
        gravity_force_matrix[i] = gravity_force_temp
    return gravity_force_matrix


def diffusion_matrix(array_of_positions, dipole_radius):
    # positions of particle centres
    # dipole_radius is actually considering the spehre radius here
    list_of_displacements = [u - v for u, v in it.combinations(array_of_positions, 2)]
    array_of_displacements = np.zeros(len(list_of_displacements), dtype=object)
    for i in range(len(list_of_displacements)):
        array_of_displacements[i] = list_of_displacements[i]
    array_of_distances = np.array([np.linalg.norm(w) for w in array_of_displacements])
    number_of_dipoles = len(array_of_positions)
    number_of_displacements = len(array_of_displacements)
    D_matrix = np.zeros([number_of_dipoles, number_of_dipoles], dtype=object)
    Djk_array = np.zeros(
        number_of_displacements, dtype=object
    )  # initialize array to store D_jk matrices
    Djj_array = np.zeros(
        number_of_dipoles, dtype=object
    )  # initialize array to store D_jj matrices
    iu = np.triu_indices(number_of_dipoles, 1)
    di = np.diag_indices(number_of_dipoles)
    for i in range(number_of_displacements):
        Djk_array[i] = Djk(
            array_of_displacements[i][0],
            array_of_displacements[i][1],
            array_of_displacements[i][2],
            array_of_distances[i],
        )
    for i in range(number_of_dipoles):
        Djj_array[i] = Djj(dipole_radius)
    D_matrix[iu] = Djk_array
    D_matrix.T[iu] = D_matrix[iu]
    D_matrix[di] = Djj_array
    temporary_array = np.zeros(number_of_dipoles, dtype=object)
    for i in range(number_of_dipoles):
        temporary_array[i] = np.concatenate(D_matrix[i])
    D = np.hstack(temporary_array)
    return D



def sphere_size(args, dipole_radius):
    #
    # Counts number of points in object
    #
    dipole_diameter = 2*dipole_radius
    sphere_radius = args[0]
    dd2 = dipole_diameter**2
    sr2 = sphere_radius**2
    print(sphere_radius,dipole_radius)
    num = int(sphere_radius//dipole_diameter)
    
    number_of_dipoles = 0
    for i in range(-num,num+1):
        i2 = i*i
        for j in range(-num,num+1):
            j2 = j*j
            for k in range(-num,num+1):
                k2 = k*k
                rad2 = (i2+j2+k2)*dd2
                if rad2 < sr2:
                    number_of_dipoles += 1
    return number_of_dipoles

def sphere_positions(args, dipole_radius, number_of_dipoles_total):
    #
    # With pts size known now, particles are added to this array
    #
    dipole_diameter = 2*dipole_radius
    sphere_radius = args[0]
    dd2 = dipole_diameter**2
    sr2 = sphere_radius**2
    pts = np.zeros((number_of_dipoles_total, 3))
    number_of_dipoles = 0
    num = int(sphere_radius//dipole_diameter)
    for i in range(-num,num+1):
        i2 = i*i
        x = i*dipole_diameter
        for j in range(-num,num+1):
            j2 = j*j
            y = j*dipole_diameter
            for k in range(-num,num+1):
                k2 = k*k
                z = k*dipole_diameter
                rad2 = (i2+j2+k2)*dd2
                if rad2 < sr2:
                    pts[number_of_dipoles][0] = x+1e-20     # Softening factor
                    pts[number_of_dipoles][1] = y+1e-20     #
                    pts[number_of_dipoles][2] = z
                    number_of_dipoles += 1
    print(number_of_dipoles," dipoles generated")
    return pts


def torus_sector_size(args, dipole_radius):
    #
    # Only supports torus flat in XY plane
    # torus_centre_radius = distance from origin to centre of tube forming the torus
    # torus_tube_radius = radius of the tube cross section of the torus
    # phi_lower = smaller angle in XY plane, from positive X axis, to start torus sector from (0,2*PI)
    # phi_upper = larger angle in XY plane, from positive X axis, to end torus sector at (0,2*PI)
    #
    # ** Could be extended to be tilted
    # ** Could also move x,y,z calcualtion into if to speed up program -> reduce waste on non-dipole checks
    #
    torus_centre_radius, torus_tube_radius, phi_lower, phi_upper = args
    dipole_diameter = 2*dipole_radius
    dd2 = dipole_diameter**2
    ttr2 = torus_tube_radius**2
    print(torus_centre_radius, torus_tube_radius, dipole_radius)
    num_xy = int( (torus_tube_radius+torus_centre_radius)//dipole_diameter)     #Number of dipoles wide in each direction (XY, wide directions)
    num_z  = int( torus_centre_radius//dipole_diameter)                         #Number of dipoles tall (shorter)
    #x_shift = torus_centre_radius*np.cos( (phi_lower+phi_upper)/2.0 )
    #y_shift = torus_centre_radius*np.sin( (phi_lower+phi_upper)/2.0 )
    #
    # Counts number of points in object
    #
    number_of_dipoles = 0

    for i in range(-num_xy,num_xy+1):
        i2 = i*i
        for j in range(-num_xy,num_xy+1):
            j2 = j*j
            phi = arctan2(j,i);

            phi = phi%(2.0*np.pi)
            phi_lower = phi_lower%(2.0*np.pi)
            phi_upper = phi_upper%(2.0*np.pi)
            withinBounds = False
            if(phi_lower <= phi_upper):
                withinBounds = ( (phi_lower < phi) and (phi < phi_upper) )
            else:
                withinBounds = ( (phi_lower < phi) or (phi < phi_upper) )
            
            if(withinBounds):
                for k in range(-num_z,num_z+1):
                    k2 = k*k
                    rad_xy_2 = (i2 + j2)*dd2
                    #pow( centre_R-sqrt( pow(point.x,2) + pow(point.y,2) ) ,2) +pow(point.z,2) <= pow(tube_R,2)
                    if (torus_centre_radius -np.sqrt(rad_xy_2))**2 +k2*dd2 < ttr2:
                        number_of_dipoles += 1

    return number_of_dipoles

def torus_sector_positions(args, dipole_radius, number_of_dipoles_total):
    #
    # Only supports torus flat in XY plane
    # torus_centre_radius = distance from origin to centre of tube forming the torus
    # torus_tube_radius = radius of the tube cross section of the torus
    # phi_lower = smaller angle in XY plane, from positive X axis, to start torus sector from (0,2*PI)
    # phi_upper = larger angle in XY plane, from positive X axis, to end torus sector at (0,2*PI)
    #
    # ** Could be extended to be tilted
    # ** Could also move x,y,z calcualtion into if to speed up program -> reduce waste on non-dipole checks
    #
    torus_centre_radius, torus_tube_radius, phi_lower, phi_upper = args
    dipole_diameter = 2*dipole_radius
    dd2 = dipole_diameter**2
    ttr2 = torus_tube_radius**2
    print(torus_centre_radius, torus_tube_radius, dipole_radius)
    num_xy = int( (torus_tube_radius+torus_centre_radius)//dipole_diameter)     #Number of dipoles wide in each direction (XY, wide directions)
    num_z  = int( torus_centre_radius//dipole_diameter)                         #Number of dipoles tall (shorter)
    x_shift = torus_centre_radius*np.cos( (phi_lower+phi_upper)/2.0 )
    y_shift = torus_centre_radius*np.sin( (phi_lower+phi_upper)/2.0 )

    pts = np.zeros((number_of_dipoles_total, 3))
    number_of_dipoles = 0
    for i in range(-num_xy,num_xy+1):
        i2 = i*i
        x = i*dipole_diameter
        for j in range(-num_xy,num_xy+1):
            j2 = j*j
            y = j*dipole_diameter
            phi = arctan2(j,i);
            
            phi = phi%(2.0*np.pi)
            phi_lower = phi_lower%(2.0*np.pi)
            phi_upper = phi_upper%(2.0*np.pi)
            withinBounds = False
            if(phi_lower <= phi_upper):
                withinBounds = ( (phi_lower < phi) and (phi < phi_upper) )
            else:
                withinBounds = ( (phi_lower < phi) or (phi < phi_upper) )

            if(withinBounds):
                for k in range(-num_z,num_z+1):
                    k2 = k*k
                    z = k*dipole_diameter
                    rad_xy_2 = (i2 + j2)*dd2
                    #pow( centre_R-sqrt( pow(point.x,2) + pow(point.y,2) ) ,2) +pow(point.z,2) <= pow(tube_R,2)
                    if (torus_centre_radius -np.sqrt(rad_xy_2))**2 +k2*dd2 < ttr2:
                        pts[number_of_dipoles][0] = x+1e-20 -x_shift     # Softening factor
                        pts[number_of_dipoles][1] = y+1e-20 -y_shift     #
                        pts[number_of_dipoles][2] = z
                        number_of_dipoles += 1
    print(number_of_dipoles," dipoles generated")
    return pts


def simulation(number_of_particles, positions, shapes, args):
    #
    # shapes = List of shape types used
    # args   = List of arguments about system and particles; [dipole_radius, particle_parameters]
    # particle_parameters; Sphere = [sphere_radius]
    #                      Torus  = [torus_centre_radius, torus_tube_radius]
    #

    ####
    ## NOTE; EEK CURRENTLY DISABLED FOR BUGIXING
    ####

    #    MyFileObject = open('anyfile.txt','w')
    #position_vectors = positions(number_of_particles)
    position_vectors = positions    #Of each particle centre
    print(positions)
    #position_vectors = np.zeros((number_of_particles,3),dtype=np.float64)
    temp = np.zeros(6)
    temp[0] = 1e-8
    temp[3] = 1e-8
    temp[4] = 0e-8
    number_of_timesteps = frames
    number_of_dipoles = len(position_vectors)
    mean = np.zeros(number_of_dipoles * 3)
    vectors_list = []
    vectors_array = np.zeros(number_of_timesteps, dtype=object)
    temp_array1 = np.zeros(number_of_timesteps, dtype=object)
    #
    # Generate a list of dipoles for one sphere
    #

    # Get total number of particles involved over all particles
    dipole_primitive_num = np.zeros(number_of_particles, dtype=int)
    for particle_i in range(number_of_particles):
        match shapes[particle_i]:
            case "sphere":
                dipole_primitive_num[particle_i] = sphere_size(args[particle_i], dipole_radius)
            case "torus":
                dipole_primitive_num[particle_i] = torus_sector_size(args[particle_i], dipole_radius)
    dipole_primitive_num_total = np.sum(dipole_primitive_num);
    # Get dipole primitive positions for each particle
    dipole_primitives = np.zeros( (dipole_primitive_num_total,3), dtype=float)   #Flattened 1D list of all dipoles for all particles
    dpn_start_indices = np.append(0, np.cumsum(dipole_primitive_num[:-1]))
    for particle_i in range(number_of_particles):
        match shapes[particle_i]:
            case "sphere":
                dipole_primitives[dpn_start_indices[particle_i]: dpn_start_indices[particle_i]+dipole_primitive_num[particle_i]] = sphere_positions(args[particle_i], dipole_radius, dipole_primitive_num[particle_i])
            case "torus":
                dipole_primitives[dpn_start_indices[particle_i]: dpn_start_indices[particle_i]+dipole_primitive_num[particle_i]] = torus_sector_positions(args[particle_i], dipole_radius, dipole_primitive_num[particle_i])
    
    if excel_output==True:
        optpos = np.zeros((frames,n_particles,3))
        if include_force==True:
            optforce = np.zeros((frames,n_particles,3))
        else:
            optforce = None
        if include_couple==True:
            optcouple = np.zeros((frames,n_particles,3))
        else:
            optcouple = None

    for i in range(number_of_timesteps):
        #        print("positions: ",position_vectors)
        
        #
        # Pass in list of dipole positions to generate total dipole array;
        # All changes inside optical_force_array().
        #
        #optical,couples = optical_force_array(position_vectors, E0, dipole_radius, dipole_primitive)

        optical, torques, couples = Dipoles.py_optical_force_torque_array(position_vectors, np.asarray(dipole_primitive_num), dipole_radius, dipole_primitives, inverse_polarizability, beam_collection)

        #couples = None
        #include_couple==False
        if excel_output==True:
            for j in range(n_particles):
                for k in range(3):
                    optpos[i,j,k] = position_vectors[j][k]
            if include_force==True:
                for j in range(n_particles):
                    for k in range(3):
                        optforce[i,j,k] = optical[j][k]
            if include_couple==True:
                for j in range(n_particles):
                    for k in range(3):
                        optcouple[i,j,k] = couples[j][k] + torques[j][k]

        if i%10 == 0:
            print("Step ",i)
            print(i,optical)



        # Finds a characteristic radius for each shape to calcualte Buckingham forces
        effective_radii = np.zeros(number_of_particles, dtype=np.float64)
        for i in range(number_of_particles):
            match shapes[i]:
                case "sphere":
                    effective_radii[i] = args[i][0]
                case "torus":
                    effective_radii[i] = (args[i][0] + args[i][1])

        #
        # Call diffusion_matrix with sphere radius not dipole radius
        #

        ##
        ## TODO; NEEDS TO BE PER PARTICLE, Not the same for every particle (as is currently done)
        ##
        D = diffusion_matrix(position_vectors, args[0][0])
        #D = diffusion_matrix(position_vectors, dipole_radius)


        buckingham = buckingham_force_array(position_vectors, effective_radii)
#        spring = spring_force_array(position_vectors, radius)
#        driver = driving_force_array(position_vectors)
#        bending = bending_force_array(position_vectors, radius)
#        gravity = gravity_force_array(position_vectors, radius)
        total_force_array = optical #+ buckingham# + gravity#+ spring #+ driver#+ gravity# + spring + bending
        #        print("buckingham: ",buckingham_force_array(position_vectors,radius))
        #        print("Springs: ",spring_force_array(position_vectors,radius))
        F = np.hstack(total_force_array)
        # print(F)
        cov = 2 * timestep * D
        # print(cov)
        R = np.random.multivariate_normal(mean, cov)
        # print(R)
        SumDijFj = (1 / (k_B * temperature)) * np.dot(D, F)
        # print(SumDijFj*timestep)
        positions_stacked = np.hstack(position_vectors)
        new_positions = positions_stacked + SumDijFj * timestep + R
        #        new_positions = positions_stacked + temp

        #        print("%6.4g" % new_positions[0], "%6.4g" % F[0],"%6.4g" % F[1],"%6.4g" % F[2],"%6.4g" % F[3],"%6.4g" % F[4],"%6.4g" % F[5], sep=', ', file=MyFileObject)

##        print(new_positions)
        new_positions_list = np.hsplit(new_positions, number_of_dipoles)
        new_positions_array = np.zeros((number_of_dipoles,3), dtype=np.float64)
        for j in range(len(new_positions_list)):
            new_positions_array[j] = new_positions_list[j]
        position_vectors = new_positions_array
        vectors_list.append(
            position_vectors
        )  # returns list of position vector arrays of all particles
    for k in range(number_of_timesteps):
        vectors_array[k] = vectors_list[k]
        temp_array1[k] = np.hstack(vectors_array[k])

    xyz_list1 = np.vsplit(np.vstack(temp_array1).T, number_of_particles)

    return xyz_list1,optpos,optforce,optcouple



###################################################################################
# Start of program
###################################################################################

if int(len(sys.argv)) != 2:
    sys.exit("Usage: python {} <FILESTEM>".format(sys.argv[0]))

filestem = sys.argv[1]
filename_vtf = filestem+".vtf"
filename_xl = filestem+".xlsx"
filename_yaml = filestem+".yml"
#===========================================================================
# Read the yaml file into a system parameter dictionary
#===========================================================================
sys_params = ReadYAML.load_yaml(filename_yaml)
print(sys_params)
#===========================================================================
# Parse the sys_params yaml file
#===========================================================================
beaminfo = ReadYAML.read_section(sys_params,'beams')
paraminfo = ReadYAML.read_section(sys_params,'parameters')
optioninfo = ReadYAML.read_section(sys_params,'options')
displayinfo = ReadYAML.read_section(sys_params,'display')
outputinfo = ReadYAML.read_section(sys_params,'output')
particleinfo = ReadYAML.read_section(sys_params,'particles')
#===========================================================================
# Read simulation parameters (this should be done externally)
#===========================================================================
wavelength = float(paraminfo['wavelength'])
dipole_radius = float(paraminfo['dipole_radius'])
timestep = float(paraminfo['time_step'])
#===========================================================================
# Read simulation options (this should be done externally)
#===========================================================================
frames = int(optioninfo['frames'])
#===========================================================================
# Read output options (this should be done externally)
#===========================================================================
vmd_output = bool(outputinfo.get('vmd_output',True))
excel_output = bool(outputinfo.get('excel_output',True))
include_force = bool(outputinfo.get('include_force',True))
include_couple = bool(outputinfo.get('include_couple',True))
#===========================================================================
# Read display options (this should be done externally)
#===========================================================================
display = Display.DisplayObject(displayinfo,frames)
#===========================================================================
# Read beam options and create beam collection
#===========================================================================
beam_collection = Beams.create_beam_collection(beaminfo,wavelength)
#n_beams = len(beam_collection)
#===========================================================================
# Read particle options and create particle collection
#===========================================================================
particle_collection = Particles.ParticleCollection(particleinfo)
print(particle_collection.num_particles)
n_particles = particle_collection.num_particles
#c = 3e8
#n1 = 3.9
#n1a = 1.5
ref_ind = particle_collection.get_refractive_indices()
particle_types = particle_collection.get_particle_types()
colors = particle_collection.get_particle_colours()
vtfcolors = particle_collection.get_particle_vtfcolours()
#radii = particle_collection.get_particle_radii()
shapes = particle_collection.get_particle_shape()
args   = particle_collection.get_particle_args()
#radius = radii[0] # because we cannot handle variable radii yet.
density = particle_collection.get_particle_density()
rho = density[0] # not yet implemented.
positions = particle_collection.get_particle_positions()

for i in range(n_particles):
    print(i,particle_types[i],ref_ind[i],colors[i],shapes[i],args[i],density[i],positions[i])
#===========================================================================
# Set up particle polarisabilities and other spurious options
#===========================================================================
ep1 = ref_ind**2
#ep1a = n1a * n1a
ep2 = 1.0
#radius = 200e-9
#rho = 2200 # glass density
#mass = (4/3)*rho*np.pi*radius**3
masses  = particle_collection.get_particle_masses()
gravity = np.zeros( (n_particles,3) ,dtype=np.float64)
# ??
# ?? POSSIBLE ERROR HERE WITH gravity[1] AS OPPOSED TO gravity[2]
# ??
gravity[:,1] = -9.81*masses
#gravity = np.zeros(3,dtype=np.float64)
#gravity[1] = -9.81*mass
print("dipole radius is:",dipole_radius,type(dipole_radius))
water_permittivity = 80.4
vacuum_permittivity = 1
k = 2 * np.pi / wavelength
epm = 1.333 # water
#a0 = (4 * np.pi * 8.85e-12) * (radius ** 3) * ((ep1 - 1) / (ep1 + 2))
#a0 = (4 * np.pi * 8.85e-12) * (dipole_radius ** 3) * ((ep1 - 1) / (ep1 + 2))
a0 = (4 * np.pi * 8.85e-12) * (dipole_radius ** 3) * ((ep1 - epm) / (ep1 + 2*epm))
#a0a = (4 * np.pi * 8.85e-12) * (dipole_radius ** 3) * ((ep1a - 1) / (ep1a + 2))
a = a0 / (1 - (2 / 3) * 1j * k ** 3 * a0/(4*np.pi*8.85e-12))
#aa = a0a / (1 - (2 / 3) * 1j * k ** 3 * a0a)
#a = a0
polarizability = a#a*np.ones(n_particles)
inverse_polarizability = (1.0+0j)/a0 # added this for the C++ wrapper (Chaumet's alpha bar)
E0 = None#0.0003e6  # V/m possibly # LEGACY REMOVE

BENDING = 0
stiffness = 0  # Errors when this is bigger than 1e-3

k_B = 1.38e-23
temperature = 293
viscosity = 8.90e-4

z_offset = wavelength / 4.0 # needed for odd order Bessel beams
z_offset = 0.0 # for most other situations

beam = "plane"  # LEGACY REMOVE



#===========================================================================
# Perform the simulation
#===========================================================================

initialT = time.time()
particles,optpos, optforces,optcouples = simulation(n_particles, positions, shapes, args)
finalT = time.time()
print("Elapsed time: {:8.6f} s".format(finalT-initialT))

# =====================================
# This code for matplotlib animation output and saving

for i in range(optforces.shape[0]):
    print("optforces "+str(i)+"= ",optforces[i]);

if display.show_output==True:
    # Plot beam, particles, forces and tracers (forces and tracers optional)
    fig, ax = None, None                                   #
    fig, ax = display.plot_intensity3d(beam_collection)    # Hash out if beam profile [NOT wanted]
    display.animate_system3d(optpos, shapes, args, colors, fig=fig, ax=ax, ignore_coords=["Z"], forces=optforces, include_quiver=True, include_tracer=False)

    ## ===
    ## Legacy Plotting Functions -> Remove
    ## ===
    # 2D animation
    #fig,ax = display.plot_intensity(beam_collection)
    #display.animate_particles(fig,ax,particles,radius,colors)





# writer = animation.PillowWriter(fps=30)

# ani.save("bessel-ang-mom-test.gif", writer=writer)
# =====================================

# =====================================
# This code for vtf file format output

# MyFileObject = open( #local save
#    "/Users/Tom/Documents/Uni/Optical_Forces_Project/Figures/vtf_files/local-test.vtf",
#    "w",
# )
#===========================================================================
# Write out data to files
#===========================================================================

if vmd_output==True:
    #
    # Uses old radius system, NOT shape,args
    # Must be fixed
    #
    pass
    #Output.make_vmd_file(filename_vtf,n_particles,frames,timestep,particles,optpos,beam_collection,finalT-initialT,radius,dipole_radius,z_offset,particle_types,vtfcolors)

if excel_output==True:
    Output.make_excel_file(filename_xl,n_particles,frames,timestep,particles,optpos,include_force,optforces,include_couple,optcouples)

