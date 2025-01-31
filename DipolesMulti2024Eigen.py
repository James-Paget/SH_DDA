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
import Generate_yaml
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


def Ajk(x, y, z, r, k):
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


def Djj(dipole_radius, k_B, temperature, viscosity):  # For Diffusion
    #
    # This is valid for a sphere, but not other shapes e.g. a torus
    # This will need to be changed when considering the dynamics of other particle shapes
    #
    djj = (k_B * temperature) / (6 * np.pi * viscosity * dipole_radius)
    D = np.zeros([3, 3])
    np.fill_diagonal(D, djj)

    return D


def Djk(x, y, z, r, k_B, temperature, viscosity):
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
    #r_max = 1.1 * (radius_i +radius_j)     # Original r_max
    r_max = 1.05 * (radius_i +radius_j)    # Reduced r_max, for closer interactions. NOTE; Should only be used with smaller time-steps (<1e-4)
    r_abs = np.linalg.norm(r)
    if r_abs < r_max:
        ##
        ## TEMPORARILY DISABLED
        ##
        # print("Eeek!! r_abs = ", r_abs)
        r_abs = r_max  # capping the force

    ##
    ## NOTE; WILL NEED REWORKING FOR BETTER RADIUS MANAGEMENT
    ##  -> Does still work with 2 spheres of equal radii
    ##
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


def spring_force(stiffness_const, natural_length, r):
    r_abs = np.linalg.norm(r)
    force = [stiffness_const * (r_abs - natural_length) * (r[i] / r_abs) for i in range(3)]  # Previous method
    # force = np.zeros(3)
    # force[0] = constant1*(r_abs-2*dipole_radius)*(r[0]/r_abs)
    # force[1] = constant1*(r_abs-2*dipole_radius)*(r[1]/r_abs)
    # force[2] = constant1*(r_abs-2*dipole_radius)*(r[2]/r_abs)

    return force


def driving_force(constant1, r, dipole_radius):
    #print("Dipole Radius:",dipole_radius)
    r_abs = np.linalg.norm(r)
    force = [constant1 * (r_abs - 2.0*dipole_radius) * (r[i] / r_abs) for i in range(3)]
    # force = np.zeros(3)
    # force[0] = constant1*(r_abs-2*dipole_radius)*(r[0]/r_abs)
    # force[1] = constant1*(r_abs-2*dipole_radius)*(r[1]/r_abs)
    # force[2] = constant1*(r_abs-2*dipole_radius)*(r[2]/r_abs)

    return force

def rot_vector_in_plane(r, r_plane, theta):
    """
    r = vector to be rotated
    r_plane = the plane to rotate r within (NOTE must be a unit vector)
    theta = angle to be rotated by
    """
    # Using the Rodrigues' rotation formula
    comp_a = r*np.cos(theta)
    comp_b = np.cross(r_plane, r)*np.sin(theta)
    comp_c = r_plane*np.dot(r_plane, r)*(1-np.cos(theta))
    return comp_a +comp_b + comp_c

def bending_force(bond_stiffness, ri, rj, rk, eqm_angle):
    # Calculates the bending force on particles j-i-k.
    # This is for any equilibrium angle so the system is partially rotated so forces restore towards that angle.
    # The rotation is undone before the forces are returned.
    # Rotation is only done when eqm_angle != 0 or pi (else plane undefined).

    # rj and rk relative to ri.
    rij = rj - ri
    rik = rk - ri
    
    rij_abs = np.linalg.norm(rij)
    rik_abs = np.linalg.norm(rik)
    rijrik = rij_abs * rik_abs
    rij2 = rij_abs * rij_abs
    rik2 = rik_abs * rik_abs

    is_plane_defined = ( np.linalg.norm( np.cross(rij, rik) ) != 0 )

    if is_plane_defined:
        # Normal to plane of rotation.
        r_plane = -np.cross(rij, rik) / np.linalg.norm( np.cross(rij, rik) ) 
        theta = np.pi - eqm_angle
        # Rotate rij.
        rij = rot_vector_in_plane(rij, r_plane, theta)    # Rotate by equilibrium angle in the plane of the points

    force = np.zeros([3, 3])

    if np.isnan(rij).any() or np.isnan(rik).any():
        print(f"Bending force received NaN,returning 0. rij; rik; r_plane = {rij}; {rik}; {r_plane}")
        return np.zeros([3, 3])
    
    # Rotate rij so that if it were at eqm_angle, it would be at a stable eqm.
    costhetajik = np.dot(rij, rik) / rijrik
    Fj = bond_stiffness * (costhetajik * rij / rij2 - rik / rijrik)
    Fk = bond_stiffness * (costhetajik * rik / rik2 - rij / rijrik)

    if is_plane_defined:
        # Unrotate Fj.
        Fj = rot_vector_in_plane(Fj, r_plane, -theta)

    # Fi used to make net force zero.
    i = 1
    force[i - 1] = Fj
    force[i + 1] = Fk
    force[i] = -(Fj + Fk)

    # OLD
    # i = 1
    # force[i] = bond_stiffness * (
    #     (rik + rij) / rijrik - costhetajik * (rij / rij2 + rik / rik2)
    # )
    # force[i - 1] = bond_stiffness * (costhetajik * rij / rij2 - rik / rijrik)
    # force[i + 1] = bond_stiffness * (costhetajik * rik / rik2 - rij / rijrik)

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


def dipole_moment_array(array_of_positions, E0, dipole_radius, number_of_dipoles_in_primitive, polarizability, beam_collection, k):

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
            k
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


def optical_force_array(array_of_particles, E0, dipole_radius, dipole_primitive, k, n_particles, a0, excel_output, include_couple, beam_collection, polarizability):
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
    p_array = dipole_moment_array(array_of_positions, E0, dipole_radius, number_of_dipoles_in_primitive, polarizability, beam_collection, k)
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



def buckingham_force_array(array_of_positions, effective_radii, particle_neighbours):
    number_of_particles = len(array_of_positions)
    displacements_matrix = displacement_matrix(array_of_positions)
    displacements_matrix_T = np.transpose(displacements_matrix)
    #Hamaker = (np.sqrt(30e-20) - np.sqrt(4e-20))**2
    Hamaker = 0
    ConstantA = 1.0e23
    ConstantB = 2.0e8  # 4.8e8
    # ConstantA = (1e-34) *1.0e23
    # ConstantB = (2e-1) *2.0e8
    buckingham_force_matrix = np.zeros(
        [number_of_particles, number_of_particles], dtype=object
    )
    
    for i in range(number_of_particles):
        for j in range(number_of_particles):
            # if i == j:
            if j in particle_neighbours[i]: # particle_neighbours includes itself
                buckingham_force_matrix[i][j] = 0 #[0, 0, 0]
            else:
                buckingham_force_matrix[i][j] = buckingham_force(
                    Hamaker,
                    ConstantA,
                    ConstantB,
                    displacements_matrix_T[i][j],
                    effective_radii[i],
                    effective_radii[j]
                )  
    buckingham_force_array = np.zeros((number_of_particles,3),dtype=np.float64)
    temp = np.sum(buckingham_force_matrix, axis=1)
    for i in range(number_of_particles):
        buckingham_force_array[i] = temp[i]
    return buckingham_force_array


def stop_particles_overlapping(array_of_positions, effective_radii, particle_neighbours):
    # for each particle, stop it overlapping with ones it's within N connections of (based on particle_neighbours).
    # no return as changes are made directly to array_of_positions.

    epsilon = 1e-10 # prevent small float errors.
    done = False
    count = 0

    while not done:
        count += 1
        done = True
        for i in range(len(particle_neighbours)):
            ri = array_of_positions[i]
            for j in particle_neighbours[i]:
                if i == j: # skip itself
                    continue

                rj = array_of_positions[j]
                rij = rj - ri
                abs_rij = np.linalg.norm(rij)
                difference = effective_radii[i] + effective_radii[j] - abs_rij
                if difference > 0: # if overlapping
                    done = False
                    array_of_positions[i] -= (difference + epsilon)/2 *rij/abs_rij
                    array_of_positions[j] += (difference + epsilon)/2 *rij/abs_rij
                    # print(f"Now {array_of_positions[i]} and {array_of_positions[j]}")

        if count > 10:
            print("stop_particles_overlapping: could not resolve overlaps, continuing.")
            break

            

def generate_connection_indices(array_of_positions, mode="manual", args=[]):
    """
    Return a list of matrix indices (i,j) of connected particles
    mode (args): num (num_connections), line (), dist ()
    Defaults to no connections with mode="manual", args=[]
    """

    num_particles = len(array_of_positions)
    connection_indices = []

    print("Generating connection indices with mode= ",mode)

    match mode:
        case "num":
            # this finds the pairs based on the num_connections closest particles.
            # args: [num_connections]
            num_connections = args[0]
            
            if num_connections > num_particles-1:
                num_connections = num_particles-1
                print(f"generate_connection_indices, mode='num': num_connections larger than max, so set to {num_connections}")

            displacements_matrix = displacement_matrix(array_of_positions)
            distances_matrix = np.zeros((num_particles, num_particles))
            current_connections = np.zeros(num_particles)

            for i in range(num_particles):
                # make scalar distance matrix
                for j in range(i, num_particles):
                    distance = np.linalg.norm(displacements_matrix[i][j])
                    distances_matrix[i][j] = distance
                    distances_matrix[j][i] = distance
                
                # order to test the closest first
                closest_js = np.argsort(distances_matrix[i])

                for j in closest_js:    
                    # check if more connections needed and no duplicates (only search search upper triangle).
                    if current_connections[i] < num_connections and j>i:
                        # add in pairs
                        connection_indices.append((i,j))
                        connection_indices.append((j,i))
                        current_connections[i] += 1
                        current_connections[j] += 1

            print(f"Num connections are {current_connections}")
            for i in range(num_particles):
                if current_connections[i] != num_connections:
                    print(f"Warning, particle {i} not properly connected, connections={current_connections[i]}")
                    

        case "line":
            # this links them in a line ordered by index.
            # args: [] for a line, or [True] for a ring.

            # For each point (i,i) along the diagonal except the last, add the point below and the point to the right.
            for i in range(num_particles-1):
                connection_indices.append((i+1,i))
                connection_indices.append((i,i+1))
            if(len(args) > 0):
                if(args[0] == True):
                    # Enable looping, connects last particle back to 0th
                    connection_indices.append((num_particles-1, 0))
                    connection_indices.append((0, num_particles-1))

        case "dist":
            # this links each particle to every other particle within a certain distance.
            # args: [] will approximate a dist, or can be passed in: [dist]
            if num_particles < 2:
                print("generate_connection_indices: dist num_particles error, setting connections=[]")

            if len(args) == 0:
                approx_radius = np.linalg.norm(array_of_positions[0])
                approx_min_spacing = 2*approx_radius / np.sqrt(num_particles-1) # N=2, dist= 2*rad and dist^2 proportional to area, area per particle proportional to 1/N
                dist = 1.5 * approx_min_spacing
            else:
                dist = args[0]

            current_connections = np.zeros(num_particles) # (only used for print data collection)

            for i in range(num_particles):
                for j in range(i+1, num_particles):
                    if np.linalg.norm( array_of_positions[i] - array_of_positions[j] ) < dist:
                        connection_indices.append((i,j))
                        connection_indices.append((j,i))

                        current_connections[i] += 1
                        current_connections[j] += 1

            current_connections = np.array(current_connections)
            #print(f"avg connections {np.average(current_connections):.2f}, max diff {np.max(current_connections)-np.min(current_connections)}")
        
        case "dist_beads":
            # Links every non-bead particle to eachother by distance (arg[0] distance chosen)
            # Then links beads to any other particle by distance (arg[1] distance chosen)
            # Assumes there are N beads (arg[3]) all located at the end of the particles list
            # args: [dist_p, dist_b, number_of_beads]
            if num_particles < 2:
                print("generate_connection_indices: dist num_particles error, setting connections=[]")

            if(len(args) < 3):
                print("Invalid number of arguements for connections, require 3; "+str(len(args))+" given")
            else:
                dist_p = args[0]
                dist_b = args[1]
                number_of_beads = int(args[2])

                # Connect all non-bead particles 
                for i in range(num_particles-number_of_beads):
                    for j in range(i+1, num_particles-number_of_beads):
                        # If Other-Other interaction
                        if np.linalg.norm( array_of_positions[i] - array_of_positions[j] ) < dist_p:
                            connection_indices.append((i,j))
                            connection_indices.append((j,i))
                            # current_connections[i] += 1
                            # current_connections[j] += 1

                # If Bead-Any interaction
                for i in range(num_particles-number_of_beads, num_particles):
                    for j in range(num_particles):
                        if(i!=j):
                            if np.linalg.norm( array_of_positions[i] - array_of_positions[j] ) < dist_b:
                                connection_indices.append((i,j))
                                connection_indices.append((j,i))
                                # current_connections[i] += 1
                                # current_connections[j] += 1

        case "dist_shells":
            # Makes connections for concentric shells of particles
            # args: [should_connect_shells, connect_fraction,    shell_type, radius ,connection_dist, ...  <optional more shell_type, radius, connection_dist> ]
            # should_connect_shells, connect_fraction allows the shells to be connected.
            # Each shell has 3 values: shell_type, radius ,connection_dist
            # Shell types are sphere, cylinderx, cylindery, cylinderz
            
            # Functions for different shapes. x,y,z,r,t: stand for coords, radius and tolerance
            shell_shapes_info = {"sphere": lambda x,y,z,r,t: x*x+y*y+z*z>=(r-t)**2 and x*x+y*y+z*z<=(r+t)**2,
                                 "cylinderx": lambda x,y,z,r,t: y*y+z*z>=(r-t)**2 and y*y+z*z<=(r+t)**2,
                                 "cylindery": lambda x,y,z,r,t: x*x+z*z>=(r-t)**2 and x*x+z*z<=(r+t)**2,
                                 "cylinderz": lambda x,y,z,r,t: x*x+y*y>=(r-t)**2 and x*x+y*y<=(r+t)**2,
                                 }
            
            # Extract args input.
            should_connect_shells = args[0]
            connect_fraction = args[1]
            array_of_positions = np.array(array_of_positions)
            shell_is_list = []
            shell_types = []
            shell_radii = []
            shell_dists = []
            arg_i = 2
            while arg_i < len(args):
                shell_types.append(args[arg_i+0])
                shell_radii.append(args[arg_i+1])
                shell_dists.append(args[arg_i+2])
                arg_i+=3

            if connect_fraction == 0:
                mod_period=1.0
                if should_connect_shells:
                    print(f"Set mod_period to 1")
            else:
                mod_period = int(np.ceil(1/connect_fraction))
            
            # Make connections for each shell
            for shell_i in range(len(shell_types)):
                shell_type = shell_types[shell_i]
                radius = shell_radii[shell_i]
                connection_dist = shell_dists[shell_i]
                point_is = []
                tolerance = radius/10

                # Build up list of points in the shell.
                shell_function = shell_shapes_info[shell_type]
                for i in range(len(array_of_positions)):
                    x,y,z = array_of_positions[i]
                    if shell_function(x,y,z,radius,tolerance):
                        point_is.append(i)
                shell_is_list.append(point_is)

                # Connect close points in the shell.
                for i in point_is:
                    for j in point_is:
                        if i!=j and np.linalg.norm( array_of_positions[i] - array_of_positions[j] ) < connection_dist:
                            connection_indices.append((i,j))
                            connection_indices.append((j,i))

                # Option to connect some shell points to close ones on the next inner shell.
                if should_connect_shells and shell_i != 0:
                    inner_shell_is = shell_is_list[shell_i-1][::mod_period] # Slice to get a subset of inner points
                    # Connect each of those to the closest point in the current shell.
                    for i in inner_shell_is:
                        print(np.linalg.norm(array_of_positions[point_is] - array_of_positions[i], axis=1))
                        closest_is = np.argsort(np.linalg.norm(array_of_positions[point_is] - array_of_positions[i], axis=1))
                        for closest_i in closest_is:
                            if closest_i != i:
                                connection_indices.append((i,closest_i))
                                connection_indices.append((closest_i,i))
                                break

        case "manual":
            # Manually state which particles will be connected in arguments when more specific connection patterns required
            connection_indices = args
        case _:
            sys.exit(f"get_connected_pairs error: modes are 'num', 'line', 'loop', 'manual'.\nInputted mode, args: {mode}, {args}")

    #print("Connections established= ",connection_indices)
    return connection_indices

def get_equilibrium_angles(initial_positions, connection_indices):
    # build list of [i,j,k,eqm_angle]; i is central index, j,k are connected particles.
    # Calculating equilibrium angles from initial_positions assumes connections are initially in equilibrium.

    if len(connection_indices) == 0:
        return []

    ijkangles = []

    # sort the indices by the first argument of each pair
    connection_indices = np.array(connection_indices)
    sorted_indices = connection_indices[np.argsort(connection_indices[:,0])]

    idx = 0
    while idx < len(sorted_indices):
        # i is the central particle's index
        i = sorted_indices[idx,0]

        # step through sorted_indices until i would change, and form connections list.
        connections = []
        while idx < len(sorted_indices) and sorted_indices[idx,0] == i:
            connections.append(sorted_indices[idx,1])
            idx += 1

        # find angle for each combination of connections.
        for j,k in it.combinations(connections, 2):
            u = initial_positions[j] - initial_positions[i]
            v = initial_positions[k] - initial_positions[i]
            angle = np.arccos(np.clip(np.dot(u,v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0,1.0))
            ijkangles.append([i,j,k,angle])

    return ijkangles

def group_particles_into_objects(number_of_particles, connection_indices):
    # Returns a list of the particle indices of each object
    # CURRENTLY UNUSED.

    if len(connection_indices) == 0: # test trivial unconnected case
        return [ [i] for i in range(number_of_particles)]

    else:
        # initialise object with particle 0 and its direct connections in.
        connection_indices = np.array(connection_indices)
        start = [0] 
        for i in connection_indices[connection_indices[:,0]==0][:,1]:
            start.append(i)
        object_indices_list = [start]

        for i in range(1, number_of_particles):
            # get i's connections
            i_connections = connection_indices[connection_indices[:,0]==i][:,1]
            
            # search through object_indices_list if any of its arrays contain a connection, else make new array
            found = -1
            arrays_to_merge = []
            for j in range(len(object_indices_list)):

                if set(object_indices_list[j]) & set(i_connections): # test for shared elements
                    
                    # if already found, will need to merge the two objects
                    if found != -1:
                        arrays_to_merge.append(object_indices_list[j])

                    else:
                        # add if not pre existing
                        if not (i in object_indices_list[j]):
                            object_indices_list[j].append(i)
                        for i_connection in i_connections[i_connections>i]:
                            if not (i_connection in object_indices_list[j]):
                                object_indices_list[j].append(i_connection)

                        found = j

            # make new object if not found
            if found == -1:
                object_indices_list.append([i])

            # merge here
            for array in arrays_to_merge:
                object_indices_list[found].extend(array)
                object_indices_list.remove(array)
    
    return object_indices_list

def get_nearest_neighbours(number_of_particles, connection_indices, max_connections_dist=2):
    # Particles that are within "max_connections_dist" connections of each other are considered nearby.
    # Returns [ [particles nearby to 0th particle], [particles nearby to 1st particle], ... ]

    if len(connection_indices) == 0: # test trivial unconnected case
        return [ [i] for i in range(number_of_particles)]

    nearby_list = []
    connection_indices = np.array(connection_indices)
    # Search for each particle
    for particle_i in range(number_of_particles):
        i_list = [particle_i]
        checked_i = 0 # index to start searching i_list from
        # Up to max searches, get neighbours of unchecked particles, add them to the list if not already there
        for _ in range(max_connections_dist):
            for i_list_idx in range(checked_i, len(i_list)):
                idx_connections = connection_indices[connection_indices[:,0]==i_list[i_list_idx]][:,1] # get connections starting with idx then pull out what it's connected to with [:,1]
                for i in idx_connections:
                    if i not in i_list:
                        i_list.append(i)
                checked_i += 1 # start looping through i_list later as more i_list_idx's are checked
        nearby_list.append(i_list)

    return nearby_list

        

""" OLD SPRINGS
# def spring_force_array(array_of_positions, dipole_radius):
#     number_of_dipoles = len(array_of_positions)
#     displacements_matrix = displacement_matrix(array_of_positions)
#     displacements_matrix_T = np.transpose(displacements_matrix)
#     stiffness = 1.0e-5
#     spring_force_matrix = np.zeros([number_of_dipoles, number_of_dipoles], dtype=object)

#     for i in range(number_of_dipoles):
#         for j in range(number_of_dipoles):
#             spring_force_matrix[i][j] = np.zeros(3)
#     for i in range(0,number_of_dipoles,2):
#         j = i + 1
#         spring_force_matrix[i][j] = spring_force(stiffness, displacements_matrix_T[i][j], dipole_radius)
#     for i in range(1,number_of_dipoles,2):
#         j = i - 1
#         spring_force_matrix[i][j] = spring_force(stiffness, displacements_matrix_T[i][j], dipole_radius)

#     # print(f"Spring matrix is\n{np.array(spring_force_matrix)}")

#     #print("Spring force array shape before:",spring_force_matrix.shape)
#     spring_force_array = np.sum(spring_force_matrix, axis=1)
#     #    print("Springs",spring_force_array)
#     #print("Spring force array shape after:",spring_force_array.shape)
#     return spring_force_array
"""

def generate_stiffness_matrix(number_of_particles, connection_indices, stiffness_spec={"type":"", "default_value":5e-7}):
    #
    # Generates a matrix of spring stiffness for each particle pair
    #
    spring_stiffness_matrix = np.zeros( (number_of_particles, number_of_particles), dtype=float )
    for i,j in connection_indices:
        spring_stiffness_element = generate_spring_stiffness_element(stiffness_spec, i, j)
        spring_stiffness_matrix[i,j] = spring_stiffness_element
        spring_stiffness_matrix[j,i] = spring_stiffness_element
    return spring_stiffness_matrix

def generate_naturallength_matrix(number_of_particles, connection_indices, initial_shape):
    #
    # Generates a matrix of natural lengths for each particle pair
    #
    spring_naturallength_matrix = np.zeros( (number_of_particles, number_of_particles), dtype=float )
    for i,j in connection_indices:
        spring_naturallength_element = generate_spring_naturallength_element(initial_shape[i], initial_shape[j])
        spring_naturallength_matrix[i,j] = spring_naturallength_element
        spring_naturallength_matrix[j,i] = spring_naturallength_element
    return spring_naturallength_matrix

def spring_force_array(array_of_positions, connection_indices, spring_stiffness_matrix, spring_naturallength_matrix):
    """
    . Calculates the spring forces along the connections specified
    . Pulls spring data from an initial particle arrangement given

    . initial_shape = [ [position1], [position2], ..., [positionN] ] for a system of N particles
    .       Where positionN = [x,y,z] of the Nth particle
    . stiffness_regime = String to specify a pattern for the spring constants, if left blank will default 
    to a constant value for all springs (set inside the function)
    """
    number_of_dipoles = len(array_of_positions)
    displacements_matrix = displacement_matrix(array_of_positions)
    displacements_matrix_T = np.transpose(displacements_matrix)

    spring_force_matrix = np.zeros([number_of_dipoles, number_of_dipoles, 3], dtype=object)

    # Populate elements in each matrix
    for i,j in connection_indices:
        spring_force_matrix[i][j] = spring_force(spring_stiffness_matrix[i,j], spring_naturallength_matrix[i,j], displacements_matrix_T[i][j])
    # Non-matrix stiffness and natural length approach
    # spring_force_matrix[i][j] = spring_force(stiffness, natural_length, displacements_matrix_T[i][j])

    # Gets total spring force on each
    spring_force_array = np.sum(spring_force_matrix, axis=1)
    return spring_force_array


def generate_spring_stiffness_element(stiffness_spec, i, j):
    """
    . Fetch the stiffness of the spring according to the regime specified

    . stiffness_spec = {"type":..., "default_value":..., <OTHER ARGS>}
        "type" = the regime used to set stiffness values, e.g. uniform for all, uniform except beads, etc
        Arguements aer interpretted differently by each type
    . i,j = the indices of the 2 particles being connected
    """
    stiffness = 0.0
    match stiffness_spec["type"]:
        case "beads":       # {"type", "default_value", "bead_value", "bead_indices"}
            # Allows connections between beads and any other particle to have different stiffness to 
            # the default stiffness between non-bead particles
            if(len(stiffness_spec) >= 4):
                if( (i in stiffness_spec["bead_indices"]) or (j in stiffness_spec["bead_indices"]) ):
                    stiffness = stiffness_spec["bead_value"]
                else:
                    stiffness = stiffness_spec["default_value"]
            else:
                print("-- Invalid stiffness_spec length given --")
        case _:             # {"type", "default_value"}
            # Defaults to all springs having same, constant, value
            if(len(stiffness_spec) >= 2):
                stiffness = stiffness_spec["default_value"]
            else:
                print("-- Invalid stiffness_spec length given --")
    return stiffness


def generate_spring_naturallength_element(initial_shape_p1, initial_shape_p2):
    """
    . Fetch the natural length of a given spring element using the initial_shape specified
    .   This will set the distance between the particles in the initial_shape as the natural length (e.g. wants to return to this shape)
    """
    # Set natural length to be the distance between the particles
    return np.sqrt(np.sum(pow(initial_shape_p2-initial_shape_p1,2)))


# def driving_force_array(array_of_positions):
#     number_of_dipoles = len(array_of_positions)
#     displacements_matrix = displacement_matrix(array_of_positions)
#     displacements_matrix_T = np.transpose(displacements_matrix)
#     driver = 3.0e-7#6
#     driving_force_array = np.zeros(number_of_dipoles, dtype=object)
#     for i in range(0,number_of_dipoles,2):
#         j = i+1
#         driving_force_array[i] = driving_force(driver, displacements_matrix_T[i][j])
#         driving_force_array[j] = driving_force_array[i]
#     return driving_force_array

def driving_force_array(array_of_positions, driving_type, args={}):
    """
    . Apply custom forces to the system's particles
    """
    number_of_dipoles = len(array_of_positions)
    driving_force_array = np.zeros((number_of_dipoles,3), dtype=object)
    match driving_type:
        case "circ_push":
            #
            # Applies a force to particles within some circular XY plane radius (any Z height)
            #
            # args = {
            #       driver_magnitude,
            #       influence_radius
            #   }
            # e.g.{5.0e-12, 0.5e-6}
            #
            driver_magnitude = args["driver_magnitude"]
            influence_radius = args["influence_radius"]
            for p in range(len(array_of_positions)):
                drive_condition = np.sqrt( pow(array_of_positions[p,0],2) + pow(array_of_positions[p,1],2) ) < influence_radius
                if(drive_condition):
                    driving_force_array[p] = np.array([0.0, 0.0, driver_magnitude*1.0])
        case "timed_circ_push":
            #
            # Applies a force to particles within some circular XY plane radius (any Z height), but only for frames 
            # less than some specified cutoff
            #
            # args = {
            #       driver_magnitude,
            #       influence_radius,
            #       current_frame,
            #       cutoff_frame
            #   }
            # e.g.{5.0e-12, 0.5e-6, frame, 10}
            #
            driver_magnitude = args["driver_magnitude"]
            influence_radius = args["influence_radius"]
            current_frame = args["current_frame"]
            cutoff_frame = args["cutoff_frame"]
            if(current_frame < cutoff_frame):
                for p in range(len(array_of_positions)):
                    drive_condition = np.sqrt( pow(array_of_positions[p,0],2) + pow(array_of_positions[p,1],2) ) < influence_radius
                    if(drive_condition):
                        driving_force_array[p] = np.array([0.0, 0.0, driver_magnitude*1.0])
        case "osc_circ_push":
            #
            # Applies a force to particles within some circular XY plane radius (any Z height) with amplitude 
            # oscillating between ±driver_magnitude
            #
            # args = {
            #       driver_magnitude,
            #       influence_radius,
            #       current_frame,
            #       frame_period
            #   }
            # e.g.{5.0e-12, 0.5e-6, frame, 50}
            #
            driver_magnitude = args["driver_magnitude"]
            influence_radius = args["influence_radius"]
            current_frame = args["current_frame"]
            frame_period = args["frame_period"]
            for p in range(len(array_of_positions)):
                drive_condition = np.sqrt( pow(array_of_positions[p,0],2) + pow(array_of_positions[p,1],2) ) < influence_radius
                if(drive_condition):
                    driving_force_array[p] = np.array([0.0, 0.0, driver_magnitude*np.sin(current_frame/frame_period * 2*np.pi)])

        case "stretch":
            #
            # Applies a force to particles with the highest/lowest values on the specified axis.
            #
            # args = {
            #       driver_magnitude,
            #       axes,
            #       influence_distance
            #       initial_positions,
            #   }
            # e.g.{5.0e-12, "x", 0.5e-7, [[0,0,0],..]}
            #
            driver_magnitude = args["driver_magnitude"]
            axes = args["axes"]
            influence_distance = args["influence_distance"] # distance into the object after the min/max is found that is influenced.
            initial_positions = args["initial_positions"]

            axis_description = {"x":[0,np.array([1,0,0])], "y":[1,np.array([0,1,0])], "z":[2,np.array([0,0,1])]} # holds index and normal for each axis
            for axis in axes:
                ax_index, ax_normal = axis_description[axis]
                min_threshold = np.min(initial_positions[:,ax_index]) + influence_distance
                max_threshold = np.max(initial_positions[:,ax_index]) - influence_distance

                for p in range(len(array_of_positions)):
                    if initial_positions[p,ax_index] < min_threshold:
                        driving_force_array[p] += -ax_normal * driver_magnitude 
                    
                    if initial_positions[p,ax_index] > max_threshold:
                        driving_force_array[p] += ax_normal * driver_magnitude 
            
        
        case _:
            print("Driving force type not recognised, (0,0,0) force returned; ",driving_type)
    return driving_force_array

""" OLD BENDING
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
"""

def bending_force_array(array_of_positions, ijkangles, bond_stiffness):
    number_of_particles = len(array_of_positions)
    bending_force_matrix = np.zeros([number_of_particles,3])
    bending_force_temp = np.zeros([3,3])

    for i,j,k,eqm_angle in ijkangles:
        bending_force_temp = bending_force(
                    bond_stiffness,
                    array_of_positions[i],
                    array_of_positions[j],
                    array_of_positions[k],
                    eqm_angle
                )

        bending_force_matrix[j] += bending_force_temp[0]
        bending_force_matrix[i] += bending_force_temp[1]
        bending_force_matrix[k] += bending_force_temp[2]
        # print(f"{i}, {j}, {k}: bending_force_temp is {bending_force_temp}")

    return bending_force_matrix
    
    
def gravity_force_array(array_of_positions, dipole_radius):
    number_of_beads = len(array_of_positions)
    # bond_stiffness = BENDING
    gravity_force_matrix = np.zeros([number_of_beads], dtype=object)
    gravity_force_temp = np.zeros([3], dtype=object)
    gravity_force_temp[2] = -1e-12#-9.81*mass
    for i in range(number_of_beads):
        gravity_force_matrix[i] = gravity_force_temp
    return gravity_force_matrix


def diffusion_matrix(array_of_positions, particle_radii, k_B, temperature, viscosity):
    # positions of particle centres
    # dipole_radius is actually considering the spehre radius here
    list_of_displacements = [u - v for u, v in it.combinations(array_of_positions, 2)]
    array_of_displacements = np.zeros(len(list_of_displacements), dtype=object)
    for i in range(len(list_of_displacements)):
        array_of_displacements[i] = list_of_displacements[i]
    array_of_distances = np.array([np.linalg.norm(w) for w in array_of_displacements])
    number_of_particles = len(array_of_positions)
    number_of_displacements = len(array_of_displacements)
    D_matrix = np.zeros([number_of_particles, number_of_particles], dtype=object)
    Djk_array = np.zeros(
        number_of_displacements, dtype=object
    )  # initialize array to store D_jk matrices
    Djj_array = np.zeros(
        number_of_particles, dtype=object
    )  # initialize array to store D_jj matrices
    iu = np.triu_indices(number_of_particles, 1)
    di = np.diag_indices(number_of_particles)
    for i in range(number_of_displacements):
        Djk_array[i] = Djk(
            array_of_displacements[i][0],
            array_of_displacements[i][1],
            array_of_displacements[i][2],
            array_of_distances[i],
            k_B,
            temperature,
            viscosity
        )
    for i in range(number_of_particles):
        Djj_array[i] = Djj(particle_radii[i], k_B, temperature, viscosity)
    D_matrix[iu] = Djk_array
    D_matrix.T[iu] = D_matrix[iu]
    D_matrix[di] = Djj_array
    temporary_array = np.zeros(number_of_particles, dtype=object)
    for i in range(number_of_particles):
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
    number_of_dipoles = 0
    num = int(2*sphere_radius/dipole_diameter)
    nums = np.arange(-(num-1)/2,(num+1)/2,1)
    for i in nums:
        i2 = i*i
        for j in nums:
            j2 = j*j
            for k in nums:
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
    num = int(2*sphere_radius/dipole_diameter)
    nums = np.arange(-(num-1)/2,(num+1)/2,1)
    for i in nums:
        i2 = i*i
        x = i*dipole_diameter +1e-20
        for j in nums:
            j2 = j*j
            y = j*dipole_diameter +1e-20
            for k in nums:
                k2 = k*k
                z = k*dipole_diameter
                rad2 = (i2+j2+k2)*dd2
                if rad2 < sr2:
                    pts[number_of_dipoles] = [x, y, z]
                    number_of_dipoles += 1
    print(number_of_dipoles," dipoles generated")
    # print(f"Z PTS ARE\n{np.unique(pts[:, 2])}\n\n") # test the full/half int shift.
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
    # ** Could also move x,y,z calculation into if to speed up program -> reduce waste on non-dipole checks
    #
    torus_centre_radius, torus_tube_radius, phi_lower, phi_upper = args
    dipole_diameter = 2*dipole_radius
    dd2 = dipole_diameter**2
    ttr2 = torus_tube_radius**2
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


def cylinder_size(args, dipole_radius):
    #
    # Only supports torus flat in XY plane
    # args = [radius, width, theta_azimuthal, theta_zenith]
    #   radius = radius of the circular cylinder face
    #   width  = distance between 2 circular faces, witht the origin located half way between them (COM of cylinder)
    #   theta_Z = rotation of centre of cylinder about Z axis, in radians
    #   theta_pitch = rotation of the Z-rotated-cylinder about its perpendicular axis (pitch up / down)
    #
    # e.g. Unrotated cylinder lies in the X axis
    #
    radius, width, theta_Z, theta_pitch = args
    dipole_diameter = 2*dipole_radius
    #dd2 = dipole_diameter**2
    r2 = radius**2
    limiting_radius = np.sqrt( pow(width/2.0,2) + pow(radius,2) )   # Radius of the sphere covering the entire cylinder (slightly more than width/2.0)
    num = int( (limiting_radius)//dipole_diameter )                 # Check cubic area of this limiting space
    ##
    ## NOTE; approach above can be sped up by limiting X,Y,Z range of each by considering which rotations are given
    ##      However, for a 1 time calcualtion like this it is not hugely important
    ##

    #
    # Counts number of points in object
    #
    number_of_dipoles = 0

    for i in range(-num,num+1):
        #i2 = i*i
        x = i*dipole_diameter
        for j in range(-num,num+1):
            j2 = j*j
            y = j*dipole_diameter        
            for k in range(-num,num+1):
                k2 = k*k
                z = k*dipole_diameter

                # Apply rotations backwards to an unrotated frame
                Z_rotation_matrix = np.array(
                    [
                        [np.cos(-theta_Z), -np.sin(-theta_Z), 0.0],
                        [np.sin(-theta_Z),  np.cos(-theta_Z), 0.0],
                        [0.0, 0.0, 1.0]
                    ]
                )
                pitch_vec = [-np.sin(-theta_Z), np.cos(-theta_Z), 0] # Front facing vector (1,0,0) rotated, then perpendicular taken (-y,x,0)
                pitch_rotation_matrix = np.array(
                    [
                        [( (pitch_vec[0]*pitch_vec[0])*(1.0-np.cos(-theta_pitch)) +(np.cos(-theta_pitch))              ), ( (pitch_vec[1]*pitch_vec[0])*(1.0-np.cos(-theta_pitch)) -(np.sin(-theta_pitch)*pitch_vec[2]) ), ( (pitch_vec[2]*pitch_vec[0])*(1.0-np.cos(-theta_pitch)) +(np.sin(-theta_pitch)*pitch_vec[1]) )],
                        [( (pitch_vec[0]*pitch_vec[1])*(1.0-np.cos(-theta_pitch)) +(np.sin(-theta_pitch)*pitch_vec[2]) ), ( (pitch_vec[1]*pitch_vec[1])*(1.0-np.cos(-theta_pitch)) +(np.cos(-theta_pitch)             ) ), ( (pitch_vec[2]*pitch_vec[1])*(1.0-np.cos(-theta_pitch)) -(np.sin(-theta_pitch)*pitch_vec[0]) )],
                        [( (pitch_vec[0]*pitch_vec[2])*(1.0-np.cos(-theta_pitch)) -(np.sin(-theta_pitch)*pitch_vec[1]) ), ( (pitch_vec[1]*pitch_vec[2])*(1.0-np.cos(-theta_pitch)) +(np.sin(-theta_pitch)*pitch_vec[0]) ), ( (pitch_vec[2]*pitch_vec[2])*(1.0-np.cos(-theta_pitch)) +(np.cos(-theta_pitch)             ) )]
                    ]
                )
                xyz_rotated = np.dot( Z_rotation_matrix, np.array([x,y,z]) )    # Apply Z rotation
                xyz_rotated = np.dot( pitch_rotation_matrix, xyz_rotated )      # Apply pitch rotation

                within_radial = (xyz_rotated[1]**2 + xyz_rotated[2]**2) <= r2
                within_width  = abs(xyz_rotated[0]) <= width/2.0
                if(within_radial and within_width):
                    number_of_dipoles += 1

    return number_of_dipoles

def cylinder_positions(args, dipole_radius, number_of_dipoles_total):
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
    radius, width, theta_Z, theta_pitch = args
    dipole_diameter = 2*dipole_radius
    #dd2 = dipole_diameter**2
    r2 = radius**2
    limiting_radius = np.sqrt( pow(width/2.0,2) + pow(radius,2) )   # Radius of the sphere covering the entire cylinder (slightly more than width/2.0)
    num = int( (limiting_radius)//dipole_diameter )                 # Check cubic area of this limiting space
    ##
    ## NOTE; approach above can be sped up by limiting X,Y,Z range of each by considering which rotations are given
    ##      However, for a 1 time calcualtion like this it is not hugely important
    ##

    #
    # Counts number of points in object
    #
    pts = np.zeros((number_of_dipoles_total, 3))
    number_of_dipoles = 0

    for i in range(-num,num+1):
        #i2 = i*i
        x = i*dipole_diameter
        for j in range(-num,num+1):
            j2 = j*j   
            y = j*dipole_diameter         
            for k in range(-num,num+1):
                k2 = k*k
                z = k*dipole_diameter

                # Apply rotations backwards to an unrotated frame
                Z_rotation_matrix = np.array(
                    [
                        [np.cos(-theta_Z), -np.sin(-theta_Z), 0.0],
                        [np.sin(-theta_Z),  np.cos(-theta_Z), 0.0],
                        [0.0, 0.0, 1.0]
                    ]
                )
                pitch_vec = [-np.sin(-theta_Z), np.cos(-theta_Z), 0] # Front facing vector (1,0,0) rotated, then perpendicular taken (-y,x,0)
                pitch_rotation_matrix = np.array(
                    [
                        [( (pitch_vec[0]*pitch_vec[0])*(1.0-np.cos(-theta_pitch)) +(np.cos(-theta_pitch))              ), ( (pitch_vec[1]*pitch_vec[0])*(1.0-np.cos(-theta_pitch)) -(np.sin(-theta_pitch)*pitch_vec[2]) ), ( (pitch_vec[2]*pitch_vec[0])*(1.0-np.cos(-theta_pitch)) +(np.sin(-theta_pitch)*pitch_vec[1]) )],
                        [( (pitch_vec[0]*pitch_vec[1])*(1.0-np.cos(-theta_pitch)) +(np.sin(-theta_pitch)*pitch_vec[2]) ), ( (pitch_vec[1]*pitch_vec[1])*(1.0-np.cos(-theta_pitch)) +(np.cos(-theta_pitch)             ) ), ( (pitch_vec[2]*pitch_vec[1])*(1.0-np.cos(-theta_pitch)) -(np.sin(-theta_pitch)*pitch_vec[0]) )],
                        [( (pitch_vec[0]*pitch_vec[2])*(1.0-np.cos(-theta_pitch)) -(np.sin(-theta_pitch)*pitch_vec[1]) ), ( (pitch_vec[1]*pitch_vec[2])*(1.0-np.cos(-theta_pitch)) +(np.sin(-theta_pitch)*pitch_vec[0]) ), ( (pitch_vec[2]*pitch_vec[2])*(1.0-np.cos(-theta_pitch)) +(np.cos(-theta_pitch)             ) )]
                    ]
                )
                xyz_rotated = np.dot( Z_rotation_matrix, np.array([x,y,z]) )    # Apply Z rotation
                xyz_rotated = np.dot( pitch_rotation_matrix, xyz_rotated )      # Apply pitch rotation

                within_radial = (xyz_rotated[1]**2 + xyz_rotated[2]**2) <= r2
                within_width  = abs(xyz_rotated[0]) <= width/2.0
                if(within_radial and within_width):
                    pts[number_of_dipoles][0] = x+1e-20     # Softening factor
                    pts[number_of_dipoles][1] = y+1e-20     #
                    pts[number_of_dipoles][2] = z+1e-20
                    number_of_dipoles += 1
    print(number_of_dipoles," dipoles generated")
    return pts

def rotate_arbitrary(theta, v, n):
    #
    # theta = angle to rotate by (ccw)
    # v = vector to rotate
    # n = axis to rotate about
    #
    arb_rotation_matrix = np.array(
        [
            [( (n[0]*n[0])*(1.0-np.cos(-theta)) +(np.cos(-theta))              ), ( (n[1]*n[0])*(1.0-np.cos(-theta)) -(np.sin(-theta)*n[2]) ), ( (n[2]*n[0])*(1.0-np.cos(-theta)) +(np.sin(-theta)*n[1]) )],
            [( (n[0]*n[1])*(1.0-np.cos(-theta)) +(np.sin(-theta)*n[2]) ), ( (n[1]*n[1])*(1.0-np.cos(-theta)) +(np.cos(-theta)             ) ), ( (n[2]*n[1])*(1.0-np.cos(-theta)) -(np.sin(-theta)*n[0]) )],
            [( (n[0]*n[2])*(1.0-np.cos(-theta)) -(np.sin(-theta)*n[1]) ), ( (n[1]*n[2])*(1.0-np.cos(-theta)) +(np.sin(-theta)*n[0]) ), ( (n[2]*n[2])*(1.0-np.cos(-theta)) +(np.cos(-theta)             ) )]
        ]
    )
    v_rotated = np.dot( arb_rotation_matrix, v )    # Apply rotation
    return v_rotated

def cube_size(args, dipole_radius):
    dipole_diameter = 2*dipole_radius
    cube_radius = args[0]
    num = int(2*cube_radius/dipole_diameter) # mult by 2 for half int lattices
    number_of_dipoles = num**3
    print(number_of_dipoles," dipoles generated")
    return number_of_dipoles

def cube_positions(args, dipole_radius, number_of_dipoles_total):
    # NOTE: cube uses int or half int lattice depending what can fit more dipoles.
    dipole_diameter = 2*dipole_radius
    cube_radius = args[0]
    num = int(2*cube_radius/dipole_diameter)
    pts = np.zeros((number_of_dipoles_total, 3))
    number_of_dipoles = 0
    nums = np.arange(0,num,1)
    for i in nums:
        x = i*dipole_diameter +1e-20
        for j in nums:
            y = j*dipole_diameter +1e-20
            for k in nums:
                z = k*dipole_diameter
                pts[number_of_dipoles] = [x, y, z]
                number_of_dipoles += 1
    pts -= (num-1)/2 * dipole_diameter # shift back to origin. odd num on lattice, even num on half integer lattice.
    # print(f"Z PTS ARE\n{pts[:num, 2]}\n\n") # test the full/half int shift.
    return pts

def simulation(frames, dipole_radius, excel_output, include_force, include_couple, temperature, k_B, inverse_polarizability, beam_collection, viscosity, timestep, number_of_particles, positions, shapes, args, connection_mode, connection_args, constants, force_terms, stiffness_spec, beam_collection_list):
    """
    shapes = List of shape types used
    args   = List of arguments about system and particles; [dipole_radius, particle_parameters]
    particle_parameters; Sphere = [sphere_radius]
                         Torus  = [torus_centre_radius, torus_tube_radius]
    constants = {"spring":..., "bending":..., ...}
          spring = 4e-6 # 5e-7
          bending= 0.25e-18 # 0.5e-18
    
    """
    ####
    ## NOTE; EEK CURRENTLY DISABLED FOR BUGIXING
    ####

    #    MyFileObject = open('anyfile.txt','w')
    #position_vectors = positions(number_of_particles)
    position_vectors = positions    # Of each particle centre
    # print(positions)
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
            case "cylinder":
                dipole_primitive_num[particle_i] = cylinder_size(args[particle_i], dipole_radius)
            case "cube":
                dipole_primitive_num[particle_i] = cube_size(args[particle_i], dipole_radius)
    dipole_primitive_num_total = np.sum(dipole_primitive_num);

    # Check dipole_primitive_num_total is an expected number
    dipole_primitive_num_max = 4000
    if not (dipole_primitive_num_total >= 0 and dipole_primitive_num_total <= dipole_primitive_num_max):
        sys.exit(f"Too many dipoles requested: {dipole_primitive_num_total}.\nMaximum has been set to {dipole_primitive_num_max}. Please raise this cap.")

    # Get dipole primitive positions for each particle
    dipole_primitives = np.zeros( (dipole_primitive_num_total,3), dtype=float)   #Flattened 1D list of all dipoles for all particles
    dpn_start_indices = np.append(0, np.cumsum(dipole_primitive_num[:-1]))
    for particle_i in range(number_of_particles):
        match shapes[particle_i]:
            case "sphere":
                dipole_primitives[dpn_start_indices[particle_i]: dpn_start_indices[particle_i]+dipole_primitive_num[particle_i]] = sphere_positions(args[particle_i], dipole_radius, dipole_primitive_num[particle_i])
            case "torus":
                dipole_primitives[dpn_start_indices[particle_i]: dpn_start_indices[particle_i]+dipole_primitive_num[particle_i]] = torus_sector_positions(args[particle_i], dipole_radius, dipole_primitive_num[particle_i])
            case "cylinder":
                dipole_primitives[dpn_start_indices[particle_i]: dpn_start_indices[particle_i]+dipole_primitive_num[particle_i]] = cylinder_positions(args[particle_i], dipole_radius, dipole_primitive_num[particle_i])
            case "cube":
                dipole_primitives[dpn_start_indices[particle_i]: dpn_start_indices[particle_i]+dipole_primitive_num[particle_i]] = cube_positions(args[particle_i], dipole_radius, dipole_primitive_num[particle_i])
    
    if excel_output==True:
        optpos = np.zeros((frames,number_of_particles,3))
        if include_force==True:
            optforce = np.zeros((frames,number_of_particles,3))
            totforces = np.zeros((frames,number_of_particles,3))
        else:
            optforce = None
        if include_couple==True:
            optcouple = np.zeros((frames,number_of_particles,3))
        else:
            optcouple = None

    # (1) Set constants
    stiffness = constants["spring"]
    BENDING   = constants["bending"]

    # (2) Get Connections
    connection_indices = generate_connection_indices(position_vectors, connection_mode, connection_args)
    # print(f"connection indices are\n{connection_indices}")
    
    # (3) Get Initial Positions
    initial_shape = np.array(position_vectors)
    #print(f"Initial shape is\n{initial_shape}")

    # (4) Get stiffness & natural length matrix
    spring_stiffness_matrix     = generate_stiffness_matrix(number_of_particles, connection_indices, stiffness_spec=stiffness_spec)
    spring_naturallength_matrix = generate_naturallength_matrix(number_of_particles, connection_indices, initial_shape)

    # (5) Get Equilibrium Angles
    ijkangles = get_equilibrium_angles(position_vectors, connection_indices)
    #print(f"Equil. Angles \n{ijkangles}")

    # Finds a characteristic radius for each shape
    effective_radii = np.zeros(number_of_particles, dtype=np.float64)
    for p in range(number_of_particles):
        match shapes[p]:
            case "sphere":
                effective_radii[p] = args[p][0]
            case "torus":
                effective_radii[p] = (args[p][0] + args[p][1])
            case "cylinder":
                effective_radii[p] = (np.sqrt( (args[p][0]/2.0)**2 + (args[p][1])**2 ))  # arg[0] for cylinder is total length
            case "cube":
                effective_radii[p] = args[p][0] * np.sqrt(2)

    # find which particles are connected in objects
    # particle_groups = group_particles_into_objects(number_of_particles, connection_indices)
    # print(f"particle_groups is {particle_groups}")
    particle_neighbours = get_nearest_neighbours(number_of_particles, connection_indices, max_connections_dist=2)

    for i in range(number_of_timesteps):
        #
        # Pass in list of dipole positions to generate total dipole array;
        # All changes inside optical_force_array().
        #
        # optical,couples = optical_force_array(position_vectors, E0, dipole_radius, dipole_primitive)

        # Use translating beam_collections if not None
        if beam_collection_list != None:
            beam_collection = beam_collection_list[i]


        optical, torques, couples = Dipoles.py_optical_force_torque_array(position_vectors, np.asarray(dipole_primitive_num), dipole_radius, dipole_primitives, inverse_polarizability, beam_collection)

        #couples = None
        #include_couple==False
        if excel_output==True:
            for j in range(number_of_particles):
                for k in range(3):
                    optpos[i,j,k] = position_vectors[j][k]
            if include_force==True:
                for j in range(number_of_particles):
                    for k in range(3):
                        optforce[i,j,k] = optical[j][k]
            if include_couple==True:
                for j in range(number_of_particles):
                    for k in range(3):
                        optcouple[i,j,k] = couples[j][k] + torques[j][k]

        if i%10 == 0:
            print("Step ",i)
            #print(i,optical)

        D = diffusion_matrix(position_vectors, effective_radii, k_B, temperature, viscosity)

        total_force_array = np.zeros( (number_of_particles,3), dtype=np.float64 )
        for force_param in force_terms:
            match force_param:
                case "optical":
                    total_force_array += optical
                case "spring":
                    spring = spring_force_array(position_vectors, connection_indices, spring_stiffness_matrix, spring_naturallength_matrix)
                    spring = spring.astype(np.float64)
                    total_force_array += spring
                case "bending":
                    bending = bending_force_array(position_vectors, ijkangles, BENDING)
                    total_force_array += bending
                case "buckingham":
                    buckingham = buckingham_force_array(position_vectors, effective_radii, particle_neighbours)
                    total_force_array += buckingham
                case "driver":
                    driver = driving_force_array(position_vectors, "stretch", args={"driver_magnitude":3.0e-12, "axes":["y"], "influence_distance":1e-10, "initial_positions":initial_shape})      # USED with python DipolesMulti2024Eigen.py 7  
                    total_force_array += driver
                case "gravity":
                    gravity = gravity_force_array(position_vectors, effective_radii[0])
                    total_force_array += gravity
        ##
        ## LEGACY METHOD
        ##
        # gravity = gravity_force_array(position_vectors, effective_radii[0])
        # buckingham = buckingham_force_array(position_vectors, effective_radii, particle_neighbours)
        # driver = driving_force_array(position_vectors, "stretch", args={"driver_magnitude":3.0e-12, "axes":["y"], "influence_distance":1e-10, "initial_positions":initial_shape})      # USED with python DipolesMulti2024Eigen.py 7  
        # driver = driving_force_array(position_vectors, "osc_circ_push", args={"driver_magnitude":1.0e-12, "influence_radius":1.1e-6, "current_frame":i, "frame_period":30})
        # driver = driving_force_array(position_vectors, "timed_circ_push", args={"driver_magnitude":5.0e-12, "influence_radius":1.6e-6, "current_frame":i, "cutoff_frame":10})
        # driver = driving_force_array(position_vectors, "timed_circ_push", args={"driver_magnitude":5.0e-12, "influence_radius":0.5e-6, "current_frame":i, "cutoff_frame":10})
        # bending = bending_force_array(position_vectors, ijkangles, BENDING)
        # NOTE; Initial shape stored earlier before any timesteps are taken
        # spring = spring_force_array(position_vectors, connection_indices, initial_shape, stiffness_spec={"type":"", "default_value":stiffness})
        # total_force_array = bending + spring + optical #+ buckingham# + driver#+ gravity #


        # Record total forces too if required
        if include_force==True:
            for j in range(number_of_particles):
                for k in range(3):
                    totforces[i,j,k] = total_force_array[j][k]
        
        F = np.hstack(total_force_array)
        cov = 2 * timestep * D
        R = np.random.multivariate_normal(mean, cov)
        SumDijFj = (1 / (k_B * temperature)) * np.dot(D, F)
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

        # particles not experiencing mutual Buckingham force are moved apart if overlapping
        stop_particles_overlapping(position_vectors, effective_radii, particle_neighbours)

        vectors_list.append(
            position_vectors
        )  # returns list of position vector arrays of all particles
    for k in range(number_of_timesteps):
        vectors_array[k] = vectors_list[k]
        temp_array1[k] = np.hstack(vectors_array[k])

    xyz_list1 = np.vsplit(np.vstack(temp_array1).T, number_of_particles)

    return xyz_list1,optpos,optforce,optcouple,totforces,connection_indices



###################################################################################
# Start of program
###################################################################################

def main(YAML_name=None, constants={"spring":5e-7, "bending":0.5e-18}, force_terms=["optical", "spring", "bending", "buckingham"], stiffness_spec={"type":"", "default_value":...}):
    #
    # Runs the full program
    # YAML_name = the name (excluding the '.yml') of the YAML file to specify this simulation.
    #             If 'None' is used, main() will read the first terminal arguement as the name instead, e.g. "python DipolesMulti2024Eigen.py <YAML_name>"
    #             If a name is parsed in, then this will be used as the YAML to read instead of sys.argv[1]
    # constants = list of constants that tend to be varied in the simulatiom, which can simply be adjusted for different runs of the simulation
    # force_terms = names of forces to be included in the simualtion, gathered here for convenience when running varying simulations
    #               e.g. "optical", "spring", "bending", "buckingham", "driver", "gravity", ...
    #

    if(YAML_name==None):
        # No name provided, hence use sys.argv[1] as the name
        print("Using YAML: "+str(sys.argv[1])+".yml")
        if int(len(sys.argv)) != 2:
            sys.exit("Usage: python {} <FILESTEM>".format(sys.argv[0]))
    else:
        # Name given, hence use this name provided as the YAML
        print("Using YAML: "+str(YAML_name)+".yml")
        sys.argv[1] = YAML_name

    #===========================================================================
    # Read the yaml file into a system parameter dictionary
    #===========================================================================

    # Check if sys.argv[1] is in the generate presets, else it must be a YAML file name
    preset_filestem = "Preset"
    is_preset_yaml_used = Generate_yaml.generate_yaml(sys.argv[1], preset_filestem)
    if is_preset_yaml_used:
        filestem = preset_filestem

    else:
        filestem = sys.argv[1]

    filename_vtf = filestem+".vtf"
    filename_xl = filestem+".xlsx"
    filename_yaml = filestem+".yml"

    sys_params = ReadYAML.load_yaml(filename_yaml)
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

    
    # Test if beam should be translated each time step (requiring new beam collections each time)
    is_beam_changing = False
    for beam_params in beaminfo.values():
        if "translationargs" in beam_params.keys():
            if beam_params["translationargs"] != "None" and beam_params["translation"] != beam_params["translationargs"]: # python None gets converted to "None" string when read from YAML
                is_beam_changing = True

    # If so, make a list of collections
    if is_beam_changing:
        #
        # translationargs = args for translation
        # translationtype  = regime to translate by e.g. "linear", "circle"
        #       linear => translationargs={x y z}
        #       circle => translationargs={N nx ny nz vx vy vz}
        #               N = number of loops of circle to perform -> can be negative for reverse dir, and fractions for sectors
        #               n = vector of normal to circular plane to move in
        #               v = vector point to centre of circle from inital translation point (will let you determine how to traverse any circle with this point located on it)
        #
        number_of_timesteps = frames
        beam_collection_list = []
        beam_translations = {}
        # Note, translations MUST come in and leave as strings

        for beam_name in beaminfo.keys():
            # Convert strings to float arrays
            if beaminfo[beam_name]["translation"] == "None": # Ensure translation is defined
                print(f"Set beam: {beam_name} translation to [0.0,0.0,0.0] from None")
                beaminfo[beam_name]["translation"] = [0.0,0.0,0.0]
            else:
                beaminfo[beam_name]["translation"] = [float(x) for x in beaminfo[beam_name]["translation"].split()]

            if beaminfo[beam_name]["translationargs"] == "None":
                beaminfo[beam_name]["translationargs"] = beaminfo[beam_name]["translation"]
            else:
                beaminfo[beam_name]["translationargs"] = [float(x) for x in beaminfo[beam_name]["translationargs"].split()]
            
            # Specify points for translation
            match beaminfo[beam_name]["translationtype"]:
                case "linear":
                    # Move in a straight line between two points
                    beam_translations[beam_name] = np.linspace(beaminfo[beam_name]["translation"], beaminfo[beam_name]["translationargs"], number_of_timesteps)
                case "circle":
                    # Move around a circle N times
                    beam_loops  = beaminfo[beam_name]["translationargs"][0]
                    beam_origin    = [ beaminfo[beam_name]["translation"][0], beaminfo[beam_name]["translation"][1], beaminfo[beam_name]["translation"][2] ]
                    beam_normal    = [ beaminfo[beam_name]["translationargs"][1], beaminfo[beam_name]["translationargs"][2], beaminfo[beam_name]["translationargs"][3] ]
                    beam_centreDir = [ beaminfo[beam_name]["translationargs"][4], beaminfo[beam_name]["translationargs"][5], beaminfo[beam_name]["translationargs"][6] ]
                    beam_theta_step= (beam_loops*2.0*np.pi)/(number_of_timesteps)
                    beam_translations[beam_name] = np.zeros((number_of_timesteps,3), dtype=float)
                    for i in range(number_of_timesteps):
                        beam_translations[beam_name][i] = np.array(beam_origin) +np.array(beam_centreDir) -rotate_arbitrary(beam_theta_step*i, beam_centreDir, beam_normal)
                case "point_set":
                    pass
                case _:
                    print("-- YAML 'translationtype' not known;"+str(beaminfo[beam_name]["translationtype"])+" --")

        for t in range(number_of_timesteps):
            for (beam_name, beam_params) in beaminfo.items():
                beaminfo[beam_name]["translation"] = " ".join([str(x) for x in beam_translations[beam_name][t]]) # join floats to a string, translationargs untouched an no longer needed.
            beam_collection_list.append(Beams.create_beam_collection(beaminfo,wavelength))
            
    # Else just use one collection like normal
    else:
        beam_collection_list = None
            
    #n_beams = len(beam_collection)
    #===========================================================================
    # Read particle options and create particle collection
    #===========================================================================
    particle_collection = Particles.ParticleCollection(particleinfo)
    print(f"Number of particles = {particle_collection.num_particles}")
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
    connection_mode = particle_collection.get_connection_mode()
    connection_args = particle_collection.get_connection_args()

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
    particles,optpos, optforces,optcouples,totforces,connection_indices = simulation(frames, dipole_radius, excel_output, include_force, include_couple, temperature, k_B, inverse_polarizability, beam_collection, viscosity, timestep, n_particles, positions, shapes, args, connection_mode, connection_args, constants, force_terms, stiffness_spec, beam_collection_list)
    finalT = time.time()
    print("Elapsed time: {:8.6f} s".format(finalT-initialT))

    # =====================================
    # This code for matplotlib animation output and saving

    #for i in range(optforces.shape[0]):
    #    print("optforces "+str(i)+"= ",optforces[i]);

    if display.show_output==True:
        # Plot beam, particles, forces and tracers (forces and tracers optional)
        fig, ax = None, None                                   #
        fig, ax = display.plot_intensity3d(beam_collection)    # Hash out if beam profile [NOT wanted] <-- For a stationary beam only (will overlay if using translating beam)
        display.animate_system3d(optpos, shapes, args, colors, fig=fig, ax=ax, connection_indices=connection_indices, ignore_coords=[], forces=optforces, include_quiver=True, include_tracer=False, include_connections=True, beam_collection_list=beam_collection_list)



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
        Output.make_excel_file(filename_xl,n_particles,frames,timestep,particles,optpos,include_force,optforces,totforces,include_couple,optcouples)

if __name__ == "__main__":  # To prevent running when imported in other files
    main(constants={"spring":5e-6, "bending":0.1e-18}, force_terms=["optical", "spring", "bending"], stiffness_spec={"type":"", "default_value":5e-6})
    ##
    ## STIFFNESS IS NOW CONTROLLED BY STIFFNES_SPEC, CAN BE MOVED OUT OF CONSTANTS
    ##