#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 16:27:06 2022

@author: phsh
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import Beams


file1 = open('force_xy_0_0.05RCP.txt', 'r')

size = 50

xx = np.zeros((size,size))
yy = np.zeros((size,size))
zz = np.zeros((size,size))

Fx = np.zeros((size,size))
Fy = np.zeros((size,size))
Fz = np.zeros((size,size))

def plot_intensity_xy_contour(nx, ny, num_plots, beam):
    Ex = np.zeros((nx, ny), dtype=complex)
    Ey = np.zeros((nx, ny), dtype=complex)
    Ez = np.zeros((nx, ny), dtype=complex)
    z = np.linspace(2e-6, 2e-6, num_plots)
    I = []
    E = np.zeros(3,dtype=np.complex128)
    fig, ax = plt.subplots(1, num_plots)
    for k in range(num_plots):
        x = np.linspace(lower, upper, nx)
        y = np.linspace(lower, upper, ny)
        for i in range(nx):
            for j in range(ny):
#                Beams.all_incident_fields((x[i], y[j], z[k]), beam_collection, E)
                Beams.all_incident_fields((x[i], y[j], 0.0), beam_collection, E)
                Ex[i][j] = E[0]
                Ey[i][j] = E[1]
                Ez[i][j] = E[2]

        X, Y = np.meshgrid(x, y, indexing="ij")
        I.append(np.square(np.abs(Ex)) + np.square(np.abs(Ey)) + np.square(np.abs(Ez)))

        I0 = np.max(I)
        if num_plots > 1:
            ax[k].plot_surface(
                X, Y, I[k] / I0, cmap=cm.coolwarm, linewidth=0, antialiased=False
            )
            ax[k].set_xlabel("x / wavelength")
            ax[k].set_ylabel("y / wavelength")
            ax[k].set_zlabel("Relative Intensity")
            ax[k].set_zlim(0, 1)
            ax[k].set_title("z = {:.1e}".format(z[k]))

        else:
#            ax.axis('equal')
            ax.set_aspect('equal','box')
            cs=ax.contourf(X, Y, I[k], cmap=cm.summer, levels=30)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            cbar = fig.colorbar(cs)
            # ax.set_zlabel("Relative Intensity")
            # ax.set_zlim(0, 1)
            # ax.set_title("z = {:.1e}".format(z[k]))
    return ax
    # plt.show()



for j in range(50):
    line = file1.readline()
    for i in range(50):
        fields = file1.readline().split(' ')
        xx[j,i] = float(fields[0])
        yy[j,i] = float(fields[1])
        zz[j,i] = float(fields[2])
        Fx[j,i] = float(fields[3])
        Fy[j,i] = float(fields[4])
        Fz[j,i] = float(fields[5])
  
file1.close()
lower = xx.min()
upper = xx.max()

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

E0 = 3e6 

###################################################################################
# New code for BEAM class
###################################################################################
n_beams = 2
beam_collection = np.zeros(n_beams,dtype=object)
mybeam = Beams.BEAM()
kk = 2*np.pi / wavelength
kt_by_kz = 0.05  # ratio of transverse to longitudinal wavevector, kz currently set to 2pi/wavelength (in general_bessel_constants)
kz = kk / np.sqrt(1+kt_by_kz**2)
kt = kt_by_kz*kz
order = 0
mybeam.kz = kz
mybeam.kt = kt
mybeam.kt_by_kz = kt_by_kz
mybeam.E0 = E0
#mybeam.beamtype = Beams.BEAMTYPE_PLANE
#mybeam.beamtype = Beams.BEAMTYPE_GAUSS_BARTON5
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
jones_matrix[1][1] = 1/np.sqrt(2)  # imaginary part
mybeam.jones = np.ctypeslib.as_ctypes(jones_matrix.flatten())
#
# Beam orientation matrix
# Beam is by default parallel to z.  Take a rotation about x axis, keeping beam in z-y plane
# with final axis parallel to x.
#
angle = 90.0 # degrees (+ve in anticlockwise sense)
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
kt_by_kz = 0.1  # ratio of transverse to longitudinal wavevector, kz currently set to 2pi/wavelength (in general_bessel_constants)
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
jones_matrix[1][1] = 1/np.sqrt(2)  # imaginary part
mybeam.jones = np.ctypeslib.as_ctypes(jones_matrix.flatten())
#
# Beam orientation matrix
# Beam is by default parallel to z.  Take a rotation about x axis, keeping beam in z-y plane
# with final axis parallel to x.
#
angle = 90.0 # degrees (+ve in anticlockwise sense)
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
beam_collection[1] = mybeam"""
###################################################################################


ax = plot_intensity_xy_contour(100, 100, 1, "beam")
ax.quiver(xx, yy, Fx*3, Fy*3, units="x")
ax.set_title("CP Bessel Beam Order {:d}".format(order))
