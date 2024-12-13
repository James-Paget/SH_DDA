# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 12:52:31 2022
Animated trajectories
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
import ctypes
import datetime
import matplotlib.animation as animation

def init():
    for trajectory in trajectories:
        trajectory.set_data([], [])
    return trajectories


def animate(fff):
    frames = fff * frame_interval
    for trajectory, particle in zip(trajectories, particles):
        trajectory.set_data(
            particle[0, 0 : frames], particle[1, 0 : frames])
#            particle[0, frames - 2 : frames], particle[1, frames - 2 : frames]
#        )  # change which axes youre looking at here
    #        if frames<1300:
    #            trajectory.set_data(particle[2, 0:frames],particle[1, 0:frames])
    #        else:
    #            trajectory.set_data(particle[2, frames-1300:frames],particle[1, frames-1300:frames])
    return trajectories

def plot_intensity_xy_contour(nx, ny, num_plots, beam):
    Ex = np.zeros((nx, ny), dtype=complex)
    Ey = np.zeros((nx, ny), dtype=complex)
    Ez = np.zeros((nx, ny), dtype=complex)
    z = np.linspace(2e-6, 2e-6, num_plots)
    I = []
    E = np.zeros(3,dtype=np.complex128)
    #fig, ax = plt.subplots(1, num_plots)
    for k in range(num_plots):
        x = np.linspace(lower, upper, nx)
        y = np.linspace(lower, upper, ny)
        for i in range(nx):
            for j in range(ny):
#                Beams.all_incident_fields((x[i], y[j], z[k]), beam_collection, E)
                Beams.all_incident_fields((x[i], y[j], z_offset), beam_collection, E)
#                Beams.all_incident_fields((x[i], y[j], 0.0), beam_collection, E)
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
#            cs=ax.contourf(X, Y, I[k], cmap=cm.gray, levels=30)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            cbar = fig.colorbar(cs)
            # ax.set_zlabel("Relative Intensity")
            # ax.set_zlim(0, 1)
            # ax.set_title("z = {:.1e}".format(z[k]))
    return ax
    # plt.show()


###################################################################################
# Start of program
###################################################################################

if int(len(sys.argv)) != 2:
    sys.exit("Usage: python {} <FILESTEM>".format(sys.argv[0]))
else:
    filestem = sys.argv[1]
    filename_vtf = filestem+".vtf"


###################################################################################
# Read in the vtf file header
###################################################################################
MyFileObject = open(filename_vtf,"r")
for i in range(6):
    myline = MyFileObject.readline()
    print(myline[0:-1])
fields = MyFileObject.readline().split(' ')
n_beams = int(fields[-1])
print("# Number of beams: {:d}".format(n_beams))
fields = MyFileObject.readline().split(' ')
n_particles = int(fields[-1])
print("# Number of particles: {:d}".format(n_particles))
fields = MyFileObject.readline().split(' ')
radius = float(fields[-1])
print("# Particle radius (m): {:e}".format(radius))
fields = MyFileObject.readline().split(' ')
dipole_radius = float(fields[-1])
print("# Dipole radius (m): {:e}".format(dipole_radius))
fields = MyFileObject.readline().split(' ')
z_offset = float(fields[-1])
print("# z-offset for plot (m): {:e}".format(z_offset))
myline = MyFileObject.readline()
print(myline[0:-1])
fields = MyFileObject.readline().split(' ')
frames = int(fields[-1])
print("# Number of timesteps: {:d}".format(frames))
fields = MyFileObject.readline().split(' ')
timestep = float(fields[-1])
print("# Time step (s): {:e}".format(timestep))
for i in range(8):
    myline = MyFileObject.readline()
    print(myline[0:-1])

###################################################################################
# Read in and define the beams
###################################################################################
beam_collection = np.zeros(n_beams,dtype=object)
for i in range(n_beams):
    myline = MyFileObject.readline()
    print(myline[0:-1])
    fields = MyFileObject.readline().split(' ')
    beamtype = int(fields[-1])
    print("#  -beamtype = {:d}".format(beamtype))
    fields = MyFileObject.readline().split(' ')
    E0 = float(fields[-1])
    print("#  -E0 = {:f}".format(E0))
    fields = MyFileObject.readline().split(' ')
    kk = float(fields[-1])
    print("#  -k = {:f}".format(kk))
    fields = MyFileObject.readline().split(' ')
    kz = float(fields[-1])
    print("#  -kz = {:f}".format(kz))
    fields = MyFileObject.readline().split(' ')
    kt = float(fields[-1])
    print("#  -kt = {:f}".format(kt))
    fields = MyFileObject.readline().split(' ')
    kt_by_kz = float(fields[-1])
    print("#  -kt_by_kz = {:f}".format(kt_by_kz))
    fields = MyFileObject.readline().split(' ')
    order = int(fields[-1])
    print("#  -order = {:d}".format(order))
    jones = np.zeros(4,dtype=np.float64)
    fields = MyFileObject.readline().split(' ')
    for j in range(4):
        jones[j] = float(fields[j-4])
    print("#  -jones = {:f} {:f} {:f} {:f}".format(jones[0], jones[1], jones[2], jones[3]))
    translation = np.zeros(3,dtype=np.float64)
    fields = MyFileObject.readline().split(' ')
    for j in range(3):
        translation[j] = float(fields[j-3])
    print("#  -translation = {:e} {:e} {:e}".format(translation[0], translation[1], translation[2]))
    rotation = np.zeros(9,dtype=np.float64)
    fields = MyFileObject.readline().split(' ')
    for j in range(9):
        rotation[j] = float(fields[j-9])
    print("#  -rotation = {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(rotation[0], rotation[1], rotation[2], rotation[3], rotation[4], rotation[5], rotation[6], rotation[7], rotation[8]))
    fields = MyFileObject.readline().split(' ')
    w0 = float(fields[-1])
    print("#  -w0 = {:e}".format(w0))
    myline = MyFileObject.readline()
    print(myline[0:-1])
    beam_collection[i] = Beams.make_beam(beamtype, E0, kk, kt_by_kz=kt_by_kz, order=order, w0=w0, jones=jones, rotation=rotation, translation=translation)

###################################################################################
# Plot the field intensity for the beam configuration
###################################################################################
fig = plt.figure()
#
lower = -0.25e-5
upper = -lower

ax = plt.axes(xlim=(lower, upper), ylim=(lower, upper))
plot_intensity_xy_contour(200, 200, 1, " ")
ax.set_aspect('equal','box')
#plt.show()

###################################################################################
# Read the atom types and set up the colours
###################################################################################
particle_spec = {'Silicon': [3.9, '#ffc107'], 'Sapphire': [2.5, '#d81b60'],'Diamond': [2.417, '#d81b60'], 'SiN': [2.046, '#66ddbb'], 'Glass': [1.5, '#1e88e5'],'Low': [1.446, '#8e8e8e'],'FusedSilica': [1.458, '#8e8e8e']}
particle_types = ['Silicon'] * n_particles
ref_ind = np.ones(n_particles)
colors = np.ndarray(n_particles,dtype=np.object)

for i in range(n_particles):
    myline = MyFileObject.readline()
    atomtype = myline[-2]
#    print("Atom {:s}".format(atomtype))
    if atomtype == 'S':
        particle_types[i] = 'Silicon'
        ref_ind[i] = particle_spec['Silicon'][0]
        colors[i] = particle_spec['Silicon'][1]
        if i/2 == i//2:
            colors[i] = '#e0a105'
        else:
            colors[i] = '#ffe10a'
    elif atomtype == 'O':
        particle_types[i] = 'Sapphire'
        ref_ind[i] = particle_spec['Sapphire'][0]
        colors[i] = particle_spec['Sapphire'][1]
        if i/2 == i//2:
            colors[i] = '#b01240'
        else:
            colors[i] = '#e81e80'
    elif atomtype == 'C':
        particle_types[i] = 'Diamond'
        ref_ind[i] = particle_spec['Diamond'][0]
        colors[i] = particle_spec['Diamond'][1]
#        if i/2 == i//2:
#            colors[i] = '#b01240'
#        else:
#            colors[i] = '#e81e80'
    elif atomtype == 'N':
        particle_types[i] = 'Glass'
        ref_ind[i] = particle_spec['Glass'][0]
        colors[i] = particle_spec['Glass'][1]
        if i/2 == i//2:
            colors[i] = '#1870d0'
        else:
            colors[i] = '#2298f5'
    elif atomtype == 'H':
        particle_types[i] = 'Low'
        ref_ind[i] = particle_spec['Low'][0]
        colors[i] = particle_spec['Low'][1]
        if i/2 == i//2:
            colors[i] = '#7e7e7e'
        else:
            colors[i] = '#aeaeae'
    elif atomtype == 'B':
        particle_types[i] = 'FusedSilica'
        ref_ind[i] = particle_spec['FusedSilica'][0]
        colors[i] = particle_spec['FusedSilica'][1]
        if i/2 == i//2:
            colors[i] = '#7e7e7e'
        else:
            colors[i] = '#aeaeae'
        if i/2 == i//2:
            colors[i] = '#1870d0'
            colors[i] = '#2298f5'
        else:
            colors[i] = '#2298f5'
            colors[i] = '#1870d0'
    elif atomtype == 'P':
        particle_types[i] = 'SiN'
        ref_ind[i] = particle_spec['SiN'][0]
        colors[i] = particle_spec['SiN'][1]
#        if i/2 == i//2:
#            colors[i] = '#7e7e7e'
#        else:
#            colors[i] = '#aeaeae'
    else:
        print("Error - atom type not recognised",atomtype)

print("# Particle types: ",particle_types)
print("# Refractive indices: ", ref_ind)
marker_size = 1.25*10.0*(radius/200e-9)*(5e-6/upper) # Size 10 for 200nm radius and upper limit 5 microns.

###################################################################################
# Set up arrays and read in the coordinates
###################################################################################
particles = np.zeros((n_particles,3,frames),dtype=np.float64)

for i in range(frames):
    myline = MyFileObject.readline()
    myline = MyFileObject.readline()
    myline = MyFileObject.readline()
    for j in range(n_particles):
        fields = MyFileObject.readline().split(' ')
        for k in range(3):
            particles[j][k][i] = float(fields[k])*1e-6
            #print(j,k,i,particles[j][k][i])

MyFileObject.close()  # closes the file again

###################################################################################
# Do the animation
###################################################################################

print(particles)
frame_interval = 10000

trajectories = [
    ax.plot([], [], markersize=marker_size/50, marker=".", c=colors[i], animated=True)[0]
#    ax.plot([], [], markersize=marker_size, marker="o", c=colors[i], mec='white', mew=0.75, alpha=1, animated=True)[0]
    for i in np.arange(n_particles)
]

ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=frames // frame_interval, interval=25, blit=True
)
plt.show()

# writer = animation.PillowWriter(fps=30)

# ani.save("bessel-ang-mom-test.gif", writer=writer)
# =====================================

