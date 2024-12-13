from matplotlib.colors import Normalize
import numpy as np
from numpy import sin, cos, pi, arctan2, sqrt
import matplotlib.pyplot as plt
import cmath
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.special import j0, j1, jvp, jv
import Beams
import ctypes

c = 3e8



def optical_force(gradE_transpose, E, alpha):
    """Calulates the optical force from the TRANSPOSE of the gradient of the  field"""
    Force = np.zeros(3)
    p = alpha * E
    Force[0] = (1 / 2) * np.real(
        p[0] * gradE_transpose[0, 0]
        + p[1] * gradE_transpose[0, 1]
        + p[2] * gradE_transpose[0, 2]
    )
    Force[1] = (1 / 2) * np.real(
        p[0] * gradE_transpose[1, 0]
        + p[1] * gradE_transpose[1, 1]
        + p[2] * gradE_transpose[1, 2]
    )
    Force[2] = (1 / 2) * np.real(
        p[0] * gradE_transpose[2, 0]
        + p[1] * gradE_transpose[2, 1]
        + p[2] * gradE_transpose[2, 2]
    )
    return Force


def plot_intensity_xy(nx, ny, num_plots, beam):
    Ex = np.zeros((nx, ny), dtype=complex)
    Ey = np.zeros((nx, ny), dtype=complex)
    Ez = np.zeros((nx, ny), dtype=complex)
    z = np.linspace(2e-6, 2e-6, num_plots)
    I = []
    E = []
    fig, ax = plt.subplots(1, num_plots, subplot_kw={"projection": "3d"})
    for k in range(num_plots):

        if beam == "gaussian":
            x = np.linspace(-20e-5, 20e-5, nx)
            y = np.linspace(-20e-5, 20e-5, ny)
            # z = np.linspace(0,0,nx)
            for l in range(n_beams):
                x_prime, y_prime, z_prime = coord_transformation(
                    (x, y, z), beam_angles[l], beam_positions[l]
                )
                for i in range(nx):
                    for j in range(ny):
                        Ex[i][j] = gaussian_E_x(
                            x_prime[i], y_prime[j], z[k], E0, wavelength, w0
                        )
                        Ey[i][j] = gaussian_E_y(
                            x_prime[i], y_prime[j], z[k], E0, wavelength, w0
                        )
                        Ez[i][j] = gaussian_E_z(
                            x_prime[i], y_prime[j], z[k], E0, wavelength, w0
                        )
                E.append(np.array([Ex, Ey, Ez]))
            E_tot = np.sum(E, axis=0)


        elif beam == "general bessel":
            x = np.linspace(lower, upper, nx)
            y = np.linspace(lower, upper, ny)
            for i in range(nx):
                for j in range(ny):
                    Ex[i][j] = general_bessel_E_x(
                        x[i], y[j], z[k], E0, wavelength, order
                    )
                    Ey[i][j] = general_bessel_E_y(
                        x[i], y[j], z[k], E0, wavelength, order
                    )
                    Ez[i][j] = general_bessel_E_z(
                        x[i], y[j], z[k], E0, wavelength, order
                    )


        X, Y = np.meshgrid(x / wavelength, y / wavelength, indexing="ij")
        # Ex, Ey, Ez = E_tot[0], E_tot[1], E_tot[2]
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
            ax.plot_surface(
                X, Y, I[k] / I0, cmap=cm.coolwarm, linewidth=0, antialiased=False
            )
            ax.set_xlabel("x / wavelength")
            ax.set_ylabel("y / wavelength")
            ax.set_zlabel("Relative Intensity")
            ax.set_zlim(0, 1)
            ax.set_title("z = {:.1e}".format(z[k]))

    plt.show()


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


def intensity_xy_animation(nx, ny, frames, beam):
    x = np.linspace(-20e-6, 20e-6, nx)
    y = np.linspace(-20e-6, 20e-6, ny)
    Ex = np.zeros((nx, ny), dtype=complex)
    Ey = np.zeros((nx, ny), dtype=complex)
    Ez = np.zeros((nx, ny), dtype=complex)
    z = np.linspace(-100e-6, 100e-6, frames)
    I = []

    for k in range(frames):
        if beam == "gaussian":
            for i in range(nx):
                for j in range(ny):
                    Ex[i][j] = gaussian_E_x(x[i], y[j], z[k], E0, wavelength, w0)
                    Ey[i][j] = gaussian_E_y(x[i], y[j], z[k], E0, wavelength, w0)
                    # Ez[i][j] = gaussian_E_z(x[i], y[j], z[k], E0, wavelength, w0)


        X, Y = np.meshgrid(x / wavelength, y / wavelength)
        I.append(np.square(np.abs(Ex)) + np.square(np.abs(Ey)) + np.square(np.abs(Ez)))

        I0 = np.max(I)

    return X, Y, I / I0, z


def update(i):
    """
    Update function tells the animator what to change in every frame.
    """
    ax.clear()
    surf = ax.plot_surface(X, Y, I[i], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title("z = {:.1e}".format(z[i]))
    ax.set_zlim(0, 1)
    return surf


def plot_intensity_xz(nx, nz, y):

    x = np.linspace(-2e-6, 2e-6, nx)
    z = np.linspace(-4e-6, 4e-6, nz)
    Ex = np.zeros((nx, nz), dtype=complex)
    Ey = np.zeros((nx, nz), dtype=complex)
    Ez = np.zeros((nx, nz), dtype=complex)

    for i in range(nx):
        for j in range(nz):
            Ex[i][j] = gaussian_E_x(x[i], y, z[j], E0, wavelength, w0)
            Ey[i][j] = gaussian_E_y(x[i], y, z[j], E0, wavelength, w0)
            Ez[i][j] = gaussian_E_z(x[i], y, z[j], E0, wavelength, w0)

    X, Z = np.meshgrid(x / wavelength, z / wavelength)
    I = np.square(np.abs(Ex)) + np.square(np.abs(Ey) + np.square(np.abs(Ez)))
    I0 = np.max(I)
    fig, ax = plt.subplots(
        subplot_kw={"projection": "3d"}
    )  # Intensity plot of a cross section of the beam at z
    ax.plot_surface(X, Z, I / I0, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel("x/wavelength")
    ax.set_ylabel("z/wavelength")
    plt.show()


def force_z_curve(x_position, y_position, z_start, z_end, num, beam):
    z_position = np.linspace(z_start, z_end, num)
    Fx_distance_array = []
    Fy_distance_array = []
    Fz_distance_array = []
    E = np.zeros(3,dtype=np.complex128)
    gradE = np.zeros((3,3),dtype=np.complex128)

    for i in range(num):
        Beams.all_incident_fields((x_position, y_position, z_position[i]), beam_collection, E)
        Beams.all_incident_field_gradients((x_position, y_position, z_position[i]), beam_collection, gradE)
        force_on_particle = optical_force(gradE.T, E, a)
        Fx = force_on_particle[0]
        Fy = force_on_particle[1]
        Fz = force_on_particle[2]
        Fx_distance_array.append(Fx)
        Fy_distance_array.append(Fy)
        Fz_distance_array.append(Fz)

    return z_position, Fx_distance_array, Fy_distance_array, Fz_distance_array


def force_y_curve(x_position, z_position, y_start, y_end, num, beam):
    y_position = np.linspace(y_start, y_end, num)
    Fx_distance_array = []
    Fy_distance_array = []
    Fz_distance_array = []
    E = np.zeros(3,dtype=np.complex128)
    gradE = np.zeros((3,3),dtype=np.complex128)

    for i in range(num):
        Beams.all_incident_fields((x_position, y_position[i], z_position), beam_collection, E)
        Beams.all_incident_field_gradients((x_position, y_position[i], z_position), beam_collection, gradE)
        force_on_particle = optical_force(gradE.T, E, a)
        Fx = force_on_particle[0]
        Fy = force_on_particle[1]
        Fz = force_on_particle[2]
        Fx_distance_array.append(Fx)
        Fy_distance_array.append(Fy)
        Fz_distance_array.append(Fz)

    return y_position, Fx_distance_array, Fy_distance_array, Fz_distance_array


def force_x_curve(y_position, z_position, x_start, x_end, num, beam):
    """scans along x and calulates forces in x,y,z directions"""
    x_position = np.linspace(x_start, x_end, num)
    Fx_distance_array = []
    Fy_distance_array = []
    Fz_distance_array = []
    E = np.zeros(3,dtype=np.complex128)
    gradE = np.zeros((3,3),dtype=np.complex128)

    for i in range(num):
        Beams.all_incident_fields((x_position[i], y_position, z_position), beam_collection, E)
        Beams.all_incident_field_gradients((x_position[i], y_position, z_position), beam_collection, gradE)
        force_on_particle = optical_force(gradE.T, E, a)
        Fx = force_on_particle[0]
        Fy = force_on_particle[1]
        Fz = force_on_particle[2]
        Fx_distance_array.append(Fx)
        Fy_distance_array.append(Fy)
        Fz_distance_array.append(Fz)

    return x_position, Fx_distance_array, Fy_distance_array, Fz_distance_array


def force_distance_plots(beam):
    # order 100 for bessel beam, order 1e-5 for gaussian
    x, fxx, fxy, fxz = force_x_curve(0, 0, -5e-6, 5e-6, 500, beam)
    y, fyx, fyy, fyz = force_y_curve(0, 0, -5e-6, 5e-6, 500, beam)
    z, fzx, fzy, fzz = force_z_curve(1e-19, 1e-19, -5e-6, 5e-6, 500, beam)

    fig, ax = plt.subplots(1, 3)
    ax[0].set_xlabel("Distance in x direction")
    ax[1].set_xlabel("Distance in y direction")
    ax[2].set_xlabel("Distance in z direction")
    ax[0].set_ylabel("Force in x direction")
    ax[1].set_ylabel("Force in y direction")
    ax[2].set_ylabel("Force in z direction")
    ax[0].plot(x, fxx)
    ax[1].plot(y, fyy)
    ax[2].plot(z, fzz)
    for label in ax[0].get_xaxis().get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax[1].get_xaxis().get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax[2].get_xaxis().get_ticklabels()[::2]:
        label.set_visible(False)

    # fig, ax = plt.subplots(1, 2)
    # ax[0].set_xlabel("Distance in x direction")
    # ax[1].set_xlabel("Distance in x direction")
    # ax[0].set_ylabel("Force in x direction")
    # ax[1].set_ylabel("Force in y direction")
    # ax[0].plot(x, fxx)
    # ax[1].plot(x, fxy)
    # for label in ax[0].get_xaxis().get_ticklabels()[::2]:
    #     label.set_visible(False)
    # for label in ax[1].get_xaxis().get_ticklabels()[::2]:
    #     label.set_visible(False)
    # plt.tight_layout()

    # plt.plot(y, fyy)
    # plt.plot(y, fyx)

    plt.show()



def force_tranform(Fx, Fy, x, y):
    """Transforms Fx and Fy components to Fphi component, and expresses this in terms of x and y unit vectors for plotting"""
    phi = arctan2(y, x)
    fx = -Fx * sin(phi)
    fy = Fy * cos(phi)
    F_phi = -Fx * sin(phi) + Fy * cos(phi)
    Fx_prime = F_phi * -sin(phi)
    Fy_prime = F_phi * cos(phi)
    return Fx_prime, Fy_prime


def force_map(beam):
    nx = 51
    ny = 51
    x = np.linspace(lower, upper, nx)
    y = np.linspace(lower, upper, ny)
    z = 0.0#4e-3
    Fx = np.zeros((nx, ny))
    Fy = np.zeros((nx, ny))
    Fx_azimuthal = np.zeros((nx, ny))
    Fy_azimuthal = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
#            gradE = incident_field_gradient(beam, (x[i], y[j], z))
            gradE = np.zeros((3,3),dtype=np.complex128)
            Beams.all_incident_field_gradients((x[i], y[j], z), beam_collection, gradE)
#            E = incident_field(beam, (x[i], y[j], z))
            E = np.zeros(3,dtype=np.complex128)
            Beams.all_incident_fields((x[i], y[j], z), beam_collection, E)
            print(E)
            Force = optical_force(gradE.T, E, a)
            #print(x[i],y[j],Force)
            Fx[i][j] = Force[0]  # [j][i] due to how plt.quiver plots the coordinates
            Fy[i][j] = Force[1]
            Fx_temp, Fy_temp = force_tranform(Force[0], Force[1], x[i], y[j])
            Fx_azimuthal[i][j] = Fx_temp
            Fy_azimuthal[i][j] = Fy_temp

    X, Y = np.meshgrid(x, y, indexing="ij")
    ax = plot_intensity_xy_contour(201, 201, 1, beam)
    #ax.quiver(X, Y, Fx_azimuthal, Fy_azimuthal, units="x")
#    ax.quiver(X, Y, Fx, Fy, units="x")
    if beam_collection[0].beamtype == Beams.BEAMTYPE_GAUSS_CSP:
        ax.set_title("CSP Gaussian beam width {:f}".format(beam_collection[0].w0))
    else:
        ax.set_title("CP Bessel Beam Order {:d}".format(order))
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.show()
    return


def coord_transformation(coords, angles, trans):
    """a is angle around x axis, b is angle around y axis, g is angle around z axis"""
    a, b, g = angles[0], angles[1], angles[2]
    dx, dy, dz = trans[0], trans[1], trans[2]

    R_z = np.array(((cos(g), -sin(g), 0), (sin(g), cos(g), 0), (0, 0, 1)))
    R_y = np.array(((cos(b), 0, sin(b)), (0, 1, 0), (-sin(b), 0, cos(b))))
    R_x = np.array(((1, 0, 0), (0, cos(a), -sin(a)), (0, sin(a), cos(a))))

    rotation = R_z @ R_y @ R_x
    rotated_coords = rotation @ np.array(coords, dtype=object)

    x_prime, y_prime, z_prime = rotated_coords[0], rotated_coords[1], rotated_coords[2]

    x_prime = x_prime + dx
    y_prime = y_prime + dy
    z_prime = z_prime + dz

    # x_prime, y_prime, z_prime = (
    # np.round(x_prime, 10),
    # np.round(y_prime, 10),
    # np.round(z_prime, 10),
    # )

    # Ex = gaussian_E_x(x_prime, y_prime, z_prime, E0, wavelength, w0)
    # Ey = gaussian_E_y(x_prime, y_prime, z_prime, E0, wavelength, w0)
    # Ez = gaussian_E_z(x_prime, y_prime, z_prime, E0, wavelength, w0)
    # return Ex, Ey, Ez
    return x_prime, y_prime, z_prime


wavelength = 1.0e-6
w0 = wavelength / np.pi  # breaks down when w0 < wavelength
#w0 = wavelength * 1.2  # breaks down when w0 < wavelength
#w0 = wavelength * 1.0  # breaks down when w0 < wavelength
alpha_by_k = 0.5

n1 = 1.446#3.9
ep1 = n1 * n1
ep2 = 1.333
radius = 200e-9  # half a micron to one micron
water_permittivity = 80.4
k = 2 * np.pi / wavelength
a0 = (4 * np.pi * 8.85e-12) * (radius ** 3) * ((ep1 - ep2) / (ep1 + 2*ep2))
a = a0 / (1 - (2 / 3) * 1j * k ** 3 * a0)  # complex form from Chaumet (2000)
# a = a0

jones_vector = np.zeros(2,dtype=complex)
jones_vector[0] = (1 + 0j) / sqrt(2)
jones_vector[1] = (0 + 1j) / sqrt(2)  # change the polarisation of the beam
E0 = 3e6

###################################################################################
# New code for BEAM class
###################################################################################
n_beams = 5
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
#mybeam.beamtype = Beams.BEAMTYPE_PLANE
#mybeam.beamtype = Beams.BEAMTYPE_GAUSS_BARTON5
#mybeam.beamtype = Beams.BEAMTYPE_BESSEL
mybeam.beamtype = Beams.BEAMTYPE_GAUSS_CSP
mybeam.order = order
mybeam.w0 = w0
mybeam.k = kk
#
# Build the Jones matrix
#
jones_matrix = np.zeros((2,2),dtype=np.float64)
jones_matrix[0][0] = 1/sqrt(2)  # real part
jones_matrix[0][1] = 0          # imaginary part
jones_matrix[1][0] = 0          # real part
jones_matrix[1][1] = 1/sqrt(2)  # imaginary part
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
beamposition = np.array((-2.0e-6,0.0,0.0),dtype=np.float64) # specify position in metres
mybeam.translation = np.ctypeslib.as_ctypes(beamposition)
#
# Store in collection
#
beam_collection[0] = mybeam
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
#mybeam.beamtype = Beams.BEAMTYPE_PLANE
#mybeam.beamtype = Beams.BEAMTYPE_GAUSS_BARTON5
#mybeam.beamtype = Beams.BEAMTYPE_BESSEL
mybeam.beamtype = Beams.BEAMTYPE_GAUSS_CSP
mybeam.order = order
mybeam.w0 = w0
mybeam.k = kk
#
# Build the Jones matrix
#
jones_matrix = np.zeros((2,2),dtype=np.float64)
jones_matrix[0][0] = 1/sqrt(2)  # real part
jones_matrix[0][1] = 0          # imaginary part
jones_matrix[1][0] = 0          # real part
jones_matrix[1][1] = 1/sqrt(2)  # imaginary part
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
beamposition = np.array((-1.0e-6,0.0,0.0),dtype=np.float64) # specify position in metres
mybeam.translation = np.ctypeslib.as_ctypes(beamposition)
#
# Store in collection
#
beam_collection[1] = mybeam
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
#mybeam.beamtype = Beams.BEAMTYPE_PLANE
#mybeam.beamtype = Beams.BEAMTYPE_GAUSS_BARTON5
#mybeam.beamtype = Beams.BEAMTYPE_BESSEL
mybeam.beamtype = Beams.BEAMTYPE_GAUSS_CSP
mybeam.order = order
mybeam.w0 = w0
mybeam.k = kk
#
# Build the Jones matrix
#
jones_matrix = np.zeros((2,2),dtype=np.float64)
jones_matrix[0][0] = 1/sqrt(2)  # real part
jones_matrix[0][1] = 0          # imaginary part
jones_matrix[1][0] = 0          # real part
jones_matrix[1][1] = 1/sqrt(2)  # imaginary part
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
beamposition = np.array((0.0e-6,0.0,0.0),dtype=np.float64) # specify position in metres
mybeam.translation = np.ctypeslib.as_ctypes(beamposition)
#
# Store in collection
#
beam_collection[2] = mybeam
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
#mybeam.beamtype = Beams.BEAMTYPE_PLANE
#mybeam.beamtype = Beams.BEAMTYPE_GAUSS_BARTON5
#mybeam.beamtype = Beams.BEAMTYPE_BESSEL
mybeam.beamtype = Beams.BEAMTYPE_GAUSS_CSP
mybeam.order = order
mybeam.w0 = w0
mybeam.k = kk
#
# Build the Jones matrix
#
jones_matrix = np.zeros((2,2),dtype=np.float64)
jones_matrix[0][0] = 1/sqrt(2)  # real part
jones_matrix[0][1] = 0          # imaginary part
jones_matrix[1][0] = 0          # real part
jones_matrix[1][1] = 1/sqrt(2)  # imaginary part
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
beamposition = np.array((1.0e-6,0.0,0.0),dtype=np.float64) # specify position in metres
mybeam.translation = np.ctypeslib.as_ctypes(beamposition)
#
# Store in collection
#
beam_collection[3] = mybeam
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
#mybeam.beamtype = Beams.BEAMTYPE_PLANE
#mybeam.beamtype = Beams.BEAMTYPE_GAUSS_BARTON5
#mybeam.beamtype = Beams.BEAMTYPE_BESSEL
mybeam.beamtype = Beams.BEAMTYPE_GAUSS_CSP
mybeam.order = order
mybeam.w0 = w0
mybeam.k = kk
#
# Build the Jones matrix
#
jones_matrix = np.zeros((2,2),dtype=np.float64)
jones_matrix[0][0] = 1/sqrt(2)  # real part
jones_matrix[0][1] = 0          # imaginary part
jones_matrix[1][0] = 0          # real part
jones_matrix[1][1] = 1/sqrt(2)  # imaginary part
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
beamposition = np.array((2.0e-6,0.0,0.0),dtype=np.float64) # specify position in metres
mybeam.translation = np.ctypeslib.as_ctypes(beamposition)
#
# Store in collection
#
beam_collection[4] = mybeam

"""
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
mybeam.beamtype = Beams.BEAMTYPE_GAUSS_CSP
mybeam.order = order
mybeam.w0 = w0
mybeam.k = kk
#
# Build the Jones matrix
#
jones_matrix = np.zeros((2,2),dtype=np.float64)
jones_matrix[0][0] = 1/sqrt(2)  # real part
jones_matrix[0][1] = 0          # imaginary part
jones_matrix[1][0] = 0          # real part
jones_matrix[1][1] = -1/sqrt(2)  # imaginary part
mybeam.jones = np.ctypeslib.as_ctypes(jones_matrix.flatten())
#
# Beam orientation matrix
# Beam is by default parallel to z.  Take a rotation about x axis, keeping beam in z-y plane
# with final axis parallel to x.
#
angle = -90.0 # degrees (+ve in anticlockwise sense)
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
"""



beam_angles = np.array(
    ((0, 0, 0), (0, 0, 0))
)  # each vector represents each beam, the angles are written as (angle around x axis, y..., z...) in radians
beam_positions = np.array(((0, 0, 0), (0, 0, 0)))

lower = -5e-6
upper = -lower

# plot_intensity_xy(100, 100, 1, "general bessel")

force_distance_plots("plane wave")
#force_distance_plots("gaussian")

force_map("general bessel")
#force_map("gaussian")
#force_map("plane wave")

#~~~testing code~~~
# x = np.linspace(-5e-5, 5e-5, 200)
# y = np.linspace(-5e-5, 5e-5, 200)
# z = np.linspace(-5e-5, 5e-5, 200)
# Ex = np.zeros(200, dtype=complex)
# Ex_prime = np.zeros(200, dtype=complex)
# Ez = np.zeros(200, dtype=complex)
# dum = 0
# for i in range(200):
#     x_prime, y_prime, z_prime = coord_transformation(
#         (x[i], 0, 0), (0, pi/2, 0), (0, 0, 0)
#     )
#     Ex_prime[i] = gaussian_E_x(x_prime, y_prime, z_prime, E0, wavelength, w0)
#     Ex[i] = gaussian_E_x(0, 0, x[i], E0, wavelength, w0)
# plt.plot(x, Ex)
# # plt.plot(x, np.imag(Ex))
# plt.show()

# x = np.linspace(-5e-5, 5e-5, 200)
# y = np.linspace(-5e-5, 5e-5, 200)
# Ex = np.zeros(200, dtype=complex)
# dExdx = np.zeros(200, dtype=complex)
# Ez = np.zeros(200, dtype=complex)
# dEzdx = np.zeros(200, dtype=complex)
# Ey = np.zeros(200, dtype=complex)
# dEydx = np.zeros(200, dtype=complex)
# for i in range(200):
#     Ex[i] = first_bessel_E_x(0, y[i], 0, E0, wavelength)
#     dExdx[i] = first_bessel_dE_xdx(0, y[i], 0, E0, wavelength)
#     Ey[i] = first_bessel_E_y(0, y[i], 0, E0, wavelength)
#     dEydx[i] = first_bessel_dE_ydx(0, y[i], 0, E0, wavelength)
#     Ez[i] = first_bessel_E_z(0, y[i], 0, E0, wavelength)
#     dEzdx[i] = first_bessel_dE_zdx(0, y[i], 0, E0, wavelength)
# plt.plot(x, np.real(np.conjugate(dEzdx) * Ez))
# # plt.plot(x, np.imag(dEzdx))
# plt.show()


# frames = 100
# X, Y, I, z = intensity_xy_animation(50, 50, frames, "gaussian")
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X, Y, I[0], cmap=cm.coolwarm, linewidth=0, antialiased=False)

# ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
# # plt.show()
# writer = animation.PillowWriter(fps=20)

# ani.save("gaussian_intensity_plot_w0=2wavl.gif", writer=writer)

# print(gaussian_dE_xdx(0,10,10,10,10,0.5))
