import matplotlib.pyplot as plt
import numpy as np

def generate_unit_sphere_positions(N, num_steps=1000):
    # returns N points on the unit sphere which are approximately evenly distributed.
    # num_steps can be changed to change the accuracy.

    pi = np.pi
    # generate random coords on the unit sphere
    spherical_coords = np.random.rand(N,3) * [0,pi, 2*pi] + [1,0,0]
    coords = np.transpose([np.sin(spherical_coords[:,1])*np.cos(spherical_coords[:,2]), np.sin(spherical_coords[:,1])*np.sin(spherical_coords[:,2]), np.cos(spherical_coords[:,1])] * spherical_coords[:,0])


    # simulate similar to charges repelling.
    C = 5e1
    dt = 1e-3
    N = len(coords)
    velocities = np.zeros((N,3))

    for _ in range(num_steps):
        for i in range(N):
            r = coords[i]-coords
            Fdt = C * r / (np.linalg.norm(r, axis=1)[:,None]**3 +1e-6) * dt
            Fdt[i] *= -1
            velocities -= Fdt
        coords += velocities * dt
        for i in range(N):
            coords[i] /= np.linalg.norm(coords[i])
    
    # option to plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(*np.transpose(coords), color="blue")
    # plt.show()

    return coords

generate_unit_sphere_positions(30)