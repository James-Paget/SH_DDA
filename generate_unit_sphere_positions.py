import matplotlib.pyplot as plt
import numpy as np
import itertools as it

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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*np.transpose(coords), color="blue")
    plt.title(f"{N} points")
    plt.show()

    return coords

# generate_unit_sphere_positions(40, num_steps=1000)

def print_particles(coords, radius=200e-9):
    for i, coord in enumerate(coords):
        coords_str = " ".join([str(x) for x in coord])
        print("\n".join([
            f"    part_{i+1}:",
            f"      material: FusedSilica",
            f"      shape: sphere",
            f"      args: {radius}",
            f"      coords: {coords_str}",
            f"      altcolour: True"
        ]))

def get_tetrahedron_points(radius=1e-6):
    def f(t,p):
        return np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)]) * radius

    coords = [[0.0,0.0,radius]]
    for i in range(3):
        coord = f(109.5/180*np.pi, i*2*np.pi/3)
        coords.append(coord)
    return coords

def get_cube_points(radius=1e-6):
    # 8 vertices
    return radius * np.array([ [-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1], [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1] ])

def get_octahedron_points(radius=1e-6):
    # 6 vertices
    return radius * np.array([ [0,0,1], [0,1,0], [1,0,0], [0,0,-1], [0,-1,0], [-1,0,0]])

def get_icosahedron_points(radius=1e-6):
    # 12 vertices
    phi = round((1 + np.sqrt(5))/2, 5)
    return radius * np.array([ [0,-1,-phi], [-1,-phi,0],  [-phi,0,-1], [0,-1,phi], [-1,phi,0],  [-phi,0,1], [0,1,-phi], [1,-phi,0],  [phi,0,-1], [0,1,phi], [1,phi,0],  [phi,0,1]])

# print_particles(get_tetrahedron_points(1e-6))
# print_particles(generate_unit_sphere_positions(12, 5000)*1e-6)
# print_particles(get_icosahedron_points())


def get_sheet_points(num_radius, separation, num_angular=None):
    # Creates a grid of points on the z=0 plane.
    # If num_angular=None, makes it a cubic grid
    # Else, num_angular is the points per radius

    coords = []
    if num_angular == None:
        # for a cubic grid
        values = separation * np.linspace(-num_radius, num_radius, 2*num_radius+1)
        for x in values:
            for y in values:
                if x**2 + y**2 <= (num_radius * separation)**2:
                    coords.append((x,y,0))

    else:
        # for a polar grid
        # !! could be awkward for connections
        ang_values = np.linspace(0, 2*np.pi, num_angular+1)[:-1] # exclude 2pi
        r_values = separation * np.linspace(1, num_radius, num_radius)
        print(ang_values)
        print(r_values)
        for r in r_values:
            for ang in ang_values:
                coords.append((r*np.cos(ang), r*np.sin(ang),0))

    return coords

print_particles(get_sheet_points(2, 0.7e-6), radius=100e-9)
# print_particles(get_sheet_points(6, 0.5e-6, 6), radius=100e-9) # Polar version



def min_dists():
    # plot how the min dist between any two particles decays as N increases.
    nums = np.arange(2,25,1)
    final = []
    for i in nums:
        ans = []
        for j in range(3):
            coords = generate_unit_sphere_positions(i)
            ans.append(np.min(np.array([np.linalg.norm(u - v) for u, v in it.combinations(coords, 2)])))
        final.append(np.average(np.array(ans)))

    plt.plot(np.log(nums), np.log(final))
    plt.xlabel("log(Points on sphere)")
    plt.ylabel("log(Min dist between points on a unit sphere)")
    plt.show()

# min_dists()


# CODE with fibonacci sunflower - works better and faster...
# https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

# from numpy import pi, cos, sin, arccos, arange
# import mpl_toolkits.mplot3d
# import matplotlib.pyplot as pp

# num_pts = 1000
# indices = arange(0, num_pts, dtype=float) + 0.5

# phi = arccos(1 - 2*indices/num_pts)
# theta = pi * (1 + 5**0.5) * indices

# x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi);

# pp.figure().add_subplot(111, projection='3d').scatter(x, y, z);
# pp.show()