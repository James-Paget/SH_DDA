import sys
import numpy as np

"""
write_* functions have no parameters set.
use_* functions have some paremeters set, making them quicker to create presets

"""

# each added preset needed to be added to BOTH 'get_preset_options' and 'generate_yaml'

def get_preset_options():
    # Return list of strings which name a preset.
    # Names corresponding to the same preset are put on the same line, although the list is 1D.
    return [
        "0", "TETRAHEDRON",
        "TETRAHEDRON_BESSEL",
        "TETRAHEDRON_ROTATED",
        "1", "ICOSAHEDRON",
        "2" , "LINE",
        "3", "NSPHERE",
        "4", "TORUS",
        "5", "CONNECTED_RING",
        "6", "UNCONNECTED_RING",
    ]

def generate_yaml(preset, filename="Preset"):

    # Reset YAML contents
    with open(f"{filename}.yml", "w") as _:
        pass

    # Match to a preset.
    match str(preset):
        case "0" | "TETRAHEDRON":
            use_default_options(filename, frames=50, show_output=True)
            use_laguerre3_beam(filename)
            use_tetrahedron(filename, tetrahedron_radius=1e-6, particle_radius=0.2e-6)

        case "TETRAHEDRON_BESSEL":
            use_default_options(filename, frames=50, show_output=True)
            use_bessel_beam(filename)
            use_tetrahedron(filename, tetrahedron_radius=1e-6, particle_radius=0.2e-6)

        case "TETRAHEDRON_ROTATED":
            use_default_options(filename, frames=50, show_output=True)
            use_laguerre3_beam(filename)
            use_tetrahedron(filename, tetrahedron_radius=1e-6, particle_radius=0.2e-6, rotation_axis=[1,0,0], rotation_theta=np.pi)

        case "1" | "ICOSAHEDRON":
            use_default_options(filename, frames=20, show_output=True)
            use_laguerre3_beam(filename)
            use_icosahedron(filename, icosahedron_radius=1e-6, particle_radius=0.2e-6)

        case "2" | "LINE":
            use_default_options(filename, frames=20, show_output=True)
            use_laguerre3_beam(filename)
            use_line(filename, num_particles=5, separation=0.5e-6, particle_radius=0.2e-6, rotation_axis=[1,0,0], rotation_theta=np.pi/2)

        case "3" | "NSPHERE":
            # Approximately distributes N points over a sphere and connects them.
            use_default_options(filename, frames=10, show_output=True)
            use_laguerre3_beam(filename)
            use_NSphere(filename, num_particles=40, sphere_radius=2e-6, particle_radius=0.15e-6, connection_mode="num", connection_args=5, rotation_axis=[0,0,1], rotation_theta=0)

        case "4" | "TORUS":
            use_default_options(filename, frames=1, show_output=True)
            use_laguerre3_beam(filename)
            use_torus(filename, num_particles=6, inner_radius=1.15e-6, tube_radius=0.2e-6, separation=0.5e-7)

        case "5" | "CONNECTED_RING":
            use_default_options(filename, frames=50, show_output=True)
            use_laguerre3_beam(filename)
            use_connected_ring(filename, num_particles=6, ring_radius=1e-6, particle_radius=0.2e-6, rotation_axis=[0,0,1], rotation_theta=0)

        case "6" | "UNCONNECTED_RING":
            use_default_options(filename, frames=50, show_output=True)
            use_laguerre3_beam(filename)
            use_unconnected_ring(filename, num_particles=6, ring_radius=1e-6, particle_radius=0.2e-6, rotation_axis=[0,0,1], rotation_theta=0)


        case _:
            sys.exit(f"Generate_yaml error: preset '{preset}' not found")


#=======================================================================
# Particle configurations
#=======================================================================

def use_tetrahedron(filename, tetrahedron_radius, particle_radius, rotation_axis=[0,0,1], rotation_theta=0):
    args_list = [[particle_radius]] * 4
    coords_list = get_tetrahedron_points(tetrahedron_radius)
    if rotation_theta != 0:
        coords_list = rotate_coords_list(coords_list, rotation_axis, rotation_theta)
    use_default_particles(filename, "sphere", args_list, coords_list, "num", 3)

def use_icosahedron(filename, icosahedron_radius, particle_radius, rotation_axis=[0,0,1], rotation_theta=0):
    args_list = [[particle_radius]] * 12
    coords_list = get_icosahedron_points(icosahedron_radius)
    if rotation_theta != 0:
        coords_list = rotate_coords_list(coords_list, rotation_axis, rotation_theta)
    use_default_particles(filename, "sphere", args_list, coords_list, "num", 5)

def use_line(filename, num_particles, separation, particle_radius, rotation_axis=[0,0,1], rotation_theta=0):
    args_list = [[particle_radius]] * num_particles
    coords_list = [[0,0,separation*(zi - (num_particles-1)/2)] for zi in range(num_particles)]
    if rotation_theta != 0:
        coords_list = rotate_coords_list(coords_list, rotation_axis, rotation_theta)
    use_default_particles(filename, "sphere", args_list, coords_list, "dist", separation*1.5)

def use_NSphere(filename, num_particles, sphere_radius, particle_radius, connection_mode, connection_args, rotation_axis=[0,0,1], rotation_theta=0):
    args_list = [[particle_radius]] * num_particles
    coords_list = get_sunflower_points(num_particles, sphere_radius)
    if rotation_theta != 0:
        coords_list = rotate_coords_list(coords_list, rotation_axis, rotation_theta)
    use_default_particles(filename, "sphere", args_list, coords_list, connection_mode, connection_args)

def use_torus(filename, num_particles, inner_radius, tube_radius, separation):
    coords_list = []
    args_list = []

    # Writing specific parameters for particle formation
    torus_gap_theta    = separation/inner_radius    # Full angle occupied by gap between torus sectors
    torus_sector_theta = (2.0*np.pi -num_particles*torus_gap_theta) / (num_particles) #Full angle occupied by torus sector
    for particle_index in range(num_particles):
        lower_phi = ( particle_index*(torus_sector_theta +torus_gap_theta) -torus_sector_theta/2.0 )
        upper_phi = ( particle_index*(torus_sector_theta +torus_gap_theta) +torus_sector_theta/2.0 )
        particle_position = [
            inner_radius*np.cos( particle_index*(torus_sector_theta +torus_gap_theta) ), 
            inner_radius*np.sin( particle_index*(torus_sector_theta +torus_gap_theta) ), 
            1.0e-6
        ]
        coords_list.append(particle_position)
        args_list.append([inner_radius, tube_radius, lower_phi, upper_phi])

    use_default_particles(filename, "torus", args_list, coords_list, connection_mode="num", connection_args=0)

def use_connected_ring(filename, num_particles, ring_radius, particle_radius, rotation_axis=[0,0,1], rotation_theta=0):
    args_list = [[particle_radius]] * num_particles
    coords_list = [[ring_radius*np.cos(t*2*np.pi/num_particles), ring_radius*np.sin(t*2*np.pi/num_particles), 0] for t in range(0, num_particles)]
    if rotation_theta != 0:
        coords_list = rotate_coords_list(coords_list, rotation_axis, rotation_theta)
    use_default_particles(filename, "sphere", args_list, coords_list, "num", 2)

def use_unconnected_ring(filename, num_particles, ring_radius, particle_radius, rotation_axis=[0,0,1], rotation_theta=0):
    args_list = [[particle_radius]] * num_particles
    coords_list = [[ring_radius*np.cos(t*2*np.pi/num_particles), ring_radius*np.sin(t*2*np.pi/num_particles), 0] for t in range(0, num_particles)]
    if rotation_theta != 0:
        coords_list = rotate_coords_list(coords_list, rotation_axis, rotation_theta)
    use_default_particles(filename, "sphere", args_list, coords_list, "num", 0)


def use_default_particles(filename, shape, args_list, coords_list, connection_mode, connection_args):
    """
    Fills in typical particle parameters e.g. material, but leaves particles general.
    """
    default_radius = 1e-07
    default_material = "FusedSilica"
    particle_list = [{"material":"FusedSilica", "shape":shape, "args":args_list[i], "coords":coords_list[i], "altcolour":True} for i in range(len(coords_list))]
    write_particles(filename, particle_list, default_radius, default_material, connection_mode, connection_args )


#=======================================================================
# Beam configurations
#=======================================================================

def use_laguerre3_beam(filename):
    """
    Makes a Laguerre-Gaussian beam.
    """
    beam = {"beamtype":"BEAMTYPE_LAGUERRE_GAUSSIAN", "E0":300, "order":3, "w0":0.6, "jones":"POLARISATION_LCP", "translation":None, "rotation":None}
    write_beams(filename, [beam])

def use_bessel_beam(filename):
    """
    Makes a Laguerre-Gaussian beam.
    """
    beam = {"beamtype":"BEAMTYPE_BESSEL", "E0":1.5e7, "order":1, "jones":"POLARISATION_LCP", "translation":None, "rotation":None}
    write_beams(filename, [beam])

#=======================================================================
# Option configurations
#=======================================================================

def use_default_options(filename, frames, show_output, wavelength=1e-6, dipole_radius=4e-8, time_step=0.0001, vmd_output=True, excel_output=True, include_force=True, include_couple=True, frame_interval=2, max_size=2e-6, resolution=201, frame_min=0, frame_max=None, z_offset=0.0):
    """
    Make the default options, requiring just filename, frames, show_output
    """
    if frame_max == None:
        frame_max = frames
    write_options(filename, frames, wavelength, dipole_radius, time_step, vmd_output, excel_output, include_force, include_couple, show_output, frame_interval, max_size, resolution, frame_min, frame_max, z_offset)

#=======================================================================
# Core functions which write options, beams and particles
#=======================================================================

def write_options(filename, frames, wavelength, dipole_radius, time_step, vmd_output, excel_output, include_force, include_couple, show_output, frame_interval, max_size, resolution, frame_min, frame_max, z_offset):
    """
    Base function to write the passed options to filename.
    """
    with open(f"{filename}.yml", "a") as file:
        file.write(f"options:\n")
        file.write(f"  frames: {frames}\n")
        file.write(f"parameters:\n")
        file.write(f"  wavelength: {wavelength}\n")
        file.write(f"  dipole_radius: {dipole_radius}\n")
        file.write(f"  time_step: {time_step}\n")
        file.write(f"output:\n")
        file.write(f"  vmd_output: {vmd_output}\n")
        file.write(f"  excel_output: {excel_output}\n")
        file.write(f"  include_force: {include_force}\n")
        file.write(f"  include_couple: {include_couple}\n")
        file.write(f"display:\n")
        file.write(f"  show_output: {show_output}\n")
        file.write(f"  frame_interval: {frame_interval}\n")
        file.write(f"  max_size: {max_size}\n")
        file.write(f"  resolution: {resolution}\n")
        file.write(f"  frame_min: {frame_min}\n")
        file.write(f"  frame_max: {frame_max}\n")
        file.write(f"  z_offset: {z_offset}\n")

def write_beams(filename, beams_list):
    """
    Base function to write the passed beams to filename.

    Each beam is a dictionary with keys.
    Not all keys must be given values, but some options are: "beamtype", "E0", "order", "w0", "jones", "translation", "rotation"
    """
    with open(f"{filename}.yml", "a") as file:
        file.write(f"beams:\n")
        for i, beam in enumerate(beams_list):
            file.write(f"  beam_{i}:\n")
            for arg in beam.keys(): # typically ["beamtype", "E0", "order", "w0", "jones", "translation", "rotation"]
                file.write(f"    {arg}: {beam[arg]}\n")

def write_particles(filename, particle_list, default_radius, default_material, connection_mode, connection_args):
    """
    Base function to write the passed particles and their parameters to filename.

    Each particle in particle_list is a dictionary with keys:  "material", "shape", "args", "coords", "altcolour"
    """
    with open(f"{filename}.yml", "a") as file:
        file.write(f"particles:\n")
        file.write(f"  default_radius: {default_radius}\n")
        file.write(f"  default_material: {default_material}\n")
        file.write(f"  connection_mode: {connection_mode}\n")
        file.write(f"  connection_args: {connection_args}\n")
        file.write(f"  particle_list:\n")

        for i, particle in enumerate(particle_list):
            # Test correct number of args
            if not (
                (particle["shape"] == "sphere" and len(particle["args"]) == 1) or
                (particle["shape"] == "torus" and len(particle["args"]) == 4)
            ):
                print(f"Particle {i} has invalid args: {particle}")
                break

            file.write(f"    part_{i}:\n")
            file.write(f"      material: {particle['material']}\n")
            file.write(f"      shape: {particle['shape']}\n")
            file.write(f"      args: {' '.join([str(elem) for elem in particle['args']])}\n")
            file.write(f"      coords: {' '.join([str(elem) for elem in particle['coords']])}\n")
            file.write(f"      altcolour: {particle['altcolour']}\n")


#=======================================================================
# Coords helper functions
#=======================================================================

def rotate_coords_list(coords_list, axis, theta):
    """
    Rotates every coordinate in coords_list by theta about axis.
    Theta in radians.
    """
    # Using the Rodrigues' rotation formula
    coords_list = np.array(coords_list)
    axis = np.array(axis)
    for i in range(len(coords_list)):
        r = coords_list[i]
        comp_a = r*np.cos(theta)
        comp_b = np.cross(axis, r)*np.sin(theta)
        comp_c = axis*np.dot(axis, r)*(1-np.cos(theta))
        coords_list[i] = comp_a +comp_b + comp_c
    return coords_list

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

def get_dodecahedron_points(radius=1.4e-6):
    # 20 vertices
    phi = round((1 + np.sqrt(5))/2, 5)
    return radius * np.array([ [-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1], [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1], [0,-1/phi,-phi], [-1/phi,-phi,0],  [-phi,0,-1/phi], [0,-1/phi,phi], [-1/phi,phi,0],  [-phi,0,1/phi], [0,1/phi,-phi], [1/phi,-phi,0],  [phi,0,-1/phi], [0,1/phi,phi], [1/phi,phi,0],  [phi,0,1/phi]])

def get_sunflower_points(N, radius):
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    indices = np.arange(0, N, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/N)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return radius * np.array([(x[i], y[i], z[i]) for i in range(N)])