import sys
import numpy as np

"""
write_* functions have no parameters set.
use_* functions have some paremeters set, making them quicker to create presets
make_yaml_* makes a full yaml using the input parameters, with default values which can be used.
"""


def generate_yaml(preset, filename="Preset"):
    # Return True if preset used, else False

    # Reset YAML contents
    with open(f"{filename}.yml", "w") as _:
        pass

    # Match to a preset.
    match str(preset):
        case "0" | "TETRAHEDRON":
            make_yaml_tetrahedron(filename)

        case "TRANSLATE_BEAM":
            use_default_options(filename, frames=30, show_output=True)
            use_laguerre3_beam(filename, translation=None, translationargs="1.5e-6 0 0", translationtype="linear") # can be linear or a circle
            use_tetrahedron(filename, 1e-6, 0.2e-6, [0,0,1], 0)

        case "TETRAHEDRON_BESSEL":
            make_yaml_tetrahedron(filename, beam="BESSEL")

        case "TETRAHEDRON_ROTATED":
            make_yaml_tetrahedron(filename, rotation_axis=[1,0,0], rotation_theta=np.pi)
            
        case "1" | "ICOSAHEDRON":
            make_yaml_icosahedron(filename)

        case "2" | "LINE":
            make_yaml_line(filename)

        case "3" | "NSPHERE":
            # Approximately distributes N points over a sphere and connects them.
            make_yaml_Nsphere(filename)

        case "4" | "TORUS":
            make_yaml_torus(filename)

        case "5" | "CONNECTED_RING":
            make_yaml_connected_ring(filename)

        case "6" | "UNCONNECTED_RING":
            make_yaml_unconnected_ring(filename)
            
        case "7" | "SHEET_TRIANGLE":
            make_yaml_sheet_triangle(filename)
        
        case "8" | "SHEET_SQUARE":
            make_yaml_sheet_square(filename)

        case "9" | "SHEET_HEXAGON":
            make_yaml_sheet_hexagon(filename)

        case "10" | "FILAMENT":
            make_yaml_filament(filename)

        case "11" | "FIBRE_1D_SPHERE":
            make_yaml_fibre_1d_sphere(filename)

        case "12" | "FIBRE_1D_CYLINDER":
            make_yaml_fibre_1d_cylinder(filename)

        case "13" | "FIBRE_2D_SPHERE_HOLLOWSHELL":
            make_yaml_fibre_2d_sphere_hollowshell(filename)

        case "14" | "FIBRE_2D_CYLINDER_HOLLOWSHELL":
            make_yaml_fibre_2d_cylinder_hollowshell(filename)

        case "15" | "FIBRE_2D_SPHERE_THICK_UNI":
            make_yaml_fibre_2d_sphere_thick_uni(filename)

        case "16" | "FIBRE_2D_CYLINDER_THICK_UNI":
            make_yaml_fibre_2d_cylinder_thick_uni(filename)
        
        case "17" | "FIBRE_2D_SPHERE_SHELLLAYERS":
            make_yaml_fibre_2d_sphere_shelllayers(filename)
        
        case "18" | "FIBRE_2D_CYLINDER_SHELLLAYERS":
            make_yaml_fibre_2d_cylinder_shelllayers(filename)

        case "SPHERICAL_SHELLLAYERS":
            use_default_options(filename, frames=1, show_output=True, time_step=0.0001)
            use_beam(filename, beam="LAGUERRE")
            particle_radius=0.15e-6
            shell_radii = [0.3e-6, 2e-6]
            numbers_per_shell = [4,10]
            connection_dists = [0.5e-6, 3e-6]
            coords_list = get_NsphereShell_points(shell_radii, numbers_per_shell)
            args_list = [[particle_radius]] * len(coords_list)
            connection_args = "False 0.0 " + " ".join(["sphere " + str(shell_radii[i]) + " " + str(connection_dists[i]) for i in range(len(shell_radii))])
            # connection_args = "True 1.0 " + " ".join(["sphere " + str(shell_radii[i]) + " " + str(connection_dists[i]) for i in range(len(shell_radii))])
            use_default_particles(filename, "sphere", args_list, coords_list, connection_mode="dist_shells", connection_args=connection_args)

        case "CUBE":
            use_default_options(filename, frames=10, show_output=True, time_step=0.0001)
            use_beam(filename, "LAGUERRE")
            args_list = [[0.20e-6]] * 4
            coords_list = get_tetrahedron_points(1e-6)
            use_default_particles(filename, "cube", args_list, coords_list, "num", 3)
        
        case "ONE_PARTICLE":
            use_default_options(filename, frames=10, show_output=True, time_step=0.0001)
            use_beam(filename, "LAGUERRE")
            args_list = [[0.20e-6]]
            coords_list = [[0.0, 0.0, 0.0]]
            use_default_particles(filename, "sphere", args_list, coords_list, "num", 0)

        case _:
            return False
    
    return True

#=======================================================================
# Make yamls
#=======================================================================

def make_yaml_tetrahedron(filename, frames=50, show_output=True, tetrahedron_radius=1e-6, particle_radius=0.2e-6, beam="LAGUERRE", rotation_axis=[0,0,1], rotation_theta=0):
    use_default_options(filename, frames, show_output)
    use_beam(filename, beam)
    use_tetrahedron(filename, tetrahedron_radius, particle_radius, rotation_axis, rotation_theta)

def make_yaml_icosahedron(filename, frames=20, show_output=True, icosahedron_radius=1e-6, particle_radius=0.2e-6, beam="LAGUERRE", rotation_axis=[0,0,1], rotation_theta=0):
    use_default_options(filename, frames, show_output)
    use_beam(filename, beam)
    use_icosahedron(filename, icosahedron_radius, particle_radius, rotation_axis, rotation_theta)

def make_yaml_line(filename, frames=20, show_output=True, num_particles=5, separation=0.5e-6, particle_radius=0.2e-6, beam="LAGUERRE", rotation_axis=[1,0,0], rotation_theta=np.pi/2):
    use_default_options(filename, frames, show_output)
    use_beam(filename, beam)
    use_line(filename, num_particles, separation, particle_radius, rotation_axis, rotation_theta)

def make_yaml_Nsphere(filename, frames=10, show_output=True, num_particles=40, sphere_radius=2e-6, particle_radius=0.1e-6, connection_mode="num", connection_args=5, beam="LAGUERRE", rotation_axis=[0,0,1], rotation_theta=0):
    use_default_options(filename, frames, show_output)
    use_beam(filename, beam)
    use_NSphere(filename, num_particles, sphere_radius, particle_radius, connection_mode, connection_args, rotation_axis, rotation_theta)

def make_yaml_torus(filename, frames=1, show_output=True, num_particles=6, inner_radius=1.15e-6, tube_radius=0.2e-6, separation=0.5e-7, beam="LAGUERRE"):
    use_default_options(filename, frames, show_output)
    use_beam(filename, beam)
    use_torus(filename, num_particles, inner_radius, tube_radius, separation)

def make_yaml_connected_ring(filename, frames=50, show_output=True, num_particles=6, ring_radius=1e-6, particle_radius=0.2e-6, beam="LAGUERRE", rotation_axis=[0,0,1], rotation_theta=0):
    use_default_options(filename, frames, show_output)
    use_beam(filename, beam)
    use_connected_ring(filename, num_particles, ring_radius, particle_radius, rotation_axis, rotation_theta)

def make_yaml_unconnected_ring(filename, frames=50, show_output=True, num_particles=6, ring_radius=1e-6, particle_radius=0.2e-6, beam="LAGUERRE", rotation_axis=[0,0,1], rotation_theta=0):
    use_default_options(filename, frames, show_output)
    use_beam(filename, beam)
    use_unconnected_ring(filename, num_particles, ring_radius, particle_radius, rotation_axis, rotation_theta)

def make_yaml_sheet_triangle(filename, frames=25, show_output=True, num_length=4, num_width=4, separation=0.9e-6, particle_radius=0.15e-6, beam="LAGUERRE", rotation_axis=[0,0,1], rotation_theta=0):
    use_default_options(filename, frames, show_output)
    use_beam(filename, beam)
    use_sheet_triangle(filename, num_length, num_width, separation, particle_radius, rotation_axis, rotation_theta)

def make_yaml_sheet_square(filename, frames=25, show_output=True, num_length=4, num_width=4, separation=0.9e-6, particle_radius=0.15e-6, beam="LAGUERRE", rotation_axis=[0,0,1], rotation_theta=0):
    use_default_options(filename, frames, show_output)
    use_beam(filename, beam)
    use_sheet_square(filename, num_length, num_width, separation, particle_radius, rotation_axis, rotation_theta)

def make_yaml_sheet_hexagon(filename, frames=25, show_output=True, num_length=3, num_width=3, separation=0.7e-6, particle_radius=0.12e-6, beam="LAGUERRE", rotation_axis=[0,0,1], rotation_theta=0):
    use_default_options(filename, frames, show_output)
    use_beam(filename, beam)
    use_sheet_hexagon(filename, num_length, num_width, separation, particle_radius, rotation_axis, rotation_theta)

def make_yaml_filament(filename, frames=50, show_output=True, length=4e-6, radius=0.8e-6, separation=0.7e-6, particle_radius=0.1e-6, beam="LAGUERRE", rotation_axis=[0,0,1], rotation_theta=0):
    use_default_options(filename, frames, show_output)
    use_beam(filename, beam)
    use_filament(filename, length, radius, separation, particle_radius, rotation_axis, rotation_theta)

def make_yaml_fibre_1d_sphere(filename, option_parameters, length=2e-6, particle_radius=0.2e-6, particle_number=10, connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_parameter_options(filename, option_parameters)
    use_beam(filename, beam)
    use_fibre_1d_sphere(filename, length, particle_radius, particle_number, connection_mode, connection_args)

def make_yaml_fibre_1d_cylinder(filename, option_parameters, length=3e-6, particle_length=0.4e-6, particle_radius=0.1e-6, particle_number=5, connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_parameter_options(filename, option_parameters)
    use_beam(filename, beam)
    use_fibre_1d_cylinder(filename, length, particle_length, particle_radius, particle_number, connection_mode, connection_args)

def make_yaml_fibre_2d_sphere_hollowshell(filename, E0, option_parameters, length=3e-6, shell_radius=0.3e-6, particle_radius=0.1e-6, particle_number_radial=6, particle_number_angular=4, connection_mode="dist", connection_args=0.0, beam="LAGUERRE", include_beads=False):
    use_parameter_options(filename, option_parameters)
    #use_beam(filename, beam, translation="2.5e-6 0.0 0.0", translationargs="1.0e-6 0.0 0.0")  ### DOES NOT ALLOW PARAMETER VARIATION EASILY ###
    # use_gaussCSP_beam(filename, E0=2.5e7, w0=0.4, translation="2.5e-6 0.0 0.0", translationargs="2.5e-6 1.5e-6 0.0")

    #2.5e-6 1.5e-6 0.0
    beam_1 = {"beamtype":"BEAMTYPE_GAUSS_CSP", "E0":E0, "order":3, "w0":0.4, "jones":"POLARISATION_LCP", "translation": "2.6e-6 0.0 0.0", "translationargs": "-0.5 0.0 0.0 1.0 -2.2e-6 0.0 0.0", "translationtype":"circle", "rotation":None}
    beam_2 = {"beamtype":"BEAMTYPE_GAUSS_CSP", "E0":E0, "order":3, "w0":0.4, "jones":"POLARISATION_LCP", "translation":"-2.6e-6 0.0 0.0", "translationargs": "0.5 0.0 0.0 1.0 2.2e-6 0.0 0.0", "translationtype":"circle", "rotation":None}
    write_beams(filename, [beam_1, beam_2])

    # Varies depending on if beads are included within this function
    use_fibre_2d_sphere_hollowshell(filename, length, shell_radius, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, include_beads=include_beads)

def make_yaml_fibre_2d_cylinder_hollowshell(filename, option_parameters, length=2e-6, shell_radius=1e-6, particle_length=0.5e-6, particle_radius=0.2e-6, particle_number_radial=3, particle_number_angular=8, connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_parameter_options(filename, option_parameters)
    use_beam(filename, beam)
    use_fibre_2d_cylinder_hollowshell(filename, length, shell_radius, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args)

def make_yaml_fibre_2d_sphere_thick_uni(filename, option_parameters, length=3e-6, shell_radius=1e-6, shell_number=1, particle_radius=0.2e-6, particle_number_radial=4, particle_number_angular=6, connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_parameter_options(filename, option_parameters)
    use_beam(filename, beam)
    use_fibre_2d_sphere_thick_uni(filename, length, shell_radius, shell_number, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args)

def make_yaml_fibre_2d_cylinder_thick_uni(filename, option_parameters, length=3e-6, shell_radius=1e-6, shell_number=1, particle_length=0.5e-6, particle_radius=0.2e-6, particle_number_radial=3, particle_number_angular=6, connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_parameter_options(filename, option_parameters)
    use_beam(filename, beam)
    use_fibre_2d_cylinder_thick_uni(filename, length, shell_radius, shell_number, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args)

def make_yaml_fibre_2d_sphere_shelllayers(filename, option_parameters, length=1.5e-6, shell_radius_max=1.5e-6, shell_number=2, particle_radius=0.15e-6, particle_separation=(np.pi*2.0*1.0e-6)/(10.0), connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_parameter_options(filename, option_parameters)
    use_beam(filename, beam)
    use_fibre_2d_sphere_shelllayers(filename, length, shell_radius_max, shell_number, particle_radius, particle_separation, connection_mode, connection_args)

def make_yaml_fibre_2d_cylinder_shelllayers(filename, option_parameters,  length=1.0e-6, shell_radius_max=1.5e-6, shell_number=2, particle_length=0.4e-6, particle_radius=0.15e-6, particle_separation=(np.pi*2.0*1.0e-6)/(10.0), connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_parameter_options(filename, option_parameters)
    use_beam(filename, beam)
    use_fibre_2d_cylinder_shelllayers(filename, length, shell_radius_max, shell_number, particle_length, particle_radius, particle_separation, connection_mode, connection_args)

def make_yaml_refine_cuboid(filename, dimensions, separations, object_offset, particle_size, particle_shape, option_parameters, beam="LAGUERRE"):
    use_parameter_options(filename, option_parameters)
    use_beam(filename, beam)
    num_particles = use_refine_cuboid(filename, dimensions, separations, object_offset, particle_size, particle_shape)
    return num_particles

def make_yaml_refine_arch_prism(filename, dimensions, separations, particle_size, deflection, object_offset, particle_shape, place_regime, prism_type, prism_args, option_parameters, beam="LAGUERRE"):
    use_parameter_options(filename, option_parameters)
    use_beam(filename, beam)
    num_particles = use_refine_arch_prism(filename, dimensions, separations, deflection, object_offset, particle_size, particle_shape, place_regime, prism_type, prism_args)
    return num_particles

def make_yaml_refine_sphere(filename, dimension, separations, particle_size, object_offset, particle_shape, place_regime, option_parameters, dipole_size=None, beam="LAGUERRE", makeCube=False, material="FusedSilica"):
    if dipole_size != None: option_parameters["dipole_radius"] = dipole_size # keep the passed in dipoles size so that it can be varied despite partial functions
    use_parameter_options(filename, option_parameters)
    use_beam(filename, beam)
    num_particles = use_refine_sphere(filename, dimension, separations, object_offset, particle_size, particle_shape, place_regime, makeCube=makeCube, material=material)
    return num_particles

def make_yaml_refine_cube_showcase(filename, dimension, object_offset, particle_shape, option_parameters, dipole_size=None, beam="LAGUERRE", material="FusedSilica"):
    #
    # Generates multiple cubes, refined by base amounts, that demonstrates the refinement process
    #
    if dipole_size != None: option_parameters["dipole_radius"] = dipole_size # keep the passed in dipoles size so that it can be varied despite partial functions
    use_parameter_options(filename, option_parameters)
    use_beam(filename, beam)
    num_particles = use_refine_cube_showcase(filename, dimension, object_offset, particle_shape, material=material)
    return num_particles

def make_yaml_single_dipole_exp(filename, test_type, test_args, object_offset, option_parameters, rotation=None, beam="LAGUERRE", extra_args=[]):
    use_parameter_options(filename, option_parameters)
    use_beam(filename, beam, rotation=rotation)
    num_particles = use_single_dipole_exp(filename, test_type, test_args, option_parameters["dipole_radius"], object_offset=object_offset, extra_args=extra_args)
    return num_particles

def make_yaml_spheredisc_model(filename, dimension, separations, particle_size, object_offset, particle_shape, option_parameters, dipole_size=None, mode="disc", beam="LAGUERRE", material="FusedSilica", fix_to_ring=True):
    if dipole_size != None: option_parameters["dipole_radius"] = dipole_size # keep the passed in dipoles size so that it can be varied despite partial functions
    use_parameter_options(filename, option_parameters)
    use_beam(filename, beam)
    num_particles = use_fill_spheredisc(filename, dimension, separations, particle_size, object_offset, particle_shape, mode=mode, material=material, fix_to_ring=fix_to_ring)
    return num_particles

def make_yaml_stretcher_springs(filename, option_parameters, num_particles, sphere_radius, particle_radius, connection_mode, connection_args, E0, w0, translation):
    use_parameter_options(filename, option_parameters)
    use_beam(filename, "STRETCHER", E0=E0, w0=w0, translation=translation)
    use_NSphere(filename, num_particles, sphere_radius, particle_radius, connection_mode, connection_args)

def make_yaml_stretch_sphere(filename, option_parameters, particle_shape, E0, w0, dimension, particle_size, transform_factor, critical_transform_factor, func_transform, object_offset, translation=None, connection_mode="dist", connection_args=0.0, material="FusedSilica"):
    use_parameter_options(filename, option_parameters)
    use_beam(filename, "STRETCHER", E0=E0, w0=w0, translation=translation)
    num_particles = use_stretch_sphere(filename, dimension, particle_size, transform_factor, critical_transform_factor, func_transform, object_offset, particle_shape=particle_shape, connection_mode=connection_mode, connection_args=connection_args, material=material)
    return num_particles

def make_yaml_stretcher_dipole_shape(filename, coords_list, dipole_size, connection_mode, connection_args, E0, w0, show_output, time_step=1e-4):
    use_default_options(filename, frames=1, show_output=show_output, time_step=time_step, dipole_radius=dipole_size)
    use_beam(filename, "STRETCHER", E0=E0, w0=w0)
    args_list = [[dipole_size]] * len(coords_list)
    use_default_particles(filename, "cube", args_list, coords_list, connection_mode=connection_mode, connection_args=connection_args, material="FusedSilica")


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
    coords_list, args_list = get_torus_points_args(num_particles, separation, inner_radius, tube_radius)
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

def use_sheet_triangle(filename, num_length, num_width, separation, particle_radius, rotation_axis=[0,0,1], rotation_theta=0, formation="square", bounds=[2e-6], connection_factor=1.001):
    args_list = [[particle_radius]] * num_length * num_width
    coords_list = get_sheet_points(num_length, num_width, separation, mode="triangle", formation=formation, bounds=bounds)
    if rotation_theta != 0:
        coords_list = rotate_coords_list(coords_list, rotation_axis, rotation_theta)
    use_default_particles(filename, "sphere", args_list, coords_list, "dist", connection_factor*separation)

def use_sheet_square(filename, num_length, num_width, separation, particle_radius, rotation_axis=[0,0,1], rotation_theta=0):
    args_list = [[particle_radius]] * num_length * num_width
    coords_list = get_sheet_points(num_length, num_width, separation, mode="square")
    if rotation_theta != 0:
        coords_list = rotate_coords_list(coords_list, rotation_axis, rotation_theta)
    use_default_particles(filename, "sphere", args_list, coords_list, "dist", 1.001*separation)

def use_sheet_hexagon(filename, num_length, num_width, separation, particle_radius, rotation_axis=[0,0,1], rotation_theta=0):
    coords_list = get_sheet_points(num_length, num_width, separation, mode="hexagon")
    args_list = [[particle_radius]] * len(coords_list)
    if rotation_theta != 0:
        coords_list = rotate_coords_list(coords_list, rotation_axis, rotation_theta)
    use_default_particles(filename, "sphere", args_list, coords_list, "dist", 1.001*separation)

def use_filament(filename, length, radius, separation, particle_radius, rotation_axis=[0,0,1], rotation_theta=0):
    coords_list = get_filament_points(length, radius, separation)
    args_list = [[particle_radius]] * len(coords_list)
    if rotation_theta != 0:
        coords_list = rotate_coords_list(coords_list, rotation_axis, rotation_theta)
    use_default_particles(filename, "sphere", args_list, coords_list, "dist", 1.001*separation)

def use_fibre_1d_sphere(filename, length, particle_radius, particle_number, connection_mode, connection_args):
    separation = length/particle_number
    args_list = [[particle_radius]] * particle_number
    coords_list = [[separation*(zi - (particle_number-1)/2), 0.0, 0.0] for zi in range(particle_number)]
    use_default_particles(filename, "sphere", args_list, coords_list, connection_mode, connection_args)

def use_fibre_1d_cylinder(filename, length, particle_length, particle_radius, particle_number, connection_mode, connection_args):
    separation = length/particle_number
    coords_list = [[separation*(zi - (particle_number-1)/2), 0.0, 0.0] for zi in range(particle_number)]
    args_list = [[particle_radius, particle_length, 0.0, 0.0]] * particle_number
    use_default_particles(filename, "cylinder", args_list, coords_list, connection_mode, connection_args)

def use_fibre_2d_sphere_hollowshell(filename, length, shell_radius, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, include_beads=False):
    args_list = [[particle_radius]] * (particle_number_radial*particle_number_angular)
    coords_list = coords_list = get_fibre_2d_hollowshell_points(length, shell_radius, particle_number_radial, particle_number_angular)
    
    bead_positions = [ [-1.5*length/2.0, 0.0, 0.0], [1.5*length/2.0, 0.0, 0.0] ]

    connection_args[0] = 1.01*max( shell_radius*(2.0*np.pi/particle_number_angular), (length/(particle_number_radial-1)) )   # NOTE; with this approach to separation, you want the two separations to be similar (your angular and radial) to avoid excess connections
    if(include_beads):
        connection_args[1] = 1.1*(abs(bead_positions[0][0]) - length/2.0)  # How far is each bead from the particles at the very edge +some tolerance 

    # Shell material
    default_radius = 1e-07
    default_material = "RBC"
    particle_list = [{"material":"FusedSilica", "shape":"sphere", "args":args_list[i], "coords":coords_list[i], "altcolour":True} for i in range(len(coords_list))]
    if(include_beads):
        # Bead material
        connection_args[2] = 2
        bead_radius = default_radius    # NOTE; The dynamics will not be valid if the radius is different (with the current implementation of the dynamics Jan.2025)
        bead_material = "FusedSilica"
        particle_list.append( {"material":bead_material, "shape":"sphere", "args":[bead_radius], "coords":bead_positions[0], "altcolour":True} )
        particle_list.append( {"material":bead_material, "shape":"sphere", "args":[bead_radius], "coords":bead_positions[1], "altcolour":True} )

    # Manually converting connection args back to nicer readable state
    ####
    ## THIS SHOULD BE CHECK FOR BEFORE WRITING INSIDE write_particles()
    ####
    connection_args_str = str(connection_args[0])
    if(include_beads):
        connection_args_str = str(connection_args[0])+" "+str(connection_args[1])+" "+str(connection_args[2])

    write_particles(filename, particle_list, default_radius, default_material, connection_mode, connection_args_str)

    #use_default_particles(filename, "sphere", args_list, coords_list, connection_mode, connection_args)

def use_fibre_2d_cylinder_hollowshell(filename, length, shell_radius, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args):
    args_list = [[particle_radius, particle_length, 0.0, 0.0]] * (particle_number_radial*particle_number_angular)
    coords_list = coords_list = get_fibre_2d_hollowshell_points(length, shell_radius, particle_number_radial, particle_number_angular)
    connection_args = 1.01*max( shell_radius*(2.0*np.pi/particle_number_angular), (length/(particle_number_radial-1)) )   # NOTE; with this approach to separation, you want the two separations to be similar (your angular and radial) to avoid excess connections
    use_default_particles(filename, "cylinder", args_list, coords_list, connection_mode, connection_args)

def use_fibre_2d_sphere_thick_uni(filename, length, shell_radius, shell_number, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args):
    args_list = [[particle_radius]] * (particle_number_radial*particle_number_angular*(shell_number) +particle_number_radial)   # N shells + 1 line
    coords_list = coords_list = get_fibre_2d_thick_points(length, shell_radius, shell_number, particle_number_radial, particle_number_angular, include_center_line=True)
    
    connection_args = max( shell_radius*(2.0*np.pi/particle_number_angular), (length/(particle_number_radial-1)) )   # NOTE; with this approach to separation, you want the two separations to be similar (your angular and radial) to avoid excess connections
    connection_args = 1.1*max( connection_args, shell_radius/shell_number )

    use_default_particles(filename, "sphere", args_list, coords_list, connection_mode, connection_args)

def use_fibre_2d_cylinder_thick_uni(filename, length, shell_radius, shell_number, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args):
    args_list = [[particle_radius, particle_length, 0.0, 0.0]] * (particle_number_radial*particle_number_angular*(shell_number) +particle_number_radial)   # N shells + 1 line
    coords_list = coords_list = get_fibre_2d_thick_points(length, shell_radius, shell_number, particle_number_radial, particle_number_angular, include_center_line=True)

    connection_args = max( shell_radius*(2.0*np.pi/particle_number_angular), (length/(particle_number_radial-1)) )   # NOTE; with this approach to separation, you want the two separations to be similar (your angular and radial) to avoid excess connections
    connection_args = 1.1*max( connection_args, shell_radius/shell_number )
    
    use_default_particles(filename, "cylinder", args_list, coords_list, connection_mode, connection_args)

def use_fibre_2d_sphere_shelllayers(filename, length, shell_radius_max, shell_number, particle_radius, particle_separation, connection_mode, connection_args):
    coords_list = get_fibre_2d_shelllayers_points(length, shell_radius_max, shell_number, particle_separation)
    args_list = [[particle_radius]] * len(coords_list)

    use_default_particles(filename, "sphere", args_list, coords_list, connection_mode, connection_args)

def use_fibre_2d_cylinder_shelllayers(filename, length, shell_radius_max, shell_number, particle_length, particle_radius, particle_separation, connection_mode, connection_args):
    coords_list = get_fibre_2d_shelllayers_points(length, shell_radius_max, shell_number, particle_separation)
    args_list = [[particle_radius, particle_length, 0.0, 0.0]] * len(coords_list)
    
    use_default_particles(filename, "cylinder", args_list, coords_list, connection_mode, connection_args)

def use_refine_cuboid(filename, dimensions, separations, object_offset, particle_size, particle_shape="sphere"):
    #
    # particle_size = radius of sphere OR half width of cube
    #
    coords_list = get_refine_cuboid(dimensions, separations, particle_size)
    num_particles = len(coords_list)
    coords_list = np.array(coords_list) + object_offset
    args_list = [[particle_size]] * num_particles
    
    use_default_particles(filename, particle_shape, args_list, coords_list, connection_mode="dist", connection_args=0.0)
    return num_particles

def use_refine_arch_prism(filename, dimensions, separations, deflection, object_offset, particle_size, particle_shape="sphere", place_regime="squish", prism_type="rect", prism_args=[1.0e-6]):
    #
    # particle_size = radius of sphere OR half width of cube
    #
    coords_list = get_refine_arch_prism(dimensions, separations, particle_size, deflection, place_regime, prism_type, prism_args)
    num_particles = len(coords_list)
    coords_list = np.array(coords_list) + object_offset
    args_list = [[particle_size]] * num_particles
    
    use_default_particles(filename, particle_shape, args_list, coords_list, connection_mode="dist", connection_args=0.0)
    return num_particles

def use_refine_sphere(filename, dimension, separations, object_offset, particle_size, particle_shape="sphere", place_regime="squish", makeCube=False, material="FusedSilica"):
    #
    # particle_size = radius of sphere OR half width of cube
    #
    coords_list = get_refine_sphere(dimension, separations, particle_size, place_regime, makeCube)
    num_particles = len(coords_list)
    coords_list = np.array(coords_list) + object_offset
    args_list = [[particle_size]] * num_particles
    
    use_default_particles(filename, particle_shape, args_list, coords_list, connection_mode="dist", connection_args=0.0, material=material)
    return num_particles

def use_refine_cube_showcase(filename, dimension, object_offset, particle_shape="sphere", material="FusedSilica"):
    #
    # particle_size = radius of sphere OR half width of cube
    #
    coords_list, args_list = get_refine_cube_showcase(dimension)
    num_particles = len(coords_list)
    
    sphere_coords_list = np.array(coords_list) + object_offset
    cube_coords_list = np.array(coords_list) + object_offset +np.array([0.0, 0.0, -3.0*dimension])

    particle_list = []
    for i in range(len(sphere_coords_list)):
        particle_list.append( {"material":material, "shape":"sphere", "args":args_list[i], "coords":sphere_coords_list[i], "altcolour":True} )
        particle_list.append( {"material":material, "shape":"cube"  , "args":args_list[i], "coords":cube_coords_list[i]  , "altcolour":True} )
    write_particles(filename, particle_list, 1e-07, material, "dist", 0.0 )
   
    return num_particles

def use_single_dipole_exp(filename, test_type, test_args, dipole_size, object_offset=[0.0, 0.0, 0.0], extra_args=[]):
    #
    # Sets up particles for a given test and arguements supplied
    # Experiments consdiered here concern single dipoles being setup in various systems to test how radiation forces vary and if problems will occur with single dipoles
    # Since individual dipoles will be considered, all particle primitives will cubes to reflect this
    #
    coords_list = get_single_dipole_exp(test_type, test_args, dipole_size, extra_args)
    num_particles = len(coords_list)
    coords_list = np.array(coords_list) + object_offset
    args_list = [[dipole_size]] * num_particles
    
    use_default_particles(filename, "cube", args_list, coords_list, connection_mode="dist", connection_args=0.0)
    return num_particles


def use_fill_spheredisc(filename, disc_radius, separation, particle_size, object_offset, particle_shape, mode="disc", material="FusedSilica", fix_to_ring=False):
    match mode:
        case "disc":
            coords_list = get_fill_disc(disc_radius, separation, particle_size, fix_to_ring=fix_to_ring)
        case "sphere":
            coords_list = get_fill_sphere(disc_radius, separation, particle_size, fix_to_ring=fix_to_ring)
        case _:
            print("Unknown mode for sphere-disc model, generating no particles")
    num_particles = len(coords_list)
    coords_list = np.array(coords_list) + object_offset
    args_list = [[particle_size]] * num_particles
    
    use_default_particles(filename, particle_shape, args_list, coords_list, connection_mode="dist", connection_args=0.0, material=material)
    return num_particles

def use_stretch_sphere(filename, dimension, particle_size, transform_factor, critical_transform_factor, func_transform, object_offset, particle_shape="sphere", connection_mode="dist", connection_args=0.0, material="FusedSilica"):
    coords_list, connection_args = get_stretch_sphere(dimension, particle_size, transform_factor, critical_transform_factor, func_transform, connection_mode, connection_args)
    num_particles = len(coords_list)
    coords_list = np.array(coords_list) + object_offset
    args_list = [[particle_size]] * num_particles
    
    use_default_particles(filename, particle_shape, args_list, coords_list, connection_mode=connection_mode, connection_args=connection_args, material=material)
    return num_particles


def use_default_particles(filename, shape, args_list, coords_list, connection_mode, connection_args, material="FusedSilica"):
    """
    Fills in typical particle parameters e.g. material, but leaves particles general.
    """
    default_radius = 1e-07
    default_material = "FusedSilica"
    particle_list = [{"material":material, "shape":shape, "args":args_list[i], "coords":coords_list[i], "altcolour":True} for i in range(len(coords_list))]
    write_particles(filename, particle_list, default_radius, default_material, connection_mode, connection_args )


#=======================================================================
# Beam configurations
#=======================================================================

def use_beam(filename, beam, translation=None, translationargs=None, translationtype=None, rotation=None, E0=1.5e7, w0=0.4, z_offset=0):
    match beam:
        case "GAUSS_CSP":
            use_gaussCSP_beam(filename,translation=translation, translationargs=translationargs, translationtype=translationtype, rotation=rotation)
        case "LAGUERRE":
            use_laguerre3_beam(filename,translation, translationargs, translationtype, rotation=rotation)
        case "BESSEL":
            use_bessel_beam(filename, translation, translationargs, translationtype, rotation=rotation)
        case "STRETCHER":
            use_stretcher_beam(filename, E0, w0, translation)

        case _:
            print(f"Beam '{beam}' unknown, using LAGUERRE. Options are LAGUERRE, BESSEL")

def use_gaussCSP_beam(filename, E0=1.5e7, w0=0.4, translation="0.0 0.0 0.0", translationargs=None, translationtype=None, rotation=None):
    """
    Makes a Gaussian complex source point beam
    """
    beam = {"beamtype":"BEAMTYPE_GAUSS_CSP", "E0":E0, "order":3, "w0":w0, "jones":"POLARISATION_LCP", "translation":translation, "translationargs":translationargs, "translationtype":translationtype, "rotation":rotation}
    write_beams(filename, [beam])

def use_laguerre3_beam(filename, translation, translationargs, translationtype=None, rotation=None):
    """
    Makes a Laguerre-Gaussian beam.
    """
    beam = {"beamtype":"BEAMTYPE_LAGUERRE_GAUSSIAN", "E0":300, "order":3, "w0":0.6, "jones":"POLARISATION_LCP", "translation":translation, "translationargs":translationargs, "translationtype":translationtype, "rotation":rotation}
    write_beams(filename, [beam])

def use_bessel_beam(filename, translation, translationargs, translationtype=None, rotation=None):
    """
    Makes a Laguerre-Gaussian beam.
    """
    beam = {"beamtype":"BEAMTYPE_BESSEL", "E0":1.5e7, "order":1, "jones":"POLARISATION_LCP", "translation":translation, "translationargs":translationargs, "translationtype":translationtype, "rotation":rotation}
    write_beams(filename, [beam])

def use_stretcher_beam(filename, E0=1.5e7, w0=0.4, translation=None):
    """
    Makes two counter-propagating Gaussian beams.
    """
    # NOTE; Same translation for both since translation applied after rotation, therefore will translate in the opposite direction
    beam1 = {"beamtype":"BEAMTYPE_GAUSS_CSP", "E0":E0, "order":3, "w0":w0, "jones":"POLARISATION_X", "translation":translation, "translationargs":None, "translationtype":None, "rotation":None}
    beam2 = {"beamtype":"BEAMTYPE_GAUSS_CSP", "E0":E0, "order":3, "w0":w0, "jones":"POLARISATION_X", "translation":translation, "translationargs":None, "translationtype":None, "rotation":"180 90.0"}
    write_beams(filename, [beam1, beam2])   #POLARISATION_X


#=======================================================================
# Option configurations
#=======================================================================

def use_default_options(filename, frames, show_output, time_step=1e-4, dipole_radius=40e-9, show_stress=False):
    """
    Make the default options, requiring just filename, frames, show_output
    """

    # frames, show_output, wavelength=1e-6, dipole_radius=4e-8, time_step=0.0001, vmd_output=True, excel_output=True, include_force=True, include_couple=True, frame_interval=2, max_size=2e-6, resolution=201, frame_min=0, frame_max=None, z_offset=0.0, show_stress=False, verbosity=0, include_dipole_forces=False, polarisability_type="RR", force_terms=["optical", "spring", "bending"]
    # To clear the old YAML before writing the new
    # NOTE; Requires this function to be run before any other writes occur
    with open(f"{filename}.yml", "w") as _:
        pass
    # Continue writing in blank slate
    option_parameters = fill_yaml_options({"frames":frames, "show_output": show_output, "time_step":time_step, "dipole_radius":dipole_radius, "show_stress":show_stress})
    write_options(filename, option_parameters)

def use_parameter_options(filename, option_parameters):
    """
    Make the default options, requiring just filename, frames, show_output
    """

    # frames, show_output, wavelength=1e-6, dipole_radius=4e-8, time_step=0.0001, vmd_output=True, excel_output=True, include_force=True, include_couple=True, frame_interval=2, max_size=2e-6, resolution=201, frame_min=0, frame_max=None, z_offset=0.0, show_stress=False, verbosity=0, include_dipole_forces=False, polarisability_type="RR", force_terms=["optical", "spring", "bending"]
    # To clear the old YAML before writing the new
    # NOTE; Requires this function to be run before any other writes occur
    with open(f"{filename}.yml", "w") as _:
        pass
    # Continue writing in blank slate
    write_options(filename, option_parameters)

def fill_yaml_options(non_default_params):
    """
    Keys are:

    frames, wavelength, dipole_radius, time_step, polarisability_type, constants (bending), stiffness_spec (type, default_value),
    equilibrium_shape, vmd_output, excel_output, include_force, include_couple, verbosity, include_dipole_forces, force_terms, show_output,
    show_stress, frame_interval, max_size, resolution, frame_min, frame_max, z_offset, beam_planes, quiver_setting.
    """
    # Fills in the non default YAML options.
    # Used to set the sets of options at the top of each YAML - so doesn't include beams or particles
    option_parameters = {
        "frames" : 1,
        "wavelength": 1.0e-6,
        "dipole_radius": 40e-9,
        "time_step": 1e-4,
        "polarisability_type": "RR",
        "constants": {"bending":0.1e-18},
        "stiffness_spec": {"type":"", "default_value":5e-6},
        "equilibrium_shape": None,

        "vmd_output": True,
        "excel_output": True,
        "include_force": True,
        "include_couple": True,
        "verbosity": 0,
        "include_dipole_forces": False,
        "force_terms": ["optical", "spring", "bending", "buckingham"],

        "show_output": True,
        "show_stress": False,
        "frame_interval": 2,
        "max_size": 2e-6,
        "resolution": 201,
        "frame_min": 0,
        "frame_max": 1,
        "z_offset": 0.0e-6,
        "beam_planes": [["z", 0.0]],
        "quiver_setting": 1,
    }
    option_parameters.update(non_default_params)
    if option_parameters["frame_max"] > option_parameters["frames"]: option_parameters["frame_max"] = option_parameters["frames"]
    return option_parameters

#=======================================================================
# Core functions which write options, beams and particles
#=======================================================================

def write_options(filename, option_parameters):
    """
    Base function to write the passed options to filename.
    """
    # join forces terms into one string when writing.
    if isinstance(option_parameters["force_terms"], list):
        option_parameters["force_terms"] = " ".join(option_parameters["force_terms"]) # max options are ["optical", "spring", "bending", "buckingham"]

    with open(f"{filename}.yml", "a") as file:
        file.write(f"options:\n")
        file.write(f"  frames: {option_parameters['frames']}\n")

        file.write(f"parameters:\n")
        for var in ["wavelength", "dipole_radius", "time_step", "polarisability_type"]:
            file.write(f"  {var}: {option_parameters[var]}\n")

        # write dictionaries into parameters list:
        for dict_name in ["constants", "stiffness_spec"]:
            file.write(f"  {dict_name}:\n")
            for key, val in option_parameters[dict_name].items():
                file.write(f"    {key}: {val}\n")
        
        # Write spring natural length override (specify custom single float natural length for all springs to use, not auto-generated)
        file.write(f"  equilibrium_shape: {option_parameters['equilibrium_shape']}\n")

        file.write(f"output:\n")
        for var in ["vmd_output", "excel_output", "include_force", "include_couple", "verbosity", "include_dipole_forces", "force_terms"]:
            file.write(f"  {var}: {option_parameters[var]}\n")

        file.write(f"display:\n")
        for var in ["show_output", "show_stress", "frame_interval", "resolution", "frame_min", "frame_max", "z_offset", "beam_planes", "quiver_setting"]:
            file.write(f"  {var}: {option_parameters[var]}\n")


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
                (particle["shape"] == "torus" and len(particle["args"]) == 4) or
                (particle["shape"] == "cylinder" and len(particle["args"]) == 4) or
                (particle["shape"] == "cube" and len(particle["args"]) == 1)
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

def get_torus_points_args(num_particles, separation, inner_radius, tube_radius):
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

    return coords_list, args_list

def get_sheet_points(num_length, num_width, separation, mode="triangle", formation="square", bounds=[2e-6]):
    # Makes a sheet in the z=0 plane. width in x-axis, length in y-axis.
    # modes are "triangle", "square", "hexagon"

    # For triangle and square, each point gives a shape so the nums are numbers of points.
    # For hexagon, the nums are number of hexagons to prevent unformed hexagons.

    def check_bounds(formation, point, bounds):
        #
        # formation = string name of formation type; exclude points outside this
        # point  = [X,Y,Z]
        # bounds = Assumes centered at origin
        #       for square: [width, height]
        #       for circle: [radius]
        #
        withinBounds=False
        match formation:
            case "square":
                width, height = bounds
                withinX = ( -width/2.0 <= point[0]) and (point[0] <= width/2.0)
                withinY = (-height/2.0 <= point[1]) and (point[1] <= height/2.0)
                withinBounds = withinX and withinY
            case "circle":
                radius = bounds[0]
                withinBounds = (point[0]**2 + point[1]**2) <= radius**2
            case _:
                withinBounds=True
                print("Unrecognised bounds for sheet formation, accepting all points")
        return withinBounds

    coords_list = []
    match mode:
        case "triangle" | "square":
            # use the same method for all sheets of this type, where each row is offset. So shape_angle is the angle between lattice vectors.
            # shape_angle can be set to anything else as well, although may not produce good connections.
            if mode == "triangle":
                shape_angle = np.pi/3
            elif mode == "square":
                shape_angle = np.pi/2
            
            centralise_shift = separation * np.array([(num_width-1)/2, (num_length-1)/2, 0]) # centre the sheet at the origin.
            row = np.zeros((num_width,3)) - centralise_shift
            row[:,0] += np.arange(0.0, separation*num_width, separation) # += so centralise_shift stays.
            offset = np.zeros(3)

            for i in range(num_length):
                coords_list.extend(row + offset)

                # offset each row by a cumulative offset
                if i%2:
                    angle = np.pi - shape_angle
                else:
                    angle = shape_angle
                offset += separation * np.array([np.cos(angle), np.sin(angle), 0])

        case "hexagon":
            # make a grid of triangles then place hexagons around that. These triangles are separated by an extra factor of sqrt3 compared to the hexagon sides.
            tri_grid = []     
            sqrt3 = np.sqrt(3)       
            
            # start with unit distances, then multiply by separation later.
            row = np.zeros((num_width,3))
            row[:,0] += np.arange(0.0, num_width, 1)*sqrt3
            offset = np.zeros(3)

            for i in range(num_length):
                tri_grid.append(row + offset) # tri_grid is not as flat as in the triangle mode.

                # offset each row by a cumulative offset
                if i%2:
                    angle = 2*np.pi/3
                else:
                    angle = np.pi/3
                offset += np.array([np.cos(angle), np.sin(angle), 0])*sqrt3

            # Make points around a hexagon, then add a subset of them to each tri_grid point depending on its placement.
            hex_points = np.array([(np.cos(i/6*2*np.pi - np.pi/6), np.sin(i/6*2*np.pi - np.pi/6), 0) for i in range(6)])
            for row_i in range(num_length):
                for j in range(num_width):
                    tri_coord = tri_grid[row_i][j]
                    # Right vertices
                    coords_list.append(tri_coord + hex_points[0]) 
                    coords_list.append(tri_coord + hex_points[1])
                    if row_i == num_length-1: # Top vertex
                        coords_list.append(tri_coord + hex_points[2])
                    if j == 0: # Left vertices
                        coords_list.append(tri_coord + hex_points[3])
                        coords_list.append(tri_coord + hex_points[4])
                    if row_i == 0: # Bottom vertex
                        coords_list.append(tri_coord + hex_points[5])

            coords_list = np.array(coords_list)
            centralise_shift = separation * np.array([(np.max(coords_list[:,0]) + np.min(coords_list[:,0]))/2, (np.max(coords_list[:,1]) + np.min(coords_list[:,1]))/2, 0]) # centre the sheet at the origin.
            coords_list = coords_list * separation - centralise_shift

        case _:
            sys.exit(f"Generate_yaml: get_sheet_points: unknown mode, {mode}")

    coords_list_reduced = []
    for coord in coords_list:
        if(check_bounds(formation, coord, bounds)):
            coords_list_reduced.append(coord)

    return coords_list_reduced

def get_filament_points(length, radius, separation):
    # Filment running along the y-axis.
    # Separation is the particle separation.
    coords_list = []
    circle_list = []
    ros2 = (radius/separation)**2
    num_rad = int(radius//separation)
    num_len = int(length//separation)
    for x in range(-num_rad, num_rad+1):
        for z in range(-num_rad, num_rad+1):
            if x**2 + z**2 <= ros2:
                circle_list.append([x*separation,-length/2,z*separation])

    circle_list = np.array(circle_list)
    for l in range(num_len):
        coords_list.extend(circle_list + [0,l*separation,0])
    return coords_list

def get_fibre_2d_hollowshell_points(length, shell_radius, particle_number_radial, particle_number_angular):
    coords_list = []

    angular_separation = (2.0*np.pi) / particle_number_angular      # Theta separation between points
    radial_separation  = (length) / (particle_number_radial-1)          # Spacing anlong width of the hollow cylinder shell being generated
    radial_offset = length/2.0    # To centre the cylinder at the origin

    for i in range(particle_number_radial):
        for j in range(particle_number_angular):
            coords_list.append( [radial_separation*i -radial_offset, shell_radius*np.cos(angular_separation*j), shell_radius*np.sin(angular_separation*j)] )

    return coords_list

def get_fibre_2d_thick_points(length, shell_radius, shell_number, particle_number_radial, particle_number_angular, include_center_line=True):
    coords_list = []

    angular_separation = (2.0*np.pi) / particle_number_angular      # Theta separation between points
    radial_separation  = (length) / (particle_number_radial-1)          # Spacing anlong width of the hollow cylinder shell being generated
    radial_offset = length/2.0    # To centre the cylinder at the origin

    for k in range(shell_number+1):
        shell_sub_radius = k*shell_radius/shell_number
        for i in range(particle_number_radial):
            for j in range(particle_number_angular):
                if(k==0):
                    if(include_center_line and j==0):
                        # Line at centre
                        coords_list.append( [radial_separation*i -radial_offset, 0.0, 0.0] )
                else:
                    # Surrounding shells
                    coords_list.append( [radial_separation*i -radial_offset, shell_sub_radius*np.cos(angular_separation*j), shell_sub_radius*np.sin(angular_separation*j)] )
    return coords_list

def get_fibre_2d_shelllayers_points(length, shell_radius_max, shell_number, particle_separation):
    coords_list = []

    particle_number_radial = int(np.ceil(length / particle_separation)) ##### COULD DO BETTER BY JUST MANUALLY CONNECTING IN THE CONNECTIONS INDEX, HOWEVER WITH THIS MAY GET GAPS THAT LET PARTICLES THROUGH TOO EASILY ####
    radial_offset = (particle_number_radial*particle_separation)/2.0

    for k in range(1, shell_number+1):
        shell_sub_radius = k*shell_radius_max/shell_number
        angular_separation = particle_separation/shell_sub_radius
        particle_number_angular = int(np.ceil( (2.0*np.pi*shell_sub_radius)/(particle_separation) ))
        for j in range(particle_number_radial):
            for i in range(particle_number_angular):
                coords_list.append([particle_separation*j -radial_offset, shell_sub_radius*np.cos(i*angular_separation), shell_sub_radius*np.sin(i*angular_separation)])

    return coords_list

def get_refine_cuboid(dimensions, separations, particle_size):
    #
    # particle_size = radius of sphere OR half width of cube
    #
    coords_list = []
    dimensions = np.array(dimensions)
    separations = np.array(separations)

    particle_numbers = np.floor((dimensions - separations) / (2*particle_size))      # Number of particles in each axis
    particle_step = np.zeros(3)
    for i in range(3):
        if particle_numbers[i] != 1.0:
            particle_step[i] = 2.0*particle_size + separations[i]/(particle_numbers[i]-1)               # Step in position to each particle in each axis
    
    for i in range(int(particle_numbers[0])):
        for j in range(int(particle_numbers[1])):
            for k in range(int(particle_numbers[2])):
                coords_list.append([i*particle_step[0] -dimensions[0]/2.0+particle_size, j*particle_step[1] -dimensions[1]/2.0+particle_size, k*particle_step[2] -dimensions[2]/2.0+particle_size])

    return coords_list

def get_refine_arch_prism(dimensions, separations, particle_size, deflection, place_regime="squish", prism_type="rect", prism_args=[1.0e-6]):
    #
    # particle_size = radius of sphere OR half width of cube
    #

    def check_prism_bounds(prism_type, point, args):
        #
        # Checks whether a point is within the bounds for a given prism
        #
        withinBounds=False
        match prism_type:
            case "circle":
                # args = [radius]
                withinBounds = np.sqrt( pow(point[0],2) + pow(point[1],2) ) <= args[0]
            case "rect":
                # args = [half_width, half_height]
                withinBounds = ( (-(args[0]+sys.float_info.epsilon) <= point[0]) and (point[0] <= (args[0]+sys.float_info.epsilon)) ) and ( (-(args[1]+sys.float_info.epsilon) <= point[1]) and (point[1] <= (args[1]+sys.float_info.epsilon)) )
            case "triangle":
                pass
        return withinBounds
    
    # def get_deflected_width(b_prime):
    #     # Works best for low eccentricity -> Close to circle in plane

    #     a = dimensions[0]#2.0e-6      # Width (original, to conserve)
    #     b = 0.01e-6#1.0e-6     # Deflection (original, to conserve)
    #     #b_prime = 0.01e-6   # Deflection (New)
    #     c = (3.0/2.0)*(a+b) -np.sqrt(a*b)
    #     d = b_prime -(c*(2.0/3.0))
    #     a_tild = (np.sqrt(b_prime)/3.0) +np.sqrt(-d +(b_prime/9.0))
    #     a_prime = pow(a_tild, 2)    # Width (New)

    #     # print("===")
    #     # print("a= ",a)
    #     # print("b= ",b)
    #     # print("a_prime= ",a_prime)
    #     # print("b_prime= ",b_prime)
    #     return a_prime

    def get_deflection_offset(x, deflection, dimension):
        #
        # Z coordinate change caused by an overall deflection
        # 
        # x = position to get deflected height for, from the overall deflection
        # deflection = raw height deflection in regular metres => e-6 included
        # dimension = width of rod (dimensions[0])
        #

        # Sinusoidal bending
        return deflection*np.sin( np.pi*((x/dimension) +1.0/2.0))

    coords_list = []
    # Get number of particles to place in each axis
    particle_number = np.floor((dimensions-separations) / (2.0*particle_size))

    # Get particle displacement in each axis [X,Y,Z]
    displacement = np.array([0.0, 0.0, 0.0])
    for i in range(3):
        if(particle_number[i] != 1.0):
            match place_regime:
                case "squish":
                    displacement[i] = 2.0*particle_size +separations[i]/(particle_number[i]-1.0)
                case "spaced":
                    displacement[i] = 2.0*particle_size +separations[i]/(particle_number[i]-1.0) +(dimensions[i]-separations[i]-particle_number[i]*2.0*particle_size)/(particle_number[i]+1)
                case "ends":
                    pass
                case _:
                    print("--Unrecgonised place_regime: "+str(place_regime)+"--")
    # Get particle origin -> Location of 0th particle to iterate from
    origin = np.array([0.0, 0.0, 0.0])
    for i in range(3):
        match place_regime:
            case "squish":
                origin[i] = -(particle_number[i]*displacement[i])/2.0 +particle_size
            case "spaced":
                origin[i] = -(particle_number[i]*displacement[i])/2.0 +particle_size
            case "ends":
                pass
            case _:
                print("--Unrecgonised place_regime: "+str(place_regime)+"--")
    # Generate grids in the YZ plane
    grid_coords = []
    for j in range(int(particle_number[1])):
        j_coord = origin[1] +j*displacement[1]
        for k in range(int(particle_number[2])):
            k_coord = origin[2] +k*displacement[2]
            if(check_prism_bounds(prism_type, [j_coord, k_coord], args=prism_args)):
                grid_coords.append(np.array([0.0, j_coord, k_coord]))
    # Place grids of particles along X axis
    for i in range(int(particle_number[0])):
        for grid_coord in grid_coords:
            coords_list.append( 
                [
                    origin[0] +i*displacement[0] +grid_coord[0],
                    grid_coord[1],
                    grid_coord[2] + get_deflection_offset(origin[0] +i*displacement[0] +grid_coord[0], deflection, dimensions[0])
                ] 
            )

    # print("====")
    # print("particle_number = ", particle_number)
    # print("grid_coords = ", grid_coords)
    # print("coords_list = ", coords_list)
    return coords_list

def get_refine_sphere(dimension, separations, particle_size, place_regime="squish", makeCube=False):
    #
    # particle_size = radius of sphere OR half width of cube
    #

    def check_bounds(point, radius):
        #
        # Checks whether a point is within the spherical/cubic bounds
        #
        if makeCube:
            return (abs(point[0])<radius) and (abs(point[1])<radius) and (abs(point[2])<radius)
        else:
            return pow(point[0],2) + pow(point[1],2) + pow(point[2],2) <= radius**2


    coords_list = []
    # Get number of particles to place in each axis
    particle_number = np.floor((dimension-np.array(separations)) / (2.0*particle_size))

    # Get particle displacement in each axis [X,Y,Z]
    displacement = np.array([0.0, 0.0, 0.0])
    for i in range(3):
        if(particle_number[i] != 1.0):
            match place_regime:
                case "squish":
                    displacement[i] = 2.0*particle_size +separations[i]/(particle_number[i]-1.0)
                case "spaced":
                    displacement[i] = 2.0*particle_size +separations[i]/(particle_number[i]-1.0) +(dimension-separations[i]-particle_number[i]*2.0*particle_size)/(particle_number[i]+1)
                case "ends":
                    pass
                case _:
                    print("--Unrecgonised place_regime: "+str(place_regime)+"--")
    # Get particle origin -> Location of 0th particle to iterate from
    origin = np.array([0.0, 0.0, 0.0])
    for i in range(3):
        match place_regime:
            case "squish":
                if(particle_number[i] > 1):
                    origin[i] = -(particle_number[i]*displacement[i])/2.0 +particle_size
            case "spaced":
                if(particle_number[i] > 1):
                    origin[i] = -(particle_number[i]*displacement[i])/2.0 +particle_size
            case "ends":
                pass
            case _:
                print("--Unrecgonised place_regime: "+str(place_regime)+"--")
    # Generate grids in the YZ plane
    for i in range(int(particle_number[0])):
        i_coord = origin[0] +i*displacement[1]
        for j in range(int(particle_number[1])):
            j_coord = origin[1] +j*displacement[1]
            for k in range(int(particle_number[2])):
                k_coord = origin[2] +k*displacement[2]
                if(check_bounds([i_coord, j_coord, k_coord], dimension/2.0)):
                    coords_list.append( 
                        [
                            i_coord,
                            j_coord,
                            k_coord
                        ] 
                    )

    # print("====")
    # print("particle_number = ", particle_number)
    # print("coords_list = ", coords_list)
    return coords_list

def get_refine_cube_showcase(dimension):
    #
    # Shows side-by-side different refinements of a cube
    #
    coords_list = []
    args_list = []
    mesh_spacing = 3.0*dimension
    particle_spacing = dimension*0.01
    
    for p in range(3):   # Number of meshes to generate side-by-side
        particle_size = (dimension-p*particle_spacing)/(p+1)
        for i in range(p+1):            # Generate cubes of particles to create each mesh
            for j in range(p+1):        #
                for k in range(p+1):    #
                    coords_list.append( [p*mesh_spacing +(i)*particle_spacing +(2.0*i+1.0)*(particle_size), (j)*particle_spacing +(2.0*j+1.0)*(particle_size), (k)*particle_spacing +(2.0*k+1.0)*(particle_size)] )
                    args_list.append([particle_size])
    return coords_list, args_list

def get_NsphereShell_points(radii, numbers_per_shell):
    coords_list = []
    for i in range(len(radii)):
        coords_list.extend(get_sunflower_points(numbers_per_shell[i], radii[i]))
    return coords_list

def get_single_dipole_exp(test_type, test_args, dipole_size, extra_args):
    coords_List = []
    invalidArgs = False
    match test_type:
        case "single":
            if( len(test_args) == 3 ):
                coords_List.append([0.0, 0.0, 0.0])
            else:
                invalidArgs=True

        case "7shell":
            if( len(test_args) == 3 ):
                coords_List.append([0.0, 0.0, 0.0])

                coords_List.append([-2.0*dipole_size, 0.0, 0.0])
                coords_List.append([ 2.0*dipole_size, 0.0, 0.0])
                coords_List.append([0.0, -2.0*dipole_size, 0.0])
                coords_List.append([0.0,  2.0*dipole_size, 0.0])
                coords_List.append([0.0, 0.0, -2.0*dipole_size])
                coords_List.append([0.0, 0.0,  2.0*dipole_size])
            else:
                invalidArgs=True

        case "7shell_difference":
            if( len(test_args) == 4 ):
                coords_List.append([0.0, 0.0, 0.0])

                coords_List.append([-2.0*dipole_size, 0.0, 0.0])
                coords_List.append([ 2.0*dipole_size, 0.0, 0.0])
                coords_List.append([0.0, -2.0*dipole_size, 0.0])
                coords_List.append([0.0,  2.0*dipole_size, 0.0])
                coords_List.append([0.0, 0.0, -2.0*dipole_size])
                coords_List.append([0.0, 0.0,  2.0*dipole_size])
            else:
                invalidArgs=True

        case "multi_separated":
            if( len(test_args) == 4 ):
                particle_separation = extra_args[0]
                particle_number = test_args[0]
                total_width = (particle_number-1)*particle_separation
                for i in range(particle_number):
                    coords_List.append([i*particle_separation -total_width/2.0, 0.0, 0.0])
            else:
                invalidArgs=True
        case _:
            print("Invalid test_type: ",test_type)
    if(invalidArgs):
        print("Invalid test_args: "+str(test_type)+", "+str(test_args))
    return coords_List

def get_fill_disc(disc_radius, separation, particle_size, fix_to_ring=False):
    coord_list = []

    layer_number = int(np.floor( (disc_radius+particle_size) / (2.0*particle_size +separation[1]) ))
    if(fix_to_ring):
        layer_number = 1

    for i in range(layer_number):
        sub_radius = disc_radius -i*(2.0*particle_size +separation[1])
        
        # Find number of particles in each ring, ignore if can only fit 1 particles in that ring (not enough symmetry)
        layer_particle_number = int(np.floor((2.0*np.pi*sub_radius) / (2.0*particle_size +separation[0])))  # Make sure this doesn't cause overlap
        if(layer_particle_number==1):
            layer_particle_number = 0

        theta_step = 0.0
        if(layer_particle_number!=0):
            if(sub_radius != 0.0):
                theta_step = (2.0*np.pi) / layer_particle_number

        for j in range(layer_particle_number):
            coord_list.append([sub_radius*np.cos(theta_step*j), sub_radius*np.sin(theta_step*j), 0.0])
    return coord_list

def get_fill_sphere(sphere_radius, separation, particle_size, fix_to_ring=False):
    #
    # Builds a filled sphere from layers of discs
    # Priorities having a central layer
    #
    coord_list = []

    layer_number = int(np.floor( (sphere_radius) / (2.0*particle_size +separation[2]) ))
    for i in range(layer_number):
        sub_radius = np.sqrt(sphere_radius**2 -(i*(2.0*particle_size +separation[2]))**2)
        disc_coord_list = get_fill_disc(sub_radius, separation, particle_size, fix_to_ring=fix_to_ring)
        offset = i*(2.0*particle_size +separation[2])
        for coord in disc_coord_list:
            coord_list.append( [coord[0], coord[1], coord[2]+offset] )      # Central / Upper
            if(i!=0):
                coord_list.append( [coord[0], coord[1], coord[2]-offset] )  # Lower
    
    return coord_list

def get_stretch_sphere(dimension, particle_size, transform_factor, critical_transform_factor, func_transform, connection_mode, connection_args):
    #
    # dimension = full diameter of sphere
    # particle_size = radius of particle
    #
    coords_list = []

    # mesh_radius = dimension/2.0
    # base_separation = (2.0*particle_size)*np.sqrt(critical_transform_factor)
    # #number_of_particles_side = int(np.floor( (mesh_radius-particle_size) / (2.0*particle_size) ))
    # number_of_particles_side = int(np.floor( mesh_radius/base_separation ))
    # number_of_particles = 2*number_of_particles_side +1

    # # Generate some base sphere shape
    # base_separation = dimension/number_of_particles
    # for i in range(-number_of_particles_side, number_of_particles_side+1):
    #     i_coord = i*base_separation
    #     for j in range(-number_of_particles_side, number_of_particles_side+1):
    #         j_coord = j*base_separation
    #         for k in range(-number_of_particles_side, number_of_particles_side+1):
    #             k_coord = k*base_separation
    #             withinBounds = (i_coord**2 +j_coord**2 +k_coord**2) < mesh_radius**2
    #             if(withinBounds):   # Check will fit within a base sphere shape
    #                 coords_list.append([i_coord, j_coord, k_coord])

    coords_list, connection_mode, connection_args = get_stretch_sphere_equilibrium(dimension, particle_size, critical_transform_factor, connection_mode=connection_mode, connection_args=connection_args)

    # Modify this base sphere to get the ellipsoid / other shape to be generated
    coords_list = np.array(coords_list)
    transformed_coords_list = func_transform(coords_list, transform_factor)

    return transformed_coords_list, connection_args

def get_stretch_sphere_equilibrium(dimension, particle_size, critical_transform_factor, connection_mode=None, connection_args=None):
    coords_list = []
    
    mesh_radius = dimension/2.0
    base_separation = (2.0*particle_size)*np.sqrt(critical_transform_factor)    ### Should rename this, not really base sep, more like a min sep ###
    #number_of_particles_side = int(np.floor( (mesh_radius-particle_size) / (2.0*particle_size) ))
    number_of_particles_side = int(np.floor( mesh_radius/base_separation ))
    number_of_particles = 2*number_of_particles_side +1

    # Generate some base sphere shape
    base_separation = dimension/number_of_particles
    for i in range(-number_of_particles_side, number_of_particles_side+1):
        i_coord = i*base_separation
        for j in range(-number_of_particles_side, number_of_particles_side+1):
            j_coord = j*base_separation
            for k in range(-number_of_particles_side, number_of_particles_side+1):
                k_coord = k*base_separation
                withinBounds = (i_coord**2 +j_coord**2 +k_coord**2) < mesh_radius**2
                if(withinBounds):   # Check will fit within a base sphere shape
                    coords_list.append([i_coord, j_coord, k_coord])

    # Get connections if in manual mode; if in another mode ignore this
    if(connection_mode == "manual"):    # => connections to be made based on NN for original sphere
        print("Generating new connections...")
        connection_args = ""
        for i in range(len(coords_list)):    # Go through all particles
            for j in range(i, len(coords_list)):# Connect any adjacent (based on lattice grid)
                if(i != j):
                    dist = np.sqrt( (coords_list[i][0]-coords_list[j][0])**2 +(coords_list[i][1]-coords_list[j][1])**2 +(coords_list[i][2]-coords_list[j][2])**2 )
                    if(dist <= base_separation*1.01):
                        connection_args += f"{i} {j} "
                        connection_args += f"{j} {i} "
    
    return coords_list, connection_mode, connection_args