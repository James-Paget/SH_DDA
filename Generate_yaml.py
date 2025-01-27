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
            use_laguerre3_beam(filename, translation=None, translationargs="1.5e-6 0 0")
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

        case _:
            return False
    
    return True

        # case "11" | "CYLINDER":
        #     use_default_options(filename, frames=1, show_output=True)
        #     use_laguerre3_beam(filename)
        #     use_cylinder(filename, num_particles=3, length=1e-6, radius=2e-7, separation=0.2e-6, rotation_axis=[1,0,0], rotation_theta=0.7)
        #    # axis x and y rotation is SWAPPED!


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

def make_yaml_fibre_1d_sphere(filename, time_step=1e-4, frames=20, show_output=True, length=2e-6, particle_radius=0.2e-6, particle_number=10, connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_default_options(filename, frames, show_output, time_step=time_step)
    use_beam(filename, beam)
    use_fibre_1d_sphere(filename, length, particle_radius, particle_number, connection_mode, connection_args)

def make_yaml_fibre_1d_cylinder(filename, time_step=1e-4, frames=10, show_output=True, length=3e-6, particle_length=0.4e-6, particle_radius=0.1e-6, particle_number=5, connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_default_options(filename, frames, show_output, time_step=time_step)
    use_beam(filename, beam)
    use_fibre_1d_cylinder(filename, length, particle_length, particle_radius, particle_number, connection_mode, connection_args)

def make_yaml_fibre_2d_sphere_hollowshell(filename, time_step=1e-4, frames=1, show_output=True, length=3e-6, shell_radius=0.3e-6, particle_radius=0.1e-6, particle_number_radial=6, particle_number_angular=4, connection_mode="dist", connection_args=0.0, beam="LAGUERRE", include_beads=False):
    use_default_options(filename, frames, show_output, time_step=time_step)
    #use_beam(filename, beam, translation="2.5e-6 0.0 0.0", translationargs="1.0e-6 0.0 0.0")  ### DOES NOT ALLOW PARAMETER VARIATION EASILY ###
    # use_gaussCSP_beam(filename, E0=2.5e7, w0=0.4, translation="2.5e-6 0.0 0.0", translationargs="2.5e-6 1.5e-6 0.0")

    #2.5e-6 1.5e-6 0.0
    beam_1 = {"beamtype":"BEAMTYPE_GAUSS_CSP", "E0":3.1e7, "order":3, "w0":0.4, "jones":"POLARISATION_LCP", "translation": "2.3e-6 0.0 0.0", "translationargs": "-0.5 0.0 0.0 1.0 -1.2e-6 0.0 0.0", "translationtype":"circle", "rotation":None}
    beam_2 = {"beamtype":"BEAMTYPE_GAUSS_CSP", "E0":3.1e7, "order":3, "w0":0.4, "jones":"POLARISATION_LCP", "translation":"-2.3e-6 0.0 0.0", "translationargs":None, "translationtype":"linear", "rotation":None}
    write_beams(filename, [beam_1, beam_2])

    # Varies depending on if beads are included within this function
    use_fibre_2d_sphere_hollowshell(filename, length, shell_radius, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, include_beads=include_beads)

def make_yaml_fibre_2d_cylinder_hollowshell(filename, time_step=1e-4, frames=1, show_output=True, length=2e-6, shell_radius=1e-6, particle_length=0.5e-6, particle_radius=0.2e-6, particle_number_radial=3, particle_number_angular=8, connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_default_options(filename, frames, show_output, time_step=time_step)
    use_beam(filename, beam)
    use_fibre_2d_cylinder_hollowshell(filename, length, shell_radius, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args)

def make_yaml_fibre_2d_sphere_thick_uni(filename, time_step=1e-4, frames=1, show_output=True, length=3e-6, shell_radius=1e-6, shell_number=1, particle_radius=0.2e-6, particle_number_radial=4, particle_number_angular=6, connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_default_options(filename, frames, show_output, time_step=time_step)
    use_beam(filename, beam)
    use_fibre_2d_sphere_thick_uni(filename, length, shell_radius, shell_number, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args)

def make_yaml_fibre_2d_cylinder_thick_uni(filename, time_step=1e-4, frames=1, show_output=True, length=3e-6, shell_radius=1e-6, shell_number=1, particle_length=0.5e-6, particle_radius=0.2e-6, particle_number_radial=3, particle_number_angular=6, connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_default_options(filename, frames, show_output, time_step=time_step)
    use_beam(filename, beam)
    use_fibre_2d_cylinder_thick_uni(filename, length, shell_radius, shell_number, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args)

def make_yaml_fibre_2d_sphere_shelllayers(filename, time_step=1e-4, frames=1, show_output=True, length=1.5e-6, shell_radius_max=1.5e-6, shell_number=2, particle_radius=0.15e-6, particle_separation=(np.pi*2.0*1.0e-6)/(10.0), connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_default_options(filename, frames, show_output, time_step=time_step)
    use_beam(filename, beam)
    use_fibre_2d_sphere_shelllayers(filename, length, shell_radius_max, shell_number, particle_radius, particle_separation, connection_mode, connection_args)

def make_yaml_fibre_2d_cylinder_shelllayers(filename, time_step=1e-4, frames=1, show_output=True,  length=1.0e-6, shell_radius_max=1.5e-6, shell_number=2, particle_length=0.4e-6, particle_radius=0.15e-6, particle_separation=(np.pi*2.0*1.0e-6)/(10.0), connection_mode="dist", connection_args=0.0, beam="LAGUERRE"):
    use_default_options(filename, frames, show_output, time_step=time_step)
    use_beam(filename, beam)
    use_fibre_2d_cylinder_shelllayers(filename, length, shell_radius_max, shell_number, particle_length, particle_radius, particle_separation, connection_mode, connection_args)


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

def use_sheet_triangle(filename, num_length, num_width, separation, particle_radius, rotation_axis=[0,0,1], rotation_theta=0):
    args_list = [[particle_radius]] * num_length * num_width
    coords_list = get_sheet_points(num_length, num_width, separation, mode="triangle")
    if rotation_theta != 0:
        coords_list = rotate_coords_list(coords_list, rotation_axis, rotation_theta)
    use_default_particles(filename, "sphere", args_list, coords_list, "dist", 1.001*separation)

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

# def use_cylinder(filename, num_particles, length, radius, separation, rotation_axis=[0,0,1], rotation_theta=0):
#     # makes a row of separated cylinders
#     coords_list = get_cylinder_points(num_particles, length, separation)
#     if rotation_theta != 0:
#         coords_list = rotate_coords_list(coords_list, rotation_axis, rotation_theta)
#     # radius, width, theta_Z, theta_pitch
#     test_pt = rotate_coords_list([[0,1,0]], rotation_axis, rotation_theta)[0]
#     theta_Z, theta_pitch = np.arctan2(test_pt[1],test_pt[0]), np.arccos(np.clip(test_pt[2], -1, 1))
#     args_list = [[radius, length, theta_Z, theta_pitch] for (x,y,z) in coords_list] # the spherical angles of the piece positions ARE their individual rotation.
#     # XXX need to fix the 0,0,0 case.
#     print(f"ARGSLIST IS {args_list}")
    
#     use_default_particles(filename, "cylinder", args_list, coords_list, "num", 0)
    # use_default_particles(filename, "sphere", [[0.15e-6]] * len(coords_list), coords_list, "num", 0)



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

def use_beam(filename, beam, translation=None, translationargs=None, translationtype=None):
    match beam:
        case "GAUSS_CSP":
            use_gaussCSP_beam(filename,translation, translationargs, translationtype)
        case "LAGUERRE":
            use_laguerre3_beam(filename,translation, translationargs, translationtype)
        case "BESSEL":
            use_bessel_beam(filename, translation, translationargs, translationtype)
        case _:
            print(f"Beam '{beam}' unknown, using LAGUERRE. Options are LAGUERRE, BESSEL")

def use_gaussCSP_beam(filename, E0=1.5e7, w0=0.4, translation="0.0 0.0 0.0", translationargs=None, translationtype=None):
    """
    Makes a Gaussian complex source point beam
    """
    beam = {"beamtype":"BEAMTYPE_GAUSS_CSP", "E0":E0, "order":3, "w0":w0, "jones":"POLARISATION_LCP", "translation":translation, "translationargs":translationargs, "translationtype":translationtype, "rotation":None}
    write_beams(filename, [beam])

def use_laguerre3_beam(filename, translation, translationargs, translationtype=None):
    """
    Makes a Laguerre-Gaussian beam.
    """
    beam = {"beamtype":"BEAMTYPE_LAGUERRE_GAUSSIAN", "E0":300, "order":3, "w0":0.6, "jones":"POLARISATION_LCP", "translation":translation, "translationargs":translationargs, "translationtype":translationtype, "rotation":None}
    write_beams(filename, [beam])

def use_bessel_beam(filename, translation, translationargs, translationtype=None):
    """
    Makes a Laguerre-Gaussian beam.
    """
    beam = {"beamtype":"BEAMTYPE_BESSEL", "E0":1.5e7, "order":1, "jones":"POLARISATION_LCP", "translation":translation, "translationargs":translationargs, "translationtype":translationtype, "rotation":None}
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
    print(" WAVELENGTH = ",wavelength)
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
                (particle["shape"] == "torus" and len(particle["args"]) == 4) or
                (particle["shape"] == "cylinder" and len(particle["args"]) == 4)
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

def get_sheet_points(num_length, num_width, separation, mode="triangle"):
    # Makes a sheet in the z=0 plane. width in x-axis, length in y-axis.
    # modes are "triangle", "square", "hexagon"

    # For triangle and square, each point gives a shape so the nums are numbers of points.
    # For hexagon, the nums are number of hexagons to prevent unformed hexagons.

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

    return coords_list

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

# def get_cylinder_points(num_particles, length, separation):
#     # Cyclinders running along the y-axis.
#     # length is length of each cylinder
#     # coords_list = []
#     y_coords = np.linspace(-(num_particles-1)/2, (num_particles+1)/2, num_particles+1)[:-1] * (length + separation)
#     return [[0,y,0] for y in y_coords]
