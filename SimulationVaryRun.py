"""
Runs the core simulation using various YAML files to produce a dataset 
from a single python run
"""

import sys
import subprocess
import Generate_yaml
import numpy as np
import xlsxwriter
import pandas as pd
import random
import math
import pickle
import itertools as it
import os
from functools import partial, reduce
from operator import mul

import Display
import DipolesMulti2024Eigen as DM


def generate_yaml(filename, particle_list, parameters_arg, beam_type="BEAMTYPE_LAGUERRE_GAUSSIAN"):

    # All possible parameters_arg keys as strings are: 
    # frames, wavelength, dipole_radius, time_step, vmd_output, excel_output, include_force, include_couple, show_output, frame_interval,
    # max_size, resolution, frame_min, frame_max, z_offset, beamtype, E0, order, w0, jones, translation, rotation, default_radius, default_material

    # particle_list contains dictionaries with keys: material, shape, args, coords, altcolour

    def fetch_beam_parameters(beam_type):
        """
        . Specifies defualt values for different beams to be included in a YAML
        . NOTE; These values can be overwritten in generate_yaml()
        """
        parameters = {}
        match beam_type:
            case "BEAMTYPE_LAGUERRE_GAUSSIAN":
                parameters = {
                    "beamtype": "BEAMTYPE_LAGUERRE_GAUSSIAN",
                    "E0": 300,
                    "order": 3,
                    "w0": 0.6,
                    "jones": "POLARISATION_LCP",
                    "translation": None,
                    "rotation": None,
                }
            case "BEAMTYPE_BESSEL":
                parameters = {
                    "beamtype": "BEAMTYPE_BESSEL",
                    "E0": 25e6,
                    "w0": 0.6,
                    "jones": "POLARISATION_LCP",
                    "translation": None,
                    "rotation": None,
                }
            case _:
                print("Beam type not recognised in YAML generation; ",beam_type)
        return parameters

    # Set default parameters
    parameters = {
        "frames" : 1,
        "wavelength": 1.0e-6,
        "dipole_radius": 40e-9,
        "time_step": 1e-4,
        "polarisability_type": "RR",
        "equilibrium_shape":None,

        "vmd_output": True,
        "excel_output": True,
        "include_force": True,
        "include_couple": True,
        "verbosity": 0,
        "include_dipole_forces": False,
        "force_terms": "optical",

        "show_output": True,
        "show_stress":False,
        "frame_interval": 2,
        "max_size": 2e-6,
        "resolution": 201,
        "frame_min": 0,
        "frame_max": 1,
        "z_offset": 0.0e-6,

        "beamtype": "BEAMTYPE_LAGUERRE_GAUSSIAN",
        "E0": 0,
        "order": 3,
        "w0": 0.6,
        "jones": "POLARISATION_LCP",
        "translation": None,
        "rotation": None,

        "default_radius": 100e-9,
        "default_material": "FusedSilica",
        "connection_mode": "manual",
        "connection_args": "",
    }
    # Update with beam specific parameters -> Fills in any fields specific to beam as default before the next .update is applied
    parameters.update( fetch_beam_parameters(beam_type) )

    # Overwrite parameters with any passed in with parameters_arg
    parameters.update(parameters_arg)

    # Make connection_args a string
    parameters['connection_args'] = " ".join([str(x) for x in parameters['connection_args']])
    
    # Write into a YAML file
    print(f"Generated YAML : {filename}")
    file = open(f"{filename}.yml", "w")
    
    file.write("options:\n")
    file.write(f"  frames: {parameters['frames']}\n")

    file.write("parameters:\n")
    for arg in ["wavelength", "dipole_radius", "time_step", "polarisability_type"]:
        file.write(f"  {arg}: {parameters[arg]}\n")

    file.write("output:\n")
    for arg in ["vmd_output", "excel_output", "include_force", "include_couple", "verbosity", "include_dipole_forces", "force_terms"]:
        file.write(f"  {arg}: {parameters[arg]}\n")

    file.write("display:\n")
    for arg in ["show_output", "show_stress", "frame_interval", "max_size", "resolution", "frame_min", "frame_max", "z_offset"]:
        file.write(f"  {arg}: {parameters[arg]}\n")

    ##
    ## MAKE WORK FOR BOTH BESSEL AND LAGUERRE
    ##
    file.write("beams:\n")
    file.write("  beam_1:\n")
    if(parameters["beamtype"]=="BEAMTYPE_LAGUERRE_GAUSSIAN"):
        for arg in ["beamtype", "E0", "order", "w0", "jones", "translation", "rotation"]:
            file.write(f"    {arg}: {parameters[arg]}\n")
    else:
        for arg in ["beamtype", "E0", "w0", "jones", "translation", "rotation"]:
            file.write(f"    {arg}: {parameters[arg]}\n")

    file.write("particles:\n")
    for arg in ["default_radius", "default_material", "connection_mode", "connection_args"]:
        file.write(f"  {arg}: {parameters[arg]}\n")


    # Write particle list.
    file.write("  particle_list:\n")
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

    file.close()


def generate_sphere_yaml(particle_formation, number_of_particles, particle_material="FusedSilica", characteristic_distance=1e-6, particle_radii = 200e-9, parameters={}):
    #
    # Generates a YAML file for a set of identical spheres with given parameters
    # This will overwrite files with the same name
    #
    # number_of_particles = Number of spheres to generate
    # characteristic_distance = key distance for each formation, e.g. radii of circle particles are placed on, length of edge for cubic formation, etc
    #
    
    # Create / overwrite YAML file
    # Writing core system parameters

    filename = "SingleLaguerre"
    particle_list = []

    # Writing specific parameters for particle formation
    for particle_index in range(number_of_particles):
        match(particle_formation):
            case "circle":
                theta_jump = (2.0*np.pi)/number_of_particles
                particle_theta = theta_jump*particle_index
                particle_position = [characteristic_distance*np.cos(particle_theta), characteristic_distance*np.sin(particle_theta), 1.0e-6]   #0.0
                position_offsets  = [
                    0.0,#random.random()*0.02*characteristic_distance, 
                    0.0,#random.random()*0.02*characteristic_distance, 
                    0.0 #random.random()*0.02*characteristic_distance
                ]
                coords = np.array(particle_position) + np.array(position_offsets)
                particle_list.append({"material": particle_material, "shape": "sphere", "args": [particle_radii], "coords": coords, "altcolour": True})
            
            case _:
                print("Particle formation invalid: ",particle_formation);

    generate_yaml(filename, particle_list, parameters)
    

def generate_torus_particle_list(number_of_particles, inner_radii, tube_radii, separating_dist, particle_material="FusedSilica"):
    # generates dicts of torus sections with keys: material, shape, args, coords, altcolour.
    # This is used to generate the yaml, and to count the number of dipoles that would be generated at an early stage in the calculation.
    particle_list = []

    # Writing specific parameters for particle formation
    torus_gap_theta    = separating_dist/inner_radii    # Full angle occupied by gap between torus sectors
    torus_sector_theta = (2.0*np.pi -number_of_particles*torus_gap_theta) / (number_of_particles) #Full angle occupied by torus sector
    for particle_index in range(number_of_particles):
        lower_phi = ( particle_index*(torus_sector_theta +torus_gap_theta) -torus_sector_theta/2.0 )
        upper_phi = ( particle_index*(torus_sector_theta +torus_gap_theta) +torus_sector_theta/2.0 )
        particle_position = [
            inner_radii*np.cos( particle_index*(torus_sector_theta +torus_gap_theta) ), 
            inner_radii*np.sin( particle_index*(torus_sector_theta +torus_gap_theta) ), 
            1.0e-6
        ]
        position_offsets  = [
            0.0,#random.random()*0.02*characteristic_distance, 
            0.0,#random.random()*0.02*characteristic_distance, 
            0.0#random.random()*0.02*characteristic_distance
        ]

        coords = np.array(particle_position) + np.array(position_offsets)
        particle_list.append({"material": particle_material, "shape": "torus", "args": [inner_radii, tube_radii, lower_phi, upper_phi], "coords": coords, "altcolour": True})
    return particle_list


def generate_torus_yaml(number_of_particles, inner_radii, tube_radii, separating_dist, particle_material="FusedSilica", parameters={}):
    #
    # Generates a YAML file for a set of identical torus sectors with given parameters
    # This will overwrite files with the same name
    #
    # number_of_particles = Number of torus sectors to generate
    # inner_radii = radial distance from origin to center of torus tube
    # tube_radii = radius of torus tube
    # separting_dist = arc length between each torus
    #
    
    # Create / overwrite YAML file
    # Writing core system parameters

    filename = "SingleLaguerre"
    particle_list = generate_torus_particle_list(number_of_particles, inner_radii, tube_radii, separating_dist, particle_material)
    generate_yaml(filename, particle_list, parameters)


def generate_torus_fixedPhi_yaml(number_of_particles, inner_radii, tube_radii, fixedPhi, particle_material="FusedSilica", parameters={}):
    #
    # Generates a YAML file for a set of identical torus sectors with given parameters
    # This will overwrite files with the same name
    #
    # number_of_particles = Number of torus sectors to generate
    # inner_radii = radial distance from origin to center of torus tube
    # tube_radii = radius of torus tube
    # separting_dist = arc length between each torus
    #
    
    # Create / overwrite YAML file
    # Writing core system parameters

    filename = "SingleLaguerre"
    particle_list = []

    # Writing specific parameters for particle formation
    centre_phi = 2.0*np.pi/number_of_particles
    for particle_index in range(number_of_particles):
        lower_phi = particle_index*(centre_phi) -fixedPhi/2.0
        upper_phi = particle_index*(centre_phi) +fixedPhi/2.0
        particle_position = [
            inner_radii*np.cos( particle_index*(centre_phi) ), 
            inner_radii*np.sin( particle_index*(centre_phi) ), 
            1.0e-6
        ]
        position_offsets  = [
            0.0,#random.random()*0.02*characteristic_distance, 
            0.0,#random.random()*0.02*characteristic_distance, 
            0.0#random.random()*0.02*characteristic_distance
        ]
        coords = np.array(particle_position) + np.array(position_offsets)
        particle_list.append({"material": particle_material, "shape": "torus", "args": [inner_radii, tube_radii, lower_phi, upper_phi], "coords": coords, "altcolour": True})

    generate_yaml(filename, particle_list, parameters)


def generate_sphere_slider_yaml(particle_formation, number_of_particles, slider_theta, particle_material="FusedSilica", characteristic_distance=1e-6, particle_radii = 200e-9, parameters={}):
    #
    # Generates a YAML file for a set of identical spheres with given parameters, PLUS an additional sphere that has initial positions between
    # two angles, tested over N steps between this angle range
    # This will overwrite files with the same name
    #
    # number_of_particles = Number of spheres to generate
    # characteristic_distance = key distance for each formation, e.g. radi of circle particles are placed on, length of edge for cubic formation, etc
    # slider_theta = angle that the slider particle sits at initially
    #
    
    # Create / overwrite YAML file
    # Writing core system parameters
    filename = "SingleLaguerre"
    particle_list = []

    # Writing specific parameters for particle formation
    for particle_index in range(number_of_particles):
        match(particle_formation):
            case "circle":
                theta_jump = (2.0*np.pi)/number_of_particles
                particle_theta = theta_jump*particle_index
                particle_position = [characteristic_distance*np.cos(particle_theta), characteristic_distance*np.sin(particle_theta), 1.0e-6]   #0.0
                position_offsets  = [
                    0.0,#random.random()*0.02*characteristic_distance, 
                    0.0,#random.random()*0.02*characteristic_distance, 
                    0.0#random.random()*0.02*characteristic_distance
                ]
                coords = np.array(particle_position) + np.array(position_offsets)
                particle_list.append({"material": particle_material, "shape": "sphere", "args": [particle_radii], "coords": coords, "altcolour": True})

            case _:
                print("Particle formation invalid: ",particle_formation);
    
    slider_position = [characteristic_distance*np.cos(slider_theta), characteristic_distance*np.sin(slider_theta), particle_position[2]]
    particle_list.append({"material": particle_material, "shape": "sphere", "args": [particle_radii], "coords": slider_position, "altcolour": True})

    generate_yaml(filename, particle_list, parameters)

def generate_sphereGrid_yaml(particle_radius, particle_spacing, bounding_sphere_radius, isCircular=True, wavelength=1.0e-6, particle_material="FusedSilica", parameters={}):
    
    # Create / overwrite YAML file
    # Writing core system parameters
    filename = "SingleLaguerre"
    particle_list = []

    # Get coords of particles in a grid
    x_base_coords = np.arange(-bounding_sphere_radius, bounding_sphere_radius, 2.0*particle_radius+particle_spacing)
    y_base_coords = np.arange(-bounding_sphere_radius, bounding_sphere_radius, 2.0*particle_radius+particle_spacing)
    for j in range(len(y_base_coords)):
        for i in range(len(x_base_coords)):
            offset = 0.0
            if(j%2 == 0):
                offset = particle_radius +particle_spacing/2.0
            if(isCircular):
                if(pow(x_base_coords[i] +offset, 2) + pow(y_base_coords[j], 2) <= pow(bounding_sphere_radius, 2)):
                    particle_list.append({"material": particle_material, "shape": "sphere", "args": [particle_radius], "coords": np.array([x_base_coords[i] +offset, y_base_coords[j], 0.0]), "altcolour": True})
            else:
                particle_list.append({"material": particle_material, "shape": "sphere", "args": [particle_radius], "coords": np.array([x_base_coords[i] +offset, y_base_coords[j], 0.0]), "altcolour": True})

    generate_yaml(filename, particle_list, parameters, beam_type="BEAMTYPE_BESSEL")   # BEAMTYPE_LAGUERRE_GAUSSIAN

def generate_sphereShell_yaml(particle_radius, num_pts, shell_radius, wavelength=1.0e-6, particle_material="FusedSilica", parameters={}):
    
    # Create / overwrite YAML file
    # Writing core system parameters
    filename = "SingleLaguerre"
    particle_list = []

    # Get coords of particles in a grid
    # CODE with fibonacci sunflower - works better and faster...
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = shell_radius*np.cos(theta)*np.sin(phi), shell_radius*np.sin(theta)*np.sin(phi), shell_radius*np.cos(phi);
    for i in range(num_pts):
        particle_list.append({"material": particle_material, "shape": "sphere", "args": [particle_radius], "coords": np.array([x[i], y[i], z[i]]), "altcolour": True})
    
    generate_yaml(filename, particle_list, parameters, beam_type="BEAMTYPE_BESSEL")

def generate_sphere_arbitrary_yaml(particles, wavelength=1.0e-6, particle_material="FusedSilica", frames_of_animation=1):
    #
    # Generates a YAML file for a set of arbitrary spheres specified
    #
    
    # Create / overwrite YAML file
    # Writing core system parameters
    filename = "SingleLaguerre"
    parameters = {"frames": frames_of_animation, "frame_max": frames_of_animation, "wavelength": wavelength}
    particle_list = []

    # Writing specific parameters for particle formation
    for particle in particles:
        particle_list.append({"material": particle_material, "shape": "sphere", "args": [particle["radius"]], "coords": particle["position"], "altcolour": True})

    generate_yaml(filename, particle_list, parameters)


def record_particle_info(filename, particle_info, record_parameters=["F"]):
    #
    # Store key details about particle from xlsx into a data structure here
    # This information is stored in particle_info (altered by reference)
    #
    # record_parameters = Which data points to save alongside position from file given
    # Data is stored as follows;
    #   [ [scenario1], [scenario2],... ]
    # where [scenarioN] = [x1, y1, z1, Fx1, Fy1, Fz1, ..., xi, yi, zi, Fxi, Fyi, Fzi], for the <i> particles involved in the scenario
    #
    info = []
    data = pd.read_excel(filename+".xlsx")
    particle_number = int(np.floor( ( len(data.iloc[0])-1 )/(3.0*4.0) ))    # NOTE; 3x4 as vectors are 3 long, and expect 4 vector data types (pos, F, F_T, C)
    for i in range(particle_number):
        # For each particle, fetch its (x,y,z,Fx,Fy,Fz)
        info.append( data.iloc[0, 1 +3*(i)] )    #X
        info.append( data.iloc[0, 2 +3*(i)] )    #Y
        info.append( data.iloc[0, 3 +3*(i)] )    #Z
        for r_param in record_parameters:
            match r_param:
                case "F":   # Optical force
                    info.append( data.iloc[0, 1 +3*(i+1*particle_number)] ) #Fx
                    info.append( data.iloc[0, 2 +3*(i+1*particle_number)] ) #Fy
                    info.append( data.iloc[0, 3 +3*(i+1*particle_number)] ) #Fz
                case "FT":  # Total force (e.g. Buckingham, etc included)
                    info.append( data.iloc[0, 1 +3*(i+2*particle_number)] ) #F_Tx
                    info.append( data.iloc[0, 2 +3*(i+2*particle_number)] ) #F_Ty
                    info.append( data.iloc[0, 3 +3*(i+2*particle_number)] ) #F_Tz
                case "C":   # Torque
                    info.append( data.iloc[0, 1 +3*(i+3*particle_number)] ) #Cx
                    info.append( data.iloc[0, 2 +3*(i+3*particle_number)] ) #Cy
                    info.append( data.iloc[0, 3 +3*(i+3*particle_number)] ) #Cz
                case _:
                    print("Record parameter not recognised while recording data; ",r_param)
    particle_info.append(info)

def store_combined_particle_info(filename, particle_info, record_parameters=["F"]):
    #
    # Moves particle info stored in python into an xlsx file
    #
    # In the xlsx, this is stored as follows;
    #   x1, y1, z1, Fx1, Fy1, Fz1, ...
    #   ... ... ...  ...  ...  ... ... <- Parameter values for scenario N
    #
    print("-- Writing To Combined File --")
    workbook = xlsxwriter.Workbook(filename+"_combined_data.xlsx")
    worksheet = workbook.add_worksheet()

    # Label the 1st particle section
    worksheet.write(0,0, "x0")
    worksheet.write(0,1, "y0")
    worksheet.write(0,2, "z0")
    for r_param_index in range(len(record_parameters)):
        match record_parameters[r_param_index]:
            case "F":   # Optical force
                worksheet.write(0,3*(r_param_index+1) +0, "Fx0")
                worksheet.write(0,3*(r_param_index+1) +1, "Fy0")
                worksheet.write(0,3*(r_param_index+1) +2, "Fz0")
            case "FT":  # Total force (e.g. Buckingham, etc included)
                worksheet.write(0,3*(r_param_index+1) +0, "F_Tx0")
                worksheet.write(0,3*(r_param_index+1) +1, "F_Ty0")
                worksheet.write(0,3*(r_param_index+1) +2, "F_Tz0")
            case "C":   # Torque
                worksheet.write(0,3*(r_param_index+1) +0, "Cx0")
                worksheet.write(0,3*(r_param_index+1) +1, "Cy0")
                worksheet.write(0,3*(r_param_index+1) +2, "Cz0")
            case _:
                print("Record parameter not recognised while recording data; ",record_parameters[r_param_index])
    worksheet.write(0,3*(len(record_parameters)+1), "...")

    # Fill in data stored from particle_info
    for j in range( len(particle_info) ):
        for i in range( len(particle_info[j]) ):
            worksheet.write(j+1, i, particle_info[j][i])

    workbook.close()

def get_closest_particle(point, output_data):
    #
    # Looks through all particles in a given frame (from some output data), and finds the 
    # index of the particles closest to the point given
    #
    # expects point as numpy array
    # output_data in the format of (x0,y0,z0, ...) for each particle, and no other readings
    #
    low_dist  = 1.0e10  # Distance of closest particle to point (easily beatable placeholder when initialised)
    low_index = 0       # Index of the particle that is closest to the point (placeholder of 0 when initialised)
    for i in range(0,len(output_data[0]),3):
        # NOTE; Plus epsilon included to prevent tie-breaker cases when convieniently picking halfway points
        pos  = np.array([ output_data[0,i+0]+sys.float_info.epsilon, output_data[0,i+1]+sys.float_info.epsilon, output_data[0,i+2]+sys.float_info.epsilon ])
        dist = np.linalg.norm( point-pos )
        if(dist <= low_dist):
            low_dist  = dist
            low_index = int(np.floor(i/3.0))
    return low_index

def select_particle_indices(filename, particle_selection, parameters_stored, read_frames=[0]):
    #
    # Calculates the particle indices using different modes.
    # Will be passed particle_selections = "all", [i,j,k,...] or [[rx,ry,rz],...]
    #
    particle_list = None
    if particle_selection == "all":
        # Sum force over all particles
        number_of_particles = get_number_of_particles_XLSX(filename, parameters_stored)
        particle_list = np.arange(0, number_of_particles, 1)

    # Check if particle selection is a list
    elif isinstance(particle_selection, list): # want lists so the force on several indices can be summed.
        if isinstance(particle_selection[0], (int, np.integer)):
            particle_list = particle_selection



        # Check if element of particle selection are a list
        elif isinstance(particle_selection[0], list):
            if isinstance(particle_selection[0][0], (float, np.floating)):
                # Sum force on particle closest to object_offset (the centre)
                number_of_particles = get_number_of_particles_XLSX(filename, parameters_stored)
                read_parameters_point = [{"type":"X", "particle":p, "subtype":s} for s, p in it.product(range(3), range(number_of_particles))]
                particle_list = []
                for pos in particle_selection:
                    point_particle_number = get_closest_particle(     # Get particle i nearest to pos.
                        np.array(pos),
                        output_data = pull_file_data(
                            filename, 
                            parameters_stored, 
                            read_frames, 
                            read_parameters_point, 
                            invert_output=False
                        )
                    )
                    particle_list.append(point_particle_number)

    if particle_list is None:
        print(f"ERROR, unexpected type, {type(particle_selection)} for {particle_selection}")
        sys.exit()

    return particle_list

def get_number_of_particles_XLSX(filename, parameters_stored):
    #
    # Gets the number of particles directly from the YAML given
    # Used in situations where exact number of particles from a system is hard to calculate
    #
    def get_parameters_stored_length(parameters_stored):
        parameter_total = 0
        for param in parameters_stored:
            parameter_total += len(param["args"])
        return parameter_total
    
    data = pd.read_excel(filename+".xlsx")
    data_length = data.count(axis='columns')
    parameters_per_particle = get_parameters_stored_length(parameters_stored)
    number_of_particles = int((data_length[0]-1)/parameters_per_particle)
    return number_of_particles

def pull_file_data(filename, parameters_stored, read_frames, read_parameters, invert_output=False):
    ####
    #### NOW JUST USES ORIGINAL XLSX MADE
    ####
    #### CORRECT DESCRIPTION GIVEN FOR THIS
    ####

    #
    # Pulls specified data from a .xlsx file, returns a list of all data requested
    #
    # Output is given as EITHER;
    #       #NON-INVERTED#
    #       [ [A_p, B_p, C_p], [A_q, B_q, C_q] ]
    #       If requesting read_frames=[p, q] and read_parameters = [A, B, C]
    #       e.g. stored as lists of data from each frame, for each parameter in that frame (in the order provided)
    #                   OR
    #       #INVERTED#
    #       [ [A_p, A_q], [B_p, B_q], [C_p, C_q] ]
    #       If requesting read_frames=[p, q] and read_parameters = [A, B, C]
    #       e.g. stored as lists of data from each frame, for each parameter in that frame (in the order provided)
    #       NOTE; Each parameter A,B,C,etc is a scalar value, e.g. If you want to extract a vector, will 
    #             need to request parameters "Px", "Py", "Pz" individually
    #
    # Works for data formatted as;
    #       Xn Yn Zn P11n P12n P13n ... PK1n PK2n PK3n ... Xm Ym Zm P11m P12m P13m ... PK1m PK2m PK3m
    #       ------------------------------------------     ------------------------------------------
    #       A particle with (x,y,z) pos and 'K'             'm-n' particles in total, all storing 
    #       parameters which can have arbitrary             the same format of parameters
    #       length (e.g. length 3 for vectors)
    #
    # NOTE; This is the format of the "_combined.xlsx" files generated in SimuationVary() tests
    #       This format assumes frame_i is located on row_i
    #
    # filename = file to pull data from WITHOUT the extension type e.g. "<name>" NOT "<name>.xlsx"
    # parameters_stored = [ [x,y,z], [Fx,Fy,Fz], ... ]
    #       Blocks of particles being stored
    # read_frames = List of which rows to read from
    # read_parameters = List of which columns to read from
    #           Format as;
    #               [{param_1}, {param_2}, ...]
    #               For {param_i} = {"type", "particle", "subtype"}
    #                   "type"     = Which general quantity is it, e.g. "X" for pos, "F" for force, "FT" for total force, "C" for torque, ...
    #                   "particle" = Which particle number do you want to read it for e.g. 0 for 0th particle
    #                   "subtype"  = Which component of the type do you want e.g. if "type"="F", then "subtype"=0 => want to read "Fx" for the given particle
    #       Using both read_frames and read_parameters can specify and and subset of data to retrieve (in complex cases, with some 
    #       extra data stored as well)
    # invert_ouput = What format to store found data in; [frames X params] OR [params X frames]
    #
    
    def get_block_index(parameter, parameters_stored, number_of_particles):
        #
        # Sum previous sections visited, to get an index offset to bring you to the 'block' of 
        # values relevent to this parameter, + an extra offset for which particle this is for, 
        # + another offset for which of these parameters in the block you want
        #
        block_index = 0
        for i in range(len(parameters_stored)):     # parameters_stored is ordered correctly
            number_of_block_parameters = len(parameters_stored[i]["args"])
            if(parameters_stored[i]["type"] == parameter["type"]):
                block_index += param["particle"]*number_of_block_parameters +param["subtype"] +1 # +1 to avoid time column
                break
            else:
                block_index += number_of_block_parameters*number_of_particles
        return block_index
    
    def get_parameters_stored_length(parameters_stored):
        parameter_total = 0
        for param in parameters_stored:
            parameter_total += len(param["args"])
        return parameter_total

    # Setup values
    if(invert_output):
        output = np.zeros( (len(read_parameters), len(read_frames)) )
    else:
        output = np.zeros( (len(read_frames), len(read_parameters)) )
    data = pd.read_excel(filename+".xlsx")
    data_length = data.count(axis='columns')
    parameters_per_particle = get_parameters_stored_length(parameters_stored)
    number_of_particles = int((data_length[0]-1)/parameters_per_particle)

    # Pull data
    for j in range(len(read_parameters)):
        param = read_parameters[j]
        block_index    = get_block_index(param, parameters_stored, number_of_particles)
        for i in range(len(read_frames)):
            frame = read_frames[i]
            data_fragment = 0
            try:
                data_fragment = data.iloc[frame, block_index]
            except:
                print("--Invalid XLSX reading, either an invalid row or column [x,y]=["+str(block_index)+","+str(frame)+"] => Data defaults to 0--")
            if(invert_output):
                output[j,i] = data_fragment
            else:
                output[i,j] = data_fragment

    # Output data
    return output

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

def simulations_singleFrame_optForce_spheresInCircle(particle_numbers, filename, include_additionalForces=False):
    #
    # Performs a DDA calculation for various particles in a circular ring on the Z=0 plane
    #
    # particle_numbers = list of particle numbers to be tested in sphere e.g. [1,2,3,4,8]
    #
    
    particle_info = []
    place_radius = 1.15e-6#152e-6         #1.15e-6
    particle_radii = 200e-9         #200e-9
    parameters = {"frames": 1, "frame_max": 1, "show_output": True}

    record_parameters = ["F"]
    if(include_additionalForces):   # Record total forces instead of just optical forces
        record_parameters = ["FT"]  #

    #For each scenario to be tested
    for i, particle_number in enumerate(particle_numbers):
        print(f"\n{i}/{len(particle_numbers)}: Performing calculation for {particle_number} particles")
        #Generate required YAML, perform calculation, then pull force data
        generate_sphere_yaml("circle", particle_number, particle_material="FusedSilica", characteristic_distance=place_radius, particle_radii=particle_radii, parameters=parameters)     # Writes to SingleLaguerre.yml
        #Run DipolesMulti2024Eigen.py
        run_command = "python DipolesMulti2024Eigen.py "+filename
        run_command = run_command.split(" ")
        print("=== Log ===")
        result = subprocess.run(run_command) #, stdout=subprocess.DEVNULL

        #Pull data from xlsx into a local list in python
        record_particle_info(filename, particle_info, record_parameters=record_parameters)
    #Write combined data to a new xlsx file
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    # parameter_text = "\n".join(
    #     (
    #         "Spheres",
    #         "R_placed   (m)= "+str(place_radius),
    #         "R_particle (m)= "+str(particle_radii)
    #     )
    # )
    parameter_text = ""
    return parameter_text

def simulations_singleFrame_optForce_spheresInCircleSlider(particle_total, slider_range, filename):
    #
    # Performs a DDA calculation for various particles in a circular ring on the Z=0 plane
    #
    # particle_total = Number of particles other than slider involved in simulation -> These will be placed in a circular formation
    # slider_range = [start, stop, steps] -> angles to palce slider at
    #
    
    particle_info = [];
    place_radius = 1.15e-6      #1.15e-6
    particle_radii = 200e-9     #200e-9
    parameters = {"frames": 1, "frame_max": 1}

    #For each scenario to be tested
    for slider_index in range(slider_range[2]):
        slider_theta = slider_range[0] +slider_index*(slider_range[1]-slider_range[0])/slider_range[2]
        print("")
        print(str(slider_index)+"/"+str(slider_range[2]))
        print("Performing calculation for "+str(particle_total)+"+1 particles, slider_theta=",slider_theta)
        #Generate required YAML, perform calculation, then pull force data
        generate_sphere_slider_yaml("circle", particle_total, slider_theta, characteristic_distance=place_radius, particle_radii=particle_radii, parameters=parameters)     # Writes to SingleLaguerre.yml

        #Run DipolesMulti2024Eigen.py
        run_command = "python DipolesMulti2024Eigen.py "+filename
        run_command = run_command.split(" ")
        print("=== Log ===")
        result = subprocess.run(run_command, stdout=subprocess.DEVNULL) #, stdout=subprocess.DEVNULL

        #Pull data from xlsx into a local list in python
        record_particle_info(filename, particle_info)
    #Write combined data to a new xlsx file
    store_combined_particle_info(filename, particle_info)
    parameter_text = "\n".join(
        (
            "Spheres +Slider",
            "R_placed   (m)= "+str(place_radius),
            "R_particle (m)= "+str(particle_radii),
            "Slider range= "+ f"{slider_range[0]:.3f}, {slider_range[1]:.3f}, {slider_range[2]:.3f}"
        )
    )
    return parameter_text

def simulations_singleFrame_optForce_wavelengthTrial(wave_start, wave_jump, beam_radius, target_pos, target_radius, filename, wavelength=None, reducedSet=0):
    #
    # Performs a DDA calculation for various particles in a circular ring on the Z=0 plane
    #
    # reducedSet = whether to consider just the first 2 intersections found, or to consider theb entire set
    # target_args = sphere args, marked as the target particle -> forces measured on this particle
    # wave_range = [start_wave_spacing, stop_wave_spacing, jump_wave_spacing]
    # If wavelength is given, it will plot in terms of fractions of the wavelength, else it will just plot int erms of the spacing
    #
    ####
    ## ************************************************************************************************
    ## NOTE THIS WILL NOT ACCOUNT FOR THE SCATTERING BETWEEN EACH OF THE OTHER PARTICLES
    ## ************************************************************************************************
    ####

    def calculate_XY_intersection_positions(r1, p1, r2, p2):
        #
        # Calculates the positions where two rings collide
        # The [1st] ring is of radius r1, centred at position p1
        # The [2nd] ring is for r2 and p2
        #
        # NOTE; This is only in the XY palne currently -> will NOT account for Z separation, assumes the spheres are at equal Z heights
        #
        intersect_positions = []
        centre_vec = [p2[0]-p1[0], p2[1]-p1[1]]
        centre_dist = np.sqrt( pow(centre_vec[0], 2) +pow(centre_vec[1], 2)  )    #Distance between centres
        # Find 'flat' intersect coords (transformed to (0,0), (0,d) system)
        x_intersect_flat = ( pow(centre_dist,2) -pow(r2,2) +pow(r1,2) ) / (2.0*centre_dist)
        y_intersect_flat_p = +np.sqrt( pow(r1,2) -pow(x_intersect_flat,2) )
        y_intersect_flat_m = -np.sqrt( pow(r1,2) -pow(x_intersect_flat,2) )
        intersect_positions.append( [x_intersect_flat, y_intersect_flat_p])
        intersect_positions.append( [x_intersect_flat, y_intersect_flat_m] )
        # Find 'true' intersect coords (rotated back to real positions)
        theta = math.atan2(centre_vec[1], centre_vec[0])     #Angle of p2 relative to p1 in XY plane
        intersect_positions.append( 
            [
                 intersect_positions[0][0]*np.cos(theta) + intersect_positions[0][1]*np.sin(theta),
                -intersect_positions[0][0]*np.sin(theta) + intersect_positions[0][1]*np.cos(theta)
            ]
        )
        intersect_positions.append( 
            [
                 intersect_positions[1][0]*np.cos(theta) + intersect_positions[1][1]*np.sin(theta),
                -intersect_positions[1][0]*np.sin(theta) + intersect_positions[1][1]*np.cos(theta)
            ]
        )
        # Remove previous flat intersections
        intersect_positions.pop(0)
        intersect_positions.pop(0)
        # Return true intersections
        return intersect_positions

    def calculate_interference_positions(wave_spacing, beam_radius, target_pos):
        #
        # Calculates the positions, relative to some target particle, on the circle defined by the beam radius where 
        # constructive/destructive interference would occur with the target
        #
        # wavelength = wavelength of incident light
        # beam_radius = defines circle where interfereing particles will be allowed to be placed, typically on the beam's 
        #               high intensity ring, so it will scatter the most light
        # target_pos = position of the target particle to feel the interference affect chosen
        #
        # NOTE; Assumes the beam is centered at the origin
        #
        positions = []
        # Select radii to probe
        interfere_radii = float_range(wave_spacing, abs(target_pos[0])+beam_radius, wave_spacing)
        # Calculate intersections at this radii
        for rad in interfere_radii:
            intersect_positions = calculate_XY_intersection_positions(beam_radius, [0.0, 0.0], rad, target_pos[:2])
            for pos in intersect_positions:
                positions.append(pos)
        return positions
    
    def float_range(start, stop, jump):
        set = []
        value = start
        while value < stop:
            set.append(value)
            value += jump
        return set

    particle_info = [];
    wave_range = [wave_start, abs(target_pos[0])+beam_radius, wave_jump]     #NOTE; Assumes the target is offset in X axis, NOT Y or Z

    # Sweep through positions to place particles at, at different separations relative to the target
    for wave_spacing in float_range(wave_range[0], wave_range[1], wave_range[2]):
        particle_positions = calculate_interference_positions(wave_spacing, beam_radius, target_pos)
        match reducedSet:
            case 1:     # First 2 particles
                particle_positions = particle_positions[:2]
            case 2:     # First particle only
                particle_positions = particle_positions[:1]
            # Case 0 => Keep full set
        print("")
        print("Performing calculation for "+str( len(particle_positions) )+"+1 particles, wave_spacing=",wave_spacing)
        
        #Generate particle setup
        particles = []
        particles.append( {"radius": target_radius, "position": target_pos} )                                               # Add target
        for particle_pos in particle_positions:
            particles.append( {"radius": target_radius, "position": [particle_pos[0], particle_pos[1], target_pos[2]]} )    # Add others

        if(wavelength != None):
            generate_sphere_arbitrary_yaml(particles, frames_of_animation=1, wavelength=wavelength)     # Writes to SingleLaguerre.yml
        else:
            generate_sphere_arbitrary_yaml(particles, frames_of_animation=1)

        #Run DipolesMulti2024Eigen.py
        run_command = "python DipolesMulti2024Eigen.py "+filename
        run_command = run_command.split(" ")
        print("=== Log ===")
        result = subprocess.run(run_command, stdout=subprocess.DEVNULL) #, stdout=subprocess.DEVNULL

        #Pull data from xlsx into a local list in python
        record_particle_info(filename, particle_info)

    #Write combined data to a new xlsx file
    store_combined_particle_info(filename, particle_info)
    parameter_text = "\n".join(
        (
            "Spheres Wave Spacing",
            "R_beam   (m)= "+str(beam_radius),
            "R_target (m)= "+str(target_radius)
        )
    )
    return parameter_text
        
        
        
def simulations_singleFrame_optForce_spheresInCircleDipoleSize(particle_total, dipole_size_range, filename):
    #
    # Performs a DDA calculation for particles in a circular ring for various dipole sizes. 
    #
    # dipole_size_range = [size_min, size_max, num]
    #
    
    dipole_sizes = make_array(dipole_size_range)

    particle_info = []
    place_radius = 1.15e-6      #1.15e-6
    particle_radii = 200e-9     #200e-9
    frames_of_animation = 1

    parameters = {"frames": frames_of_animation, "frame_max": frames_of_animation, "show_output": True}

    # For each scenario to be tested
    for i, dipole_size in enumerate(dipole_sizes):
        print(f"\n{i}/{len(dipole_sizes)}: Performing calculation for dipole size {dipole_size}")

        # Change parameters to use each dipole_size, then generate YAML
        parameters["dipole_radius"] = dipole_size
        generate_sphere_yaml("circle", particle_total, particle_material="FusedSilica", characteristic_distance=place_radius, particle_radii = particle_radii, parameters=parameters)

        # Run DipolesMulti2024Eigen.py   
        
        run_command = "python DipolesMulti2024Eigen.py "+filename
        run_command = run_command.split(" ")
        print("=== Log ===")
        result = subprocess.run(run_command, stdout=subprocess.DEVNULL) #, stdout=subprocess.DEVNULL
        
        # Pull data from xlsx into a local list in python
        record_particle_info(filename, particle_info)

    # Write combined data to a new xlsx file
    store_combined_particle_info(filename, particle_info)
    parameter_text = "\n".join(
        (
            "Spheres",
            "R_placed   (m)= "+str(place_radius),
            "R_particle (m)= "+str(particle_radii)
        )
    )
    return parameter_text, dipole_sizes


def simulations_singleFrame_optForce_torusInCircle(particle_numbers, filename):
    #
    # Performs a DDA calculation for various particles in a circular ring on the Z=0 plane
    #
    # particle_numbers = list of particle numbers to be tested in sphere e.g. [1,2,3,4,8]
    #
    
    particle_info = [];
    inner_radii = 1.15e-6
    tube_radii  = 200e-9
    separation  = 0.3e-6
    parameters = {"frames": 1, "frame_max": 1}
    #For each scenario to be tested
    for i, particle_number in enumerate(particle_numbers):
        print(f"\n{i}/{len(particle_numbers)}: Performing calculation for {particle_number} particles")
        #Generate required YAML, perform calculation, then pull force data
        generate_torus_yaml(particle_number, inner_radii, tube_radii, separation, parameters=parameters)     # Writes to <filename>.yml

        #Run DipolesMulti2024Eigen.py
        run_command = "python DipolesMulti2024Eigen.py "+filename
        run_command = run_command.split(" ")
        print("=== Log ===")
        result = subprocess.run(run_command, stdout=subprocess.DEVNULL) #, stdout=subprocess.DEVNULL

        #Pull data from xlsx into a local list in python
        record_particle_info(filename, particle_info)
    #Write combined data to a new xlsx file
    store_combined_particle_info(filename, particle_info)
    parameter_text = "\n".join(
        (
            "Torus Sectors= ",
            "R_inner (m)= "+str(inner_radii),
            "R_tube  (m)= "+str(tube_radii),
            "Arc Sep.(m)= "+str(separation)
        )
    )
    return parameter_text

def simulations_singleFrame_optForce_torusInCircleFixedPhi(particle_numbers, filename):
    #
    # Performs a DDA calculation for various particles in a circular ring on the Z=0 plane
    #
    # particle_numbers = list of particle numbers to be tested in sphere e.g. [1,2,3,4,8]
    #
    
    particle_info = [];
    inner_radii = 1.15e-6
    tube_radii  = 200e-9
    sector_phi  = (2.0*np.pi)/16.0  #Angular torus width
    #For each scenario to be tested
    for i, particle_number in enumerate(particle_numbers):
        print(f"\n{i}/{len(particle_numbers)}: Performing calculation for {particle_number} particles")
        #Generate required YAML, perform calculation, then pull force data
        generate_torus_fixedPhi_yaml(particle_number, inner_radii, tube_radii, sector_phi)     # Writes to <filename>.yml

        #Run DipolesMulti2024Eigen.py
        run_command = "python DipolesMulti2024Eigen.py "+filename
        run_command = run_command.split(" ")
        print("=== Log ===")
        result = subprocess.run(run_command, stdout=subprocess.DEVNULL) #, stdout=subprocess.DEVNULL

        #Pull data from xlsx into a local list in python
        record_particle_info(filename, particle_info)
    #Write combined data to a new xlsx file
    store_combined_particle_info(filename, particle_info)
    parameter_text = "\n".join(
        (
            "Torus Sectors= ",
            "R_inner   (m)= "+str(inner_radii),
            "R_tube    (m)= "+str(tube_radii),
            "Phi Sector(m)= "+str(sector_phi)
        )
    )
    return parameter_text


def simulations_singleFrame_optForce_torusInCircleDipoleSize(particle_total, dipole_size_range, filename, separating_dist=0.1e-6):
    #
    # Performs a DDA calculation for particles in a circular ring for various dipole sizes. 
    #
    # dipole_size_range = [size_min, size_max, num]
    #
    dipole_sizes = make_array(dipole_size_range)
    
    particle_info = []
    inner_radii = 1.15e-6
    tube_radii  = 200e-9
    frames_of_animation = 1

    parameters = {"frames": frames_of_animation, "frame_max": frames_of_animation, "show_output": False}
    torus_gap_theta    = separating_dist/inner_radii    # Full angle occupied by gap between torus sectors
    torus_sector_theta = (2.0*np.pi -particle_total*torus_gap_theta) / (particle_total) #Full angle occupied by torus sector
 
    # For each scenario to be tested
    for i, dipole_size in enumerate(dipole_sizes):
        print(f"\n{i}/{len(dipole_sizes)}: Performing calculation for dipole size {dipole_size}")

        # Change parameters to use each dipole_size, then generate YAML
        parameters["dipole_radius"] = dipole_size
        generate_torus_yaml(particle_total, inner_radii, tube_radii, separating_dist, particle_material="FusedSilica", parameters=parameters)

        # Run DipolesMulti2024Eigen.py
        run_command = "python DipolesMulti2024Eigen.py "+filename
        run_command = run_command.split(" ")
        print("=== Log ===")
        result = subprocess.run(run_command, stdout=subprocess.DEVNULL) #, stdout=subprocess.DEVNULL

        # Pull data from xlsx into a local list in python
        record_particle_info(filename, particle_info)

    # Write combined data to a new xlsx file
    store_combined_particle_info(filename, particle_info)
    parameter_text = "\n".join(
        (
            "Torus Sectors= ",
            "R_inner   (m)= "+str(inner_radii),
            "R_tube    (m)= "+str(tube_radii),
            f"Phi Sector(m)= {torus_sector_theta:.3f}"
        )
    )
    return parameter_text, dipole_sizes


def simulations_singleFrame_optForce_torusInCircleSeparation(particle_total, separation_range, filename, dipole_size):
    #
    # Performs a DDA calculation for particles in a circular ring for various dipole sizes. 
    #
    # separation_range = [sep_min, sep_max, num]
    #
    
    particle_info = []
    inner_radii = 1.15e-6
    tube_radii  = 200e-9
    frames_of_animation = 1

    parameters = {"frames": frames_of_animation, "frame_max": frames_of_animation, "dipole_radius": dipole_size, "show_output": False}
    separations = np.linspace(*separation_range) # unpack list to fill the 3 arguments
 
    # For each scenario to be tested
    for i, separation in enumerate(separations):
        print(f"\n{i}/{len(separations)}: Performing calculation for separation {separation}")

        torus_gap_theta    = separation/inner_radii    # Full angle occupied by gap between torus sectors
        torus_sector_theta = (2.0*np.pi -particle_total*torus_gap_theta) / (particle_total) #Full angle occupied by torus sector
        
        # Generate YAML
        generate_torus_yaml(particle_total, inner_radii, tube_radii, separation, particle_material="FusedSilica", parameters=parameters)

        # Run DipolesMulti2024Eigen.py
        run_command = "python DipolesMulti2024Eigen.py "+filename
        run_command = run_command.split(" ")
        print("=== Log ===")
        result = subprocess.run(run_command, stdout=subprocess.DEVNULL) #, stdout=subprocess.DEVNULL

        # Pull data from xlsx into a local list in python
        record_particle_info(filename, particle_info)

    # Write combined data to a new xlsx file
    store_combined_particle_info(filename, particle_info)
    parameter_text = "\n".join(
        (
            "Torus Sectors= ",
            "R_inner   (m)= "+str(inner_radii),
            "R_tube    (m)= "+str(tube_radii),
            f"Phi Sector(m)= {torus_sector_theta:.3f}"
        )
    )
    return parameter_text, separations

def simulations_singleFrame_optForce_torusInCircle_FixedSep_SectorDipole(particle_numbers, dipoleSize_numbers, separation, filename):
    #
    # Performs a DDA calculation for torus particles in a ring
    # Varies the number of particles with constant separation
    # This is considered for multiple dipole sizes
    #
    # multi_plot_data structured as follows;
    #       [ [data_set], ... ]
    #   where data_set = [ [sweep_1], ..., [sweep_N] ]
    #
    
    inner_radii = 1.15e-6
    tube_radii  = 150e-9
    frames_of_animation = 1

    # For each scenario to be tested
    multi_plot_data = []
    for dipoleSize_num in dipoleSize_numbers:
        particle_info = []
        for particle_num in particle_numbers:
            print("Performing calculation for [particle#="+str(particle_num)+" ,dipoleSize="+str(dipoleSize_num)+"]")

            # Generate particles
            parameters = {"frames": frames_of_animation, "frame_max": frames_of_animation, "dipole_radius": dipoleSize_num, "show_output": False}
            generate_torus_yaml(particle_num, inner_radii, tube_radii, separation, particle_material="FusedSilica", parameters=parameters)

            # Run DipolesMulti2024Eigen.py
            run_command = "python DipolesMulti2024Eigen.py "+filename
            run_command = run_command.split(" ")
            print("=== Log ===")
            result = subprocess.run(run_command, stdout=subprocess.DEVNULL) #, stdout=subprocess.DEVNULL

            # Pull data from xlsx into a local list in python
            record_particle_info(filename, particle_info)
        multi_plot_data.append(particle_info)
    #Plot multi data
    #
    # NOTE; The multi-plotter now uses raw python files, rather than saving this data for simplicity, hence storeing a combined file is not necessary
    #
    #store_combined_particle_info(filename, particle_info)
    parameter_text = "\n".join(
        (
            "Torus Sectors= ",
            "R_inner   (m)= "+str(inner_radii),
            "R_tube    (m)= "+str(tube_radii),
            "Separation(m)= "+str(separation)
        )
    )
    return parameter_text, multi_plot_data

def simulations_singleFrame_connected_sphereGrid(particle_radius, particle_spacing, bounding_sphere_radius, connection_mode, connection_args, filename):
    particle_info = [];
    parameters = {"frames": 10, "frame_max": 10, "show_output": True, "connection_mode":connection_mode, "connection_args":connection_args}

    print("Generating sphereGrid")
    #Generate required YAML, perform calculation, then pull force data
    generate_sphereGrid_yaml(particle_radius, particle_spacing, bounding_sphere_radius, parameters=parameters)     # Writes to SingleLaguerre.yml
    #Run DipolesMulti2024Eigen.py
    run_command = "python DipolesMulti2024Eigen.py "+filename
    run_command = run_command.split(" ")
    print("=== Log ===")
    result = subprocess.run(run_command) #, stdout=subprocess.DEVNULL
    
    return ""

def simulations_singleFrame_connected_sphereShell(particle_radius, particle_spacing, shell_radius, connection_mode, connection_args, filename):
    particle_info = [];
    parameters = {"frames": 50, "frame_max": 50, "show_output": True, "connection_mode":connection_mode, "connection_args":connection_args, "time_step": 0.2e-4}

    print("Generating sphereShell")
    #Generate required YAML, perform calculation, then pull force data
    generate_sphereShell_yaml(particle_radius, particle_spacing, shell_radius, parameters=parameters)     # Writes to SingleLaguerre.yml
    #Run DipolesMulti2024Eigen.py
    run_command = "python DipolesMulti2024Eigen.py "+filename
    run_command = run_command.split(" ")
    print("=== Log ===")
    result = subprocess.run(run_command) #, stdout=subprocess.DEVNULL
    
    return ""


def calc_sphere_volumes(particle_total, dipole_size_range, radii):
    # used by "get_sphere_volumes"
    # this functions returns a list of volumes for each dipole size in the range.
    volumes = []
    dipole_sizes = np.linspace(*dipole_size_range) # unpack list to fill the 3 arguments
    
    for dipole_size in dipole_sizes:
        number_of_dipoles = 0
            
        for particle_i in range(particle_total):
            sphere_radius = radii[particle_i]
            dipole_diameter = 2*dipole_size
            dd2 = dipole_diameter**2
            sr2 = sphere_radius**2
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

        volume = number_of_dipoles * dipole_size**3
        print(volume)
        volumes.append(volume)

    print(volumes)
    return volumes


def calc_SphereOrCube_volumes(dipole_size_range, radius, isSphere):
    # this functions returns a list of volumes for each dipole size in the range. For either a sphere of a cube (radius is half side length).
    # dipole_size_range = [min, max, num]
    volumes = []
    dipole_sizes = make_array(dipole_size_range)
    
    for dipole_size in dipole_sizes:
        number_of_dipoles = 0
        dipole_diameter = 2*dipole_size
        dd2 = dipole_diameter**2
        sr2 = radius**2
        num = int(2*radius/dipole_diameter)
        nums = np.arange(-(num-1)/2,(num+1)/2,1)
        for i in nums:
            i2 = i*i
            for j in nums:
                j2 = j*j
                for k in nums:
                    k2 = k*k
                    rad2 = (i2+j2+k2)*dd2
                    if isSphere:
                        if rad2 < sr2:
                            number_of_dipoles += 1
                    else:
                        number_of_dipoles += 1

        volume = number_of_dipoles * dipole_size**3
        volumes.append(volume)

    return volumes



def make_array(values):
    # values can be an array, or a range [min, max, num] and will return an array
    if len(values) == 3 and isinstance(values[2], int):
        values = np.linspace(*values)
    return values


def calc_torus_volumes(particle_total, dipole_size_range, separating_dist, inner_radii, tube_radii):
    # used by "get_torus_volumes"
    # this functions returns a list of volumes for each dipole size in the range.

    volumes = []
    dipole_sizes = np.linspace(*dipole_size_range) 
    torus_gap_theta    = separating_dist/inner_radii    # Full angle occupied by gap between torus sectors
    torus_sector_theta = (2.0*np.pi -particle_total*torus_gap_theta) / (particle_total) #Full angle occupied by torus sector
    
    # For each dipole size, calculate the number of dipoles across all torus sections.
    for dipole_size in dipole_sizes:
        number_of_dipoles = 0

        for particle_i in range(particle_total):
            dipole_diameter = 2*dipole_size
            dd2 = dipole_diameter**2
            ttr2 = tube_radii**2
            num_xy = int( (tube_radii+inner_radii)//dipole_diameter)     #Number of dipoles wide in each direction (XY, wide directions)
            num_z  = int( inner_radii//dipole_diameter)                  #Number of dipoles tall (shorter)
            phi_lower = ( particle_i*(torus_sector_theta +torus_gap_theta) -torus_sector_theta/2.0 ) %(2.0*np.pi)
            phi_upper = ( particle_i*(torus_sector_theta +torus_gap_theta) +torus_sector_theta/2.0 ) %(2.0*np.pi)

            for i in range(-num_xy,num_xy+1):
                i2 = i*i
                for j in range(-num_xy,num_xy+1):
                    j2 = j*j
                    phi = math.atan2(j,i)%(2.0*np.pi)

                    withinBounds = False
                    if (phi_lower <= phi_upper):
                        withinBounds = ( (phi_lower < phi) and (phi < phi_upper) )
                    else:
                        withinBounds = ( (phi_lower < phi) or (phi < phi_upper) )
                    
                    if (withinBounds):
                        for k in range(-num_z,num_z+1):
                            k2 = k*k
                            rad_xy_2 = (i2 + j2)*dd2
                            if (inner_radii -np.sqrt(rad_xy_2))**2 +k2*dd2 < ttr2:
                                number_of_dipoles += 1

        volume = number_of_dipoles * dipole_size**3
        volumes.append(volume)

    return volumes

def store_volumes(shape, args, dipole_size_range, volumes, dict):
    # used by "get_sphere_volumes" and "get_torus_volumes"
    # Stores all calculated volumes in volume_store.p which is a dict of a dict of a dict of an array.
    args_key = " ".join([str(i) for i in args])
    sizes_key = " ".join([str(i) for i in dipole_size_range])

    # Create nested dictionaries if needed.
    if not shape in dict.keys():
        dict[shape] = {}
    if not args_key in dict[shape].keys():
        dict[shape][args_key] = {}

    dict[shape][args_key][sizes_key] = volumes

    with open("volume_store.p", "wb") as f:
        pickle.dump(dict, f)

def get_sphere_volumes(particle_total, radii, dipole_size_range):
    # Read values from storage or calculate new ones, producing the volumes for specific parameters over a dipole size range.

    # if radii is a float, make it a list
    if isinstance(radii, float):
        radii = np.ones(particle_total) * radii
    elif isinstance(radii, list) and len(radii) == particle_total:
        pass
    else:
        sys.exit(f"get_sphere_volumes: incorrect radii {radii}")

    # Prepare dictionary keys
    shape = "sphere"
    radii_key = " ".join([str(i) for i in radii])
    sizes_key = " ".join([str(i) for i in dipole_size_range])
    print(radii_key)

    try:
        with open("volume_store.p", "rb") as f:
            dict = pickle.load(f)
    except:
        dict = {}

    # Return existing store
    if shape in dict.keys() and radii_key in dict[shape].keys() and sizes_key in dict[shape][radii_key].keys():
        return dict[shape][radii_key][sizes_key]

    # Calculate new set
    else:
        print("\nSpending time to calculate new volumes\n")
        volumes = calc_sphere_volumes(particle_total, dipole_size_range, radii)
        store_volumes(shape, radii, dipole_size_range, volumes, dict)
        return volumes


def get_torus_volumes(particle_total, inner_radii, tube_radii, separation, dipole_size_range):
    # Read values from storage or calculate new ones, producing the volumes for specific parameters over a dipole size range.

    # Prepare dictionary keys
    radssep = [particle_total, inner_radii, tube_radii, separation]
    shape = "torus"
    rrs_key = " ".join([str(i) for i in radssep])
    sizes_key = " ".join([str(i) for i in dipole_size_range])

    try:
        with open("volume_store.p", "rb") as f:
            dict = pickle.load(f)
    except:
        dict = {}

    # Return existing store
    if shape in dict.keys() and rrs_key in dict[shape].keys() and sizes_key in dict[shape][rrs_key].keys():
        return dict[shape][rrs_key][sizes_key]

    # Calculate new set
    else:
        print("\nSpending time to calculate new volumes\n")
        volumes = calc_torus_volumes(particle_total, dipole_size_range, separation, inner_radii, tube_radii)
        store_volumes(shape, radssep, dipole_size_range, volumes, dict)
        return volumes
    

def filter_dipole_sizes(volumes, dipole_size_range, num, target_volume=None, should_remove_non_extrema=False):
    # used to filter the results of "get_sphere_volumes" and "get_torus_volumes"
    # * This is so that the simulated objects have more similar volumes.
    # Finds the dipole sizes with volumes closest to the target volume (defaults to the average volume).
    # Returns these best sizes, volumes, and the maximum error.
    if target_volume == None:
        target_volume = np.average(volumes)

    dipole_sizes =  make_array(dipole_size_range)
    num_sizes = len(dipole_sizes)

    if num > num_sizes:
        sys.exit(f"filter_dipole_sizes: too many points requested, max is {num_sizes}")
    
    # scale = 2
    # if num * scale < num_sizes:
    #     original_num = num
    #     num *= scale
    #     isScaled = True
    # else:
    #     isScaled = False

    # finds the <num> min values, the rest are unsorted and are sliced off.
    indices = np.argpartition(np.abs(np.array(volumes)-target_volume), num)[:num] 
    max_error = abs(volumes[indices[num-1]]-target_volume)/target_volume

    # if isScaled:
    #     # Pick ones with most different dipoles sizes
    #     almost_filtered_dipole_sizes = np.array(dipole_sizes)[indices]
    #     filtered_dipole_sizes = np.array([almost_filtered_dipole_sizes])
    #     for _ in range(original_num-1): # (-1 as init with 1 elem)


    filtered_dipole_sizes = np.array(dipole_sizes)[indices]
    sort_is = np.argsort(filtered_dipole_sizes)
    final_is = list(indices[sort_is])
    print(f"Filtered dipole sizes to {num} values, with max volume error: {max_error:.02%}.")


    # In testing: keep sizes only at a max or min volume, this helps but not significantly.
    if should_remove_non_extrema:
        for i in final_is:
            if i == 0 or i == num_sizes-1:
                pass
            else:
                if not( volumes[i] > volumes[i-1] != volumes[i+1] > volumes[i] ): # not (True if gradients are opposite ie max or min)
                    final_is.remove(i)
                    print(f"Removed {dipole_sizes[i]}")

    return np.array(dipole_sizes)[final_is], np.array(volumes)[final_is], max_error

def simulations_fibre_1D_sphere(filename, chain_length, particle_radius, particle_number, connection_mode, connection_args, option_parameters):
    particle_info = [];
    record_parameters = ["F"]

    # Generate set of particle in chain
    print(f"Performing calculation for {particle_number} particle chain")
    Generate_yaml.make_yaml_fibre_1d_sphere(filename, option_parameters, chain_length, particle_radius, particle_number, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_1D_cylinder(filename, chain_length, particle_length, particle_radius, particle_number, connection_mode, connection_args, option_parameters):
    particle_info = [];
    record_parameters = ["F"]

    # Generate set of particle in chain
    print(f"Performing calculation for {particle_number} particle chain")
    Generate_yaml.make_yaml_fibre_1d_cylinder(filename, option_parameters, chain_length, particle_length, particle_radius, particle_number, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_2D_sphere_hollowShell(filename, E0, option_parameters, object_offset, chain_length, shell_radius, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, include_beads=False):
    particle_info = []
    record_parameters = ["F"]

    # Generate YAML for set of particles and beams
    print(f"Performing calculation for {particle_number_radial*particle_number_angular} particles")
    with open(f"{filename}.yml", "w") as _:     # Used to reset file each time this is run
        pass                                    #
    Generate_yaml.make_yaml_fibre_2d_sphere_hollowshell(filename, E0, option_parameters, object_offset, chain_length, shell_radius, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, beam="GAUSS_CSP", include_beads=include_beads)

    # Run simulation
    DM.main(YAML_name=filename)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_2D_cylinder_hollowShell(filename, chain_length, shell_radius, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, option_parameters):
    particle_info = [];
    record_parameters = ["F"]

    # Generate YAML for set of particles and beams
    print(f"Performing calculation for {particle_number_radial*particle_number_angular} particles")
    Generate_yaml.make_yaml_fibre_2d_cylinder_hollowshell(filename, option_parameters, chain_length, shell_radius, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_2D_sphere_thick_connectUniform(filename, chain_length, shell_radius, shell_number, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args,option_parameters):
    particle_info = [];
    record_parameters = ["F"]

    # Generate YAML for set of particles and beams
    print(f"Performing calculation for {particle_number_radial*particle_number_angular} particles")
    Generate_yaml.make_yaml_fibre_2d_sphere_thick_uni(filename, option_parameters, chain_length, shell_radius, shell_number, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_2D_cylinder_thick_connectUniform(filename, chain_length, shell_radius, shell_number, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args,option_parameters):
    particle_info = [];
    record_parameters = ["F"]

    # Generate YAML for set of particles and beams
    print(f"Performing calculation for {particle_number_radial*particle_number_angular} particles")
    Generate_yaml.make_yaml_fibre_2d_cylinder_thick_uni(filename, option_parameters, chain_length, shell_radius, shell_number, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_2D_sphere_shellLayers(filename, chain_length, shell_radius, shell_number, particle_radius, particle_separation, connection_mode, connection_args,option_parameters):
    particle_info = [];
    record_parameters = ["F"]

    # Generate YAML for set of particles and beams
    print("Performing calculation for N particles")
    Generate_yaml.make_yaml_fibre_2d_sphere_shelllayers(filename, option_parameters, chain_length, shell_radius, shell_number, particle_radius, particle_separation, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_2D_cylinder_shellLayers(filename, chain_length, shell_radius, shell_number, particle_length, particle_radius, particle_separation, connection_mode, connection_args,option_parameters):
    particle_info = [];
    record_parameters = ["F"]

    # Generate YAML for set of particles and beams
    print("Performing calculation for N particles")
    Generate_yaml.make_yaml_fibre_2d_cylinder_shelllayers(filename, option_parameters, chain_length, shell_radius, shell_number, particle_length, particle_radius, particle_separation, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_refine_arch_prism(dimensions, variables_list, separations_list, particle_sizes, dipole_sizes, deflections, object_offsets, particle_shapes, place_regime, prism_type, prism_args, beam_type, option_parameters, force_measure_point=[0.0, 0.0, 0.0], indep_vector_component=2):
    #
    # Consider a prism of given parameters, calculate the path it should be located on when deflected by some amount
    #

    # Specify parameters for data pulling later
    parameters_stored = [
        {"type":"X", "args":["x", "y", "z"]},
        {"type":"F", "args":["Fx", "Fy", "Fz"]},
        {"type":"F_T", "args":["F_Tx", "F_Ty", "F_Tz"]},
        {"type":"C", "args":["Cx", "Cy", "Cz"]}
    ]
    read_frames = [
        0
    ]

    # Begin calculations
    # get list of variables from the dictionary.
    dipole_sizes = variables_list["dipole_sizes"]
    separations_list = variables_list["separations_list"]
    particle_sizes = variables_list["particle_sizes"]
    particle_shapes = variables_list["particle_shapes"]
    object_offsets = variables_list["object_offsets"]
    deflections = variables_list["deflections"]
    # get the independent variable (the one to be plotted against)
    indep_name = variables_list["indep_var"]
    indep_list = np.array(variables_list[indep_name] )

    # Based on the indep var, set what variable are varied over different lines of the graph.
    match indep_name:
        case "dipole_sizes": 
            line_vars = [separations_list, particle_sizes, particle_shapes, object_offsets, deflections]
            indep_axis_list = indep_list
        case "separations_list": 
            line_vars = [dipole_sizes, particle_sizes, particle_shapes, object_offsets, deflections]
            indep_axis_list = indep_list[:,indep_vector_component] # vector var so pick which component to plot against
        case "particle_sizes": 
            line_vars = [dipole_sizes, separations_list, particle_shapes, object_offsets, deflections]
            indep_axis_list = indep_list
        case "particle_shapes": 
            line_vars = [dipole_sizes, separations_list, particle_sizes, object_offsets, deflections]
            indep_axis_list = indep_list
        case "object_offsets": 
            line_vars = [dipole_sizes, separations_list, particle_sizes, particle_shapes, deflections]
            indep_axis_list = indep_list[:,indep_vector_component] # vector var so pick which component to plot against
        case "deflections": 
            line_vars = [dipole_sizes, separations_list, particle_sizes, particle_shapes, object_offsets]
            indep_axis_list = indep_list

    print("Performing refinement calculation for deflected prism")
    data_set = []
    data_set_params = []
    particle_nums_set = []
    dpp_nums_set = []
    num_indep = len(indep_axis_list)
    # forceMag_data = np.array([ indep_axis_list, np.zeros(num_indep) ])  
    # forceX_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
    # forceY_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
    # forceZ_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
    # data_vary_dipoleSize_F       = np.array([ indep_axis_list, np.zeros(num_indep) ])
    # data_vary_dipoleSize_FperDip = np.array([ indep_axis_list, np.zeros(num_indep) ])
    # data_vary_dipoleSize_modF2   = np.array([ indep_axis_list, np.zeros(num_indep) ])   #[ [dipole_sizes], [recorded_data]-> e.g. force magnitude ]
    # particle_nums = np.array([ indep_axis_list, np.zeros(num_indep) ])
    # dpp_nums = np.array([ indep_axis_list, np.zeros(num_indep) ])


    # Iterate though every combination of variables that are varied across the lines of the graph.
    var_set_length = len(indep_list)
    for i in line_vars:
        var_set_length *= len(i)
    
    for index_params, params in enumerate(it.product(*line_vars)):

        forceMag_data = np.array([ indep_axis_list, np.zeros(num_indep) ])  
        forceX_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
        forceY_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
        forceZ_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
        data_vary_dipoleSize_F       = np.array([ indep_axis_list, np.zeros(num_indep) ])
        data_vary_dipoleSize_FperDip = np.array([ indep_axis_list, np.zeros(num_indep) ])
        data_vary_dipoleSize_modF2   = np.array([ indep_axis_list, np.zeros(num_indep) ])   #[ [dipole_sizes], [recorded_data]-> e.g. force magnitude ]
        particle_nums = np.array([ indep_axis_list, np.zeros(num_indep) ])
        dpp_nums = np.array([ indep_axis_list, np.zeros(num_indep) ])

        match indep_name:
            case "dipole_sizes": separations, particle_size, particle_shape, object_offset, deflection = params
            case "separations_list": dipole_size, particle_size, particle_shape, object_offset, deflection = params
            case "particle_sizes": dipole_size, separations, particle_shape, object_offset, deflection = params
            case "particle_shapes": dipole_size, separations, particle_size, object_offset, deflection = params
            case "object_offsets": dipole_size, separations, particle_size, particle_shape, deflection = params
            case "deflections": dipole_size, separations, particle_size, particle_shape, object_offset = params

        # Iterate over independent variable to get the data for each line.
        for i, indep_var in enumerate(indep_list):
            print("");print("Progress;"+str( "="*(int(np.floor(10*(i+ index_params*len(indep_list))/var_set_length))) )+str("-"*(int(np.floor(10*(1.0- (i+ index_params*len(indep_list))/var_set_length)))) )+"; "+str((i+ index_params*len(indep_list)))+"/"+str(var_set_length))
            match indep_name:
                case "dipole_sizes": dipole_size = indep_var
                case "separations_list": separations = indep_var
                case "particle_sizes": particle_size = indep_var
                case "particle_shapes": particle_shape = indep_var
                case "object_offsets": object_offset = indep_var
                case "deflections": deflection = indep_var
        
            # Generate YAML & Run Simulation
            option_parameters["dipole_radius"] = dipole_size
            #Generate_yaml.make_yaml_refine_cuboid(filename, time_step, dimensions, dipole_size, separations, object_offset, particle_size, particle_shape, frames=1, show_output=show_output, beam="LAGUERRE")
            particle_num = Generate_yaml.make_yaml_refine_arch_prism(filename, dimensions, separations, particle_size, deflection, object_offset, particle_shape, place_regime, prism_type, prism_args, option_parameters, beam=beam_type)
            DM.main(YAML_name=filename)

            match particle_shape:
                case "sphere": dpp_num = DM.sphere_size([particle_size], dipole_size)
                case "cube": dpp_num = DM.cube_size([particle_size], dipole_size)
                case _: sys.exit("UNIMPLEMENTED SHAPE")

            #---
            # Get details needed for force calculations
            #
            number_of_particles = get_number_of_particles_XLSX(filename, parameters_stored)

            read_parameters = []
            for p in range(number_of_particles):
                read_parameters.append({"type":"F", "particle":p, "subtype":0})
                read_parameters.append({"type":"F", "particle":p, "subtype":1})
                read_parameters.append({"type":"F", "particle":p, "subtype":2})
            read_parameters_point = []
            for p in range(number_of_particles):
                read_parameters_point.append({"type":"X", "particle":p, "subtype":0})
                read_parameters_point.append({"type":"X", "particle":p, "subtype":1})
                read_parameters_point.append({"type":"X", "particle":p, "subtype":2})
            point_particle_number = get_closest_particle(     # Measure forces on central particle for all systems
                np.array(force_measure_point),
                output_data = pull_file_data(
                    filename, 
                    parameters_stored, 
                    read_frames, 
                    read_parameters_point, 
                    invert_output=False
                )
            )
            #
            # Get details needed for force calculations
            #---


            # Pull data needed from this frame, add it to another list tracking
            output_data = pull_file_data(
                filename, 
                parameters_stored, 
                read_frames, 
                read_parameters, 
                invert_output=False
            )
            #print("read_parameters = ", read_parameters)


            #---
            # Calculate required quantities
            #
            # (0) F_mag, Fx, Fy, Fz on 0th particle
            particle_list = select_particle_indices(filename, "all", parameters_stored, read_frames) # note, "central" currently deprecated
            for p_i in range(len(particle_list)):
                # Calculate required quantities
                recorded_force = np.array([output_data[0, 3*p_i+0], output_data[0, 3*p_i+1], output_data[0, 3*p_i+2]])    # Only pulling at a single frame, => only 1 list inside output, holding each 
                
                # Store quantities
                forceX_data[1][i] += recorded_force[0]
                forceY_data[1][i] += recorded_force[1]
                forceZ_data[1][i] += recorded_force[2]
            forceMag_data[1][i] = np.sqrt(forceX_data[1][i]**2 + forceY_data[1][i]**2 + forceZ_data[1][i]**2) # Magnitude of the net force.
            # recorded_force = np.array([output_data[0, 0], output_data[0, 1], output_data[0, 2]])    # Only pulling at a single frame, => only 1 list inside output, holding each 
            # recorded_force_mag = np.sqrt(np.dot(recorded_force, recorded_force))                    # Calculate dep. var. to be plotted
            # forceMag_data[1][i] = recorded_force_mag
            # forceX_data[1][i] = recorded_force[0]
            # forceY_data[1][i] = recorded_force[1]
            # forceZ_data[1][i] = recorded_force[2]

            # (1)&(2) Get magnitude of force per dipole on central particle
            point_force_vec = np.array(
                [
                    output_data[0, 3*point_particle_number +0],
                    output_data[0, 3*point_particle_number +1],
                    output_data[0, 3*point_particle_number +2]
                ]
            )
            point_force = np.sqrt(np.dot(point_force_vec, point_force_vec))  # Finding |F| for point particle
            data_vary_dipoleSize_F[1][i] = abs(point_force)
            data_vary_dipoleSize_FperDip[1][i] = abs(point_force/dpp_num)

            # (3) Get magnitude^2 of force across full mesh
            total_force_vec = np.zeros(3)
            for p in range(0, len(output_data[0]), 3):    # Go through each particle in sets of 3 (Fx,Fy,Fz measurements)
                #print("     -> output_data[0]      = ", output_data[0])
                #print("     -> output_data[0, p+0] = ", output_data[0, p+0])
                #print("p=",p)
                total_force_vec[0] += output_data[0, p+0]
                total_force_vec[1] += output_data[0, p+1]
                total_force_vec[2] += output_data[0, p+2]
            total_force = np.sqrt(np.dot(total_force_vec, total_force_vec))  # Finding |F|
            data_vary_dipoleSize_modF2[1][i] = total_force
            particle_nums[1][i] = particle_num
            dpp_nums[1][i] = dpp_num
            #
            # Calculate required quantities
            #---

        data_set.append(np.array(forceMag_data))
        data_set.append(np.array(forceX_data))
        data_set.append(np.array(forceY_data))
        data_set.append(np.array(forceZ_data))
        data_set.append(np.array(data_vary_dipoleSize_F))
        data_set.append(np.array(data_vary_dipoleSize_FperDip))
        data_set.append(np.array(data_vary_dipoleSize_modF2))
        data_set_params.append(params)
        particle_nums_set.append(np.array(particle_nums))
        dpp_nums_set.append(np.array(dpp_nums))

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    parameter_text = "" #### LEGACY ####
    return parameter_text, np.array(data_set), data_set_params, np.array(particle_nums_set), np.array(dpp_nums_set)


def simulations_refine(dimension, variables_list, separations_list, particle_sizes, dipole_sizes, object_offsets, force_terms, particle_shapes, place_regime, beam_type, include_dipole_forces=False, polarisability_type="RR", force_measure_point=[0.0, 0.0, 0.0], show_output=True, indep_vector_component=2, isObjectCube=True):
    #
    # Consider an object of given parameters. Default to cube object, but can be sphere
    #
    time_step = 1e-4

    # Specify parameters for data pulling later
    parameters_stored = [
        {"type":"X", "args":["x", "y", "z"]},
        {"type":"F", "args":["Fx", "Fy", "Fz"]},
        {"type":"F_T", "args":["F_Tx", "F_Ty", "F_Tz"]},
        {"type":"C", "args":["Cx", "Cy", "Cz"]}
    ]
    read_frames = [
        0
    ]

    # Begin calculations
    # get list of variables from the dictionary.
    dipole_sizes = variables_list["dipole_sizes"]
    separations_list = variables_list["separations_list"]
    particle_sizes = variables_list["particle_sizes"]
    particle_shapes = variables_list["particle_shapes"]
    object_offsets = variables_list["object_offsets"]
    # get the independent variable (the one to be plotted against)
    indep_name = variables_list["indep_var"]
    indep_list = np.array(variables_list[indep_name])

    # Based on the indep var, set what variable are varied over different lines of the graph.
    match indep_name:
        case "dipole_sizes": 
            line_vars = [separations_list, particle_sizes, particle_shapes, object_offsets]
            indep_axis_list = indep_list
        case "separations_list": 
            line_vars = [dipole_sizes, particle_sizes, particle_shapes, object_offsets]
            indep_axis_list = indep_list[:,indep_vector_component] # vector var so pick which component to plot against
        case "particle_sizes": 
            line_vars = [dipole_sizes, separations_list, particle_shapes, object_offsets]
            indep_axis_list = indep_list
        case "particle_shapes": 
            line_vars = [dipole_sizes, separations_list, particle_sizes, object_offsets]
            indep_axis_list = indep_list
        case "object_offsets": 
            line_vars = [dipole_sizes, separations_list, particle_sizes, particle_shapes]
            indep_axis_list = indep_list[:,indep_vector_component] # vector var so pick which component to plot against

    print("Performing refinement calculation for sphere")
    data_set = []
    data_set_params = []
    particle_nums_set = []
    dpp_nums_set = []
    num_indep = len(indep_axis_list)


    # Iterate though every combination of variables that are varied across the lines of the graph.
    var_set_length = len(indep_list)
    for i in line_vars:
        var_set_length *= len(i)
    for index_params, params in enumerate(it.product(*line_vars)):

        forceMag_data = np.array([ indep_axis_list, np.zeros(num_indep) ])  
        forceX_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
        forceY_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
        forceZ_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
        data_vary_dipoleSize_F       = np.array([ indep_axis_list, np.zeros(num_indep) ])
        data_vary_dipoleSize_FperDip = np.array([ indep_axis_list, np.zeros(num_indep) ])
        data_vary_dipoleSize_modF2   = np.array([ indep_axis_list, np.zeros(num_indep) ])   #[ [dipole_sizes], [recorded_data]-> e.g. force magnitude ]
        particle_nums = np.array([ indep_axis_list, np.zeros(num_indep) ])
        dpp_nums = np.array([ indep_axis_list, np.zeros(num_indep) ])

        match indep_name:
            case "dipole_sizes": separations, particle_size, particle_shape, object_offset = params
            case "separations_list": dipole_size, particle_size, particle_shape, object_offset = params
            case "particle_sizes": dipole_size, separations, particle_shape, object_offset = params
            case "particle_shapes": dipole_size, separations, particle_size, object_offset = params
            case "object_offsets": dipole_size, separations, particle_size, particle_shape = params

        # Iterate over independent variable to get the data for each line.
        for i, indep_var in enumerate(indep_list):
            print("");print("Progress;"+str( "="*(int(np.floor(10*(i+ index_params*len(indep_list))/var_set_length))) )+str("-"*(int(np.floor(10*(1.0- (i+ index_params*len(indep_list))/var_set_length)))) )+"; "+str((i+ index_params*len(indep_list)))+"/"+str(var_set_length))
            match indep_name:
                case "dipole_sizes": dipole_size = indep_var
                case "separations_list": separations = indep_var
                case "particle_sizes": particle_size = indep_var
                case "particle_shapes": particle_shape = indep_var
                case "object_offsets": object_offset = indep_var
        
            # Generate YAML & Run Simulation
            option_parameters["dipole_radius"] = dipole_size
            particle_num = Generate_yaml.make_yaml_refine_sphere(filename, dimension, separations, particle_size, object_offset, particle_shape, place_regime, option_parameters, beam=beam_type, makeCube=isObjectCube)
            DM.main(YAML_name=filename, force_terms=force_terms, include_dipole_forces=include_dipole_forces, polarisability_type=polarisability_type, verbosity=0)


            match particle_shape:
                case "sphere": dpp_num = DM.sphere_size([particle_size], dipole_size)
                case "cube": dpp_num = DM.cube_size([particle_size], dipole_size)
                case _: sys.exit("UNIMPLEMENTED SHAPE")

            #---
            # Get details needed for force calculations
            #
            number_of_particles = get_number_of_particles_XLSX(filename, parameters_stored)

            read_parameters = []
            for p in range(number_of_particles):
                read_parameters.append({"type":"F", "particle":p, "subtype":0})
                read_parameters.append({"type":"F", "particle":p, "subtype":1})
                read_parameters.append({"type":"F", "particle":p, "subtype":2})
            read_parameters_point = []
            for p in range(number_of_particles):
                read_parameters_point.append({"type":"X", "particle":p, "subtype":0})
                read_parameters_point.append({"type":"X", "particle":p, "subtype":1})
                read_parameters_point.append({"type":"X", "particle":p, "subtype":2})
            point_particle_number = get_closest_particle(     # Measure forces on central particle for all systems
                np.array(force_measure_point),
                output_data = pull_file_data(
                    filename, 
                    parameters_stored, 
                    read_frames, 
                    read_parameters_point, 
                    invert_output=False
                )
            )
            #
            # Get details needed for force calculations
            #---


            # Pull data needed from this frame, add it to another list tracking
            output_data = pull_file_data(
                filename, 
                parameters_stored, 
                read_frames, 
                read_parameters, 
                invert_output=False
            )
            #print("read_parameters = ", read_parameters)


            #---
            # Calculate required quantities
            #            
            # (0) F_mag, Fx, Fy, Fz on 0th particle
            particle_list = select_particle_indices(filename, "all", parameters_stored, read_frames) # note, "central" currently deprecated
            for p_i in range(len(particle_list)):
                # Calculate required quantities
                recorded_force = np.array([output_data[0, 3*p_i+0], output_data[0, 3*p_i+1], output_data[0, 3*p_i+2]])    # Only pulling at a single frame, => only 1 list inside output, holding each 
                # Store quantities
                forceX_data[1][i] += recorded_force[0]
                forceY_data[1][i] += recorded_force[1]
                forceZ_data[1][i] += recorded_force[2]
            forceMag_data[1][i] = np.sqrt(forceX_data[1][i]**2 + forceY_data[1][i]**2 + forceZ_data[1][i]**2) # Magnitude of the net force.
            # recorded_force = np.array([output_data[0, 0], output_data[0, 1], output_data[0, 2]])    # Only pulling at a single frame, => only 1 list inside output, holding each 
            # recorded_force_mag = np.sqrt(np.dot(recorded_force, recorded_force))                    # Calculate dep. var. to be plotted
            # forceMag_data[1][i] = recorded_force_mag
            # forceX_data[1][i] = recorded_force[0]
            # forceY_data[1][i] = recorded_force[1]
            # forceZ_data[1][i] = recorded_force[2]

            # (1)&(2) Get magnitude of force per dipole on central particle
            point_force_vec = np.array(
                [
                    output_data[0, 3*point_particle_number +0],
                    output_data[0, 3*point_particle_number +1],
                    output_data[0, 3*point_particle_number +2]
                ]
            )
            point_force = np.sqrt(np.dot(point_force_vec, point_force_vec))  # Finding |F| for point particle
            data_vary_dipoleSize_F[1][i] = abs(point_force)
            data_vary_dipoleSize_FperDip[1][i] = abs(point_force/dpp_num)

            # (3) Get magnitude^2 of force across full mesh
            total_force_vec = np.zeros(3)
            for p in range(0, len(output_data[0]), 3):    # Go through each particle in sets of 3 (Fx,Fy,Fz measurements)
                #print("     -> output_data[0]      = ", output_data[0])
                #print("     -> output_data[0, p+0] = ", output_data[0, p+0])
                #print("p=",p)
                total_force_vec[0] += output_data[0, p+0]
                total_force_vec[1] += output_data[0, p+1]
                total_force_vec[2] += output_data[0, p+2]
            total_force = np.sqrt(np.dot(total_force_vec, total_force_vec))  # Finding |F|
            data_vary_dipoleSize_modF2[1][i] = total_force
            particle_nums[1][i] = particle_num
            dpp_nums[1][i] = dpp_num
            ####
            ## TEMPORARILY MODIFIED THIS TO TEST TANGENTIAL AND RADIAL FORCES ---> WHEN REWORKING THIS MAKE THESE THEIR OWN FUNCTION TO JUST BE CALLED
            ####
            # tangenital_force_total = np.sqrt(forceX_data[1][i]**2 + forceY_data[1][i]**2 + forceZ_data[1][i]**2)
            # radial_force_total = np.sqrt(forceX_data[1][i]**2 + forceY_data[1][i]**2 + forceZ_data[1][i]**2)
            # data_vary_dipoleSize_modF2[1][i] = tangenital_force_total
            # data_vary_dipoleSize_FperDip[1][i] = radial_force_total
            # particle_nums[1][i] = particle_num
            # dpp_nums[1][i] = dpp_num
            #
            # Calculate required quantities
            #---

        data_set.append(np.array(forceMag_data))
        data_set.append(np.array(forceX_data))
        data_set.append(np.array(forceY_data))
        data_set.append(np.array(forceZ_data))
        data_set.append(np.array(data_vary_dipoleSize_F))
        data_set.append(np.array(data_vary_dipoleSize_FperDip))
        data_set.append(np.array(data_vary_dipoleSize_modF2))
        data_set_params.append(params)
        particle_nums_set.append(np.array(particle_nums))
        dpp_nums_set.append(np.array(dpp_nums))

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    parameter_text = "" #### LEGACY ####
    return parameter_text, np.array(data_set), data_set_params, np.array(particle_nums_set), np.array(dpp_nums_set)


# def simulations_refine_general(dimensions, variables_list, force_terms, time_step=1e-4, show_output=False, indep_vector_component=2, particle_selection="all"):
#     #
#     # Consider an object of given parameters, vary its aspects, take force measurements for each scenario
#     # particle_selection can be "all", "central" or [int]
#     #

#     # Specify parameters for data pulling later
#     parameters_stored = [
#         {"type":"X", "args":["x", "y", "z"]},
#         {"type":"F", "args":["Fx", "Fy", "Fz"]},
#         {"type":"F_T", "args":["F_Tx", "F_Ty", "F_Tz"]},
#         {"type":"C", "args":["Cx", "Cy", "Cz"]}
#     ]
#     read_frames = [
#         0
#     ]
    
#     # get list of variables from the dictionary.
#     dipole_sizes = variables_list["dipole_sizes"]
#     separations_list = variables_list["separations_list"]
#     particle_sizes = variables_list["particle_sizes"]
#     particle_shapes = variables_list["particle_shapes"]
#     object_offsets = variables_list["object_offsets"]
#     # get the independent variable (the one to be plotted against)
#     indep_name = variables_list["indep_var"]
#     indep_list = np.array(variables_list[indep_name] )

#     # Based on the indep var, set what variable are varied over different lines of the graph.
#     match indep_name:
#         case "dipole_sizes": 
#             line_vars = [separations_list, particle_sizes, particle_shapes, object_offsets]
#             indep_axis_list = indep_list
#         case "separations_list": 
#             line_vars = [dipole_sizes, particle_sizes, particle_shapes, object_offsets]
#             indep_axis_list = indep_list[:,indep_vector_component] # vector var so pick which component to plot agaisnt
#         case "particle_sizes": 
#             line_vars = [dipole_sizes, separations_list, particle_shapes, object_offsets]
#             indep_axis_list = indep_list
#         case "particle_shapes": 
#             line_vars = [dipole_sizes, separations_list, particle_sizes, object_offsets]
#             indep_axis_list = indep_list
#         case "object_offsets": 
#             line_vars = [dipole_sizes, separations_list, particle_sizes, particle_shapes]
#             indep_axis_list = indep_list[:,indep_vector_component] # vector var so pick which component to plot agaisnt


#     # Begin calculations
#     print("Performing refinement calculation for cuboid")
#     data_set = []
#     data_set_params = []
#     particle_nums_set = []
#     dpp_nums_set = []
#     num_indep = len(indep_axis_list)

#     # Iterate though every combination of variables that are varied across the lines of the graph.
#     for params in it.product(*line_vars):

#         forceMag_data = np.array([ indep_axis_list, np.zeros(num_indep) ])  
#         forceX_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
#         forceY_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
#         forceZ_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
#         particle_nums = np.array([ indep_axis_list, np.zeros(num_indep) ])
#         dpp_nums = np.array([ indep_axis_list, np.zeros(num_indep) ])

#         match indep_name:
#             case "dipole_sizes": separations, particle_size, particle_shape, object_offset = params
#             case "separations_list": dipole_size, particle_size, particle_shape, object_offset = params
#             case "particle_sizes": dipole_size, separations, particle_shape, object_offset = params
#             case "particle_shapes": dipole_size, separations, particle_size, object_offset = params
#             case "object_offsets": dipole_size, separations, particle_size, particle_shape = params

#         # Iterate over independent variable to calculate the forces on every particle.
#         for i, indep_var in enumerate(indep_list):
#             match indep_name:
#                 case "dipole_sizes": dipole_size = indep_var
#                 case "separations_list": separations = indep_var
#                 case "particle_sizes": particle_size = indep_var
#                 case "particle_shapes": particle_shape = indep_var
#                 case "object_offsets": object_offset = indep_var
        
#             # Generate YAML for set of particles and beams
#             particle_num = Generate_yaml.make_yaml_refine_cuboid(filename, time_step, dimensions, dipole_size, separations, object_offset, particle_size, particle_shape, frames=1, show_output=show_output, beam="LAGUERRE")
#             # Run simulation
#             DM.main(YAML_name=filename, force_terms=force_terms)
#             match particle_shape:
#                 case "sphere": dpp_num = DM.sphere_size([particle_size], dipole_size)
#                 case "cube": dpp_num = DM.cube_size([particle_size], dipole_size)
#                 case _: sys.exit("UNIMPLEMENTED SHAPE")

#             # Calculate what particles to sum the forces over.
#             particle_list = select_particle_indices(filename, particle_selection, parameters_stored, read_frames)
#             read_parameters = [{"type":"F", "particle":p, "subtype":st} for p, st in it.product(particle_list, [0,1,2])]

#             # Pull data needed from this frame, add it to another list tracking
#             output_data = pull_file_data(
#                 filename, 
#                 parameters_stored, 
#                 read_frames, 
#                 read_parameters, 
#                 invert_output=False
#             )
#             #print("read_parameters = ",read_parameters)
#             for p_i in range(len(particle_list)):
#                 # print("     -> output_data[0]      = ", output_data[0])
#                 # print("     -> output_data[0, 3*p_i+0] = ", output_data[0, 3*p_i+0])
#                 # print("p=",p_i)
#                 # print("3p=",3*p_i)
#                 # Calculate required quantities
#                 recorded_force = np.array([output_data[0, 3*p_i+0], output_data[0, 3*p_i+1], output_data[0, 3*p_i+2]])    # Only pulling at a single frame, => only 1 list inside output, holding each 
                
#                 # Store quantities
#                 forceX_data[1][i] += recorded_force[0]
#                 forceY_data[1][i] += recorded_force[1]
#                 forceZ_data[1][i] += recorded_force[2]

#             forceMag_data[1][i] = np.sqrt(forceX_data[1][i]**2 + forceY_data[1][i]**2 + forceZ_data[1][i]**2) # Magnitude of the net force.
#             particle_nums[1][i] = particle_num
#             dpp_nums[1][i] = dpp_num

#         data_set.append(np.array(forceMag_data))
#         data_set.append(np.array(forceX_data))
#         data_set.append(np.array(forceY_data))
#         data_set.append(np.array(forceZ_data))
#         data_set_params.append(params)
#         particle_nums_set.append(np.array(particle_nums))
#         dpp_nums_set.append(np.array(dpp_nums))
        

#     # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
#     parameter_text = ""
#     return parameter_text, np.array(data_set), data_set_params, np.array(particle_nums_set), np.array(dpp_nums_set)

def simulations_single_dipole(filename, read_parameters, beam_type, test_type, test_args, object_offset, option_parameters, rotation=None):
    #
    # Test the forces experience by single dipole systems in different setups
    #

    # Fixed values initialised
    invalidArgs=False
    parameters_stored = [
        {"type":"X", "args":["x", "y", "z"]},
        {"type":"F", "args":["Fx", "Fy", "Fz"]},
        {"type":"F_T", "args":["F_Tx", "F_Ty", "F_Tz"]},
        {"type":"C", "args":["Cx", "Cy", "Cz"]}
    ]
    read_frames = [
        0
    ]

    data_set        = []
    data_set_labels = []
    graphlabel_set  = {"title":"", "xAxis":"", "yAxis":""}
    match test_type:
        case "single":
            if(len(test_args)==3):
                graphlabel_set  = {"title":"Single Dipole", "xAxis":"X Offset(m)", "yAxis":"Force(N)"}
                data_set_Fx = [[], []]
                data_set_Fy = [[], []]
                data_set_Fz = [[], []]

                offset_lower, offset_upper, offset_number = test_args

                object_offset_set = np.linspace(offset_lower, offset_upper, offset_number)
                for offset in object_offset_set:
                    # Setup and run simulation
                    Generate_yaml.make_yaml_single_dipole_exp(filename, test_type=test_type, test_args=test_args, object_offset=[offset, 0.0, 0.0], option_parameters=option_parameters, rotation=rotation, beam=beam_type)
                    DM.main(YAML_name=filename)

                    # Pull forces found
                    output_data = pull_file_data(
                        filename, 
                        parameters_stored, 
                        read_frames, 
                        read_parameters, 
                        invert_output=False
                    )[0]    # NOTE; [0] To immediately get the 0th frame from results

                    # Populate data set to visualise
                    data_set_Fx[0].append(offset) # X-axis
                    data_set_Fx[1].append(output_data[0]) # Y-axis

                    data_set_Fy[0].append(offset) # X-axis
                    data_set_Fy[1].append(output_data[1]) # Y-axis

                    data_set_Fz[0].append(offset) # X-axis
                    data_set_Fz[1].append(output_data[2]) # Y-axis

                data_set.append(data_set_Fx)
                data_set_labels.append("Fx")

                data_set.append(data_set_Fy)
                data_set_labels.append("Fy")
                
                data_set.append(data_set_Fz)
                data_set_labels.append("Fz")
            else:invalidArgs=True
        
        case "7shell":
            if(len(test_args)==3):
                graphlabel_set  = {"title":"7 Single Dipoles", "xAxis":"X Offset(m)", "yAxis":"Force(N)"}
                data_set_Fx = [[], []]
                data_set_Fy = [[], []]
                data_set_Fz = [[], []]

                offset_lower, offset_upper, offset_number = test_args

                object_offset_set = np.linspace(offset_lower, offset_upper, offset_number)
                for offset in object_offset_set:
                    # Setup and run simulation
                    Generate_yaml.make_yaml_single_dipole_exp(filename, test_type=test_type, test_args=test_args, object_offset=[offset, 0.0, 0.0], option_parameters=option_parameters, rotation=rotation, beam=beam_type)
                    DM.main(YAML_name=filename)

                    # Pull forces found
                    output_data = pull_file_data(
                        filename, 
                        parameters_stored, 
                        read_frames, 
                        read_parameters, 
                        invert_output=False
                    )[0]    # NOTE; [0] To immediately get the 0th frame from results

                    # Populate data set to visualise
                    data_set_Fx[0].append(offset) # X-axis
                    data_set_Fx[1].append(output_data[0]) # Y-axis

                    data_set_Fy[0].append(offset) # X-axis
                    data_set_Fy[1].append(output_data[1]) # Y-axis

                    data_set_Fz[0].append(offset) # X-axis
                    data_set_Fz[1].append(output_data[2]) # Y-axis

                data_set.append(data_set_Fx)
                data_set_labels.append("Fx")

                data_set.append(data_set_Fy)
                data_set_labels.append("Fy")
                
                data_set.append(data_set_Fz)
                data_set_labels.append("Fz")
            else:invalidArgs=True

        case "7shell_difference":
            if(len(test_args)==4):
                offset_lower, offset_upper, offset_number, polarisabilities = test_args

                graphlabel_set  = {"title":"Single Dipole Difference"+str(polarisabilities), "xAxis":"X Offset(m)", "yAxis":"Force(N)"}

                object_offset_set = np.linspace(offset_lower, offset_upper, offset_number)
                print("LIST IS ", polarisabilities)
                for polarisability in polarisabilities:
                    option_parameters["polarisability_type"] = polarisability
                    data_set_Fx = [[], []]
                    data_set_Fy = [[], []]
                    data_set_Fz = [[], []]
                    for offset in object_offset_set:
                        # Setup and run simulation
                        Generate_yaml.make_yaml_single_dipole_exp(filename, test_type=test_type, test_args=test_args, object_offset=[offset, 0.0, 0.0], option_parameters=option_parameters, rotation=rotation, beam=beam_type)
                        DM.main(YAML_name=filename)

                        # Pull forces found
                        output_data = pull_file_data(
                            filename, 
                            parameters_stored, 
                            read_frames, 
                            read_parameters, 
                            invert_output=False
                        )[0]    # NOTE; [0] To immediately get the 0th frame from results

                        # Populate data set to visualise
                        data_set_Fx[0].append(offset) # X-axis
                        data_set_Fx[1].append(output_data[0]) # Y-axis
                        data_set_Fy[0].append(offset) # X-axis
                        data_set_Fy[1].append(output_data[1]) # Y-axis
                        data_set_Fz[0].append(offset) # X-axis
                        data_set_Fz[1].append(output_data[2]) # Y-axis

                    data_set.append(data_set_Fx)
                    data_set_labels.append("Fx-"+polarisability)
                    data_set.append(data_set_Fy)
                    data_set_labels.append("Fy="+polarisability)
                    data_set.append(data_set_Fz)
                    data_set_labels.append("Fz-"+polarisability)
                # Calculate and add difference plot
                data_set_FxDiff=[[],[]]
                data_set_FyDiff=[[],[]]
                data_set_FzDiff=[[],[]]
                for offset_ind in range(len(object_offset_set)):
                    data_set_FxDiff[0].append(object_offset_set[offset_ind]) # X-axis
                    data_set_FxDiff[1].append(data_set[0][1][offset_ind] -data_set[3+0][1][offset_ind]) # Y-axis
                    data_set_FyDiff[0].append(object_offset_set[offset_ind]) # X-axis
                    data_set_FyDiff[1].append(data_set[1][1][offset_ind] -data_set[3+1][1][offset_ind]) # Y-axis
                    data_set_FzDiff[0].append(object_offset_set[offset_ind]) # X-axis
                    data_set_FzDiff[1].append(data_set[2][1][offset_ind] -data_set[3+2][1][offset_ind]) # Y-axis
                data_set.append(data_set_FxDiff)
                data_set_labels.append("Fx-Difference")
                data_set.append(data_set_FyDiff)
                data_set_labels.append("Fy-Difference")
                data_set.append(data_set_FzDiff)
                data_set_labels.append("Fz-Difference")
            else:invalidArgs=True

        case "multi_separated":
            if(len(test_args)==4):
                particle_number, lower_separation, upper_separation, separation_number = test_args

                graphlabel_set  = {"title":str(particle_number)+" dipoles separated in X axis", "xAxis":"X Dipole Center Separation(m)", "yAxis":"Force(N)"}
                data_set_Fx = [[], []]
                data_set_Fy = [[], []]
                data_set_Fz = [[], []]
                
                separations = np.linspace(lower_separation, upper_separation, separation_number)
                for separation in separations:
                    # Setup and run simulation
                    Generate_yaml.make_yaml_single_dipole_exp(filename, test_type=test_type, test_args=test_args, object_offset=object_offset, option_parameters=option_parameters, rotation=rotation, beam=beam_type, extra_args=[separation])
                    DM.main(YAML_name=filename)

                    # Pull forces found
                    output_data = pull_file_data(
                        filename, 
                        parameters_stored, 
                        read_frames, 
                        read_parameters, 
                        invert_output=False
                    )[0]    # NOTE; [0] To immediately get the 0th frame from results

                    # Populate data set to visualise
                    data_set_Fx[0].append(separation) # X-axis
                    data_set_Fx[1].append(output_data[0]) # Y-axis
                    data_set_Fy[0].append(separation) # X-axis
                    data_set_Fy[1].append(output_data[1]) # Y-axis
                    data_set_Fz[0].append(separation) # X-axis
                    data_set_Fz[1].append(output_data[2]) # Y-axis

                data_set.append(data_set_Fx)
                data_set_labels.append("Fx")
                data_set.append(data_set_Fy)
                data_set_labels.append("Fy")
                data_set.append(data_set_Fz)
                data_set_labels.append("Fz")
            else:invalidArgs=True

        case _:
            print("Invalid test_type: ",test_type)
    if(invalidArgs):
        print("Invalid test_args: "+str(test_type)+", "+str(test_args))

    # Format and return data to be plotted
    return np.array(data_set), data_set_labels, graphlabel_set


def simulation_single_cubeSphere(filename, dimensions, object_shape, separations, object_offset, particle_size, particle_shape, option_parameters, beam="LAGUERRE"):
    # Get the forces, particle number, and dipoles per particle for a SINGLE set of parameters for either a cuboid or sphere object.
    parameters_stored = [
        {"type":"X", "args":["x", "y", "z"]},
        {"type":"F", "args":["Fx", "Fy", "Fz"]},
        {"type":"F_T", "args":["F_Tx", "F_Ty", "F_Tz"]},
        {"type":"C", "args":["Cx", "Cy", "Cz"]}
    ]
    read_frames = [
        0
    ]
    
    match object_shape:
        case "sphere": particle_num = Generate_yaml.make_yaml_refine_sphere(filename, dimensions[0], separations, particle_size, object_offset, particle_shape, place_regime="squish", option_parameters=option_parameters, beam=beam)
        case "cube": particle_num = Generate_yaml.make_yaml_refine_cuboid(filename, dimensions, separations, object_offset, particle_size, particle_shape, option_parameters, beam=beam)
        case _: sys.exit("UNIMPLEMENTED SHAPE object_shape")
    
    # Run simulation
    DM.main(YAML_name=filename)
    dipole_size = option_parameters["dipole_radius"]
    match particle_shape:
        case "sphere": dpp_num = DM.sphere_size([particle_size], dipole_size) # get dipoles per particle.
        case "cube": dpp_num = DM.cube_size([particle_size], dipole_size)
        case _: sys.exit("UNIMPLEMENTED SHAPE particle_shape")

    read_parameters = [{"type":{t}, "particle":p, "subtype":st} for t, p, st in it.product(["X","F"], range(particle_num), [0,1,2])]

    # Pull data needed from this frame, add it to another list tracking
    output_data = pull_file_data(
        filename, 
        parameters_stored, 
        read_frames, 
        read_parameters, 
        invert_output=False
    )
    positions = np.zeros((particle_num,3))
    forces = np.zeros((particle_num,3))
    for p_i in range(particle_num):
        positions[p_i] = [output_data[0, 3*p_i+0], output_data[0, 3*p_i+1], output_data[0, 3*p_i+2]] # 0th frame data.
        forces[p_i] = [output_data[0, 3*p_i+3], output_data[0, 3*p_i+4], output_data[0, 3*p_i+5]] # 0th frame data.
    
    return positions, forces, particle_num, dpp_num

# def simulations_spheredisc_model(filename, variables_list, dda_forces_returned, beam_type, forces_output, particle_selections, include_dipole_forces=False, polarisability_type="RR", mode="disc", torque_centre=[0.0, 0.0, 0.0], indep_vector_component=2, time_step=1e-4, frames=1, show_output=True):    
#     #
#     # Consider an object of given parameters. Default to cube object, but can be sphere
#     # variables_list contains all of the parameters to be changed (dict).
#     # dda_forces_returned are the forces returned from the DDA - usually ["optical"]. object_shape is either "sphere" or "cube"
#     # beam_type is str for the beam, usually "LAGUERRE"
#     # forces_output is the force types returned, subset of ["Fmag", "Fx", "Fy", "Fz", "Cmag", "Cx", "Cy", "Cz"]
#     # particle_selections is a list of string keywords ("all"), ints (particle indices), or float vectors ([0,0,0], particle closest to this point)
#     #
#     # Return data_set, data_set_params, particle_nums_set, dpp_nums_set
#     # data set is [ [[indep var list], [force list]] for F, p in zip(forces_output, particle_selections) for each set of line parameters]
#     # other returns give different data for the same pattern.
#     #

#     if len(forces_output) != len(particle_selections): sys.exit(f"ERROR, forces_output and particle_selections must be the same length, not {len(forces_output)} and {len(particle_selections)}.")
    
#     # Specify all parameters in the file so a subset can be pulled later based on read_parameters
#     parameters_stored = [
#         {"type":"X", "args":["x", "y", "z"]},
#         {"type":"F", "args":["Fx", "Fy", "Fz"]},
#         {"type":"F_T", "args":["F_Tx", "F_Ty", "F_Tz"]}, # (total force including hydrodynamics, not used here)
#         {"type":"C", "args":["Cx", "Cy", "Cz"]}
#     ]
#     parameters_stored_torque = [
#         {"type":"X", "args":["x", "y", "z"]},
#         {"type":"F", "args":["Fx", "Fy", "Fz"]},
#     ]
#     read_frames = [0]
#     # Record what parameters each force output would require
#     read_parameters_lookup = {
#         "Fmag": [["F",0], ["F",1], ["F",2]],
#         "Fx":   [["F",0]],
#         "Fy":   [["F",1]],
#         "Fz":   [["F",2]],
#         "Cmag": [["C",0], ["C",1], ["C",2]],
#         "Cx":   [["C",0]],
#         "Cy":   [["C",1]],
#         "Cz":   [["C",2]],
#         "Tmag": [["X",0], ["X",1], ["X",2], ["F",0], ["F",1], ["F",2]], # normal order
#         "Tx":   [["X",1], ["X",2], ["F",1], ["F",2]], # y,z,Fy,Fz
#         "Ty":   [["X",2], ["X",0], ["F",2], ["F",0]], # z,x,Fz,Fx
#         "Tz":   [["X",0], ["X",1], ["F",0], ["F",1]], # x,y,Fx,Fy
#     }
#     # Pull variables from list supplied
#     dipole_sizes = variables_list["dipole_sizes"]
#     separations_list = variables_list["separations_list"]
#     particle_sizes = variables_list["particle_sizes"]
#     particle_shapes = variables_list["particle_shapes"]
#     object_offsets = variables_list["object_offsets"]
#     dimensions = variables_list["dimensions"]
#     # get the independent variable (the one to be plotted against)
#     indep_name = variables_list["indep_var"]
#     indep_list = np.array(variables_list[indep_name])
#     # Based on the indep var, set what variable are varied over different lines of the graph.
#     match indep_name:
#         case "dipole_sizes": 
#             line_vars = [separations_list, particle_sizes, particle_shapes, object_offsets, dimensions]
#             indep_axis_list = indep_list
#         case "separations_list": 
#             line_vars = [dipole_sizes, particle_sizes, particle_shapes, object_offsets, dimensions]
#             indep_axis_list = indep_list[:,indep_vector_component] # vector var so pick which component to plot against
#         case "particle_sizes": 
#             line_vars = [dipole_sizes, separations_list, particle_shapes, object_offsets, dimensions]
#             indep_axis_list = indep_list
#         case "particle_shapes": 
#             line_vars = [dipole_sizes, separations_list, particle_sizes, object_offsets, dimensions]
#             indep_axis_list = indep_list
#         case "object_offsets": 
#             line_vars = [dipole_sizes, separations_list, particle_sizes, particle_shapes, dimensions]
#             indep_axis_list = indep_list[:,indep_vector_component] # vector var so pick which component to plot against
#         case "dimensions": 
#             line_vars = [dipole_sizes, separations_list, particle_sizes, particle_shapes, object_offsets]
#             indep_axis_list = indep_list # note, this is 1D (sphere and cube only)
            
    
#     print("Performing sphere-disc calculation")
#     data_set_params = []
#     num_indep = len(indep_axis_list)
#     # Iterate though every combination of variables that are varied across the lines of the graph.
#     var_set_length = len(indep_list)
#     for i in line_vars: # count total number of variable combinations
#         var_set_length *= len(i)

#     num_expts_per_param = len(forces_output)
#     data_set_length = int(var_set_length * num_expts_per_param/len(indep_list))

#     data_set = np.array([[indep_axis_list, np.zeros(num_indep)] for _ in range(data_set_length)], dtype=object)
#     particle_nums_set = np.array([[indep_axis_list, np.zeros(num_indep)] for _ in range(var_set_length)], dtype=object)
#     dpp_nums_set = np.array([[indep_axis_list, np.zeros(num_indep)] for _ in range(var_set_length)], dtype=object)

#     # Only make dipoles file if torque about given centre are needed.
#     if "Tmag" in forces_output or "Tx" in forces_output or "Ty" in forces_output or "Tz" in forces_output: include_dipole_forces = True
#     else: include_dipole_forces = False

#     # START OF LOOP OVER PARAMS
#     for params_i, params in enumerate(it.product(*line_vars)):
#         data_set_params.append(params)

#         match indep_name:
#             case "dipole_sizes": separations, particle_size, particle_shape, object_offset, dimension = params
#             case "separations_list": dipole_size, particle_size, particle_shape, object_offset, dimension = params
#             case "particle_sizes": dipole_size, separations, particle_shape, object_offset, dimension = params
#             case "particle_shapes": dipole_size, separations, particle_size, object_offset, dimension = params
#             case "object_offsets": dipole_size, separations, particle_size, particle_shape, dimension = params
#             case "dimensions": dipole_size, separations, particle_size, particle_shape, object_offset = params

#         # Iterate over independent variable to get the data for each line.
#         for i, indep_var in enumerate(indep_list):
#             print("\nProgress;"+str( "="*(int(np.floor(10*(i+ params_i*len(indep_list))/var_set_length))) )+str("-"*(int(np.floor(10*(1.0- (i+ params_i*len(indep_list))/var_set_length)))) )+"; "+str((i+ params_i*len(indep_list)))+"/"+str(var_set_length))
#             match indep_name:
#                 case "dipole_sizes": dipole_size = indep_var
#                 case "separations_list": separations = indep_var
#                 case "particle_sizes": particle_size = indep_var
#                 case "particle_shapes": particle_shape = indep_var
#                 case "object_offsets": object_offset = indep_var
#                 case "dimensions": dimension = indep_var
        
#             # Generate YAML & Run Simulation
#             particle_num = Generate_yaml.make_yaml_spheredisc_model(filename, dimension, separations, particle_size, dipole_size, object_offset, particle_shape, mode=mode, beam=beam_type, time_step=time_step, frames=frames, show_output=show_output)
#             DM.main(YAML_name=filename, force_terms=dda_forces_returned, include_dipole_forces=include_dipole_forces, polarisability_type=polarisability_type, verbosity=0)

#             print("PATICLE NUM IS ", particle_num)
#             match particle_shape:
#                 case "sphere": dpp_num = DM.sphere_size([particle_size], dipole_size)
#                 case "cube": dpp_num = DM.cube_size([particle_size], dipole_size)
#                 case _: sys.exit("UNIMPLEMENTED SHAPE")

#             #------------------
#             #------------------
#             # Simulation has run so have all the forces. Now do all experiments with force and particle selections
#             for expt_i in range(num_expts_per_param):
#                 force_type = forces_output[expt_i]
#                 particles = select_particle_indices(filename, particle_selections[expt_i], parameters_stored, read_frames=[0])
#                 read_parameters_args = read_parameters_lookup[force_type]
#                 read_parameters = []

#                 # Lookup values from <filename>.xlsx
#                 if force_type[0] == "F" or force_type[0] == "C":
#                     for p in particles:
#                         if p >= particle_num: p=particle_num-1; print(f"WARNING, set particle index to {particle_num}")
#                         read_parameters.extend([{"type":f, "particle":p, "subtype":s} for f,s in read_parameters_args])

#                     pulled_data = pull_file_data(
#                         filename, 
#                         parameters_stored, 
#                         read_frames, 
#                         read_parameters
#                     )
                
#                 # Lookup values from <filename>_dipoles.xlsx
#                 elif force_type[0] == "T":
#                     for p in particles:
#                         if p >= particle_num: p=particle_num-1; print(f"WARNING, set particle index to {particle_num}")
#                         # Now, loop over all dipoles in each desired particle.
#                         for d in range(dpp_num):
#                             read_parameters.extend([{"type":f, "particle":p*dpp_num+d, "subtype":s} for f,s in read_parameters_args])

#                     pulled_data = pull_file_data(
#                         filename+"_dipoles", 
#                         parameters_stored_torque, 
#                         read_frames, 
#                         read_parameters
#                     )

#                 # Calculate output from results
#                 value_list = pulled_data[0] # frame 0
#                 match force_type:
#                     case "Fmag" | "Cmag":
#                         output = np.zeros(3)
#                         for p in range(len(particles)):
#                             output += [pulled_data[0, 3*p+0], pulled_data[0, 3*p+1], pulled_data[0, 3*p+2]]
#                         output = np.linalg.norm(output)

#                     case "Tx" | "Ty" | "Tz":
#                         if force_type == "Tx": centre = torque_centre[1], torque_centre[2]
#                         elif force_type == "Ty": centre = torque_centre[2], torque_centre[0]
#                         elif force_type == "Tz": centre = torque_centre[0], torque_centre[1]
#                         output = 0
#                         for d in range(len(particles) * dpp_num):
#                             output += (value_list[4*d+0]-centre[0]) * value_list[4*d+3] - (value_list[4*d+1]-centre[0]) * value_list[4*d+2] # order for cross product comes from read_parameters_args
#                         # output *=2 # TEMP XXX REMOVE
#                     case "Tmag":
#                         output = np.zero(3)
#                         for d in range(len(particles) * dpp_num):
#                             output += np.cross(value_list[4*d+0:4*d+3] - torque_centre, value_list[4*d+3:4*d+6]) # order for cross product comes from read_parameters_args
#                         output = np.linalg.norm(output)
                    
#                     # Normally just sum all the force components in pulled_data
#                     case _:
#                         output = np.sum(pulled_data[0])
                    
#                 data_set[params_i*num_expts_per_param + expt_i, 1, i] = output
#             #------------------
#             #------------------
            
#             particle_nums_set[params_i, 1, i] = particle_num
#             dpp_nums_set[params_i, 1, i] = dpp_num

#     # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
#     return data_set, data_set_params, particle_nums_set, dpp_nums_set
            
def simulations_refine_all(filename, variables_list, partial_yaml_func, forces_output, particle_selections, indep_vector_component=2, torque_centre=[0,0,0]):
    #
    # Calculates the forces/torques for a single frame based on a set of parameters. Arbitrary YAML, based on partial_yaml_func.
    # variables_list contains all of the parameters to be changed (dict). One is set to be the independent variable, to be plotted on the x-axis, all combinations of the other variables are taken and plotted on different graph lines.
    #
    # INPUT
    # dda_forces_returned are the forces returned from the DDA - usually ["optical"]. object_shape is either "sphere" or "cube"
    # forces_output is the force types returned, optoins are ["Fmag", "Fx", "Fy", "Fz", "Cmag", "Cx", "Cy", "Cz", "Tmag", "Tx", "Ty", "Tz"]
    # particle_selections is a list of: string keywords ("all"), list of ints (particle indices), or list of float vectors ([0,0,0], particle closest to this point)
    # Requires form partial_yaml_func(dimension=dimension, separations=separations, particle_size=particle_size, dipole_size=dipole_size, object_offset=object_offset, particle_shape=particle_shape)
    #
    # OUTPUT
    # Returns data_set, data_set_params, particle_nums_set, dpp_nums_set
    # data set is [ [[indep var list], [force list]] for F, p in zip(forces_output, particle_selections) for each set of line parameters]
    # the other returns give different data for the same pattern.
    #
    if len(forces_output) != len(particle_selections): sys.exit(f"ERROR, forces_output and particle_selections must be the same length, not {len(forces_output)} and {len(particle_selections)}.")

    # Specify all parameters in the xlsx file so a subset can be pulled later based on read_parameters. Gives information about the structure of the data in the file.
    parameters_stored = [
        {"type":"X", "args":["x", "y", "z"]},
        {"type":"F", "args":["Fx", "Fy", "Fz"]},
        {"type":"F_T", "args":["F_Tx", "F_Ty", "F_Tz"]}, # (total force including hydrodynamics, not used here)
        {"type":"C", "args":["Cx", "Cy", "Cz"]}
    ]
    # Dipole file stores fewer params. Used in torque calculation.
    parameters_stored_torque = [
        {"type":"X", "args":["x", "y", "z"]},
        {"type":"F", "args":["Fx", "Fy", "Fz"]},
    ]
    read_frames = [0]

    # Record what parameters each force output would require
    read_parameters_lookup = {
        "Fmag": [["F",0], ["F",1], ["F",2]],
        "Fx":   [["F",0]],
        "Fy":   [["F",1]],
        "Fz":   [["F",2]],
        "Cmag": [["C",0], ["C",1], ["C",2]],
        "Cx":   [["C",0]],
        "Cy":   [["C",1]],
        "Cz":   [["C",2]],
        "Tmag": [["X",0], ["X",1], ["X",2], ["F",0], ["F",1], ["F",2]], # normal order
        "Tx":   [["X",1], ["X",2], ["F",1], ["F",2]], # y,z,Fy,Fz
        "Ty":   [["X",2], ["X",0], ["F",2], ["F",0]], # z,x,Fz,Fx
        "Tz":   [["X",0], ["X",1], ["F",0], ["F",1]], # x,y,Fx,Fy
    }

    # Begin calculations
    # get list of variables from the dictionary.
    dipole_sizes = variables_list["dipole_sizes"]
    separations_list = variables_list["separations_list"]
    particle_sizes = variables_list["particle_sizes"]
    particle_shapes = variables_list["particle_shapes"]
    object_offsets = variables_list["object_offsets"]
    dimensions = variables_list["dimensions"]
    # (make a default material)
    if "materials" in variables_list.keys(): materials = variables_list["materials"]
    else: materials = ["FusedSilica"]
    # get the independent variable (the one to be plotted against)
    indep_name = variables_list["indep_var"]
    indep_list = np.array(variables_list[indep_name])

    # Based on the indep var, set what variables are varied over different lines of the graph.
    match indep_name:
        case "dipole_sizes": 
            line_vars = [separations_list, particle_sizes, particle_shapes, object_offsets, dimensions, materials]
            indep_axis_list = indep_list
        case "separations_list": 
            line_vars = [dipole_sizes, particle_sizes, particle_shapes, object_offsets, dimensions, materials]
            indep_axis_list = indep_list[:,indep_vector_component] # vector var so pick which component to plot against
        case "particle_sizes": 
            line_vars = [dipole_sizes, separations_list, particle_shapes, object_offsets, dimensions, materials]
            indep_axis_list = indep_list
        case "particle_shapes": 
            line_vars = [dipole_sizes, separations_list, particle_sizes, object_offsets, dimensions, materials]
            indep_axis_list = indep_list
        case "object_offsets": 
            line_vars = [dipole_sizes, separations_list, particle_sizes, particle_shapes, dimensions, materials]
            indep_axis_list = indep_list[:,indep_vector_component] # vector var so pick which component to plot against
        case "dimensions": 
            line_vars = [dipole_sizes, separations_list, particle_sizes, particle_shapes, object_offsets, materials]
            indep_axis_list = indep_list # note, this is 1D (sphere and cube only)
        case "materials": 
            line_vars = [dipole_sizes, separations_list, particle_sizes, particle_shapes, object_offsets, dimensions]
            indep_axis_list = indep_list 
            
    print("Performing refinement calculation")
    data_set_params = []
    num_indep = len(indep_axis_list)
    # Iterate though every combination of variables that are varied across the lines of the graph.
    var_set_length = len(indep_list)
    for i in line_vars: # count total number of variable combinations
        var_set_length *= len(i)

    # Each expt is a pair of elements: forces_output[i], particle_selections[i]
    num_expts_per_param = len(forces_output)
    data_set_length = int(var_set_length * num_expts_per_param/len(indep_list))

    data_set = np.array([[indep_axis_list, np.zeros(num_indep)] for _ in range(data_set_length)], dtype=object)
    particle_nums_set = np.array([[indep_axis_list, np.zeros(num_indep)] for _ in range(int(var_set_length/len(indep_list)))], dtype=object)
    dpp_nums_set = np.array([[indep_axis_list, np.zeros(num_indep)] for _ in range(int(var_set_length/len(indep_list)))], dtype=object)

    # Start the loop over all parameters to be varied across different lines of the graph.
    for params_i, params in enumerate(it.product(*line_vars)):
        data_set_params.append(params)

        # Recover what variables params contains
        match indep_name:
            case "dipole_sizes": separations, particle_size, particle_shape, object_offset, dimension, material = params
            case "separations_list": dipole_size, particle_size, particle_shape, object_offset, dimension, material = params
            case "particle_sizes": dipole_size, separations, particle_shape, object_offset, dimension, material = params
            case "particle_shapes": dipole_size, separations, particle_size, object_offset, dimension, material = params
            case "object_offsets": dipole_size, separations, particle_size, particle_shape, dimension, material = params
            case "dimensions": dipole_size, separations, particle_size, particle_shape, object_offset, material = params
            case "materials": dipole_size, separations, particle_size, particle_shape, object_offset, dimension = params

        # Iterate over independent variable to get the data for each line.
        for i, indep_var in enumerate(indep_list):
            print("\nProgress;"+str( "="*(int(np.floor(10*(i+ params_i*len(indep_list))/var_set_length))) )+str("-"*(int(np.floor(10*(1.0- (i+ params_i*len(indep_list))/var_set_length)))) )+"; "+str((i+ params_i*len(indep_list)))+"/"+str(var_set_length))
            # Very precautious clearing of file here, as there were issues with incorrect reading before
            if(os.path.exists(filename+".xlsx")):
                os.remove(filename+".xlsx")
            if(os.path.exists(filename+"_dipoles.xlsx")):
                os.remove(filename+"_dipoles.xlsx")

            match indep_name:
                case "dipole_sizes": dipole_size = indep_var
                case "separations_list": separations = indep_var
                case "particle_sizes": particle_size = indep_var
                case "particle_shapes": particle_shape = indep_var
                case "object_offsets": object_offset = indep_var
                case "dimensions": dimension = indep_var
        
            # Generate YAML & Run Simulation
            particle_num = partial_yaml_func(dimension=dimension, separations=separations, particle_size=particle_size, dipole_size=dipole_size, object_offset=object_offset, particle_shape=particle_shape, material=material)
            DM.main(YAML_name=filename)

            match particle_shape:
                case "sphere": dpp_num = DM.sphere_size([particle_size], dipole_size)
                case "cube": dpp_num = DM.cube_size([particle_size], dipole_size)
                case _: sys.exit("UNIMPLEMENTED SHAPE")

            particle_nums_set[params_i, 1, i] = particle_num
            dpp_nums_set[params_i, 1, i] = dpp_num

            # Simulation has run so have all the forces. Now do all experiments with force and particle selections
            data_set = get_forces_via_lookup(filename, data_set, particle_num, i, params_i, forces_output, particle_selections, read_frames, read_parameters_lookup, parameters_stored, parameters_stored_torque, torque_centre=torque_centre, dpp_num=dpp_num)
            
    return data_set, data_set_params, particle_nums_set, dpp_nums_set

def get_forces_via_lookup(filename, data_set, particle_num, i, params_i, expt_output, particle_selections, read_frames, read_parameters_lookup, parameters_stored, parameters_stored_torque=None, torque_centre=None, dpp_num=None):
    #
    # Gets the forces specified for the particles specified from some file
    #
    # param_i = iteration through line variables (in context of the parameter varying method)
    # i = iteration through indepent variables (in context of the parameter varying method)
    #       These both control where elements should be placed in the output data_set
    #
    num_expts_per_param = len(expt_output)
    for expt_i in range(num_expts_per_param):
        expt_type = expt_output[expt_i]
        particles = select_particle_indices(filename, particle_selections[expt_i], parameters_stored, read_frames=[0])
        read_parameters_args = read_parameters_lookup[expt_type]
        read_parameters = []

        
        if expt_type[0] == "T": 
            # Look up torques in a different file.
            # !! NOTE Careful not to start non-torque expts with "T"
            # Lookup values from <filename>_dipoles.xlsx
            for p in particles:
                if p >= particle_num: p=particle_num-1; print(f"WARNING, set particle index to {particle_num}")
                # Now, loop over all dipoles in each desired particle.
                for d in range(dpp_num):
                    # print(f"particle number {particle_num}, dpp {dpp_num}, p*dpp_num+d={p*dpp_num+d}")
                    read_parameters.extend([{"type":f, "particle":p*dpp_num+d, "subtype":s} for f,s in read_parameters_args])

            pulled_data = pull_file_data(
                filename+"_dipoles", 
                parameters_stored_torque, 
                read_frames, 
                read_parameters
            )
        
        else:
            # Default is the normal xlsx, and look up used passed in params from read_parameters_lookup[expt_type]
            # Lookup values from <filename>.xlsx
            for p in particles:
                if p >= particle_num: p=particle_num-1; print(f"WARNING, set particle index to {particle_num}")
                read_parameters.extend([{"type":f, "particle":p, "subtype":s} for f,s in read_parameters_args])

            pulled_data = pull_file_data(
                filename, 
                parameters_stored, 
                read_frames, 
                read_parameters
            )
        
        # Calculate output from results
        value_list = pulled_data[0] # frame 0
        match expt_type:
            # These used in refine_all
            case "Fmag" | "Cmag":
                output = np.zeros(3)
                for p in range(int(len(value_list)/3)):
                    output += [value_list[3*p+0], value_list[3*p+1], value_list[3*p+2]] 
                output = np.linalg.norm(output)
                data_set[params_i*num_expts_per_param + expt_i, 1, i] = output
            case "FZmag" | "CZmag":
                output = np.zeros(3)
                for p in range(int(len(value_list)/3)):
                    output += [value_list[3*p+0], value_list[3*p+1], value_list[3*p+2]] 
                output = output[2]
                data_set[params_i*num_expts_per_param + expt_i, 1, i] = output

            case "Tx" | "Ty" | "Tz":
                if expt_type == "Tx": centre = torque_centre[1], torque_centre[2]
                elif expt_type == "Ty": centre = torque_centre[2], torque_centre[0]
                elif expt_type == "Tz": centre = torque_centre[0], torque_centre[1]
                output = 0
                for d in range(len(particles) * dpp_num):
                    output += (value_list[4*d+0]-centre[0]) * value_list[4*d+3] - (value_list[4*d+1]-centre[0]) * value_list[4*d+2] # order for cross product comes from read_parameters_args
                data_set[params_i*num_expts_per_param + expt_i, 1, i] = output

            case "Tmag":
                output = np.zeros(3)
                for d in range(len(particles) * dpp_num):
                    output += np.cross(value_list[4*d+0:4*d+3] - torque_centre, value_list[4*d+3:4*d+6]) # order for cross product comes from read_parameters_args
                output = np.linalg.norm(output)
                data_set[params_i*num_expts_per_param + expt_i, 1, i] = output

            # These used in stretcher: note listed as one expt, but cases designed so 3 (x,y,z) outputs are put into data set.

            case "Z_split":
                #
                # Sum forces on the upper and lower half planes separately, then add magnitudes together at the end => opposing forces
                # on each side won't cancel, but double-up
                #
                output = np.zeros(3)
                uhp_output = np.zeros(3)
                lhp_output = np.zeros(3)
                for p in range(int(len(value_list)/6)):
                    force = value_list[ p*6+0 : p*6+3 ]
                    pos   = value_list[ p*6+3 : p*6+6 ]
                    if( not((-sys.float_info.epsilon < pos[2]) and (pos[2] < sys.float_info.epsilon)) ):   # If outside the central layer
                        if( pos[2] > 0.0 ):     # If in upper half plane, sum forces
                            uhp_output += [force[0], force[1], force[2]]
                        else:                   # If in lower half plane, sum forces
                            lhp_output += [force[0], force[1], force[2]]
                    # Ignore forces at the central plane of the system
                # Add magnitude of these forces together
                output += [ (uhp_output[0]),  (uhp_output[1]),  (uhp_output[2])]
                output += [-(lhp_output[0]), -(lhp_output[1]), -(lhp_output[2])]
                # note currently params_i*num_expts_per_param + expt_i = 0
                data_set[params_i*num_expts_per_param + expt_i : params_i*num_expts_per_param + expt_i +3, 1, i] = output

            case "XYZ_split":
                #
                # Just sum force in the positive XYZ corner, assume symmetry for others hence this will show force pushing / pulling on either side
                #
                output = np.zeros(3)
                for p in range(int(len(value_list)/6)):
                    force = value_list[ p*6+0 : p*6+3 ]
                    pos   = value_list[ p*6+3 : p*6+6 ]
                    if( (pos[0] +sys.float_info.epsilon > 0.0) and (pos[1] +sys.float_info.epsilon > 0.0) and (pos[2] +sys.float_info.epsilon > 0.0) ):   # If not +X,+Y,+Z corner, then sum forces
                        output += [force[0], force[1], force[2]]
                # note currently params_i*num_expts_per_param + expt_i = 0
                data_set[params_i*num_expts_per_param + expt_i : params_i*num_expts_per_param + expt_i +3, 1, i] = output

            case "RTZ_split":
                #
                # x,y forces replaced by r,theta. z forces summed if in UHP, else minused.
                #
                output = np.zeros(3)
                sign = lambda x: 1 if x>=sys.float_info.epsilon else ( -1 if x<=-sys.float_info.epsilon else 0) # sign, but =0 very close to x=0
                for p in range(int(len(value_list)/6)):
                    force = value_list[ p*6+0 : p*6+3 ]
                    pos   = value_list[ p*6+3 : p*6+6 ]
                    theta = np.arctan2(pos[1], pos[0])
                    force[0], force[1] = force[0]*np.cos(theta) + force[1]*np.sin(theta), -force[0]*np.sin(theta) + force[1]*np.cos(theta) # decompose x,y into r,theta
                    output += [force[0], force[1], force[2] * sign(pos[2])]
                # note currently params_i*num_expts_per_param + expt_i = 0
                data_set[params_i*num_expts_per_param + expt_i : params_i*num_expts_per_param + expt_i +3, 1, i] = output


            # For most force_types, just sum all the force components in value_list
            case _:
                output = np.sum(value_list)
                data_set[params_i*num_expts_per_param + expt_i, 1, i] = output

    return data_set


def make_param_strs(data_set_params, legend_params, indep_name):
    #
    # Concatenate display strings for each parameter in the legend, presenting its name and value.
    #
    param_strs = []
    i_dict = {"dipole_sizes":0, "separations_list":1, "particle_sizes":2, "particle_shapes":3, "object_offsets":4, "dimensions":5, "materials":6} # convert between names and list index.
    indep_val = i_dict[indep_name]
    for params in data_set_params:
        # Pick what variables to show in the legend.
        param_str = ""
        for key, value in i_dict.items():
            if value > indep_val:
                value -= 1 # params DOESN'T include the indep var so the indices in i_dict beyond the indep var must be shifted (-1) to fill the gap.
            if key in legend_params:
                param_str += f", {display_var(key, params[value])}"
        param_strs.append(param_str)
    return param_strs

def filter_data_set(force_filter, data_set, data_set_params, legend_params, indep_name, N=4):
    #
    # Filter for what force types are wanted
    # Options, force_filter=["Fmag", "Fx", "Fy", "Fz"] XXX could add Ftheta, Fr
    # Returns filtered data_set, datalabel_set
    # The label only use the variables in legend_params since others do not change so are shown in the title.
    #
    filtered_i = []
    datalabel_set = []
    param_strs = make_param_strs(data_set_params, legend_params, indep_name)
    # Iterate over the experiments
    for i in range(len(data_set_params)):
        param_str = param_strs[i]

        # Pick what forces to plot and create the legend string.
        if "Fmag" in force_filter:
            filtered_i.append(N*i)
            datalabel_set.append(f"F Mag{param_str}")
        if "Fx" in force_filter:
            filtered_i.append(N*i+1)
            datalabel_set.append(f"Fx{param_str}")
        if "Fy" in force_filter:
            filtered_i.append(N*i+2)
            datalabel_set.append(f"Fy{param_str}")
        if "Fz" in force_filter:
            filtered_i.append(N*i+3)
            datalabel_set.append(f"Fz{param_str}")
        if "Fpoint" in force_filter:
            filtered_i.append(N*i+4)
            datalabel_set.append(f"Fpoint{param_str}")
        if "Fpoint_perDip" in force_filter:
            filtered_i.append(N*i+5)
            datalabel_set.append(f"Fpoint_perDip{param_str}")
        if "F_T" in force_filter:
            filtered_i.append(N*i+6)
            datalabel_set.append(f"F_T{param_str}")
    
    return data_set[filtered_i], datalabel_set, filtered_i

def display_var(variable_type, value=None):
    """
    * Returns a string based on the arguments
    * if value=None, returns the name and units of variable_type as a tuple e.g. "dipole size", "/m"
    * Else returns a single formatted string using the value e.g. "dipole size = 4e-8m"
    """
    if value == None:
        # Used to make the x-axis label
        match variable_type:
            # note, these strings should match the starts of the value!=None case, as used in get_colourline to match strings (force_output is an exception)
            case "dipole_sizes": return "dipole size", "[m]"
            case "separations_list": return "separation", "[m]"
            case "particle_sizes": return "particle size", "[m]"
            case "particle_shapes": return " particle shape", ""
            case "object_offsets": return "offset", "[m]"
            case "deflections": return "deflection", "[m]"
            case "dimensions": return "dimension", "[m]"
            case "materials": return "material", ""
            # the below cases are for matching linestyle_var_str in get_colourline, not for axis labels
            case "particle_selections": return "particle selection", ""

            case _: return f"{variable_type} UNKNOWN", "UNITS"
            # Note, particle selection and force type shouldn't be on the x-axis so are not cases here.
    
    else:
        # Used to make the title/lege§nd labels
        mkstr = lambda s: "0.0" if s==0 else f"{s:.2e}" # function to format a float in scientific notation, except for 0.
        match variable_type:
            case "dipole_sizes": return f"dipole size = {value:.2e}m"
            case "separations_list": return f"separation = [{mkstr(value[0])},{mkstr(value[1])},{mkstr(value[2])}]m"
            case "particle_sizes": return f"particle size = {value:.2e}m"
            case "particle_shapes": return f" particle shape = {value}"
            case "object_offsets": return f"offset = [{mkstr(value[0])},{mkstr(value[1])},{mkstr(value[2])}]m"
            case "deflections": return f" deflection = {value:.2e}"
            case "dimensions": return f" dimension = {value:.1e}"
            case "materials": return f" material = {value}"
            case "forces_output": return f"{value}"
            case "particle_selections":
                if isinstance(value, str): return f"particle selection = {value}"
                elif isinstance(value[0], int): return f"particle selection = [{','.join(str(value[i]) for i in range(len(value)))}]"
                elif isinstance(value[0][0], (int, float, np.floating)): return f"particle selection = [{','.join(['['+','.join([mkstr(value[i][j]) for j in range(3)])+']' for i in range(len(value))])}]"
    
            case _: return f"{variable_type} UNKNOWN"

def get_titlelegend(variables_list, indep_name, particle_selection, dimensions):
    #
    # Formats the graph title.
    # Variables that don't change are put in the graph title. Otherwise, they are recorded to go in the legend.
    # This excludes the independent variable.
    #
    titlestr = f" against {display_var(indep_name)[0]}. dimensions = {dimensions}, particle selection = {particle_selection}\n"
    legend_params = []
    newline_count = 0
    # For each variable, print it in the legend or the title.
    for key, value in variables_list.items():
        if key == indep_name or key == "indep_var": 
            continue # Indep var isn't in title or legend.
        if len(value) == 1: # It doesn't change so put in the title
            titlestr += f", {display_var(key, value[0])}"
            if newline_count == 2:
                newline_count = 0
                titlestr += "\n"
            newline_count += 1
        else: # variable changes so keep in legend
            legend_params.append(key)
    if titlestr[-1]=="\n": # remove trailing \n
        titlestr = titlestr[:-1]
    return titlestr, legend_params

def get_colourline(datalabel_set, legend_params, variables_list, linestyle_var=None, cgrad=lambda x: (1/4+3/4*x, x/3, 1-x)):
    #
    # makes line styles and colours for each line on the graph.
    # linestyle_var can be "dipole_sizes", "separations_list", "particle_sizes", "particle_shapes", "object_offsets", "deflections", "forces_output", "particle_selections", "dimensions", "materials"
    # If linestyle_var=None, it will be set automatically
    # cgrad determines the colour tuples.
    #

    line_options = ["solid", "dashed", "dotted", "dashdot"] # Note, more could be added or some could be repeated.
    num_line_options = len(line_options)

    # Automatically select linestyle_var if useful, list below gives a priority order.
    if (linestyle_var == None or linestyle_var not in legend_params) and len(legend_params) > 1:
        for vars in ["particle_shapes", "object_offsets", "dipole_sizes", "separations_list", "particle_sizes", "forces_output", "particle_selections", "dimensions", "materials", "deflections"]:
            if vars in legend_params and len(variables_list[vars]) < num_line_options:
                linestyle_var = vars
                print(f"Note, set linestyle_var to {linestyle_var}")
                break
    
    # Trivial case: only colours change
    if linestyle_var == None: 
        linestyle_set = ["solid" for _ in range(len(datalabel_set))]
        data_colour_set = [i for i in range(len(datalabel_set))]

    # Non-trivial case
    else:
        # display_var will always be in the legend if param is present, so search for it, except force as that is just shown as e.g. "Fx".
        if linestyle_var == "forces_output":
            linestyle_var_str_list = variables_list["forces_output"]
            count = 0
        else:
            linestyle_var_str = display_var(linestyle_var)[0] # e.g. dipole_size -> dipole size

        linestyle_set = []
        data_colour_set = []
        # record params seen before, and get their index, else create new entry.
        linestyle_var_list = []
        other_var_list = []

        for label in datalabel_set:
            # forces output doesn't have a string label for it so, so need to test the value e.g. "Fx"
            if linestyle_var == "forces_output":
                linestyle_var_str = linestyle_var_str_list[count % len(linestyle_var_str_list)]
                count += 1

            # Split each legend label into the params it contains
            pieces = label.split(", ")
            # Find the piece corresponding to the line style parameter. 
            linestyle_var_piece = None
            for piece in pieces:
                if linestyle_var_str in piece:
                    linestyle_var_piece = piece
                    break
            if linestyle_var_piece == None: print(f"WARNING, could not find '{linestyle_var_str}' in pieces: {pieces}")

            # Build list of unique parameters, grouped into linestyle var, and non linestyle var params.
            if linestyle_var_piece not in linestyle_var_list:
                linestyle_var_list.append(linestyle_var_piece)

            pieces.remove(linestyle_var_piece)
            pieces_str = " ".join(pieces)
            if pieces_str not in other_var_list:
                other_var_list.append(pieces_str)

            # Use the index to enumerate unique parameters
            linestyle_count = linestyle_var_list.index(linestyle_var_piece)
            colour_count = other_var_list.index(pieces_str)

            linestyle_set.append(line_options[linestyle_count % num_line_options])
            data_colour_set.append(colour_count)

    # Convert indices to colour tuples.
    data_colour_set = np.array(data_colour_set, dtype=object)
    if np.max(data_colour_set) != 0:
        data_colour_set = data_colour_set/np.max(data_colour_set) # normalise
    for i in range(len(data_colour_set)):
        data_colour_set[i] =  cgrad(data_colour_set[i]) # turn into colours

    return linestyle_set, np.array(data_colour_set)

def get_title_label_line_colour(variables_list, data_set_params, forces_output, particle_selections, indep_name, linestyle_var=None, cgrad=lambda x: (1/4+3/4*x, x/3, 1-x)):
    #
    # Calculates: titlestrbase, datalabel_set, linestyle_set, datacolor_set, graphlabel_set
    # Works with data from simulations_refine_all
    #
    
    # 1) split variables over legend and title
    variables_list["forces_output"] = forces_output
    variables_list["particle_selections"] = particle_selections
    title_params, legend_params = split_title_legend(variables_list, indep_name)

    # 2) make title
    title_start= "Torques" if forces_output[0][0]=="C" else "Forces" +f" against {display_var(indep_name)[0]}" # try to determine if it is a torque or force plot.
    title_str = make_title(title_start, title_params, variables_list)

    # 3) make legend labels
    datalabel_set = make_legend_labels(data_set_params, legend_params, indep_name, forces_output, particle_selections)

    # 4) make linestyles and colours
    linestyle_set, datacolor_set = get_colourline(datalabel_set, legend_params, variables_list, linestyle_var=linestyle_var, cgrad=cgrad)

    # 5) axis labels
    y_axis = "Forces [N]" if forces_output[0][0]=="F" else "Torques [Nm]"
    graphlabel_set = {"title":title_str, "xAxis":f"{display_var(indep_var)[0]} {display_var(indep_var)[1]}", "yAxis":y_axis} 

    return title_str, datalabel_set, linestyle_set, datacolor_set, graphlabel_set

def split_title_legend(variables_list, indep_name):
    #
    # make lists of variable names which will go into the title and the legend
    #
    title_params = []
    legend_params = []
    for key, value in variables_list.items():
        if key == "indep_var" or key == indep_name: continue # skip indep variable name and list
        if len(value) == 1 or all(x == value[0] for x in value): # check if only one unique value
            title_params.append(key)
        else:
            legend_params.append(key)
    return title_params, legend_params

def make_title(title_start, title_params, variables_list):
    #
    # makes title string
    #
    title_str = title_start
    newline_count = 1
    for key, value in variables_list.items():
        if key in title_params: 
            title_str += f", {display_var(key, value[0])}"
            if newline_count == 2:
                newline_count = 0
                title_str += "\n"
            newline_count += 1
    if title_str[-1]=="\n": # remove trailing \n
        title_str = title_str[:-1]
    return title_str

def make_legend_labels(data_set_params, legend_params, indep_name, forces_output, particle_selections):
    #
    # make array of line labels
    #
    param_strs = make_param_strs(data_set_params, legend_params, indep_name) # strs with the params varying (not forces or particle selections)
    labels = []
    num_expts = len(forces_output)
    # Make force and particle base strings.
    base_strs = [""]*num_expts
    for i in range(num_expts):
        if "forces_output" in legend_params:
            base_strs[i] += display_var("forces_output", forces_output[i])
        if "particle_selections" in legend_params:
            if "forces_output" in legend_params: base_strs[i] += ", "
            base_strs[i] += display_var("particle_selections", particle_selections[i])

    for param_str in param_strs:
        if base_strs[0] == "": param_str = param_str[2:] # remove ", " prefix if no base string
        for i in range(num_expts):
            labels.append(f"{base_strs[i]}{param_str}")
    return labels



def dynamic_stretcher_vary(filename, variables_list, option_parameters, yaxis_label):
    # Makes a data set of either the eccentricity, the ratio of the longest and shortest radii against the frame number, or the bounding box ratio of the z height/ sqrt(x width * y width)
    # yaxis_label switches between the modes listed about, values should be "Eccentricity", "Height/width ratio", or "Bounding box ratio".
    # num_averaged is how many positions the longest/shortest are averaged over - reduces anomalous effects but dilutes differences.
    # Different variables are used, depending on varaibles list.
    # Other parameters are given in option_parameters.
    # If the "try" fails, the values will be set to impossible ones - all zeros

    # Initialise variables
    max_time = option_parameters["frames"] * max(variables_list["time_step"])
    parameters_stored = [{"type":"X", "args":["x", "y", "z"]},{"type":"F", "args":["Fx", "Fy", "Fz"]},{"type":"FT", "args":["FTx", "FTy", "FTz"]}, {"type":"C", "args":["Cx", "Cy", "Cz"]}]
    data_set = [] # it will be inhomogeneous
    datalabel_set = []
    pulled_data_set = []
    count = 0
    making_title = True
    title_str = yaxis_label + " against frame number"
    len_title = len(title_str)
    max_title_len = 100 # roughly
    num_expts = reduce(mul, [len(v) for v in variables_list.values()])
    material = "RBC" # Red blood cell

    for params in it.product(*variables_list.values()):
        print(f"Progress; {count}/{num_expts}")

        # extract params values. NOTE order of this is important
        option_parameters["stiffness_spec"]["default_value"] = params[0]
        option_parameters["constants"]["bending"] = params[1]
        translation = params[2]
        num_particles = params[3]
        particle_radius = params[4]
        option_parameters["dipole_radius"] = params[4]
        E0 = params[5]
        w0 = params[6]
        time_step = params[7]
        num_averaged = params[8]
        sphere_radius = params[9]
        # repeat = params[10] # does nothing, just used to see the effect of randomness.

        option_parameters["time_step"] = time_step
        times = np.arange(0, max_time+time_step, time_step)
        num_frames = len(times)
        option_parameters["frames"] = num_frames
        read_frames = np.arange(0, num_frames, 1)

        # Make title/legend
        line_label = ""
        for i, key in enumerate(variables_list.keys()):
            if len(variables_list[key])>1: 
                if line_label!="": line_label += ", "
                line_label += f"{key} = {params[i]}"
            elif making_title:
                if len_title > max_title_len: len_title=0; title_str += "\n"
                string = f", {key} = {params[i]}"
                title_str += string
                len_title += len(string)
        
        if title_str[-1] == "\n": title_str = title_str[:-1] # pop trailing \n
        making_title = False # just add params to the title for the first time

        # Run simulation for the current set of params.
        Generate_yaml.make_yaml_stretcher_springs(filename, option_parameters, num_particles, sphere_radius, particle_radius, connection_mode, connection_args, E0, w0, translation, material=material)
        particles = np.arange(0, num_particles, 1)
        data_set_values = np.zeros(num_frames) # default to zeros in case the "try" fails

        try:
            DM.main(filename)
            read_parameters = [{"type":"X", "particle":p, "subtype":s} for s, p in it.product(range(3), particles)]
            pulled_data = pull_file_data(filename, parameters_stored, read_frames, read_parameters)

            
            for f in read_frames:
                data_of_frame = pulled_data[f]
                positions = np.array([data_of_frame[0:num_particles], data_of_frame[num_particles:2*num_particles], data_of_frame[2*num_particles:3*num_particles]])

                match yaxis_label:
                    case "Eccentricity":
                        centre = np.average(positions, axis=1)
                        rs = np.linalg.norm(positions - centre[:,None], axis=0)
                        smallest_rs = rs[np.argpartition(rs, num_averaged)[:num_averaged]]  # sort for the num_averaged min values, then slice for them.
                        largest_rs = rs[np.argpartition(rs, -num_averaged)[-num_averaged:]] # sort for the num_averaged max values, then slice for them.
                        output = np.sqrt(1 - np.average(smallest_rs)**2/np.average(largest_rs)**2)
                    
                    case "Height/width ratio":
                        centre = np.average(positions, axis=1)
                        rs = np.linalg.norm(positions - centre[:,None], axis=0)
                        smallest_rs = rs[np.argpartition(rs, num_averaged)[:num_averaged]]  
                        largest_rs = rs[np.argpartition(rs, -num_averaged)[-num_averaged:]]
                        output = np.average(largest_rs)/np.average(smallest_rs) # long/short ratio
                    
                    case "Bounding box ratio":
                        smallest_xs = positions[0, np.argpartition(positions[0], num_averaged)[:num_averaged]]
                        largest_xs = positions[0, np.argpartition(positions[0], -num_averaged)[-num_averaged:]]
                        smallest_ys = positions[1, np.argpartition(positions[1], num_averaged)[:num_averaged]]
                        largest_ys = positions[1, np.argpartition(positions[1], -num_averaged)[-num_averaged:]]
                        smallest_zs = positions[2, np.argpartition(positions[2], num_averaged)[:num_averaged]]
                        largest_zs = positions[2, np.argpartition(positions[2], -num_averaged)[-num_averaged:]]

                        # print("\nx", smallest_xs, largest_xs)
                        # print("y", smallest_ys, largest_ys) 
                        # print("z", smallest_zs, largest_zs)                       
                        output = (np.average(largest_zs)-np.average(smallest_zs))/( np.sqrt( (np.average(largest_xs)-np.average(smallest_xs)) * (np.average(largest_ys)-np.average(smallest_ys))))

                data_set_values[f] = output

        except Exception as error: 
            pulled_data = None
            print(f"\nERROR, failed and continuing on params:")
            for i, key in enumerate(variables_list.keys()):
                print(f"{key} = {params[i]}")
            print(error)

        data_set.append( [times, data_set_values] )
        datalabel_set.append(line_label)
        pulled_data_set.append(pulled_data)
        count +=1
    
    data_set = np.array(data_set, dtype=object) # object as inhomogeneous
    graphlabel_set={"title":title_str, "xAxis":"Time [s]", "yAxis":f"{yaxis_label}"}
    graphlabel_set["yAxis"] = "Ratio of major to minor axis length"
    return data_set, datalabel_set, graphlabel_set, pulled_data_set

def calculate_MoI(pulled_data_set, data_set, axes=["z"]):
    """
    Finds the moment of inertia for each set of particles in pulled_data_set
    * Normal: sum over all particles of mr^2 (r relative to centre of mass)
    * Assuming spheroid: finds object height/width to get the MoI

    pulled_data_set contains the particle positions with indices [experiment][frame][flatted positions: all x's, then all y's, then all z's]

    Returns data_set_moi, data_set_ideal: 
        moi of particle positions, moi of spheroid with dimensions from the data respectively
    Indices of returned lists are for [select from "axes"][experiment][0: time vals, 1: mois][specific time val/moi]
    """

    pulled_data_set = np.array(pulled_data_set)
    data_set = np.array(data_set)
    times = data_set[0,0,:]
    num_axes = len(axes)
    num_expts = len(data_set)
    num_frames = len(times)
    num_particles = int(len(pulled_data_set[0,0])/3)
    particle_mass = 1/num_particles # XXX

    data_set_moi = np.array([ [[times, np.zeros(times.shape)] for _ in range(num_expts)]  for _ in range(num_axes)])
    data_set_ideal = np.array([ [[times, np.zeros(times.shape)] for _ in range(num_expts)]  for _ in range(num_axes)])
    axis_lookup = {"x":[False, True, True], "y":[True, False, True], "z":[True, True, False]} # make np filters

    for expt_i in range(num_expts):
        for f in range(num_frames):
            data_of_frame = pulled_data_set[expt_i, f]
            positions = np.array([data_of_frame[0:num_particles], data_of_frame[num_particles:2*num_particles], data_of_frame[2*num_particles:3*num_particles]])
            centre = np.average(positions, axis=1)
            shifted_positions = positions - centre[:,None]
            

            for axi in range(num_axes):
                ax_filter = axis_lookup[axes[axi]]

                # actual moi
                rs = np.linalg.norm(shifted_positions[ax_filter], axis=0)
                moi = particle_mass * np.sum(rs**2)
                data_set_moi[axi, expt_i, 1, f] = moi

                # ideal spheroid moi
                # TODO using:
                # smallest_rs = rs[np.argpartition(rs, num_averaged)[:num_averaged]]  
                # largest_rs = rs[np.argpartition(rs, -num_averaged)[-num_averaged:]]
                # output = np.average(largest_rs)/np.average(smallest_rs) # long/short ratio
                # smallest_xs = positions[0, np.argpartition(positions[0], num_averaged)[:num_averaged]]
                # largest_xs = positions[0, np.argpartition(positions[0], -num_averaged)[-num_averaged:]]
                # smallest_ys = positions[1, np.argpartition(positions[1], num_averaged)[:num_averaged]]
                # largest_ys = positions[1, np.argpartition(positions[1], -num_averaged)[-num_averaged:]]
                # smallest_zs = positions[2, np.argpartition(positions[2], num_averaged)[:num_averaged]]
                # largest_zs = positions[2, np.argpartition(positions[2], -num_averaged)[-num_averaged:]]

    return data_set_moi, data_set_ideal

#=================#
# Perform Program #
#=================#
if int(len(sys.argv)) != 2:
    sys.exit("Usage: python <RUN_TYPE>")

match(sys.argv[1]):
    case "spheresInCircle":
        filename = "SingleLaguerre"
        #1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
        particle_numbers = [6,7,8,9,10,11,12,13,14,15,16]
        parameter_text = simulations_singleFrame_optForce_spheresInCircle(particle_numbers, filename, include_additionalForces=False)
        #Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, particle_numbers, "Particle number", "", parameter_text)
        Display.plot_tangential_force_against_number_averaged(filename+"_combined_data", parameter_text)
    case "torusInCircle":
        filename = "SingleLaguerre"
        particle_numbers = [2,3,4,5,6,7,8,9]
        parameter_text = simulations_singleFrame_optForce_torusInCircle(particle_numbers, filename)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, particle_numbers, "Particle number", "", parameter_text)
    case "torusInCircleFixedPhi":
        filename = "SingleLaguerre"
        particle_numbers = [1,2,3,4,5,6,7,8,9,10,11,12]
        parameter_text = simulations_singleFrame_optForce_torusInCircleFixedPhi(particle_numbers, filename)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, particle_numbers, "Particle number", "", parameter_text)
    case "spheresInCircleSlider":
        filename = "SingleLaguerre"
        theta_range = [np.pi/6.0, np.pi, 50] #np.pi/2.0, 3.0*np.pi/2.0,
        parameter_text = simulations_singleFrame_optForce_spheresInCircleSlider(1, theta_range, filename)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, np.linspace(*theta_range), "slider theta", "(radians)", parameter_text)
    case "spheres_wavelengthTrial":
        filename      = "SingleLaguerre"
        wavelength    = 1.0e-6
        beam_radius   = 1.15e-6
        target_pos    = [2.0*beam_radius, 0.0, 1.0e-6]
        target_radius = 200e-9
        wave_jump = wavelength/8.0
        wave_start = 4.0*wave_jump
        x_values = np.arange(wave_start, abs(target_pos[0])+beam_radius, wave_jump) /wavelength 
        #NOTE; Make sure the start is a multiple of jump in order for constructive to be nice
        parameter_text = simulations_singleFrame_optForce_wavelengthTrial(wave_start, wave_jump, beam_radius, target_pos, target_radius, filename, wavelength=wavelength, reducedSet=2)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, x_values, "Wave spacing", "(wavelengths)", parameter_text)
    case "spheresInCircleDipoleSize":
        filename = "SingleLaguerre"
        particle_total = 12
        dipole_size_range = [50e-9, 150e-9, 20]
        parameter_text, dipole_sizes = simulations_singleFrame_optForce_spheresInCircleDipoleSize(particle_total, dipole_size_range, filename)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, np.linspace(*dipole_size_range), "Dipole size", "(m)", parameter_text)
    case "torusInCircleDipoleSize":
        filename = "SingleLaguerre"
        particle_total = 6
        separation = 0.5e-7
        dipole_sizes = [35e-9, 200e-9, 20]

        # Option to filter dipole sizes so that the objects have a similar volume.
        filter_dipoleSizes_by_volume = False
        if filter_dipoleSizes_by_volume:
            particle_total = 6
            separation = 1e-7
            dipole_sizes = [60e-9, 30e-9, 150]
            inner_radii = 1.15e-6
            tube_radii = 200e-9
            filter_num = 25
            old_dipole_sizes = dipole_sizes
            volumes = get_torus_volumes(particle_total, inner_radii, tube_radii, separation, dipole_sizes)
            dipole_sizes, indices, _ = filter_dipole_sizes(volumes, dipole_sizes, filter_num)
        parameter_text, dipole_sizes = simulations_singleFrame_optForce_torusInCircleDipoleSize(particle_total, dipole_sizes, filename, separation)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, make_array(dipole_sizes), "Dipole size", "(m)", parameter_text)
    case "torusInCircleSeparation":
        filename = "SingleLaguerre"
        particle_total = 6
        separation_range = [0.2e-7, 3e-7, 25]
        dipole_size = 40e-9
        parameter_text, dipole_sizes = simulations_singleFrame_optForce_torusInCircleSeparation(particle_total, separation_range, filename, dipole_size)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, np.linspace(*separation_range), "Separation", "(m)", parameter_text)
    case "torusInCircle_FixedSep_SectorDipole":
        #
        # Compares force for different sector numbers, for a dipole sizes
        #

        #particle_numbers = [1,2,3,4,5,6,7,8, 12, 16]
        #dipoleSize_numbers = [40e-9, 50e-9, 60e-9, 70e-9] #np.linspace(...)
        #separation = 300e-9
        
        filename = "SingleLaguerre"
        particle_numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        dipoleSize_numbers = [40e-9, 50e-9, 60e-9, 70e-9] #np.linspace(...)
        data_axes = [dipoleSize_numbers, particle_numbers]
        separation = 0#1e-7
        parameter_text, data_set = simulations_singleFrame_optForce_torusInCircle_FixedSep_SectorDipole(particle_numbers, dipoleSize_numbers, separation, filename)
        Display.plotMulti_tangential_force_against_arbitrary(data_set, data_axes, 0, ["Dip.Rad", "Particle Number"], ["(m)", ""], parameter_text)
    case "testVolumes":
        # use this mode to make a new volume storage entry or to plot it.
        particle_total = 6
        dipole_size_range = [60e-9, 30e-9, 150]
        separation = 1e-7
        inner_radii = 1.15e-6
        tube_radii = 200e-9
        filter_num = 25

        volumes = get_torus_volumes(particle_total, inner_radii, tube_radii, separation, dipole_size_range)
        # sphere_radius = 200e-9
        # volumes = get_sphere_volumes(particle_total, sphere_radius, dipole_size_range)

        filtered_dipole_sizes, filtered_volumes, max_volume_error = filter_dipole_sizes(volumes, dipole_size_range, filter_num)
        Display.plot_volumes_against_dipoleSize(np.linspace(*dipole_size_range), volumes, filtered_dipole_sizes, filtered_volumes)
    case "connected_sphereGrid":
        #
        # Currently just runs the simulation for observation, no data is recorded or stored in .xlsx files here
        #
        filename = "SingleLaguerre"
        particle_radius  = 100e-9#100e-9
        particle_spacing = 400e-9#60e-9
        bounding_sphere_radius = 2e-6
        connection_mode = "dist"
        connection_args = [2*100e-9 +500e-9] # [2*100e-9 +100e-9]
        parameter_text = simulations_singleFrame_connected_sphereGrid(particle_radius, particle_spacing, bounding_sphere_radius, connection_mode, connection_args, filename)
    case "connected_sphereShell":
        #
        # Currently just runs the simulation for observation, no data is recorded or stored in .xlsx files here
        #
        filename = "SingleLaguerre"
        particle_radius  = 100e-9#100e-9
        num_pts = 50
        shell_radius = 1e-6
        connection_mode = "dist"
        connection_args = [600e-9] # [2*100e-9 +100e-9] # these connection_args copied from connected_sphereGrid
        parameter_text = simulations_singleFrame_connected_sphereShell(particle_radius, num_pts, shell_radius, connection_mode, connection_args, filename)
    
    #
    # A series of tests for fibres using different models, measuring the flexibility, forces, etc experienced by each
    #
    case "fibre_1D_sphere":
        # fibre/line of sphere particles
        # Save file
        filename = "SingleLaguerre"
        # Args
        chain_length    = 3e-6
        particle_radius = 100e-9
        particle_number = 5
        option_parameters = Generate_yaml.fill_yaml_options({
            "time_step": 1e-5,
            "frames": 100,
            "constants": {"bending": 0.5e-18},
            "stiffness_spec": {"type":"", "default_value":5e-7},
            "force_terms": ["optical", "spring", "bending", "buckingham"],
            "show_output": True,
        })

        particle_separation = chain_length/particle_number
        connection_mode = "dist"
        connection_args = 1.1*particle_separation
        # Run
        parameter_text = simulations_fibre_1D_sphere(filename, chain_length, particle_radius, particle_number, connection_mode, connection_args, option_parameters)
    case "fibre_1D_cylinder":
        # fibre/line of cylinder particles
        # Save file
        filename = "SingleLaguerre"
        # Args
        chain_length    = 3e-6
        particle_length = 300e-9
        particle_radius = 200e-9
        particle_number = 5
        option_parameters = Generate_yaml.fill_yaml_options({
            "time_step": 1e-5,
            "frames": 100,
            "constants": {"bending": 0.5e-18},
            "stiffness_spec": {"type":"", "default_value":5e-7},
            "force_terms": ["optical", "spring", "bending", "buckingham"],
            "show_output": True,
        })

        particle_separation = chain_length/particle_number
        connection_mode = "dist"
        connection_args = 1.1*particle_separation
        # Run
        parameter_text = simulations_fibre_1D_cylinder(filename, chain_length, particle_length, particle_radius, particle_number, connection_mode, connection_args, option_parameters)
    
    case "fibre_2D_sphere_hollowShell":
        # Save file
        filename = "SingleLaguerre"
        # Args
        chain_length    = 3e-6
        particle_radius = 100e-9
        shell_radius    = 500e-9
        particle_number_radial  = 6
        particle_number_angular = 8
        E0 = 4.6e7
        object_offset=np.array([0.0, -1.0e-6, 0.0])

        stiffness = 1.5e-6
        include_beads = True  # Silica beads attached to either side of the rod, used to deform the rod

        # Get connections
        connection_mode = "dist"
        connection_args = [0.0]   # NOTE; Is overwritten in the function to pick the correct value
        if(include_beads):
            connection_mode = "dist_beads"
            connection_args = [0.0, 0.0, 2] # NOTE; 0.0 values will be overwritten later in the function for correct values
            # "dist_beads" has args = [
            #       distance to connect non-bead particles,
            #       distance to connect bead particles,
            #       number of bead particles (NOTE; Assumes all beads are located at the end of the particle list)
            #   ]

        # Get stiffness matrix specs
        bead_indices = []
        for i in range(1, connection_args[2]+1):
            bead_indices.append((particle_number_radial*particle_number_angular)-i)

        option_parameters = Generate_yaml.fill_yaml_options({
            "time_step": 0.25e-4,
            "frames": 90,
            "max_size":3e-6,    #2e-6
            "constants": {"bending": 0.1e-18},
            "stiffness_spec": {"type":"beads", "default_value":stiffness, "bead_value":5.0*stiffness, "bead_indices":bead_indices}, # for uniform stiffness: {"type":"", "default_value":1e-6}
            "force_terms": ["optical", "spring", "bending"],
            "show_output": True,
            "beam_planes": [['z',0]],
            "quiver_setting": 0,
            "frames": 1,
        })

        # Run
        parameter_text = simulations_fibre_2D_sphere_hollowShell(filename, E0, option_parameters, object_offset, chain_length, shell_radius, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, include_beads=include_beads)
    case "fibre_2D_cylinder_hollowShell":
        # Save file
        filename = "SingleLaguerre"
        # Args
        chain_length    = 3e-6
        particle_length = 300e-9
        particle_radius = 100e-9
        shell_radius    = 300e-9
        particle_number_radial  = 8
        particle_number_angular = 6

        connection_mode = "dist"
        connection_args = 0.0   # NOTE; Is overwritten in the function to pick the correct value

        option_parameters = Generate_yaml.fill_yaml_options({
            "time_step": 5e-5,
            "frames": 90,
            "constants": {"bending": 0.65e-18},
            "stiffness_spec": {"type":"", "default_value":5e-7},
            "force_terms": ["optical", "spring", "bending"],
            "show_output": True,
            "beam_planes": [],
            "quiver_setting": 0
        })

        # Run
        parameter_text = simulations_fibre_2D_cylinder_hollowShell(filename, chain_length, shell_radius, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, option_parameters)
    case "fibre_2D_sphere_thick_connectUniform":
        # Save file
        filename = "SingleLaguerre"
        # Args
        chain_length    = 0.5e-6
        particle_radius = 100e-9
        shell_radius    = 1600e-9
        shell_number    = 2
        particle_number_radial  = 4
        particle_number_angular = 12

        option_parameters = Generate_yaml.fill_yaml_options({
            "time_step": 1e-7,
            "frames": 10,
            "dipole_radius":particle_radius,
            "constants": {"bending": 0.1e-18},
            "stiffness_spec": {"type":"", "default_value":7.5e-6},
            "force_terms": ["optical", "spring", "bending"],
            "show_output": True,
            "beam_planes": [],
            "quiver_setting": 0
        })

        connection_mode = "dist"
        connection_args = 0.0   # NOTE; Is overwritten in the function to pick the correct value
        # Run
        parameter_text = simulations_fibre_2D_sphere_thick_connectUniform(filename, chain_length, shell_radius, shell_number, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, option_parameters)
    case "fibre_2D_cylinder_thick_connectUniform":
        # Save file
        filename = "SingleLaguerre"
        # Args
        chain_length    = 3e-6
        particle_length = 300e-9
        particle_radius = 100e-9
        shell_radius    = 400e-9
        shell_number    = 1
        particle_number_radial  = 6
        particle_number_angular = 6
        
        option_parameters = Generate_yaml.fill_yaml_options({
            "time_step": 1e-5,
            "frames": 50,
            "constants": {"bending": 0.1e-18},
            "stiffness_spec": {"type":"", "default_value":7.5e-6},
            "force_terms": ["optical", "spring", "bending"],
            "show_output": True,
        })

        connection_mode = "dist"
        connection_args = 0.0   # NOTE; Is overwritten in the function to pick the correct value
        # Run
        parameter_text = simulations_fibre_2D_cylinder_thick_connectUniform(filename, chain_length, shell_radius, shell_number, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, option_parameters)
    case "fibre_2D_sphere_shellLayers":
        # Save file
        filename = "SingleLaguerre"
        # Args
        chain_length    = 1.0e-6
        particle_radius = 0.15e-6
        shell_radius    = 1.0e-6
        shell_number    = 2
        particle_separation = (np.pi*2.0*shell_radius)/(15.0)

        option_parameters = Generate_yaml.fill_yaml_options({
            "time_step": 1e-7,
            "frames": 10,
            "quiver_setting":0,
            "dipole_radius": particle_radius,
            "constants": {"bending": 0.1e-18},
            "stiffness_spec": {"type":"", "default_value":7.5e-6},
            "force_terms": ["optical"], #["optical", "spring", "bending"],
            "show_output": True,
        })

        connection_mode = "dist"
        connection_args = 1.01*particle_separation
        # Run
        parameter_text = simulations_fibre_2D_sphere_shellLayers(filename, chain_length, shell_radius, shell_number, particle_radius, particle_separation, connection_mode, connection_args, option_parameters)
    case "fibre_2D_cylinder_shellLayers":
        # Save file
        filename = "SingleLaguerre"
        # Args
        chain_length    = 1.0e-6
        particle_length = 300e-9
        particle_radius = 0.15e-6
        shell_radius    = 1.0e-6
        shell_number    = 2
        particle_separation = (np.pi*2.0*shell_radius)/(15.0)

        option_parameters = Generate_yaml.fill_yaml_options({
            "time_step": 1e-4,
            "frames": 1,
            "constants": {"bending": 0.1e-18},
            "stiffness_spec": {"type":"", "default_value":7.5e-6},
            "force_terms": ["optical"], #["optical", "spring", "bending"],
            "show_output": True,
        })

        connection_mode = "dist"
        connection_args = 1.01*particle_separation
        # Run
        parameter_text = simulations_fibre_2D_cylinder_shellLayers(filename, chain_length, shell_radius, shell_number, particle_length, particle_radius, particle_separation, connection_mode, connection_args, option_parameters)
    
    case "refine_arch_prism":
        ##
        ## Dipole data new can be pulled from simualtion, can rewrite get_bound_indices() and get_index_forces() 
        ## again if want to look at force on a bounded area of dipoles again
        ##
        ## --> MAKE SURE ALL THE DIPOLE STUFF IS FINE AND NOT CAUSING PROBLEMS
        ##
        ## GENERALISE FORCE FOR REFINE EMTHOD, AND FOR PLOTTER SO IS NICER TO USE
        ##

        # Save file
        filename = "SingleLaguerre"

        #-----------------------
        #-----------------------
        # Variable args
        dimensions      = np.array([200e-9, 200e-9, 200e-9])        # Bounding box for prism
        separations_list= [[0.0e-6, 0.0, 0.0]]   #[[i*0.01*1.0e-6, 0.0, 0.0] for i in range(100)]
        particle_sizes  = np.linspace(0.07e-6, 0.1e-6, 25)#np.linspace(0.06125e-6, 0.25e-6, 10)      # Radius or half-width
        dipole_sizes    = [30e-9, 40e-9]#[30e-9, 40e-9]#np.linspace(40e-9, 60e-9, 15)#[30e-9, 40e-9, 50e-9]
        deflections     = [0, 10e-7]                                  # Of centre in micrometres (also is deflection to centre of rod, not underside)
        #object_offsets  = [[-dimensions[0]/2.0, -dimensions[1]/2.0, -dimensions[2]/2.0]]#[[-dimensions[0]/2.0, -dimensions[1]/2.0, 0e-6]]  # Offset the whole object
        object_offsets  = [[0.0e-6, 0.0, 0.0]]
        force_measure_point = [0.0, 0.0, 0.0] # NOTE; This is the position measured at AFTER all shifts applied (e.g. measure at Dimensions[0]/2.0 would be considering the end of the rod, NOT the centre)
        particle_shapes = ["cube"]
        indep_vector_component = 0          # Which component to plot when dealing with vector quantities to plot (Often not used)
        force_filter=["Fx", "Fy", "Fmag"]     # options are ["Fmag","Fx", "Fy", "Fz", "Fpoint", "Fpoint_perDip", "F_T"] 
        indep_var = "particle_sizes" #"dipole_sizes"    #"particle_sizes"
        beam_type = "GAUSS_CSP"  #LAGUERRE
        place_regime = "spaced"   # Format to place particles within the overall rod; "squish", "spaced", ...
        prism_type   = "rect"   # Prism generation to use; "circle", "rect", ...
        prism_args   = [dimensions[1]/2.0, dimensions[2]/2.0]   #[dimensions[1]/2.0]      #"circle" => [radius], "rect" => [half-Y dim, half-Z dim]

        option_parameters = Generate_yaml.fill_yaml_options({
            "time_step": 1e-4,
            "force_terms": ["optical"],
            "show_output": False,
            "include_dipole_forces": False,
        })
        #-----------------------
        #-----------------------

        variables_list = {
            "indep_var": indep_var, # Must be one of the other keys: dipole_sizes, separations_list, particle_sizes, particle_shapes, deflections
            "dipole_sizes": dipole_sizes,
            "separations_list": separations_list,
            "particle_sizes": particle_sizes,
            "particle_shapes": particle_shapes,
            "object_offsets": object_offsets,
            "deflections": deflections
        }
        # Only used for when indep var is a vector (e.g.object_offsets): Set what component to plot against
        indep_name = variables_list["indep_var"]
        if indep_name == "separations_list": indep_vector_component = 0
        elif indep_name == "object_offsets": indep_vector_component = 2
        else: indep_vector_component = 0

        # Run
        parameter_text, data_set, data_set_params, particle_nums_set, dpp_nums_set = simulations_refine_arch_prism(
            dimensions, 
            variables_list,
            separations_list, 
            particle_sizes, 
            dipole_sizes, 
            deflections, 
            object_offsets, 
            particle_shapes, 
            place_regime,
            prism_type,
            prism_args,
            beam_type,
            option_parameters,
            force_measure_point=force_measure_point,
            indep_vector_component=indep_vector_component
        )
        
        # Format output and make legend/title strings
        titlestrbase, legend_params = get_titlelegend(variables_list, indep_name, "", dimensions)
        data_set, datalabel_set, filtered_i = filter_data_set(force_filter, data_set, data_set_params, legend_params, indep_name, N=7)
        linestyle_set, datacolor_set = get_colourline(datalabel_set, legend_params, variables_list, linestyle_var=None, cgrad=lambda x: (1/4+3/4*x, x/3, 1-x))

        xAxis_varname, xAxis_units = display_var(indep_name)
        graphlabel_set = {"title":"Forces"+titlestrbase, "xAxis":f"{xAxis_varname} {xAxis_units}", "yAxis":"Force /N"} 
        Display.plot_multi_data(data_set, datalabel_set, graphlabel_set=graphlabel_set, linestyle_set=linestyle_set, datacolor_set=datacolor_set)

        # Plot particle number and dipoles per particle against the independent variable.
        pd_legend_labels = make_param_strs(data_set_params, legend_params, indep_name)
        particlelabel_set = {"title":"Particle number"+titlestrbase, "xAxis":f"{display_var(indep_name)[0]} {display_var(indep_name)[1]}", "yAxis":"Particle number"}
        Display.plot_multi_data(particle_nums_set, pd_legend_labels, graphlabel_set=particlelabel_set, linestyle_set=linestyle_set[::len(force_filter)], datacolor_set=datacolor_set[::len(force_filter)]) # jumps of len(force_filter)
        dipolelabel_set = {"title":"Dipoles per particle"+titlestrbase, "xAxis":f"{display_var(indep_name)[0]} {display_var(indep_name)[1]}", "yAxis":"Dipoles per particle"}
        Display.plot_multi_data(dpp_nums_set, pd_legend_labels, graphlabel_set=dipolelabel_set, linestyle_set=linestyle_set[::len(force_filter)], datacolor_set=datacolor_set[::len(force_filter)]) 


    # case "refine_sphere_model":
    #     #
    #     # Consider a spherical mesh modelled with primitive objects such as spheres or cubes
    #     # Allows for plots to be found for these situations with several varying parameters at once
    #     #

    #     # Save file
    #     filename = "SingleLaguerre"

    #     #-----------------------
    #     #-----------------------
    #     # Variable args

    #     #
    #     # Cube of cubes & cube of spheres tending to force of perfect system (1particle cube, inf cube dipoles)
    #     #
    #     # show_output     = False
    #     # dimension       = 200e-9    # Radius of the total spherical mesh
    #     # separations_list= [[0.0e-6, 0.0, 0.0]]   #[[i*0.01*1.0e-6, 0.0, 0.0] for i in range(100)]
    #     # particle_sizes  = [0.2e-7] # Radius or half-width
    #     # dipole_sizes    = np.linspace(10e-9, 20e-9, 30) # np.linspace(10e-9, 50e-9, 30)
    #     # object_offsets  = [[1.0e-6, 0.0, 0.0e-6]]       # Offset the whole object
    #     # force_measure_point = [1.15e-6, 0.0, 0.0]       # NOTE; This is the position measured at AFTER all shifts applied (e.g. measure at Dimensions[0]/2.0 would be considering the end of the rod, NOT the centre)
    #     # force_terms     = ["optical"]
    #     # particle_shapes = ["cube"]
    #     # indep_vector_component = 0          # Which component to plot when dealing with vector quantities to plot (Often not used)
    #     # force_filter=["Fx", "Fy", "Fmag"]     # options are ["Fmag","Fx", "Fy", "Fz", "Fpoint", "Fpoint_perDip", "F_T"] 
    #     # indep_var = "dipole_sizes"    #"dipole_sizes"    #"particle_sizes"
    #     # beam_type = "LAGUERRE"          #"GAUSS_CSP"
    #     # place_regime = "squish"             # Format to place particles within the overall rod; "squish", "spaced", ...
    #     # include_dipole_forces = False
    #     # linestyle_var = "dipole_sizes"


    #     #
    #     # Force on cube as it moves in a Bessel beam --> Comparing to single dipole particle with RR, LDR or CM
    #     #
    #     show_output     = False
    #     dimension       = 200e-9    # Full width of sphere/cube
    #     separations_list= [[0.0e-6, 0.0, 0.0]]
    #     particle_sizes  = [100e-9] # Radius or half-width
    #     dipole_sizes    = np.linspace(9e-9, 100e-9, 200)#[12.5e-9, 25e-9, 50e-9, 100e-9]
    #     object_offsets  = [[1.0e-6, 0.0, 0.0]]#[[i*0.06e-6, 0.0, 0.0] for i in range(25)]       # Offset the whole object
    #     force_measure_point = [1.15e-6, 0.0, 0.0]       # NOTE; This is the position measured at AFTER all shifts applied (e.g. measure at Dimensions[0]/2.0 would be considering the end of the rod, NOT the centre)
    #     force_terms     = ["optical"]
    #     particle_shapes = ["cube","sphere"]
    #     indep_vector_component = 0          # Which component to plot when dealing with vector quantities to plot (Often not used)
    #     force_filter=["Fz"]     # options are ["Fmag","Fx", "Fy", "Fz", "Fpoint", "Fpoint_perDip", "F_T"] 
    #     indep_var = "dipole_sizes"#"object_offsets"
    #     beam_type = "BESSEL" 
    #     place_regime = "squish"             # Format to place particles within the overall rod; "squish", "spaced", ...
    #     include_dipole_forces = False
    #     linestyle_var = None#"dipole_sizes"
    #     polarisability_type = "CM"

    #     #
    #     # Testing plots to ensure working correctly
    #     #
    #     # show_output     = False
    #     # dimension       = 200e-9    # Full width of sphere/cube
    #     # separations_list= [[0.0e-6, 0.0, 0.0]]   #[[i*0.01*1.0e-6, 0.0, 0.0] for i in range(100)]
    #     # particle_sizes  = [1e-7, 100e-9]#[0.2e-7, 0.6e-7, 1.0e-7]#np.linspace(0.2e-7, 1.0e-7, 20)#np.linspace(0.04e-6, 0.1e-6, 25)#np.linspace(0.06125e-6, 0.25e-6, 10)      # Radius or half-width
    #     # dipole_sizes    = np.linspace(10e-9, 50e-9, 4)#np.linspace(10e-9, 60e-9, 20)#[30e-9, 40e-9, 50e-9]
    #     # object_offsets  = [[1.0e-6, 0.0, 0.0e-6]]      # Offset the whole object
    #     # particle_shapes = ["cube"]
    #     # force_terms     = ["optical"]
    #     # indep_vector_component = 0
    #     # force_measure_point = [1.15e-6, 0.0, 0.0]
    #     # force_filter= ["Fx"]     # options are ["Fmag","Fx", "Fy", "Fz", "Fpoint", "Fpoint_perDip", "F_T"] 
    #     # indep_var = "dipole_sizes"    #"dipole_sizes"    #"particle_sizes"
    #     # beam_type = "LAGUERRE"          #"GAUSS_CSP"
    #     # place_regime = "squish"             # Format to place particles within the overall rod; "squish", "spaced", ...
    #     # include_dipole_forces = False
    #     # linestyle_var = None # (it will pick the best) "particle_sizes"
    #     # polarisability_type = "RR"

    #     #-----------------------
    #     #-----------------------

    #     variables_list = {
    #         "indep_var": indep_var, # Must be one of the other keys: dipole_sizes, separations_list, particle_sizes, particle_shapes, deflections
    #         "dipole_sizes": dipole_sizes,
    #         "separations_list": separations_list,
    #         "particle_sizes": particle_sizes,
    #         "particle_shapes": particle_shapes,
    #         "object_offsets": object_offsets
    #     }

    #     indep_name = variables_list["indep_var"]

    #     # Run
    #     parameter_text, data_set, data_set_params, particle_nums_set, dpp_nums_set = simulations_refine(
    #         dimension, 
    #         variables_list,
    #         separations_list, 
    #         particle_sizes, 
    #         dipole_sizes, 
    #         object_offsets, 
    #         force_terms, 
    #         particle_shapes, 
    #         place_regime,
    #         beam_type,
    #         include_dipole_forces,
    #         polarisability_type=polarisability_type,
    #         force_measure_point=force_measure_point,
    #         show_output=show_output,
    #         indep_vector_component=indep_vector_component,
    #         isObjectCube=True
    #     )

    #     # Format output and make legend/title strings
    #     titlestrbase, legend_params = get_titlelegend(variables_list, indep_name, "all", [dimension, dimension, dimension])
    #     data_set, datalabel_set, filtered_i = filter_data_set(force_filter, data_set, data_set_params, legend_params, indep_name, N=7)
    #     linestyle_set, datacolor_set = get_colourline(datalabel_set, legend_params, variables_list, linestyle_var=linestyle_var, cgrad=lambda x: (1/4+3/4*x, x/3, 1-x))

    #     xAxis_varname, xAxis_units = display_var(indep_name)
    #     graphlabel_set = {"title":"Forces"+titlestrbase, "xAxis":f"{xAxis_varname} {xAxis_units}", "yAxis":"Force /N"} 
    #     Display.plot_multi_data(data_set, datalabel_set, graphlabel_set=graphlabel_set, linestyle_set=linestyle_set, datacolor_set=datacolor_set)

    #     # Plot particle number and dipoles per particle against the independent variable.
    #     # pd_legend_labels = make_param_strs(data_set_params, legend_params, indep_name)
    #     # particlelabel_set = {"title":"Particle number"+titlestrbase, "xAxis":f"{display_var(indep_name)[0]} {display_var(indep_name)[1]}", "yAxis":"Particle number"}
    #     # Display.plot_multi_data(particle_nums_set, pd_legend_labels, graphlabel_set=particlelabel_set, linestyle_set=linestyle_set[::len(force_filter)], datacolor_set=datacolor_set[::len(force_filter)]) # jumps of len(force_filter)
    #     # dipolelabel_set = {"title":"Dipoles per particle"+titlestrbase, "xAxis":f"{display_var(indep_name)[0]} {display_var(indep_name)[1]}", "yAxis":"Dipoles per particle"}
    #     # Display.plot_multi_data(dpp_nums_set, pd_legend_labels, graphlabel_set=dipolelabel_set, linestyle_set=linestyle_set[::len(force_filter)], datacolor_set=datacolor_set[::len(force_filter)]) 





    # case "refine_cuboid_general":
    #     #====================================================================================
    #     # Save file
    #     filename = "SingleLaguerre"
    #     # Args
    #     dimensions  =  [0.8e-6]*3 #[0.8e-6, 0.8e-6, 0.8e-6]  # Full Dimensions of each side of the cuboid
    #     force_terms=["optical"]                # ["optical", "spring", "bending", "buckingham"]
    #     force_filter=["Fmag", "Fy", "Fx"]                    # Options are ["Fmag","Fx", "Fy", "Fz"]
    #     indep_name = "dipole_sizes"          # Options: dipole_sizes, separations_list, particle_sizes, particle_shapes, object_offsets
    #     particle_selection = "all"          # Options are "all", "central" or a list of ints (manual)
    #     # Iterables
    #     separations_list = [[0,0,0]] # Separation in each axis of the cuboid, as a total separation (e.g. more particles => smaller individual separation between each)
        
    #     ### DIPS FRACTIONS OF PARTICLE SIZE
    #     # particle_sizes = np.linspace(0.16e-6, 0.2e-6, 1)  
    #     # dipole_sizes = [2*particle_sizes[0]/n - 1e-12 for n in [1,2,3,4,5,6,7,8]]
        
    #     ### NORMAL
    #     # separations_list = [[s, s, s] for s in np.linspace(0, 0.5e-6, 10)]  # Separation in each axis of the cuboid, as a total separation (e.g. more particles => smaller individual separation between each)
    #     dipole_sizes = np.linspace(40e-9, 100e-9, 40)         
    #     particle_sizes = np.linspace(0.1e-6, 0.3e-6, 3)

    #     particle_shapes = ["cube"] 
    #     object_offsets = [[1e-6, 0e-6, 0e-6]]
    #     #====================================================================================
        
    #     # Run
    #     variables_list = {"indep_var": indep_name, "dipole_sizes": dipole_sizes,"separations_list": separations_list,"particle_sizes": particle_sizes,"particle_shapes": particle_shapes,"object_offsets": object_offsets}
    #     parameter_text, data_set, data_set_params, particle_nums_set, dpp_nums_set = simulations_refine_general(dimensions, variables_list, force_terms, show_output=False , indep_vector_component=0, particle_selection=particle_selection) # indep_vector_component only used for when indep var is a vector (e.g.object_offsets): Set what component to plot against
        
    #     # Format output then plot graph
    #     titlestrbase, legend_params = get_titlelegend(variables_list, indep_name, particle_selection, dimensions)
    #     data_set, datalabel_set, filtered_i = filter_data_set(force_filter, data_set, data_set_params, legend_params, indep_name)
    #     linestyle_set, datacolor_set = get_colourline(datalabel_set, legend_params, variables_list, linestyle_var=None, cgrad=lambda x: (1/4+3/4*x, x/3, 1-x))
    #     graphlabel_set = {"title":"Forces"+titlestrbase, "xAxis":f"{display_var(indep_name)[0]} {display_var(indep_name)[1]}", "yAxis":"Force /N"} 
    #     Display.plot_multi_data(data_set, datalabel_set, graphlabel_set=graphlabel_set, linestyle_set=linestyle_set, datacolor_set=datacolor_set) 
        
    #     # Plot particle number and dipoles per particle against the independent variable.
    #     pd_legend_labels = make_param_strs(data_set_params, legend_params, indep_name)
    #     particlelabel_set = {"title":"Particle number"+titlestrbase, "xAxis":f"{display_var(indep_name)[0]} {display_var(indep_name)[1]}", "yAxis":"Particle number"}
    #     Display.plot_multi_data(particle_nums_set, pd_legend_labels, graphlabel_set=particlelabel_set, linestyle_set=linestyle_set[::len(force_filter)], datacolor_set=datacolor_set[::len(force_filter)]) # jumps of len(force_filter)
    #     dipolelabel_set = {"title":"Dipoles per particle"+titlestrbase, "xAxis":f"{display_var(indep_name)[0]} {display_var(indep_name)[1]}", "yAxis":"Dipoles per particle"}
    #     Display.plot_multi_data(dpp_nums_set, pd_legend_labels, graphlabel_set=dipolelabel_set, linestyle_set=linestyle_set[::len(force_filter)], datacolor_set=datacolor_set[::len(force_filter)]) 

    #     # Can be used to filter the dipole sizes: (Should change to filter for maxs/mins)
    #     # volumes = calc_SphereOrCube_volumes(dipole_sizes, particle_sizes[0], isSphere=False) # particle_size is 0th!
    #     # filtered_dipole_sizes, filtered_volumes, error = filter_dipole_sizes(volumes, dipole_sizes, num=10, target_volume=None)
    #     # Display.plot_volumes_against_dipoleSize(dipole_sizes, volumes, best_sizes=filtered_dipole_sizes, best_volumes=filtered_volumes)


    case "single_dipole_exp":
        #
        # Considers the force on single dipoles / sets of single dipoles using different polarisability prescriptions
        # and in different fields
        #

        # Save file
        filename = "SingleLaguerre"

        #-----------------------
        #-----------------------
        # Variable args

        beam_type = "BESSEL"          # Which beam to use
        object_offset = [0.0, 0.0, 0.0]
        test_type = "7shell"  # Particle setup to test
        linestyle_set = None
        rotation = None#"180 0.0 0.0"

        option_parameters = Generate_yaml.fill_yaml_options({
            "show_output": False,
            "force_terms": ["optical"],
            "polarisability_type": "RR",    # Which polarisability to test
            "time_step": 1e-4,
            "dipole_radius": 100e-9, # Half-width/radius of dipole
            "frames": 1,
        })

        # Test parameters
        match test_type:
            case "single":
                test_args = [0.0, 1.5e-6, 50]  # [offset_lower, offset_upper, offset_number]

                # Read forces on the only dipole present
                read_parameters = [
                    {"type":"F", "particle":0, "subtype":0},
                    {"type":"F", "particle":0, "subtype":1},
                    {"type":"F", "particle":0, "subtype":2}
                ]

            case "7shell":
                test_args = [0.0, 1.5e-6, 50]  # [offset_lower, offset_upper, offset_number]

                # Read forces on the only dipole present
                read_parameters = []
                for p in range(7):
                    read_parameters.append({"type":"F", "particle":p, "subtype":0})
                    read_parameters.append({"type":"F", "particle":p, "subtype":1})
                    read_parameters.append({"type":"F", "particle":p, "subtype":2})
            
            case "7shell_difference":
                test_args = [0.0, 1.5e-6, 50, ["CM", "RR"]]  # [offset_lower, offset_upper, offset_number]

                # Read forces on the only dipole present
                read_parameters = []
                for p in range(7):
                    read_parameters.append({"type":"F", "particle":p, "subtype":0})
                    read_parameters.append({"type":"F", "particle":p, "subtype":1})
                    read_parameters.append({"type":"F", "particle":p, "subtype":2})

                linestyle_set=["dotted","dotted","dotted", "dashed","dashed","dashed", "solid","solid","solid"]
            
            case "multi_separated":
                dipole_size = option_parameters["dipole_radius"]
                test_args = [15, dipole_size*2.0, dipole_size*10.0, 50]  # [particle_number, lower_separation, upper_separation, separation_number]

                # Read forces from all dipoles
                read_parameters=[]
                for p in range(test_args[0]):
                    read_parameters.append({"type":"F", "particle":p, "subtype":0})
                    read_parameters.append({"type":"F", "particle":p, "subtype":1})
                    read_parameters.append({"type":"F", "particle":p, "subtype":2})

            case _:
                test_args=[]
                read_parameters=[]


        # Run simulation
        data_set, data_set_labels, graphlabel_set = simulations_single_dipole(filename, read_parameters, beam_type, test_type, test_args, object_offset, option_parameters, rotation=rotation)
        Display.plot_multi_data(data_set, data_set_labels, graphlabel_set=graphlabel_set, linestyle_set=linestyle_set)

    case "sphere_spheredisc_model":
        #
        # Consider a disc of spherical particles
        # Leads into modelling a sphere through a series of discs
        #

        # Save file
        filename = "SingleLaguerre"
        if(os.path.exists(filename+".xlsx")):
            os.remove(filename+".xlsx")
        if(os.path.exists(filename+"_dipoles.xlsx")):
            os.remove(filename+"_dipoles.xlsx")

        #-----------------------
        #-----------------------
        # Variable args

        #
        # Sphere / Disc for measurements on far right particle
        #
        # show_output     = False
        # disc_radius     = 1.09e-6                   # Radius of full disc
        # particle_sizes  = [200e-9]                  # Radius of spherical particles used to model the disc
        # separation_min = 0.0e-6
        # separation_max = 1.4e-6#1.4e-6
        # separation_iter = 20
        # separations_list= [[separation_min+i*( (separation_max-separation_min)/separation_iter ), 0.0, 0.0e-6] for i in range(separation_iter)]     # NOTE; Currently just uses separation[0] as between particles in a layer, and separation[1] as between layers in a disc, and separation[2] as between discs in a sphere
        # dipole_sizes    = [75e-9]#np.linspace(80e-9, 100e-9, 20)
        # object_offsets  = [[0.0e-6, 0.0, 1.0e-6]]      # Offset the whole object
        # dda_forces_returned     = ["optical"]
        # particle_shapes         = ["sphere"]
        # indep_vector_component  = 0              # Which component to plot when dealing with vector quantities to plot (Often not used)
        # indep_var               = "separations_list"
        # beam_type               = "LAGUERRE" 
        # include_dipole_forces   = False
        # linestyle_var           = None
        # polarisability_type     = "RR"
        # mode        = "sphere"     #"disc", "sphere"
        # frames      = 1
        # time_step   = 1e-4
        # # NOTE; The following lists must be the same length.
        # forces_output= ["Fx", "Fy"]     # options are ["Fmag","Fx", "Fy", "Fz", "Cmag","Cx", "Cy", "Cz",] 
        # particle_selections = [ [0], [0] ]#[ [[disc_radius, 0.0, 0.0]], [[disc_radius, 0.0, 0.0]] ]#[[[0.0,0.0,0.0], [1.0,0.0,0.0]]] # list of "all", [i,j,k...], [[rx,ry,rz]...]

        #
        # Measure torque experienced by entire shape (sphere/disc/ring)
        #
        disc_radius     = [1.14e-6]    #1.14e-6 #1.09e-6                   # Radius of full disc
        particle_sizes  = [100e-9]                  # Radius of spherical particles used to model the disc
        separation_min = 0.0e-6
        separation_max = 1.4e-6#1.4e-6
        separation_iter = 20
        separations_list= [[separation_min+i*( (separation_max-separation_min)/separation_iter ), 0.0, 0.0e-6] for i in range(separation_iter)]     # NOTE; Currently just uses separation[0] as between particles in a layer, and separation[1] as between layers in a disc, and separation[2] as between discs in a sphere
        dipole_sizes    = [75e-9] #[40e-9, 50e-9, 60e-9, 70e-9]
        object_offsets  = [[0.0e-6, 0.0, 1.0e-6]]      # Offset the whole object
        particle_shapes         = ["sphere"]
        indep_vector_component  = 0              # Which component to plot when dealing with vector quantities to plot (Often not used)
        indep_var               = "separations_list"
        beam_type               = "LAGUERRE" 
        linestyle_var           = None
        mode        = "sphere"     #"disc", "sphere"
        frames      = 1
        time_step   = 1e-4
        materials = ["FusedSilica"]
        fix_to_ring = False
        # NOTE; The following lists must be the same length.
        forces_output= ["Fx", "Fy"]     # options are ["Fmag","Fx", "Fy", "Fz", "Cmag","Cx", "Cy", "Cz",] 
        particle_selections = [[0], [0]]#[ [[disc_radius, 0.0, 0.0]], [[disc_radius, 0.0, 0.0]] ]#[[[0.0,0.0,0.0], [1.0,0.0,0.0]]] # list of "all", [i,j,k...], [[rx,ry,rz]...]
        # forces_output= ["Fx", "Fy"]
        # particle_selections = [ [0],[0] ]

        option_parameters = Generate_yaml.fill_yaml_options({
            "show_output": False,
            "show_stress": False,
            "force_terms": ["optical"],
            "polarisability_type": "RR",
        })
        # Only make dipoles file if torque about given centre are needed.
        if "Tmag" in forces_output or "Tx" in forces_output or "Ty" in forces_output or "Tz" in forces_output: option_parameters["include_dipole_forces"] = True
        else: option_parameters["include_dipole_forces"] = False

        #-----------------------
        #-----------------------

        variables_list = {
            "indep_var": indep_var, # Must be one of the other keys: dipole_sizes, separations_list, particle_sizes, particle_shapes, deflections
            "dipole_sizes": dipole_sizes,
            "separations_list": separations_list,
            "particle_sizes": particle_sizes,
            "particle_shapes": particle_shapes,
            "object_offsets": object_offsets,
            "dimensions": disc_radius,
            "materials": materials
        }
        # Only used for when indep var is a vector (e.g.object_offsets): Set what component to plot against
        indep_name = variables_list["indep_var"]
        
        partial_yaml_func = partial(Generate_yaml.make_yaml_spheredisc_model, filename=filename, option_parameters=option_parameters, mode=mode, beam=beam_type, fix_to_ring=fix_to_ring)
        data_set, data_set_params, particle_nums_set, dpp_nums_set = simulations_refine_all(
            filename,
            variables_list, 
            partial_yaml_func, 
            forces_output, 
            particle_selections, 
            indep_vector_component=indep_vector_component, 
            torque_centre=[0,0,0]
        )

        # Format output and make legend/title strings
        title_str, datalabel_set, linestyle_set, datacolor_set, graphlabel_set = get_title_label_line_colour(variables_list, data_set_params, forces_output, particle_selections, indep_name, linestyle_var=linestyle_var, cgrad=lambda x: (1/4+3/4*x, x/3, 1-x))
      
        graphlabel_set["title"] += f", mesh_shape={mode}, fix_ring={fix_to_ring}"
        Display.plot_multi_data(data_set, datalabel_set, graphlabel_set=graphlabel_set, linestyle_set=linestyle_set, datacolor_set=datacolor_set)

        # Plot particle number and dipoles per particle against the independent variable.
        particlelabel_set = {"title":"Particle number", "xAxis":f"{display_var(indep_name)[0]} {display_var(indep_name)[1]}", "yAxis":"Particle number"}
        Display.plot_multi_data(particle_nums_set, datalabel_set[::len(forces_output)], graphlabel_set=particlelabel_set, linestyle_set=linestyle_set[::len(forces_output)], datacolor_set=datacolor_set[::len(forces_output)]) 
        dipolelabel_set = {"title":"Dipoles per particle", "xAxis":f"{display_var(indep_name)[0]} {display_var(indep_name)[1]}", "yAxis":"Dipoles per particle"}
        Display.plot_multi_data(dpp_nums_set, datalabel_set[::len(forces_output)], graphlabel_set=dipolelabel_set, linestyle_set=linestyle_set[::len(forces_output)], datacolor_set=datacolor_set[::len(forces_output)]) 


    case "surface_stresses":
        #
        # Simple 3D plot of the forces on each dipole relative to the average force to show how it would deform relative to the centre of mass.
        # Uses quiver_setting=2 in Display.animate_system3d
        #
        #====================================================================================
        # Save file
        filename = "SingleLaguerre"
        # force_terms=["optical"]              # ["optical", "spring", "bending", "buckingham"]
        # Args
        dimensions  =  [2.0e-6]*3            # Total dimension of the object, NOTE: only 0th value used by a sphere object
        object_shape = "sphere" # cube or sphere
        separations = [0,0,0]
        dipole_size = 40e-9
        num_particles_in_diameter = 15
        particle_size = dimensions[0]/(2*num_particles_in_diameter) # (assumes dimensions are isotropic)
        dipole_size=particle_size  # Done to fix the dipoles to reduce computation time
        # particle_size = 0.15e-6 # NOTE *2 for diameter
        object_offset = [0.5e-6, 0e-6, 0e-6]
        # show_output = False
        option_parameters = Generate_yaml.fill_yaml_options({
            "show_output": False,
            "show_stress": True,
            "force_terms": ["optical"],
            "beam_planes": [['z',0]],
            "dipole_size": dipole_size,
            "quiver_setting": 0,
            "max_size": 2e-6,
        })
        #====================================================================================
        
        # Run
        if particle_size < dipole_size: 
            dipole_size = particle_size
            print(f"WARNING: particle size smaller than dipoles size, setting dipole size to particle size ({particle_size})")
        print(f"\nSimulation for 1 frame of a {object_shape} object with cube particles.\nDimensions = {dimensions}, dipole size = {dipole_size}m, particle size = {particle_size:.3e}m, separations = {separations}m, object offset = {object_offset}m\n")
        positions, forces, particle_num, dpp_num = simulation_single_cubeSphere(filename, dimensions, object_shape, separations, object_offset, particle_size, "cube", option_parameters, beam="LAGUERRE")
    
    case "force_torque_sim":
        #
        # Cube or sphere object in an LG beam, all combinations of variables can be iterated over
        # Can be used to simulate a single large sphere in the centre of a LG to test the torque.
        #

        # Save file
        filename = "SingleLaguerre"

        #
        # Run cube generation out of sub-cubes / sub-spheres
        # Showing that particle + dipole refinement results in accuracy results (to a numerically exact TRUE result, for cube when 2.0*particle_size=dimension)
        #
        # dimensions      = [400e-9]                       # Full width of sphere/cube
        # separations_list= [[0.0e-6, 0.0, 0.0]]           # For each axis, sum of the separations between each particle
        # particle_sizes  = [dimensions[0]/12, dimensions[0]/6, dimensions[0]/2]              # Single particle
        # dipole_sizes    = np.linspace(15e-9, 200e-9, 100)  
        # object_offsets  = [[1.13e-6, 0.0, 0.0e-6]]          # Offset the whole object
        # particle_shapes = ["cube", "sphere"]
        # materials = ["FusedSilica"] # , "FusedSilica01"
        # indep_var = "dipole_sizes"                       # Must be one of the keys in variables_list, excluding "indep_var".
        # beam_type = "LAGUERRE"     
        # object_shape = "cube"   
        # torque_centre = [0,0,0]
        # place_regime = "squish"                          # Format to place particles within the overall rod; "squish", "spaced", ...
        # linestyle_var = None # (it will pick the best if None) strings: dipole_sizes, particle_sizes, particle_shapes, forces_output, particle_selections, deflections, separations_list
        # # The following lists must be the same length.
        # forces_output= ["Fmag"] #["Tz", "Cz"]     # options are ["Fmag","Fx", "Fy", "Fz", "Cmag","Cx", "Cy", "Cz",] 
        # particle_selections = ["all"]   #["all", "all"] # list of "all", [i,j,k...], [[rx,ry,rz]...] - (get all particles, specific indices, or indices close to a position; then forces summed over all particles in the list)

        #
        # 2nd data set for sub-cube, sub-sphere test
        #
        dimensions      = [400e-9]                       # Full width of sphere/cube
        separations_list= [[0.0e-6, 0.0, 0.0]]           # For each axis, sum of the separations between each particle
        particle_sizes  = [40e-9, 100e-9, 200e-9]              # Single particle
        dipole_sizes    = np.linspace(20e-9, 200e-9, 200)  
        object_offsets  = [[1.13e-6, 0.0, 0.0e-6]]          # Offset the whole object
        particle_shapes = ["sphere", "cube"]
        materials = ["FusedSilica"] # , "FusedSilica01"
        indep_var = "dipole_sizes"                       # Must be one of the keys in variables_list, excluding "indep_var".
        beam_type = "LAGUERRE"     
        object_shape = "cube"   
        torque_centre = [0,0,0]
        place_regime = "squish"                          # Format to place particles within the overall rod; "squish", "spaced", ...
        linestyle_var = None # (it will pick the best if None) strings: dipole_sizes, particle_sizes, particle_shapes, forces_output, particle_selections, deflections, separations_list
        # The following lists must be the same length.
        forces_output= ["Fmag"] #["Tz", "Cz"]     # options are ["Fmag","Fx", "Fy", "Fz", "Cmag","Cx", "Cy", "Cz",] 
        particle_selections = ["all"]   #["all", "all"] # list of "all", [i,j,k...], [[rx,ry,rz]...] - (get all particles, specific indices, or indices close to a position; then forces summed over all particles in the list)

        option_parameters = Generate_yaml.fill_yaml_options({
            "show_output": False,
            "show_stress": False,
            "force_terms": ["optical"],
        })

        # Only make dipoles file if torque about given centre are needed.
        if "Tmag" in forces_output or "Tx" in forces_output or "Ty" in forces_output or "Tz" in forces_output: option_parameters["include_dipole_forces"] = True
        else: option_parameters["include_dipole_forces"] = False

        # Make YAML function
        # partial_yaml_func args left to call: dimension, separations, particle_size, dipole_size, object_offset, particle_shape
        match object_shape:
            case "cube": isObjectCube = True
            case "sphere": isObjectCube = False
            case _: isObjectCube = False; print("WARNING, object shape set to sphere")
        partial_yaml_func = partial(Generate_yaml.make_yaml_refine_sphere, makeCube=isObjectCube, filename=filename, place_regime=place_regime, beam=beam_type, option_parameters=option_parameters)

        #-----------------------
        #-----------------------

        variables_list = {
            "indep_var": indep_var,
            "dipole_sizes": dipole_sizes,
            "separations_list": separations_list,
            "particle_sizes": particle_sizes,
            "particle_shapes": particle_shapes,
            "object_offsets": object_offsets,
            "dimensions": dimensions,
            "materials": materials
        }

        data_set, data_set_params, particle_nums_set, dpp_nums_set = simulations_refine_all(filename, variables_list, partial_yaml_func, forces_output, particle_selections, indep_vector_component=2, torque_centre=torque_centre)

        title_str, datalabel_set, linestyle_set, datacolor_set, graphlabel_set = get_title_label_line_colour(variables_list, data_set_params, forces_output, particle_selections, indep_var, linestyle_var=linestyle_var, cgrad=lambda x: (1/4+3/4*x, x/3, 1-x))

        Display.plot_multi_data(data_set, datalabel_set, graphlabel_set=graphlabel_set, linestyle_set=linestyle_set, datacolor_set=datacolor_set)


    case "stretcher_with_springs":
        #
        # Simulation of a sphere stretched between two oppsing Gaussian beams
        #
        filename = "Optical_stretcher"
        param_set = "1" # "1", "guck"

        if param_set == "1":
            num_particles = 120   # 40, 72, 160
            sphere_radius = 1.3e-6
            particle_radius = 0.1e-6
            connection_mode = "num"
            connection_args = "5"
            E0 = 14e6 #1.5e7
            w0 = 1.0
            translation = "0.0 0.0 4.5e-6"

            option_parameters = Generate_yaml.fill_yaml_options({
                "show_output": True,
                "show_stress": False,
                "force_terms": ["optical", "spring", "bending"], #, "buckingham"
                "constants": {"bending": 1e-19}, # 5e-20  # 0.5e-18 # 5e-19
                "stiffness_spec": {"type":"", "default_value":5e-6}, #5e-8  # 5e-7
                "dipole_radius": 100e-9,
                "frames": 21,
                "time_step": 10e-5, 
                "max_size": 5e-6,
                "resolution": 401,
                "quiver_setting": 0,
                "wavelength": 1.0e-6,
                "max_size": 1.8e-6,
                "beam_planes": [["y", 0]], #  [["z", 0], ["x", 0]]  [["z", 0]]
                "beam_alpha": 0.4,
            })
        elif param_set == "guck":
            num_particles = 160   # 40, 72, 84, 120 160
            sphere_radius = 3.36e-6
            particle_radius = 0.1e-6
            connection_mode = "num"
            connection_args = "5"
            E0 = 14e6 #1.5e7
            w0 = 5
            translation = "0.0 0.0 30e-6"

            option_parameters = Generate_yaml.fill_yaml_options({
                "force_terms": ["optical", "spring", "bending"], #, "buckingham"
                "constants": {"bending": 1e-19}, # 5e-20  # 0.5e-18 # 5e-19
                "stiffness_spec": {"type":"", "default_value":5e-6}, #5e-8  # 5e-7
                "dipole_radius": 100e-9,
                "time_step": 10e-5, 
                "wavelength": 785e-9,

                "show_output": True,
                "show_stress": False,
                "frames": 40,
                "max_size": 5e-6,
                "quiver_setting": 0,
                "resolution": 401,
                "beam_planes": [["y", 0]], #  [["z", 0], ["x", 0]]  [["z", 0]]
                "beam_alpha": 0.4,
            })

        print(f"\ntime step = {option_parameters['time_step']}, stiffness = {option_parameters['stiffness_spec']['default_value']}, bending = {option_parameters['constants']['bending']}, particle number = {num_particles}, dipole size = {option_parameters['dipole_radius']}, particle size = {particle_radius}, sphere object radius = {sphere_radius}, beam E0 = {E0:.2e}, beam width = {w0}, wavelength = {option_parameters['wavelength']}, translation = {translation}\n")
        
        Generate_yaml.make_yaml_stretcher_springs(filename, option_parameters, num_particles, sphere_radius, particle_radius, connection_mode, connection_args, E0, w0, translation)
        DM.main(filename)

    case "dynamic_stretcher_eccentricity":
        #
        # Dynamics simulation of a sphere stretched between two opposing Gaussian beams
        #
        filename = "Optical_stretcher"
        connection_mode = "num"
        connection_args = "5"
        yaxis_label = "Bounding box ratio" # "Eccentricity", "Height/width ratio", "Bounding box ratio"

        option_parameters = Generate_yaml.fill_yaml_options({
            "force_terms": ["optical", "spring", "bending"], #, "buckingham"
            "dipole_radius": 100e-9,
            "wavelength": 785e-9,

            "show_output": False,
            "show_stress": False,
            "frames": 2000,
            "frame_min": 1,
            "max_size": 5e-6,
            "quiver_setting": 0,
            "resolution": 401,
            "beam_planes": [["y", 0]], #  [["z", 0], ["x", 0]]  [["z", 0]]
            "beam_alpha": 0.4,
        })
        # MAIN
        variables_list = { # NOTE order of this is important
            "stiffness": [2.7e-6],  #6.5e-6
            "bending": [0.75e-19, 1e-19],
            "translation": ["0.0 0.0 130e-6"],
            "num_particles": [160], # 40, 72, 84, 100, 120, 160, 200
            "particle_radius": [0.1e-6], # adjust dipole size to match this.
            "E0": [14e6],
            "w0": [5],
            "time_step": [5e-5], # largest one used to calc actual frames, shorter ones only have more frames.
            "num_averaged": [1], # num min and max to average the positions of to get the eccentricity / ratio, this also acts as a repeat.
            "sphere_radius": [3.36e-6], # sphere radius from Guck's paper is 3.36e-6m
            "repeat": [i+1 for i in range(2)],
        }

        def pickle_write(dict, filename): 
            with open(filename, "wb") as f: pickle.dump(dict, f)
        def pickle_read(filename): 
            with open(filename, "rb") as f: return pickle.load(f)
        def pickle_merge(file_a, file_b): # append b into a
            dict_a = pickle_read(file_a)
            dict_b = pickle_read(file_b)
            for key, value in dict_b.items(): # assuming they have the same keys, and that they contain arrays that can be extended.
                np.append(dict_a[key], value)
            pickle_write(dict_a, file_a)
            

        should_recalculate = True # if data should be calculated, not read from a file.
        should_merge = False # if recalculating, option to extend existing data.

        store_name = "dynamic_stretcher_store"
        if should_recalculate:
            data_set, datalabel_set, graphlabel_set, pulled_data_set = dynamic_stretcher_vary(filename, variables_list, option_parameters, yaxis_label)
            
            # Store data
            data_dict = {"data_set":data_set, "datalabel_set":datalabel_set, "graphlabel_set":graphlabel_set, "pulled_data_set":pulled_data_set}
            unique_filename = f"{store_name}{np.random.randint(0, 10000000)}.p"
            # store in unique file so data not overwritten later. Comment this for fewer *.p files.
            pickle_write(data_dict, unique_filename)

            if should_merge:
                pickle_merge(f"{store_name}.p", unique_filename) # 
            else:
                pickle_write(data_dict, f"{store_name}.p")
        
        # else, don't calculate just read from file.
        else:
            data_dict = pickle_read(f"{store_name}.p")
            data_set, datalabel_set, graphlabel_set, pulled_data_set = data_dict["data_set"], data_dict["datalabel_set"], data_dict["graphlabel_set"], data_dict["pulled_data_set"]

        Display.plot_multi_data(np.array(data_set), datalabel_set, graphlabel_set=graphlabel_set)

        axes = ["z", "x"]
        data_set_moi, data_set_ideal = calculate_MoI(pulled_data_set, data_set, axes=axes)
        for axi in range(len(axes)):
            graphlabel_set["yAxis"] = f"MoI, axis {axes[axi]}"
            Display.plot_multi_data(data_set_moi[axi], datalabel_set, graphlabel_set=graphlabel_set)

        



    case "stretcher_springs_cubic_sphere":
        #
        # Simulation of a sphere mode of cubic particles placed in an optical stretcher (counter-propagating beams, Gaussian usually)
        # Stresses on the particles are then considered, and the placement of the particles changed to try and minimise these forces
        # Spring forces are also used to allow an equilibrium to be reached
        #

        def func_transform(coordinates, transform_factor, transform_type="linear", args={}):
            #
            # coordinates = [[x,y,z],...] positions of particles to be transformed
            # transform_factor=1.0 => no transform for linear map, >1.0 => stretching of Z axis, shrinking other XY, vice verse for <1.0
            #
            match transform_type:
                case "linear":
                    # Linear transform
                    transformed_coords_list = coordinates * [1/np.sqrt(transform_factor), 1/np.sqrt(transform_factor), transform_factor]
                    return transformed_coords_list

                case "radial_meridional":
                    # Radial and meridional transform given in "Guck, Jochen, et al. "The optical stretcher: a novel laser tool to micromanipulate cells." Biophysical journal 81.2 (2001): 767-784"
                    nu = 0.5    # Poisson ratio
                    Eh = 3.9e-5 # Young's Modulus*shell thickness -> experimental average used
                    stress0 = pow(transform_factor,3)*3.0   # Scales with transform factor; starts at 0 -> required deformation
                    transformed_coords_list = []
                    for coord in coordinates:
                        rho  = np.sqrt(pow(coord[0],2) + pow(coord[1],2) + pow(coord[2],2))
                        rho2 = rho*rho
                        phi = np.arctan2(coord[1], coord[0])
                        theta = np.pi/2 - np.arctan2(coord[2], rho)
                        radial_comp = ( (rho2*stress0)/(4.0*Eh) )*( (5.0+nu)*pow(np.cos(theta),2) - (1.0+nu) )  
                        meridional_comp = ( (rho2*stress0*(1.0+nu))/(2.0*Eh) )*( np.sin(theta)*np.cos(theta) )

                        coord_base   = np.array([coord[0], coord[1], coord[2]])
                        coord_radial = np.array([coord[0], coord[1], coord[2]])*radial_comp/rho
                        coord_theta  = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])*meridional_comp

                        transformed_coord_unrot = coord_base+coord_radial+coord_theta
                        transformed_coords_list.append(transformed_coord_unrot)

                    return np.array(transformed_coords_list)

                case "adjusted_linear":
                    # Linear transform with some alteration
                    # NOTE; This does NOT preserve volume, you could add this though by scaling the Z down by rad_factor^2, but this would likely cause more problems
                    #   --> Need a function with positive and negative integral contribution => Takes away and awards volume in XY plane without influencing Z plane
                    
                    #
                    # Original adjustment -> zero seen
                    #
                    # mesh_unstretched_radius = 1000e-9   ##### Hard coded to test if shape profile works -> Fits current test, see if works then implement fully ######
                    # influence = 0.6     # Acts to inflate the transformation in the XY plane
                    # transformed_coords_list = []
                    # for coord in coordinates:
                    #     rad_factor = 1.0 +influence*(transform_factor-1.0)*(np.sqrt( pow(coord[0],2) + pow(coord[1],2) ) / (mesh_unstretched_radius))
                    #     transformed_coords_list.append( [rad_factor*coord[0]/np.sqrt(transform_factor), rad_factor*coord[1]/np.sqrt(transform_factor), coord[2]*transform_factor] )

                    #
                    # Compressive adjustment
                    #
                    mesh_unstretched_radius = 1000e-9   ##### HARD CODED FOR NOW -> Fits current test, see if works then implement fully ######
                    influence = 0.25     # Acts to inflate the transformation in the XY plane
                    transformed_coords_list = []
                    for coord in coordinates:
                        rad_factor = 1.0 -influence*(transform_factor-1.0)*(np.sqrt( pow(coord[0],2) + pow(coord[1],2) ) / (mesh_unstretched_radius))
                        transformed_coords_list.append( [rad_factor*coord[0]/np.sqrt(transform_factor), rad_factor*coord[1]/np.sqrt(transform_factor), coord[2]*transform_factor] )

                    return np.array(transformed_coords_list)

                case "power":
                    power = args["power"] # stretch in z is transform_factor^power
                    transformed_coords_list = coordinates * [transform_factor**(-power/2), transform_factor**(-power/2), transform_factor**power]
                    return transformed_coords_list
                
                case "singular":
                    # Transform just one axis
                    transformed_coords_list = []
                    for coord in coordinates:
                        if(coord[0] > 0.0):
                            transformed_coords_list.append( [coord[0]*transform_factor, coord[1], coord[2]] )
                        else:
                            transformed_coords_list.append( [coord[0], coord[1], coord[2]] )
                    return transformed_coords_list      
                case "inverse_area":
                    # z stretch inversely proportional to the z-layer's area; like springs in parallel
                    # assumes the shape is symmetric in z, so pairs of ±z planes are transformed at a time

                    eps = sys.float_info.epsilon
                    transformed_coords_list = np.array(coordinates)
                    z_values = np.sort(np.unique(np.abs(coordinates[:,2]))) # get all unique absolute values of planes, then sort for the lowest.
                    print("z_values are", z_values)
                    plane_spacing = z_values[1] # XXX !!! assumes there are enough layers
                    accumulated_factor = 0

                    for z_plane in z_values: # note, these are ABS z values.
                        print("for transform ", transform_factor, "; accum factor is", accumulated_factor)
                        upper_z_indices = np.argwhere(abs(coordinates[:,2]-z_plane) <= eps)
                        lower_z_indices = np.argwhere(abs(coordinates[:,2]+z_plane) <= eps)

                        effective_transform_factor = 1 + (transform_factor-1)/len(upper_z_indices)

                        # transform x,y values by the factor.
                        transformed_coords_list[upper_z_indices] *= [1/np.sqrt(effective_transform_factor), 1/np.sqrt(effective_transform_factor), 1]
                        transformed_coords_list[lower_z_indices] *= [1/np.sqrt(effective_transform_factor), 1/np.sqrt(effective_transform_factor), 1]

                        # shift z values based on effective_transform_factor and the stretches of previous planes.
                        if np.abs(z_plane) < eps:
                            # special behaviour for 0th plane - it doesn't need to shift, but it stretching with still affect others.
                            accumulated_factor += (effective_transform_factor - 1)/2
                        
                        else:
                            shift = plane_spacing * ( (effective_transform_factor - 1)/2 + accumulated_factor )
                            # print(f"starts at {transformed_coords_list[upper_z_indices,2]}, shift by {shift}")
                            transformed_coords_list[upper_z_indices,2] += shift
                            transformed_coords_list[lower_z_indices,2] -= shift
                            accumulated_factor += (effective_transform_factor - 1)
                            # print(f"now at {transformed_coords_list[upper_z_indices,2]}\n")

                    return transformed_coords_list
                case _:
                    print("Invalid transform function type, returning 0 coord: ")
                    return [0.0, 0.0, 0.0]
                
        # System variables
        filename = "Optical_stretcher"

        #
        # Original test runs
        #
        # Particle variables
        # dimension = 2.0e-6#2.4e-6      # Base diameter of the full untransformed sphere
        # transform_factor = 1.0  # Factor to multiply/dividing separation by; Will have XYZ total scaling to conserve volume
        # critical_transform_factor = 1.5 # The max transform you want to apply, which sets the default separation of particles in the system
        # num_factors_tested = 10
        # particle_size = 100e-9 #100e-9      # Will fit as many particles into the dimension space as the transform factor (e.g. base separation) allows
        # object_offset = [0.0, 0.0, 0.0e-6]
        # material = "FusedSilica"
        # connection_mode = "manual"  #"dist", 0.0
        # connection_args = []    # NOTE; This gets populated with arguments when the particles are generated (connections must stay the same at any stretching degree, based on the original sphere, hence must be made when the original sphere is generated)
        # particle_shape = "sphere"
        # #forces_output= ["FTx", "FTy", "FTz"]     # options are ["Fmag","Fx", "Fy", "Fz", "Cmag","Cx", "Cy", "Cz",] 
        # #particle_selections = [[0], [0]]
        # force_reading = "XYZ_split"       #"Z_split", "XYZ_split", "RT_Z_split"
        # transform_type = "linear" # "linear", "inverse_area"
        # # Beam variables
        # E0 = 8.0e6 #4.75e6
        # w0 = 0.5
        # translation = "0.0 0.0 2.0e-6"  # Offset applied to both beams
        # coords_List, nullMode, nullArgs = Generate_yaml.get_stretch_sphere_equilibrium(dimension, particle_size, critical_transform_factor) # Get positions of unstretched sphere to set the spring natural lengths and bending equilibrium angles.
        # option_parameters = Generate_yaml.fill_yaml_options({
        #     "show_output": False,
        #     "show_stress": False,
        #     "quiver_setting": 0,
        #     "force_terms": ["spring"], #"optical", "spring", "bending"
        #     "constants": {"bending": 0.75e-19}, # 0.75e-19 # 5e-20  # 0.5e-18 # 5e-19
        #     "stiffness_spec": {"type":"", "default_value": 5.0e-6}, #5e-8  # 5e-7
        #     "equilibrium_shape": coords_List,
        #     "dipole_radius": 100e-9,
        #     "frames": 1,
        #     "time_step": 0.5e-4, 
        #     "beam_planes": [["z", 0], ["x", 0]],
        # })


        #
        # Half-sized solid sphere experiment match
        
        # dimension = 2.50e-6     # Base diameter of the full untransformed sphere
        # transform_factor = 1.0  # Factor to multiply/dividing separation by; Will have XYZ total scaling to conserve volume
        # critical_transform_factor = 6.75 # The max transform you want to apply, which sets the default separation of particles in the system
        # num_factors_tested = 6
        # particle_size = 100e-9 #100e-9      # Will fit as many particles into the dimension space as the transform factor (e.g. base separation) allows
        # object_offset = [0.0, 0.0, 0.0e-6]
        # material = "FusedSilica"
        # particle_shape = "sphere"
        # connection_mode = "manual"      # "dist", 0.0
        # connection_args = []    # NOTE; This gets populated with arguments when the particles are generated (connections must stay the same at any stretching degree, based on the original sphere, hence must be made when the original sphere is generated)
        # force_reading = "RTZ_split"       #"Z_split", "XYZ_split", "RTZ_split"
        # transform_type = "inverse_area" # "linear", "inverse_area"
        # E0 = 14e6 # 4.75e6
        # w0 = 5.0
        # translation = "0.0 0.0 130.0e-6"  # Offset applied to both beams
        # coords_List, nullMode, nullArgs = Generate_yaml.get_stretch_sphere_equilibrium(dimension, particle_size, critical_transform_factor) # Get positions of unstretched sphere to set the spring natural lengths and bending equilibrium angles.
        # option_parameters = Generate_yaml.fill_yaml_options({
        #     "show_output": True,
        #     "show_stress": False,
        #     "quiver_setting": 0,
        #     "wavelength": 1.0e-6,
        #     "force_terms": ["optical", "spring", "bending"], #"optical", "spring", "bending"
        #     "constants": {"bending": 0.75e-19}, # 0.75e-19 # 5e-20  # 0.5e-18 # 5e-19
        #     "stiffness_spec": {"type":"", "default_value": 1.0e-7}, #5e-6 #5e-8  # 5e-7
        #     "equilibrium_shape": coords_List,
        #     "dipole_radius": particle_size,
        #     "frames": 1,
        #     "time_step": 0.5e-4, 
        #     "beam_planes": [["z", 0]], #  [["z", 0], ["x", 0]]  [["z", 0]]
        #     "beam_alpha": 0.6,
        # })

        #
        # Other
        #
        # dimension = 400e-9     # Base diameter of the full untransformed sphere
        # transform_factor = 1.0  # Factor to multiply/dividing separation by; Will have XYZ total scaling to conserve volume
        # critical_transform_factor = 1.75 # The max transform you want to apply, which sets the default separation of particles in the system
        # num_factors_tested = 100
        # particle_size = 50e-9 #100e-9      # Will fit as many particles into the dimension space as the transform factor (e.g. base separation) allows

        # object_offset = [0.0, 0.0, 0.0e-6]
        # material = "FusedSilica"
        # particle_shape = "sphere"
        # connection_mode = "manual"      # "dist", 0.0
        # connection_args = []    # NOTE; This gets populated with arguments when the particles are generated (connections must stay the same at any stretching degree, based on the original sphere, hence must be made when the original sphere is generated)
        # force_reading = "XYZ_split"       #"Z_split", "XYZ_split", "RTZ_split"
        # transform_type = "linear" # "linear", "inverse_area"

        # E0 = 4.5e6 # 4.75e6
        # w0 = 0.5
        # translation = "0.0 0.0 2.0e-6"  # Offset applied to both beams
        # coords_List, nullMode, nullArgs = Generate_yaml.get_stretch_sphere_equilibrium(dimension, particle_size, critical_transform_factor) # Get positions of unstretched sphere to set the spring natural lengths and bending equilibrium angles.
        # option_parameters = Generate_yaml.fill_yaml_options({
        #     "show_output": False,
        #     "show_stress": False,
        #     "quiver_setting": 0,
        #     "wavelength": 1.0e-6,
        #     "force_terms": ["optical", "spring", "bending"], #"optical", "spring", "bending"
        #     "constants": {"bending": 0.75e-19}, # 0.75e-19 # 5e-20  # 0.5e-18 # 5e-19
        #     "stiffness_spec": {"type":"", "default_value": 5.0e-6}, #5e-6 #5e-8  # 5e-7
        #     "equilibrium_shape": coords_List,
        #     "dipole_radius": 25e-9,
        #     "frames": 1,
        #     "time_step": 0.5e-4, 
        #     "beam_planes": [["x", 0],["z", 0]], #  [["z", 0], ["x", 0]]  [["z", 0]]
        #     "beam_alpha": 0.6,
        # })


        #
        # Stretching for experimentally accurate BUT scaled shape (1/3 scale)
        #
        dimension = 6720e-9     # Base diameter of the full untransformed sphere
        transform_factor = 1.0  # Factor to multiply/dividing separation by; Will have XYZ total scaling to conserve volume
        critical_transform_factor = 1.75 # The max transform you want to apply, which sets the default separation of particles in the system
        num_factors_tested = 5
        particle_size = 100e-9   #100e-9      # Will fit as many particles into the dimension space as the transform factor (e.g. base separation) allows
        object_offset = [0.0, 0.0, 0.0e-6]
        material = "FusedSilica"
        particle_shape = "sphere"
        force_reading = "XYZ_split"         # "Z_split", "XYZ_split", "RTZ_split"
        transform_type = "radial_meridional"         # "linear", "inverse_area", "radial_meridional"
        E0 = 14.0e6 #14e6
        w0 = 4.4   #5.4
        translation = "0.0 0.0 135.0e-6"  # Offset applied to both beams

        sphere_type = "shell" # options are "solid" or "shell"
        
        match sphere_type:
            case "solid":
                # currently not working
                num_particles = None
                connection_mode = "manual"  
                connection_args = []    # NOTE; This gets populated with arguments when the particles are generated (connections must stay the same at any stretching degree, based on the original sphere, hence must be made when the original sphere is generated)
                coords_list, _, _ = Generate_yaml.get_stretch_sphere_equilibrium(dimension, particle_size, critical_transform_factor) # Get positions of unstretched sphere to set the spring natural lengths and bending equilibrium angles.
            case "shell":
                num_particles = 100
                connection_mode = "num"
                connection_args = 5
                coords_list = []
                coords_list_raw = Generate_yaml.get_sunflower_points(num_particles, dimension/2.0)
                for i in range(len(coords_list_raw)):
                    coords_list.append(list(coords_list_raw[i]))

        option_parameters = Generate_yaml.fill_yaml_options({
            "show_output": True,
            "show_stress": False,
            "quiver_setting": 0,
            "wavelength": 0.785e-6,
            "frames": 1,
            "max_size": 5e-6,

            "time_step": 0.0625e-4,  #0.125e-4 
            "force_terms": ["optical", "spring", "bending"], #"optical", "spring", "bending"
            "constants": {"bending": 0.75e-19}, # 0.75e-19 # 5e-20  # 0.5e-18 # 5e-19
            "stiffness_spec": {"type":"", "default_value": 2.5e-6}, #3.35e-5 #3.5e-5
            "equilibrium_shape": coords_list,
            "dipole_radius": 100e-9,
            "beam_planes": [], #  [["x", 0],["z", 0]]
            "beam_alpha": 0.4,
        })

        if option_parameters["show_output"] == False: option_parameters["frames"] = 1
        power_args = {"power": 1}

        # Run a varying simulation over transforms
        # Specify all parameters in the xlsx file so a subset can be pulled later based on read_parameters. Gives information about the structure of the data in the file.
        
        read_parameters_lookup = {
            "Fmag":      [["FT",0], ["FT",1], ["FT",2]],
            "FZmag":     [["FT",0], ["FT",1], ["FT",2]],
            "Z_split":   [["FT",0], ["FT",1], ["FT",2], ["X",0], ["X",1], ["X",2]],
            "XYZ_split": [["FT",0], ["FT",1], ["FT",2], ["X",0], ["X",1], ["X",2]],
            "RTZ_split": [["FT",0], ["FT",1], ["FT",2], ["X",0], ["X",1], ["X",2]], 
        }
        parameters_stored = [{"type":"X", "args":["x", "y", "z"]},{"type":"F", "args":["Fx", "Fy", "Fz"]},{"type":"FT", "args":["FTx", "FTy", "FTz"]}, {"type":"C", "args":["Cx", "Cy", "Cz"]}]
        read_frames = [0]

        datalabel_set = ["FTx", "FTy", "FTz"] if force_reading != "RTZ_split" else ["FTr", "FTtheta", "FTz"]
        if(force_reading=="FZmag"):
            datalabel_set = ["FTz"]

        func_transform_partial = partial(func_transform, transform_type=transform_type, args=power_args)
        params_i = 0
        expt_output = [force_reading] # NOTE listed as one expt, but cases designed so 3 (x,y,z) outputs are put into data set.

        #
        # Vary transform factor
        #
        transform_factor_list = np.linspace(transform_factor, critical_transform_factor, num_factors_tested)
        num_transforms = len(transform_factor_list)
        data_set = np.array([[transform_factor_list, np.zeros(num_transforms)] for _ in range(len(datalabel_set))], dtype=object)        
        graphlabel_set={"title":f"Stretched sphere model, mode = {force_reading}", "xAxis":"Transform_Factor", "yAxis":"Forces(N)"}
        for i in range(len(transform_factor_list)):
            print("\nProgress; "+str(i)+"/"+str(num_transforms))
            particle_num = Generate_yaml.make_yaml_stretch_sphere(filename, option_parameters, particle_shape, E0, w0, dimension, particle_size, transform_factor_list[i], critical_transform_factor, func_transform_partial, object_offset, translation, connection_mode=connection_mode, connection_args=connection_args, material=material, num_particles=num_particles)
            print("made yaml")
            DM.main(filename)
            data_set = get_forces_via_lookup(filename, data_set, particle_num, i, params_i, expt_output, ["all"], read_frames, read_parameters_lookup, parameters_stored, parameters_stored_torque=None, torque_centre=None)

            # Get positions of particles out
            read_parameters_pos = []
            for p in range(particle_num):
                read_parameters_pos.append({"type":"X", "particle":p, "subtype":0})
                read_parameters_pos.append({"type":"X", "particle":p, "subtype":1})
                read_parameters_pos.append({"type":"X", "particle":p, "subtype":2})
            output_data = pull_file_data(
                filename, 
                parameters_stored, 
                [0], 
                read_parameters_pos, 
                invert_output=False
            )[0]
            # Get bounding box
            ##
            ## Could extend to find average instead for each bound
            ##
            bounding_box = [ [0.0, 0.0], [0.0, 0.0], [0.0, 0.0] ]   # [ [xl, xu], [yl, yu], [zl, zu] ]
            for j in range(particle_num):
                for k in range(3):
                    if(output_data[3*j+k] < bounding_box[k][0]):
                        bounding_box[k][0] = output_data[3*j+k]
                    if(output_data[3*j+k] > bounding_box[k][1]):
                        bounding_box[k][1] = output_data[3*j+k]
            # Get Height/Width ratio
            hw_ratio = (bounding_box[2][1] - bounding_box[2][0]) / ( ((bounding_box[0][1] - bounding_box[0][0])+(bounding_box[1][1] - bounding_box[1][0]))/2.0 )
            print(f"T={transform_factor_list[i]},  Height/Width ratio={hw_ratio}")

        #
        # Vary offset of object (wanted for a small scale test)
        #
        # graphlabel_set={"title":f"Translation, mode = {force_reading}", "xAxis":"translation[m]", "yAxis":"Forces[N]"}
        # min_offset = -10.0e-6
        # max_offset = 10.0e-6
        # offset_list = np.linspace(min_offset, max_offset, num_factors_tested)
        # data_set = np.array([[offset_list, np.zeros(num_factors_tested)] for _ in range(len(datalabel_set))], dtype=object)
        # for i in range(len(offset_list)):
        #     print("\nProgress; "+str(i)+"/"+str(len(offset_list)))
        #     object_offset = [0.0, 0.0, offset_list[i]]
        #     particle_num = Generate_yaml.make_yaml_stretch_sphere(filename, option_parameters, particle_shape, E0, w0, dimension, particle_size, transform_factor, critical_transform_factor, func_transform_partial, object_offset, translation, connection_mode=connection_mode, connection_args=connection_args, material=material)
        #     DM.main(filename)
        #     data_set = get_forces_via_lookup(filename, data_set, particle_num, i, params_i, expt_output, ["all"], read_frames, read_parameters_lookup, parameters_stored, parameters_stored_torque=None, torque_centre=None)

        # Plot forces for each step considered to see if equilibrium is being reached
        Display.plot_multi_data(np.array(data_set), datalabel_set, graphlabel_set=graphlabel_set) 


    case "showcase_refinement":
        filename = "SingleLaguerre"
        option_parameters = Generate_yaml.fill_yaml_options({
            "show_output": True,
            "show_stress": False,
            "dipole_radius": 300e-9,
            "quiver_setting":0,
            "force_terms": ["optical"],
            "beam_planes": []
        })
        Generate_yaml.make_yaml_refine_cube_showcase(filename, 0.8e-6, [-2.8e-6, 0.0, 0.0], "sphere", option_parameters, beam="LAGUERRE", material="FusedSilica")
        DM.main(YAML_name=filename)



    case _:
        print("Unknown run type: ",sys.argv[1])
        # print("Allowed run types are; 'spheresInCircle', 'torusInCircle', 'torusInCircleFixedPhi', 'spheresInCircleSlider', 'spheresInCircleDipoleSize', 'torusInCircleDipoleSize', 'testVolumes', 'connected_sphereGrid', 'connected_sphereShell'")
