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

        "vmd_output": True,
        "excel_output": True,
        "include_force": True,
        "include_couple": True,

        "show_output": True,
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
    for arg in ["wavelength", "dipole_radius", "time_step"]:
        file.write(f"  {arg}: {parameters[arg]}\n")

    file.write("output:\n")
    for arg in ["vmd_output", "excel_output", "include_force", "include_couple"]:
        file.write(f"  {arg}: {parameters[arg]}\n")

    file.write("display:\n")
    for arg in ["show_output", "frame_interval", "max_size", "resolution", "frame_min", "frame_max", "z_offset"]:
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

    filename = "SingleLaguerre_SphereVary"
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

    filename = "SingleLaguerre_TorusVary"
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

    filename = "SingleLaguerre_TorusVary"
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
    filename = "SingleLaguerre_SphereVary"
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
    filename = "SingleLaguerre_SphereVary"
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
    filename = "SingleLaguerre_SphereVary"
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
    filename = "SingleLaguerre_SphereVary"
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
        pos  = np.array([ output_data[0,i+0], output_data[0,i+1], output_data[0,i+2] ])
        dist = np.linalg.norm( point-pos )
        if(dist <= low_dist):
            low_dist  = dist
            low_index = int(np.floor(i/3.0))
    return low_index

def get_number_of_particles_YAML(filename, parameters_stored):
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
        print("number_of_particles= ", number_of_particles)
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

def simulations_singleFrame_optForce_spheresInCircle(particle_numbers, filename, include_additionalForces=False):
    #
    # Performs a DDA calculation for various particles in a circular ring on the Z=0 plane
    #
    # particle_numbers = list of particle numbers to be tested in sphere e.g. [1,2,3,4,8]
    #
    
    particle_info = [];
    place_radius = 1.15e-6#152e-6         #1.15e-6
    particle_radii = 200e-9         #200e-9
    parameters = {"frames": 1, "frame_max": 1, "show_output": False}

    record_parameters = ["F"]
    if(include_additionalForces):   # Record total forces instead of just optical forces
        record_parameters = ["FT"]  #

    #For each scenario to be tested
    for i, particle_number in enumerate(particle_numbers):
        print(f"\n{i}/{len(particle_numbers)}: Performing calculation for {particle_number} particles")
        #Generate required YAML, perform calculation, then pull force data
        generate_sphere_yaml("circle", particle_number, characteristic_distance=place_radius, particle_radii=particle_radii, parameters=parameters)     # Writes to SingleLaguerre_SphereVary.yml
        #Run DipolesMulti2024Eigen.py
        run_command = "python DipolesMulti2024Eigen.py "+filename
        run_command = run_command.split(" ")
        print("=== Log ===")
        result = subprocess.run(run_command, stdout=subprocess.DEVNULL) #, stdout=subprocess.DEVNULL

        #Pull data from xlsx into a local list in python
        record_particle_info(filename, particle_info, record_parameters=record_parameters)
    #Write combined data to a new xlsx file
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = "\n".join(
        (
            "Spheres",
            "R_placed   (m)= "+str(place_radius),
            "R_particle (m)= "+str(particle_radii)
        )
    )
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
        generate_sphere_slider_yaml("circle", particle_total, slider_theta, characteristic_distance=place_radius, particle_radii=particle_radii, parameters=parameters)     # Writes to SingleLaguerre_SphereVary.yml

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
            generate_sphere_arbitrary_yaml(particles, frames_of_animation=1, wavelength=wavelength)     # Writes to SingleLaguerre_SphereVary.yml
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

    parameters = {"frames": frames_of_animation, "frame_max": frames_of_animation, "show_output": False}

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
    generate_sphereGrid_yaml(particle_radius, particle_spacing, bounding_sphere_radius, parameters=parameters)     # Writes to SingleLaguerre_SphereVary.yml
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
    generate_sphereShell_yaml(particle_radius, particle_spacing, shell_radius, parameters=parameters)     # Writes to SingleLaguerre_SphereVary.yml
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
    

def filter_dipole_sizes(volumes, dipole_size_range, num, target_volume=None):
    # used to filter the results of "get_sphere_volumes" and "get_torus_volumes"
    # * This is so that the simulated objects have more similar volumes.
    # Finds the dipole sizes with volumes closest to the target volume (defaults to the average volume).
    # Returns these best sizes, volumes, and the maximum error.
    if target_volume == None:
        target_volume = np.average(volumes)

    dipole_sizes = np.linspace(*dipole_size_range)
    num_sizes = dipole_size_range[2]

    if num > num_sizes:
        sys.exit(f"filter_dipole_sizes: too many points requested, max is {num_sizes}")

    # finds the <num> min values, the rest are unsorted and are sliced off.
    indices = np.argpartition(np.abs(np.array(volumes)-target_volume), num)[:num] 
    max_error = abs(volumes[indices[num-1]]-target_volume)/target_volume

    filtered_dipole_sizes = np.array(dipole_sizes)[indices]
    sort_is = np.argsort(filtered_dipole_sizes)
    final_is = list(indices[sort_is])
    print(f"Filtered dipole sizes to {num} values, with max volume error: {max_error:.02%}.")

    # In testing: keep sizes only at a max or min volume, this helps but not significantly.
    for i in final_is:
        if i == 0 or i == num_sizes-1:
            pass
        else:
            if not( volumes[i] > volumes[i-1] != volumes[i+1] > volumes[i] ): # not (True if gradients are opposite ie max or min)
                final_is.remove(i)
                print(f"Removed {dipole_sizes[i]}")

    return np.array(dipole_sizes)[final_is], np.array(volumes)[final_is], max_error

def simulations_fibre_1D_sphere(filename, chain_length, particle_radius, particle_number, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True):
    particle_info = [];
    record_parameters = ["F"]

    # Generate set of particle in chain
    print(f"Performing calculation for {particle_number} particle chain")
    Generate_yaml.make_yaml_fibre_1d_sphere(filename, time_step, frames, show_output, chain_length, particle_radius, particle_number, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename, constants=constants, force_terms=force_terms)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_1D_cylinder(filename, chain_length, particle_length, particle_radius, particle_number, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True):
    particle_info = [];
    record_parameters = ["F"]

    # Generate set of particle in chain
    print(f"Performing calculation for {particle_number} particle chain")
    Generate_yaml.make_yaml_fibre_1d_cylinder(filename, time_step, frames, show_output, chain_length, particle_length, particle_radius, particle_number, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename, constants=constants, force_terms=force_terms)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_2D_sphere_hollowShell(filename, chain_length, shell_radius, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, time_step, constants, force_terms, stiffness_spec, frames, show_output=True, include_beads=False):
    particle_info = [];
    record_parameters = ["F"]

    # Generate YAML for set of particles and beams
    print(f"Performing calculation for {particle_number_radial*particle_number_angular} particles")
    with open(f"{filename}.yml", "w") as _:     # Used to reset file each time this is run
        pass                                    #
    Generate_yaml.make_yaml_fibre_2d_sphere_hollowshell(filename, time_step, frames, show_output, chain_length, shell_radius, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, beam="GAUSS_CSP", include_beads=include_beads)

    # Run simulation
    DM.main(YAML_name=filename, constants=constants, force_terms=force_terms, stiffness_spec=stiffness_spec)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_2D_cylinder_hollowShell(filename, chain_length, shell_radius, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True):
    particle_info = [];
    record_parameters = ["F"]

    # Generate YAML for set of particles and beams
    print(f"Performing calculation for {particle_number_radial*particle_number_angular} particles")
    Generate_yaml.make_yaml_fibre_2d_cylinder_hollowshell(filename, time_step, frames, show_output, chain_length, shell_radius, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename, constants=constants, force_terms=force_terms)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_2D_sphere_thick_connectUniform(filename, chain_length, shell_radius, shell_number, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True):
    particle_info = [];
    record_parameters = ["F"]

    # Generate YAML for set of particles and beams
    print(f"Performing calculation for {particle_number_radial*particle_number_angular} particles")
    Generate_yaml.make_yaml_fibre_2d_sphere_thick_uni(filename, time_step, frames, show_output, chain_length, shell_radius, shell_number, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename, constants=constants, force_terms=force_terms)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_2D_cylinder_thick_connectUniform(filename, chain_length, shell_radius, shell_number, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True):
    particle_info = [];
    record_parameters = ["F"]

    # Generate YAML for set of particles and beams
    print(f"Performing calculation for {particle_number_radial*particle_number_angular} particles")
    Generate_yaml.make_yaml_fibre_2d_cylinder_thick_uni(filename, time_step, frames, show_output, chain_length, shell_radius, shell_number, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename, constants=constants, force_terms=force_terms)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_2D_sphere_shellLayers(filename, chain_length, shell_radius, shell_number, particle_radius, particle_separation, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True):
    particle_info = [];
    record_parameters = ["F"]

    # Generate YAML for set of particles and beams
    print("Performing calculation for N particles")
    Generate_yaml.make_yaml_fibre_2d_sphere_shelllayers(filename, time_step, frames, show_output, chain_length, shell_radius, shell_number, particle_radius, particle_separation, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename, constants=constants, force_terms=force_terms)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_fibre_2D_cylinder_shellLayers(filename, chain_length, shell_radius, shell_number, particle_length, particle_radius, particle_separation, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True):
    particle_info = [];
    record_parameters = ["F"]

    # Generate YAML for set of particles and beams
    print("Performing calculation for N particles")
    Generate_yaml.make_yaml_fibre_2d_cylinder_shelllayers(filename, time_step, frames, show_output, chain_length, shell_radius, shell_number, particle_length, particle_radius, particle_separation, connection_mode, connection_args, beam="LAGUERRE")

    # Run simulation
    DM.main(YAML_name=filename, constants=constants, force_terms=force_terms)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    record_particle_info(filename, particle_info, record_parameters=record_parameters)
    store_combined_particle_info(filename, particle_info, record_parameters=record_parameters)
    parameter_text = ""
    return parameter_text

def simulations_refine_cuboid(dimensions, dipole_size, separations, particle_size, force_terms, particle_shape, object_offset, time_step=1e-4, show_output=True):
    #
    # Consider a cuboid of given parameters, vary aspects of cuboid, take force measurements for each scenario
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
    read_parameters = [
        {"type":"F", "particle":0, "subtype":0},
        {"type":"F", "particle":0, "subtype":1},
        {"type":"F", "particle":0, "subtype":2}
    ]

    # Begin calculations
    ####
    #### REMOVE DIPOLE_SIZE ARGS
    ####
    print("Performing refinement calculation for cuboid")
    dipole_sizes = np.linspace(40e-9, 120e-9, 15)
    data_set = []
    dipVary_forceMag_data = np.array([ np.array(dipole_sizes), np.zeros( len(dipole_sizes) ) ])   #[ [dipole_sizes], [recorded_data]-> e.g. force magnitude ]
    dipVary_forceX_data = np.array([ np.array(dipole_sizes), np.zeros( len(dipole_sizes) ) ])
    dipVary_forceY_data = np.array([ np.array(dipole_sizes), np.zeros( len(dipole_sizes) ) ])
    dipVary_forceZ_data = np.array([ np.array(dipole_sizes), np.zeros( len(dipole_sizes) ) ])
    
    for i in range(len(dipole_sizes)):
        # Generate YAML for set of particles and beams
        Generate_yaml.make_yaml_refine_cuboid(filename, time_step, dimensions, dipole_sizes[i], separations, object_offset, particle_size, particle_shape, frames=1, show_output=show_output, beam="LAGUERRE")
        # Run simulation
        DM.main(YAML_name=filename, force_terms=force_terms)
        # Pull data needed from this frame, add it to another list tracking
        output_data = pull_file_data(
            filename, 
            parameters_stored, 
            read_frames, 
            read_parameters, 
            invert_output=False
        )
        # Calculate required quantities
        recorded_force = np.array([output_data[0, 0], output_data[0, 1], output_data[0, 2]])    # Only pulling at a single frame, => only 1 list inside output, holding each 
        recorded_force_mag = np.sqrt(np.dot(recorded_force, recorded_force.conjugate()))        # Calculate dep. var. to be plotted
        # Store quantities
        dipVary_forceMag_data[1][i] = recorded_force_mag
        dipVary_forceX_data[1][i] = recorded_force[0]
        dipVary_forceY_data[1][i] = recorded_force[1]
        dipVary_forceZ_data[1][i] = recorded_force[2]
        
    data_set.append(dipVary_forceMag_data)
    data_set.append(dipVary_forceX_data)
    data_set.append(dipVary_forceY_data)
    data_set.append(dipVary_forceZ_data)

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    parameter_text = "\n".join(
        (
            "Refined_Cuboid",
            "dimensions   (m)= "+str(dimensions),
            "dipole_size  (m)= "+str(dipole_size),
            "separations  (m)= "+str(separations),
            "particle_size(m)= "+str(particle_size)
        )
    )
    return parameter_text, np.array(data_set)

def simulations_refine_arch_prism(dimensions, separations, particle_length, dipole_size, deflection, object_offset, force_terms, particle_shape, show_output=True):
    #
    # Consider a cuboid of given parameters, calculate the path it should be located on when deflected by some amount
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
    print("Performing refinement calculation for cuboid")
    
    ####
    ## ADD DIPOLE DIRECT PLOTTER TO GET BETTER VIEW OF WHERE DIPOLES ARE
    ####
    ####
    ## SWITCH TO DAVID'S VARY METHOD
    ####
    vary_deflection = np.linspace(0.0e-6, 0.5e-6, 3) # Varying deflection
    vary_dipoleSize = np.linspace(25e-9, 60e-9, 10) # Varying dipoleSize
    vary_particleLength = np.linspace(100e-9, 75e-9, 1) # Varying particleLength
    vary_separationX    = np.linspace(0.0e-6, 0.5e-6, 5) # Varying particleLength
    ####
    ## -> SHOULD BE TESTING PARTICLE LENGTHS THAT ACTUALLY FIR NICELY INTO THE REGION
    ####

    data_set = []
    data_vary_dipoleSize_F       = np.array([ np.array(vary_dipoleSize), np.zeros( len(vary_dipoleSize) ) ])
    data_vary_dipoleSize_FperDip = np.array([ np.array(vary_dipoleSize), np.zeros( len(vary_dipoleSize) ) ])
    data_vary_dipoleSize_modF2   = np.array([ np.array(vary_dipoleSize), np.zeros( len(vary_dipoleSize) ) ])   #[ [dipole_sizes], [recorded_data]-> e.g. force magnitude ]
    
    for j in range(len(vary_particleLength)):
        for i in range(len(vary_dipoleSize)):
            # Generate YAML for set of particles and beams
            Generate_yaml.make_yaml_refine_arch_prism(filename, time_step, dimensions, separations, vary_particleLength[j], vary_dipoleSize[i], deflection, object_offset, particle_shape, frames=1, show_output=show_output, beam="LAGUERRE")
            
            number_of_particles = get_number_of_particles_YAML(filename, parameters_stored)
            read_parameters_central = []
            for p in range(number_of_particles):
                read_parameters_central.append({"type":"X", "particle":p, "subtype":0})
                read_parameters_central.append({"type":"X", "particle":p, "subtype":1})
                read_parameters_central.append({"type":"X", "particle":p, "subtype":2})
            read_parameters = []
            for p in range(number_of_particles):
                read_parameters.append({"type":"F", "particle":p, "subtype":0})
                read_parameters.append({"type":"F", "particle":p, "subtype":1})
                read_parameters.append({"type":"F", "particle":p, "subtype":2})
            
            #central_particle_number = 0
            central_particle_number = get_closest_particle(     # Measure forces on central particle for all systems
                np.array([0.0, 0.0, deflection]),
                output_data = pull_file_data(
                    filename, 
                    parameters_stored, 
                    read_frames, 
                    read_parameters_central, 
                    invert_output=False
                )
            )
            centre_dipoles = DM.sphere_size([vary_particleLength[j]], vary_dipoleSize[i])   # Number of dipoles of centre particle (ALL HAVE SAME NUMBER)
            
            # Run simulation
            DM.main(YAML_name=filename, force_terms=force_terms)
            # Pull data needed from this frame, add it to another list tracking
            output_data = pull_file_data(
                filename, 
                parameters_stored, 
                read_frames, 
                read_parameters, 
                invert_output=False
            )
            # Calculate required quantities
            # (1)&(2) Get magnitude of force per dipole on central particle
            centre_force_vec = np.array(
                [
                    output_data[0, 3*central_particle_number +0],
                    output_data[0, 3*central_particle_number +1],
                    output_data[0, 3*central_particle_number +2]
                ]
            )
            centre_force = np.sqrt(np.dot(centre_force_vec, centre_force_vec.conjugate()))  # Finding |F| for centre particle
            data_vary_dipoleSize_F[1][i] = centre_force
            data_vary_dipoleSize_FperDip[1][i] = centre_force/centre_dipoles

            # (3) Get magnitude^2 of force across full mesh
            total_force_vec = np.zeros(3, dtype=complex)
            for p in range(0, len(output_data[0]), 3):    # Go through each particle in sets of 3 (Fx,Fy,Fz measurements)
                total_force_vec[0] += output_data[0, p+0]
                total_force_vec[1] += output_data[0, p+1]
                total_force_vec[2] += output_data[0, p+2]
            total_force = np.dot(total_force_vec, total_force_vec.conjugate())  # Finding |F|^2
            data_vary_dipoleSize_modF2[1][i] = total_force

            
            
            ### OLD QUANITTIES ###
            # recorded_force = np.array([output_data[0, 0], output_data[0, 1], output_data[0, 2]])    # Only pulling at a single frame, => only 1 list inside output, holding each 
            # recorded_force_mag = np.sqrt(np.dot(recorded_force, recorded_force.conjugate()))        # Calculate dep. var. to be plotted
            # Store quantities
            # dipVary_forceMag_data[1][i] = recorded_force_mag
            # dipVary_forceX_data[1][i] = recorded_force[0]
            # dipVary_forceY_data[1][i] = recorded_force[1]
            # dipVary_forceZ_data[1][i] = recorded_force[2]
            
        data_set.append(data_vary_dipoleSize_F)
        data_set.append(data_vary_dipoleSize_FperDip)
        data_set.append(data_vary_dipoleSize_modF2)

        # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
        parameter_text = "\n".join(
            (
                "Refined_Cuboid",
                "dimensions   (m)= "+str(dimensions),
                "separations  (m)= "+str(separations),
                "particle_size(m)= "+str(particle_length)
            )
        )
    return parameter_text, np.array(data_set)

def simulations_refine_general(dimensions, variables_list, force_terms, time_step=1e-4, show_output=False, indep_vector_component=2):
    #
    # Consider an object of given parameters, vary its aspects, take force measurements for each scenario
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
    # type must match with that of parameters_stored, subtype just indexes the args.
    read_parameters = [
        {"type":"F", "particle":0, "subtype":0},
        {"type":"F", "particle":0, "subtype":1},
        {"type":"F", "particle":0, "subtype":2}
    ]
    
    # get list of variables from the dictionary.
    dipole_sizes = variables_list["dipole_sizes"]
    separations_list = variables_list["separations_list"]
    particle_sizes = variables_list["particle_sizes"]
    particle_shapes = variables_list["particle_shapes"]
    object_offsets = variables_list["object_offsets"]
    # get the independent variable (the one to be plotted against)
    indep_name = variables_list["indep_var"]
    indep_list = np.array(variables_list[indep_name] )

    # Based on the indep var, set what variable are varied over different lines of the graph.
    match indep_name:
        case "dipole_sizes": 
            line_vars = [separations_list, particle_sizes, particle_shapes, object_offsets]
            indep_axis_list = indep_list
        case "separations_list": 
            line_vars = [dipole_sizes, particle_sizes, particle_shapes, object_offsets]
            indep_axis_list = indep_list[:,indep_vector_component] # vector var so pick which component to plot agaisnt
        case "particle_sizes": 
            line_vars = [dipole_sizes, separations_list, particle_shapes, object_offsets]
            indep_axis_list = indep_list
        case "particle_shapes": 
            line_vars = [dipole_sizes, separations_list, particle_sizes, object_offsets]
            indep_axis_list = indep_list
        case "object_offsets": 
            line_vars = [dipole_sizes, separations_list, particle_sizes, particle_shapes]
            indep_axis_list = indep_list[:,indep_vector_component] # vector var so pick which component to plot agaisnt


    # Begin calculations
    print("Performing refinement calculation for cuboid")
    data_set = []
    data_set_params = []
    num_indep = len(indep_axis_list)
    forceMag_data = np.array([ indep_axis_list, np.zeros(num_indep) ])  
    forceX_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
    forceY_data = np.array([ indep_axis_list, np.zeros(num_indep) ])
    forceZ_data = np.array([ indep_axis_list, np.zeros(num_indep) ])

    # Iterate though every combination of variables that are varied across the lines of the graph.
    for params in it.product(*line_vars):
        match indep_name:
            case "dipole_sizes": separations, particle_size, particle_shape, object_offset = params
            case "separations_list": dipole_size, particle_size, particle_shape, object_offset = params
            case "particle_sizes": dipole_size, separations, particle_shape, object_offset = params
            case "particle_shapes": dipole_size, separations, particle_size, object_offset = params
            case "object_offsets": dipole_size, separations, particle_size, particle_shape = params

        # Iterate over independent variable to get the data for each line.
        for i, indep_var in enumerate(indep_list):
            match indep_name:
                case "dipole_sizes": dipole_size = indep_var
                case "separations_list": separations = indep_var
                case "particle_sizes": particle_size = indep_var
                case "particle_shapes": particle_shape = indep_var
                case "object_offsets": object_offset = indep_var
        
            # Generate YAML for set of particles and beams
            Generate_yaml.make_yaml_refine_cuboid(filename, time_step, dimensions, dipole_size, separations, object_offset, particle_size, particle_shape, frames=1, show_output=show_output, beam="LAGUERRE")
            # Run simulation
            DM.main(YAML_name=filename, force_terms=force_terms)
            # Pull data needed from this frame, add it to another list tracking
            output_data = pull_file_data(
                filename, 
                parameters_stored, 
                read_frames, 
                read_parameters, 
                invert_output=False
            )
            # Calculate required quantities
            recorded_force = np.array([output_data[0, 0], output_data[0, 1], output_data[0, 2]])    # Only pulling at a single frame, => only 1 list inside output, holding each 
            recorded_force_mag = np.sqrt(np.dot(recorded_force, recorded_force.conjugate()))        # Calculate dep. var. to be plotted
            
            # Store quantities
            forceMag_data[1][i] = recorded_force_mag
            forceX_data[1][i] = recorded_force[0]
            forceY_data[1][i] = recorded_force[1]
            forceZ_data[1][i] = recorded_force[2]
            # print(f"\n{indep_var} has z-force: {recorded_force[2]}\n")

        data_set.append(np.array(forceMag_data))
        data_set.append(np.array(forceX_data))
        data_set.append(np.array(forceY_data))
        data_set.append(np.array(forceZ_data))
        data_set_params.append(params)
        

    # Pull data from xlsx into a local list in python, Write combined data to a new xlsx file
    parameter_text = ""
    return parameter_text, np.array(data_set), data_set_params


def filter_data_set(force_filter, data_set, data_set_params, legend_params, indep_name):
    #
    # Filter for what force types are wanted
    # Options, force_filter=["Fmag", "Fx", "Fy", "Fz"] XXX could add Ftheta, Fr
    # Returns filtered data_set, datalabel_set
    # The label only use the variables in legend_params since others do not change so are shown in the title.
    #
    i_dict = {"dipole_sizes":0, "separations_list":1, "particle_sizes":2, "particle_shapes":3, "object_offsets":4} # convert between names and list index.
    indep_val = i_dict[indep_name]

    filtered_i = []
    datalabel_set = []
    # Iterate over the experiments
    for i, params in enumerate(data_set_params):
        # Pick what variables to show in the legend.
        param_str = ""
        for key, value in i_dict.items():
            if value > indep_val: value -= 1 # params DOESN'T include the indep var so the indices in i_dict beyond the indep var must be shifted (-1) to fill the gap.
            if key in legend_params:
                param_str += f"{display_var(key, params[value])} "

        # Pick what forces to plot and create the legend string.
        if "Fmag" in force_filter:
            filtered_i.append(4*i)
            datalabel_set.append(f"F Mag, {param_str}")
        if "Fx" in force_filter:
            filtered_i.append(4*i+1)
            datalabel_set.append(f"Fx, {param_str}")
        if "Fy" in force_filter:
            filtered_i.append(4*i+2)
            datalabel_set.append(f"Fy, {param_str}")
        if "Fz" in force_filter:
            filtered_i.append(4*i+3)
            datalabel_set.append(f"Fz, {param_str}")
    
    return data_set[filtered_i], datalabel_set

def display_var(variable_type, value=None):
    #
    # Returns a string based on the arguments
    # if value=None, returns the name and units of variable_type as a tuple e.g. "dipole size", "/m"
    # Else returns a single formatted string using the value e.g. "dipole size = 4e-8m"
    #
    if value == None:
        match variable_type:
            case "dipole_sizes": return "dipole size", "/m"
            case "separations_list": return "separation", "/m"
            case "particle_sizes": return "particle size", "/m"
            case "particle_shapes": return " particle shape"
            case "object_offsets": return "offset", "/m"
            case _: return f"{variable_type} UNKNOWN", "UNITS"
    
    else:
        match variable_type:
            case "dipole_sizes": return f"dipole size = {value}m"
            case "separations_list": return f"separation = [{value[0]},{value[1]},{value[2]}]m"
            case "particle_sizes": return f"particle size = {value}m"
            case "particle_shapes": return f" particle shape = {value}"
            case "object_offsets": return f"offset = [{value[0]},{value[1]},{value[2]}]m"
            case _: return f"{variable_type} UNKNOWN"

def get_titlelegend(variables_list, indep_name):
    #
    # Formats the graph title.
    # Variables that don't change are put in the graph title. Otherwise, they are recorded to go in the legend.
    # This excludes the independent variable.
    #
    titlestr = f"Forces against {display_var(indep_name)[0]}"
    legend_params = []
    newline_count = 1
    # For each variable, print it in the legend or the title.
    for key, value in variables_list.items():
        if key == indep_name: continue # Indep var isn't in title or legend.
        if len(value) == 1: # It doesn't change so put in the title
            titlestr += f", {display_var(key, value[0])}"
            if newline_count == 2:
                newline_count = 0
                titlestr += "\n"
            newline_count += 1
        else: # variable changes so keep in legend
            legend_params.append(key)
    return titlestr, legend_params



#=================#
# Perform Program #
#=================#
if int(len(sys.argv)) != 2:
    sys.exit("Usage: python <RUN_TYPE>")

match(sys.argv[1]):
    case "spheresInCircle":
        filename = "SingleLaguerre_SphereVary"
        #1,2,3,4,5,6,7,8,9,10,11,12
        particle_numbers = [1,2,3,4,5,6,7,8,9,10,11,12]
        parameter_text = simulations_singleFrame_optForce_spheresInCircle(particle_numbers, filename, include_additionalForces=False)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, particle_numbers, "Particle number", "", parameter_text)
        Display.plot_tangential_force_against_number_averaged(filename+"_combined_data", parameter_text)
    case "torusInCircle":
        filename = "SingleLaguerre_TorusVary"
        particle_numbers = [2,3,4,5,6,7,8,9]
        parameter_text = simulations_singleFrame_optForce_torusInCircle(particle_numbers, filename)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, particle_numbers, "Particle number", "", parameter_text)
    case "torusInCircleFixedPhi":
        filename = "SingleLaguerre_TorusVary"
        particle_numbers = [1,2,3,4,5,6,7,8,9,10,11,12]
        parameter_text = simulations_singleFrame_optForce_torusInCircleFixedPhi(particle_numbers, filename)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, particle_numbers, "Particle number", "", parameter_text)
    case "spheresInCircleSlider":
        filename = "SingleLaguerre_SphereVary"
        theta_range = [np.pi/6.0, np.pi, 50] #np.pi/2.0, 3.0*np.pi/2.0,
        parameter_text = simulations_singleFrame_optForce_spheresInCircleSlider(1, theta_range, filename)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, np.linspace(*theta_range), "slider theta", "(radians)", parameter_text)
    case "spheres_wavelengthTrial":
        filename      = "SingleLaguerre_SphereVary"
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
        filename = "SingleLaguerre_SphereVary"
        particle_total = 12
        dipole_size_range = [50e-9, 150e-9, 20]
        parameter_text, dipole_sizes = simulations_singleFrame_optForce_spheresInCircleDipoleSize(particle_total, dipole_size_range, filename)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, np.linspace(*dipole_size_range), "Dipole size", "(m)", parameter_text)
    case "torusInCircleDipoleSize":
        filename = "SingleLaguerre_TorusVary"
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
        filename = "SingleLaguerre_TorusVary"
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
        
        filename = "SingleLaguerre_TorusVary"
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
        filename = "SingleLaguerre_SphereVary"
        particle_radius  = 100e-9#100e-9
        particle_spacing = 400e-9#60e-9
        bounding_sphere_radius = 2e-6
        connection_mode = "dist"
        connection_args = [2*100e-9 +300e-9] # [2*100e-9 +100e-9]
        parameter_text = simulations_singleFrame_connected_sphereGrid(particle_radius, particle_spacing, bounding_sphere_radius, connection_mode, connection_args, filename)
    case "connected_sphereShell":
        #
        # Currently just runs the simulation for observation, no data is recorded or stored in .xlsx files here
        #
        filename = "SingleLaguerre_SphereVary"
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
        # Save file
        filename = "SingleLaguerre"
        # Args
        chain_length    = 3e-6
        particle_radius = 100e-9
        particle_number = 5
        time_step = 1e-5
        frames = 100
        constants={"spring":5e-7, "bending":0.5e-18}
        force_terms=["optical", "spring", "bending", "buckingham"]

        particle_separation = chain_length/particle_number
        connection_mode = "dist"
        connection_args = 1.1*particle_separation
        # Run
        parameter_text = simulations_fibre_1D_sphere(filename, chain_length, particle_radius, particle_number, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True)
    case "fibre_1D_cylinder":
        # Save file
        filename = "SingleLaguerre"
        # Args
        chain_length    = 3e-6
        particle_length = 300e-9
        particle_radius = 100e-9
        particle_number = 5
        time_step = 1e-5
        frames = 100
        constants={"spring":5e-6, "bending":0.5e-18}
        force_terms=["optical", "spring", "bending", "buckingham"]

        particle_separation = chain_length/particle_number
        connection_mode = "dist"
        connection_args = 1.1*particle_separation
        # Run
        parameter_text = simulations_fibre_1D_cylinder(filename, chain_length, particle_length, particle_radius, particle_number, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True)
    case "fibre_2D_sphere_hollowShell":
        # Save file
        filename = "SingleLaguerre"
        # Args
        chain_length    = 3e-6
        particle_radius = 100e-9
        shell_radius    = 300e-9
        particle_number_radial  = 6
        particle_number_angular = 6
        time_step = 0.5e-4
        frames = 60

        stiffness = 1.0e-6
        constants={"spring":stiffness, "bending":0.1e-18}
        force_terms=["optical", "spring", "bending"]
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
        stiffness_spec = {"type":"beads", "default_value":stiffness, "bead_value":3.0*stiffness, "bead_indices":bead_indices}
        #stiffness_spec = {"type":"", "default_value":stiffness}   # Default version for uniform stiffness

        # Run
        parameter_text = simulations_fibre_2D_sphere_hollowShell(filename, chain_length, shell_radius, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, time_step, constants, force_terms, stiffness_spec, frames, show_output=True, include_beads=include_beads)
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
        time_step = 0.5e-4
        frames = 10
        constants={"spring":5e-6, "bending":0.65e-18}
        force_terms=["optical", "spring", "bending"]

        connection_mode = "dist"
        connection_args = 0.0   # NOTE; Is overwritten in the function to pick the correct value
        # Run
        parameter_text = simulations_fibre_2D_cylinder_hollowShell(filename, chain_length, shell_radius, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True)
    case "fibre_2D_sphere_thick_connectUniform":
        # Save file
        filename = "SingleLaguerre"
        # Args
        chain_length    = 3e-6
        particle_radius = 100e-9
        shell_radius    = 400e-9
        shell_number    = 1
        particle_number_radial  = 10
        particle_number_angular = 6
        time_step = 1e-5
        frames = 50
        constants={"spring":7.5e-6, "bending":0.1e-18}
        force_terms=["optical", "spring", "bending"]

        connection_mode = "dist"
        connection_args = 0.0   # NOTE; Is overwritten in the function to pick the correct value
        # Run
        parameter_text = simulations_fibre_2D_sphere_thick_connectUniform(filename, chain_length, shell_radius, shell_number, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True)
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
        time_step = 1e-5
        frames = 50
        constants={"spring":7.5e-6, "bending":0.1e-18}
        force_terms=["optical", "spring", "bending"]

        connection_mode = "dist"
        connection_args = 0.0   # NOTE; Is overwritten in the function to pick the correct value
        # Run
        parameter_text = simulations_fibre_2D_cylinder_thick_connectUniform(filename, chain_length, shell_radius, shell_number, particle_length, particle_radius, particle_number_radial, particle_number_angular, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True)
    case "fibre_2D_sphere_shellLayers":
        # Save file
        filename = "SingleLaguerre"
        # Args
        chain_length    = 1.0e-6
        particle_radius = 0.15e-6
        shell_radius    = 1.0e-6
        shell_number    = 2
        particle_separation = (np.pi*2.0*shell_radius)/(15.0)
        time_step = 1e-4
        frames = 10
        constants={"spring":7.5e-6, "bending":0.1e-18}
        force_terms=[]#["optical", "spring", "bending", "buckingham"]

        connection_mode = "dist"
        connection_args = 1.01*particle_separation
        # Run
        parameter_text = simulations_fibre_2D_sphere_shellLayers(filename, chain_length, shell_radius, shell_number, particle_radius, particle_separation, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True)
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
        time_step = 1e-4
        frames = 10
        constants={"spring":7.5e-6, "bending":0.1e-18}
        force_terms=[]#["optical", "spring", "bending", "buckingham"]

        connection_mode = "dist"
        connection_args = 1.01*particle_separation
        # Run
        parameter_text = simulations_fibre_2D_cylinder_shellLayers(filename, chain_length, shell_radius, shell_number, particle_length, particle_radius, particle_separation, connection_mode, connection_args, time_step, constants, force_terms, frames, show_output=True)
    case "refine_cuboid":
        # Save file
        filename = "SingleLaguerre"
        # Args
        dimensions  = [1.0e-6, 0.6e-6, 0.6e-6] # Dimensions of each side of the cuboid
        object_offset = [1e-6, 0e-6, 0e-6]     # Offset the whole object
        dipole_size = 40e-9     # 40e-9
        separations = [0.4e-6, 0.2e-6, 0.2e-6]    # Separation in each axis of the cuboid, as a total separation (e.g. more particles => smaller individual separation between each)
        object_offset = [0e-6, 0e-6, 0e-6]     # Offset the whole object
        particle_size = 0.2e-6      # e.g radius of sphere, width of cube
        force_terms=["optical"] # ["optical", "spring", "bending", "buckingham"]
        particle_shape = "cube"

        # Run
        parameter_text, data_set = simulations_refine_cuboid(dimensions, dipole_size, separations, particle_size, force_terms, particle_shape, object_offset, show_output=False)
        # Plot graph here
        datalabel_set = np.array([ 
            "F Mag",
            "Fx",
            "Fy",
            "Fz"
        ])
        # datacolor_set = np.array([ 
        #     "red"
        # ])
        graphlabel_set = {"title":"Title", "xAxis":"Dipole Size (micro m)", "yAxis":"some Y"}
        Display.plot_multi_data(data_set=data_set, datalabel_set=datalabel_set, graphlabel_set=graphlabel_set)  #, datacolor_set=datacolor_set

    case "refine_arch_prism":
        # Save file
        filename = "SingleLaguerre"
        # Args
        show_output     = True
        dimensions      = np.array([2.0e-6, 0.5e-6, 0.5e-6])    # Bounding box for prism
        separations     = np.array([0.0, 0.0, 0.0])
        particle_length = 100e-9                                # Radius or half-width
        dipole_size     = 40e-9
        deflection      = 0.25e-6           # Of centre in micrometres (also is deflection to centre of rod, not underside)
        object_offset   = [-dimensions[0]/2.0, -dimensions[1]/2.0, 0e-6]     # Offset the whole object
        force_terms     = ["optical"]   # ["optical", "spring", "bending", "buckingham"]
        particle_shape  = "sphere"

        # Run
        parameter_text, data_set = simulations_refine_arch_prism(dimensions, separations, particle_length, dipole_size, deflection, object_offset, force_terms, particle_shape, show_output=show_output)
        # Plot graph here
        datalabel_set = np.array([ 
            "|F| (centre)",
            "|F| per dipole (centre)",
            "|F|^2 total mesh"
        ])
        # datacolor_set = np.array([ 
        #     "red"
        # ])
        graphlabel_set = {"title":"Title", "xAxis":"Dipole Size (micro m)", "yAxis":"some Y"}
        Display.plot_multi_data(data_set=data_set, datalabel_set=datalabel_set, graphlabel_set=graphlabel_set)  #, datacolor_set=datacolor_set

    case "refine_cuboid_general":
        # Save file
        filename = "SingleLaguerre"
        # Args
        dimensions  = [1.0e-6, 0.6e-6, 0.6e-6] # Dimensions of each side of the cuboid
        force_terms=["optical"]                # ["optical", "spring", "bending", "buckingham"]

        # Iterables
        dipole_sizes = np.linspace(50e-9, 100e-9, 1)         
        particle_sizes = [0.16e-6] # e.g radius of sphere, half-width of cube
        separations_list = [[0.4e-6, 0.0, 0.0]]  # Separation in each axis of the cuboid, as a total separation (e.g. more particles => smaller individual separation between each)
        particle_shapes = ["cube", "sphere"] 
        object_offsets = [[1e-6, 0e-6, 0e-6], [1e-6, 0e-6, 0.5e-6], [1e-6, 0e-6, 1e-6], [1e-6, 0e-6, 1.5e-6], [1e-6, 0e-6, 2e-6]]

        # dipole_sizes = np.linspace(50e-9, 100e-9, 20)         
        # particle_sizes = [0.15e-6, 0.1e-6] # e.g radius of sphere, half-width of cube
        # separations_list = [[0.4e-6, 0.0, 0.0]]  # Separation in each axis of the cuboid, as a total separation (e.g. more particles => smaller individual separation between each)
        # particle_shapes = ["cube", "sphere"] # ["cube", "sphere"]   
        # object_offsets = [[1e-6, 0e-6, 0e-6], [1e-6, 0e-6, 1e-6]]     # Offset the whole object 

        variables_list = {
            "indep_var": "object_offsets", # Must be one of the other keys: dipole_sizes, separations_list, particle_sizes, particle_shapes
            "dipole_sizes": dipole_sizes,
            "separations_list": separations_list,
            "particle_sizes": particle_sizes,
            "particle_shapes": particle_shapes,
            "object_offsets": object_offsets
        }

        # Only used for when indep var is a vector (e.g.object_offsets): Set what component to plot against
        indep_name = variables_list["indep_var"]
        if indep_name == "separations_list": indep_vector_component = 0
        elif indep_name == "object_offsets": indep_vector_component = 2
        else: indep_vector_component = 0

        # Run
        parameter_text, data_set, data_set_params = simulations_refine_general(dimensions, variables_list, force_terms, show_output=False, indep_vector_component=indep_vector_component)

        # Format output and make legend/title strings
        force_filter=["Fx"] # options are ["Fmag","Fx", "Fy", "Fz"]
        titlestr, legend_params = get_titlelegend(variables_list, indep_name)
        data_set, datalabel_set = filter_data_set(force_filter, data_set, data_set_params, legend_params, indep_name)
        
        # Plot graph here
        # linestyle_set = np.repeat(["-", "--", ":", "-."], int(len(data_set)/4))
        # print(f"DATASET IS {data_set}\ndatalabel_set is {datalabel_set}\ndata_set_params is {data_set_params}")
        linestyle_set = np.repeat(["-"], len(data_set))
        xAxis_varname, xAxis_units = display_var(indep_name)
        graphlabel_set = {"title":titlestr, "xAxis":f"{xAxis_varname} {xAxis_units}", "yAxis":"Force /N"} # single quotes needed to prevent strings clashing.
        Display.plot_multi_data(data_set, datalabel_set, graphlabel_set=graphlabel_set, linestyle_set=linestyle_set)

    case _:
        print("Unknown run type: ",sys.argv[1]);
        print("Allowed run types are; 'spheresInCircle', 'torusInCircle', 'torusInCircleFixedPhi', 'spheresInCircleSlider', 'spheresInCircleDipoleSize', 'torusInCircleDipoleSize', 'testVolumes', 'connected_sphereGrid', 'connected_sphereShell'")
