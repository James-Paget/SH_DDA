"""
Runs the core simulation using various YAML files to produce a dataset 
from a single python run
"""

import sys
import subprocess
import numpy as np
import xlsxwriter
import pandas as pd
import random
import math

import Display


def generate_yaml(filename, particle_list, parameters_arg):

    # All possible parameters_arg keys as strings are: 
    # frames, wavelength, dipole_radius, time_step, vmd_output, excel_output, include_force, include_couple, show_output, frame_interval,
    # max_size, resolution, frame_min, frame_max, z_offset, beamtype, E0, order, w0, jones, translation, rotation, default_radius, default_material

    # particle_list contains dictionaries with keys: material, shape, args, coords, altcolour

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
        "E0": 300,
        "order": 3,
        "w0": 0.6,
        "jones": "POLARISATION_LCP",
        "translation": None,
        "rotation": None,

        "default_radius": 100e-9,
        "default_material": "FusedSilica",
    }

    # Overwrite parameters with any passed in with parameters_arg
    parameters.update(parameters_arg)
    
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

    file.write("beams:\n")
    file.write("  beam_1:\n")
    for arg in ["beamtype", "E0", "order", "w0", "jones", "translation", "rotation"]:
        file.write(f"    {arg}: {parameters[arg]}\n")

    file.write("particles:\n")
    for arg in ["default_radius", "default_material"]:
        file.write(f"  {arg}: {parameters[arg]}\n")


    # Write particle list.
    file.write("  particle_list:\n")
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

def simulations_singleFrame_optForce_spheresInCircle(particle_numbers, filename, include_additionalForces=False):
    #
    # Performs a DDA calculation for various particles in a circular ring on the Z=0 plane
    #
    # particle_numbers = list of particle numbers to be tested in sphere e.g. [1,2,3,4,8]
    #
    
    particle_info = [];
    place_radius = 1.152e-6         #1.15e-6
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
    
    particle_info = []
    place_radius = 1.15e-6      #1.15e-6
    particle_radii = 200e-9     #200e-9
    frames_of_animation = 1

    parameters = {"frames": frames_of_animation, "frame_max": frames_of_animation, "show_output": False}
    dipole_sizes = np.linspace(*dipole_size_range) # unpack list to fill the 3 arguments

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


def simulations_singleFrame_optForce_torusInCircleDipoleSize(particle_total, dipole_size_range, filename):
    #
    # Performs a DDA calculation for particles in a circular ring for various dipole sizes. 
    #
    # dipole_size_range = [size_min, size_max, num]
    #
    
    particle_info = []
    inner_radii = 1.15e-6
    tube_radii  = 200e-9
    separating_dist  = 0.1e-6
    frames_of_animation = 1

    parameters = {"frames": frames_of_animation, "frame_max": frames_of_animation, "show_output": False}
    dipole_sizes = np.linspace(*dipole_size_range) # unpack list to fill the 3 arguments
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


def simulations_singleFrame_optForce_torusInCircleSeparation(particle_total, separation_range, filename):
    #
    # Performs a DDA calculation for particles in a circular ring for various dipole sizes. 
    #
    # separation_range = [sep_min, sep_max, num]
    #
    
    particle_info = []
    inner_radii = 1.15e-6
    tube_radii  = 200e-9
    frames_of_animation = 1

    parameters = {"frames": frames_of_animation, "frame_max": frames_of_animation, "show_output": False}
    separations = np.linspace(*separation_range) # unpack list to fill the 3 arguments
 
    # For each scenario to be tested
    for i, separation in enumerate(separations):
        print(f"\n{i}/{len(separations)}: Performing calculation for dipole size {separation}")

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
            parameters = {"frames": frames_of_animation, "frame_max": frames_of_animation, "dipole_radius": dipoleSize_num, "show_output": True}
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


#=================#
# Perform Program #
#=================#
if int(len(sys.argv)) != 2:
    sys.exit("Usage: python <RUN_TYPE>")

match(sys.argv[1]):
    case "spheresInCircle":
        filename = "SingleLaguerre_SphereVary"
        #1,2,3,4,5,6,7,8,9,10,11,12
        particle_numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        parameter_text = simulations_singleFrame_optForce_spheresInCircle(particle_numbers, filename, include_additionalForces=True)
        # Display.plot_tangential_force_against_number(filename+"_combined_data", 0, parameter_text)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, particle_numbers, "Particle number", "", parameter_text)
        Display.plot_tangential_force_against_number_averaged(filename+"_combined_data", parameter_text)
    case "torusInCircle":
        filename = "SingleLaguerre_TorusVary"
        particle_numbers = [2,3,4,5,6,7,8,9,10,11,12]
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
        x_values = np.arange(wave_start, abs(target_pos[0])+beam_radius, wave_jump) /wavelength # XXX BASED ON: [wave_start, abs(target_pos[0])+beam_radius, wave_jump]
        #NOTE; Make sure the start is a multiple of jump in order for constructive to be nice
        parameter_text = simulations_singleFrame_optForce_wavelengthTrial(wave_start, wave_jump, beam_radius, target_pos, target_radius, filename, wavelength=wavelength, reducedSet=2)
        # OLD_ARB_CALL: Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, wave_jump/wavelength, "Wave spacing (wavelengths)", parameter_text=parameter_text)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, x_values, "Wave spacing", "(wavelengths)", parameter_text)
    case "spheresInCircleDipoleSize":
        filename = "SingleLaguerre_SphereVary"
        particle_total = 12
        dipole_size_range = [6e-8, 4e-8, 5]
        parameter_text, dipole_sizes = simulations_singleFrame_optForce_spheresInCircleDipoleSize(particle_total, dipole_size_range, filename)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, np.linspace(*dipole_size_range), "Dipole size", "(m)", parameter_text)
    case "torusInCircleDipoleSize":
        filename = "SingleLaguerre_TorusVary"
        particle_total = 6
        dipole_size_range = [40e-9, 100e-9, 30]
        parameter_text, dipole_sizes = simulations_singleFrame_optForce_torusInCircleDipoleSize(particle_total, dipole_size_range, filename)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, np.linspace(*dipole_size_range), "Dipole size", "(m)", parameter_text)
    case "torusInCircleSeparation":
        filename = "SingleLaguerre_TorusVary"
        particle_total = 6
        separation_range = [20e-9, 300e-9, 40]
        parameter_text, dipole_sizes = simulations_singleFrame_optForce_torusInCircleSeparation(particle_total, separation_range, filename)
        Display.plot_tangential_force_against_arbitrary(filename+"_combined_data", 0, np.linspace(*separation_range), "Separation", "(m)", parameter_text)
    case "torusInCircle_FixedSep_SectorDipole":
        #
        # Compares force for different sector numbers, for a dipole sizes
        #

        #particle_numbers = [2,3,4,5,6,7,8, 12, 16]
        #dipoleSize_numbers = [40e-9, 50e-9, 60e-9, 70e-9] #np.linspace(...)
        #separation = 300e-9
        
        filename = "SingleLaguerre_TorusVary"
        particle_numbers = [1,2,3,4,5,6,7]
        dipoleSize_numbers = [40e-9, 50e-9, 60e-9, 70e-9] #np.linspace(...)
        data_axes = [dipoleSize_numbers, particle_numbers]
        separation = 0.0e-9
        parameter_text, data_set = simulations_singleFrame_optForce_torusInCircle_FixedSep_SectorDipole(particle_numbers, dipoleSize_numbers, separation, filename)
        Display.plotMulti_tangential_force_against_arbitrary(data_set, data_axes, 0, ["Dip.Rad", "Particle Number"], ["(m)", ""], parameter_text)
    case _:
        print("Unknown run type: ",sys.argv[1]);
        print("Allowed run types are; 'spheresInCircle', 'torusInCircle', 'torusInCircleFixedPhi', 'spheresInCircleSlider', 'spheresInCircleDipoleSize', 'torusInCircleDipoleSize, 'torusInCircle_FixedSep_SectorDipole'")
