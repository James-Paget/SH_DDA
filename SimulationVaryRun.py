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

import Display


def generate_sphere_yaml(particle_formation, number_of_particles, particle_material="FusedSilica", characteristic_distance=1e-6, particle_radii = 200e-9):
    #
    # Generates a YAML file for a set of identical spheres with given parameters
    # This will overwrite files with the same name
    #
    # number_of_particles = Number of spheres to generate
    # characteristic_distance = key distance for each formation, e.g. radi of circle particles are placed on, length of edge for cubic formation, etc
    #
    
    # Create / overwrite YAML file
    # Writing core system parameters
    print("Generated sphere YAML")
    file = open("SingleLaguerre_SphereVary.yml", "w")
    
    file.write("options:\n")
    file.write("  frames: 1\n")

    file.write("parameters:\n")
    file.write("  wavelength: 1.0e-6\n")
    file.write("  dipole_radius: 40e-9\n")
    file.write("  time_step: 1e-4\n")

    file.write("output:\n")
    file.write("  vmd_output: True\n")
    file.write("  excel_output: True\n")
    file.write("  include_force: True\n")
    file.write("  include_couple: True\n")

    file.write("display:\n")
    file.write("  show_output: True\n")
    file.write("  frame_interval: 2\n")
    file.write("  max_size: 2e-6\n")
    file.write("  resolution: 201\n")
    file.write("  frame_min: 0\n")
    file.write("  frame_max: 1\n")
    file.write("  z_offset: 0.0e-6\n")

    file.write("beams:\n")
    file.write("  beam_1:\n")
    file.write("    beamtype: BEAMTYPE_LAGUERRE_GAUSSIAN\n")
    file.write("    E0: 300\n")
    file.write("    order: 3\n")
    file.write("    w0: 0.6\n")
    file.write("    jones: POLARISATION_LCP\n")
    file.write("    translation: None\n")
    file.write("    rotation: None\n")

    file.write("particles:\n")
    file.write("  default_radius: 100e-9\n")
    file.write("  default_material: FusedSilica\n")
    file.write("  particle_list:\n")

    # Writing specific parameters for particle formation
    for particle_index in range(number_of_particles):
        match(particle_formation):
            case "circle":
                theta_jump = (2.0*np.pi)/number_of_particles
                particle_theta = theta_jump*particle_index
                particle_position = [characteristic_distance*np.cos(particle_theta), characteristic_distance*np.sin(particle_theta), 1.0e-6]
                position_offsets  = [
                    0.0,#random.random()*0.02*characteristic_distance, 
                    0.0,#random.random()*0.02*characteristic_distance, 
                    0.0#random.random()*0.02*characteristic_distance
                ]
                file.write("    part_"+str(2*particle_index)+":\n")
                file.write("      material: "+str(particle_material)+"\n")
                file.write("      shape: sphere\n")
                file.write("      args: "+str(particle_radii)+"\n")
                file.write("      coords: "+str(particle_position[0] +position_offsets[0])+" "+str(particle_position[1] +position_offsets[1])+" "+str(1.0e-6 +position_offsets[2])+"\n")
                file.write("      altcolour: True\n")
            case _:
                print("Particle formation invalid: ",particle_formation);

    file.close()

def generate_torus_yaml(number_of_particles, inner_radii, tube_radii, separating_dist, particle_material="FusedSilica"):
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
    print("Generated torus YAML")
    file = open("SingleLaguerre_TorusVary.yml", "w")
    
    file.write("options:\n")
    file.write("  frames: 10\n")

    file.write("parameters:\n")
    file.write("  wavelength: 1.0e-6\n")
    file.write("  dipole_radius: 40e-9\n")
    file.write("  time_step: 1e-4\n")

    file.write("output:\n")
    file.write("  vmd_output: True\n")
    file.write("  excel_output: True\n")
    file.write("  include_force: True\n")
    file.write("  include_couple: True\n")

    file.write("display:\n")
    file.write("  show_output: True\n")
    file.write("  frame_interval: 2\n")
    file.write("  max_size: 2e-6\n")
    file.write("  resolution: 201\n")
    file.write("  frame_min: 0\n")
    file.write("  frame_max: 10\n")
    file.write("  z_offset: 0.0e-6\n")

    file.write("beams:\n")
    file.write("  beam_1:\n")
    file.write("    beamtype: BEAMTYPE_LAGUERRE_GAUSSIAN\n")
    file.write("    E0: 300\n")
    file.write("    order: 3\n")
    file.write("    w0: 0.6\n")
    file.write("    jones: POLARISATION_LCP\n")
    file.write("    translation: None\n")
    file.write("    rotation: None\n")

    file.write("particles:\n")
    file.write("  default_radius: 100e-9\n")
    file.write("  default_material: FusedSilica\n")
    file.write("  particle_list:\n")

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
        file.write("    part_"+str(2*particle_index)+":\n")
        file.write("      material: "+str(particle_material)+"\n")
        file.write("      shape: torus\n")
        file.write("      args: "+str(inner_radii)+" "+str(tube_radii)+" "+str(lower_phi)+" "+str(upper_phi)+"\n")
        file.write("      coords: "+str(particle_position[0] +position_offsets[0])+" "+str(particle_position[1] +position_offsets[1])+" "+str(particle_position[2] +position_offsets[2])+"\n")
        file.write("      altcolour: True\n")

    file.close()

def simulations_singleFrame_optForce_spheresInCircle(particle_numbers, filename):
    #
    # Performs a DDA calcualtion for various particles in a circular ring on the Z=0 plane
    #
    # particle_numbers = list of particle numbers to be tested in sphere e.g. [1,2,3,4,8]
    #
    
    particle_info = [];
    #For each scenario to be tested
    for particle_number in particle_numbers:
        print("")
        print("Performing calculation for "+str(particle_number)+" particles")
        #Generate required YAML, perform calculation, then pull force data
        generate_sphere_yaml("circle", particle_number, characteristic_distance=1.15e-6)     # Writes to SingleLaguerre_SphereVary.yml

        #Run DipolesMulti2024Eigen.py
        run_command = "python DipolesMulti2024Eigen.py "+filename
        run_command = run_command.split(" ")
        print("=== Log ===")
        result = subprocess.run(run_command, stdout=subprocess.DEVNULL) #, stdout=subprocess.DEVNULL

        #Pull data from xlsx into a local list in python
        record_particle_info(particle_info)
    #Write combined data to a new xlsx file
    store_combined_particle_info(particle_info, filename)

def simulations_singleFrame_optForce_torusInCircle(particle_numbers, filename):
    #
    # Performs a DDA calcualtion for various particles in a circular ring on the Z=0 plane
    #
    # particle_numbers = list of particle numbers to be tested in sphere e.g. [1,2,3,4,8]
    #
    
    particle_info = [];
    #For each scenario to be tested
    for particle_number in particle_numbers:
        print("")
        print("Performing calculation for "+str(particle_number)+" particles")
        #Generate required YAML, perform calculation, then pull force data
        generate_torus_yaml(particle_number, 1.15e-6, 200e-9, 0.3e-6)     # Writes to <filename>.yml

        #Run DipolesMulti2024Eigen.py
        run_command = "python DipolesMulti2024Eigen.py "+filename
        run_command = run_command.split(" ")
        print("=== Log ===")
        result = subprocess.run(run_command, stdout=subprocess.DEVNULL) #, stdout=subprocess.DEVNULL

        #Pull data from xlsx into a local list in python
        record_particle_info(particle_info)
    #Write combined data to a new xlsx file
    store_combined_particle_info(particle_info, filename)

def record_particle_info(particle_info):
    #
    # Store key details about particle from xlsx into a data structure here
    # This information is stored in particle_info (altered by reference)
    #
    # Data is stored as follows;
    #   [ [scenario1], [scenario2],... ]
    # where [scenarioN] = [x1, y1, z1, Fx1, Fy1, Fz1, ..., xi, yi, zi, Fxi, Fyi, Fzi], for the <i> particles involved in the scenario
    #
    info = []
    data = pd.read_excel("SingleLaguerre_SphereVary.xlsx")
    particle_number = int(np.floor( ( len(data.iloc[0])-1 )/(3.0*3.0) ))
    for i in range(particle_number):
        # For each particle, fetch its (x,y,z,Fx,Fy,Fz)
        info.append( data.iloc[0, 1 +3*(i)] )    #X
        info.append( data.iloc[0, 2 +3*(i)] )    #Y
        info.append( data.iloc[0, 3 +3*(i)] )    #Z
        info.append( data.iloc[0, 1 +3*(i+particle_number)] ) #Fx
        info.append( data.iloc[0, 2 +3*(i+particle_number)] ) #Fy
        info.append( data.iloc[0, 3 +3*(i+particle_number)] ) #Fz
    particle_info.append(info)

def store_combined_particle_info(particle_info, filename):
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
    worksheet.write(0,0, "x1")
    worksheet.write(0,1, "y1")
    worksheet.write(0,2, "z1")
    worksheet.write(0,3, "Fx1")
    worksheet.write(0,4, "Fy1")
    worksheet.write(0,5, "Fz1")
    worksheet.write(0,6, "...")

    # Fill in data stored from particle_info
    for j in range( len(particle_info) ):
        for i in range( len(particle_info[j]) ):
            worksheet.write(j+1, i, particle_info[j][i])

    workbook.close()

#=================#
# Perform Program #
#=================#
if int(len(sys.argv)) != 2:
    sys.exit("Usage: python <RUN_TYPE>")

match(sys.argv[1]):
    case "spheresInCircle":
        #simulations_singleFrame_optForce_spheresInCircle([1,2,4,8,12], "SingleLaguerre_SphereVary");
        Display.plot_tangential_force_against_number("SingleLaguerre_SphereVary_combined_data", 0)
    case "torusInCircle":
        simulations_singleFrame_optForce_torusInCircle([10], "SingleLaguerre_TorusVary");
    case _:
        print("Unknown run type: ",sys.argv[1]);


"""
====
TODO
====
(1) NEED TO FIX THE PROBLEM WITH PARTICLES AT FRAME 1 HAVING A NON-NORMAL FORCE --> TAKE CMOP / MOVE OUT INITIALLY
    --> Jiggle and take averages perhaps --> using brownian motion force stepper
(2) Do for torus too
(3) Do plots for results gathered --> tangential force plots for 0th (the un-moving particle)
(4) comapre magnitudes of tangential for other particles too
        -> do for both sphere and torus case
"""