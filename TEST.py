import numpy as np

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
            particle_step[i] = 2.0*particle_size + separations[i]/(particle_numbers[i]-1)                       # Step in position to each particle in each axis
    
    for i in range(int(particle_numbers[0])):
        for j in range(int(particle_numbers[1])):
            for k in range(int(particle_numbers[2])):
                coords_list.append([i*particle_step[0] -dimensions[0]/2.0+particle_size, j*particle_step[1] -dimensions[1]/2.0+particle_size, k*particle_step[2] -dimensions[2]/2.0+particle_size])

    return coords_list

print(len(get_refine_cuboid(dimensions=[1,1,2], separations=[0,0,0], particle_size=0.25)))