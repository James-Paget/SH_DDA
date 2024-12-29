#
# Output functions
#
import datetime
import socket
import xlsxwriter

def make_vmd_file(filename_vtf,n_particles,frames,timestep,particles,optpos,beam_collection,duration,radius,dipole_radius,z_offset,particle_types,vtfcolors):
    """
    Function to generate VMD output file.
    Inputs:
    filename_vtf (string): filename with .vtf extension for
      storing the frames.
    beam_collection (Ctypes struct array): contains all the information regarding the beams used.
    positions, particle definitions
    duration (float): time elapsed in seconds.
    Outputs:
    None
    """
    
    MyFileObject = open(filename_vtf,"w",)
    n_beams = beam_collection.beams
    print("####################################################################", file=MyFileObject)
    print("# Output from multi-bead simulation", file=MyFileObject)
    now = datetime.datetime.now()
    print("# File written: {:s}".format(now.strftime("%Y-%m-%d %H:%M:%S")), file=MyFileObject)
    print("# Elapsed time: {:8.6f} s".format(duration),file=MyFileObject)
    print("# System: {:s}".format(socket.gethostname()),file=MyFileObject)
    print("####################################################################", file=MyFileObject)
    print("# Number of beams: {:d}".format(n_beams), file=MyFileObject)
    print("# Number of particles: {:d}".format(n_particles), file=MyFileObject)
    print("# Particle radius (m): {:e}".format(radius), file=MyFileObject)
    print("# Dipole radius (m): {:e}".format(dipole_radius), file=MyFileObject)
    print("# z-offset for plot (m): {:e}".format(z_offset), file=MyFileObject)
    print("####################################################################", file=MyFileObject)
    print("# Number of timesteps: {:d}".format(frames), file=MyFileObject)
    print("# Time step (s): {:e}".format(timestep), file=MyFileObject)
    print("####################################################################", file=MyFileObject)
    print("# Beam type parameters:", file=MyFileObject)
    print("#   BEAMTYPE_PLANE = 0", file=MyFileObject)
    print("#   BEAMTYPE_GAUSS_BARTON5 = 1", file=MyFileObject)
    print("#   BEAMTYPE_GAUSS_CSP = 2", file=MyFileObject)
    print("#   BEAMTYPE_BESSEL = 3", file=MyFileObject)
    print("#   BEAMTYPE_LAGUERRE_GAUSSIAN = 4", file=MyFileObject)
    print("####################################################################", file=MyFileObject)
    for i in range(n_beams):
        print("# Beam number: {:d}".format(i), file=MyFileObject)
        print("#  -beamtype = {:d}".format(beam_collection.BEAM_ARRAY[i].beamtype), file=MyFileObject)
        print("#  -E0 = {:f}".format(beam_collection.BEAM_ARRAY[i].E0), file=MyFileObject)
        print("#  -k = {:f}".format(beam_collection.BEAM_ARRAY[i].k), file=MyFileObject)
        print("#  -kz = {:f}".format(beam_collection.BEAM_ARRAY[i].kz), file=MyFileObject)
        print("#  -kt = {:f}".format(beam_collection.BEAM_ARRAY[i].kt), file=MyFileObject)
        print("#  -kt_by_kz = {:f}".format(beam_collection.BEAM_ARRAY[i].kt_by_kz), file=MyFileObject)
        print("#  -order = {:d}".format(beam_collection.BEAM_ARRAY[i].order), file=MyFileObject)
        print("#  -jones = {:f} {:f} {:f} {:f}".format(beam_collection.BEAM_ARRAY[i].jones[0], beam_collection.BEAM_ARRAY[i].jones[1], beam_collection.BEAM_ARRAY[i].jones[2], beam_collection.BEAM_ARRAY[i].jones[3]), file=MyFileObject)
        print("#  -translation = {:e} {:e} {:e}".format(beam_collection.BEAM_ARRAY[i].translation[0], beam_collection.BEAM_ARRAY[i].translation[1], beam_collection.BEAM_ARRAY[i].translation[2]), file=MyFileObject)
        print("#  -rotation = {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}".format(beam_collection.BEAM_ARRAY[i].rotation[0], beam_collection.BEAM_ARRAY[i].rotation[1], beam_collection.BEAM_ARRAY[i].rotation[2], beam_collection.BEAM_ARRAY[i].rotation[3], beam_collection.BEAM_ARRAY[i].rotation[4], beam_collection.BEAM_ARRAY[i].rotation[5], beam_collection.BEAM_ARRAY[i].rotation[6], beam_collection.BEAM_ARRAY[i].rotation[7], beam_collection.BEAM_ARRAY[i].rotation[8]), file=MyFileObject)
        print("#  -w0 = {:e}".format(beam_collection.BEAM_ARRAY[i].w0), file=MyFileObject)
        print("####################################################################", file=MyFileObject)

    for i in range(n_particles):
        print("atom {:d} radius {:4.2f} name {:s}".format(i,1.25*radius*1e6,vtfcolors[i]), file=MyFileObject)
        """        if particle_types[i] == 'Silicon':
            print("atom {:d} radius {:4.2f} name S".format(i,1.25*radius*1e6), file=MyFileObject)
        elif particle_types[i] == 'Sapphire':
            print("atom {:d} radius {:4.2f} name O".format(i,1.25*radius*1e6), file=MyFileObject)
        elif particle_types[i] == 'Glass':
            print("atom {:d} radius {:4.2f} name N".format(i,1.25*radius*1e6), file=MyFileObject)
        elif particle_types[i] == 'Low':
            print("atom {:d} radius {:4.2f} name H".format(i,1.25*radius*1e6), file=MyFileObject)
        elif particle_types[i] == 'FusedSilica':
            print("atom {:d} radius {:4.2f} name B".format(i,1.25*radius*1e6), file=MyFileObject)
        elif particle_types[i] == 'Diamond':
            print("atom {:d} radius {:4.2f} name C".format(i,1.25*radius*1e6), file=MyFileObject)
        elif particle_types[i] == 'SiN':
            print("atom {:d} radius {:4.2f} name P".format(i,1.25*radius*1e6), file=MyFileObject)
        """
    for i in range(0, frames, 1):
        print("\n", file=MyFileObject)
        print("timestep", file=MyFileObject)

        for j in range(n_particles):
            print(
                "{:.4f} {:.4f} {:.4f}".format(
                    particles[j][0][i] * 1e6,
                    particles[j][1][i] * 1e6,
                    particles[j][2][i] * 1e6,
                ),
                file=MyFileObject,
            )

    MyFileObject.close()  # closes the file again
    return

def make_excel_file(filename_xl,n_particles,frames,timestep,particles,optpos,include_force,optforces,totforces,include_couple,optcouples):
    """
    Function to generate excel output file.
    Inputs:
    filename_xl (string): filename with .xlsx extension for
      storing the positions etc.
    Outputs:
    None
    """

    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(filename_xl)
    worksheet = workbook.add_worksheet()

    # Start from the first cell. Rows and columns are zero indexed.
    worksheet.write(0,0,"time(s)")

    for j in range (n_particles):
        worksheet.write(0,j*3+1,"x{:d}(m)".format(j))
        worksheet.write(0,j*3+2,"y{:d}(m)".format(j))
        worksheet.write(0,j*3+3,"z{:d}(m)".format(j))
    if include_force==True:
        for j in range (n_particles):
            # Optical force
            worksheet.write(0,(j+1*n_particles)*3+1,"Fx{:d}(N)".format(j))
            worksheet.write(0,(j+1*n_particles)*3+2,"Fy{:d}(N)".format(j))
            worksheet.write(0,(j+1*n_particles)*3+3,"Fz{:d}(N)".format(j))
            # Total force
            worksheet.write(0,(j+2*n_particles)*3+1,"F_Tx{:d}(N)".format(j))
            worksheet.write(0,(j+2*n_particles)*3+2,"F_Ty{:d}(N)".format(j))
            worksheet.write(0,(j+2*n_particles)*3+3,"F_Tz{:d}(N)".format(j))
    if include_couple==True:
        offset = 3*n_particles
        if include_force==False:
            offset = n_particles
        for j in range (n_particles):
            worksheet.write(0,(j+offset)*3+1,"Cx{:d}(Nm)".format(j))
            worksheet.write(0,(j+offset)*3+2,"Cy{:d}(Nm)".format(j))
            worksheet.write(0,(j+offset)*3+3,"Cz{:d}(Nm)".format(j))

    # Iterate over the data and write it out row by row.
    for i in range(0, frames, 1):
        worksheet.write(i+1,0,timestep*i)
        for j in range (n_particles):
            for k in range (3):
                worksheet.write(i+1,j*3+k+1,optpos[i][j][k])
        if include_force==True:
            for j in range (n_particles):
                for k in range (3):
                    worksheet.write(i+1,(j+1*n_particles)*3+k+1,optforces[i][j][k])
                    worksheet.write(i+1,(j+2*n_particles)*3+k+1,totforces[i][j][k])
        if include_couple==True:
            offset = 3*n_particles
            if include_force==False:
                offset = n_particles
            for j in range (n_particles):
                for k in range (3):
                    worksheet.write(i+1,(j+offset)*3+k+1,optcouples[i][j][k])

    workbook.close()
    return

def append_to_excel_file(filename_xl,n_particles,frames,timestep,particles,optpos,include_force,optforces,include_couple,optcouples):
    """
    Adds elements to an existsing excel file
    Used by SimulationVaryRun.py to build excel files from multiple setups
    """

    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(filename_xl)
    worksheet = workbook.add_worksheet()

    # Start from the first cell. Rows and columns are zero indexed.
    worksheet.write(0,0,"time(s)")

    for j in range (n_particles):
        worksheet.write(0,j*3+1,"x{:d}(m)".format(j))
        worksheet.write(0,j*3+2,"y{:d}(m)".format(j))
        worksheet.write(0,j*3+3,"z{:d}(m)".format(j))
    if include_force==True:
        for j in range (n_particles):
            worksheet.write(0,(j+n_particles)*3+1,"Fx{:d}(N)".format(j))
            worksheet.write(0,(j+n_particles)*3+2,"Fy{:d}(N)".format(j))
            worksheet.write(0,(j+n_particles)*3+3,"Fz{:d}(N)".format(j))
    if include_couple==True:
        offset = 2*n_particles
        if include_force==False:
            offset = n_particles
        for j in range (n_particles):
            worksheet.write(0,(j+offset)*3+1,"Cx{:d}(Nm)".format(j))
            worksheet.write(0,(j+offset)*3+2,"Cy{:d}(Nm)".format(j))
            worksheet.write(0,(j+offset)*3+3,"Cz{:d}(Nm)".format(j))

    # Iterate over the data and write it out row by row.
    for i in range(0, frames, 1):
        worksheet.write(i+1,0,timestep*i)
        for j in range (n_particles):
            for k in range (3):
                worksheet.write(i+1,j*3+k+1,optpos[i][j][k])
        if include_force==True:
            for j in range (n_particles):
                for k in range (3):
                    worksheet.write(i+1,(j+n_particles)*3+k+1,optforces[i][j][k])
        if include_couple==True:
            offset = 2*n_particles
            if include_force==False:
                offset = n_particles
            for j in range (n_particles):
                for k in range (3):
                    worksheet.write(i+1,(j+offset)*3+k+1,optcouples[i][j][k])

    workbook.close()
    return
    
