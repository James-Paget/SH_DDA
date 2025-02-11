"""
Make a display object for animations etc
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import pandas as pd
import sys
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import Beams


class DisplayObject (object):
    defaults = {"show_output":'True',
                "frame_interval":10,
                "max_size":3.2e-6,
                "resolution":201,
                "frame_min":0,
                "frame_max":1000, # ignore and default to "frames"
                "z_offset":'0.0e-6'
                }

    def __init__(self,displayinfo,frames):
        if displayinfo==None:
            # Set defaults
            self.show_output = bool(DisplayObject.defaults['show_output'])
            self.frame_interval = int(DisplayObject.defaults['frame_interval'])
            self.max_size = float(DisplayObject.defaults['max_size']) # range will be 2 times this
            self.resolution = int(DisplayObject.defaults['resolution']) # number of points in each direction of plot
            self.frame_min = int(DisplayObject.defaults['frame_min']) # starting frame for animation
            self.frame_max = frames # will default to number of frames
            self.z_offset = float(DisplayObject.defaults['z_offset']) # this is the z value for the intensity plot
        else:
            # Read from file
            self.show_output = bool(displayinfo.get('show_output',DisplayObject.defaults['show_output']))
            self.frame_interval = int(displayinfo.get('frame_interval',DisplayObject.defaults['frame_interval']))
            self.max_size = float(displayinfo.get('max_size',DisplayObject.defaults['max_size']))
            self.resolution = int(displayinfo.get('resolution',DisplayObject.defaults['resolution']))
            self.frame_min = abs(int(displayinfo.get('frame_min',DisplayObject.defaults['frame_min'])))
            self.frame_max = min(frames,int(displayinfo.get('frame_max',frames)))
            self.z_offset = float(displayinfo.get('z_offset',DisplayObject.defaults['z_offset']))


    def get_intensity_points(self, beam_collection, n=None):
        # Prepare beam surface values
        if n == None:
            nx = self.resolution
        else:
            nx = n
        ny = nx
        Ex = np.zeros((nx, ny), dtype=complex)
        Ey = np.zeros((nx, ny), dtype=complex)
        Ez = np.zeros((nx, ny), dtype=complex)
        E = np.zeros(3,dtype=np.complex128)
        upper = self.max_size
        lower = -upper
        x = np.linspace(lower, upper, nx)
        y = np.linspace(lower, upper, ny)
        X, Y = np.meshgrid(x, y)
        for j in range(ny):
            for i in range(nx):
                Beams.all_incident_fields((x[i], x[j], self.z_offset), beam_collection, E)
                Ex[j][i] = E[0]
                Ey[j][i] = E[1]
                Ez[j][i] = E[2]

        Z = np.zeros(X.shape) + self.z_offset
        I = np.square(np.abs(Ex)) + np.square(np.abs(Ey)) + np.square(np.abs(Ez))
        I0 = np.max(I)
        return X, Y, Z, I, I0

    def plot_intensity(self, beam_collection):
        #I = []
        #fig, ax = plt.subplots(1, num_plots)
        upper = self.max_size
        lower = -upper
        _,_,_, I, I0 = self.get_intensity_points(beam_collection)
        print(np.shape(I))

        fig = plt.figure()
        ax = plt.axes(xlim=(lower, upper), ylim=(lower, upper))

        ax.set_aspect('equal','box')
        #cs=ax.contourf(X, Y, I, cmap=cm.viridis, levels=30)
        #cs=ax.imshow(I[k],cmap=cm.summer)
        extents = (lower,upper,lower,upper)
        cs=ax.imshow(I,cmap=cm.viridis,vmin=0.0,vmax=I0,origin="lower",extent=extents)
        ax.set_aspect('equal','box')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        cbar = fig.colorbar(cs)
        # ax.set_title("z = {:.1e}".format(z[k]))
        return fig,ax


    def animate_particles(self,fig,ax,positions,radius,colors):
        n_particles = len(colors)
        marker_size = 1.25*10.0*(radius/200e-9)*(5e-6/self.max_size) # Size 10 for 200nm radius and upper limit 5 microns.
        self.particles = positions
        self.trajectories = [ax.plot([], [], markersize=marker_size, marker="o", c=colors[i], mec='white', mew=0.75, alpha=1, animated=True)[0] for i in np.arange(n_particles)]

        ani = animation.FuncAnimation(fig, self.animate, init_func=self.init_anim, frames=(self.frame_max-self.frame_min) // self.frame_interval, interval=25, blit=True)
        plt.show()
# writer = animation.PillowWriter(fps=30)

# ani.save("bessel-ang-mom-test.gif", writer=writer)
# =====================================


    def init_anim(self):
        for trajectory in self.trajectories:
            trajectory.set_data([], [])
        return self.trajectories


    def animate(self,fff):
        frames = self.frame_min + fff * self.frame_interval
        for trajectory, particle in zip(self.trajectories, self.particles):
            trajectory.set_data(
                particle[0, frames - 2 : frames], particle[1, frames - 2 : frames]
            )
        return self.trajectories

    def plot_intensity3d(self, beam_collection):
        #
        # Plots intensity of beam overlayed on figure
        #
        
        upper = self.max_size
        lower = -upper
        X, Y, Z, I, I0 = self.get_intensity_points(beam_collection)

        # Make figure objects
        fig = plt.figure()
        zlower = lower
        zupper = upper
        ax = fig.add_subplot(111, projection='3d', xlim=(lower, upper), ylim=(lower, upper), zlim=(zlower, zupper))
        cs = ax.plot_surface(X, Y, Z, facecolors=cm.viridis(I/I0), edgecolor='none', alpha=0.6)

        ax.set_aspect('equal','box')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        # cbar = fig.colorbar(cs)
        # ax.set_title("z = {:.1e}".format(z[k]))
        return fig,ax


    def make_sphere_surface(self, args, centre):
        radius = args[0]
        samples = 12 #20
        u = np.linspace(0, 2 * np.pi, samples)
        v = np.linspace(0, np.pi, samples)
        x = radius * np.outer(np.cos(u), np.sin(v)) + centre[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + centre[1]
        z = radius * np.outer(np.ones(samples), np.cos(v)) + centre[2]
        return x, y, z
    
    def make_torus_sector_surface(self, args, centre):
        torus_centre_radius = args[0]
        torus_beam_radius = args[1]
        phi_lower = args[2]
        phi_upper = args[3]
        #
        # NOTE; Does not pull these values automatically
        #
        x_shift = torus_centre_radius*np.cos( (phi_lower+phi_upper)/2.0 )
        y_shift = torus_centre_radius*np.sin( (phi_lower+phi_upper)/2.0 )
        samples = 20
        phi_lower = phi_lower%(2.0*np.pi)
        phi_upper = phi_upper%(2.0*np.pi)
        if(phi_lower < phi_upper):
            u = np.linspace(phi_lower, phi_upper, samples)
        else:
            set_a = np.linspace(phi_lower, 2.0*np.pi, int(samples/2.0))
            set_b = np.linspace(0.0, phi_upper, int(samples/2.0))
            u = np.concatenate( (set_a, set_b) )
        v = np.linspace(0, 2.0*np.pi, samples)
        x = np.outer(torus_centre_radius + torus_beam_radius*np.cos(v), np.cos(u)) +centre[0] -x_shift
        y = np.outer(torus_centre_radius + torus_beam_radius*np.cos(v), np.sin(u)) +centre[1] -y_shift
        z = np.outer(torus_beam_radius*np.sin(v), np.ones(samples)) +centre[2]
        return x, y, z
    
    def make_cylinder_surface(self, args, centre):
        # radius, width, angle_Z, angle_pitch = args

        # samples = 20
        # u = np.linspace(-width/2.0, width/2.0, samples)
        # v = np.linspace(0, 2.0*np.pi, samples)

        # ###
        # ### NEEDS TO BE FIXED FOR CYLINDER -> AR ARBITRARY ANGLES
        # ###
        # x = np.outer(radius*np.cos(v), np.cos(u)) +centre[0]
        # y = np.outer(radius*np.cos(v), np.sin(u)) +centre[1]
        # z = np.outer(radius*np.sin(v), np.ones(samples)) +centre[2]
        # return x, y, z
        radius, width, theta_Z, theta_pitch = args
        samples = 10 #20
        u = np.linspace(0, 2.0*np.pi, samples)              # Parameterised values
        v = np.linspace(-width/2.0, width/2.0, samples)     #
        u, v = np.meshgrid(u, v)

        x = v                   # X-coordinates
        y = radius*np.cos(u)    # Y-coordinates
        z = radius*np.sin(u)    # Shift along Z-axis

        Z_rotation_matrix = np.array(
            [
                [np.cos(theta_Z), -np.sin(theta_Z), 0.0],
                [np.sin(theta_Z),  np.cos(theta_Z), 0.0],
                [0.0, 0.0, 1.0]
            ]
        )
        pitch_vec = [-np.sin(theta_Z), np.cos(theta_Z), 0] # Front facing vector (1,0,0) rotated, then perpendicular taken (-y,x,0)
        pitch_rotation_matrix = np.array(
            [
                [( (pitch_vec[0]*pitch_vec[0])*(1.0-np.cos(theta_pitch)) +(np.cos(theta_pitch))              ), ( (pitch_vec[1]*pitch_vec[0])*(1.0-np.cos(theta_pitch)) -(np.sin(theta_pitch)*pitch_vec[2]) ), ( (pitch_vec[2]*pitch_vec[0])*(1.0-np.cos(theta_pitch)) +(np.sin(theta_pitch)*pitch_vec[1]) )],
                [( (pitch_vec[0]*pitch_vec[1])*(1.0-np.cos(theta_pitch)) +(np.sin(theta_pitch)*pitch_vec[2]) ), ( (pitch_vec[1]*pitch_vec[1])*(1.0-np.cos(theta_pitch)) +(np.cos(theta_pitch)             ) ), ( (pitch_vec[2]*pitch_vec[1])*(1.0-np.cos(theta_pitch)) -(np.sin(theta_pitch)*pitch_vec[0]) )],
                [( (pitch_vec[0]*pitch_vec[2])*(1.0-np.cos(theta_pitch)) -(np.sin(theta_pitch)*pitch_vec[1]) ), ( (pitch_vec[1]*pitch_vec[2])*(1.0-np.cos(theta_pitch)) +(np.sin(theta_pitch)*pitch_vec[0]) ), ( (pitch_vec[2]*pitch_vec[2])*(1.0-np.cos(theta_pitch)) +(np.cos(theta_pitch)             ) )]
            ]
        )
    
        xyz_rotated = np.dot( Z_rotation_matrix, np.array([x.flatten(), y.flatten(), z.flatten()]) )    # Apply Z rotation
        xyz_rotated = np.dot( pitch_rotation_matrix, xyz_rotated )                                      # Apply pitch rotation

        x_rotated = xyz_rotated[0].reshape(x.shape) + centre[0]
        y_rotated = xyz_rotated[1].reshape(y.shape) + centre[1]
        z_rotated = xyz_rotated[2].reshape(z.shape) + centre[2]

        return x_rotated, y_rotated, z_rotated
    
    def make_cube_surface(self, args, centre):
        radius = args[0]
        x1 = centre[0] - radius
        x2 = centre[0] + radius
        y1 = centre[1] - radius
        y2 = centre[1] + radius
        z1 = centre[2] - radius
        z2 = centre[2] + radius
        # bottom, top, left, right, back front

        x = [[x1, x2, x2, x1, x1],
            [x1, x2, x2, x1, x1],
            [x1, x2, x2, x1, x1],
            [x1, x2, x2, x1, x1],
            [x1, x1, x1, x1, x1],
            [x2, x2, x2, x2, x2]]
        y = [[y1, y1, y2, y2, y1],
            [y1, y1, y2, y2, y1],  
            [y1, y1, y1, y1, y1],  
            [y2, y2, y2, y2, y2],  
            [y1, y2, y2, y1, y1],  
            [y1, y2, y2, y1, y1]]   
        z = [[z1, z1, z1, z1, z1],
            [z2, z2, z2, z2, z2], 
            [z1, z1, z2, z2, z1], 
            [z1, z1, z2, z2, z1], 
            [z1, z1, z2, z2, z1], 
            [z1, z1, z2, z2, z1]]               
        return np.array(x), np.array(y), np.array(z)
    

    def animate_system3d(self, positions, shapes, args, colours, fig=None, ax=None, connection_indices=[], ignore_coords=[], forces=[], quiver_setting=0, include_tracer=True, include_connections=True, quiver_scale=6e5, beam_collection_list=None):
        #
        # Plots particles with optional quiver (force forces) and tracer (for positions) plots too
        # NOTE; If a quiver plot is wanted, a list of forces must be provided as well (in the format of optforces)
        # 
        # ignore_coords = list of coordinates to ignore force components for in the quiver plot, e.g. 'X', 'Y', 'Z'
        # quiver_scale  = Scale the force arrows to be visible; Default=3e5
        # quiver_setting - 0 = no quiver; 1 = force on each particle; 2 = F-F_total on each particle & average force at centre of mass
        #
        
        # Animation function
        def update(t):
            # Clear old plot elements (particles, quivers, etc)
            for plot in plots:
                plot.remove()
            plots.clear()

            # Add frame counter
            textplot = ax.text2D(0.0, 1.0, "Frame: "+str(t), transform=ax.transAxes)
            plots.append(textplot)

            # Add new particle plot elements
            #colours[62] = "#fc3232"
            for i in range(num_particles):
                #colours[i] = "#fc3232"
                match shapes[i]:
                    case "sphere":
                        x, y, z = self.make_sphere_surface(args[i], positions[t, i])
                    case "torus":
                        x, y, z = self.make_torus_sector_surface(args[i], positions[t, i])
                    case "cylinder":
                        x, y, z = self.make_cylinder_surface(args[i], positions[t, i])
                    case "cube":
                        x, y, z = self.make_cube_surface(args[i], positions[t, i])
                plot = ax.plot_surface(x, y, z, color=colours[i], alpha=0.7)    #1.0
                plots.append(plot)

            # print("positions[t]= ",positions[t])
            # Add new spring connections to the plot
            if(include_connections):
                for connection in connection_indices:
                    # Assuming connections stored in pairs
                    # print("connection    = ",connection)
                    # print("connection[0] = ",connection[0])
                    p1 = positions[t, connection[0]]
                    p2 = positions[t, connection[1]]
                    # print("p1= ",p1)
                    # print("p2= ",p2)
                    lineplot = ax.plot(
                        np.array([p1[0],p2[0]]),
                        np.array([p1[1],p2[1]]), 
                        np.array([p1[2],p2[2]]),
                        linewidth=4,
                        color='blue'
                    )
                    for line in lineplot:
                        plots.append(line)
            
            # Add new quiver plot elements
            if (quiver_setting!=0):
                if( len(forces) == 0 ):
                    print("!! No forces provided, Unable to plot quiver plot !!")
                else:
                    pos_x, pos_y, pos_z = np.transpose(positions[t, :, :])
                    force_x, force_y, force_z = np.transpose(forces[t, :, :]) * quiver_scale
                    for ignore_coord in ignore_coords:
                        match ignore_coord:
                            case "X":
                                force_x = np.zeros(force_x.shape)
                            case "Y":
                                force_y = np.zeros(force_y.shape)
                            case "Z":
                                force_z = np.zeros(force_z.shape)
                    
                    if quiver_setting == 1: # As normal: forces on each particle
                        quiver = ax.quiver(pos_x, pos_y, pos_z, force_x, force_y, force_z)
                        plots.append(quiver)

                    elif quiver_setting == 2:
                        force_x_avg = np.average(force_x)
                        force_y_avg = np.average(force_y)
                        force_z_avg = np.average(force_z)

                        quiver = ax.quiver(pos_x, pos_y, pos_z, force_x-force_x_avg, force_y-force_y_avg, force_z-force_z_avg)
                        plots.append(quiver)
                        # CoM F_tot
                        quiver_tot = ax.quiver(np.average(pos_x), np.average(pos_y), np.average(pos_z), np.average(force_x), np.average(force_y), np.average(force_z), color="r")
                        plots.append(quiver_tot)
            
            # Plot tracers
            if(include_tracer):
                if(t % 5 == 0): # Only place tracers periodically to reduce lag
                    for i in range(num_particles):
                        t1 = t
                        t2 = t+1 if(t+1 < positions.shape[0]) else t
                        pos_x1, pos_y1, pos_z1 = np.transpose(positions[t1, :, :])
                        pos_x2, pos_y2, pos_z2 = np.transpose(positions[t2, :, :])
                        ax.quiver(pos_x1, pos_y1, pos_z1, pos_x2-pos_x1, pos_y2-pos_y1, pos_z2-pos_z1)


            if(beam_collection_list!=None):
                X, Y, Z, I, I0 = self.get_intensity_points(beam_collection_list[t], n=61) # lowered resolution otherwise the animation slows down.
                beam_plane = ax.plot_surface(X, Y, Z, facecolors=cm.viridis(I/I0), edgecolor='none', alpha=0.6)
                plots.append(beam_plane)

        # Initialise
        positions = np.array(positions)
        steps = len(positions)
        num_particles = len(positions[0])

        # If no axes given
        if fig == None or ax == None:
            fig = plt.figure()
            upper = self.max_size
            lower = -upper
            zlower = -2e-6
            zupper = 2e-6
            ax = fig.add_subplot(111, projection='3d', xlim=(lower, upper), ylim=(lower, upper), zlim=(zlower, zupper))

            ax.set_aspect('equal','box')
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

        plots = []
        for i in range(num_particles):
            # Convert hex colours to tuples
            hex = colours[i][1:]
            colour = tuple(int(hex[j:j+2], 16) / 255 for j in (0, 2, 4))
            match shapes[i]:
                case "sphere":
                    x, y, z = self.make_sphere_surface(args[i], positions[0, i])
                case "torus":
                    x, y, z = self.make_torus_sector_surface(args[i], positions[0, i])
                case "cylinder":
                    x, y, z = self.make_cylinder_surface(args[i], positions[0, i])
                case "cube":
                    x, y, z = self.make_cube_surface(args[i], positions[0, i])
                
            plot = ax.plot_surface(x, y, z, color=colour, alpha=0.6)
            plots.append(plot)

        ani = animation.FuncAnimation(fig, update, frames=steps, interval=100)  #100

        plt.show()


    def plot_stresses(self, positions, forces, shapes, all_args, beam_collection, include_quiver=True):
        # For cube particles, and for 1 frame.

        for shape in shapes:
            if shape != "cube": print("WARNING: plot_stresses plots all particles as cubes")

        particle_radius = all_args[0][0] # for cubic particles
        positions = np.array(positions)
        forces = np.array(forces)
        num_particles = len(positions)
        cube_corners = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])*particle_radius
        cube_faces = [
            [cube_corners[i] for i in [0, 1, 3, 2]],  # Back
            [cube_corners[i] for i in [4, 5, 7, 6]],  # Front
            [cube_corners[i] for i in [0, 1, 5, 4]],  # Left
            [cube_corners[i] for i in [2, 3, 7, 6]],  # Right
            [cube_corners[i] for i in [0, 2, 6, 4]],  # Bottom 
            [cube_corners[i] for i in [1, 3, 7, 5]],  # Top
        ]

        # Initialise figure
        fig = plt.figure()
        upper = self.max_size
        lower = -upper
        zlower = -2e-6
        zupper = 2e-6
        ax = fig.add_subplot(111, projection='3d', xlim=(lower, upper), ylim=(lower, upper), zlim=(zlower, zupper))
        ax.set_aspect('equal','box')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # Plot beam
        X, Y, Z, I, I0 = self.get_intensity_points(beam_collection, n=61)
        ax.plot_surface(X, Y, Z, facecolors=cm.viridis(I/I0), edgecolor='none', alpha=0.25)

        # Shift forces to be relative to the net force on the object.
        shifted_forces = forces[0] - np.average(forces[0], axis=0)

        # Scale forces to [0,1] for the cmap.
        maximum = np.max(shifted_forces, axis=0)
        minimum = np.min(shifted_forces, axis=0)
        scaled_forces = np.zeros((num_particles, 3))
        for i in range(3):
            # negative shifted forces mapped to [-1,0], posititives to [0,1]
            print(minimum[i], maximum[i])
            scaled_forces[:,i] = shifted_forces[:,i]/np.array([maximum[i] if x[i]>=0 else -minimum[i] for x in shifted_forces]) # -min as min only used when x[i]<0, then want to keep the vals -ve so div by +ve.
        scaled_forces = scaled_forces/2 + 0.5 # /2 and +1 so scaled force range is [0,1] with 0 force at 0.5 - midpoint of the cmap.
        
        for p_i in range(num_particles): # XXX could chance this loop to select planes of the object - 0.1 alpha to fade out the other parts?
            pos = positions[p_i]
            faces = cube_faces + pos
            
            cols = [
                cm.coolwarm(1 - scaled_forces[p_i][0]), # Back # 1-... as normal in the negative direction.
                cm.coolwarm(scaled_forces[p_i][0]),     # Front
                cm.coolwarm(1 - scaled_forces[p_i][1]), # Left
                cm.coolwarm(scaled_forces[p_i][1]),     # Right
                cm.coolwarm(1 - scaled_forces[p_i][2]), # Bottom
                cm.coolwarm(scaled_forces[p_i][2]),     # Top
            ]

            ax.add_collection3d(Poly3DCollection(faces, facecolors=cols, linewidths=0, alpha=1.0))
            if include_quiver:
                # XXX could add feature to only plot arrows from the object surface
                quiver_scale = 1e-7/(4*particle_radius**2)# /area as stress = Force/Area
                ax.quiver(pos[0], pos[1], pos[2], shifted_forces[p_i,0]*quiver_scale, shifted_forces[p_i,1]*quiver_scale, shifted_forces[p_i,2]*quiver_scale)

        plt.show()


# End of display object.

#
# A series of plots for analysis of forces on particles being brought close together
# These plots use the '*_combined_data.xlsx' files generated in 'SimulationVaryRun.py'
#
def plot_tangential_force_against_number(filename, particle_target, parameter_text=""):
    #
    # Generates a plot of tangential force magnitude of the Nth particle for a system of M particles as a function of the numebr of particles in the system
    # Applies for spherical and torus particles
    #
    data = pd.read_excel(filename+".xlsx")
    data_num = data.count(axis='columns')
    total_force_magnitudes = []
    tangential_force_magnitudes = []
    particle_numbers = []
    for scenario_index in range(len(data) ):
        #Look through each scenario setup, get number of particles involved, try extract data from this scenario
        number_of_particles = int(np.floor(data_num[scenario_index]/(6.0)))
        force_value = 0.0
        if(particle_target < number_of_particles):
            # Total Force Magnitude
            total_force_mag = np.sqrt(
                 pow(data.iloc[scenario_index, 3 +6*particle_target], 2) 
                +pow(data.iloc[scenario_index, 4 +6*particle_target], 2)
            )

            # Tangential Force Magnitude
            position_xy_mag = np.sqrt(
                pow(data.iloc[scenario_index, 0 +6*particle_target],2) +
                pow(data.iloc[scenario_index, 1 +6*particle_target],2)
            )
            position_xy_vector_norm = [
                data.iloc[scenario_index, 0 +6*particle_target] / position_xy_mag,
                data.iloc[scenario_index, 1 +6*particle_target] / position_xy_mag
            ]
            tangential_xy_vector = [
                -position_xy_vector_norm[1],
                 position_xy_vector_norm[0]
            ]
            tangential_force_mag = tangential_xy_vector[0]*data.iloc[scenario_index, 3 +6*particle_target] + tangential_xy_vector[1]*data.iloc[scenario_index, 4 +6*particle_target]
        #Add values to plot
        total_force_magnitudes.append(total_force_mag)
        tangential_force_magnitudes.append(tangential_force_mag)
        particle_numbers.append(number_of_particles)
    #Plot data
    print("particle_numbers = ", particle_numbers)
    print("force_magnitudes = ", total_force_magnitudes)
    print("force_magnitudes = ", tangential_force_magnitudes)

    fig, ax = plt.subplots()

    #Count lines of parameter text to align position (shift down by ~0.05 per line, calibrated for default size.)
    text_ypos = 1 - 0.05*(parameter_text.count("\n")+1)

    ax.plot(particle_numbers, total_force_magnitudes, label="total", color="red")
    ax.plot(particle_numbers, tangential_force_magnitudes, label="tangential", color="blue")
    ax.text(
        0.0, text_ypos,
        parameter_text,
        transform=ax.transAxes,
        fontsize=12
    )
    plt.xlabel("Particle Number")
    plt.ylabel("Force (N)")
    plt.title("Tangential force for varying particle numbers")
    plt.legend()
    plt.show()

def plot_tangential_force_against_number_averaged(filename, parameter_text=""):
    #
    # Works like "plot_tangential_force_against_number()" function, but averages all particls force rather than considering a target particle
    # Generates a plot of tangential force magnitude of the Nth particle for a system of M particles as a function of the numebr of particles in the system
    # Applies for spherical and torus particles
    #
    # NOTE; For 0th frame calculations this has no impact (if all particles are placed symmetrically), but for Brownian motion included beforehand this could be helpful to use
    #
    data = pd.read_excel(filename+".xlsx")
    data_num = data.count(axis='columns')
    total_force_magnitudes = []
    tangential_force_magnitudes = []
    particle_numbers = []
    for scenario_index in range(len(data) ):
        #Look through each scenario setup, get number of particles involved, try extract data from this scenario
        number_of_particles = int(np.floor(data_num[scenario_index]/(6.0)))
        total_force_mag      = 0.0
        tangential_force_mag = 0.0
        for particle_index in range(number_of_particles):
            # Total Force Magnitude
            total_force_mag += np.sqrt(
                 pow(data.iloc[scenario_index, 3 +6*particle_index], 2) 
                +pow(data.iloc[scenario_index, 4 +6*particle_index], 2)
            )

            # Tangential Force Magnitude
            position_xy_mag = np.sqrt(
                pow(data.iloc[scenario_index, 0 +6*particle_index],2) +
                pow(data.iloc[scenario_index, 1 +6*particle_index],2)
            )
            position_xy_vector_norm = [
                data.iloc[scenario_index, 0 +6*particle_index] / position_xy_mag,
                data.iloc[scenario_index, 1 +6*particle_index] / position_xy_mag
            ]
            tangential_xy_vector = [
                -position_xy_vector_norm[1],
                 position_xy_vector_norm[0]
            ]
            tangential_force_mag += tangential_xy_vector[0]*data.iloc[scenario_index, 3 +6*particle_index] + tangential_xy_vector[1]*data.iloc[scenario_index, 4 +6*particle_index]
        #Add values to plot
        total_force_magnitudes.append(total_force_mag/number_of_particles)
        tangential_force_magnitudes.append(tangential_force_mag/number_of_particles)
        particle_numbers.append(number_of_particles)
    #Plot data
    print("particle_numbers = ", particle_numbers)
    print("force_magnitudes = ", total_force_magnitudes)
    print("force_magnitudes = ", tangential_force_magnitudes)

    fig, ax = plt.subplots()
    #Count lines of parameter text to align position (shift down by ~0.05 per line, calibrated for default size.)
    text_ypos = 1 - 0.05*(parameter_text.count("\n")+1)

    ax.plot(particle_numbers, total_force_magnitudes, label="total", color="red")
    ax.plot(particle_numbers, tangential_force_magnitudes, label="tangential", color="blue")
    ax.text(
        0.0, text_ypos,
        parameter_text,
        transform=ax.transAxes,
        fontsize=12
    )

    plt.xlabel("Particle Number")
    plt.ylabel("Averaged Force (N)")
    plt.title("Tangential force (averaged) for varying particle numbers")
    plt.legend()
    plt.show()

def plot_tangential_force_against_arbitrary(filename, particle_target, x_values, x_label, x_units, parameter_text="", parameters_per_particle=2):
    #
    # Generates a plot of tangential force magnitude of the Nth particle for a system of M particles as a function of the number of particles in the system
    # Applies for spherical and torus particles
    #
    # parameters_per_particle = number of parameters (e.g. position, force, force_total, torque, etc) expected to be seen for this file -> NOTE; Assumes ALL parameters a vector quantities
    #
    data = pd.read_excel(filename+".xlsx")
    data_num = data.count(axis='columns')
    total_force_magnitudes = []
    tangential_force_magnitudes = []

    for scenario_index in range(len(data) ):
        #Look through each scenario setup, get number of particles involved, try extract data from this scenario
        elements_per_particle = int(3*parameters_per_particle)
        number_of_particles = int(np.floor(data_num[scenario_index]/(elements_per_particle)))
        if(particle_target < number_of_particles):
            
            posX = data.iloc[scenario_index, 0 +elements_per_particle*particle_target]
            posY = data.iloc[scenario_index, 1 +elements_per_particle*particle_target]
            #posZ = data.iloc[scenario_index, 2 +elements_per_particle*particle_target]

            FX = data.iloc[scenario_index, 3 +elements_per_particle*particle_target]
            FY = data.iloc[scenario_index, 4 +elements_per_particle*particle_target]
            #FZ = data.iloc[scenario_index, 5 +elements_per_particle*particle_target]

            # Total Force Magnitude
            total_force_mag = np.sqrt(
                 pow(FX, 2) 
                +pow(FY, 2)
            )

            # Tangential Force Magnitude
            position_xy_mag = np.sqrt(
                pow(posX,2) +
                pow(posY,2)
            )
            position_xy_vector_norm = [
                posX / position_xy_mag,
                posY / position_xy_mag
            ]
            tangential_xy_vector = [
                -position_xy_vector_norm[1],
                 position_xy_vector_norm[0]
            ]
            tangential_force_mag = tangential_xy_vector[0]*FX + tangential_xy_vector[1]*FY
        #Add values to plot
        total_force_magnitudes.append(total_force_mag)
        tangential_force_magnitudes.append(tangential_force_mag)

    #Plot data
    print(f"{x_label}: {x_values}")
    print("force_magnitudes = ", total_force_magnitudes)
    print("tangential_force_magnitudes = ", tangential_force_magnitudes)

    fig, ax = plt.subplots()

    #Count lines of parameter text to align position (shift down by ~0.05 per line, calibrated for default size.)
    text_ypos = 1 - 0.05*(parameter_text.count("\n")+1)

    ax.plot(x_values, total_force_magnitudes, label="total", color="red")
    ax.plot(x_values, tangential_force_magnitudes, label="tangential", color="blue")
    ax.text(
        0.0, text_ypos,
        parameter_text,
        transform=ax.transAxes,
        fontsize=12
    )
    plt.xlabel(f"{x_label} {x_units}") # Brackets included in units so not left over if unitless i.e. ()
    plt.ylabel("Force (N)")
    plt.title(f"Tangential force against {x_label}")
    plt.legend()
    plt.show()


def plotMulti_tangential_force_against_arbitrary(data_set, data_axes, particle_target, set_label, set_units, parameter_text=""):
    #
    # Plots multiple sets of data together
    # Generates a plot of tangential force magnitude of the Nth particle for a system of M particles as a function of the number of particles in the system
    # Applies for spherical and torus particles
    #
    # data_set  = python list of all data
    # data_axes = axes values tested -> [ [varying parameter], [x_values] ]
    #
    ##
    ## DEPRECATED
    ##

    #Plot data
    fig, ax = plt.subplots()

    for i in range( len(data_axes[0]) ):
        # Go through each extra parameter (different coloured plots)
        total_force_magnitudes      = []
        tangential_force_magnitudes = []
        plotColour = fetchCycleColour(len(data_axes[1]), i)
        data = data_set[i]
        for j in range( len(data_axes[1]) ):
            # Go through each swept X value in plot
            particle_total = int(np.floor(len(data[j])/6.0))
            if(particle_target < particle_total):
                # Total Force Magnitude
                total_force_mag = np.sqrt(
                    pow(data[j][3 +6*particle_target], 2) 
                    +pow(data[j][4 +6*particle_target], 2)
                )

                # Tangential Force Magnitude
                position_xy_mag = np.sqrt(
                    pow(data[j][0 +6*particle_target],2) +
                    pow(data[j][1 +6*particle_target],2)
                )
                position_xy_vector_norm = [
                    data[j][0 +6*particle_target] / position_xy_mag,
                    data[j][1 +6*particle_target] / position_xy_mag
                ]
                tangential_xy_vector = [
                    -position_xy_vector_norm[1],
                        position_xy_vector_norm[0]
                ]
                tangential_force_mag = tangential_xy_vector[0]*data[j][3 +6*particle_target] + tangential_xy_vector[1]*data[j][4 +6*particle_target]
                total_force_magnitudes.append(total_force_mag)
                tangential_force_magnitudes.append(tangential_force_mag)
        ax.plot(data_axes[1], total_force_magnitudes, label="total, "+str(set_label[0])+str(set_units[0])+"="+str(data_axes[0][i]), color=plotColour, linestyle="dashed")
        ax.plot(data_axes[1], tangential_force_magnitudes, label="tangential, "+str(set_label[0])+str(set_units[0])+"="+str(data_axes[0][i]), color=plotColour, linestyle="solid")
    #Count lines of parameter text to align position (shift down by ~0.05 per line, calibrated for default size.)
    text_ypos = 1 - 0.05*(parameter_text.count("\n")+1)
    ax.text(
        0.0, text_ypos,
        parameter_text,
        transform=ax.transAxes,
        fontsize=12
    )
    plt.xlabel(f"{set_label[1]} {set_units[1]}") # Brackets included in units so not left over if unitless i.e. ()
    plt.ylabel("Force (N)")
    plt.title(f"Force against {set_label[1]}")
    plt.legend()
    plt.show()


def fetchCycleColour(set_size, index):
    #
    # Generates a varied colour the ith component in a set
    #
    return (abs(np.sin(1.0*np.pi*(index+1)/set_size)), 0.0, abs(np.cos(1.0*np.pi*(index+1)/set_size)))


def plot_volumes_against_dipoleSize(dipole_sizes, volumes, best_sizes=None, best_volumes=None):
    # plots volume of all particles in the object against the dipole size.
    fig, ax = plt.subplots()
    ax.plot(dipole_sizes, volumes, color="blue")
    try:
        ax.scatter(best_sizes, best_volumes, color="orange")
    except:
        pass
    plt.xlabel("Dipole sizes (m)")
    plt.ylabel("Object volume (m^3)")
    plt.title("Total volume against dipole size")
    plt.show()

def plot_multi_data(data_set, datalabel_set, datacolor_set=np.array([]), graphlabel_set={"title":"", "xAxis":"", "yAxis":""}, linestyle_set=None, show_legend=True):
    #
    # Plots multiple sets of data on a single axis
    # Data and labels are parsed in to fit the scenario required
    #
    # data_set = [ [set_1], [set_2], ... ]
    #       Such that [set_1] = [ [x_data], [y_data] ] to be plotted, NOTE; Assumed to be numpy arrays
    # datalabel_set  = [ "set_1_label", "set_2_label", ... ]
    #       Labels for each data set, shown in the legend
    # graphlabel_set = {"title":..., "xAxis":..., "yAxis":...}
    #       Labels for the graph in general
    #

    # Go through each data set and plot
    fig, ax = plt.subplots()
    for i in range(len(data_set)):
        label = ""  # Default, therefore can leave blank
        if( len(datalabel_set) > i ):
            label = datalabel_set[i]
        if(len(datacolor_set) < len(data_set)):
            if linestyle_set != None:
                ax.plot(data_set[i,0], data_set[i,1], label=label, linestyle=linestyle_set[i])
            else:
                ax.plot(data_set[i,0], data_set[i,1], label=label)
        else:
            if linestyle_set != None:
                ax.plot(data_set[i,0], data_set[i,1], label=label, color=datacolor_set[i], linestyle=linestyle_set[i])
            else:
                ax.plot(data_set[i,0], data_set[i,1], label=label, color=datacolor_set[i])

    plt.title(graphlabel_set["title"])
    plt.xlabel(graphlabel_set["xAxis"])
    plt.ylabel(graphlabel_set["yAxis"])
    if(show_legend and datalabel_set!=[""]):
        ax.legend()
    plt.show()