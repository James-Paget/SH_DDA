"""
Make a display object for animations etc
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import pandas as pd

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

    def plot_intensity(self, beam_collection):
        nx = self.resolution
        ny=nx
        Ex = np.zeros((nx, ny), dtype=complex)
        Ey = np.zeros((nx, ny), dtype=complex)
        Ez = np.zeros((nx, ny), dtype=complex)
        z = self.z_offset
        #I = []
        E = np.zeros(3,dtype=np.complex128)
        #fig, ax = plt.subplots(1, num_plots)
        upper = self.max_size
        lower = -upper
        x = np.linspace(lower, upper, nx)
        y = np.linspace(lower, upper, ny)
        X, Y = np.meshgrid(x, y)
        for j in range(ny):
            for i in range(nx):
                Beams.all_incident_fields((x[i], y[j], self.z_offset), beam_collection, E)
                Ex[j][i] = E[0]
                Ey[j][i] = E[1]
                Ez[j][i] = E[2]

        I = np.square(np.abs(Ex)) + np.square(np.abs(Ey)) + np.square(np.abs(Ez))
        print(np.shape(I))
#        for j in range(ny):
#            print(j,I[0][100][j])
        I0 = np.max(I)
#            ax.axis('equal')

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
        
        # Prepare beam surface values
        nx = self.resolution
        ny=nx
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

        I = np.square(np.abs(Ex)) + np.square(np.abs(Ey)) + np.square(np.abs(Ez))
        I0 = np.max(I)

        # Make figure objects
        fig = plt.figure()
        zlower = -2e-6
        zupper = 2e-6
        ax = fig.add_subplot(111, projection='3d', xlim=(lower, upper), ylim=(lower, upper), zlim=(zlower, zupper))

        Z = np.zeros(X.shape) + self.z_offset
        cs = ax.plot_surface(X, Y, Z, facecolors=cm.viridis(I/I0), edgecolor='none', alpha=0.6)

        ax.set_aspect('equal','box')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        # cbar = fig.colorbar(cs)
        # ax.set_title("z = {:.1e}".format(z[k]))
        return fig,ax


    def make_sphere_surface(self, args, center):
        radius = args[0]
        samples = 20
        u = np.linspace(0, 2 * np.pi, samples)
        v = np.linspace(0, np.pi, samples)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(samples), np.cos(v)) + center[2]
        return x, y, z
    
    def make_torus_sector_surface(self, args, center):
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
        x = np.outer(torus_centre_radius + torus_beam_radius*np.cos(v), np.cos(u)) +center[0] -x_shift
        y = np.outer(torus_centre_radius + torus_beam_radius*np.cos(v), np.sin(u)) +center[1] -y_shift
        z = np.outer(torus_beam_radius*np.sin(v), np.ones(samples)) +center[2]
        return x, y, z
    

    def animate_system3d(self, positions, shapes, args, colours, fig=None, ax=None, ignore_coords=[], forces=[], include_quiver=False, include_tracer=True, quiver_scale=3e5):
        #
        # Plots particles with optional quiver (force forces) and tracer (for positions) plots too
        # NOTE; If a quiver plot is wanted, a list of forces must be provided as well (in the format of optforces)
        # 
        # ignore_coords = list of coordinates to ignore force components for in the quiver plot, e.g. 'X', 'Y', 'Z'
        # quiver_scale  = Scale the force arrows to be visible
        #

        # Animation function
        def update(t):
            # Clear old plot elements (particles, quivers, etc)
            for plot in plots:
                plot.remove()
            plots.clear()

            # Add new particle plot elements
            for i in range(num_particles):
                match shapes[i]:
                    case "sphere":
                        x, y, z = self.make_sphere_surface(args[i], positions[t, i])
                    case "torus":
                        x, y, z = self.make_torus_sector_surface(args[i], positions[t, i])
                plot = ax.plot_surface(x, y, z, color=colours[i], alpha=1.0)
                plots.append(plot)
            
            # Add new quiver plot elements
            if(include_quiver):
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
                    quiver = ax.quiver(pos_x, pos_y, pos_z, force_x, force_y, force_z)
                    plots.append(quiver)
            
            # Plot tracers
            if(include_tracer):
                if(t % 5 == 0): # Only place tracers periodically to reduce lag
                    for i in range(num_particles):
                        t1 = t
                        t2 = t+1 if(t+1 < positions.shape[0]) else t
                        pos_x1, pos_y1, pos_z1 = np.transpose(positions[t1, :, :])
                        pos_x2, pos_y2, pos_z2 = np.transpose(positions[t2, :, :])
                        ax.quiver(pos_x1, pos_y1, pos_z1, pos_x2-pos_x1, pos_y2-pos_y1, pos_z2-pos_z1)

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
            plot = ax.plot_surface(x, y, z, color=colour, alpha=0.6)
            plots.append(plot)

        ani = animation.FuncAnimation(fig, update, frames=steps, interval=100)

        plt.show()


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

def plot_tangential_force_against_arbitrary(filename, particle_target, x_values, x_label, x_units, parameter_text=""):
    #
    # Generates a plot of tangential force magnitude of the Nth particle for a system of M particles as a function of the number of particles in the system
    # Applies for spherical and torus particles
    #
    data = pd.read_excel(filename+".xlsx")
    data_num = data.count(axis='columns')
    total_force_magnitudes = []
    tangential_force_magnitudes = []

    for scenario_index in range(len(data) ):
        #Look through each scenario setup, get number of particles involved, try extract data from this scenario
        number_of_particles = int(np.floor(data_num[scenario_index]/(6.0)))
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


def plot_volumes_against_dipoleSize(dipole_sizes, volumes, best_sizes=None, best_volumes=None):
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