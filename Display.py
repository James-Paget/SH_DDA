"""
Make a display object for animations etc
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

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

        ax.set_aspect('equal','box')
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
    

    def animate_particles3d(self, fig, ax, positions, shapes, args, colours):
        #pow( centre_R-sqrt( pow(point.x,2) + pow(point.y,2) ) ,2) +pow(point.z,2) <= pow(tube_R,2)

        # Animation function
        def update(t):
            # Clear old particles
            for plot in plots:
                plot.remove()
            plots.clear()

            for i in range(num_particles):
                match shapes[i]:
                    case "sphere":
                        x, y, z = self.make_sphere_surface(args[i], positions[t, i])
                    case "torus":
                        x, y, z = self.make_torus_sector_surface(args[i], positions[t, i])
                plot = ax.plot_surface(x, y, z, color=colours[i], alpha=1.0)
                plots.append(plot)

        # Initialise
        positions = np.array(positions)
        steps = len(positions)
        num_particles = len(positions[0])

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
            #x, y, z = self.make_particle_surface(radii[i], positions[0, i])
            plot = ax.plot_surface(x, y, z, color=colour, alpha=0.6)
            plots.append(plot)

        ani = animation.FuncAnimation(fig, update, frames=steps, interval=100)

        plt.show()



