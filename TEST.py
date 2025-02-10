import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import pandas as pd
import sys
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import Beams

def plot_stresses(positions, forces, shapes, all_args):
        # For cube particles, and for 1 frame.

        for shape in shapes:
            if shape != "cube": sys.exit(f"plot_stresses requires cubes only, not {shape}")
        args = all_args[0]
        positions = np.array(positions)
        num_particles = len(positions)

        fig = plt.figure()
        upper = 2e-6
        lower = -upper
        zlower = -2e-6
        zupper = 2e-6
        ax = fig.add_subplot(111, projection='3d', xlim=(lower, upper), ylim=(lower, upper), zlim=(zlower, zupper))
        ax.set_aspect('equal','box')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        
        # Plot beam
        # X, Y, Z, I, I0 = self.get_intensity_points(beam_collection, n=61) # lowered resolution otherwise the animation slows down.
        # ax.plot_surface(X, Y, Z, facecolors=cm.viridis(I/I0), edgecolor='none', alpha=0.2)

        shifted_forces = np.zeros((num_particles, 3))
        for p_i in range(num_particles):
            shifted_forces[p_i,0] = forces[0, p_i, 0] - np.average(forces[0,:,0])
            shifted_forces[p_i,1] = forces[0, p_i, 1] - np.average(forces[0,:,1])
            shifted_forces[p_i,2] = forces[0, p_i, 2] - np.average(forces[0,:,2])

        maximum = np.max(shifted_forces, axis=0)
        mimimum = np.min(shifted_forces, axis=0)
        print("shifted forces,", shifted_forces)
        scaled_forces = np.zeros((num_particles, 3))
        for i in range(3):
            # scale to -1, 1, then /2 and +1
            scaled_forces[:,i] = shifted_forces[:,i]/np.array([maximum[i] if x[i]>=0 else -mimimum[i] for x in shifted_forces])/2 + 0.5

        print("scaled", scaled_forces)
        corners = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])*args[0]

        for p_i in range(num_particles):
            pos = positions[p_i]
            vertices = corners + pos

            faces = [
                [vertices[i] for i in [0, 1, 3, 2]],  # Back
                [vertices[i] for i in [4, 5, 7, 6]],  # Front
                [vertices[i] for i in [0, 2, 6, 4]],  # Left
                [vertices[i] for i in [1, 3, 7, 5]],  # Right
                [vertices[i] for i in [0, 1, 5, 4]],  # Bottom 
                [vertices[i] for i in [2, 3, 7, 6]],  # Top
            ]
            
            cols = [
                cm.coolwarm(1 - scaled_forces[p_i][1]), # Left
                cm.coolwarm(scaled_forces[p_i][1]),     # Right
                cm.coolwarm(1-scaled_forces[p_i][2]),   # Bottom
                cm.coolwarm(scaled_forces[p_i][2]),     # Top
                cm.coolwarm(1 - scaled_forces[p_i][0]), # Back
                cm.coolwarm(scaled_forces[p_i][0]),     # Front
            ]
            print(p_i, "->", cols)
            ax.add_collection3d(Poly3DCollection(faces, facecolors=cols, linewidths=0, alpha=1.0))
            quiver_scale = 9e5
            ax.quiver(pos[0], pos[1], pos[2], shifted_forces[p_i,0]*quiver_scale, shifted_forces[p_i,1]*quiver_scale, shifted_forces[p_i,2]*quiver_scale)

        plt.show()


plot_stresses([[-1e-6,0,0], [1e-6, 0, 0], [1e-6, 0, 2e-6]], np.array([[[1e-12,0e-12,0e-12], [-0e-12,1e-12,-0e-12], [-0e-12, 0e-12,1e-12]]]), ["cube"], [[0.2e-6]])