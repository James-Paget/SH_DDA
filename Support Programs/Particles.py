"""
Particle creation routines
"""
import numpy as np

class ParticleCollection (object):
    num_particles = 0
    particle_spec = {'Silicon': [3.9, '#e0a105', '#ffe10a'], 'Sapphire': [2.5, '#b01240', '#e81e80'],'Diamond': [2.417, '#b01240', '#e81e80'], 'SiN': [2.046, '#d8ff60', '#e8ff80'], 'Glass': [1.5, '#1870d0', '#2298f5'],'Low': [1.446, '#7e7e7e', '#aeaeae'],'FusedSilica': [1.458-0.0j, '#7e7e7e', '#aeaeae'],'FusedSilica0001': [1.458-0.0001j, '#7e7e7e', '#aeaeae'],'FusedSilica001': [1.458-0.001j, '#7e7e7e', '#aeaeae'],'FusedSilica001p': [1.458+0.001j, '#7e7e7e', '#aeaeae'],'FusedSilica002': [1.458-0.002j, '#7e7e7e', '#aeaeae'],'FusedSilica005': [1.458-0.005j, '#7e7e7e', '#aeaeae']}
    default_material = 'FusedSilica'
    default_radius = 200e-9
    
#particle_types = ['FusedSilica'] * n_particles
#particle_types[0] = 'Sapphire'
#particle_types[1] = 'SiN'
#particle_types[3] = 'Diamond'
#ref_ind = np.ones(n_particles,dtype=complex)
#colors = np.ndarray(n_particles,dtype=object)

#for i in range(n_particles):
#    if particle_types[i] == 'Silicon':
#        ref_ind[i] = particle_spec['Silicon'][0]
#        colors[i] = particle_spec['Silicon'][1]
#        if i/2 == i//2:
#            colors[i] = '#e0a105'
#        else:
#            colors[i] = '#ffe10a'

 

    
    def __init__(self,particleinfo):
        if particleinfo==None:
            # Set defaults
            ParticleCollection.num_particles = 1
            self.particle_type = np.asarray([ParticleCollection.default_material])
            self.particle_radius = np.asarray([ParticleCollection.default_radius])
            self.particle_indices = np.asarray([1.458],dtype=complex)
            self.particle_colour = np.asarray(['#8e8e8e'])
            self.particle_density = np.asarray([2200])
            self.particle_positions = np.zeros((1,3),dtype=float)
        else:
            # Read from file
            self.default_material = particleinfo.get('default_material','FusedSilica')
            self.default_radius = float(particleinfo.get('default_radius',200e-9))
            self.particle_list = particleinfo.get('particle_list',None)
            if self.particle_list==None or self.particle_list==False:
                # Set defaults
                ParticleCollection.num_particles = 1
                self.particle_type = np.asarray(['FusedSilica'])
                self.particle_radius = np.asarray([200e-9])
                self.particle_indices = np.asarray([1.458],dtype=complex)
                self.particle_colour = np.asarray(['#8e8e8e'])
                self.particle_density = np.asarray([2200])
                self.particle_positions = np.zeros((1,3),dtype=float)
            else:
                # Read individual particles
                i=0
                self.particle_type = []
                self.particle_radius = []
                self.particle_colour = []
                self.particle_positions = []
                self.particle_indices = []
                self.particle_density = []
                for newparticle in self.particle_list:
                    particle = self.particle_list[newparticle]
                    print("Loading particle",particle)
                    if particle != None:
                        self.particle_type.append(particle.get('material',self.default_material))
                        self.particle_radius.append(float(particle.get('radius',self.default_radius)))
                        self.altcolour = bool(particle.get('altcolour',False))
                        if self.altcolour==False:
                            self.particle_colour.append(ParticleCollection.particle_spec[self.particle_type[i]][1])
                        else:
                            self.particle_colour.append(ParticleCollection.particle_spec[self.particle_type[i]][2])
                        self.particle_indices.append(ParticleCollection.particle_spec[self.particle_type[i]][0])
                        self.coords = particle.get('coords',"0.0 0.0 0.0")
                        self.fields = self.coords.split(" ")
                        if self.fields[0]=="None":
                            self.particle_positions.append(np.array((0.0,0.0,0.0),dtype=np.float64))
                        else:
                            self.particle_positions.append(np.array((0.0,0.0,0.0),dtype=np.float64))
                            for j in range(min(len(self.fields),3)):
                                self.particle_positions[i][j] = float(self.fields[j])
                        self.particle_density.append(2200)
                    else:
                        ParticleCollection.num_particles = 1
                        self.particle_type.append('FusedSilica')
                        self.particle_radius.append(200e-9)
                        self.particle_indices.append(1.458)
                        self.particle_colour.append('#8e8e8e')
                        self.particle_density.append(2200)
                        self.particle_positions.append(np.array((0.0,0.0,0.0),dtype=np.float64))
                    i+=1
                ParticleCollection.num_particles = i

    def get_refractive_indices(self):
        return np.asarray(self.particle_indices,dtype=complex)
        
    def get_particle_types(self):
        return np.asarray(self.particle_type)
        
    def get_particle_colours(self):
        return np.asarray(self.particle_colour)
        
    def get_particle_radii(self):
        return np.asarray(self.particle_radius,dtype=float)

    def get_particle_density(self):
        return np.asarray(self.particle_density,dtype=float)

    def get_particle_positions(self):
        return np.asarray(self.particle_positions,dtype=float).reshape((ParticleCollection.num_particles,3))
    
