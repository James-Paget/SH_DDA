"""
Particle creation routines
"""
import numpy as np

class ParticleCollection (object):
    num_particles = 0
    particle_spec = {'Silicon': [3.9, '#e0a105', '#ffe10a','S'],
                     'Sapphire': [2.5, '#b01240', '#e81e80','O'],
                     'Diamond': [2.417, '#b01240', '#e81e80','C'],
                     'SiN': [2.046, '#d8ff60', '#e8ff80','P',],
                     'Calcite': [1.55, '#e0a105', '#ffe10a','S'],
                     'Glass': [1.5, '#1870d0', '#2298f5','N'],
                     'LeadGlass': [1.6, '#1870d0', '#2298f5','N'],
                     'Low': [1.446, '#7e7e7e', '#aeaeae','H',],
                     'FusedSilica': [1.458+0.0j, '#7e7e7e', '#aeaeae','B',],
                     'FusedSilica0001': [1.458+0.0001j, '#7e7e7e', '#aeaeae','B'],
                     'FusedSilica0002': [1.458+0.0002j, '#7e7e7e', '#aeaeae','B'],
                     'FusedSilica0005': [1.458+0.0005j, '#7e7e7e', '#aeaeae','B'],
                     'FusedSilica001': [1.458+0.001j, '#7e7e7e', '#aeaeae','B'],
                     'FusedSilica002': [1.458+0.002j, '#7e7e7e', '#aeaeae','B'],
                     'FusedSilica005': [1.458+0.005j, '#7e7e7e', '#aeaeae','B'],
                     'FusedSilica01': [1.458+0.01j, '#7e7e7e', '#aeaeae','B'],
                     'RBC': [1.41, '#ffffff', '#eeeeee', 'Cr'],
                     'Chromium': [3.41 + 3.57j, '#ffffff', '#eeeeee', 'Cr'],
                     'Air': [1.00, '#0f0f0f', '#1f1f1f', 'Ar'],
                     }
                     
    defaults = {"default_material":'FusedSilica',
                #"default_radius":'200e-9',
                "default_shape":'sphere',
                "default_args":'200e-9',
                "default_density":'2200', # kg/m^3 typical glass value
                "default_connection_mode":'manual',
                "default_connection_args":''
                }
    default_material = defaults['default_material']
    #default_radius   = float(defaults['default_radius'])
    default_shape    = defaults['default_shape']
    default_args     = defaults['default_args']
    default_density  = float(defaults['default_density'])
    default_connection_mode    = defaults['default_connection_mode']
    default_connection_args    = defaults['default_connection_args']

    
    def __init__(self,particleinfo):
        if particleinfo==None:
            # Set defaults
            ParticleCollection.num_particles = 1
            self.particle_type = np.asarray([ParticleCollection.default_material])
            #self.particle_radius = np.asarray([ParticleCollection.default_radius])
            self.particle_shape.append(self.default_shape)
            self.particle_args.append( self.default_args.split(" ") )
            self.particle_indices = np.asarray([ParticleCollection.particle_spec[ParticleCollection.default_material][0]],dtype=complex)
            self.particle_colour = np.asarray([ParticleCollection.particle_spec[ParticleCollection.default_material][1]])
            self.particle_vtfcolour = np.asarray([ParticleCollection.particle_spec[ParticleCollection.default_material][3]])
            self.particle_density = np.asarray([ParticleCollection.default_density])
            self.particle_positions = np.zeros((1,3),dtype=float)
        else:
            # Read from file
            self.default_material = particleinfo.get('default_material',ParticleCollection.default_material)
            #self.default_radius = float(particleinfo.get('default_radius',ParticleCollection.default_radius))

            # Get connection information
            self.connection_mode = particleinfo.get('connection_mode',ParticleCollection.default_connection_mode)
            if self.connection_mode == None: 
                self.connection_mode = ParticleCollection.default_connection_mode
            self.connection_args = particleinfo.get('connection_args',ParticleCollection.default_connection_args)
            if self.connection_args == None: 
                self.connection_args = ParticleCollection.default_connection_args

            # Get particle list
            self.particle_list = particleinfo.get('particle_list',None)
            if self.particle_list==None or self.particle_list==False:
                # Set defaults
                ParticleCollection.num_particles = 1
                self.particle_type = np.asarray([ParticleCollection.default_material])
                #self.particle_radius = np.asarray([ParticleCollection.default_radius])
                self.particle_shape.append(self.default_shape)
                self.particle_args.append( self.default_args.split(" ") )
                self.particle_indices = np.asarray([ParticleCollection.particle_spec[ParticleCollection.default_material][0]],dtype=complex)
                self.particle_colour = np.asarray([ParticleCollection.particle_spec[ParticleCollection.default_material][1]])
                self.particle_vtfcolour = np.asarray([ParticleCollection.particle_spec[ParticleCollection.default_material][3]])
                self.particle_density = np.asarray([ParticleCollection.default_density])
                self.particle_positions = np.zeros((1,3),dtype=float)
            else:
                # Read individual particles
                i=0
                self.particle_type = []
                #self.particle_radius = []
                self.particle_shape = []    # Name of shape e.g "sphere", "torus", "cylinder", "cube"
                self.particle_args  = []    # Parameters for specific shape e.g. [radius], [r1,r2,phi1,phi2] ,..., [radius]
                self.particle_colour = []
                self.particle_vtfcolour = []
                self.particle_positions = []
                self.particle_indices = []
                self.particle_density = []
                for newparticle in self.particle_list:
                    particle = self.particle_list[newparticle]
                    # print("Loading particle",particle)
                    if particle != None:
                        self.particle_type.append(particle.get('material',self.default_material))
                        #self.particle_radius.append(float(particle.get('radius',self.default_radius)))

                        self.argsraw = str(particle.get('args',self.default_args))
                        self.args    = self.argsraw.split(" ")
                        match (particle.get('shape',self.default_shape), len(self.args)):
                            case ("sphere", 1):
                                self.particle_shape.append("sphere")
                                self.particle_args.append(self.args)
                            case ("torus", 4):
                                self.particle_shape.append("torus")
                                self.particle_args.append(self.args)
                            case ("cylinder", 4):
                                self.particle_shape.append("cylinder")
                                self.particle_args.append(self.args)
                            case ("cube", 1):
                                self.particle_shape.append("cube")
                                self.particle_args.append(self.args)
                            case _:
                                self.particle_shape.append(self.default_shape)
                                self.particle_args.append(self.default_args)

                        self.altcolour = bool(particle.get('altcolour',False))
                        if self.altcolour==False:
                            self.particle_colour.append(ParticleCollection.particle_spec[self.particle_type[i]][1])
                        else:
                            self.particle_colour.append(ParticleCollection.particle_spec[self.particle_type[i]][2])
                        self.particle_vtfcolour.append(ParticleCollection.particle_spec[self.particle_type[i]][3])
                        self.particle_indices.append(ParticleCollection.particle_spec[self.particle_type[i]][0])
                        self.coords = particle.get('coords',"0.0 0.0 0.0")
                        self.fields = self.coords.split(" ")
                        if self.fields[0]=="None":
                            self.particle_positions.append(np.array((0.0,0.0,0.0),dtype=np.float64))
                        else:
                            self.particle_positions.append(np.array((0.0,0.0,0.0),dtype=np.float64))
                            for j in range(min(len(self.fields),3)):
                                self.particle_positions[i][j] = float(self.fields[j])
                        self.particle_density.append(ParticleCollection.default_density)
                    else:
                        ParticleCollection.num_particles = 1
                        self.particle_type.append(ParticleCollection.default_material)
                        #self.particle_radius.append(ParticleCollection.default_radius)
                        self.particle_shape.append(self.default_shape)
                        self.particle_args.append( self.default_args.split(" ") )
                        self.particle_indices.append(ParticleCollection.particle_spec[ParticleCollection.default_material][0])
                        self.particle_colour.append(ParticleCollection.particle_spec[ParticleCollection.default_material][1])
                        self.particle_vtfcolour.append(ParticleCollection.particle_spec[ParticleCollection.default_material][3])
                        self.particle_density.append(ParticleCollection.default_density)
                        self.particle_positions.append(np.array((0.0,0.0,0.0),dtype=np.float64))
                    i+=1
                ParticleCollection.num_particles = i

    def get_refractive_indices(self):
        return np.asarray(self.particle_indices,dtype=complex)
        
    def get_particle_types(self):
        return np.asarray(self.particle_type)
        
    def get_particle_colours(self):
        return np.asarray(self.particle_colour)
        
    def get_particle_vtfcolours(self):
        return np.asarray(self.particle_vtfcolour)
        
    def get_particle_radii(self):
        #
        # OUTDATED -> USING SHAPE,ARGS SYSTEM NOW
        #
        return np.asarray(self.particle_radius,dtype=float)

    def get_particle_shape(self):
        return np.asarray(self.particle_shape)

    def get_particle_args(self):
        #
        # Try to vectorise this neatly
        #   Main problem is variable length args
        #
        set = np.zeros(ParticleCollection.num_particles, dtype=object)
        for i in range(ParticleCollection.num_particles):
            set[i] = np.asarray(self.particle_args[i],dtype=float)
        return set
    
    def get_particle_density(self):
        #
        # **NOTE; May want to have variable densities
        #
        return np.asarray(self.particle_density,dtype=float)

    def get_particle_positions(self):
        return np.asarray(self.particle_positions,dtype=float).reshape((ParticleCollection.num_particles,3))
    
    def get_particle_masses(self):
        #
        # Gets the masses of each shape
        #
        masses = np.ones(ParticleCollection.num_particles, dtype=float);
        for i in range(ParticleCollection.num_particles):
            match self.particle_shape[i]:
                case "sphere":
                    masses[i] = (4/3)*float(self.particle_density[i])*np.pi*float(self.particle_args[i][0])**3
                case "torus":
                    masses[i] = float(self.particle_density[i])*2.0*(np.pi**2)*float(self.particle_args[i][1])**2 *float(self.particle_args[i][0])
                case "cylinder":
                    masses[i] = float(self.particle_density[i])*( np.pi*(float(self.particle_args[i][0])**2)*float(self.particle_args[i][1]) )
                case "cube":
                    masses[i] = float(self.particle_density[i])*8.0*float(self.particle_args[i][0])**3
                case _:
                    print("Invalid shape: During mass calc, ",i)
        return masses
    
    def get_connection_mode(self):
        return self.connection_mode
    
    def get_connection_args(self):
        if self.connection_args == "":
            return np.asarray([], dtype=float)
        else:
            args = []
            # Cast string to appropriate type.
            for arg in str(self.connection_args).split(" "):
                if arg == "True":
                    arg = True
                elif arg == "False":
                    arg = False
                else:
                    try:
                        arg = float(arg)
                    except:
                        pass # arg left as a string
                args.append(arg)
            return args