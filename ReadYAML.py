"""
Service functions for reading the parameters in
the yaml file for a given simulation.
"""
import os
import yaml
import sys
import Simulation
import Display
import Beams
import Particles

def load_yaml(filename_yaml):
    yamlpath = './'+filename_yaml
    check_file = os.path.isfile(yamlpath)
    if not check_file:
        print("Unable to find configuration file: ",yamlpath)
        sys.exit()
    with open(yamlpath, 'r') as yamlfile:
        sys_params = yaml.safe_load(yamlfile)
    return sys_params
    
    
def read_section(sys_params, section):
    sectioninfo = sys_params.get(section,None)
    return sectioninfo


class Options(object):
    num_simulations = 0

    def __init__(self, filestem):
        filename_yaml = filestem+".yml"
        self.name = filename_yaml
        Options.num_simulations += 1
        sys_params = load_yaml(filename_yaml)
        self.beaminfo = read_section(sys_params,'beams')
        self.paraminfo = read_section(sys_params,'parameters')
        self.optioninfo = read_section(sys_params,'options')
        self.displayinfo = read_section(sys_params,'display')
        self.outputinfo = read_section(sys_params,'output')
        self.particleinfo = read_section(sys_params,'particles')
        
        self.simulation = Simulation.SimulationObject(self.paraminfo,self.optioninfo,self.outputinfo)
        self.display = Display.DisplayObject(self.displayinfo,self.simulation.frames)
        self.beam_collection = Beams.create_beam_collection(self.beaminfo,self.simulation.wavelength)
        self.particle_collection = Particles.ParticleCollection(self.particleinfo)


    def __del__(self):
        Options.num_simulations -= 1
    
    
