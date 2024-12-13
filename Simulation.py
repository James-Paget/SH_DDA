

class SimulationObject (object):
    defaults = {"wavelength":'1e-6',
                "dipole_radius":'60e-9',
                "time_step":'1e-4',
                "frames":'1000',
                "vmd_output":'True',
                "excel_output":'True',
                "include_force":'True',
                "include_couple":'True',
                }

    def __init__(self,paraminfo,optioninfo,outputinfo):
        if paraminfo==None:
            # Set defaults
            self.wavelength = float(SimulationObject.defaults['wavelength'])
            self.dipole_radius = float(SimulationObject.defaults['dipole_radius'])
            self.timestep = float(SimulationObject.defaults['timestep']) # range will be 2 times this
        else:
            # Read from file
            self.wavelength = float(paraminfo.get('wavelength',SimulationObject.defaults['wavelength']))
            self.dipole_radius = float(paraminfo.get('dipole_radius',SimulationObject.defaults['dipole_radius']))
            self.time_step = float(paraminfo.get('time_step',SimulationObject.defaults['time_step']))
        if optioninfo==None:
            # Set defaults
            self.frames = int(SimulationObject.defaults['frames'])
        else:
            # Read from file
            self.frames = int(optioninfo.get('frames',SimulationObject.defaults['frames']))
        if outputinfo==None:
            # Set defaults
            self.vmd_output = bool(SimulationObject.defaults['vmd_output'])
            self.excel_output = bool(SimulationObject.defaults['excel_output'])
            self.include_force = bool(SimulationObject.defaults['include_force'])
            self.include_couple = bool(SimulationObject.defaults['include_couple'])
        else:
            # Read from file
            self.vmd_output = bool(outputinfo.get('vmd_output',SimulationObject.defaults['vmd_output']))
            self.excel_output = bool(outputinfo.get('excel_output',SimulationObject.defaults['excel_output']))
            self.include_force = bool(outputinfo.get('include_force',SimulationObject.defaults['include_force']))
            self.include_couple = bool(outputinfo.get('include_couple',SimulationObject.defaults['include_couple']))

