class Sweep:
    def __init__(self, fs, S21z):
        '''
        Class representing a single target sweep taken with a radio frequency system on a chip (RFSoC). 
        Includes analysis functions for target sweeps of single microwave kinetic inductance detectors (MKIDs).
        '''

        # Sweep data
        self.fs = fs # Frequencies of sweep
        self.S21z = S21z # Complex transmission (S21) data of sweep
        self.S21z_norm = None # Normalized S21 data
    
    def remove_cable_delay(self):
        pass

    def remove_gain_profile(self):
        pass

    def normalize_sweep(self):
        pass

