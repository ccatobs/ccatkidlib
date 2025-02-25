import numpy as np



class Resonator:
    '''
    Class representing a single kinetic inductance detector.
    '''

    def __init__(self, network, res_num):
        self.res_num = -1 # Resonator Number

        # Resonant Frequency
        self.f0 = 0

        # Resonator Quality Factors
        self.Qi = 0
        self.Qc = 0

        # Nonlinearity Parmeter
        self.a = 100

        # Resonator Sweeps and Timestreams
        self.sweeps  = {}
        self.streams = {}
    
    def dashboard():
        pass




    #################
    # Magic Methods #
    #################

    def __str__():
        pass

    def __repr__():
        pass