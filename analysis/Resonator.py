class Resonator:
    def __init__(self):
        '''
        Class representing a single microwave kinetic inductance detector (MKID) driven with a set tone power.
        '''

        # Resonator Quality
        self.good_quality = True # Flag for discarding poor quality data

        # Unique identifiers
        self.array_pos = None # Numerical position in MKID array (sorted by smallest to largest resonant frequency)
        
        # Associated target sweep(s)
        self.sweeps = None

        # Associated timestream(s)
        self.tstreams = None

        # Define resonator frequencies
        self.nominal_res_freq = None
        self.res_freq = None # Resonant frequency
        self.min_freq = None # Frequency where |S21| is minimized
        self.sens_freq = None # Frequency where resonator is most sensitive frequency shifts

        # Define resonator quality factors
        self.Qr = None # Resonator quality factor
        self.Qe = None # Complex external quality factor

        # Bifurcation parameter
        self.a = None
        self.bifurcated = False

        # External parameters
        self.tone_power = None # Driving tone power
        self.bath_temp = None # Bath temperature
        self.optical_loading = None # Nominal optical loading 

    ################
    # 
    ################





    ###################
    # Getters/Setters #
    ###################

        
