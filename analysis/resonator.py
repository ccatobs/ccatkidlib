# Module for the resonator class

class resonator:
    def __init__(f0, f, s21):
        
        """
        f0 (float): resonant freq from RFSoC peak finder
        f (np array): freqs of targ sweep
        s21 (np array): complex transmission data at each f (above)

        """
        self.s21 = s21
        self.f = f
        self.f0 = f0
        
        # Find index of frequency closest to the resonant frequency (from the RFSoC peak-finder)
        self.f0_index = np.where(np.abs(self.f - self.f0) == np.min(np.abs(self.f - self.f0)))


        
        # measured cable delay from Cornell SD testing environment
        # WILL NEED TO CHANGE IF USING A DIFFERENT TESTING ENVIRONMENT
        self.TAU = -121.6158362029386
        
        if self.TAU is not None:
            self.s21 = self.s21 * np.exp(-1j* (self.f-self.f0)*2*np.pi*self.TAU)
    
    def fit_resonator(self, nonlinear = False, asymm = False):
        # "theory" fit of the s21 curve from the target sweep
        """
        nonlinear (boolean): to fit to a non-linear resonator model or not
        asymm (boolean): to fit to an asymmetric resonator model or not
        """
        self.res_fit = rm.full_fit(self.f, self.s21.real, self.s21.imag, nonlinear = nonlinear, asymm = asymm)
                    
            
        self.res_model = rm.fine_s21_model(self.f, self.res_fit.params)
        self.params = self.res_fit.params
        self.Q = self.res_fit.params["Q"].value
    
    def remove_cable(self):
        cable_mask_inner = self.f0/self.Q * 2
        self.cable_mask = np.where(np.abs(self.f - self.f0) > cable_mask_inner)[0]
        self.background_mean = np.mean(np.abs(self.s21[self.cable_mask]))
        self.cable_fit = rm.cable_fit(self.f[self.cable_mask].real, self.s21[self.cable_mask].real,self.s21[self.cable_mask].imag)
        self.cable = rm.fine_s21_model(self.f, self.cable_fit.params, cable=True)
        self.s21 = np.divide(self.s21, self.cable)

