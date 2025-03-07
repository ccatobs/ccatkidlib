
from ccatkidlib.analysis.res_fit.phase_utils.utils import *
from scipy.optimize import curve_fit
import scipy



class ResonanceFitterSingleTone():
    ''' Class to extract resonance parameters following algorithm in Gao Thesis Appendix E
        1) remove cable delay
        2) find circle x0,y0 and radius
        3) rotate and translate to origin
        4) fit phase versus frequency to extract fr, Qr
        Class to load resonator data and perform a simple or 
        nonlinear phase fit (user-specified) on a single tone
    '''
    def __init__(self,f,z,tau, numspan=2, tone_freq_lo=0,window_width=0,\
                 pherr_threshold_num=10,pherr_threshold=0.2,\
                 verbose=False,**keywords):
        # f in Hz (will also work with GHz?), z is complex, tau in ns
        ''' fit the resonance following method described in Jiansong Gao thesis Appendix E
            data needs to be complex (like I + jQ)
        '''

        self.f = f
        self.z = z # z_real+1j*z_imag
        self.tau = tau
        self.tone_freq_lo = tone_freq_lo
        self.numspan = numspan
        self.ang = phase2(self.z)
        self.pherr_threshold_num = int(pherr_threshold_num)
        self.pherr_threshold = pherr_threshold

        # new argument for specifying output to print to user
        self.verbose = verbose

        self.window_width = window_width

        if ('weight_type' in keywords):
            self.use_weight = True
            self.weight_type = keywords['weight_type']
            if self.window_width == 0:
                print('To use weighting, need to specify a window width')
                self.use_weight = False
        else:
            self.use_weight = False

        if ('result0' in keywords):
            self.result0 = keywords['result0'] # dictionary of fit results
            self.result = self.fits21simnlin()
        else:
            self.result = self.fits21sim() # result0 is a linear fit with fits21sim

        self.z1 = removecable(self.f,self.z,self.tau,verbose=self.verbose)
    
        if ('result0' in keywords):
            self.f0g, self.Qg, self.idf0, self.iddf = estpara3(self.f,self.z1,verbose=self.verbose,result0=self.result0)
        else:
            self.f0g, self.Qg, self.idf0, self.iddf = estpara3(self.f,self.z1,verbose=self.verbose)
    
        # fit circle using data points f0g +- 2*f0g/Qg
        self.id1 = np.amax([self.idf0-2*self.iddf,0])
        self.id2 = np.amin([self.idf0+2*self.iddf,len(self.f)])
    
        if self.id2 < len(self.f):
            self.id2 = self.id2 + 1
    
        #r, zc, c_residue = circlefittest(z1[id1:id2])
        circle_obj = Circlefit(self.z1[self.id1:self.id2])
        self.r = float(circle_obj.r)
        self.zc = circle_obj.zc
        self.c_residue = circle_obj.residue
    
        self.theta_zc = np.angle(self.zc)

        # rotation and translation to center
        self.z2 = (self.zc - self.z1)*np.exp(-1.j*self.theta_zc) # same as center_circle function

        self.ft, self.zt = trimdata(self.f,self.z2,self.f0g,self.Qg,self.numspan) # estv['f0'],estv['Q']
        self.f0gid = find_nearest_indice(self.f,self.f0g)


    def fphaselin(self, ft, Q, f0, phi): # f is the independent variable here
        xg = (ft/f0) - 1. # also written as (f-f0)/f0
    
        ang = phi + 2.*np.arctan(-2.*Q*xg)
        return ang


    def calcphasebounds(self, Q, f0, phi, fmargin = 0.02, phimargin = 0.05):
        # order is Q, f0, and phi
        bounds = ((1000., f0-fmargin*f0, phi-phimargin*np.abs(phi)), (1e7, f0+fmargin*f0, phi+phimargin*np.abs(phi)))
        return bounds


    def windowlorentz(self,f):
        # f_c is the resonant frequency or probe tone frequency + LO
        f_c = self.tone_freq_lo
        x = 2. / self.window_width * (f - f_c) # 2/W * (f - f_c)
        uncertainty = np.sqrt(1. + np.square(x)) # sqrt(1 + x^2)
        return uncertainty


    def windowgauss(self,f):
        # f_c is the resonant frequency or probe tone frequency + LO
        f_c = self.tone_freq_lo
        x = 2. / self.window_width * (f - f_c) # 2/W * (f - f_c)
        uncertainty = np.exp(np.square(x) / 2.) # exp(x^2 / 2)
        return uncertainty


    def fitphasetest(self, ft, angt, Q, f0, phi, **keywords):
    
        p0 = [Q, f0, phi]

        if self.use_weight:
            if self.weight_type == 'lorentz':
                uncertainty_angt = self.windowlorentz(ft)
            elif self.weight_type == 'gauss':
                uncertainty_angt = self.windowgauss(ft)
            else:
                uncertainty_angt = np.ones(np.shape(ft))
        else:
            uncertainty_angt = np.ones(np.shape(ft))
    
        if ('bounds' in keywords):
            bounds = keywords['bounds']
            if self.verbose:
                print('Fitting data with user-specified bounds')
                print(bounds)
            popt, pcov = curve_fit(self.fphaselin, xdata=ft, ydata=angt, p0=p0, bounds=bounds)
        elif self.use_weight:
            print('Fitting data with weighting')
            popt, pcov = curve_fit(self.fphaselin, xdata=ft, ydata=angt, p0=p0, sigma=uncertainty_angt, absolute_sigma=False)
        else:
            if self.verbose:
                print('Fitting data')
            popt, pcov = curve_fit(self.fphaselin, xdata=ft, ydata=angt, p0=p0)
    
        fit_result = self.fphaselin(ft,popt[0],popt[1],popt[2])
        p0_result = self.fphaselin(ft,p0[0],p0[1],p0[2])
    
        return popt, pcov, fit_result, p0_result, uncertainty_angt


    def fitphase2(self,f,z,f0g,Qg,numspan=2,showplot=False,use_bounds=False):
        # give z2 for correct result
    
        zinf = (z[0]+z[-1])/2.
        ezinf = zinf/np.abs(zinf)

        z1_1 = z/(-ezinf)
        ang = np.angle(z1_1) + np.angle(-ezinf) # before: np.angle(z1_1) + np.angle(-ezinf) #phase2(z1_1) #+ np.angle(-ezinf) #phase2(z1_1) + np.angle(-ezinf) #8/11/24: np.angle(z1_1) + np.angle(-ezinf) #phase2(z1_1) + np.angle(-ezinf) #before: phase2(z1_1) + np.angle(-ezinf)

        hnumsmopts = int(np.floor((f0g/Qg/3.)/(f[1]-f[0])))
        hnumsmopts = int(np.amax([np.amin([hnumsmopts, 20]),0]))

        if hnumsmopts-1 < 0:
            hnumsmopts_l = 0
        else:
            hnumsmopts_l = hnumsmopts-1

        N = int(hnumsmopts*2)
        zm = smoothdata2(z,N)

        zm_inf = (zm[0]+zm[-1])/2.
        ezminf = zm_inf/np.abs(zm_inf)
        zm1_1 = zm/(-ezminf)
        angm = np.angle(zm1_1) + np.angle(-ezminf) # before: np.angle(zm1_1) + np.angle(-ezminf) #phase2(zm1_1) #+ np.angle(-ezminf) #phase2(zm1_1) + np.angle(-ezminf) #phase2(zm1_1) + np.angle(-ezminf)
    
        dangm = angm[hnumsmopts_l:len(angm)] - angm[0:len(angm)-hnumsmopts+1]

        fd = f[hnumsmopts_l:len(angm)] - f[0:len(angm)-hnumsmopts+1]

        dangmin = np.amin(dangm[hnumsmopts_l:len(dangm)-hnumsmopts+1])
        idx = np.argmin(dangm[hnumsmopts_l:len(dangm)-hnumsmopts+1])

        f0 = f[idx + hnumsmopts]
        # print(f0) # identical to matlab

        fstep = f[1]-f[0]
        idx = find_nearest_indice(f, f0)
        wid = int(np.floor(f0/Qg/3./fstep))

        fmid = f[np.amax([idx-wid,0]):np.amin([idx+wid+1,len(f)])]
    
        xmid = (f0-fmid)/f0
        angmid = ang[np.amax([idx-wid,0]):np.amin([idx+wid+1,len(f)])]

        slope, intercept, r, p, se = scipy.stats.linregress(xmid,angmid)

        phi = intercept
        Q = np.abs(slope/4.)
    
        ft, zt = trimdata(f,z,f0,Q,numspan)
        ft, angt = trimdata(f,ang,f0,Q,numspan)
    
        if use_bounds:
            bounds = self.calcphasebounds(Q, f0, phi)
            popt, pcov, fit_result, p0_result, uncertainty_angt = self.fitphasetest(ft, angt, Q, f0, phi, bounds = bounds)
        else:
            popt, pcov, fit_result, p0_result, uncertainty_angt = self.fitphasetest(ft, angt, Q, f0, phi)

        # order is Q, f0, phi
        fphasetest = self.fphaselin(ft, popt[0], popt[1], popt[2])

        if self.verbose:
            # print('initial guess for f0 and Q (given): ', f0g, Qg)
            # print('guess for f0 from smoothed data: ', f0)
            # print('guess from linear fit for phi, Q: ', phi,Q)
            # print('fit results for Q, f0, and phi are ', popt)
            pass
    
        pherr = angt - fphasetest #np.abs(angt - fphasetest)

        return popt, pcov, pherr, fit_result, p0_result, ft, zt, angt, ang, uncertainty_angt, ezinf


    def fits21sim(self,showplot=False,use_bounds=False):
    
        result = {}
        result['tau'] = self.tau

        numspan = self.numspan
    
        z1 = removecable(self.f,self.z,self.tau,verbose=self.verbose)
    
        f0g, Qg, idf0, iddf = estpara3(self.f,z1,verbose=self.verbose)
    
        # fit circle using data points f0g +- 2*f0g/Qg
        id1 = np.amax([idf0-2*iddf,0])
        id2 = np.amin([idf0+2*iddf,len(self.f)])
    
        if id2 < len(self.f): # this is needed I think
            id2 = id2 + 1
    
        #r, zc, c_residue
        circle_obj = Circlefit(z1[id1:id2])
        result['r'] = float(circle_obj.r)
        result['zc'] = circle_obj.zc
        result['c_residue'] = circle_obj.residue
    
        if self.verbose:
            print('f0, Q, idf0, iddf guess ', f0g, Qg, idf0, iddf) # matches matlab
            print('r, zc: ', result['r'],result['zc'])

        theta_zc = np.angle(result['zc'])

        # rotation and translation to center
        z2 = (result['zc'] - z1)*np.exp(-1.j*theta_zc)
    
        # now fit to data below
        # order is Q, f0, phi
        if use_bounds:
            popt, pcov, pherr, fit_result, x0_result, ft, zt, angt, ang, uncertainty_angt, ezinf = self.fitphase2(self.f,z2,f0g,Qg,numspan,use_bounds=use_bounds)
        else:
            popt, pcov, pherr, fit_result, x0_result, ft, zt, angt, ang, uncertainty_angt, ezinf = self.fitphase2(self.f,z2,f0g,Qg,numspan)
    
        result['popt'] = popt
        result['Q'] = popt[0]
        result['f0'] = popt[1]
        result['phi'] = popt[2]
    
        result['pcov'] = pcov
        result['pherr'] = pherr
        result['popt_err'] = np.sqrt(np.diag(pcov)) # one standard deviation errors of fit parameters # std
    
        result['fit_result'] = fit_result
        result['fit_result2'] = self.fphaselin(self.f,result['Q'],result['f0'],result['phi'])
        result['fit_resultt'] = self.fphaselin(ft,result['Q'],result['f0'],result['phi']) # Q, f0, phi
        result['x0_result'] = x0_result
    
        result['ft'] = ft
        result['zt'] = zt
        result['ang'] = ang
        result['angt'] = angt
        result['uncertainty_angt'] = uncertainty_angt
        result['ezinf'] = ezinf
    
        idft1 = find_nearest_indice(self.f,ft[0])
        widft = len(ft)

        zt_n = self.z[idft1:idft1+widft] # idft1+widft-1 before, matches matlab
        ft_n = self.f[idft1:idft1+widft]
        result['zt_n'] = zt_n
        result['ft_n'] = ft_n

        result['zd'] = -result['zc']/np.abs(result['zc'])*np.exp(1.j*result['phi'])*result['r']*2.
        result['zinf'] = result['zc'] - result['zd']/2.
    
        result['Qc'] = np.abs(result['zinf'])/np.abs(result['zd'])*result['Q']
        result['Qi'] = 1./(1./result['Q']-1./result['Qc'])
    
        result['zf0'] = result['zinf'] + result['zd']

        #projection of zf0 into zinf
        l = np.real(result['zf0']*np.conj(result['zinf']))/np.abs(result['zinf'])**2

        # Q/Qi0 = l
        result['Qi0'] = result['Q']/l

        result['f00id'] = find_nearest_indice(self.f, result['f0'])
        result['f0_corr'] = result['f0'] # same for linear version
        result['f0id'] = result['f00id'] # same for linear version

        # for converting from ang to IQ space
        iq_model = 1. - (((result['Q']/result['Qc'])*np.exp(1.j*result['phi']))/(1. + 2.j*result['Q']*((self.f/result['f0']) - 1.)))
        iq_modelt = 1. - (((result['Q']/result['Qc'])*np.exp(1.j*result['phi']))/(1. + 2.j*result['Q']*((result['ft']/result['f0']) - 1.)))
        result['iq_model'] = iq_model # z2 = A*iq_model --> A = z2/iq_model
        result['iq_modelt'] = iq_modelt

        result['A_mag'] = result['r']
        result['A_magt'] = result['r']

        ang_to_iq = result['A_mag']*(np.cos(result['fit_result2']-np.angle(-result['ezinf'])) + 1.j*np.sin(result['fit_result2']-np.angle(-result['ezinf'])))*np.abs(result['iq_model'])
        angt_to_iq = result['A_magt']*(np.cos(result['fit_resultt']-np.angle(-result['ezinf'])) + 1.j*np.sin(result['fit_resultt']-np.angle(-result['ezinf'])))*np.abs(result['iq_modelt'])
        result['ang_to_z2'] = ang_to_iq*-result['ezinf']
        result['angt_to_z2'] = angt_to_iq*-result['ezinf']

        result['ang_to_z1'] = result['zc'] - (result['ang_to_z2']/np.exp(-1.j*np.angle(result['zc'])))
        result['angt_to_z1'] = result['zc'] - (result['angt_to_z2']/np.exp(-1.j*np.angle(result['zc'])))

        result['ang_to_z'] = result['ang_to_z1']/(np.exp(-1.j*2.*np.pi*(self.f/1e9)*self.tau)) # f in GHz, tau (negative) in ns
        result['angt_to_z'] = result['angt_to_z1']/(np.exp(-1.j*2.*np.pi*(result['ft']/1e9)*self.tau)) # f in GHz, tau (negative) in ns
    
        return result

 
    def fphasenlin(self, f, Q, alpha, f00, phi, r, z): # f is the independent variable here
        # self, f, Q, alpha, f00, phi, r, z
        A = z - r*np.exp(1.j*(phi+np.pi))
        f0 = f00 - alpha*np.abs(A)**2
        x = (f - f0)/f0
    
        ang = phi + 2.*np.arctan(-2.*Q*x)
        return ang


    def fphasenlin2(self, f, Q, alpha, f00, phi, r, z): # f is the independent variable here
        # self, f, Q, alpha, f00, phi, r, z
        A = z - r*np.exp(1.j*(phi+np.pi))
        f0 = f00 - alpha*np.abs(A)**2
        x = (f - f0)/f0
    
        ang = phi + 2.*np.arctan(-2.*Q*x)
        return ang, x


    def calcphasenlinbounds(self, Q, alpha, f0, phi, fmargin = 0.02, phimargin = 0.05):
        # order is Q, alpha, f0, and phi
        bounds = ((1000., -np.inf, f0-fmargin*f0, phi-phimargin*np.abs(phi)), (1e7, np.inf, f0+fmargin*f0, phi+phimargin*np.abs(phi)))
        return bounds


    def fitphasenlintest2(self, ft, angt, Q, alpha, f0, phi, r, zt, **keywords):
    
        #f0_corr = f01 - alpha*(2.*r)**2
        p0 = [Q, alpha, f0, phi] #[Q, alpha, f0_corr, phi]

        if self.use_weight:
            if self.weight_type == 'lorentz':
                uncertainty_angt = self.windowlorentz(ft)
            elif self.weight_type == 'gauss':
                uncertainty_angt = self.windowgauss(ft)
            else:
                uncertainty_angt = np.ones(np.shape(ft))
        else:
            uncertainty_angt = np.ones(np.shape(ft))

        if ('bounds' in keywords):
            if self.verbose:
                print('Fitting data with user-specified bounds')
                print(bounds)
            bounds = keywords['bounds']
            if ('err' in keywords):
                if self.verbose:
                    print('Using calculated error')
                err = keywords['err']
                popt, pcov = curve_fit(lambda f_lamb,a,b,c,d: self.fphasenlin(f_lamb,a,b,c,d,r,zt), ft, angt, p0=p0,
                               sigma=err, bounds=bounds)
            else:
                try:
                    popt, pcov = curve_fit(lambda f_lamb,a,b,c,d: self.fphasenlin(f_lamb,a,b,c,d,r,zt), ft, angt, p0=p0,
                                   bounds=bounds)
                except RuntimeError:
                    print('Failed to fit')
                    pass
        else:
            if self.verbose:
                print('Fitting data')
            if ('err' in keywords):
                if self.verbose:
                    print('Using calculated error')
                err = keywords['err']
                # this is giving me issues
                popt, pcov = curve_fit(lambda f_lamb,a,b,c,d: self.fphasenlin(f_lamb,a,b,c,d,r,zt), ft, angt, p0=p0,sigma=err)
            elif self.use_weight:
                print('Fitting data with weighting')
                try:
                    popt, pcov = curve_fit(lambda f_lamb,a,b,c,d: self.fphasenlin(f_lamb,a,b,c,d,r,zt), ft, angt, p0=p0, sigma=uncertainty_angt, absolute_sigma=False)
                    fit_result, xt = self.fphasenlin2(ft,popt[0],popt[1],popt[2],popt[3],r,zt)
                except RuntimeError:
                    print('Failed to fit data')
                    popt = [1.,1.,1.,1.]
                    pcov = [1.,1.,1.,1.]
                    fit_result = 1.
                    xt = 1.
            else:
                try:
                    popt, pcov = curve_fit(lambda f_lamb,a,b,c,d: self.fphasenlin(f_lamb,a,b,c,d,r,zt), ft, angt, p0=p0)
                    fit_result, xt = self.fphasenlin2(ft,popt[0],popt[1],popt[2],popt[3],r,zt)
                except RuntimeError:
                    print('Failed to fit data')
                    popt = [1.,1.,1.,1.]
                    pcov = [1.,1.,1.,1.]
                    fit_result = 1.
                    xt = 1.

        p0_result = self.fphasenlin(ft,p0[0],p0[1],p0[2],p0[3],r,zt)
        return popt, pcov, fit_result, p0_result, xt, uncertainty_angt


    def fitphasenlin_c2(self,f,z,estv,numspan=2,use_bounds=False):
    
        zinf = (z[0]+z[-1])/2.
        ezinf = zinf/np.abs(zinf)

        z1_1 = z/(-ezinf) # z1 in matlab, renamed due to redundancy

        ang = np.angle(z1_1) + np.angle(-ezinf) # before: np.angle(z1_1) + np.angle(-ezinf) #phase2(z1_1) #+ np.angle(-ezinf) #phase2(z1_1) + np.angle(-ezinf) #8/11/24: np.angle(z1_1) + np.angle(-ezinf) #phase2(z1_1) + np.angle(-ezinf) # before: phase2(z1_1) + np.angle(-ezinf)

        # linear regime used with estpara # -20 dBm data and linear fit # import data to test for now
        ft, angt = trimdata(f,ang,estv['f0'],estv['Q'],numspan)
        ft, zt = trimdata(f,z,estv['f0'],estv['Q'],numspan)

        f00g = estv['f0']
        Qgg = estv['Q'] # originally Qg
        z1_n = np.mean(zt[:5]) # z1
        z2_n = np.mean(zt[-5:]) # z2
        v = np.amax(np.abs(zt-z1_n) + np.abs(zt-z2_n))
        idf0 = np.argmax(np.abs(zt-z1_n) + np.abs(zt-z2_n))
        phig = estv['phi']
        alphag = np.abs(ft[idf0] - f00g)/(2.*estv['r'])**2

        # order is Q, alpha, f0, phi
        # not fitting to estv['r'], zt
        if use_bounds:
            bounds = self.calcphasenlinbounds(Qgg, alphag, f00g, phig)
            popt, pcov, fit_result, x0_result, xt, uncertainty_angt = self.fitphasenlintest2(ft, angt, Qgg, alphag, f00g, phig, estv['r'], zt, 
                                                                bounds = bounds)
        else:
            popt, pcov, fit_result, x0_result, xt, uncertainty_angt = self.fitphasenlintest2(ft, angt, Qgg, alphag, f00g, phig, estv['r'], zt)

        if np.mean(popt) == 1.:
            flag = 1
        else:
            flag = 0

        if self.verbose:
            print('fit results for Q, alpha, f0, and phi are ', popt)

        # get error
        var = np.sum((angt-fit_result)**2)/(angt.shape[0] - 1)
        err = np.ones(angt.shape[0])*np.sqrt(var)

        pherr = angt - fit_result #np.abs(angt - fit_result)
        f00 = popt[2]
    
        return popt, pcov, pherr, fit_result, x0_result, ft, zt, angt, ang, var, err, xt, flag, uncertainty_angt, ezinf


    def fits21simnlin(self,use_bounds=False,showplot=False):
        # assumes data has already been loaded and corrected
    
        numspan = self.numspan

        result = {}
        result['tau'] = self.tau
    
        z1 = removecable(self.f,self.z,self.tau,verbose=self.verbose)
    
        f0g, Qg, idf0, iddf = estpara3(self.f,z1,verbose=self.verbose,result0=self.result0)
    
        # fit circle using data points f0g +- 2*f0g/Qg
        id1 = np.amax([idf0-2*iddf,0])
        id2 = np.amin([idf0+2*iddf,len(self.f)])
    
        if id2 < len(self.f):
            id2 = id2 + 1

        #r, zc, c_residue
        circle_obj = Circlefit(z1[id1:id2])
        result['r'] = float(circle_obj.r)
        result['zc'] = circle_obj.zc
        result['c_residue'] = circle_obj.residue
    
        # if self.verbose:
        #     print('f0, Q, idf0, iddf guess ', f0g, Qg, idf0, iddf) # matches matlab
        #     print('r, zc: ', result['r'],result['zc'])

        # create estv
        estv = {}
        estv['f0'] = f0g
        estv['Q'] = Qg
        estv['r'] = result['r']
        estv['phi'] = self.result0['phi']

        theta_zc = np.angle(result['zc'])

        # rotation and translation to center
        z2 = (result['zc'] - z1)*np.exp(-1.j*theta_zc)
    
        # now fit to data below
        # print(self.f,z2)
        if use_bounds:
            popt, pcov, pherr, fit_result, x0_result, ft, zt, angt, ang, var, err, xt, flag, uncertainty_angt, ezinf = self.fitphasenlin_c2(self.f,z2,estv,numspan,use_bounds=use_bounds)
        else:
            popt, pcov, pherr, fit_result, x0_result, ft, zt, angt, ang, var, err, xt, flag, uncertainty_angt, ezinf = self.fitphasenlin_c2(self.f,z2,estv,numspan)
    
        result['popt'] = popt # Q, alpha, f0, phi
        result['Q'] = popt[0]
        result['alpha'] = popt[1]
        result['f0'] = popt[2]
        result['phi'] = popt[3]
    
        result['pcov'] = pcov
        result['pherr'] = pherr
        result['popt_err'] = np.sqrt(np.diag(pcov)) # one standard deviation errors of fit parameters # std
    
        result['fit_result'] = fit_result # on trimmed data, ft, zt
        result['fit_result2'] = self.fphasenlin(self.f,result['Q'],result['alpha'],result['f0'],result['phi'],result['r'],z2) # Q, alpha, f0, phi # fit ang
        result['fit_resultt'] = self.fphasenlin(ft,result['Q'],result['alpha'],result['f0'],result['phi'],result['r'],zt) # fit ang
        result['x0_result'] = x0_result
    
        result['ft'] = ft
        result['zt'] = zt
        result['ang'] = ang
        result['angt'] = angt
        result['uncertainty_angt'] = uncertainty_angt
        result['ezinf'] = ezinf

        result['fit_var'] = var
        result['fit_err'] = err
        result['x'] = xt
    
        idft1 = find_nearest_indice(self.f,ft[0])
        widft = len(ft)
    
        zt_n = self.z[idft1:idft1+widft] # idft1+widft-1 before, matches matlab
        ft_n = self.f[idft1:idft1+widft]
        result['zt_n'] = zt_n
        result['ft_n'] = ft_n

        result['zd'] = -result['zc']/np.abs(result['zc'])*np.exp(1.j*result['phi'])*result['r']*2
    
        result['zinf'] = result['zc'] - result['zd']/2.

        Qc_part = np.abs(np.abs(result['zc'])/(2.*result['r']*np.exp(1.j*result['phi'])) + 0.5)
        result['Qc'] = Qc_part*result['Q']

        result['Qi'] = 1./(1./result['Q']-1./result['Qc'])
        result['bif'] = (result['alpha']*(2.*result['r'])**2)/(result['f0']/result['Q'])
        result['dff0'] = result['alpha']*(2.*result['r'])**2/result['f0']
        result['zf0'] = result['zinf'] + result['zd']

        # ignored delta r term for now
        result['bif_err'] = np.sqrt((4.*result['r']**2*result['Q']/result['f0'])**2*result['popt_err'][1]**2 +
                                   (-4.*result['r']**2*result['alpha']*result['Q']/result['f0']**2)**2*result['popt_err'][2]**2 +
                                   (4.*result['r']**2*result['alpha']/result['f0'])**2*result['popt_err'][0]**2)

        result['Qc_err'] = np.sqrt((np.abs(np.abs(result['zc'])*-1.j/(2.*result['r']*np.exp(1.j*result['phi'])))*
                                    result['Q'])**2*result['popt_err'][3]**2 + 
                                    np.abs(np.abs(result['zc'])/(2.*result['r']*np.exp(1.j*result['phi'])) + 0.5)**2*
                                    result['popt_err'][0]**2)

        result['Qi_err'] = np.sqrt((Qc_part/(Qc_part-1.))**2*result['popt_err'][0]**2 + 
                                    (-1./((1./result['Q']) - (1./(Qc_part*result['Q'])))**2*(1./(Qc_part**2*result['Q']))*
                                     np.abs(-1.j*np.abs(result['zc'])/(2.*result['r']*np.exp(1.j*result['phi']))))**2*
                                    result['popt_err'][3]**2)


        # ignored delta zc and delta r terms for now
        result['zd_err'] = np.sqrt(((-result['zc']/np.abs(result['zc']))*2.*result['r']*1.j*np.exp(1.j*result['phi']))**2*result['popt_err'][3]**2)
        result['zinf_err'] = np.sqrt((result['zc']/np.abs(result['zc'])*1.j*np.exp(1.j*result['phi'])*result['r'])**2*result['popt_err'][3]**2)

        #projection of zf0 into zinf
        l = np.real(result['zf0']*np.conj(result['zinf']))/np.abs(result['zinf'])**2
        result['l'] = l
        # ignored delta zc term for now, should only be real
        result['l_err'] = np.sqrt(np.real(result['zc']/result['zinf']**2)**2*np.abs(result['zd_err'])**2)
    
    
        # Q/Qi0 = l
        result['Qi0'] = result['Q']/l
        result['Qi0_err'] = np.sqrt(result['popt_err'][0]**2/result['l']**2 + result['Q']**2/result['l']**4*result['l_err']**2)


        # f0 from fit
        result['f00id'] = find_nearest_indice(self.f, result['f0']) # fit f0 called f00 in matlab code

        # similar to corrected f0 in fphasenlin
        result['f0_corr'] = result['f0'] - result['alpha']*(2.*estv['r'])**2
        result['f0id'] = find_nearest_indice(self.f,result['f0_corr'])
        #print('f0id',result['f0id'])

        if flag == 0:
            if sum(np.abs(result['pherr']) > self.pherr_threshold) > self.pherr_threshold_num:
                flag = 2

        result['flag'] = flag

        # for converting from ang to IQ space
        iq_model = 1. - (((result['Q']/result['Qc'])*np.exp(1.j*result['phi']))/(1. + 2.j*result['Q']*((self.f/result['f0_corr']) - 1.)))
        iq_modelt = 1. - (((result['Q']/result['Qc'])*np.exp(1.j*result['phi']))/(1. + 2.j*result['Q']*((result['ft']/result['f0_corr']) - 1.)))
        result['iq_model'] = iq_model # z2 = A*iq_model --> A = z2/iq_model
        result['iq_modelt'] = iq_modelt

        result['A_mag'] = result['r']
        result['A_magt'] = result['r']

        ang_to_iq = result['A_mag']*(np.cos(result['fit_result2']-np.angle(-result['ezinf'])) + 1.j*np.sin(result['fit_result2']-np.angle(-result['ezinf'])))
        angt_to_iq = result['A_magt']*(np.cos(result['fit_resultt']-np.angle(-result['ezinf'])) + 1.j*np.sin(result['fit_resultt']-np.angle(-result['ezinf'])))
        result['ang_to_z2'] = ang_to_iq*-result['ezinf']
        result['angt_to_z2'] = angt_to_iq*-result['ezinf']

        result['ang_to_z1'] = result['zc'] - (result['ang_to_z2']/np.exp(-1.j*np.angle(result['zc'])))
        result['angt_to_z1'] = result['zc'] - (result['angt_to_z2']/np.exp(-1.j*np.angle(result['zc'])))

        result['ang_to_z'] = result['ang_to_z1']/(np.exp(-1.j*2.*np.pi*(self.f/1e9)*self.tau)) # f in GHz, tau (negative) in ns
        result['angt_to_z'] = result['angt_to_z1']/(np.exp(-1.j*2.*np.pi*(result['ft']/1e9)*self.tau)) # f in GHz, tau (negative) in ns
    
        return result
    
    
    # helper functions based on Hannes's code
    def plotComplexPlane(self,z):
        plt.plot(np.real(z),np.imag(z))
        plt.xlabel('real')
        plt.ylabel('imag')

    
    def rescaleCircle(self,z,r):
        return z/float(r)


    def rotateCircle(self,z,phi):
        return z*np.e**(-1.j*phi/180.*np.pi) # or np.exp


    def plotPolar(self,z):
        plt.subplot(111, polar=True)
        plt.plot(phase2(z),np.abs(z)) # np.arctan2(np.imag(z),np.real(z))


    def plot(self):
        # make the fitted array for plotting purposes

        s = np.linspace(0,2.*np.pi,100)

        z_circlefit = self.zc+self.r*np.e**(1.j*s)
        z_circlefit_mini = self.zc+self.r*0.001*np.e**(1.j*s)
        fig = plt.subplots(2, 2)

        # upper left, s21 magnitude.  Raw data, trimmed and fit.
        plt.subplot(221)
        plt.plot(self.f,logmag(self.z1)) #,'o-') # bo
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('$|S_{21}|^2$')


        plt.subplot(222,polar=True)
        plt.plot(phase2(self.z),np.abs(self.z),'c')
        plt.plot(phase2(self.z1),np.abs(self.z1),'blue')
        plt.plot(phase2(z_circlefit),np.abs(z_circlefit),'r--') # z1[id1:id2] # k--
        plt.plot(phase2(z_circlefit_mini),np.abs(z_circlefit_mini),'r+')
        plt.plot(np.angle(self.result['zinf']),np.abs(self.result['zinf']),'k*')
        plt.plot(np.angle(self.result['zf0']),np.abs(self.result['zf0']),'ko')
        plt.plot(phase2(self.z2),np.abs(self.z2),'b')
        plt.plot(phase2(self.zt),np.abs(self.zt),'m')

        # similar to corrected f0 in fphasenlin
        plt.plot(self.result['ang'][self.result['f0id']],np.abs(self.z2[self.result['f0id']]),'ko')
        # f0 from fit
        plt.plot(self.result['ang'][self.result['f00id']],np.abs(self.z2[self.result['f00id']]),'mo')


        plt.subplot(223)
        plt.plot(self.f,self.result['ang'],'.')
        color = plt.gca().lines[-1].get_color() # get most recent line color

        plt.plot(self.result['ft'],self.result['angt'],'o',color=color)
        plt.plot(self.result['ft'],self.result['fit_result'],'k--')

        # trimmed data that was fit
        plt.plot(self.result['ft'][0],self.result['angt'][0],'c+')
        plt.plot(self.result['ft'][-1],self.result['angt'][-1],'c+')

        # similar to corrected f0 in fphasenlin
        plt.plot(self.f[self.result['f0id']],self.result['ang'][self.result['f0id']],'ko')
        # f0 from fit
        plt.plot(self.f[self.result['f00id']],self.result['ang'][self.result['f00id']],'mo')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (rad)')


        plt.subplot(224)
        plt.plot(self.result['ft'],self.result['pherr'])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Residuals (rad)')

        plt.tight_layout()
        plt.show()
