import numpy as np

import matplotlib.pyplot as plt
import math
from scipy import interpolate, linalg, optimize


class Circlefit():
    ''' Class to perform a circle fit on data '''
    def __init__(self,z,scaling=False):
        ''' data needs to be complex (like I + jQ)
        '''
        self.z = z 
        self.scaling = scaling
        self.r, self.zc, self.residue = self.nsphere_fit_test(scaling=self.scaling)

    # based on scikit-guess package
    def nsphere_fit_test(self,scaling): # not using axis right now # axis=-1
        """
        Fit an n-sphere to ND data.
        The center and radius of the n-sphere are optimized using the Coope
        method. The sphere is described by
        .. math::
           \left \lVert \vec{x} - \vec{c} \right \rVert_2 = r
        Parameters
        ----------
        x : array-like
            The n-vectors describing the data. Usually this will be a nxm
            array containing m n-dimensional data points.
        axis : int
            The axis that determines the number of dimensions of the
            n-sphere. All other axes are effectively raveled to obtain an
            ``(m, n)`` array.
        scaling : bool
            If `True`, scale and offset the data to a bounding box of -1 to
            +1 during computations for numerical stability. Default is
            `False`.
        Return
        ------
        r : scalar
            The optimal radius of the best-fit n-sphere for `x`.
        c : array
            An array of size `x.shape[axis]` with the optimized center of
            the best-fit n-sphere.
        References
        ----------
        - [Coope]_ "\ :ref:`ref-cfblanls`\ "
        """
    
        x = np.column_stack((self.z.real, self.z.imag)) # or rather xy
        x = np.asarray(x) #preprocess(x, float=True, axis=axis)
        n = x.shape[-1]
        x = x.reshape(-1, n)
        m = x.shape[0]

        B = np.empty((m, n+1), dtype=x.dtype)
        X = B[:,:-1]
        X[:] = x
        B[:,-1] = 1 # based on Coope's method, column vector bj = [aj ... 1]
        #print(np.shape(B))
        #print(np.shape(x))                                                                                                                                                                                                                                                                                                                                                                                                                                                  kid_phase_fit_change.py                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        if scaling:
            print('Using scaling')
            xmin = X.min()
            xmax = X.max()
            scale = 0.5 * (xmax - xmin)
            offset = 0.5 * (xmax + xmin)
            X -= offset
            X /= scale

        d = np.sum(X**2, axis=-1) # square(X).sum(axis=-1) # sum of aj^2
        #print(np.shape(d))

        # least squares of By = d or minimize |d - B y|
        y, residue, rank_B, B_sing = linalg.lstsq(B, d, overwrite_a=True, overwrite_b=True) # *_
        #print(np.shape(y))

        # to transform back to original coordinates to recover c or xi and r
        # need to change residues for c and r too?
        zc = 0.5 * y[:-1] # c
        r = np.sqrt(y[-1] + np.sum(zc**2)) # sqrt(y[-1] + square(c).sum())

        if scaling:
            r *= scale
            zc *= scale
            zc += offset

        zc_complex = zc[0] + 1.j*zc[1]

        return r, zc_complex, residue


# refer to https://matplotlib.org/stable/gallery/misc/transoffset.html#sphx-glr-gallery-misc-transoffset-py
# for switching between projections using ax and subplot
def removecable(f,z,tau,verbose=False,showplot=False): 
    # f,z,tau,f1
    #f1 = f[0]
    if tau > 0:
        tau = -1.*tau
    
    if(f[0]>1e7):
        if verbose:
            print('Converting f from Hz to GHz before removing cable delay')
        f = f/1e9
    f1 = f[0]
    #print(f)
    #print(f1)
    #print(f-f1)
    z1 = z*np.exp(-1.j*2.*np.pi*(f)*tau) # (f-f1) # assumes tau is already negative

    if showplot:
        #fig = plt.figure()
        #ax = plt.subplot(221, projection='polar')
        fig = plt.subplots(2, 2)

        plt.subplot(211)
        #ax = plt.subplot(222, projection='rectilinear') # default projection
        plt.plot(f,logmag(z),'c')
        plt.plot(f,logmag(z1),'k')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('$|S_{21}|^2$')
        #plt.show()

        plt.subplot(222,polar=True) # np.arctan2(np.imag(self.z),np.real(self.z))
        plt.plot(phase2(z),np.abs(z),'c')
        plt.plot(phase2(z1),np.abs(z1),'k')
        ##polar2(z,color='c')
        ##polar2(z1,color='k')
        plt.show()
    
    return z1


def find_nearest_indice(x,val):
    idx = np.searchsorted(x, val, side="left")
    if idx > 0 and (idx == len(x) or math.fabs(val - x[idx-1]) < math.fabs(val - x[idx])):
        return idx - 1 #x[idx-1]
    else:
        return idx #x[idx]


def trimdata(f,z,f0,Q,numspan=1):
    f0id = find_nearest_indice(f, f0)
    idspan =  (f0/Q) / (f[1]-f[0])
    
    idstart = int(np.floor(np.amax([f0id - idspan*numspan,0])))
    #print('idstart',idstart)
    if idstart != 0: # new, 3/17/23, matlab comparison
        idstart = idstart + 1

    idstop = int(np.floor(np.amin([f0id + idspan*numspan,len(f)])))+1

    if idstop > len(f):
        idstop = len(f)

    fb = f[idstart:idstop]
    zb = z[idstart:idstop]
    return fb, zb


def smoothdata2(y, N): # more matlab like
    # compute moving average with window size N
    
    if N%2==0: # checks if N is even
        N = N - 1

    y_out = np.convolve(y, np.ones(N, dtype=int), mode='valid')/N
    r = np.arange(1,N-1,2)
    start = np.cumsum(y[:N-1])[::2]/r
    stop = (np.cumsum(y[:-N:-1])[::2]/r)[::-1]
    y_smooth = np.concatenate((start, y_out, stop))
    return y_smooth


# put this in one of the classes instead?
def estpara3(f,z,verbose=False,**keywords):
    # to do: add guess for phi0?

    if ('result0' in keywords):
        if verbose:
            print('Using linear regime fit result0')
        result0 = keywords['result0']
        f0g = result0['f0'] #[0][0]
        Qg = result0['Q'] #[0][0]
        v = np.amin(np.abs(f - f0g))
        idf0 = np.argmin(np.abs(f - f0g))
        iddf = np.abs(find_nearest_indice(f, f0g - f0g/Qg/2.) - find_nearest_indice(f, f0g + f0g/Qg/2.))
    else:
        z1 = np.mean(z[:10])
        z2 = np.mean(z[len(f)-11:len(f)])
        zr = (z1 + z2)/2.
        dmax = np.amax(np.abs(z - z1) + np.abs(z - z2))
        idf0 = np.argmax(np.abs(z - z1) + np.abs(z - z2))
        z0 = z[idf0]
        f0g = f[idf0]
        zcg = (zr + z0)/2.
        rg = np.abs(zr - z0)/2.
        zz1 = z[:idf0+1]
        r01 = np.amin(np.abs(np.abs(zz1-z0) - np.sqrt(2.)*rg))
        id3db1 = np.argmin(np.abs(np.abs(zz1-z0) - np.sqrt(2.)*rg))
        zz2 = z[idf0+1:len(z)]
        r02 = np.amin(np.abs(np.abs(zz2-z0) - np.sqrt(2.)*rg))
        id3db2 = np.argmin(np.abs(np.abs(zz2-z0) - np.sqrt(2.)*rg))
        iddf = np.amax([np.abs(idf0 - id3db1), np.abs(id3db2)])
        Qg = f0g / (2.*iddf*(f[1]-f[0]))

    # set to correct dtypes  
    f0g = float(f0g)
    Qg = float(Qg)
    idf0 = int(idf0)
    iddf = int(iddf)

    return f0g, Qg, idf0, iddf


def plot_summary_for_index(data,ii,fig=None,ax=None):
    if not fig: fig, ax = plt.subplots(2, 2)

    ax[0,0].plot(data[:,ii,0],data[:,ii,1])
    ax[0,0].set(ylabel='i')

    ax[0,1].plot(data[:,ii,0],data[:,ii,2])
    ax[0,1].set(ylabel='q',xlabel='Freq (Hz)')

    y=magS21(data[:,ii,1],data[:,ii,2])
    ax[1,0].plot(data[:,ii,0],y)
    ax[1,0].set(ylabel='mag S21 (lin)',xlabel='Freq (Hz)')

    ax[1,1].plot(data[:,ii,1],data[:,ii,2])
    ax[1,1].set(xlabel='i',ylabel='q')
    return fig,ax
    

def guess_tone_drive_atten(tone_range, powlist, a_guess, a_predict_threshold, Q_dict, Qc_dict, omega_r_dict, a_dict, E_star_dict, bif_flag_filter_dict):
        # pick a new a_guess and guess tone drive attenuation values based on previous fits

    a_predict_flag_dict = {}
    Pro_test_guess_dBm_dict = {}
    Pro_test_low_dBm_dict = {}
    a_predict_off_dict = {}
    for ii in tone_range:
        a_ii = a_dict[ii][1:]
        bif_flag_filter_ii = bif_flag_filter_dict[ii]
        if np.asarray(a_ii).size == 0:
            E_star_2 = 0
            a_predict = 0
            a_predict_pow2 = 0
            Pro_test_low = 0
            Pro_test_low_dBm = 0
            a_predict_off = 0
            Pro_test_low_guess = 0
            Pro_test_low_guess_dBm = 0
            a_predict_flag = 1
        elif np.asarray(a_ii)[bif_flag_filter_ii].size == 0:
            E_star_2 = 0
            a_predict = 0
            a_predict_pow2 = 0
            Pro_test_low = 0
            Pro_test_low_dBm = 0
            a_predict_off = 0
            Pro_test_low_guess = 0
            Pro_test_low_guess_dBm = 0
            a_predict_flag = 1
        else: # uses bif_flag_filter_ii to skip poor fits/failed fits in calculation
            Q_0_ii = Q_dict[ii][0]
            Qc_0_ii = Qc_dict[ii][0]
            omega_r_0_ii = omega_r_dict[ii][0]
            E_star_ii = E_star_dict[ii][1:]
            E_star_2 = np.mean(np.asarray(E_star_ii)[bif_flag_filter_ii])
            a_predict = np.asarray(a_ii)[bif_flag_filter_ii][0]
            a_predict_pow2 = np.asarray(powlist[1:])[bif_flag_filter_ii][0]
            Pro_test_low = Qc_0_ii*omega_r_0_ii*a_predict*E_star_2/(2.*Q_0_ii**3)
            Pro_test_low_dBm = 10.*np.log10(Pro_test_low) + 30. # convert W to dBm
            a_predict_off = Pro_test_low_dBm/a_predict_pow2 #self.a_predict_pow
            Pro_test_low_guess = Qc_0_ii*omega_r_0_ii*a_guess*E_star_2/(2.*Q_0_ii**3)
            Pro_test_low_guess_dBm = 10.*np.log10(Pro_test_low_guess) + 30. # convert W to dBm
            if np.abs(a_predict_off-1.) > a_predict_threshold:
                a_predict_flag = 1
            else:
                a_predict_flag = 0
        a_predict_off_dict[ii] = a_predict_off
        a_predict_flag_dict[ii] = a_predict_flag # 0 is a good guess, 1 is above threshold
        if Pro_test_low_dBm != 0:
            Pro_test_low_dBm_dict[ii] = -1.*Pro_test_low_dBm
            Pro_test_guess_dBm_dict[ii] = -1.*Pro_test_low_guess_dBm
        else:
            Pro_test_low_dBm_dict[ii] = Pro_test_low_dBm
            Pro_test_guess_dBm_dict[ii] = Pro_test_low_guess_dBm
    return a_predict_off_dict, a_predict_flag_dict, Pro_test_low_dBm_dict, Pro_test_guess_dBm_dict


def find_best_adrv(
    #cls,
    fit_obj,
    a_ref=0.2, #DriveFitConfig.model_fields["a_ref"].default,
    a_min=-0.2, #DriveFitConfig.model_fields["a_min"].default,
    a_max=1.5,
    a_tolerance=0.01,
):
    """This function determining the best driving power empirically."""
    a_values = np.array(fit_obj.bif_list)
    pows = np.array(fit_obj.powlist)
    fit_flags = np.array(fit_obj.fit_flag_list)
    m = (fit_flags == 0) & (a_values <= a_max) & (a_values >= a_min)
    print('adrv test', len(m), len(pows))
    
    if len(m) != len(pows):
        p_best = None
        a_best = None
        p_flag = 1
        print("failed on first fit, skipping search for best atten drive.")
        #logger.debug(
        #    "no valid data point for finding best atten drive.",
        #)
        return locals()
    else:
        n_interp = m.sum()
        p_interp = pows[m]
        a_interp = a_values[m]
        if n_interp == 0:
             p_best = None
             a_best = None
             p_flag = 1
             print("no valid data point for finding best atten drive.")
             #logger.debug(
             #    "no valid data point for finding best atten drive.",
             #)
             return locals()
        if n_interp < 2:
             # just get the only data point
             p_best = p_interp[0]
             a_best = a_interp[0]
             p_flag = 1
             print("only one data point found, used as best atten drive {p_best=}",p_best)
             #logger.debug(
             #    f"only one data point found, used as best atten drive {p_best=}",
             #)
             return locals()
    # sort by p and check zero crossing for nonlinearity
    isort = np.argsort(p_interp)
    p_sorted = p_interp[isort]
    a_sorted = a_interp[isort]
    p_best, y, p_flag = find_first_zerocrossing1d( #cls._find_first_zerocrossing1d(
    p_sorted, a_sorted - a_ref, tolerance=a_tolerance
    )
    a_best = y + a_ref
    print("use {n_interp} data points, found {p_best=} {a_best=} {p_flag=}",n_interp,p_best,a_best,p_flag)
    #logger.debug(
    #    f"use {n_interp} data points, found {p_best=} {a_best=} {p_flag=}",
    #)
    return locals()


def find_best_adrv_tone_amp(
    #cls,
    fit_obj,
    a_ref=0.2, #DriveFitConfig.model_fields["a_ref"].default,
    a_min=-0.2, #DriveFitConfig.model_fields["a_min"].default,
    a_max=1.5,
    a_tolerance=0.01,
):
    """This function determining the best driving power empirically."""
    a_values = np.array(fit_obj.bif_list)
    pows = np.array(fit_obj.powlist)
    fit_flags = np.array(fit_obj.fit_flag_list)
    m = (a_values <= a_max) & (a_values >= a_min)
    print('adrv test', len(m), len(pows))
    
    if len(m) != len(pows):
        p_best = None
        a_best = None
        p_flag = 1
        print("failed on first fit, skipping search for best atten drive.")
        #logger.debug(
        #    "no valid data point for finding best atten drive.",
        #)
        return locals()
    else:
        n_interp = m.sum()
        p_interp = pows[m]
        a_interp = a_values[m]
        if n_interp == 0:
             p_best = None
             a_best = None
             p_flag = 1
             print("no valid data point for finding best atten drive.")
             #logger.debug(
             #    "no valid data point for finding best atten drive.",
             #)
             return locals()
        if n_interp < 2:
             # just get the only data point
             p_best = p_interp[0]
             a_best = a_interp[0]
             p_flag = 1
             print("only one data point found, used as best atten drive {p_best=}",p_best)
             #logger.debug(
             #    f"only one data point found, used as best atten drive {p_best=}",
             #)
             return locals()
    # sort by p and check zero crossing for nonlinearity
    isort = np.argsort(p_interp)
    p_sorted = p_interp[isort]
    a_sorted = a_interp[isort]
    p_best, y, p_flag = find_first_zerocrossing1d( #cls._find_first_zerocrossing1d(
    p_sorted, a_sorted - a_ref, tolerance=a_tolerance
    )
    a_best = y + a_ref
    print("use {n_interp} data points, found {p_best=} {a_best=} {p_flag=}",n_interp,p_best,a_best,p_flag)
    #logger.debug(
    #    f"use {n_interp} data points, found {p_best=} {a_best=} {p_flag=}",
    #)
    return locals()

# new, 6/13/24 # sam's first try
#@classmethod
def find_best_adrv_extrapolate(
    #cls,
    fit_obj,
    a_ref= 0.2, #DriveFitConfig.model_fields["a_ref"].default,
    a_min=-0.2, #DriveFitConfig.model_fields["a_min"].default,
    a_max=1.5, #DriveFitConfig.model_fields["a_max"].default,
    a_tolerance=0.01,
    ):
    """This function determining the best driving power empirically."""
    a_values = np.array(fit_obj.bif_list)
    pows = np.array(fit_obj.powlist)
    fit_flags = np.array(fit_obj.fit_flag_list)
    m = (fit_flags == 0) & (a_values <= a_max) & (a_values >= a_min)
    n_interp = m.sum()
    p_interp = pows[m]
    a_interp = a_values[m]
    if n_interp == 0:
        p_best = None
        a_best = None
        p_flag = 1
        print("no valid data point for finding best atten drive.")
        #logger.debug(
        #"no valid data point for finding best atten drive.",
        #)
        return locals()
    if n_interp < 2:
        # just get the only data point
        p_best = p_interp[0]
        a_best = a_interp[0]
        p_flag = 1
        print("only one data point found, used as best atten drive {p_best=}",p_best)
        #logger.debug(
        #f"only one data point found, used as best atten drive {p_best=}",
        #)
        return locals()
        # sort by p and check zero crossing for nonlinearity
    isort = np.argsort(p_interp)
    p_sorted = p_interp[isort]
    a_sorted = a_interp[isort]
    p_best, y, p_flag = find_a_extrapolatev2(
                        p_sorted, a_sorted - a_ref, a_sorted, tolerance=a_tolerance
                        )
    a_best = y + a_ref
    print("use {n_interp} data points, found {p_best=} {a_best=} {p_flag=}", n_interp, p_best, a_best, p_flag)
    #logger.debug(
    #f"use {n_interp} data points, found {p_best=} {a_best=} {p_flag=}",
    #)
    return locals()


#@staticmethod
def find_a_extrapolatev2(x, y, a_sorted, tolerance=1e-5): # find_first_zerocrossing1d
    # no crossing and all positive, return the first one
    if y[0] >= 0:
        return x[0], y[0], 1
    for i in range(len(y) - 1):
        y0 = y[i] #, y1 = y[i + 1]
        x0 = x[i] #, x1 = x[i + 1]
        if np.isclose(y0, 0, atol=tolerance):
            return x0, y0, 0
        dec_indices = np.where(np.diff(a_sorted) < 0.0)[0] + 1 # find indices that break monotonically increasing pattern and remove
        y_new = np.delete(y,dec_indices)
        x_new = np.delete(x,dec_indices)
        y0, y1 = y_new[i], y_new[i + 1]
        x0, x1 = x_new[i], x_new[i + 1]
        #dx = x1 - x0
        #dy = y1 - y0
        #return x0 - y0 * dx / dy, 0.0, 0
        if y0 < 0 and y1 >= 0: # do anything with numpy or scipy? # want to keep this line
            # interpolate
            y_gen = np.linspace(0.01,0.77) # 0.77
            y_interp = np.interp([0.2], y_new, x_new) # swapped x and y here
            #y_interp = np.interp(y_gen, y_new, x_new) # swapped x and y here
            #dx = x1 - x0
            #dy = y1 - y0
            return y_interp, 0.0, 0 #(0.2) #a_ref #x0 - y0 * dx / dy, 0.0, 0
    else:
        # no crossing and all negative, return the last one
        return x[-1], y[-1], 1


#@staticmethod
def find_first_zerocrossing1d(x, y, tolerance=1e-5):
    # no crossing and all positive, return the first one
    if y[0] >= 0:
        return x[0], y[0], 1
    for i in range(len(y) - 1):
        y0, y1 = y[i], y[i + 1]
        x0, x1 = x[i], x[i + 1]
        if np.isclose(y0, 0, atol=tolerance):
                return x0, y0, 0
        if y0 < 0 and y1 >= 0:
                # interpolate
                dx = x1 - x0
                dy = y1 - y0
                return x0 - y0 * dx / dy, 0.0, 0
    else:
        # no crossing and all negative, return the last one
        return x[-1], y[-1], 1



def _get_tone_amps(nc):
    if "Header.Toltec.ToneAmp" in nc.variables:
        ## new change 20240316
        toneAmps = nc.variables["Header.Toltec.ToneAmp"][:].data.T[:, 0]
    else:
        toneAmps = nc.variables["Header.Toltec.ToneAmps"][:].data
    return toneAmps


def magS21(i,q):
    return np.sqrt(i**2+q**2) # similar to np.abs

    
def logmag(z): # similar to magS21_dB(i,q)
    return 20.0*np.log10(np.abs(z)) # np.sqrt(z.real**2+z.imag**2)


def complexS21(i,q):
    return i + 1.j*q # I + jQ


def phase2(z): # have another phase function below
    x = np.angle(z)
    x_cp = np.copy(x)
    y = np.unwrap(x_cp)
    return y


def get_dip_depth(s21): # from Yuhan # using target sweeps would be best I think
    """
    Input an array of s21, find the
    depth between maximal and minimal point.
    """
    s21_mag = np.abs(s21)
    return 20. * np.log10(max(s21_mag)) - 20. * np.log10(min(s21_mag))


def findcabledelay(f,z):
    '''
    f : takes np array in Hz
    z : tales np array in complex s21

    return tau in ns
    '''
    
    # assumes f in Hz, identical to MATLAB code
    f = f / 1e9
    
    theta = phase2(z) # includes phase unwrapping
    grad = np.gradient(theta)/np.gradient(f)/2./np.pi
    
    # fig = plt.figure()
    # ax = plt.subplot(211)
    # plt.plot(f, theta)
    # plt.xlabel('f [GHz]')
    # plt.ylabel('Phase')
    
    # ax = plt.subplot(212)
    # plt.plot(f, grad)
    tau = np.mean(grad)
    #plt.plot(f, tau,'r--') # horizontal line
    #plt.axhline(y=tau, color='r', linestyle='--')
    # plt.xlabel('f [GHz]')
    # plt.ylabel('tau [ns]')
    # plt.title('tau = ' + str(tau))
    # fig.tight_layout()
    # plt.show()
    # print(tau)
    
    return tau/1e9


def fit_cable_delay(gain_f,gain_z):
    # gain f in Hz
    gain_phase = phaseS21(gain_z.real,gain_z.imag)
    gain_phase_cp = np.copy(gain_phase)
    gain_phase = np.unwrap(gain_phase) # unwrap phase with numpy
    
    p_phase = np.polyfit(gain_f,gain_phase,1)
    tau = p_phase[0]/(2.*np.pi)
    poly_func_phase = np.poly1d(p_phase)
    fit_data_phase = poly_func_phase(gain_f)

    return tau*1e9,fit_data_phase,gain_phase

