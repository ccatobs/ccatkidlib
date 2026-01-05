import numpy as np
import scipy 

from lmfit import Model, Parameters
from scipy.optimize import least_squares, root_scalar
from scipy.interpolate import make_interp_spline, PchipInterpolator
from numba import njit, guvectorize, float64, prange

#=================#
# General Fitting #
#=================#

@njit(parallel=True, cache=True)
def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    '''Calculate 2D linear regression for given x and y np.arrays by solving matrix equation

    Args:
        x (np.ndarray): Array of independent variable data
        y (np.ndarray): Array of dependent variable data
    Returns:
        tuple[int, int]: Slope, intercept
    '''
    A = np.ones((2, x.size))
    A[0] = x

    m, b = np.linalg.solve(A @ A.T, A @ y.T)
    return m, b

def circle_fit(x: np.ndarray, y: np.ndarray, bounds: tuple[list[float], list[float]] = None, full_output: bool = False, loss: str = 'soft_l1', f_scale: float = 1, method: str = 'trf'):
    @njit(cache=True)
    def _res_jac(A: float, D: float, theta: float, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''Calculates residuals and Jacobian for fitting circles. 

        Notes:
            Objective function from Section 3.2 of https://doi.org/10.48550/arXiv.cs/0301001 

        Args:
            A      (float): Fitting parameter
            D      (float): Fitting parametr
            theta  (float): Fitting parameter
            x (np.ndarray): Array of measured x-axis data (I for complex transmission)
            y (np.ndarray): Array of measured y-axis data (Q for complex transmission)
        Returns:
            result (tuple[np.ndarray, np.ndarray]): Residuals array, Jacobian array
        '''
        sin = np.sin(theta)
        cos = np.sqrt(1 - sin**2)

        z = x**2 + y**2
        E = np.sqrt(1 + 4*A*D)
        u = x*cos + y*sin

        P = A*z + E*u + D
        Q = np.sqrt(1 + 4*A*P)

        d = 2*P/Q
        R = 2*(1 - A*d/Q)/(Q + 1)

        d_dA = (z + 2*D*u/E)*R - d**2 / Q
        d_dD = (2*A*u/E + 1)*R
        d_dtheta = (-x*sin + y*cos)*E*R

        jacobian = np.vstack((d_dA, d_dD, d_dtheta)).T

        return d, jacobian

    def _residuals(params, x, y):
        A, D, theta = params
        residuals, jacobian = _res_jac(A, D, theta, x, y)
        return residuals

    def _jacobian(params, x, y):
        A, D, theta = params
        residuals, jacobian = _res_jac(A, D, theta, x, y)
        return jacobian    

    def _cline_to_circ(A, B, C):
        x = -B/(2*A)
        y = -C/(2*A)
        R = 1/(2*np.abs(A))
        return x, y, R

    def _circ_to_cline(x, y, R):
        A = 1/(2*R)
        B = -2*A*x
        C = -2*A*y
        D = (B**2 + C**2 - 1)/(4*A)
        return A, B, C, D

    # Filter out NaNs from input arrays
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    # Shift the circle if too close to the origin
    mean_x, mean_y = np.mean(x), np.mean(y)
    center_mag = np.sqrt(mean_x**2 + mean_y**2)
    shift_x, shift_y = 0, 0
    if center_mag < 5: shift_x, shift_y = 50/center_mag, 50/center_mag
    x += shift_x
    y += shift_y

    init_x, init_y, init_R = init_circle_fit(x, y)
    init_A, init_B, init_C, init_D = _circ_to_cline(init_x, init_y, init_R)

    if bounds is None:
        bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    elif not bounds[0][0] < A < bounds[1][0]:
        left, right = bounds
        bounds = (-1*np.array(right), -1*np.array(left))

    # TODO: Improve fit implementation to not require two loops (or use a different method)
    for _ in range(2):
        x0 = [np.real(init_A), np.real(init_D), np.real(np.arctan(init_C/init_B))]
        result = least_squares(_residuals, x0 = x0, jac=_jacobian, args=(x, y), bounds = bounds, loss = loss, f_scale=f_scale, method=method)
        A, D, theta = result.x

        E = np.sqrt(1 + 4*A*D)
        B, C = E*np.cos(theta), E*np.sin(theta)

        x_c, y_c, R = _cline_to_circ(A, B, C)

        # The fit will fail if the center has to pass near the origin since there is a singularity there.
        # Whether the fit needs to do this is determined by the sign of A (the sign of all other variables are defined relative to A)
        # If the fit fails (will return a very small radius), we refit with the sign of our initial guesses flipped
        # https://doi.org/10.48550/arXiv.cs/0301001 have another method that involves shifting the data if the fit tries to go through the origin 
        # but this is difficult without implementing solver from scratch (could maybe be done through a callback function to least_squares)
        if 1.1*center_mag > R > center_mag/100:
            break
        else:
            init_A *= -1
            init_B *= -1
            init_C *= -1
            init_D *= -1        

    if full_output:
        return x_c - shift_x, y_c - shift_y, R, result
    else:
        return x_c - shift_x, y_c - shift_y, R

def init_circle_fit(x, y):
    @njit(cache=True)
    def _moments(x, y):
        z = x**2 + y**2

        M = np.zeros((4, 4))

        data = [z, x, y, np.ones(len(x))]

        for i in prange(4):
            for j in prange(i, 4):
                dot = np.dot(data[i], data[j])
                M[i][j] = dot
                if not i == j: M[j][i] = dot
        return M

    M = _moments(x, y)
    B_inv = np.array([[ 0, 0, 0, -0.5],
                      [ 0, 1, 0,  0],
                      [ 0, 0, 1,  0],
                      [-0.5, 0, 0,  0]])

    eigvals, eigvecs = np.linalg.eig(B_inv @ M)
    positive_eig = eigvals >= 0

    min_eig = np.argmin(eigvals[positive_eig])
    A_vec = eigvecs[:, positive_eig][:, min_eig]

    A, B, C, D = A_vec

    # Need to normalize eigenvector to satisfy B^2 + C^2 - 4AD = 1
    norm = 1/np.sqrt(B**2 + C**2 - 4*A*D)
    A, B, C, D = A*norm, B*norm, C*norm, D*norm

    x = -B/(2*A)
    y = -C/(2*A)
    R = 1/(2*np.abs(A))
    return x, y, R

#===============#
# Interpolation #
#===============#

def y_to_x_spline(x: np.ndarray, y: np.ndarray, k: int = 3, y_low = None, y_up = None) -> tuple[None | scipy.interpolate.BSpline, scipy.interpolate.BSpline | None]:
    ''' 
    Args:
        x (np.ndarray): Array of independent variables
        y (np.ndarray): Array of dependent variables
        k (int): Degree of polynomials to interpolate with. Defaults to degree 3.
    Returns:
        tuple[None | scipy.interpolate.BSpline, scipy.interpolate.BSpline | None]: y to x BSpline, x to y BSpline. Only returns one BSpline, the other will be None
    '''

    if y_low is not None: 
        mask = np.where(y > y_low)
        y = y[mask]
        x = x[mask]
    if y_up is not None:
        mask = np.where(y < y_up)
        y = y[mask]
        x = x[mask]

    # Creating a spline requires a monotonically increasing array, reverse if  decreasing
    # Save to new variables so that original x and y can be used for x to y interpolation if y to x fails
    if y[-1] - y[0] < 0:
        y_new = y[::-1]
        x_new = x[::-1]
    else:
        y_new = y
        x_new = x

    # Try to interpolate y to x, but interpolate x to y if that fails (e.g., if y is not monotonically increasing)
    try:
        y_to_x_spline = make_interp_spline(y_new, x_new, k=k)
        return y_to_x_spline, None
    except ValueError:
        x_to_y_spline = make_interp_spline(x, y, k=k)
        return None, x_to_y_spline

def y_to_x_interp(ys, y_to_x_spline=None, x_to_y_spline=None):
    if y_to_x_spline is not None:
        xs = y_to_x_spline(ys)
    elif x_to_y_spline is not None:
        min_x, max_x = x_to_y_spline.t[0], x_to_y_spline.t[-1]
        xs = [None]*len(ys)
        for i, y in enumerate(ys):
            try:
                xs[i] = root_scalar(lambda x: x_to_y_spline(x) - y, bracket = [min_x, max_x]).root
            except:
                pass
    else:
        xs = np.zeros(len(ys))
    return xs

#===================#
# Resonator Fitting #
#===================#

def phase_fit(f: np.ndarray,
              phase: np.ndarray,
              I: np.ndarray | None = None,
              Q: np.ndarray | None = None,
              R: float | None = None,
              window: float | None = None, 
              unwrap_threshold = 1.9*np.pi,
              params: Parameters | None = None,
              nonlinear: bool = False,
              method: str = 'least_squares'):
    ''' Method for fitting to the phase of a kinetic inductance detector (KID)

    Notes:
        Using method from https://doi.org/10.1117/12.3019161. 

    Args:
        f (np.ndarray): Array of frequencies (in Hz)
        phase (np.ndarray): Array of phases (in rad)
        I (np.ndarray, optional): Array of in-phase components of complex transmission. Required for nonlinear fit
        Q (np.ndarray, optional): Array of quadrature components of complex transmission. Required for nonlinear fit
        params (lmfit.Parameters, optional): lmfit Parameters object to use as initial guess for fit
        R (float, optional): Radius of the IQ circle. Only required for nonlinear fit and if not passed as a lmfit.Parameter in params
        nonlinear (bool, optional): Whether to use nonlinear fit. Defaults to False
    '''
    # TODO: Should switch out lmfit for curve_fit for faster fitting and don't really need lmfit features, Also don't want data trimming in this function since can just use IQ_trim instead

    @njit(cache=True)
    def _guess_params(f, phase):
        ninety_phase_ind = np.argmin(np.abs(np.abs(phase) - np.pi/2)) # Get the index at which the phase is closest to 90 degrees 
        
        f_0_guess = (f[zero_phase_ind_up] + f[zero_phase_ind_low])/2  # Guess resonant frequency to be near zero phase
        f_0_min, f_0_max = f[0], f[-1] # Bound resonant frequency to be within sampled frequencies
        
        FWHM_guess = 2*np.abs(f[ninety_phase_ind] - f_0_guess)
        FWHM_min, FWHM_max = f[2] - f[0], 5*(f[-1] - f[0])

        Qr_min, Qr_max = f_0_guess/FWHM_max, f_0_guess/FWHM_min
        Qr_guess = f_0_guess/FWHM_guess if FWHM_guess != 0 else np.sqrt(Qr_min*Qr_max)

        # Create param tuples in form (name, value, vary, min, max)
        f_0_param = ('f_0', f_0_guess, True, f_0_min, f_0_max)
        Qr_param =  ('Qr',  Qr_guess,  True, Qr_min,  Qr_max)
        theta_0_param = ('theta_0', 0, True, -np.pi, np.pi)
        beta_param = ('beta', 0, True, -1e4, 1e4)

        return f_0_param, Qr_param, theta_0_param, beta_param

    @njit(cache=True)
    def _phase_fit(f, I, Q, f_0, Qr, theta_0, beta, R):
        ''' Calculates phases for the given frequencies for a nonlinear KID
        Args:
            f (np.ndarray): Array of frequencies (in Hz)
            I (np.ndarray): Array of in-phase components of complex transmission
            Q (np.ndarray): Array of quadrature components of complex transmission
            f_0 (float): KID resonant frequency
            Q (float): KID total quality factor
            theta_0 (float): KID phase angle
            beta (float): Factor related to KID kinetic inductance nonlinearity. Use a = 0 for a linear resonator.
            R (float): Radius of KID IQ circle. 

        Returns:
            np.ndarray: Array of calculated phases for the given frequencies
        '''

        f_0 = np.ones(f.size)*f_0
        
        # Calculate nonlinear correction only if a nonzero beta is provided
        if not beta == 0: 
            # Calculate off resonance I and Q (I_off, Q_off)
            sin = np.sin(theta_0 + np.pi)
            cos = np.sqrt(1 - sin**2)
            I_off, Q_off = R*cos, R*sin

            # Calculate distance between arbitrary (I, Q) point and off resonance point (I_off, Q_off)
            dist = (I - I_off)**2 + (Q - Q_off)**2

            # Shift resonant frequency using nonlinear term
            beta *= 1e-8
            f_0 = f_0 - beta*dist

        x = (f - f_0)/f_0
        return theta_0 + 2*np.arctan(-2*Qr*x)

    # Unwrap phase
    # ------------
    zero_phase_ind_up, zero_phase_ind_low = np.argmin(phase[phase >= 0]), np.argmax(phase[phase <= 0])
    ref_phase = phase[zero_phase_ind_up]
    
    curr_shift = 0
    for i in range(len(phase) - 1):
        diff = phase[i+1] - phase[i]
        phase[i] -= curr_shift
        if diff > unwrap_threshold:
            curr_shift += 2*np.pi 
        elif diff < -1*unwrap_threshold:
            curr_shift -= 2*np.pi
    phase[-1] -= curr_shift
    phase += ref_phase - phase[zero_phase_ind_up]
    
    
    # Create/validate params for model
    # --------------------------------
    if params is None: params = Parameters()
    if nonlinear:
        if I is None: raise ValueError(f'I is a required argument for nonlinear fits.')
        if Q is None: raise ValueError(f'Q is a required argument for nonlinear fits.')
        if (R is None) and (not 'R' in params): raise ValueError(f'R must be passed as an argument or through params for a nonlinear fit.')
    
    in_params = [param in params for param in ['f_0', 'Qr', 'theta_0', 'beta']]
    if not all(in_params):
        guess_params = _guess_params(f, phase)

        for incl, guess in zip(in_params, guess_params):
            if not incl: params.add(*guess)
    if R is not None: params.add(*('R', R, False))
    
    # Create lmfit model for phase fit
    # --------------------------------
    phase_model = Model(_phase_fit, independent_vars=['f', 'I', 'Q'])

    beta_guess = params['beta'].value
    params['beta'].set(value=0, vary=False)

    if window is not None:
        f_0, Qr = params['f_0'].value, params['Qr'].value
        f_win = 0.5*window*(f_0/Qr)    
        mask = (f < (f_0 + f_win)) & (f > (f_0 - f_win))

    fit = phase_model.fit(phase[mask], params=params, nan_policy='omit', f=f[mask], I=I[mask], Q=Q[mask])
    params = fit.params

    if window is not None:
        f_0, Qr = params['f_0'].value, params['Qr'].value
        f_win = 0.5*window*(f_0/Qr)    
        mask = (f < (f_0 + f_win)) & (f > (f_0 - f_win))

    if nonlinear:
        if beta_guess == 0: beta_guess = 0.1e8*(params['f_0'].value / params['Qr'].value)/(2*params['R'].value)**2
        params['beta'].set(value=beta_guess, vary=True)
    
    fit = phase_model.fit(phase[mask], params=params, nan_policy='omit', f=f[mask], I=I[mask], Q=Q[mask], method=method)
    fit.mask = mask

    # Fit model 
    # ---------
    return fit