# -*- coding: utf-8 -*-
'''
Created on Thu Feb 21 10:10:49 2019

@author: Heather McCarrick

Given a complex s21 sweep, returns the data fit
to the resonator model and the resonator parameters 

Based on equation 11 from Kahlil et al. and adapted from
Columbia KIDs open source analysis code


Updated by Cody Duell 2019
And again in 2022

The nonlinear stuff came Nick Cothard

'''

import numpy as np
from lmfit import Model

'''
first the parts of the fitting equation
'''
def linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag):
    Q_e = Q_e_real + 1j*Q_e_imag
    return (1 - (Q * Q_e**(-1) /(1 + 2j * Q * (f - f_0) / f_0) ) )

def asymm_resonator(f, f_0, Q, Q_e_real, Q_e_imag):
    #This is the same resonator shape that I've seen multiple places, 
    #Asymmetric fitting doesn't really work that well at the moment

    Q_e = Q_e_real + 1j*Q_e_imag
    Qc = np.abs(Q_e)
    amp = Q/Qc
    phi = np.angle(Q_e)
    yg =  Q * ((f - f_0) / f_0)
    return (1 - amp*((np.exp(-1j*phi)/(1 + 2j*yg)) +(1/2)*(np.exp(-1j*phi) -1)))

def nonlinear_resonator(f, f_0, Q, Q_e_real, Q_e_imag,a):
    Q_e = Q_e_real + 1j*Q_e_imag
    x0 = (f-f_0)/f_0
    y0 = Q*x0
    y = res_roots(y0,a)
    return (1 - (Q * Q_e**(-1) /(1 + 2j * y ) ) )

def cable_delay(f, delay, phi, f_min):
    return np.exp(1j * (-2 * np.pi * (f - f_min) * delay + phi))

def general_cable(f, delay, phi, f_min, A_mag, A_slope):
    phase_term =  cable_delay(f,delay,phi,f_min)
    magnitude_term = ((f-f_min)*A_slope + 1)* A_mag
    return magnitude_term*phase_term
    
def resonator_cable(f, f_0, Q, Q_e_real, Q_e_imag, delay, phi, f_min, A_mag, A_slope):
    #combine above functions into our full fitting functions
    resonator_term = linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag)
    cable_term = general_cable(f, delay, phi, f_min, A_mag, A_slope)
    return cable_term*resonator_term

def asymm_resonator_cable(f, f_0, Q, Q_e_real, Q_e_imag, delay, phi, f_min, A_mag, A_slope):
    #combine above functions into our full fitting functions
    resonator_term = asymm_resonator(f, f_0, Q, Q_e_real, Q_e_imag)
    cable_term = general_cable(f, delay, phi, f_min, A_mag, A_slope)
    return cable_term*resonator_term

def nonlinear_resonator_cable(f, f_0, Q, Q_e_real, Q_e_imag, a, delay, phi, f_min, A_mag, A_slope):
    #combine above functions into our full fitting functions
    resonator_term = nonlinear_resonator(f, f_0, Q, Q_e_real, Q_e_imag,a)
    cable_term = general_cable(f, delay, phi, f_min, A_mag, A_slope)
    return cable_term*resonator_term

'''
then our main functions

'''

def cable_fit(freqs, real, imag):
    #takes numpy arrays of freq, real and imag values
    
    #turn real and imag s21 into a single complex array
    s21_complex = np.vectorize(complex)(real, imag)
    
    #set our initial guesses 
    argmin_s21 = np.abs(s21_complex).argmin()
    fmin = freqs.min()
    A_slope, A_offset = np.polyfit(freqs - fmin, np.abs(s21_complex), 1)
    A_mag = A_offset
    A_mag_slope = A_slope / A_mag
    phi_slope, phi_offset = np.polyfit(freqs - fmin, np.unwrap(np.angle(s21_complex)), 1)
    delay = -phi_slope / (2 * np.pi)

    #make our model
    totalmodel = Model(general_cable)
    params = totalmodel.make_params(delay=delay, 
                                phi=phi_offset, 
                                f_min=fmin, 
                                A_mag=A_mag, 
                                A_slope=A_mag_slope)

    params['phi'].set(min=phi_offset-np.pi, max=phi_offset+np.pi)
    params['f_min'].set(vary=False)

    #fit it
    result = totalmodel.fit(s21_complex, params, f=freqs)
    return result

def full_fit(freqs, real, imag, nonlinear = False, asymm = False, fix_cable = False):
    #takes numpy arrays of freq, real and imag values
    
    #turn real and imag s21 into a single complex array
    s21_complex = np.vectorize(complex)(real, imag)
    
    #set our initial guesses 
    argmin_s21 = np.abs(s21_complex).argmin()
    fmin = freqs.min()
    fmax = freqs.max()
    f_0_guess = freqs[argmin_s21]
    Q_min = 0.1 * (f_0_guess / (fmax - fmin))  
    delta_f = np.diff(freqs)  
    min_delta_f = delta_f[delta_f > 0].min()
    Q_max = f_0_guess / min_delta_f  
    Q_guess = np.sqrt(Q_min * Q_max) 
    s21_min = np.abs(s21_complex[argmin_s21])
    s21_max = np.abs(s21_complex).max()
    Q_e_real_guess = Q_guess / (1 - s21_min / s21_max)
    a_guess = 0.1
    
    if fix_cable:
        #print("Fixing cable...")
        A_mag = fix_cable.params['A_mag'].value;
        A_mag_slope = fix_cable.params['A_slope'].value;
        phi_offset = fix_cable.params['phi'].value;
        delay = fix_cable.params['delay'].value;
        fmin = fix_cable.params['f_min'].value;
    else:
        A_slope, A_offset = np.polyfit(freqs - fmin, np.abs(s21_complex), 1)
        A_mag = A_offset
        A_mag_slope = A_slope / A_mag
        phi_slope, phi_offset = np.polyfit(freqs - fmin, np.unwrap(np.angle(s21_complex)), 1)
        delay = -phi_slope / (2 * np.pi)
    
    #make our model
    if nonlinear:
        #print("Using nonlinear model...")
        totalmodel = Model(nonlinear_resonator_cable, nan_policy = 'omit')
        params = totalmodel.make_params(f_0=f_0_guess,
                                Q=Q_guess,
                                Q_e_real=Q_e_real_guess,
                                Q_e_imag=0,
                                a = a_guess,
                                delay=delay,
                                phi=phi_offset,
                                f_min=fmin,
                                A_mag=A_mag,
                                A_slope=A_mag_slope)
        params['a'].set(min=0, max=10)
    elif asymm:
        #print("Using asymmetric model...")
        totalmodel = Model(asymm_resonator_cable)
        params = totalmodel.make_params(f_0=f_0_guess, 
                                Q=Q_guess, 
                                Q_e_real=Q_e_real_guess, 
                                Q_e_imag=0, 
                                delay=delay, 
                                phi=phi_offset, 
                                f_min=fmin, 
                                A_mag=A_mag, 
                                A_slope=A_mag_slope)
    else:
        totalmodel = Model(resonator_cable)
        params = totalmodel.make_params(f_0=f_0_guess, 
                                Q=Q_guess, 
                                Q_e_real=Q_e_real_guess, 
                                Q_e_imag=0, 
                                delay=delay, 
                                phi=phi_offset, 
                                f_min=fmin, 
                                A_mag=A_mag, 
                                A_slope=A_mag_slope)

    #set some bounds
    params['f_0'].set(min=freqs.min(), max=freqs.max())
    params['Q'].set(min=Q_min, max=Q_max)
    params['Q_e_real'].set(min=1, max=1e7)
    params['Q_e_imag'].set(min=-1e7, max=1e7)
    
    if fix_cable:
        params['A_mag'].set(vary=False)
        params['A_slope'].set(vary=False)
        params['delay'].set(vary=False)
        params['phi'].set(vary=False)
        params['f_min'].set(vary=False)
    else:
        params['phi'].set(min=phi_offset-np.pi, max=phi_offset+np.pi)
        params['f_min'].set(vary=False)
        #params['delay'].set(min=delay-delay_var, max=delay+delay_var)

    
    #fit it
    result = totalmodel.fit(s21_complex, params, f=freqs)
    return result


def fine_s21_model(freqs_fine, fit_params, asymm = False, cable = False):
    #use this after fitting the data to get a prettier model
    try:
        a = fit_params['a'].value
    except:
        a = False

    if cable:
        totalmodel = Model(general_cable)
    elif a:
       totalmodel = Model(nonlinear_resonator_cable, nan_policy = 'omit')
       #print("Using nonlinear resonator model")
    elif asymm:
        totalmodel = Model(asymm_resonator_cable)
    else:
        totalmodel = Model(resonator_cable)

    '''
    if a:
        totalmodel = Model(nonlinear_resonator_cable)
        print("Using nonlinear resonator model")
    '''
    
    params = totalmodel.make_params(**fit_params)
    fine_model = totalmodel.eval(params, f=freqs_fine)
    return fine_model

'''
some other functions you probably want that the fitting does not directly return
'''
def get_qi(Q, Q_e_real, Q_e_imag):
    Q_e = Q_e_real + 1j*Q_e_imag
    return (Q**-1 - np.real(Q_e**-1))**-1

def get_br(Q, f_0):
    return f_0*(2 * Q)**-1

def reduced_chi_squared(ydata, ymod, n_param=9, sd=None):
    #red chi squared in lmfit does not return something reasonable 
    #so here is a handwritten function
    #you want sd to be the complex error

    chisq = np.sum((np.real(ydata) - np.real(ymod))**2/((np.real(sd))**2)) + np.sum((np.imag(ydata) - np.imag(ymod))**2/((np.imag(sd))**2))
    nu=2*ydata.size-n_param     #multiply  the usual by 2 since complex
    red_chisq = chisq/nu 
    return chisq, red_chisq

def residuals(ydata, ymod):
    return ydata-ymod

'''
functions for nonlinear resonator model.
analytic root finding algorithm.
can probably be made faster with numba.
'''

def cuberoot_complexnp(z):
    if isinstance(z,complex):
        z = np.array([z])
    return np.array([abs(zz)**(1/3)*(zz/abs(zz)) if np.isreal(zz) else zz**(1/3) for zz in z])

def cardano_roots(coeffs):
    # Finding roots of a cubic of the form ax^3 + bx^2 + cx +d = 0.
    # Based on https://proofwiki.org/wiki/Cardano's_Formula
    # and https://stackoverflow.com/questions/39474254/cardanos-formula-not-working-with-numpy

    a,b,c,d = [x+0j for x in coeffs]
    # Define helpers Q, R, and Z
    Q = (3*a*c - b**2) / (9*a**2)
    R = (9*a*b*c - 27*a**2*d - 2*b**3) / (54*a**3)
    Z = b/(3*a)


    # Define discriminant. If D>0, one real root. If D<0, three real roots.
    D = Q**3 + R**2
    S = cuberoot_complexnp(R + D**0.5)
    T = cuberoot_complexnp(R - D**0.5)

    x0 = S+T - Z
    x1 = -(S+T)/2 + (S-T)*1j*3**0.5/2 - Z
    x2 = -(S+T)/2 - (S-T)*1j*3**0.5/2 - Z

    return np.array([x0,x1,x2]),D

def res_roots(yg,a):
    coeffs = [4,-4*yg,1,-yg-a]
    roots,D = cardano_roots(coeffs)
    ys0,ys1,ys2 = roots
    ysx = np.full_like(ys0,0)
    sel = np.where(D>=0)
    ysx[sel] = ys0[sel]
    sel = np.where(D<0)
    ysx[sel] = ys1[sel]
    return ysx.real
