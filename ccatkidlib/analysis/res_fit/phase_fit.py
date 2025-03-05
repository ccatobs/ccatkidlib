
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import yaml
from ccatkidlib.analysis.res_fit.phase_utils.utils import *

from ccatkidlib.analysis.res_fit.phase_utils.fit_single_det import ResonanceFitterSingleTone as fit

def fit_single_res(fs, z, **kwargs):

    tau = findcabledelay(fs/1e9, z)
    result0 = fit(fs, z, tau, **kwargs).result
    nonlinear_result = fit(fs, z, tau, result0=result0, **kwargs).result

    return nonlinear_result

# def fit_target_sweep_plot(targ_file, cfg_file, verb=False):

#     fits = fit_target_sweep(targ_file, cfg_file, verb, keep_model=True)

#     Qs = fits['Q']

#     fig, ax = plt.subplots()

def fit_target_sweep(targ_file=None, cfg_file=None, verb=False, span=1, keep_model=False,  **kwargs):
    '''
    fits target sweep using data from targ_file. cfg_file is the config of that target sweep.
    returns a dictionary of numpy arrays for fitted values. Also filters some of the
    nonphysical fits if select filt=True
    
    inputs
    
        targ_file:  string    The target sweep data file
        cfg_file:   string    The target sweep config file
        verb:       bool      whether to output with more verbosity

        kwargs:    {
            numspan: int
            tone_freq_lo: float
            window_width: float  
            pherr_threshold_num: float
            pherr_threshold: float
        }


    returns
    
        ret = {
            'Qi':     np.array,
            'Qc':     np.array,
            'Q':      np.array,
            'bif':    np.array,
            'alpha':  np.array,
            'tau':    np.array,
            'f0':     np.array,
            'chi_sq': np.array,
            'flag':   np.array, An array of flags corresponding to each fit
                               0 is good fit; 1 is bad fit; 2 is failed fit; 3 is code error
            'fs':     np.array, 
            'fit_z':  np.array, will return if keep_model is True
            'data_z': np.array,
        }

    '''
    
    fs, s21 = np.load(targ_file)
    fs = fs.real
    with open(cfg_file, 'r') as f:
            config = yaml.safe_load(f)

    bin_width = config['tones']['N_step']
        
    dets = np.reshape(s21, (s21.shape[0]//bin_width,bin_width))
    fs = np.reshape(fs, (fs.shape[0]//bin_width,bin_width))

    dets = dets[:, bin_width//2 - int(bin_width*span*0.5) : bin_width//2 + int(bin_width*span*0.5)]
    fs = fs[:, bin_width//2 - int(bin_width*span*0.5) : bin_width//2 + int(bin_width*span*0.5)]

    ret = {
        'Qi': [],
        'Qc':[],
        'Q':[],
        'bif':[],
        'alpha':[],
        'tau':[],
        'f0':[],
        'flag':[],
        'f0_corr':[]
    }

    models = {
        'fs': [],
        'data_z': [],
    }
    if keep_model: 
        models['fit_z'] = []
        # models['phase_fit'] = []
        models['ang'] = []


    
    for i, (f, z) in enumerate(zip(fs, dets)):
        try:
    
            # tau = findcabledelay(f/1e9, z)
            # result0 = fit(f, z, tau).result
            # nonlinear_result = fit(f, z, tau, result0=result0).result

            models['fs'].append(f)
            models['data_z'].append(z)

            nonlinear_result = fit_single_res(f,z,**kwargs)

            if keep_model: 
                models['fit_z'].append(nonlinear_result['ang_to_z'])
                # models['phase_fit'].append(nonlinear_result['fit_result'])
                models['ang'].append(nonlinear_result['ang'])


            for k in ret.keys():
                ret[k].append(nonlinear_result[k])

            
        except:
            if keep_model: 
                models['fit_z'].append(z.mean()*np.ones(shape=z.shape))
                # models['phase_fit'].append(np.ones(shape=z.shape))
                models['ang'].append(np.ones(shape=z.shape))

            for k in ret.keys():
                ret[k].append(None)
            ret['flag'][-1] = 3
            
    for k in models.keys():
        ret[k] = models[k]

    for k in ret.keys():
        # if k == 'phase_fit':print(k, ret[k])
        ret[k] = np.array(ret[k])
    
    ret['chi_sq'] = np.array(
         [
              np.sum((np.abs(fit) - np.abs(data))**2 / np.abs(fit)) for fit, data in zip(ret['fit_z'], ret['data_z'])
         ]
         )
    
    # print(len(np.argwhere(ret['flag'] == 2)))
    
    for i, (Q, Qi, Qc) in enumerate(zip(ret['Q'], ret['Qi'], ret['Qc'])):
        if Q is not None and (Q < 10 and Qi < 10 and Qc < 10): ret['flag'][i] = 2

    total = len(ret['Q']) 
    good = len(np.argwhere(ret['flag'] == 0))
    bad = len(np.argwhere(ret['flag'] == 1))
    fail = len(np.argwhere(ret['flag'] == 2))
    err = len(np.argwhere(ret['flag'] == 3))

        # for i, q in enumerate(ret['alpha']):
        #         if q < 0:  
        #             for k in ret.keys(): ret[k].pop(i)
            

    if verb: 
        print(f"Successful Fit of {good + bad} / {total} detectors")
        print(f"Good Fit of {good} / {total} detectors")
        print(f"Bad Fit of {bad} / {total} detectors")
        print(f"Failed Fit of {fail} / {total} detectors")
        print(f"Error Fit of {err} / {total} detectors")
    
    return ret

if __name__ == '__main__':

    from matplotlib.widgets import Button
    from ccatkidlib.analysis.res_fit.plot_utils import interactive_plot
    from ccatkidlib.analysis.res_fit.phase_utils.utils import removecable
    
    target_file = "/mnt/md0/cooldown_dec/targ_test/targ/20250123/1737659072/B1D3/test_targ_1737659458.npy"
    target_cfg = "/mnt/md0/cooldown_dec/targ_test/rfsoc/20250123/1737659072/B1D3/targ_config_drone_1737659458.yaml"


    fs, s21 = np.load(target_file)
    fs = fs.real
    span = 1
    with open(target_cfg, 'r') as f:
            config = yaml.safe_load(f)

    bin_width = config['tones']['N_step']
        
    dets = np.reshape(s21, (s21.shape[0]//bin_width,bin_width))
    fs = np.reshape(fs, (fs.shape[0]//bin_width,bin_width))

    data = fit_target_sweep(target_file, target_cfg, verb=True, pherr_threshold=0.2, pherr_threshold_num = 10)

    s21s = []

    for i, tau in enumerate(data['tau']):
        if tau is not None:
            s21s.append(removecable(fs[i], dets[i], tau))
        else: s21s.append(dets[i])

    s21s = np.array(s21s)

    print(data['tau'])
    
    indices = np.argwhere(data['flag'] == 0)
    
    interactive_plot(s21s[indices].real, s21s[indices].imag)
    # interactive_plot(fs[indices], np.abs(dets[indices]))

    
    
    
    
