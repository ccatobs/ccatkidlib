
import numpy as np
import matplotlib.pyplot as plt

import yaml
from ccatkidlib.analysis.res_fit.phase_utils.utils import *

from ccatkidlib.analysis.res_fit.phase_utils.fit_single_det import ResonanceFitterSingleTone as fit



def fit_target_sweep(targ_file=None, cfg_file=None, filt=False, verb=False, plot=False):
    '''
    fits target sweep using data from targ_file. cfg_file is the config of that target sweep.
    returns a dictionary of numpy arrays for fitted values. Also filters some of the
    nonphysical fits if select filt=True
    
    inputs
    
        targ_file : string    The target sweep data file
        cfg_file:   string    The target sweep config file
        filt:       bool      whether to filter the fits before returning
        verb:       bool      whether to output with more verbosity


    returns
    
        ret = {
            'Qi':    np.array,
            'Qc':    np.array,
            'Q':     np.array,
            'bif':   np.array,
            'alpha': np.array,
            'tau':   np.array,
            'f0':    np.array,
        }
    '''
    
    fs, s21 = np.load(targ_file)
    fs = fs.real
    with open(cfg_file, 'r') as f:
            config = yaml.safe_load(f)

    bin_width = config['tones']['N_step']
        
    dets = np.reshape(s21, (s21.shape[0]//bin_width,bin_width))
    fs = np.reshape(fs, (fs.shape[0]//bin_width,bin_width))
    ret = {
        'Qi': [],
        'Qc':[],
        'Q':[],
        'bif':[],
        'alpha':[],
        'tau':[],
        'f0':[],
    }
    
    for i, (f, d) in enumerate(zip(fs, dets)):
        try:
            tau = findcabledelay(f/1e9, d)
            result0 = fit(f, d, tau).result
            nonlinear_result = fit(f, d, tau, result0=result0).result

            for k in ret.keys():
                ret[k].append(nonlinear_result[k])
        except:
            pass

    for k in ret.keys():
        ret[k] = np.array(ret[k])
    
    if filt:
        passed = np.argwhere(ret['Q']>10)

        for k in ret.keys():
            ret[k] = ret[k][passed]
        

        # for i, q in enumerate(ret['alpha']):
        #         if q < 0:  
        #             for k in ret.keys(): ret[k].pop(i)
            

    if verb: print(f"Successful Fit of {len(ret['Qi'])} / {len(dets)} detectors")
    if plot:
        plt.hist(ret['Qi'], np.arange(0,5e5, 2e3))
        plt.show()
    
    return ret