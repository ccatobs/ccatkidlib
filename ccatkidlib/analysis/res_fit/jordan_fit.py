import matplotlib.pyplot as plt
import sys
# sys.path.append('/home/pcs/')
import numpy as np
import yaml
import scipy
import multiprocessing
import time

from ccatkidlib.analysis.res_fit.jordan_utils.fitting import fit_nonlinear_iq_multi, fit_nonlinear_iq, fit_nonlinear_iq_ss

def multiprocess_fit_ss(fs=None, dets=None,targ_file=None, cfg_file=None, keep_model=False, verb=False, num_cores=40):
    '''
    Uses multiprocessing to make jordan's code run faster for a target sweep
    '''

    t = time.time()
    
    if fs is None and dets is None:
        fs, s21 = np.load(targ_file)
        fs = fs.real
        with open(cfg_file, 'r') as f:
                config = yaml.safe_load(f)
    
        bin_width = config['tones']['N_step']
            
        dets = np.reshape(s21, (s21.shape[0]//bin_width,bin_width))
        fs = np.reshape(fs, (fs.shape[0]//bin_width,bin_width))

    args = [(f,z) for i, (f,z) in enumerate(zip(fs, dets))]

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(fit_single_res_ss, args)

    print(time.time()-t)

    return {key: np.array([d[key] for d in results]) for key in results[0]}

    
    

def fit_single_res_ss(f, z, keep_model=False):
    ret = {}

    model = {
         'fs': [],
         'data_z': []
    }
    flags = []
    
    if keep_model:
        # ret['fit_fs'] = []
        model['fit_z'] = []

    try:
        fit = fit_nonlinear_iq_ss(f,z, verbose=False)

        ret['Qc'] = fit.result.Qc
        ret['Q'] = fit.result.Qr
        ret['Qi'] = fit.result.Qi
        ret['f0'] = fit.result.fr
        ret['a'] = fit.result.a
        ret['tau'] = fit.result.tau
        ret['chi_sq'] = fit.result.chi_sq
        
        if keep_model:
            ret['fit_z'] = fit.z_fit()

        ret['fs'] = f
        ret['data_z'] = z
        ret['flag'] = 0

    except RuntimeError:
        ret['flag'] = 3

        ret['Qc'] = None
        ret['Q'] = None
        ret['Qi'] = None
        ret['f0'] = None
        ret['a'] = None
        ret['tau'] = None
        ret['chi_sq'] = None
        
        if keep_model:
            ret['fit_z'] = z.mean()*np.ones(shape=z.shape)

        ret['fs'] = f
        ret['data_z'] = z

    return ret
    

    
    
    

def fit_target_sweep_ss(targ_file, cfg_file, verb=True, keep_model=False):

    fs, s21 = np.load(targ_file)
    fs = fs.real
    with open(cfg_file, 'r') as f:
            config = yaml.safe_load(f)

    bin_width = config['tones']['N_step']
        
    dets = np.reshape(s21, (s21.shape[0]//bin_width,bin_width))
    fs = np.reshape(fs, (fs.shape[0]//bin_width,bin_width))

    ret = {
        'Qr': [],
        'Qi': [],
        'Qc': [],
        'f0': [],
        'a': [],
        'tau': [],
        'chi_sq': [],  
    }

    model = {
         'fs': [],
         'data_z': []
    }
    flags = []

    if keep_model:
        # ret['fit_fs'] = []
        model['fit_z'] = []
         
    # fits = fit_nonlinear_iq_multi(fs, dets, verbose=False)

    for i, (f, z) in enumerate(zip(fs, dets)):

        try:

            fit = fit_nonlinear_iq_ss(f,z, verbose=False)
            for k in ret.keys():
                 exec(f"ret['{k}'].append(fit.result.{k})")
            
            if keep_model:
                model['fit_z'].append(fit.z_fit())
    
            model['fs'].append(f)
            model['data_z'].append(z)
            flags.append(0)

        except RuntimeError:
            print(f'RuntimeError for res {i}')
            flags.append(3)

            for k in ret.keys():
                 exec(f"ret['{k}'].append(None)")
            
            if keep_model:
                model['fit_z'].append(z.mean()*np.ones(shape=z.shape))
    
            model['fs'].append(f)
            model['data_z'].append(z)
            

    # for i, result in enumerate(fits):
    #     if i >= len(fs): continue
        

    #     if keep_model:
    #         # ret['fs'].append(fits._fit_results[result].f_data)
    #         model['fit_z'].append(fits._fit_results[result].z_fit())
    #     model['fs'].append(fss[i])
    #     model['data_z'].append(detss[i])

    
    for k in model.keys():
        ret[k] = model[k]
    
    ret['Q'] = ret.pop('Qr')
    ret['alpha'] = ret.pop('a')
    ret['flag'] = np.array(flags)

    for k in ret.keys():
        ret[k] = np.array(ret[k])

    return ret
    
    

def fit_target_sweep(targ_file, cfg_file, verb=True, keep_model=False):

    fs, s21 = np.load(targ_file)
    fs = fs.real
    with open(cfg_file, 'r') as f:
            config = yaml.safe_load(f)

    bin_width = config['tones']['N_step']
        
    dets = np.reshape(s21, (s21.shape[0]//bin_width,bin_width))
    fs = np.reshape(fs, (fs.shape[0]//bin_width,bin_width))

    ret = {
        'Qr': [],
        'Qi': [],
        'Qc': [],
        'f0': [],
        'a': [],
        'tau': [],
        'chi_sq': [],  
    }

    model = {
         'fs': [],
         'data_z': []
    }

    flags = []
    
    if keep_model:
        # ret['fit_fs'] = []
        model['fit_z'] = []
         
    # fits = fit_nonlinear_iq_multi(fs, dets, verbose=False)

    for i, (f, z) in enumerate(zip(fs, dets)):

        fit = fit_nonlinear_iq(f,z, verbose=False)
        for k in ret.keys():
             exec(f"ret['{k}'].append(fit.result.{k})")
        
        if keep_model:
            model['fit_z'].append(fit.z_fit())

        model['fs'].append(f)
        model['data_z'].append(z)
        flags.append(0)

    # for i, result in enumerate(fits):
    #     if i >= len(fs): continue
        

    #     if keep_model:
    #         # ret['fs'].append(fits._fit_results[result].f_data)
    #         model['fit_z'].append(fits._fit_results[result].z_fit())
    #     model['fs'].append(fss[i])
    #     model['data_z'].append(detss[i])

    
    for k in model.keys():
        ret[k] = model[k]
    
    ret['Q'] = ret.pop('Qr')
    ret['alpha'] = ret.pop('a')
    ret['flag'] = flags

    for k in ret.keys():
        ret[k] = np.array(ret[k])

    return ret
          

if __name__ == "__main__":
    target_file = "/mnt/md0/cooldown_dec/test/targ/20250201/1738368042/B1D1/att_to_tone_targ_1738368500.npy"
    target_cfg = "/mnt/md0/cooldown_dec/test/rfsoc/20250201/1738368042/B1D1/targ_config_drone_1738368500.yaml"