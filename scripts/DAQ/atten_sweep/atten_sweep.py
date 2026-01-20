from ccatkidlib.rfsoc.rfsoc_daq import R
import numpy as np
from tqdm import tqdm

def atten_sweep(R):
    R.set_atten(drive=12)
    R.find_detectors()

    drive_attens = np.arange(20, -0.5, -0.5)        
    for drive_atten in tqdm(drive_attens, desc = 'Drive Attenuations'):
        R.set_atten(drive=drive_atten)
        R.tune_tone_placement()
        R.take_target_sweep()
        R.take_timestream(10)

if __name__ == '__main__':
    RC = R()
    atten_sweep(RC)