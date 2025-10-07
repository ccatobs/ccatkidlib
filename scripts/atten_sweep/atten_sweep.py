from ccatkidlib.rfsoc.rfsoc_daq import R
import numpy as np
from tqdm import tqdm


def atten_sweep(R):
    com_to, boards = R._get_com_to()
    center_drive = R._get_drone_args(com_to, ['atten', 'drive'])
    center_sense = R._get_drone_args(com_to, ['atten', 'sense'])

    drive_offsets = np.arange(-9, 10, 1)
    sense_offsets = np.arange(-9, 12, 3)

    drive_attens = np.array([drive_offsets + atten for atten in center_drive]).T
    sense_attens = np.array([sense_offsets + atten for atten in center_sense]).T

    write_comb = True
    for drive_atten in tqdm(drive_attens, desc = 'Drive Attenuations'):
        for sense_atten in tqdm(sense_attens, desc = 'Sense Attenuations'):

            # Take VNA sweeps with different drive & sense attenuations 
            # ---------------------------------------------------------
            R.set_atten(drive=drive_atten, sense=sense_atten)
            R.take_vna_sweep(write_comb=write_comb) # Take VNA sweep with no attenuation
            write_comb = False

if __name__ == '__main__':
    RC = R(cfg_path='/home/pcs/ccatkidlib/scripts/atten_sweep/system_config.yaml', initialize_boards=True, initialize_drones=True)
    atten_sweep(RC)
    RC.close()