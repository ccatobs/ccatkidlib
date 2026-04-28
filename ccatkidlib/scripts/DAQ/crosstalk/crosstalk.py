from ocs.ocs_client import OCSClient
from tqdm import tqdm
import numpy as np
import time
import sys
import yaml

#Quick .

def main():
    # Local imports
    sys.path.append('./../../rfsoc/') # Append path with RFSoC_DAQ.py
    
    from rfsoc_daq import R

    #Load configs
    cfg_list = ["./coldload_system_config_crosstalk_1.yaml", "./coldload_system_config_crosstalk_2.yaml",
                "./coldload_system_config_crosstalk_3.yaml"]
    # Intialize RFSoC control object
    #R = R(cfg_path = "./coldload_system_config.yaml")

    # Currents and attenuations to sweep over
    drive_attens = [0,6,12]# DAC attenuations
    sense_attens = [0] # ADC attenuations
    attens = (drive_attens, sense_attens)

    # Run coldload sweep
    crosstalk_sweep(cfg_list, attens, R)

def crosstalk_sweep(cfg_list, attens, R):
    '''
    Load a config take a few vna sweeps. Get 

    Parameters:
        cfg_list: cfgs to create RFSoC data acquisition object
        attens: List of attenuations to iterate over
    '''
    # Initalize power supply PCS agent
    psu = OCSClient('psuBKP9130B', args = [])

    # Iterate over configs and take two attenuation sweeps at each.
    # -----------------------------------------------------

    with tqdm(range(len(cfg_list)), desc = f'DAQ:') as pbar:
        for i in pbar:
            # Load a cfg
            Rx = R(cfg_path = cfg_list[i]) 

            # Edit external config file
            get_updated_psu_stats(psu)
            pbar.set_postfix_str(f'Current drone: {i + 1}, Output: {(i+1)}')
            #time.sleep(90)
            # Take attenuation sweep
            # ----------------------
            atten_sweep(Rx, attens)
            #get_updated_psu_stats(psu)
            #pbar.set_postfix_str(f'Current drone: {i + 1}, Output: {(i+2)%3 + 1}')
            #time.sleep(90)
            #atten_sweep(Rx, attens)

def atten_sweep(R, attens):
    '''
    Sweep over different drive (DAC) and sense (ADC) attenuations and take sweeps and (optionally) timestreams at each. 

    Parameters:
        attens: A tuple of drive and sense attenuation lists
    '''


    # Iterate over attenuations and take sweep + timestream data at each attenuation
    # -----------------------------------------------------
    with tqdm(range(len(attens[0])), desc = f'DAQ:') as drive_pbar:
        for j in drive_pbar:
            with tqdm(range(len(attens[1])), desc = f'DAQ:') as sense_pbar:
                for k in sense_pbar:
                    # Set drive and sense attenuations
                    # --------------------------------
                    R.set_atten(drive = attens[0][j], sense = attens[1][k])
                    drive_pbar.set_postfix_str(f'Current Drive Attenuation: {attens[0][j]}')
                    sense_pbar.set_postfix_str(f'Current Sense Attenuation: {attens[1][k]}')
                    
                    # Run data acquisition commands
                    # -----------------------------

                    # Take VNA sweep
                    if j==0:
                        #If its the first sweep, generate a new vna comb, but otherwise just repeat the sweep process.
                        R.take_vna_sweep()
                    else:
                        R.take_vna_sweep(write_tones = False)
 
def get_updated_psu_stats(psu):
    """
    Grab the actual current and voltage at the psu and return them for logging
    """
    status_psu, message_psu, session_psu = psu.monitor_output.start()
    time.sleep(3)
    logged = False
    while not logged:
        try:
            status_psu, message_psu, session_psu = psu.monitor_output.status()
            current = session_psu['data']['data']['Current_1']
            voltage = session_psu['data']['data']['Voltage_1']
            logged = True
        except:
            time.sleep(0.2)        
    psu.monitor_output.stop()
    return current, voltage
    

if __name__ == "__main__":
    main()
