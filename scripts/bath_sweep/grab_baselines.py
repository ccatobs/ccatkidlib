from ocs.ocs_client import OCSClient
from tqdm import tqdm
import numpy as np
import time
import sys
import yaml

#Watch the array temp and take some sweeps when they are above 700 mK.

def main():
    # Local imports
    sys.path.append('./../../rfsoc/') # Append path with RFSoC_DAQ.py
    
    from rfsoc_daq import R

    # Intialize RFSoC control object
    R = R(cfg_path = "./bath_sweep_system_config.yaml")

    drive_attens = [0,6]# DAC attenuations
    sense_attens = [0] # ADC attenuations

    attens = (drive_attens, sense_attens)

    low_T = .75 #Temp to start taking sweeps
    cutoff = 1.6 # Temp to stop
    timeout = 8*3600 # Time to stop

    # Run bath sweep
    bath_sweep(R, low_T, cutoff, attens, timeout)

def bath_sweep(R, low_T, cutoff, attens, timeout):
    '''
    Step the bath temperature and then take sweeps and noise

    Parameters:
        R: RFSoC data acquisition object
        bath_temps: a list of bath temperatures that will be set by the MC
        attens: a list of attenuator values for vna sweeps
        noise_attens: a list of attenuator values to take noise data
    '''

    #Initialize lakeshore PCS agents
    coldload_agent = OCSClient('LSA291F')
    fp_agent = OCSClient('LSA29EX')
    mxc_agent = OCSClient('LSA25RH')

    R = atten_sweep(R, attens, coldload_agent, fp_agent, mxc_agent, rewrite=True)

    array_temps = update_logs(R, fp_agent, coldload_agent, mxc_agent)
    start_time = time.time()

    # First block to wait and check the array_temps every 20 seconds
    # -----------------------------------------------------
    while array_temps[0]<low_T:
        time.sleep(20)
        array_temps, coldload_temp, mxc_temp = update_temps(fp_agent, coldload_agent, mxc_agent)
        if time.time() - start_time>600:
            print("Still going...", time.time())#time.sleep(20)
            start_time = time.time()

    start_time = time.time()
    #Second block to take an attenuator sweep every 10 minutes until the array temp passes the cutoff
    while array_temps[0]<cutoff and time.time() - start_time<timeout:
        timestamp =time.time()
        # ----------------------
        atten_sweep(R, attens, coldload_agent, fp_agent, mxc_agent)
        #if time.time()
        array_temps, coldload_temp, mxc_temp = update_temps(fp_agent, coldload_agent, mxc_agent)
        while time.time()-timestamp<600 and array_temps[0]<cutoff:
            time.sleep(20)
            array_temps, coldload_temp, mxc_temp = update_temps(fp_agent, coldload_agent, mxc_agent)


def atten_sweep(R, attens,coldload_agent, fp_agent, mxc_agent, rewrite = False, with_noise = False):
    '''
    Sweep over different drive (DAC) and sense (ADC) attenuations and take sweeps and (optionally) timestreams at each. 

    Parameters:
        attens: A tuple of drive and sense attenuation lists
        coldload_agent: the ocs agent for the coldload (for temperature logging)
        fp_agent: the ocs agent for the Lakeshore 372 with the array thermometers (for temperature logging)
        mxc_agent: the ocs agent for the Lakeshore 372 with the mixing chamber thermometer (for temperature logging)
    '''

    timestream_length = 60 # Length of timestream in seconds

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

                    if with_noise:
                        # fetch and save cold load, mxc, and array temps
                        update_logs(R, fp_agent, coldload_agent, mxc_agent)
                        R.find_detectors() # Take VNA sweeps and find detectors
                        
                        update_logs(R, fp_agent, coldload_agent, mxc_agent)
                        R.find_detectors_fine() # Find detectors fine
                    
                        update_logs(R, fp_agent, coldload_agent, mxc_agent)
                        R.take_target_sweep() # Take target sweep at found detector locations

                        update_logs(R, fp_agent, coldload_agent, mxc_agent)
                        R.take_timestream(timestream_length) # Take timestream at found detector locations
                    else:
                        # fetch and save cold load and array temps
                        update_logs(R, fp_agent, coldload_agent, mxc_agent)

                        # Take VNA sweep
                        if rewrite == True:
                            #If its the first sweep, generate a new vna comb, but otherwise just repeat the sweep process.
                            R.take_vna_sweep()
                        else:
                            R.take_vna_sweep(write_tones = False)
 

def update_temps(fp_agent, coldload_agent, mxc_agent):
    '''
    query the current temperatures of the 100 mK plate
    '''
    
    status,msg,session = fp_agent.acq.status()
    array_temp1 = session['data']['fields']['Channel_04']['T']
    array_temp2 = session['data']['fields']['Channel_06']['T']

    status,msg,session = mxc_agent.acq.status()
    mxc_temp = session['data']['fields']['Channel_06']['T']

    cl_temps = []
    for i in range(10):
        status,msg,session = coldload_agent.acq.status()
        time.sleep(0.1)
        coldload_temp = session['data']['fields']['Channel_2']['T']
        cl_temps.append(coldload_temp)
    coldload_temp = round(np.mean(cl_temps),3)
    
    return [array_temp1, array_temp2], coldload_temp, mxc_temp

def update_logs(R, fp_agent, coldload_agent, mxc_agent):
    array_temps, coldload_temp, mxc_temp = update_temps(fp_agent, coldload_agent, mxc_agent)
    R.edit_config(R.ext_cfg, "array_temp", array_temps)
    R.edit_config(R.ext_cfg, "coldload_temp", str(coldload_temp))
    R.edit_config(R.ext_cfg, "bath_temp", str(mxc_temp))
    
    return array_temps

    
if __name__ == "__main__":
    main()
