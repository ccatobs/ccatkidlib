from ocs.ocs_client import OCSClient
from tqdm import tqdm
import numpy as np
import time
import sys
import yaml

#Script for taking bath temperature sweep data.

def main():
    # Local imports
    sys.path.append('./../../rfsoc/') # Append path with RFSoC_DAQ.py
    
    from rfsoc_daq import R

    # Intialize RFSoC control object
    R = R(cfg_path = "./bath_sweep_system_config.yaml")

    bath_temps = [65, 105, 130, 165, 190, 215, 240 ] # String of bath temperatures to servo on (in mK)
    
    drive_attens = [0,3,6,9, 12, 15, 18]# DAC attenuations
    sense_attens = [0] # ADC attenuations
    drive_attens_noise = [0, 3]# DAC attenuations to be used when taking noise
    sense_attens_noise = [0] # ADC attenuations to be used when taking noise

    attens = (drive_attens, sense_attens)
    noise_attens = (drive_attens_noise, sense_attens_noise)

    # Run bath sweep
    bath_sweep(R, bath_temps, attens, noise_attens)

def bath_sweep(R, bath_temps, attens, noise_attens):
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
    mxc_agent = OCSClient('LSA25RH',args=[])

    # Set the heater to the lower range to start out  
    mxc_agent.set_heater_range(heater='sample',range=3.16e-3, wait=2)

    # Iterate over currents and take attuenuation sweep and then noise data at each.
    # -----------------------------------------------------
    with tqdm(range(len(bath_temps)), desc = f'MXC Temp:') as pbar:
        for i in pbar:
            temp = bath_temps[i]
            
            # Set the bath temperature
            if temp>180:
                 mxc_agent.set_heater_range(heater='sample',range=10e-3, wait=2) # Set higher heater range

            mxc_agent.servo_to_temperature(temperature = temp/1000.0)
            pbar.set_postfix_str(f'Set point: {temp/1000.0}')

            # Wait for 20 minutes except for the first step.
            if temp>65:
                time.sleep(1200)
            else:
                time.sleep(60)

            # Update logs and configs
            array_temps = update_logs(R, fp_agent, coldload_agent, mxc_agent)
            pbar.set_postfix_str(f'Set point: {temp/1000.0}, Array temps: {array_temps}')
            
            # Take attenuation sweep
            # ----------------------
            atten_sweep(R, attens, coldload_agent, fp_agent, mxc_agent)

            # Take noise data
            # ----------------------
            atten_sweep(R, noise_attens, coldload_agent, fp_agent, mxc_agent, with_noise = True)

def atten_sweep(R, attens,coldload_agent, fp_agent, mxc_agent, with_noise = False):
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
                        if j==0:
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
    
    return [array_temp1, array_temp2], mxc_temp, coldload_temp 

def update_logs(R, fp_agent, coldload_agent, mxc_agent):
    array_temps, coldload_temp, mxc_temp = update_temps(fp_agent, coldload_agent, mxc_agent)
    R.edit_config(R.ext_cfg, "array_temp", array_temps)
    R.edit_config(R.ext_cfg, "coldload_temp", str(coldload_temp))
    R.edit_config(R.ext_cfg, "bath_temp", str(mxc_temp))
    
    return array_temps

    
if __name__ == "__main__":
    main()
