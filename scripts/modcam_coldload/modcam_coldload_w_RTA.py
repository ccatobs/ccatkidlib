from ocs.ocs_client import OCSClient
from tqdm import tqdm
import numpy as np
import time
import sys
import yaml

#Approach of this is to set the coldload current aggressively and then bring it down slowly to reduce the wait time from the time constant.
#Once at a coldload temperature, go through and get many vna sweeps and target sweeps at a range of powers (only vna sweeps for lower powers). 
#Fit a few higher power resonators. Choose a tone power based on fitted resonators. Take noise.

def main():
    # Local imports
    sys.path.append('./../../rfsoc/') # Append path with RFSoC_DAQ.py
    
    from rfsoc_daq import R

    # Intialize RFSoC control object
    R = R(cfg_path = "./coldload_system_config_with_RTA.yaml")

    # Currents and attenuations to sweep over
    #voltages = [10]#[0, 8,12,12,0] # Voltages in Volts. Just sets the max possible voltage. Not necessarily used.
    #currents = [0, .079, .11, .137, .158, .177, .205, .224] # Estimates to the required currents in amps
    #
    coldload_cfg = "./coldload_tuning_params.yaml"
    drive_attens = [0,3]# DAC attenuations
    drive_attens_noise = [0, 3]#DAC attenuations to be used when taking noise 
    sense_attens = [12, 18, 24, 30] # ADC attenuations
    restart = True

    attens = (drive_attens, sense_attens)
    noise_attens = (drive_attens_noise, sense_attens)

    # Run coldload sweep
    coldload_sweep(R, coldload_cfg, attens, noise_attens, restart = restart)

def coldload_sweep(R, coldload_cfg, attens, noise_attens, restart = False):
    '''
    Changes coldload temperature and takes sweep + timestream data for various attenuations

    Parameters:
        R: RFSoC data acquisition object
        current: List of currents to iterate over 
        attens: List of attenuations to iterate over
    '''

    coldload_tuning_params = load_coldload_params(coldload_cfg)
    currents = coldload_tuning_params['currents']
    remaining = coldload_tuning_params['steps_remaining']
    if remaining == 1:
        currents = [currents]
    # Initalize power supply PCS agent
    psu = OCSClient('psuBKP9130B', args = [])
    psu.init.start()
    psu.init.wait()

    # Turn on power supply output
    psu.set_output(channel = 1, state = True)
    psu.set_output.wait()

    #Set the current to zero (unless restarting) and then set a max voltage for the power supply
    if restart:
        print(currents[0])
        psu.set_current(channel = 1, current = currents[0])
    else:
        psu.set_current(channel = 1, current = 0)
    psu.set_current.wait()
    psu.set_voltage(channel = 1, volts = 10)

    #Initialize lakeshore PCS agents
    coldload_agent = OCSClient('LSA291F')
    fp_agent = OCSClient('LSA29EX')

    coldload_temp, [array_temp1, array_temp2] = update_temps(coldload_agent, fp_agent)
    
    #Start a local temperature log for the cold load.
    logname = "coldload_temps.log"
    timestamp = int(time.time())
    with open(logname, "a+") as file:
        file.write("Coldload,\t\tTimestamp\n" +f"{np.round(coldload_temp, 3)},\t\t{timestamp}")
    
    # Iterate over currents and take attuenuation sweep and then noise data at each.
    # -----------------------------------------------------
    while remaining>0:

        with tqdm(range(len(currents)), desc = f'DAQ:') as pbar:
            for i in pbar:
                # Change coldload temp 
                # This includes the wait and so forth
                # --------------------
                if restart:
                    last_cl_temp = step_coldload_power(psu, coldload_cfg, logname, coldload_agent, fp_agent, restart=True)
                    restart = False
                else:
                    last_cl_temp = step_coldload_power(psu, coldload_cfg, logname, coldload_agent, fp_agent)

                # Edit external config file
                curr, volt = get_updated_psu_stats(psu)
                R.edit_config(R.ext_cfg, "power_supply_current", curr)
                R.edit_config(R.ext_cfg, "power_supply_voltage", volt)
                pbar.set_postfix_str(f'Voltage: {volt}; Current: {curr}')
            
                # Take attenuation sweep
                # ----------------------
                #atten_sweep(R, attens, coldload_agent, fp_agent, logname)

                # Take noise data
                # ----------------------
                atten_sweep(R, noise_attens, coldload_agent, fp_agent, logname, with_noise = True)

                coldload_tuning_params = load_coldload_params(coldload_cfg)
                currents = coldload_tuning_params['currents']
                remaining = coldload_tuning_params['steps_remaining']

def atten_sweep(R, attens,coldload_agent, fp_agent, logname, with_noise = False):
    '''
    Sweep over different drive (DAC) and sense (ADC) attenuations and take sweeps and (optionally) timestreams at each. 

    Parameters:
        attens: A tuple of drive and sense attenuation lists
    '''

    timestream_length = 30 # Length of timestream in seconds

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
                        # fetch and save cold load and array temps
                        last_cl_temp = update_logs(R, coldload_agent, fp_agent, logname)
                        
                        R.find_detectors() # Take VNA sweeps and find detectors
                        
                        last_cl_temp = update_logs(R, coldload_agent, fp_agent, logname)
    
                        R.find_detectors_fine(write_tones = False) # Find detectors fine
                    
                        last_cl_temp = update_logs(R, coldload_agent, fp_agent, logname)
    
                        R.take_target_sweep(write_tones = False) # Take target sweep at found detector locations

                        last_cl_temp = update_logs(R, coldload_agent, fp_agent, logname)
                        
                        R.take_timestream(timestream_length, write_tones=False) # Take timestream at found detector locations
                    else:
                        # fetch and save cold load and array temps
                        last_cl_temp = update_logs(R, coldload_agent, fp_agent, logname)
                        
                        R.take_vna_sweep() # Take VNA sweep

def coldload_wait(t_sec):
    '''
    Wait for the coldload temp to settle. Currently a very simple sleep function but could/should be changed to a more complicated function.
    '''
    time.sleep(t_sec)

def update_temps(coldload_agent, mxc_agent):
    '''
    query the current temperatures of the coldload and 100 mK plate
    '''
    cl_temps = []
    for i in range(10):
        status,msg,session = coldload_agent.acq.status()
        time.sleep(0.1)
        coldload_temp = session['data']['fields']['Channel_2']['T']
        cl_temps.append(coldload_temp)
    
    status,msg,session = mxc_agent.acq.status()
    array_temp1 = session['data']['fields']['Channel_04']['T']
    array_temp2 = session['data']['fields']['Channel_06']['T']
    return round(np.mean(cl_temps),3), [array_temp1, array_temp2]

def update_logs(R, coldload_agent, fp_agent, logname):
    coldload_temp, array_temps = update_temps(coldload_agent, fp_agent)
    timestamp = int(time.time())
    with open(logname, "a+") as file:
        file.write(f"{np.round(coldload_temp, 3)},\t\t{timestamp}\n")
                        
    R.edit_config(R.ext_cfg, "coldload_temp", str(coldload_temp))
    R.edit_config(R.ext_cfg, "array_temp", array_temps)
    R.edit_config(R.ext_cfg, "temp_time", timestamp)
    
    return coldload_temp

def step_coldload_power(psu, coldload_cfg, logname,coldload_agent, fp_agent, restart = False):
    """
    Helper function to set the coldload power, overshooting the power and then stepping down to help work around the slow time constant. 
    Requires being passed a lot of ridiculous variables because I didn't have time to clean up the overall logic
    """
    # Change power supply current to overshoot at first

    cparams = load_coldload_params(coldload_cfg)
    currents = cparams['currents']
    temps = cparams['temps']
    try:
        len(currents)
        current = currents[0]
        temp = temps[0]
    except:
        current = currents
        currents = [current]
        temp = temp
        temps = [temp]

    updated_cparams = update_coldload_params(cparams)
    save_coldload_params(updated_cparams,coldload_cfg)

    if restart:
        print(f"Skipping long hold on restart at {current}")
        psu.set_current(channel = 1, current = current)
        coldload_temp, [array_temp1, array_temp2] = update_temps(coldload_agent, fp_agent)
        timestamp = int(time.time())
        with open(logname, "a+") as file:
            file.write(f"\n{np.round(coldload_temp, 3)},\t\t{timestamp}")
        time.sleep(cparams['restart_hold_time']*58)
        return 
        
    
    a = cparams['stage1']['current_factor']
    hold_time = cparams['stage1']['nominal_hold_time']
    psu.set_current(channel = 1, current = a*current)
    psu.set_current.wait()

    coldload_temp, array_temps = update_temps(coldload_agent, fp_agent)

    for i in range(hold_time):
        if temp - coldload_temp<0.4:
            #if it's heating up faster than expected move down to lower current faster
            break
                    
        time.sleep(59)
                
        #Updating log
        coldload_temp, [array_temp1, array_temp2] = update_temps(coldload_agent, fp_agent)
        timestamp = int(time.time())
        with open(logname, "a+") as file:
            file.write(f"\n{np.round(coldload_temp, 3)},\t\t{timestamp}")

    cparams = load_coldload_params(coldload_cfg)
    #Step down to lower overshoot on current
    a2 = cparams['stage2']['current_factor']
    hold_time = cparams['stage2']['nominal_hold_time']
    psu.set_current(channel = 1, current = a2*current)
    psu.set_current.wait()
            
    #Check every 5 minutes for up to 10 minutes before reducing power
    for i in range(hold_time):

        if temp - coldload_temp<0.05:
            #if it's heating up faster than expected move down to lower current faster
            break

        time.sleep(59)

        #Updating log
        coldload_temp, [array_temp1, array_temp2] = update_temps(coldload_agent, fp_agent)
        timestamp = int(time.time())
        with open(logname, "a+") as file:
            file.write(f"\n{np.round(coldload_temp, 3)},\t\t{timestamp}")


    #Settle
    cparams = load_coldload_params(coldload_cfg)
    a3 = cparams['stage3']['current_factor']
    hold_time = cparams['stage3']['nominal_hold_time']
    psu.set_current(channel = 1, current = a3*current)
    psu.set_current.wait()

    for i in range(hold_time):
        time.sleep(59)

        #Updating log
        coldload_temp, [array_temp1, array_temp2] = update_temps(coldload_agent, fp_agent)
        timestamp = int(time.time())
        with open(logname, "a+") as file:
            file.write(f"\n{np.round(coldload_temp, 3)},\t\t{timestamp}")

    #Set final current
    psu.set_current(channel = 1, current = current)
    psu.set_current.wait()
    cparams = load_coldload_params(coldload_cfg)
    final_hold = cparams['final_hold_time']
    time.sleep(60*final_hold)
    return

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

def load_coldload_params(param_file):
    """
    Load the tuning parameters from the file.
    """
    with open(param_file, "r") as file:
        for f in yaml.safe_load_all(file):
            coldload_tuning_params = f
    
    coldload_tuning_params['steps_remaining'] = len(coldload_tuning_params['currents'])
    
    return coldload_tuning_params

def save_coldload_params(params, param_file):
    """
    Save updated set of params as param_file
    """
    yfile = open(param_file, "w")
    yaml.dump(params, yfile)
    yfile.close()
    return

def update_coldload_params(params):
    """
    Move first item from current step list to completed list, retally remaining
    """

    completed = params['completed']
    currents = params['currents']
    completed.append(currents[0])
    params['completed']=completed
    params['currents'] = currents[1:]
    params['steps_remaining'] = False
    params['temps'] = params['temps'][1:]
    return params 

    
if __name__ == "__main__":
    main()
