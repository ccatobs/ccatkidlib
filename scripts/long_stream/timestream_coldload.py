from ocs.ocs_client import OCSClient
from tqdm import tqdm
import numpy as np
import time
import sys
import yaml

#Aim is to stream as the cold load is cooling down. Drone 3 is excluded.

def main():
    # Local imports
    sys.path.append('./../../rfsoc/') # Append path with RFSoC_DAQ.py
    
    from rfsoc_daq import R

    print("\n......\n......\n!!!!!!\nDon't forget to run watch_coldload.py!\n")
    print("!!!!!!\n......\n......")

    # Intialize RFSoC control object
    R = R(cfg_path = "./coldload_system_config.yaml")

    currents = [.305, .224]
    # Run coldload sweep
    coldload_stream(R, currents)

def coldload_stream(R, currents):
    '''
    Takes sweep + timestream data as the coldload is falling in temperature

    Parameters:
        R: RFSoC data acquisition object
        current: List of currents to iterate over 
        attens: List of attenuations to iterate over
    '''

    # Initalize power supply PCS agent
    psu = OCSClient('psuBKP9130B', args = [])
    psu.init.start()
    psu.init.wait()

    # Turn on power supply output
    psu.set_output(channel = 1, state = True)
    psu.set_output.wait()
    
    #Initialize lakeshore PCS agents
    #coldload_agent = OCSClient('LSA291F')
    fp_agent = OCSClient('LSA29EX')

    array_temp1, array_temp2 = update_temps(fp_agent)
    
    # Iterate over currents and take attuenuation sweep and then noise data at each.
    # -----------------------------------------------------
    with tqdm(range(len(currents)), desc = f'DAQ:') as pbar:
        for i in pbar:
            # Prepare the tone comb for streaming
            R.find_detectors()
            update_logs(R,fp_agent)
            R.find_detectors_fine(write_tones=False)
            update_logs(R, fp_agent)
            R.take_target_sweep(write_tones=False)

            #Step the coldload current
            psu.set_current(channel = 1, current=currents[i])
            psu.set_current.wait()

            # Take noise data
            timestream_length = 1200
            R.take_timestream(timestream_length, write_tones=False)
            if i==0:
                psu.set_current(channel = 1, current=0.1*currents[i])
                time.sleep(600)#Sleep for 10 minutes after the timestream but with current lower to help level off
                psu.set_current(channel = 1, current=currents[i])
                time.sleep(1200)#Set it back to the stable temp and sleep for 20 minutes before moving on
            # ----------------------

def update_temps(mxc_agent):
    '''
    query the current temperatures of the 100 mK plate
    '''
    status,msg,session = mxc_agent.acq.status()
    array_temp1 = session['data']['fields']['Channel_04']['T']
    array_temp2 = session['data']['fields']['Channel_06']['T']
    return [array_temp1, array_temp2]

def update_logs(R, fp_agent):
    array_temps = update_temps(fp_agent)
    timestamp = int(time.time())
    R.edit_config(R.ext_cfg, "array_temp", array_temps)
    R.edit_config(R.ext_cfg, "temp_time", timestamp)
    
    return array_temps

    
if __name__ == "__main__":
    main()
