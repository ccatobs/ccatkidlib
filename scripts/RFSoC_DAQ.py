# Import Python modules
import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm
import datetime as dt
import logging
import numpy as np
import time
import pytz

# Append file paths of local modules
sys.path.append('/home/rfsoc/ocs-site-configs/cornell/pisco/clients/')
sys.path.append('/home/rfsoc/MKID-characterization/mkid_cal/')
sys.path.append('/home/rfsoc/primecam_readout/local_notebooks')

# Import local modules
from drone import drone
from so3g.hk import load_range
from ColdloadHeater import ColdloadHeater
from BathHeater import BathHeater
from stream_data import rfsoc_udp_connection
from ocs.ocs_client import OCSClient

class RFSoC_DAQ:

    def __init__(self, num_detectors = 6, drone = 1.1, coldload_ocs_client = 'LSA2FQZ', bath_ocs_client = "LSA1940", data_dir = "/home/rfsoc/rfsoc_result/Darshan/", alcove_dir = '/home/rfsoc/primecam_readout/src', output_dir = '/home/rfsoc/primecam_readout/src/tmp', rfsoc_dir = "xilinx@192.168.2.98:/home/xilinx/primecam_readout/src/alcove_commands/"):
        # Alcove Command Numbers
        self.setNCLO = 20
        self.setFineNCLO = 21
        self.writeNewVnaComb = 31
        self.writeTargCombFromVnaSweep = 32
        self.writeTargCombFromCustomList = 37
        self.vnaSweep = 40
        self.targetSweep = 42
        self.findVnaResonators = 50
        self.findTargResonators = 51

        self.drone = drone
        self.num_detectors = num_detectors
        self.tau = None # Cable delay (in seconds)
        self.current_time = dt.datetime.now() # Fetch current time
        
        # Tail for output files
        self.file_tail = '{bandwidth}_MHz_{int(tone_power)}_dB_CL_{self.coldload_heater.get_temperature():.3f}K_BTH_{self.bath_heater.get_temperature()*10**3:.3f}mK_{self.current_time.strftime("%y%m%d_%H%M%S")}.npy'
        
        # Define code directory path
        self.alcove_dir = Path(alcove_dir)
        self.output_dir = Path(output_dir)
        self.rfsoc_dir = Path(rfsoc_dir)
        sys.path.append(alcove_dir)

        # Define data directory path
        self.data_dir = Path(f"{data_dir}{self.current_time.strftime('%y%m%d')}/")
        self.fig_dir = self.data_dir / 'fig'

        # Set up logging
        logger = logging.getLogger(__name__)
        self.send_msg(self.style("INITIALIZE COLDLOAD SWEEP", "header"))

        # Attempt to make data directory
        self.create_dir(self.fig_dir)
        
        # Create logging txt file
        logging.basicConfig(filename=self.data_dir / 'coldload_sweep_log.txt', filemode = "w",
        format='%(asctime)s %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p", level = logging.INFO)
        
        # Initialize objects to control coldload and bath (sample) heaters
        self.coldload_heater = ColdloadHeater(ocs_client = coldload_ocs_client, log = True)
        self.bath_heater = BathHeater(ocs_client = bath_ocs_client, log = True)

        # Create target sweep directory
        self.target_sweep_dir = self.data_dir / "target_sweep"
        self.create_dir(self.target_sweep_dir)

        # Create a custom frequency directory
        self.custom_freq_dir = self.data_dir / "custom_freq"
        self.create_dir(self.custom_freq_dir)

        # Create timestream directory
        self.timestream_dir = self.data_dir / "timestream"
        self.create_dir(self.timestream_dir)

        # Initialize udp_conn for timestreaming data
        self.udp_conn = rfsoc_udp_connection()

    def full_sweep(self, tone_powers, coldload_powers, bath_powers, in_dB = True, custom_freqs = None, bandwidths = [5, 3, 1], noise_data = False, window = 15e6, output = False, savefig = True, bandwidth_multiplier = 4, stream_time = 60):
        try:
            # Loop over all bath temperatures
            for bath_range, bath_pows in tqdm(bath_powers.items(), desc = "Bath Range"):
                for bath_pow in tqdm(bath_pows, desc = f"Bath powers with range {bath_range}"):
                    bath_power = {bath_range:bath_pow}

                    # Loop over all coldload temperatures
                    for coldload_range, coldload_pows in tqdm(coldload_powers.items(), desc = "Coldload Range"):
                        for coldload_pow in tqdm(coldload_pows, desc = f"Coldload powers with range {coldload_range}"):
                            coldload_power = {coldload_range:coldload_pow}

                            # Loop over all tone powers
                            for tone_power in tqdm(tone_powers, desc="Tone powers"):
                                self.collect_data(tone_power, coldload_power = coldload_power, bath_power = bath_power, custom_freqs = custom_freqs, in_dB = in_dB, bandwidths = bandwidths, window = window, noise_data = noise_data, output = output, savefig = savefig, bandwidth_multiplier = bandwidth_multiplier, stream_time = stream_time)
        except Exception as e:
            # Log stack trace of any exception that occurs
            tqdm.write(e)
            logger = logging.getLogger(__name__)
            logger.exception(self.style("EXCEPTION OCCURED", "header"))

    # Run a target sweep
    def target_sweep(self, tone_power, in_dB = True, file_head = "", custom_freqs = None, coldload_power = {"low": 0}, bath_power = {10e-3:0.025e-3}, bandwidth = 5, N_steps = 500):
        self.send_msg(self.style("TARGET SWEEP", "header"))
        
        # Convert from dB
        if in_dB:
            tone_power = self.convert_from_dB(tone_power)
        
        # Set temperatures of heaters
        self.heater_set_power(self.bath_heater, bath_power)
        self.heater_set_power(self.coldload_heater, coldload_power)
        
        # Custom frequencies are not provided
        if custom_freqs is None:
            self.send_msg("Custom frequencies not provided, attempting to find resonators.")
            custom_freqs = self.find_resonators()

        # Write target comb
        self.write_custom_params(custom_amps = tone_power, custom_freqs = custom_freqs)
        
        # Run target sweep
        self.send_msg(f"Running a {bandwidth} MHz target sweep with {tone_power} tone power, {bath_power} W bath power, and {coldload_power}% coldload power!")
        self.current_time = dt.datetime.now()
        self.run_alcove_command(self.targetSweep, chan_bandwidth = bandwidth, N_steps = N_steps)

        # Get target sweep s21 data file
        targ_sweep_file = self.get_most_recent_file(self.output_dir, "s21_targ*")

        # Save target sweep s21 data file to data_dir
        tone_power = self.convert_to_dB(tone_power)
        targ_sweep_fname = f"{file_head}_s21_targ_" + eval(f"f'{self.file_tail}'")
        os.system(f"cp {targ_sweep_file} " + f"{self.data_dir / targ_sweep_fname}")
        self.send_msg(self.style(f"Saved {targ_sweep_file} to {self.data_dir / targ_sweep_fname}", "saving"))
        return targ_sweep_fname

    # Take successively smaller targ sweeps
    def collect_data(self, tone_power, in_dB = True, bandwidths = [3, 1], custom_freqs = None, coldload_power = {"low": 0}, bath_power = {10e-3:0.025e-3},  window = 15e6, noise_data = False, output = False, savefig = True, bandwidth_multiplier = 4, stream_time = 60):
        # Sort bandwidths from widest to narrowest
        bandwidths = sorted(bandwidths, reverse = True)

        self.send_msg(self.style("CALIBRATION SWEEPS", "header"))
        # Run calibration targ sweeps at each bandwidth, using frequencies from previous each time
        for bandwidth in tqdm(bandwidths, desc = "Calibration Sweeps"):
            # Run target sweep
            targ_sweep_fname = self.target_sweep(tone_power, in_dB = in_dB, file_head = "cal", bandwidth = bandwidth, custom_freqs = custom_freqs, coldload_power = coldload_power, bath_power = bath_power)

            # Find resonators in most recent target sweep
            self.run_alcove_command(self.findTargResonators)
            custom_freqs = self.get_most_recent_file(self.output_dir, "f_res_targ*")

        # Copy final calibration resonator frequencies to data_dir
        cal_freqs_fname = "final_custom_freqs_" + eval(f"f'{self.file_tail}'")
        os.system(f"cp {custom_freqs} " + f"{self.data_dir / cal_freqs_fname}")

        # Determine cable delay (in seconds)
        if self.tau is None:
            self.send_msg("Cable delay not provided, calculating cable delay...")
            self.tau, taus = self.cable_delay(tone_power, in_dB = in_dB, custom_freqs = custom_freqs, coldload_power = coldload_power, bath_power = bath_power)

        # Create drone object
        cal_drone = drone(drone_id = "cal_drone", data_dir = str(self.data_dir) + "/", output_dir = str(self.fig_dir) + "/", targ_sweep_s21_fname = targ_sweep_fname, targ_sweep_f0s_fname = cal_freqs_fname,
         base_temp = self.bath_heater.get_temperature()*1000, load_name = "coldload", amp_gain = tone_power, tau = self.tau)
        cal_drone.init_resonators(window = window, output = output, savefig = False)
        
        # Calculate largest FWHM of resonators
        max_width = 0
        for resonator in cal_drone.resonators:
            width = bandwidth_multiplier*resonator.f0/resonator.Q
            if(width > max_width):
                max_width = width

        # Convert width from Hz to MHz
        bandwidth = max_width/1e6
        # Take targ sweep with bandwidth = max_width
        self.target_sweep(tone_power, in_dB = in_dB, file_head = "final", bandwidth = bandwidth, custom_freqs = custom_freqs, coldload_power = coldload_power, bath_power = bath_power)
        
        # Find resonators in most recent target sweep
        self.run_alcove_command(self.findTargResonators)
        custom_freqs = self.get_most_recent_file(self.output_dir, "f_res_targ*") 
        final_freqs_fname =  f"final_res_freqs_" + eval(f"f'{self.file_tail}'")
        os.system(f"cp {custom_freqs} " + f"{self.data_dir / final_freqs_fname}")
        
        # Create drone object
        final_drone = drone(drone_id = "final_drone", data_dir = str(self.data_dir) + "/", output_dir = str(self.fig_dir) + "/", targ_sweep_s21_fname = targ_sweep_fname, 
                    targ_sweep_f0s_fname = final_freqs_fname, base_temp = self.bath_heater.get_temperature()*1000, load_name = "coldload", amp_gain = tone_power, tau = self.tau)
        final_drone.init_resonators(window = window, output = output, savefig = savefig)
        final_drone.calibrate_IQ_circle(savefig = savefig)

        # Collect on resonance/off resonance noise data
        if noise_data:
            self.collect_noise_data(tone_power = tone_power, in_dB = in_dB, drone = final_drone, bandwidth = bandwidths[-2], coldload_power = coldload_power, bath_power = bath_power, stream_time = stream_time)

    def collect_noise_data(self, tone_power, in_dB = True, stream_time = 60, off_res_shift = 1, num_streams = 2, bandwidth = 4, custom_freqs = None, drone = None, coldload_power = {"low": 0}, bath_power = {10e-3:0.025e-3}):
        self.send_msg(self.style("COLLECT NOISE DATA", "header"))

        # Find most sensitive frequency for each resonator (in Hz)
        sens_freqs = np.ones(self.num_detectors)
        res_freqs = np.ones(self.num_detectors)
        for i, resonator in enumerate(drone.resonators):
            res_freqs[i], res_freq_ind = resonator.find_manual_f0()
            try:
                resonator.freq_vs_phase_fit()
                sens_freqs[i] = resonator.get_max_dphi_df()
            except:
                self.send_msg("Failed to find most sensitive frequency. Using resonant frequency instead!")
                sens_freqs[i] = res_freqs[i]
        self.send_msg(f"Most Sensitive Frequencies: {sens_freqs}")
        # Save most sensitive frequencies
        sens_freqs_fname =  f"sens_custom_freqs_" + eval(f"f'{self.file_tail}'")
        np.save(f"{self.data_dir / sens_freqs_fname}", sens_freqs)
        
        self.send_msg("Taking on resonance timestream data")
        # Collect on resonance time stream data
        self.take_timestream(0, stream_time, file_head = "timestream_on_res", bandwidth = bandwidth, tone_power = tone_power)

        # Run target sweep centered at most sensitive frequencies
        self.target_sweep(tone_power, in_dB = in_dB, file_head = "noise", bandwidth = bandwidth, custom_freqs = sens_freqs, coldload_power = coldload_power, bath_power = bath_power)
        
        self.send_msg("Taking timestream data at most sensitive frequencies and off resonance.")
        # Take off-resonance and most sensitive time-stream data
        delta_fs = np.linspace(0, off_res_shift, num_streams)
        for delta_f in tqdm(delta_fs, desc = "Timestreams"):
            self.take_timestream(delta_f, stream_time, file_head = "timestream_sens", bandwidth = bandwidth, tone_power = tone_power)

    def cable_delay(self, tone_power, in_dB = True, custom_freqs = None, coldload_power = {"low":0}, bath_power = {10e-3:0.025e-3}):
        self.send_msg(self.style("CALCULATING CABLE DELAY", "header"))
        
        from scipy.stats import linregress
        
        # Find resonator frequencies if not provided
        if custom_freqs is None:
            self.send_msg("Custom frequencies not provided, attempting to find resonators.")
            custom_freqs =self.find_resonators()

        # Load custom_freqs file    
        try:
            freqs = sorted(np.load(custom_freqs))
        # Assume an array was passed
        except:
            freqs = sorted(custom_freqs)

        # Minimum Seperation between resonators (in Hz)
        min_sep = min([freqs[i+1] - freqs[i] for i in range(len(freqs)-1)])
        bandwidth = 0.95*min_sep

        # Take target sweep with largest bandwidth without resonator overlap
        target_sweep_fname = self.target_sweep(tone_power, in_dB = in_dB, custom_freqs = custom_freqs, file_head = "cable_delay", coldload_power = coldload_power, bath_power = bath_power, bandwidth = bandwidth/1e6, N_steps = 50)
        
        # Load target sweep data
        fs, s21 = np.load(self.data_dir / target_sweep_fname, allow_pickle = True)
        
        # Calculate phase data 
        phases = np.arctan2(np.imag(s21), np.real(s21))
        taus = np.ones(self.num_detectors)
        for i, freq in enumerate(freqs):
            res_mask = np.where(np.logical_and(freq - bandwidth/2 < fs, fs < freq + bandwidth/2))
            res_fs = np.real(fs[res_mask])
            res_phases = phases[res_mask]
            offset = 0

            for j in range(len(res_phases) - 1):
                res_phases[j] += offset
                if abs((res_phases[j+1] + offset)  - res_phases[j]) > 6:
                    offset -= 2*np.pi
            res_phases[-1] += offset

            # Calculate cable delay (in seconds)
            taus[i] = linregress(res_fs, res_phases)[0]/(2*np.pi)
            avg_tau = np.average(taus)
        self.send_msg(f"Cable Delays: {taus}")
        self.send_msg(f"Average Cable Delay: {avg_tau}")
        return avg_tau, taus

    def take_timestream(self, shift, stream_time, file_head = "", bandwidth = 5, tone_power = 72):       
            self.send_msg(self.style("TAKING TIMESTREAM DATA","header"))

            # Shift frequencies by shift and take timestream data
            self.run_alcove_command(self.setFineNCLO, shift)
            self.send_msg(f"Taking {stream_time} s of timestream data!")
            timestream = self.udp_conn.stream_data(t_sec = stream_time)
            self.send_msg(f"Finished taking timestream data.")

            # Save timestream to timestream directory
            timestream_fname = file_head + f"_{shift}_MHz_" + eval(f"f'{self.file_tail}'")
            np.save(f"{self.timestream_dir / timestream_fname}", timestream)
            self.send_msg(self.style(f"Saved timestream data to: '{self.timestream_dir / timestream_fname}'", "saving"))

            # Shift frequencies back to original values
            self.run_alcove_command(self.setFineNCLO, 0)

    def setup_sweep(self, window = (400, 800)):
        center = 600
        if window is not None:
            try:
                center = int((window[0] + window[1])/2)
            except:
                pass
        bandwidth = abs(window[1] - window[0])

        # Set NCLO (center frequency of sweep)
        self.run_alcove_command(self.setNCLO, center)
        self.run_alcove_command(self.writeNewVnaComb)

        self.run_alcove_command(self.vnaSweep, bandwidth)
        vna_sweep_file = self.get_most_recent_file(self.output_dir, "s21_vna*")
        vna_sweep_fname = (f"setup_s21_vna_center_{center}_MHz_bandwidth_{bandwidth}_MHz")
        os.system(f"cp {vna_sweep_file} " + f"{self.data_dir / vna_sweep_fname}")

        return vna_sweep_fname

    # Helper Methods

    def create_dir(self, dir_path):
        try:
            dir_path = Path(dir_path)
        except:
            pass

        try:
            if not dir_path.exists():
                dir_path.mkdir(parents = True, exist_ok = False)
                self.send_msg(f"The directory '{dir_path}' was successfully created!")
            else:
                self.send_msg(f"The directory '{dir_path}' already exists! Directory was not overwritten.")
        except:
            self.send_err(f"The directory '{dir_path}' could not be created! Ensure that the file path is valid.")

    def heater_set_power(self, heater, power, dynamic = True, sampling_interval = 3*60, wait_time = 10, max_time = 60*60, tol = 0.005):
        
        # Get current heater range and power
        curr_power = heater.get_power()
        curr_range = heater.get_range()

        # Get stability of heater temperature (True if stable, False if unstable/unknown)
        stable = heater.get_stability()

        # New range and power values
        new_range, new_power = list(power.items())[0]

        # Check if the new range and power are different from the current value
        if new_power != curr_power or new_range != curr_range or not stable:
            self.send_msg(self.style("SETTING HEATER POWER", "header"))

            # Attempt to set the range and power of the heater
            try:
                heater.set_range(new_range)
                heater.set_power(new_power)

                # Set stability of heater(s)
                heater.set_stability(False)

                if dynamic: # Also wait for bath heater temperature to stabilize
                    self.bath_heater.set_stability(False)
            
            # If cannot set new heater range or power, revert to current range and power (!!!Should Errror handle in heater class)
            except:
                self.send_err("Error setting heater range or power. Resetting to prior values.")
                try:
                    heater.set_range(curr_range)
                    heater.set_power(curr_power)
                except:
                    self.send_err("|CRITICAL| Failure to revert heater to prior values. Stopping code execution!")
                    sys.exit(0)

            # Wait for the heater temperature to stabilize
            if dynamic: # Dynamic stabilization
                self.send_msg(f"Dynamically waiting for heater temperature to stabilize.")
                temps = [heater.get_temperature()] # Store temperature of heater every "sampling_interval" seconds
                
                with tqdm(total = 100, desc = "Temperature Stabilizing") as pbar:
                    start_time = time.time()
                    curr_time = start_time
                    curr_prog = 0

                    # or if it has been longer than "max_time" seconds
                    while not heater.get_stability() and (curr_time - start_time) < max_time:
                        time.sleep(sampling_interval)

                        # Check if two most recently sampled temperatures are within tolerance of each other 
                        # If within tolerance, set stability of heater to True
                        temps = np.append(temps, [heater.get_temperature()])
                        temp_change = abs(temps[-1] - temps[-2])
                        if temp_change > 0 and temp_change < tol:
                            heater.set_stability(True)
                        elif temp_change == 0:
                            curr_prog = 100 - curr_prog
                        else:
                            try:
                                invs = 1/temp_change
                                curr_prog = 100*invs/(20 + invs) - curr_prog
                            except:
                                pass
                        pbar.update(curr_prog)
                        curr_time = time.time()
                
                # Set heater stability to True
                heater.set_stability(True)    
                self.send_msg(f"Transient Heater Temps: {temps}")

                # Wait for bath heater to stabilize
                self.heater_set_power(self.bath_heater, {self.bath_heater.get_range(): self.bath_heater.get_power()}, dynamic = True, tol = tol/10)
            else: # Wait "wait_time" seconds for the temperature to stabilize
                self.send_msg(f"Waiting {wait_time/60} minutes for temperature to stabilize.")
                for i in tqdm(range(int(np.ceil(wait_time/5))), desc = "Temperature Stabilizing"):
                    time.sleep(5)
                self.send_msg(f"Final Heater Temp: {heater.get_temperature()} K")
                heater.set_stability(True)

    def run_alcove_command(self, command_num, *args, alcove_out = False, **kwargs):
        #self.send_msg(self.style(f"RUNNING ALCOVE COMMAND: {command_num}", "header"))
        # Change to directory containing alcove commands
        try:
            if Path.cwd() != self.alcove_dir:
                os.chdir(self.alcove_dir)
                self.send_msg(f"Working directory was changed to '{Path.cwd()}'")
        except:
            self.send_err(f"Failed to change working directory to '{self.alcove_dir}'")
        
        # Create command string
        cmd = f"python3 queen_cli.py {command_num} {self.drone}" 
        
        # Add additional arguments
        if len(args) > 0 or len(kwargs) > 0:
            cmd += " -a"
            for arg in args:
                cmd += f' {arg}'
            if len(kwargs) > 0:
                cmd += ' "'
                for key, value in kwargs.items():
                    cmd += f'{key}={value} '
                cmd += '"'

        # Run command
        try:        
            os.system(cmd)
            self.send_msg(self.style(f"Successfully ran alcove command: '{cmd}'", "command"))
        except:
            self.send_err(f"Failed to run command: '{cmd}'")

    def write_custom_params(self, **kwargs):
        # Update the relevant custom parameter array(s) on the rfsoc board
        for key, value in kwargs.items():
            # Check if value is a file path
            try: 
                value = Path(value)
                if not value.exists():
                    self.send_err("Exception: Invalid file path was passed to write_custom_params")
                    return
            # If not file path, assume value is a number or an array of numbers
            except:
                try: 
                    if not len(value) == self.num_detectors:
                        self.send_msg(f"Length of '{key}={value}' does not match number of detectors specified!")
                        return
                except TypeError:
                    value = value*np.ones(self.num_detectors)
                # Save the passed array
                np.save(self.data_dir / key, value)

                # Change value so that it points to the file path of saved array
                value = self.data_dir / f"{key}.npy"

            cmd = f"scp {value} {self.rfsoc_dir}"

            # Match input array to correct custom parameter array
            if key == "custom_freqs":
                cmd += "/custom_freqs.npy"
            elif key == "custom_amps":
                cmd += "/custom_amps.npy"
            elif key == "custom_phis":
                cmd += "/custom_phis.npy"
            else:
                return
            
            # Copy the array with custom parametrs onto the rfsoc board
            os.system(cmd)
            self.send_msg(self.style(f"Ran command {cmd}", "command"))
        
        # Write sweep comb based on custom parameters
        self.run_alcove_command(self.writeTargCombFromCustomList)

    def find_resonators(self, vna_center = 600, width_min=2, width_max=150, prom_dB=0.1, distance=17000):
        # Custom tone power?
        
        # Run an initial VNA sweep
        self.run_alcove_command(self.writeNewVnaComb)
        self.run_alcove_command(self.vnaSweep, vna_center)

        # Find resonators
        self.run_alcove_command(self.findVnaResonators, width_min=width_min, width_max=width_max, prom_dB=prom_dB, distance=distance)

        # Write target sweep based on VNA sweep
        self.run_alcove_command(self.writeTargCombFromVnaSweep)

        # Get most recent resonator frequencies from VNA sweep
        return self.get_most_recent_file(self.output_dir, "f_res_vna*")

    def convert_from_dB(self, power):
        return 10**(power/20)

    def convert_to_dB(self, power):
        return 20*np.log10(power)

    def send_msg(self, msg):
        logger = logging.getLogger(__name__)
        try:
            logger.info(msg)
            tqdm.write(msg)
        except:
            tqdm.write("Error writing message. Ensure that the message is a string.")

    def send_err(self, err):
        logger = logging.getLogger(__name__)
        try:
            logger.error(msg)
            tqdm.write(f"| ERROR | {msg}")
        except:
            tqdm.write("Error writing message. Ensure that the message is a string.")

    def style(self, string, styl = "default"):
        curr_str = ""
        if styl == "header":
            bar_len = 150
            for i in range(bar_len):
                curr_str += "="
            curr_str += "\n"
            for i in range(int((bar_len - len(string))/2)):
                curr_str += " "
            curr_str += string + "\n"

            for i in range(bar_len):
                curr_str += "="
            return curr_str
        if styl == "command":
            return f">>> {string}"
        if styl == "saving":
            return f"| SAVING | {string}"

    def get_creation_time(self, file):
        try:
            file = Path(file)
            return file.stat().st_ctime
        except:
            return -1
        
    def get_most_recent_file(self, directory, file_identifier):
        try:
            directory = Path(directory)
            return sorted(directory.glob(file_identifier), key = self.get_creation_time, reverse = True)[0]
        except:
            send_msg(f"Failed to fetch most recent file in {directory} with identifier {file_identifier}")
            return "invalid/path"
        
