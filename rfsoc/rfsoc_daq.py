# Import Python modules
import os
import sys
from pathlib import Path
import numpy as np
import time

import rfsoc_io
import utils

class R:
    '''
    Class for tuning and data acquisition of Microwave Kinetic Inductance Detectors (MKIDs) using
    Radio Frequency System on a Chip (RFSoC).
    '''

    def __init__(self, cfg_path = "./config.yaml"):
        '''
        Constructor for RFSoC_DAQ. Creates directories for data storage, configures logger, and starts
        RFSoC PCS agent.
        '''

        # Current date in yyyy/mm/dd
        curr_date = time.strftime('%Y%m%d', time.gmtime())

        # Create session id from first ten digits of current time
        self.sess_id = str(time.time())[:10]

        # Create a global timestamp used for file naming and pairing
        self.timestamp = str(time.time()).split('.')[0]

        # Read configuration file
        cfg = rfsoc_io.load_config(cfg_path)
        try:
            self.cfg, self.cfg_io = cfg
        except:
            self.cfg = cfg
            self.cfg_io = None

        # Assign commonly used parameters to variables
        self.com_to = self.cfg_io['rfsoc_io']['com_to'] # Which RFSoC board and drone to send commands
        self.output = self.cfg_io['io']['terminal_output']
        
        self.save_cfg = self.cfg_io['io']['save_config_copy']
        self.save_data = self.cfg_io['io']['save_data']

        self.data_dir = Path(self.cfg_io['file_paths']['base_data_dir'])
        self.tmp_data_dir = Path(self.cfg_io['file_paths']['tmp_data_dir'])

        # Save session ID to config files
        self.edit_config(self.cfg, 'sess_id', self.sess_id)
        self.edit_config(self.cfg_io, 'sess_id', self.sess_id)

        # Create book directories inside data_dir
        new_dir_paths = rfsoc_io.create_book(curr_date, self.sess_id, self.com_to, self.data_dir, output = self.output)
        self.rfsoc_dir, self.targ_dir, self.timestream_dir, self.vna_dir = new_dir_paths

        # Setup logger
        log_dir = self.rfsoc_dir
        rfsoc_io.setup_logging(log_dir / self.cfg_io['io']['logging_fname'], self.cfg_io['io']['logging_level'], output = self.output)

        # Add paths to primecam_readout modules, PCS clients, and analysis code
        sys.path.append(self.cfg_io['file_paths']['primecam_readout'])
        sys.path.append(self.cfg_io['file_paths']['pcs_dir'])
        sys.path.append(self.cfg_io['file_paths']['analysis_code_dir'])
        rfsoc_io.send_msg('DEBUG', 'Finished appending file paths.', self.output)

        # Load local modules
        from timestream import TimeStream # type: ignore # Load modules from primecam_readout
        from ocs.ocs_client import OCSClient # Import PCS client module

        # Initialize PCS clients
        self.rfsoc = OCSClient(self.cfg_io['pcs_agents']['rfsoc_agent'], args=[])
        rfsoc_io.send_msg('INFO', f'Initialized RFSoC agent. Communicating with drone: {self.com_to}', self.output)

        # Attempt Initialize timestream client
        try:
            self.streamer = TimeStream(self.cfg_io['rfsoc_io']['udp_ip'], 
                                   self.cfg_io['rfsoc_io']['udp_port'])
        except:
            rfsoc_io.send_msg('INFO', 'Another instance of rfsoc_daq.py is already running! Please close this instance and try again.', self.output)
            sys.exit()

        # Set NCLO frequency of RFSoC
        rtn = self.rfsoc.setNCLO(com_to = self.com_to, f_lo = self.cfg['rfsoc_tones']['drone_NCLO'])
        rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)

        rfsoc_io.send_msg('INFO', f"Set NCLO to {self.cfg['rfsoc_tones']['drone_NCLO']} MHz!", self.output)

        # Save init config
        rfsoc_io.save_config(self.rfsoc_dir / f'{self.timestamp}_init_config.yaml', self.cfg, self.save_cfg)
        rfsoc_io.save_config(self.rfsoc_dir / f'{self.timestamp}_init_config_io.yaml', self.cfg_io, self.save_cfg)

    ##############################
    # Data Acquisition Functions #
    ##############################

    def take_vna_sweep(self, **kwargs):
        '''
        Take a vector network analyzer (VNA) sweep using RFSoC. 

        Parameters:
            bandwidth (int): Bandwidth of the sweep in MHz. 
            N_steps (int): Number of points per tone 
        Keyword Arguments:
            NCLO (int): Numerically controlled local oscillator frequency in MHz. Center frequency of VNA sweep.
        Returns:
            output (str): VNA sweep file path
        '''

        N_steps = 500
        for key, value in kwargs.items():
            if key == 'NCLO':
                rtn = self.rfsoc.setNCLO(com_to = self.com_to, f_lo = value)
                rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)

                self.cfg['rfsoc_tones']['drone_NCLO'] = int(value)
                rfsoc_io.send_msg('INFO', f'Set NCLO to {value} MHz!', self.output)
            elif key == 'N_steps':
                N_steps = value

        # Write VNA comb
        rfsoc_io.send_msg('INFO', 'Writing new VNA comb!', self.output)
        rtn = self.rfsoc.writeNewVnaComb(com_to = self.com_to)
        rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
        rfsoc_io.send_msg('INFO', 'Successfully wrote new VNA comb!', self.output)
        time.sleep(1) # Wait before taking sweep (not waiting can affect sweep quality)

        # Take VNA sweep
        rfsoc_io.send_msg('INFO', 'Taking VNA sweep!', self.output)
        rtn = self.rfsoc.vnaSweep(com_to = self.com_to, N_steps = N_steps)
        rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
        rfsoc_io.send_msg('INFO', 'Finished taking VNA sweep!', self.output)


        # Save VNA sweep
        vna_file = rfsoc_io.get_most_recent_file(self.tmp_data_dir, 's21_vna*', self.output)

        # Set timestamp to current time and save config
        self.timestamp = str(time.time()).split('.')[0]
        self.cfg = rfsoc_io.save_config(self.rfsoc_dir / f"{self.timestamp}_vna_config.yaml", self.cfg, self.save_cfg)
        
        if self.save_data:
            fname = self.cfg_io["file_names"]["vna_fname"]
            fname = eval(f"f'{fname}'") + '.npy'
            cmd = f"cp {vna_file} {self.vna_dir / fname}"

            os.system(cmd)
            rfsoc_io.send_msg('DEBUG', f'>>> Successfully ran command: {cmd}!', self.output)
            return self.vna_dir / fname
        else:
            return vna_file

    def take_target_sweep(self, **kwargs):
        '''
        Take a target sweep around the specified tones.

        Parameters:
            bandwidth (int): Bandwidth of sweep around each tone in MHz.
            N_steps (int): Number of points per tone
            write_tones (bool): Whether to write new comb or use current comb
            tone_freqs: Frequencies at which to place tones
            tone_powers: Readout power of tones
            tone_phis: Phase of tones
        Returns:
            output (str): File path of target sweep
        '''

        write_tones = True
        # Evaluate kwargs
        for key, value in kwargs.items():
            if key == 'bandwidth':
                self.cfg['rfsoc_tones']['bandwidth'] = value
            elif key == 'N_steps':
                self.cfg['rfsoc_tones']['N_steps'] = value
            elif key == 'write_tones':
                write_tones = value
        
        # Write new target sweep comb
        if write_tones:
            self.write_config_tones(**kwargs)
            time.sleep(1) # Wait before taking sweep (not waiting can affect sweep quality)
        
        rfsoc_io.send_msg('INFO', 'Taking target sweep!', output = self.output)
        # Take target sweep
        rtn = self.rfsoc.targetSweep(com_to = self.com_to, 
                               chan_bandwidth = self.cfg['rfsoc_tones']['bandwidth'], 
                               N_steps = self.cfg['rfsoc_tones']['N_steps'])
        rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
        rfsoc_io.send_msg('INFO', 'Finished taking target sweep!', output = self.output)
        
        # Save target sweep
        target_file = rfsoc_io.get_most_recent_file(self.tmp_data_dir, 's21_targ*', output = self.output)
        
        # Set timestamp to current time and save config
        self.timestamp = str(time.time()).split('.')[0]
        self.cfg = rfsoc_io.save_config(self.rfsoc_dir / f"{self.timestamp}_targ_config.yaml", self.cfg, self.save_cfg)
        
        if self.save_data:
            fname = self.cfg_io["file_names"]["targ_fname"]
            fname = eval(f"f'{fname}'") + '.npy'

            cmd = f'cp {target_file} {self.targ_dir / fname}'

            os.system(cmd)
            rfsoc_io.send_msg('DEBUG', f'>>> Successfully ran command: {cmd}!', output = self.output)

            return self.targ_dir / fname
        else:
            return target_file
    
    def take_timestream(self, t_sec, **kwargs):
        '''
        Take timestream data using RFSoC.

        Parameters:
            time (int): Length of timestream in seconds
            tone_freqs: Frequencies at which to place tones
            tone_powers: Readout power of tones
            tone_phis: Phase of tones 
        Return:
            output (list of str): Timestream file paths if save_data = True
        Returns:
            output: Return complex S21 timestream data in array
        '''

        write_tones = True
        save_data = self.save_data
        for key, value in kwargs.items():
            if key == 'write_tones':
                write_tones = value
            elif key == 'save_data':
                save_data = value

        # Write new tones
        if write_tones:
            self.write_config_tones(**kwargs)

        rfsoc_io.send_msg('INFO', f'Taking {t_sec} seconds of timestream data!', self.output)

        # Get timestream I, Q data
        N_packets = int(eval(self.cfg_io['rfsoc_io']['sampling_freq'])*t_sec)
        I, Q = self.streamer.getTimeStreamChunk(N_packets)
        rfsoc_io.send_msg('INFO', 'Finished taking timestream data!', self.output)
        
        # Combine I, Q data into complex S21 data and discard data from unused tones
        s21z = list([])
        for i, data in enumerate(zip(I, Q)):
            if i == self.cfg['rfsoc_tones']['num_tones']: break
            s21z.append(data[0] + 1j*data[1])
        
        # Cast to an numpy array
        s21z = np.array(s21z)

        # Set timestamp to current time and save config
        self.timestamp = str(time.time()).split('.')[0]

        if save_data:
            self.cfg = rfsoc_io.save_config(self.rfsoc_dir / f"{self.timestamp}_stream_config.yaml", self.cfg, self.save_cfg)
            return self.save_timestream(s21z)
        else:
            return s21z

    ####################
    # Tuning Functions #
    ####################
    
    def find_detectors(self, new_sweep = True, peak_prom_db = 0.1, peak_prom_std = 0, peak_dis = 17000, peak_width_min = 100, peak_width_max=300, **kwargs):
        '''
        Find detectors and place tones at their minima. Length of bins is 500,000 Hz / N_steps.
        Default parameters work well for N_steps = 500 and should be scaled appropriately if N_steps is changed.

        Parameters:
            new_sweep (bool): Whether or not to take a new VNA sweep
            N_steps (int): Number of bins per tone
            peak_prom_std (float): Peak height from surroundings, in noise std multiples. Uses larger of peak_prom_db or peak_prom_std.
            peak_prom_db (float): Peak height from surroundings, in db. Uses larger of peak_prom_db or peak_prom_std.
            peak_dis (int): Min distance between peaks [bins].
            peak_width (2-tuple of ints): Min/max peak width [bins].
            stitch (bool): Whether to stitch (comb discontinuities).
            remove_cont (bool): Whether to subtract the continuum.
            continuum_wn (int): Continuum filter cutoff frequency [Hz].
            remove_noise (bool): Whether to subtract noise.
            noise_wn (int): Noise filter cutoff frequency [Hz].
        Returns:
            output (arr of floats): Returns an array of the found detector frequencies
        '''
        # Evaluate kwargs
        stitch = True
        remove_cont = True
        continuum_wn = 300
        remove_noise = True
        noise_wn = 30000
        N_steps = 500
        for key, value in kwargs.items():
            if key == 'stitch':
                stitch = value
            elif key == 'N_steps':
                N_steps = value
            elif key == 'remove_cont':
                remove_cont = value
            elif key == 'continuum_wn':
                continuum_wn = value
            elif key == 'remove_noise':
                remove_noise = value
            elif key == 'noise_wn':
                noise_wn = value
        
        # Take VNA sweep if new_sweep = True or if one does not already exist
        vna_file = rfsoc_io.get_most_recent_file(self.vna_dir, f"{self.cfg_io['file_names']['vna_fname'][0]}*", self.output)
        if not vna_file.exists() or new_sweep:
            self.take_vna_sweep(**kwargs)

        rfsoc_io.send_msg('INFO', "Finding detectors from VNA sweep!", output = self.output)
        # Find resonators from VNA sweep
        rtn = self.rfsoc.findVnaResonators(com_to = self.com_to, peak_prom_db = peak_prom_db, peak_prom_std = peak_prom_std,
                                           peak_dis = int(peak_dis), peak_width_min = int(peak_width_min), peak_width_max = int(peak_width_max),
                                           stitch = stitch, stitch_bw = N_steps, stitch_sw = int(N_steps/5),
                                           remove_cont = remove_cont, continuum_wn = continuum_wn,
                                           remove_noise=remove_noise, noise_wn = noise_wn)
        rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
        rfsoc_io.send_msg('INFO', "Writing target comb using found resonators!", output = self.output)
        
        # Write target comb from VNA sweep
        rtn = self.rfsoc.writeTargCombFromVnaSweep(com_to = self.com_to)
        rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)

        rfsoc_io.send_msg('DEBUG', 'Creating custom comb using target comb!', output = self.output)
        # Write target comb to custom comb files
        rtn = self.rfsoc.createCustomCombFilesFromCurrentComb(com_to = self.com_to)
        rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)

        # Copy custom comb files in order to write to config
        self.cp_custom_comb_files(self.data_dir / 'tmp')

        # Get detector frequencies
        det_freqs = np.load(self.data_dir / "tmp" / "f_rf_tones_comb_cust.npy").tolist()
        rfsoc_io.send_msg('INFO', f"Found detector frequencies: {det_freqs} Hz", self.output)
        self.edit_config(self.cfg, 'found_num_detectors', len(det_freqs))
        self.edit_config(self.cfg, 'found_detector_freqs', det_freqs)

        return det_freqs

    def find_detectors_fine(self, new_sweep = True, **kwargs):
        '''
        Find detectors and place tones at their minima. Length of bins is 500,000 Hz / N_steps.
        
        Parameters:
            new_sweep (bool): Whether or not to take a new target sweep
            N_steps (int): Number of bins per tone
            bandwidth (float): Bandwidth of target sweep
        Returns:
            output (arr of floats): Returns an array of the found detector frequencies
        '''
        # Evaluate kwargs
        N_steps = self.cfg['rfsoc_tones']['N_steps']
        for key, value in kwargs.items():
            if key == 'N_steps':
                N_steps = value
        
        # Take VNA sweep if new_sweep = True or if one does not already exist
        targ_file = rfsoc_io.get_most_recent_file(self.targ_dir, f"{self.cfg_io['file_names']['targ_fname'][0]}*", self.output)
        if not targ_file.exists() or new_sweep:
            self.take_target_sweep(**kwargs)

        rfsoc_io.send_msg('INFO', "Finding detectors from target sweep!", output = self.output)
        # Find resonators from target sweep
        rtn = self.rfsoc.findTargResonators(com_to = self.com_to, stitch_bw = N_steps)
        rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
        rfsoc_io.send_msg('INFO', "Writing target comb using found resonators!", output = self.output)
        
        # Write target comb from target sweep
        rtn = self.rfsoc.writeTargCombFromTargSweep(com_to = self.com_to)
        rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
        self.write_config_tones(**kwargs) # Write custom tone powers and phis if provided

        rfsoc_io.send_msg('DEBUG', 'Creating custom comb using target comb!', output = self.output)
        # Write target comb to custom comb files
        rtn = self.rfsoc.createCustomCombFilesFromCurrentComb(com_to = self.com_to)
        rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)

        # Copy custom comb files in order to write to config
        self.cp_custom_comb_files(self.data_dir / 'tmp')

        # Get detector frequencies
        det_freqs = np.load(self.data_dir / "tmp" / "f_rf_tones_comb_cust.npy").tolist()
        rfsoc_io.send_msg('INFO', f"Found detector frequencies: {det_freqs} Hz", self.output)
        self.edit_config(self.cfg, 'found_num_detectors', len(det_freqs))
        self.edit_config(self.cfg, 'found_detector_freqs', det_freqs)

        return det_freqs
    
    def bias_detectors(self):
        pass

    def get_cable_delay(self, **kwargs):
        '''
        Get the cable delay of the readout chain using specified tones. 
        '''

        # Take target sweep
        targ_file = self.take_target_sweep(**kwargs)
        
    ####################
    # Helper Functions #
    ####################

    def write_custom_tones(self, **kwargs):
        '''
        Write custom tone power(s), frequency(ies), and/or phase(s) of tones.
        Parameters:
            tone_freqs: Float, array or file path of array containing custom tone frequencies
            tone_powers: Float, array or file path of array containing custom tone powers 
            tone_phis: Float, array or file path of array containing custom tone phases
        '''

        custom_comb_dir = Path(self.cfg_io['file_paths']['rfsoc_data_dir']) / 'custom_comb'

        # Evaluate passed key word arguments
        # ----------------------------------
        for key, value in kwargs.items():
            # Check if value is a file path
            try: 
                value = Path(value)
                # If value is a file path, ensure that it is a valid path
                if not value.exists():
                    rfsoc_io.send_msg('WARNING', f"{value} is not a valid file path!", self.output)
                    continue 
            # If not file path, assume value is a number or an array of numbers
            except:
                try: 
                    # Assume an array is passed and check if its non-empty and if length equals num_tones
                    if len(value) == 0:
                        rfsoc_io.send_msg('WARNING', f"'{value}' is an empty array, not writing to custom comb files!", self.output)
                        continue
                    elif not len(value) == self.cfg['rfsoc_tones']['num_tones']:
                        rfsoc_io.send_msg('WARNING', f"Length of '{value}' does not match number of tones specified!", self.output)
                        # continue 
                except TypeError:
                    try:
                        # Assume an number is passed and use the same number for all tones
                        value = value*np.ones(self.cfg['rfsoc_tones']['num_tones'])
                    except:
                        rfsoc_io.send_msg('WARNING', f"{value} is not a file path, array, or number!")
                        continue
                
                # Save the passed array
                np.save(self.data_dir / 'tmp' / key, value)

                # Change value so that it points to the file path of saved array
                value = self.data_dir / 'tmp' / f"{key}.npy"

            cmd = f"scp -q {value} {str(custom_comb_dir)}"

            # Match input array to correct custom parameter array
            if key == "tone_freqs":
                cmd += "/f_rf_tones_comb_cust.npy"
                rfsoc_io.send_msg('INFO', 'Modified tone frequencies!', self.output)
            elif key == "tone_powers":
                cmd += "/a_tones_comb_cust.npy"

                # Convert from dB to normal units if necessary
                if self.cfg['rfsoc_tones']['dB']:
                    np.save(value, utils.convert_from_dB(np.load(value)))
                rfsoc_io.send_msg('INFO', 'Modified tone powers!', self.output)
            elif key == "tone_phis":
                cmd += "/p_tones_comb_cust.npy"
                rfsoc_io.send_msg('INFO', 'Modified tone phases!', self.output)
            else:
                return
            
            # Copy the array with custom parameters onto the rfsoc board
            os.system(cmd)
            rfsoc_io.send_msg('DEBUG', f">>> Ran command: {cmd}", self.output)

        # Save new config file
        # ----------------
        # Copy custom comb files into local tmp folder
        self.cp_custom_comb_files(self.data_dir / 'tmp')

        # Edit config files custom comb parameters
        self.edit_config(self.cfg, 'tone_freqs', np.load(self.data_dir / "tmp" / "f_rf_tones_comb_cust.npy").tolist())
        self.edit_config(self.cfg, 'tone_powers', np.load(self.data_dir / "tmp" / "a_tones_comb_cust.npy").tolist())
        self.edit_config(self.cfg, 'tone_phis', np.load(self.data_dir / "tmp" / "p_tones_comb_cust.npy").tolist())
         
        # Write sweep comb based on custom parameters
        # -------------------------------------------
        rtn = self.rfsoc.writeCombFromCustomList(com_to = self.com_to)
        rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
        rfsoc_io.send_msg('INFO', 'Sucessfully wrote custom comb.', self.output)

        # Convert tone powers to dB if original config file had dB tone powers
        if self.cfg['rfsoc_tones']['dB']:
            self.edit_config(self.cfg, 'tone_powers', utils.convert_to_dB(self.cfg['rfsoc_tones']['tone_powers']).tolist())

    def write_config_tones(self, **kwargs):
        '''
        Write custom comb using comb parameters from config file. Config file parameters are 
        superseceded by key word argument parameters.

        Parameters:
            tone_freqs: Float, array or file path of array containing custom tone frequencies
            tone_powers: Float, array or file path of array containing custom tone powers 
            tone_phis: Float, array or file path of array containing custom tone phases
        '''
        # Get tone frequencies, powers, and phis from config file
        tone_freqs = self.cfg['rfsoc_tones']['tone_freqs']
        tone_powers = self.cfg['rfsoc_tones']['tone_powers']
        tone_phis = self.cfg['rfsoc_tones']['tone_phis']

        # Evaluate kwargs
        for key, value in kwargs.items():
            if key == 'tone_freqs':
                tone_freqs = value
            elif key == 'tone_powers':
                tone_powers = value
            elif key == 'tone_phis':
                tone_phis = value
        
        # Write custom comb
        self.write_custom_tones(tone_freqs = tone_freqs, tone_powers = tone_powers, tone_phis = tone_phis)

    def cp_custom_comb_files(self, dest_dir):
        '''
        Copies custom comb files from the RFSoC into specified local directory.

        Parameters:
            dest_dir (str): Directory to copy custom comb files into
        '''
        custom_comb_dir = Path(self.cfg_io['file_paths']['rfsoc_data_dir']) / 'custom_comb'
        dest_dir = Path(dest_dir)

        os.system(f'scp -q {custom_comb_dir / "f_rf_tones_comb_cust.npy"} {dest_dir/ "f_rf_tones_comb_cust.npy"}')
        os.system(f'scp -q {custom_comb_dir / "a_tones_comb_cust.npy"} {dest_dir/ "a_tones_comb_cust.npy"}')
        os.system(f'scp -q {custom_comb_dir / "p_tones_comb_cust.npy"} {dest_dir/ "p_tones_comb_cust.npy"}')

    def save_timestream(self, s21z, **kwargs):

        '''
        Save array of timestream data by breaking it into smaller files specified by max_size.
        '''
        from math import ceil
        
        max_file_size = eval(self.cfg_io['io']['max_file_size'])
        for key, value in kwargs.items():
            if key == 'max_file_size':
                max_file_size = value

        # Split timestream into multiple files if it exceeds the max file size
        tstream_size = sys.getsizeof(s21z) # Get file size in bytes
        tstream_len = np.shape(s21z)[1]
        trimmed_len = ceil(tstream_len/ceil(tstream_size/max_file_size))
        
        rfsoc_io.send_msg('DEBUG', f'Timestream size is {tstream_size/1e6} MB!', self.output)

        # Save Timestream
        fname = self.cfg_io["file_names"]["stream_fname"]
        fname = eval(f"f'{fname}'")

        tstream_files = list([])
        for i, j in enumerate(range(0, tstream_len, trimmed_len)):
            tstream_file = self.timestream_dir / f'{fname}_{i+1:03}.npy'
            np.save(tstream_file , s21z[:, j:j+trimmed_len])
            tstream_files.append(tstream_file)
            rfsoc_io.send_msg('DEBUG', f'Successfully saved timestream {i+1}!', self.output)
        
        return tstream_files
    
    def edit_config(self, cfg, key, value, append = False):
        '''
        Update key in specified configuration file with the specified value. Preferred method for internal 
        updates to config file. For external updates to config file, see "edit_main_config".

        Parameters:
            append: Whether to append a new key, value pair to config file if key is not found
        '''

        done = utils.edit_dic(cfg, key, value)
        if done:
            rfsoc_io.send_msg('DEBUG', f'Updated key "{key}" with value "{value}" in config file"!')
        elif append:   
            cfg[key] = value
            rfsoc_io.send_msg('DEBUG', f'Added key "{key}" with value "{value}" to config file!')
        else:
            rfsoc_io.send_msg('DEBUG', f'Failed to update key "{key}" with value "{value}" in config file!')        
        return done

    ###################
    # Getters/Setters #
    ###################
    def reload_config(self, cfg_path = "./config.yaml"):
        cfg = rfsoc_io.load_config(cfg_path)
        try:
            self.cfg, self.cfg_io = cfg
        except:
            self.cfg = cfg
            self.cfg_io = None
        return self.cfg, self.cfg_io

    def get_main_config(self):
        return self.cfg
    
    def get_io_config(self):
        return self.cfg_io

    def edit_main_config(self, key, value, append = False):
        '''
        Update key in main configuration file with the specified value. For use by 
        scripts implementing RFSoC_DAQ to update main config file without the need for it be passed
        between programs. For internal updates to config file, see "edit_config"
        '''
        return self.edit_config(self.cfg, key, value, append)
