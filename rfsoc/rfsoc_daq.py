#=================================#
# rfsoc_daq.py               2024 #
# Darshan Patel dp649@cornell.edu #
#=================================#

# Import Python modules
import os
import sys
from pathlib import Path
import numpy as np
import time

# Import local modules
import rfsoc_io
import utils

class R:
    '''
    Class for tuning and data acquisition of Microwave Kinetic Inductance Detectors (MKIDs) using
    Radio Frequency System on a Chip (RFSoC).
    '''

    def __init__(self, cfg_path = "/home/rfsoc/ccatkidlib/rfsoc/system_config.yaml"):
        '''
        Constructor for R. Creates directories for data storage, configures logger, and starts
        RFSoC PCS agent.

        Parameters:
            cfg_path (str): Path to system configuration file.
        '''

        # Current date in yyyy/mm/dd
        self.curr_date = time.strftime('%Y%m%d', time.gmtime())

        # Create session id from first ten digits of current time
        self.sess_id = str(time.time())[:10]

        # Create a global timestamp used for file naming and pairing
        self.timestamp = str(time.time()).split('.')[0]

        # Load config files
        self.load_system_config(cfg_path)
            
        # Setup logger
        self.log_dir = self.rfsoc_dirs[0].parent
        rfsoc_io.setup_logging(self.log_dir / self.io_cfg['io']['logging_fname'], self.io_cfg['io']['logging_level'], output = self.output)

        # Add paths to primecam_readout modules, PCS clients, and analysis code
        sys.path.append(self.io_cfg['file_paths']['primecam_readout'])
        sys.path.append(self.io_cfg['file_paths']['pcs_dir'])
        sys.path.append(self.io_cfg['file_paths']['analysis_dir'])
        rfsoc_io.send_msg('DEBUG', 'Finished appending file paths!', self.output)

        # Load local modules
        from ocs.ocs_client import OCSClient # Import PCS client module                                                  

        # Initialize PCS clients
        self.rfsoc = OCSClient(self.io_cfg['pcs_agents']['rfsoc_agent'], args=[]) 
        self.setup_drones(com_to = self.drone_list)
        rfsoc_io.send_msg('INFO', f'Initialized RFSoC agent. Communicating with drones: {self.drone_list}!', self.output)

        # Initialize None instance of TimeStream class
        self.streamer = None

        # Set NCLO frequency of RFSoC
        for com_to, cfg in zip(self.drone_list, self.drone_cfg):
            rtn = self.rfsoc.setNCLO(com_to = com_to, f_lo = cfg['tones']['NCLO'])
            rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
        rfsoc_io.send_msg('INFO', f"Set NCLO to {[cfg['tones']['NCLO'] for cfg in self.drone_cfg]} MHz for drones: {self.drone_list}!", self.output)

        # Set drone attenuators
        self.set_atten()

        # Save init config
        rfsoc_io.save_config(self.log_dir / f'{self.timestamp}_init_config_ext.yaml', self.ext_cfg, self.save_cfg)
        rfsoc_io.save_config(self.log_dir / f'{self.timestamp}_init_config_io.yaml', self.io_cfg, self.save_cfg)
        for rfsoc_dir, cfg in zip(self.rfsoc_dirs, self.drone_cfg):
            rfsoc_io.save_config(rfsoc_dir / f'{self.timestamp}_init_config_drone.yaml', cfg, self.save_cfg)

    ###################
    # Setup Functions #
    ###################

    def setup_drones(self, com_to, **kwargs): 
        '''
        Setup drones in drone_list. 
        '''
        import drone_control 

        parallel = self.io_cfg['parallel'] # Whether to run board commands in parallel (stops drones not in drone_list)
        restart = self.io_cfg['restart'] # Whether to restart already running drones

        for key, value in kwargs.items():
            if key == 'parallel':
                parallel = value
            elif key == 'restart':
                restart = value

        # Edit master_drone_config.yaml in primecam_readout to match drone_list
        master_drone_file = self.io_cfg['file_paths']['master_drone_list']
        master_drone = rfsoc_io.load_config(master_drone_file)

        # Loop through all boards
        for board in self.board_list:
            # Loop through all drones
            for i in range(4):
                # Set to_run for drone (True if supposed to be running, False otherwise)
                com = board + f'.{i + 1}'
                master_drone[com]['ip'] = self.io_cfg['boards'][f'b{board}']['board_ip']
                master_drone[com]['to_run'] = com in com_to
        # Save edited config
        rfsoc_io.save_config(master_drone_file, master_drone)

        wait = False

        # Loop through all boards
        for board in self.board_list:
            # Loop through all drones
            for i in range(4):
                ip, to_run, running = drone_control.statusDrone(int(board), i + 1) # Get drone status

                # Stop, Start, or Restart drone as appropriate
                if running and not to_run and parallel:
                    ret = drone_control.stopDrone(int(board), i + 1)
                elif to_run and not running:
                    ret = drone_control.startDrone(int(board), i + 1)
                    wait = True
                elif to_run and running and restart:
                    ret = drone_control.restartDrone(int(board), i + 1)
                    wait = True

        # Wait if any of the drones were started/restarted
        if wait: rfsoc_io.wait(20, output = self.output, desc = "Waiting for drones to start")
    
    def load_system_config(self, cfg_path = "/home/rfsoc/ccatkidlib/rfsoc/system_config.yaml"):
        # Read configuration file
        cfg = rfsoc_io.load_config(cfg_path)
        try:
            self.ext_cfg, self.io_cfg = cfg
        except:
            print("System config must contain two config files (an external config and an io config)! Reference example system config file.")

        # Get list of drones
        self.drone_list = self.io_cfg['drone_list']

        # Get number of drones and convert to a list if only a single drone is passed
        try:
            self.drone_num = len(self.drone_list)
        except:
            self.drone_list = [self.drone_list]
            self.drone_num = len(self.drone_list)

        # Get list of boards used
        bids = set()
        for drone in self.drone_list:
            split_str = drone.split('.')
            bids.add(split_str[0])
            # Replace board strings with all bid.drid IDs
            if len(split_str) == 1:
                self.drone_list.remove(drone)
                for i in range(4):
                    self.drone_list.append(drone + f'.{i + 1}')
        self.board_list = list(bids)

        # Load drone configuration files
        self.drone_cfg = []
        board_cfgs = self.io_cfg['boards']
        for board in board_cfgs:
            if board[1:] in self.board_list:
                # Load all drone configuration files
                self.drone_cfg.append(rfsoc_io.load_config(board_cfgs[board]['drone_cfg']))

                # Keep only the configuration files corresponding to drones in self.drone_list
                drones = self.drone_cfg[-1]
                num = len(drones)
                for i in range(num):
                    if not (drones[num - i - 1]['com_to'] in self.drone_list):
                        drones.pop(num - i - 1 )
        
        # Flatten list of drone configs to match drone_list
        self.drone_cfg = [cfg for board in self.drone_cfg for cfg in board]
        # Assign commonly used parameters to variables
        self.output = self.io_cfg['io']['terminal_output']

        self.save_cfg = self.io_cfg['io']['save_config_copy']
        self.save_data = self.io_cfg['io']['save_data']

        self.data_dir = Path(self.io_cfg['file_paths']['data_dir'])
        self.drone_dir = Path(self.io_cfg['file_paths']['drone_dir'])
        self.tmp_data_dir = Path(self.io_cfg['file_paths']['tmp_data_dir'])

        # Save session ID to config files
        self. _edit_config(self.ext_cfg, 'sess_id', self.sess_id)
        self. _edit_config(self.io_cfg, 'sess_id', self.sess_id)
        for cfg in self.drone_cfg:
            self. _edit_config(cfg, 'sess_id', self.sess_id)

        new_dir_paths = rfsoc_io.create_book(self.curr_date, self.sess_id, self.drone_list, self.data_dir, output = self.output)
        self.rfsoc_dirs, self.targ_dirs, self.timestream_dirs, self.vna_dirs = new_dir_paths

    def set_atten(self, com_to = None, drive = None, sense = None):
        '''
        Set drive/sense attenutors. 

        Keyword Arguments:
            drive (float): Value of drive (DAC) attenuation in dB (must be between 0 and 31.75)
            sense (float): Value of sense (ADC) attenuation in dB (must be between 0 and 31.75)
        '''

        # Specify which attenuators to change
        if com_to is None:
            com_to = self.drone_list

        # Set drive attenuation
        self._set_atten(com_to = com_to, direction = 'drive', atten = drive)

        # Set sense attenuation
        self._set_atten(com_to = com_to, direction = 'sense', atten = sense)

    def write_config_tones(self, com_to, **kwargs):
        '''
        Write custom comb using comb parameters from config file. Config file parameters are
        superseceded by key word argument parameters.

        Parameters:
            tone_freqs: Float, array or file path of array containing custom tone frequencies
            tone_powers: Float, array or file path of array containing custom tone powers
            tone_phis: Float, array or file path of array containing custom tone phases
        '''

        tone_freqs = []
        tone_powers = []
        tone_phis = []
        for com in com_to:
            ind = self.drone_list.index(com)

            # Get tone frequencies, powers, and phis from config file
            tone_freqs.append(self.drone_cfg[ind]['tones']['tone_freqs'])
            tone_powers.append(self.drone_cfg[ind]['tones']['tone_powers'])
            tone_phis.append(self.drone_cfg[ind]['tones']['tone_phis'])

        # Evaluate kwargs
        for key, value in kwargs.items():
            if key == 'tone_freqs':
                tone_freqs = self._parse_args(com_to, value)
            elif key == 'tone_powers':
                tone_powers = self._parse_args(com_to, value)
            elif key == 'tone_phis':
                tone_phis = self._parse_args(com_to, value)

        # Write custom comb
        for com, freq, power, phi in zip(com_to, tone_freqs, tone_powers, tone_phis):
            self._write_custom_tones(com, tone_freqs = freq, tone_powers = power, tone_phis = phi)
        
        # Write sweep comb based on custom parameters
        # -------------------------------------------
        if self.io_cfg['parallel']:
            for board in self.board_list:
                rtn = self.rfsoc.writeCombFromCustomList(com_to = board)
                rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
                rfsoc_io.send_msg('INFO', f'Sucessfully wrote custom comb for board {board}!', self.output)
        else:
            for com in com_to:
                rtn = self.rfsoc.writeCombFromCustomList(com_to = com)
                rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
                rfsoc_io.send_msg('INFO', f'Sucessfully wrote custom comb for drone {com}!', self.output)

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

        #N_steps = 500
        #for key, value in kwargs.items():
            #if key == 'NCLO':
                #rtn = self.rfsoc.setNCLO(com_to = self.com_to, f_lo = value)
                #rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)

                #self.ext_cfg['rfsoc_tones']['drone_NCLO'] = int(value)
                #rfsoc_io.send_msg('INFO', f'Set NCLO to {value} MHz!', self.output)
            #elif key == 'N_steps':
            #    N_steps = value

        # Write VNA comb

        com_to = self._get_com_to(**kwargs)

        rfsoc_io.send_msg('INFO', 'Writing new VNA comb!', self.output)
        if self.io_cfg['parallel'] and len(com_to) > 1:
            for board in self.board_list:
                rtn = self.rfsoc.writeNewVnaComb(com_to = board)
                rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
            rfsoc_io.send_msg('INFO', f'Successfully wrote new VNA comb for boards {self.board_list}!', self.output)

        else:
            for com in com_to:
                rtn = self.rfsoc.writeNewVnaComb(com_to = com)
                rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
            rfsoc_io.send_msg('INFO', f'Successfully wrote new VNA comb for drones {com_to}!', self.output)
        
        # Take VNA sweep
        time.sleep(3) # Wait before taking sweep (not waiting can affect sweep quality, unsure if this is still the case: need to test) 
        
        # Get current timestamp
        self.timestamp = str(time.time()).split('.')[0]
        if self.io_cfg['parallel'] and len(com_to) > 1:
            for board in self.board_list:
                rfsoc_io.send_msg('INFO', 'Taking VNA sweep!', self.output)
                rtn = self.rfsoc.vnaSweep(com_to = board)
                rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
                rfsoc_io.send_msg('INFO', f'Finished taking VNA sweep for board {board}!', self.output)
        else:
            for com in com_to:
                rfsoc_io.send_msg('INFO', 'Taking VNA sweep!', self.output)
                rtn = self.rfsoc.vnaSweep(com_to = com)
                rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
                rfsoc_io.send_msg('INFO', f'Finished taking VNA sweep for drone {com}!', self.output)

        # Save ext config
        self.ext_cfg = rfsoc_io.save_config(self.log_dir / f"{self.timestamp}_vna_config_ext.yaml", self.ext_cfg, self.save_cfg)

        # Make sure all VNA sweeps have finished
        time.sleep(3)
        vna_files, vna_paths = [], []
        # Save VNA sweep
        for com in com_to:
            ind = self.drone_list.index(com)
            bid, drid = com.split('.')
            bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
            key = self.io_cfg['file_paths']['RSA_key']
            path = self.drone_dir / f'drone{drid}' / 'vna'

            with rfsoc_io.get_connection(bip, key) as c:
                vna_file = rfsoc_io.get_most_recent_file_board(c, path, file_identifier = "s21_vna")

                if not rfsoc_io.path_exists(c, vna_file):
                    rfsoc_io.send_msg('ERROR', "VNA sweep was unsuccessful! No files saved.", self.output)
                    vna_files.append(None)
                    continue

                vna_files.append(vna_file)
                if self.save_data:
                    fname = self.io_cfg["file_names"]["vna_fname"]
                    fname = eval(f"f'{fname}'") + '.npy'
                    vna_path = self.vna_dirs[ind] / fname
                    c.get(str(vna_file), str(vna_path))
            
            rfsoc_io.send_msg('DEBUG', f'>>> Successfully copied VNA file from drone {com}!', self.output)
            vna_paths.append(vna_path)

            # Save drone config
            self.drone_cfg[ind] = rfsoc_io.save_config(self.rfsoc_dirs[ind] / f"{self.timestamp}_vna_config_drone.yaml", self.drone_cfg[ind], self.save_cfg)
            
            
        if self.save_data:
            return vna_paths
        else:
            return vna_files

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

        com_to = self._get_com_to(**kwargs)

        write_tones = True
        # Evaluate kwargs
        for key, value in kwargs.items():
            #if key == 'bandwidth':
            #    self.ext_cfg['rfsoc_tones']['bandwidth'] = value
            #elif key == 'N_steps':
            #    self.ext_cfg['rfsoc_tones']['N_steps'] = value
            if key == 'write_tones':
                write_tones = value

        # Write new target sweep comb
        if write_tones:
            self.write_config_tones(com_to, **kwargs)
            time.sleep(3) # Wait before taking sweep (not waiting can affect sweep quality)
        
        # Get timestamp
        self.timestamp = str(time.time()).split('.')[0]
        if self.io_cfg['parallel']:
            for board in self.board_list:
                rfsoc_io.send_msg('INFO', 'Taking target sweep!', output = self.output)
                rtn = self.rfsoc.targetSweep(com_to = board)
                rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
                rfsoc_io.send_msg('INFO', f'Finished taking target sweep for board {board}!', self.output)
        else:
            for com in com_to:
                rfsoc_io.send_msg('INFO', 'Taking target sweep!', output = self.output)
                rtn = self.rfsoc.targetSweep(com_to = com)
                rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
                rfsoc_io.send_msg('INFO', f'Finished taking target sweep for drone {com}!', self.output)

        # Get current timestamp and save ext config
        self.ext_cfg = rfsoc_io.save_config(self.log_dir / f"{self.timestamp}_target_config_ext.yaml", self.ext_cfg, self.save_cfg)

        # Make sure all target sweeps have finished
        time.sleep(3)
        targ_files, targ_paths = [], []

        # Save target sweep
        for com in com_to:
            ind = self.drone_list.index(com)
            bid, drid = com.split('.')
            bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
            key = self.io_cfg['file_paths']['RSA_key']
            path = self.drone_dir / f'drone{drid}' / 'targ'

            with rfsoc_io.get_connection(bip, key) as c:
                targ_file = rfsoc_io.get_most_recent_file_board(c, path, file_identifier = "s21_targ")

                if not rfsoc_io.path_exists(c, targ_file):
                    rfsoc_io.send_msg('ERROR', "Target sweep was unsuccessful! No files saved.", self.output)
                    targ_files.append(None)
                    continue

                targ_files.append(targ_file)
                if self.save_data:
                    fname = self.io_cfg["file_names"]["targ_fname"]
                    fname = eval(f"f'{fname}'") + '.npy'
                    targ_path = self.targ_dirs[ind] / fname
                    c.get(str(targ_file), str(targ_path))
            
            rfsoc_io.send_msg('DEBUG', f'>>> Successfully copied target file from drone {com}!', self.output)
            targ_paths.append(targ_path)

            # Save drone config
            self.drone_cfg[ind] = rfsoc_io.save_config(self.rfsoc_dirs[ind] / f"{self.timestamp}_targ_config_drone.yaml", self.drone_cfg[ind], self.save_cfg)
            
        if self.save_data:
            return targ_paths
        else:
            return targ_files

    def take_timestream(self, t_sec, **kwargs):
        '''
        Take timestream data using RFSoC.

        Parameters:
            time (int): Length of timestream in seconds
            write_tones (bool): Whether to re-write comb
            tone_freqs: Frequencies at which to place tones
            tone_powers: Readout power of tones
            tone_phis: Phase of tones
        Return:
            output (list of str): Timestream file paths if save_data = True
        Returns:
            output: Return complex S21 timestream data in array
        '''
        from rfsoc_timestream import Streamer

        # Attempt to initialize timestream client
        exit = False
        while self.streamer is None:
            try:
                self.streamer = Streamer(self.io_cfg['udp_ip'],
                                    self.io_cfg['udp_port'])
            except:
                if exit:
                    rfsoc_io.send_msg('CRITICAL', 'The currently running process was not terminated. Safely exiting!', self.output)
                    sys.exit()
                rfsoc_io.send_msg('CRITICAL', 'A process is already bound to the timestream UDP port! Would you like to terminate this process?', self.output)
                os.system(f"fuser -ki -n udp {self.io_cfg['udp_port']}")
                exit = True


        com_to = self._get_com_to(**kwargs)

        # Parse key word arguments
        write_tones = True
        save_data = self.save_data
        for key, value in kwargs.items():
            if key == 'write_tones':
                write_tones = value
            elif key == 'save_data':
                save_data = value

        # Write new tones
        if write_tones:
            self.write_config_tones(com_to, **kwargs)
            time.sleep(3)

        rfsoc_io.send_msg('INFO', f'Taking {t_sec} seconds of timestream data!', self.output)
        
        # Turn off all currently running timestreams
        ret = self.rfsoc.timestreamOn(on = False)
        time.sleep(1) # Wait to ensure that all timestreams were turned off

        stream_paths = []
        timestreams = []
        self.timestamp = str(time.time()).split('.')[0]
        if self.io_cfg['parallel'] and len(com_to) > 1:
            for board in self.board_list:
                # Turn on timestreams
                ret = self.rfsoc.timestreamOn(com_to = board, on = True)

                inds = [self.drone_list.index(com) for com in com_to if com.split('.')[0] == board]


                # Get total number of packets to capture
                N_packets = len(inds)*int(eval(self.io_cfg['boards'][f'b{board}']['sampling_freq'])*t_sec)

                # Take timestream
                data, auxs, ips, ports = self.streamer.take_timestream(N_packets)

                # Turn off timestream
                ret = self.rfsoc.timestreamOn(com_to = board, on = False)

                rfsoc_io.send_msg('INFO', f'Finished taking timestream data for board {board}!', self.output)

                # Sort the data by drone based on source IP address
                drone_ips = [self.drone_cfg[ind]['udp_source_ip'] for ind in inds]
                drone_data = [ [] for _ in range(len(inds))]
                for dat, aux, ip in zip(data, auxs, ips):
                    drone_ind = drone_ips.index(str(ip))
                    drone_data[drone_ind].append(np.append(aux[3], np.array(dat)))

                # Save the timestreams for each drone
                for data, ind in zip(drone_data, inds):
                    # Convert data into I, Q
                    data = np.array(data)
                    I, Q = data[:,1::2].T, data[:,2::2].T
                    ts = data[:, 0].T

                    # Combine I and Q into complex S21 data
                    s21z = I + 1j * Q

                    # Combine array of times and complex S21z data
                    timestream_data = np.append([ts], s21z, axis = 0)

                    if save_data:
                        self.drone_cfg[ind] = rfsoc_io.save_config(self.rfsoc_dirs[ind] / f"{self.timestamp}_timestream_config_drone.yaml", self.drone_cfg[ind], self.save_cfg)
                        stream_paths.append(self._save_timestream(ind, timestream_data))
                    else:
                        timestreams.append(timstream_data)
        else:
            for com in com_to:
                ind = self.drone_list.index(com)

                # Get number of packets per drone
                N_packets = int(eval(self.io_cfg['boards'][f"b{com.split('.')[0]}"]['sampling_freq'])*t_sec)

                # Turn on timestream
                ret = self.rfsoc.timestreamOn(com_to = com, on = True)

                # Take timestream
                data, aux, ips, ports = self.streamer.take_timestream(N_packets)

                # Turn off timestream
                ret = self.rfsoc.timestreamOn(com_to = com, on = False)

                rfsoc_io.send_msg('INFO', f'Finished taking timestream data for drone {com}!', self.output)

                # Convert data into I, Q
                data = np.array(data)
                I, Q = data[:,0::2].T, data[:,1::2].T

                # Combine I and Q into complex S21 data
                s21z = I + 1j * Q

                # Get times
                ts = np.array(aux)[:, 3]
                
                # Combine array of times and complex S21z data
                timestream_data = np.append([ts], s21z, axis = 0)
                if save_data:
                    self.drone_cfg[ind] = rfsoc_io.save_config(self.rfsoc_dirs[ind] / f"{self.timestamp}_timestream_config_drone.yaml", self.drone_cfg[ind], self.save_cfg)
                    stream_paths.append(self._save_timestream(ind, timestream_data))
                else:
                    timstreams.append(timestream_data)
        
        # Save ext config
        self.ext_cfg = rfsoc_io.save_config(self.log_dir / f"{self.timestamp}_timestream_config_ext.yaml", self.ext_cfg, self.save_cfg)

        if save_data:
            return stream_paths
        else:
            return timestreams

    ####################
    # Tuning Functions #
    ####################

    def find_detectors(self, new_sweep = True, **kwargs):
        '''
        Find detectors and place tones at their minima. Length of bins is 500,000 Hz / N_steps.
        Default parameters work well for N_steps = 500 and should be scaled appropriately if N_steps is changed.

        Parameters:
            new_sweep (bool): Whether or not to take a new VNA sweep
            peak_prom_std (float): Peak height from surroundings, in noise std multiples. Uses larger of peak_prom_db or peak_prom_std.
            peak_prom_db (float): Peak height from surroundings, in db. Uses larger of peak_prom_db or peak_prom_std.
            peak_dis (int): Min distance between peaks [bins].
            width_min (int): Min peak width [bins]
            width_max (int): Max peak width [bins]
            stitch (bool): Whether to stitch (comb discontinuities).
            stitch_sw (int): Amount of bins on each end to use for stitching [bins]
            remove_cont (bool): Whether to subtract the continuum.
            continuum_wn (int): Continuum filter cutoff frequency [Hz].
            remove_noise (bool): Whether to subtract noise.
            noise_wn (int): Noise filter cutoff frequency [Hz].
        '''

        com_to = self._get_com_to(**kwargs)

        # Evaluate kwargs
        peak_prom_std = self._get_drone_args(com_to, ['det_find', 'peak_prom_std'])
        peak_prom_db = self._get_drone_args(com_to, ['det_find', 'peak_prom_db'])
        peak_dis = self._get_drone_args(com_to, ['det_find', 'peak_dis'])
        width_min = self._get_drone_args(com_to, ['det_find', 'width_min'])
        width_max = self._get_drone_args(com_to, ['det_find', 'width_max'])
        stitch = self._get_drone_args(com_to, ['det_find', 'stitch'])
        stitch_sw = self._get_drone_args(com_to, ['det_find', 'stitch_sw'])
        remove_cont = self._get_drone_args(com_to, ['det_find', 'remove_cont'])
        continuum_wn = self._get_drone_args(com_to, ['det_find', 'continuum_wn'])
        remove_noise = self._get_drone_args(com_to, ['det_find', 'remove_noise'])
        noise_wn = self._get_drone_args(com_to, ['det_find', 'noise_wn'])

        # Parse key word arguments
        for key, value in kwargs.items():
            if key == 'peak_prom_std':
                peak_prom_std = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "peak_prom_std", peak_prom_std)
            elif key == 'peak_prom_db':
                peak_prom_db = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "peak_prom_db", peak_prom_db)
            elif key == 'peak_dis':
                peak_dis = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "peak_dis", peak_dis)
            elif key == 'width_min':
                width_min = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "width_min", width_min)
            elif key == 'width_max':
                width_max = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "width_max", width_max)
            elif key == 'stitch':
                stitch = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "stitch", stitch)
            elif key == 'stitch_sw':
                stitch_sw = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "stitch_sw", stitch_sw)
            elif key == 'continuum_wn':
                continuum_wn = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "continuum_wn", continuum_wn)
            elif key == 'remove_cont':
                remove_cont = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "remove_cont", remove_cont)
            elif key == 'remove_noise':
                remove_noise = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "remove_noise", remove_noise)
            elif key == 'noise_wn':
                noise_wn = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "noise_wn", noise_wn)

        # Take VNA sweep if new_sweep = True or if one does not already exist
        to_sweep = []
        for com in com_to:
            ind = self.drone_list.index(com)
            # Check if a VNA sweep was taken in the last day
            vna_file = rfsoc_io.get_most_recent_file(self.vna_dirs[ind], f"{self.io_cfg['file_names']['vna_fname'][0]}*", self.output, time_past=24*3600)
            if not vna_file.exists() or new_sweep:
                to_sweep.append(com)
        if len(to_sweep) != 0:
            self.take_vna_sweep(com_to = to_sweep)
        
        rfsoc_io.send_msg('INFO', "Finding detectors from VNA sweep!", output = self.output)
        for i, com in enumerate(com_to):
            # Find resonators from VNA sweep
            rtn = self.rfsoc.findVnaResonators(com_to = com, peak_prom_std = peak_prom_std[i], peak_prom_db = peak_prom_db[i], peak_dis = peak_dis[i],
                                                width_min = width_min[i], width_max = width_max[i], stitch = stitch[i], stitch_sw = stitch_sw[i],
                                                remove_cont = remove_cont[i], continuum_wn = continuum_wn[i], remove_noise = remove_noise[i], noise_wn = noise_wn[i])
            rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
        
        rfsoc_io.send_msg('INFO', "Writing target comb using found resonators!", output = self.output)
        if self.io_cfg['parallel']:
            for board in self.board_list:
                start_time = time.time()
                # Write target comb using found resonators
                rtn = self.rfsoc.writeTargCombFromVnaSweep(com_to = board)
                rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
                rfsoc_io.send_msg('DEBUG', 'Creating custom comb using target comb!', output = self.output)

                rfsoc_io.wait(120 - int((time.time()-start_time)), output = self.output, desc = "Writing custom target comb") # Wait to make sure all combs have been written (large time difference depending on number of tones)
                # Write target comb to custom comb files
                rtn = self.rfsoc.createCustomCombFilesFromCurrentComb(com_to = board)
                rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
        else:
            for com in com_to:
                # Write target comb using found resonators
                rtn = self.rfsoc.writeTargCombFromVnaSweep(com_to = com)
                rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)
                rfsoc_io.send_msg('DEBUG', 'Creating custom comb using target comb!', output = self.output)
                
                # Write target comb to custom comb files
                rtn = self.rfsoc.createCustomCombFilesFromCurrentComb(com_to = com)
                rfsoc_io.send_msg('DEBUG', f'{rtn}', self.output)

        found_nums = []
        for com in com_to:
            ind = self.drone_list.index(com)
            # Copy custom comb files in order to write to config
            bid, drid = com.split('.')
            bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
            key = self.io_cfg['file_paths']['RSA_key']

            custom_freq_dir = Path(self.drone_dir / f'drone{drid}' / 'custom_comb' / self.io_cfg['file_names']['cust_comb_freq'])
            with rfsoc_io.get_connection(bip, key) as c:
                det_freqs = rfsoc_io.load_array_board(c, custom_freq_dir)

            found_num = len(det_freqs)
            found_nums.append(found_num)

            rfsoc_io.send_msg('DEBUG', f"Found detector frequencies for drone {com}: {det_freqs} Hz", self.output)
            rfsoc_io.send_msg('INFO', f"Found {found_num} detectors for drone {com}!", self.output)
    
            self. _edit_config(self.drone_cfg[ind], 'found_num_detectors', found_num)
            self. _edit_config(self.drone_cfg[ind], 'found_detector_freqs', det_freqs)
        return found_nums

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
        N_steps = self.ext_cfg['rfsoc_tones']['N_steps']
        for key, value in kwargs.items():
            if key == 'N_steps':
                N_steps = value

        # Take VNA sweep if new_sweep = True or if one does not already exist
        targ_file = rfsoc_io.get_most_recent_file(self.targ_dir, f"{self.io_cfg['file_names']['targ_fname'][0]}*", self.output)
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
        self. _edit_config(self.ext_cfg, 'found_num_detectors', len(det_freqs))
        self. _edit_config(self.ext_cfg, 'found_detector_freqs', det_freqs)

        return det_freqs

    def bias_detectors(self):
        pass

    def get_cable_delay(self, **kwargs):
        '''
        Get the cable delay of the readout chain using specified tones.
        '''

        # Take target sweep
        targ_file = self.take_target_sweep(**kwargs)
    
    #############################
    # Internal Helper Functions #
    #############################

    def _write_custom_tones(self, com, **kwargs):
        '''
        Write custom tone power(s), frequency(ies), and/or phase(s) of tones.
        Parameters:
            tone_freqs: Float, array or file path of array containing custom tone frequencies
            tone_powers: Float, array or file path of array containing custom tone powers
            tone_phis: Float, array or file path of array containing custom tone phases
        '''

        ind = self.drone_list.index(com)
        bid, drid = com.split('.')
        bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
        key = self.io_cfg['file_paths']['RSA_key']
        custom_comb_dir = self.drone_dir / f'drone{drid}' / 'custom_comb'

        # Evaluate passed key word arguments
        # ----------------------------------
        with rfsoc_io.get_connection(bip, key) as c:
            tone_num = None
            for key, value in kwargs.items():
                # Check if value is a file path
                try:
                    value = Path(value)
                    # If value is a file path, ensure that it is a valid path
                    if not value.exists():
                        rfsoc_io.send_msg('WARNING', f"{value} is not a valid file path for drone {com}!", self.output)
                        continue
                    
                    value = np.load(value)
                # If not file path, assume value is a number or an array of numbers
                except:
                    try:
                        # Assume an array is passed and check if its non-empty
                        if len(value) == 0:
                            rfsoc_io.send_msg('WARNING', f"'{value}' is an empty array, not writing to custom comb file for drone {com}!", self.output)
                            continue
                        elif tone_num is None:
                            tone_num = len(value)
                        elif len(value) != tone_num:
                            rfsoc_io.send_msg('WARNING', f"The length of one or more of the custom tone arrays do not match! Not writing custom comb for drone {com}.", self.output)
                            return
                    except TypeError:
                        try:
                            # Assume a number is passed and use the same number for all tones
                            if tone_num is None: 
                                tone_num = eval(self.drone_cfg[ind]['tones']['num_tones'])

                            # If num_tones is None. Try to use found number of detectors
                            if tone_num is None:
                                rfsoc_io.send_msg('DEBUG', 'num_tones is not specified in config. Attempting to use found_num_detectors.', self.output)
                                tone_num = self.drone_cfg[ind]['det_config']['found_num_detectors']
                            else:
                                rfsoc_io.send_msg('WARNING', f'The number of tones needs to be specified through num_tones or found_num_detectors for drone {com} alongside given value: {value}.')
                                continue

                            value = value*np.ones(tone_num)
                        except:
                            rfsoc_io.send_msg('WARNING', f"{value} is not a file path, array, or number! Not writing custom comb file for drone {com}.")
                            continue

                # Match input array to correct custom parameter array
                if key == "tone_freqs":
                    path = custom_comb_dir / self.io_cfg['file_names']['cust_comb_freq']
                    rfsoc_io.send_msg('DEBUG', f'Modified tone frequencies for drone {com}!', self.output)
                elif key == "tone_powers":
                    path = custom_comb_dir / self.io_cfg['file_names']['cust_comb_amp']

                    # Convert from dB to normal units if necessary
                    if self.drone_cfg[ind]['tones']['dB']:
                        value = utils.convert_from_dB(value)
                    
                    rfsoc_io.send_msg('DEBUG', f'Modified tone powers  for drone {com}!', self.output)
                elif key == "tone_phis":
                    path = custom_comb_dir / self.io_cfg['file_names']['cust_comb_phi']
                    rfsoc_io.send_msg('DEBUG', f'Modified tone phases  for drone {com}!', self.output)
                else:
                    continue

                # Copy the array with custom parameters onto the rfsoc board
                rfsoc_io.save_array_board(c, path, np.array(value).tolist())
                rfsoc_io.send_msg('DEBUG', f"Saved custom comb for drone {com}!", self.output)

            # Edit config file sweep parameters
            tone_freqs = rfsoc_io.load_array_board(c, custom_comb_dir / self.io_cfg['file_names']['cust_comb_freq'])
            self._edit_config(self.drone_cfg[ind], 'tone_freqs', tone_freqs)
            
            tone_powers = rfsoc_io.load_array_board(c, custom_comb_dir / self.io_cfg['file_names']['cust_comb_amp'])
            # Convert tone powers to dB if original config file had dB tone powers
            if self.drone_cfg[ind]['tones']['dB']:
                tone_powers = utils.convert_to_dB(tone_powers).tolist()
            
            self._edit_config(self.drone_cfg[ind], 'tone_powers', tone_powers)
            
            tone_phis = rfsoc_io.load_array_board(c, custom_comb_dir / self.io_cfg['file_names']['cust_comb_phi'])
            self._edit_config(self.drone_cfg[ind], 'tone_phis', tone_phis)

            num_tones = len(tone_freqs)
            self._edit_config(self.drone_cfg[ind], 'num_tones', num_tones)

    def _save_timestream(self, ind, data, **kwargs):

        '''
        Save array of timestream data by breaking it into smaller files specified by max_size.
        '''
        from math import ceil

        max_file_size = eval(self.io_cfg['io']['max_file_size'])
        for key, value in kwargs.items():
            if key == 'max_file_size':
                max_file_size = value

        # Split timestream into multiple files if it exceeds the max file size
        tstream_size = sys.getsizeof(data) # Get file size in bytes
        tstream_len = np.shape(data)[1]
        trimmed_len = ceil(tstream_len/ceil(tstream_size/max_file_size))

        rfsoc_io.send_msg('DEBUG', f'Timestream size is {tstream_size/1e6} MB!', self.output)

        # Save Timestream
        fname = self.io_cfg["file_names"]["stream_fname"]
        fname = eval(f"f'{fname}'")

        tstream_files = list([])
        timestream_dir = self.timestream_dirs[ind]
        for i, j in enumerate(range(0, tstream_len, trimmed_len)):
            tstream_file = timestream_dir / f'{fname}_{i+1:03}.npy'
            np.save(tstream_file , data[:, j:j+trimmed_len])
            tstream_files.append(tstream_file)
            rfsoc_io.send_msg('DEBUG', f'Successfully saved timestream {i+1}!', self.output)

        return tstream_files

    def _edit_config(self, cfg, key, value, append = False):
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

    def _parse_args(self, com_to, arg):
        args = None
        try:
            if len(arg) == len(com_to):
                args = arg
            else:
                rfsoc_io.send_msg('WARNING', f'{arg} is not a valid argument. Must be a single value or match the length of {com_to}!', self.output)
                return None
        except:
            try:
                args = [arg] * len(com_to)
            except:
                rfsoc_io.send_msg('WARNING', f'{arg} is not a valid argument. Must be a single value or match the length of {com_to}!', self.output)
                return None
        return args

    def _get_drone_args(self, com_to, arg):
        inds = [self.drone_list.index(com) for com in com_to]
        return [utils.dict_get(self.drone_cfg[ind], arg) for ind in inds]

    def _set_drone_args(self, com_to, key, args):
        for com, arg in zip(com_to, args):
            ind = self.drone_list.index(com)
            self._edit_config(self.drone_cfg[ind], key, arg)
    
    def _get_com_to(self, **kwargs):
        com_to = self.drone_list

        for key, value in kwargs.items():
            if key == 'com_to':
                com_to = value
                if not isinstance(com_to, list):
                    com_to = [com_to]
        
        self.setup_drones(com_to = com_to, restart = False)
        
        return com_to

    def _set_atten(self, com_to = None, direction = None, atten = None):
        # Set attenuation 
        attens = self._get_drone_args(com_to, ['atten', f'{direction}'])
        if atten is not None: 
            attens = self._parse_args(com_to, atten)
            if attens is None:
                return None
            self.set_drone_args(com_to, direction, attens)
            
        for com, att in zip(com_to, attens):
            self.rfsoc.setAtten(com_to = com, direction = direction, atten = att)
        rfsoc_io.send_msg('INFO', f'Successfully set {direction} attenuation to {attens} for drones {com_to}!', self.output)
