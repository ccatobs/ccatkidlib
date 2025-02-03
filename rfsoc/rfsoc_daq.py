#=================================#
# rfsoc_daq.py               2025 #
# Darshan Patel dp649@cornell.edu #
#=================================#

# Import Python modules
import os
import sys
import ast
import pickle
from pathlib import Path
import numpy as np
import time

# Import local modules
from rfsoc_io import *
import rfsoc_io
import utils

class R:
    '''
    Class for tuning and data acquisition of Microwave Kinetic Inductance Detectors (MKIDs) using
    Radio Frequency System on a Chip (RFSoC).
    '''

    def __init__(self, cfg_path = "/home/pcs/ccatkidlib/rfsoc/system_config.yaml"):
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
        self.output = True
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
        from rfsoc_timestream import Streamer

        # Initialize PCS clients
        self.rfsoc = OCSClient(self.io_cfg['pcs_agents']['rfsoc_agent'], args=[])

        # Setup drones
        if self.io_cfg['initialize']:
            self.setup_drones()
            rfsoc_io.send_msg('INFO', f'Initialized RFSoC agent. Communicating with drones: {self.drone_list}!', self.output)

        # Attempt to initialize timestream client
        self.streamer = None
        exit = False
        while self.streamer is None:
            try:
                self.streamer = Streamer(self.io_cfg['udp_ip'], self.io_cfg['udp_port'])
                rfsoc_io.send_msg('INFO', f"Successfully initialized timestream object using address {self.io_cfg['udp_ip']} and port {self.io_cfg['udp_port']}!" ,output = self.output)
            except:
                if exit:
                    rfsoc_io.send_msg('CRITICAL', 'The currently running process was not terminated. Safely exiting!', output = True)
                    sys.exit()
                rfsoc_io.send_msg('CRITICAL', 'A process is already bound to the timestream UDP port! Would you like to terminate this process?', output = True)
                os.system(f"fuser -ki -n udp {self.io_cfg['udp_port']}")
                time.sleep(1) # Wait for socket to close
                exit = True

        # Set NCLO frequency and accum_len for RFSoC drones
        if self.io_cfg['initialize']:
            for com, cfg in zip(self.drone_list, self.drone_cfg):
                # Set NCLO
                rtn = self.rfsoc.setNCLO(com_to = com, f_lo = cfg['tones']['NCLO'])
                rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)

                # Set accum_len
                rtn = self.rfsoc.setAccumLength(com_to=com)
                rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
            rfsoc_io.send_msg('INFO', f"Set NCLO to {[cfg['tones']['NCLO'] for cfg in self.drone_cfg]} MHz for drones: {self.drone_list}!", self.output)

            # Set drone attenuations
            self.set_atten()

        # Save init config
        rfsoc_io.save_config(self.log_dir / f'init_config_ext_{self.timestamp}.yaml', self.ext_cfg, self.save_cfg)
        rfsoc_io.save_config(self.log_dir / f'init_config_io_{self.timestamp}.yaml', self.io_cfg, self.save_cfg)
        for rfsoc_dir, cfg in zip(self.rfsoc_dirs, self.drone_cfg):
            rfsoc_io.save_config(rfsoc_dir / f'init_config_drone_{self.timestamp}.yaml', cfg, self.save_cfg)

    ###################
    # Setup Functions #
    ###################

    @header
    def setup_drones(self, **kwargs):
        '''
        Setup drones specified by com_to.

        Parameters:
            com_to (list of str): List of drones to setup
            parallel (bool): Whether to run board commands in parallel
            restart (bool): Whether to restart already running drones
        '''
        import drone_control

        com_to = self.drone_list
        parallel = self.parallel # Whether to run board commands in parallel (stops drones not in drone_list)
        restart = self.io_cfg['restart'] # Whether to restart already running drones

        for key, value in kwargs.items():
            if key == 'parallel':
                parallel = value
            elif key == 'restart':
                restart = value
            elif key == 'com_to':
                com_to = value

        # Edit master_drone_config.yaml in primecam_readout to match drone_list
        # ---------------------------------------------------------------------

        # Retrieve and load master drone config file
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

        # Setup drones
        # ------------

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
        if wait: rfsoc_io.wait(25, output = self.output, desc = "Waiting for drones to start")

    def load_system_config(self, cfg_path = "/home/rfsoc/ccatkidlib/rfsoc/system_config.yaml"):
        '''
        Load the system config file and setup file directory structure for saving data.

        Parametrs:
            cfg_path (str): File path of system configuration file
        '''

        # Load system configuration file
        # ------------------------------

        cfg = rfsoc_io.load_config(cfg_path)
        try:
            self.ext_cfg, self.io_cfg = cfg # Split config into external and IO config
        except:
            print("System config must contain two config files (an external config and an IO config)! Please reference the example system config file.")
            sys.exit()

        # Load drone list and drone configs
        # ---------------------------------
        self.drone_list = self.io_cfg['drone_list']
        self.board_list = None

        # Get number of drones and convert to a list if only a single drone is passed
        try:
            self.drone_num = len(self.drone_list)
        except:
            self.drone_list = [self.drone_list]
            self.drone_num = len(self.drone_list)

        self.drone_list, bids = self._get_com_to(com_to = self.drone_list, setup = False)
        self.board_list = bids

        # Load drone configuration files
        self.drone_cfg = []

        # Iterate over all board configs specified in system config
        board_cfgs = self.io_cfg['boards']
        for board in board_cfgs:
            # Check if board config corresponds to a board specified in drone list
            if board[1:] in self.board_list:
                # Load all drone configuration files for board
                self.drone_cfg.append(rfsoc_io.load_config(board_cfgs[board]['drone_cfg']))

                # Keep only the configuration files corresponding to drones in drone list
                drones = self.drone_cfg[-1]
                num = len(drones)
                for i in range(num):
                    # Remove drone config if drone not specified in drone list
                    if not (drones[num - i - 1]['com_to'] in self.drone_list):
                        drones.pop(num - i - 1 )

        # Flatten list of drone configs to match drone list
        self.drone_cfg = [cfg for board in self.drone_cfg for cfg in board]

        # Assign commonly used parameters as class attributes
        # ---------------------------------------------------

        # Whether to run commands in parallel
        self.parallel = self.io_cfg['parallel']

        # IO parameters for saving and printing data
        self.output = self.io_cfg['io']['terminal_output']
        self.save_cfg = self.io_cfg['io']['save_config_copy']
        self.save_data = self.io_cfg['io']['save_data']

        # Commonly used file paths
        self.data_dir = Path(self.io_cfg['file_paths']['data_dir'])
        self.drone_dir = Path(self.io_cfg['file_paths']['drone_dir'])
        self.tmp_data_dir = Path(self.io_cfg['file_paths']['tmp_data_dir'])

        # Save session ID to config files
        # -------------------------------

        self. edit_config(self.ext_cfg, 'sess_id', self.sess_id)
        self. edit_config(self.io_cfg, 'sess_id', self.sess_id)
        for cfg in self.drone_cfg:
            self.edit_config(cfg, 'sess_id', self.sess_id)
        
        # Create file directory structure for saving data
        # -----------------------------------------------
        new_dir_paths = rfsoc_io.create_book(self.curr_date, self.sess_id, self.drone_list, self.data_dir, output = self.output)

        # Assign data directories as class attributes
        self.rfsoc_dirs, self.targ_dirs, self.timestream_dirs, self.vna_dirs = new_dir_paths

    def edit_config(self, cfg, key, value, append = False):
        '''
        Update key in specified configuration file with the specified value. 

        Parameters:
            cfg (dict): Configuration file to update
            key (str): Key that should be updated
            value (any): Value with which to update key
            append (bool): Whether to append a new key, value pair to config file if key is not found
        Returns:
            done (bool): True if key was successfully created or updated. 
        '''

        # Edit config file dictionary
        done = utils.edit_dic(cfg, key, value)

        # Check if key was successfully updated
        if done:
            rfsoc_io.send_msg('DEBUG', f'Updated key "{key}" with value "{value}" in config file"!')
        elif append:
            cfg[key] = value
            done = True
            rfsoc_io.send_msg('DEBUG', f'Added key "{key}" with value "{value}" to config file!')
        else:
            rfsoc_io.send_msg('DEBUG', f'Failed to update key "{key}" with value "{value}" in config file!')
        return done

    @header
    def set_atten(self, drive = None, sense = None, **kwargs):
        '''
        Set drive/sense attenutors.

        Keyword Arguments:
            com_to (list of str): List of drones for which to set attenuation
            drive (list of float): Values of drive (DAC) attenuations in dB (must be between 0 and 31.75)
            sense (list of float): Values of sense (ADC) attenuations in dB (must be between 0 and 31.75)
        '''

        def _set_atten(com_to = None, direction = None, atten = None):
            '''
            Internal function for setting drone attenuations. 

            Parameters:
                com_to (str): List of drone com_to for which attenuations should be set
                direction (str): Which attenuator to set. Options are 'drive' for DAC and 'sense' for ADC
                atten (float): List of attenuations. Attenuation must be a float between 0 and 31.75
            Returns:
                attens: The set drone attenuations. None if invalid attenuations were used
            '''

            # Get attenuations
            # ----------------

            attens = self._get_drone_args(com_to, ['atten', f'{direction}']) # Get attenuations from drone config files

            # Override config attenuations with those passed as method argument (if any)
            if atten is not None:
                attens = self._parse_args(com_to, atten) # Parse attenuations passed as argument

                # Return None if invalid attenuations were passed
                if attens is None: return None 

                self._set_drone_args(com_to, direction, attens) # Update attenuations in drone config files

            # Set attenuations
            # ----------------

            # Iterate over drones and set attenuation
            for com, att in zip(com_to, attens):
                self.rfsoc.setAtten(com_to = com, direction = direction, atten = att)
            rfsoc_io.send_msg('INFO', f'Successfully set {direction} attenuation to {attens} for drones {com_to}!', self.output)

            return attens

        # Specify which attenuators to change
        com_to, _ = self._get_com_to(**kwargs)

        # Set drive attenuation
        _set_atten(com_to = com_to, direction = 'drive', atten = drive)

        # Set sense attenuation
        _set_atten(com_to = com_to, direction = 'sense', atten = sense)

    @header
    def write_config_comb(self, **kwargs):
        '''
        Write custom comb using comb parameters from config file. Config file parameters are
        superseceded by key word argument parameters.

        Parameters:
            tone_freqs: Float, array or file path of array containing custom tone frequencies (Hz)
            tone_powers: Float, array or file path of array containing custom tone powers
            tone_phis: Float, array or file path of array containing custom tone phases
        '''

        com_to, _ = self._get_com_to(**kwargs)

        tone_freqs = []
        tone_powers = []
        tone_phis = []

        # Iterate over each drone in drone list
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
        
        # Write custom comb for each drone
        for com, freq, power, phi in zip(com_to, tone_freqs, tone_powers, tone_phis):
            self._write_custom_comb(com, tone_freqs = freq, tone_powers = power, tone_phis = phi)

        # Write sweep comb based on custom parameters
        # -------------------------------------------
        if self.parallel:
            for board in self.board_list:
                rtn = self.rfsoc.writeTargCombFromCustomList(com_to = board)
                rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
                rfsoc_io.send_msg('INFO', f'Sucessfully wrote custom comb for board {board}!', self.output)
        else:
            for com in com_to:
                rtn = self.rfsoc.writeTargCombFromCustomList(com_to = com)
                rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
                rfsoc_io.send_msg('INFO', f'Sucessfully wrote custom comb for drone {com}!', self.output)

    @header
    def get_ADC_rms(self, **kwargs):
        ''' 
        Get the root mean squared (RMS) power at the analog to digital converter (ADC) for 
        drones specified by com_to.

        Parameters:
            com_to (list of str): List of drones to setup
        '''

        def _getSnapData(com, *args, **kwargs):
            rtn = self.rfsoc.getSnapData(com_to = com, mux_sel = 0, silent = False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
            return rtn.session['messages'][1][1][:-3].split('}, ')

        rtns = np.array(self._run_parallel(_getSnapData, **kwargs)).flatten()

        rms_list, inds = [], []
        for rtn in rtns:
            data_dic = pickle.loads(ast.literal_eval(rtn.split(': ')[-1]))
            ind = self.drone_list.index(f"{data_dic['bid']}.{data_dic['drid']}")
            inds.append(ind)

            # From primecam_readout alcove_base getADCrms function
            I, Q = data_dic['data']
            z = I + 1j * Q
            rms = np.real(np.sqrt(np.mean(z * np.conj(z)))).astype(np.float32)
            rms_list.append(rms)

            self.edit_config(self.drone_cfg[ind], 'ADC_RMS', rms)

        rms_list = [rms for _, rms in sorted(zip(inds, rms_list))]
        return rms_list

    def check_avail(self, com, **kwargs):
        '''
        Check the available storage space on RFSoC board.

        Parameters:
            com (str): Bid of board to check available space

        Will probably want to move over this functionality to the 
        queen_agent in the future.
        '''

        threshold = 2.0  # Threshold after which to send warnings (in GB)
        crit_threshold = 0.1 # Critical threshold (in GB)
        to_keep = 0.125 # Days
        clean = True

        for key, value in kwargs.items():
            if key == 'threshold':
                threshold = value
            elif key == 'crit_threshold':
                crit_threshold = value
            elif key == 'clean':
                clean = value
            elif key == 'to_keep':
                to_keep = value

        bid = com.split('.')[0]
        bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
        key = self.io_cfg['file_paths']['ssh_key']
        
        # Get space available on board in bytes
        with rfsoc_io.get_connection(bip, key) as c:
            avail_space = int(c.run("df -P -B1 | grep '/dev/root' | awk '{print $4}'",  hide = True).stdout)/1e9
        
            # Append available board space to config
            self.edit_config(self.ext_cfg, 'avail_space', avail_space, append = True)

            if avail_space < crit_threshold:
                rfsoc_io.send_msg('CRITICAL', f'Storage space on board {bid} is {avail_space} GB!', output = True)
            elif avail_space < threshold:
                rfsoc_io.send_msg('WARNING', f'Storage space on board {bid} is {avail_space} GB! Clean space on board!', self.output)
                if clean:
                    out = c.run(f"python3 clean_board.py 'drones' -l {to_keep}", hide = True).stdout
                    rfsoc_io.send_msg('DEBUG', out, self.output)
            else:
                rfsoc_io.send_msg('INFO', f'Storage space on board {bid} is {avail_space} GB!', self.output)
        
        return avail_space
    
    ##############################
    # Data Acquisition Functions #
    ##############################
    @header
    def take_vna_sweep(self, **kwargs):
        '''
        Take a vector network analyzer (VNA) sweep using RFSoC.

        Parameters:
            NCLO (int): Numerically controlled local oscillator frequency in MHz. Center frequency of VNA sweep.
        Returns:
            output (str): VNA sweep file path
        '''

        def _write_vna_comb(com, *args, **kwargs):
            avail = self.check_avail(com = com)
            rtn = self.rfsoc.writeNewVnaComb(com_to = com, silent = True)
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
            return rtn
        
        def _take_vna_sweep(com, *args, **kwargs):
            avail = self.check_avail(com = com)
            rtn = self.rfsoc.vnaSweep(com_to = com, silent = True)
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)

            self.get_ADC_rms(com_to = com)
            return rtn
        
        # Get com_to
        com_to, _ = self._get_com_to(**kwargs)

        # Parse key word arguments
        write_tones = True
        for key, value in kwargs.items():
            if key == 'NCLO':
                NCLO = self._parse_args(com_to, value)
                for com, N in zip(com_to, NCLO):
                    rtn = self.rfsoc.setNCLO(com_to = com, f_lo = N)
                    rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)

                self._set_drone_args(com_to, "NCLO", NCLO)
                rfsoc_io.send_msg('INFO', f'Set NCLO to {NCLO} MHz for drones {com_to}!', self.output)
            elif key == 'write_tones':
                write_tones = value

        # Write VNA comb
        if write_tones:
            rfsoc_io.send_msg('INFO', 'Writing new VNA combs!', self.output)
            self._run_parallel(_write_vna_comb, **kwargs)
            rfsoc_io.send_msg('INFO', f'Successfully wrote new VNA combs for drones {com_to}!', self.output)
            time.sleep(5) # Wait before taking sweep (not waiting can affect sweep quality, unsure if this is still the case: need to test)

        # Take VNA sweep
        # Get current timestamp
        self.timestamp = str(time.time()).split('.')[0]
        rfsoc_io.send_msg('INFO', 'Taking VNA sweeps!', self.output)
        self._run_parallel(_take_vna_sweep, **kwargs)
        rfsoc_io.send_msg('INFO', f'Finished taking VNA sweeps for drones {com_to}!', self.output)

        # Save ext config
        self.ext_cfg = rfsoc_io.save_config(self.log_dir / f"vna_config_ext_{self.timestamp}.yaml", self.ext_cfg, self.save_cfg)

        # Save VNA sweep(s)
        vna_files, vna_paths = [], []
        ssh_key = self.io_cfg['file_paths']['ssh_key']
        for com in com_to:
            ind = self.drone_list.index(com)
            bid, drid = com.split('.')
            bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
            path = self.drone_dir / f'drone{drid}' / 'vna'

            with rfsoc_io.get_connection(bip, ssh_key) as c:
                vna_file = rfsoc_io.get_most_recent_file_board(c, path, file_identifier = self.io_cfg['board_file_names']['vna_s21'])

                if not rfsoc_io.path_exists(c, vna_file):
                    rfsoc_io.send_msg('ERROR', "VNA sweep was unsuccessful! No files saved.", self.output)
                    vna_files.append(None)
                    continue

                vna_files.append(vna_file)
                if self.save_data:
                    fname = self.io_cfg["save_file_names"]["vna_sweep"]
                    fname = f'{fname}_{self.timestamp}.npy'
                    vna_path = self.vna_dirs[ind] / fname
                    c.get(str(vna_file), str(vna_path))

            rfsoc_io.send_msg('DEBUG', f'>>> Successfully copied VNA file from drone {com}!', self.output)
            vna_paths.append(vna_path)

            # Edit config with VNA comb used
            freq_comb, amp_comb, phi_comb = self._get_curr_comb(com)
            self.edit_config(self.drone_cfg[ind], 'tone_freqs', freq_comb)
            self.edit_config(self.drone_cfg[ind], 'tone_powers', amp_comb)
            self.edit_config(self.drone_cfg[ind], 'tone_phis', phi_comb)
            self.edit_config(self.drone_cfg[ind], 'num_tones', len(freq_comb))

            # Save drone config
            self.drone_cfg[ind] = rfsoc_io.save_config(self.rfsoc_dirs[ind] / f"vna_config_drone_{self.timestamp}.yaml", self.drone_cfg[ind], self.save_cfg)

        if self.save_data:
            return vna_paths
        else:
            return vna_files

    @header
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
        
        def _take_targ_sweep(com, *args, **kwargs):
            avail = self.check_avail(com = com)
            rtn = self.rfsoc.targetSweep(com_to = com, silent = True)
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
            return rtn

        com_to, _ = self._get_com_to(**kwargs)

        # Evaluate kwargs
        write_tones = False
        for key, value in kwargs.items():
            #if key == 'bandwidth':
            #    self.ext_cfg['rfsoc_tones']['bandwidth'] = value
            #elif key == 'N_steps':
            #    self.ext_cfg['rfsoc_tones']['N_steps'] = value
            if key == 'write_tones':
                write_tones = value
            elif key == 'NCLO':
                NCLO = self._parse_args(com_to, value)
                for com, N in zip(com_to, NCLO):
                    rtn = self.rfsoc.setNCLO(com_to = com, f_lo = N)
                    rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)

                self._set_drone_args(com_to, "NCLO", NCLO)
                rfsoc_io.send_msg('INFO', f'Set NCLO to {NCLO} MHz for drones {com_to}!', self.output)

        # Write new target sweep comb
        if write_tones:
            self.write_config_comb(**kwargs)
            time.sleep(3) # Wait before taking sweep (not waiting can affect sweep quality)

        # Get timestamp
        self.timestamp = str(time.time()).split('.')[0]
        rfsoc_io.send_msg('INFO', 'Taking target sweeps!', output = self.output)
        self._run_parallel(_take_targ_sweep, **kwargs)
        rfsoc_io.send_msg('INFO', f'Finished taking target sweeps for drones {com_to}!', self.output)

        # Get current timestamp and save ext config
        self.ext_cfg = rfsoc_io.save_config(self.log_dir / f"target_config_ext_{self.timestamp}.yaml", self.ext_cfg, self.save_cfg)

        # Save target sweep
        targ_files, targ_paths = [], []
        for com in com_to:
            ind = self.drone_list.index(com)
            bid, drid = com.split('.')
            bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
            key = self.io_cfg['file_paths']['ssh_key']
            path = self.drone_dir / f'drone{drid}' / 'targ'

            with rfsoc_io.get_connection(bip, key) as c:
                targ_file = rfsoc_io.get_most_recent_file_board(c, path, file_identifier = self.io_cfg['board_file_names']['targ_s21'])

                if not rfsoc_io.path_exists(c, targ_file):
                    rfsoc_io.send_msg('ERROR', "Target sweep was unsuccessful! No files saved.", self.output)
                    targ_files.append(None)
                    continue

                targ_files.append(targ_file)
                if self.save_data:
                    fname = self.io_cfg["save_file_names"]["targ_sweep"]
                    fname = f'{fname}_{self.timestamp}.npy'
                    targ_path = self.targ_dirs[ind] / fname
                    c.get(str(targ_file), str(targ_path))

            rfsoc_io.send_msg('DEBUG', f'>>> Successfully copied target file from drone {com}!', self.output)
            targ_paths.append(targ_path)

            # Edit config with target comb used
            freq_comb, amp_comb, phi_comb = self._get_curr_comb(com)
            self.edit_config(self.drone_cfg[ind], 'tone_freqs', freq_comb)
            self.edit_config(self.drone_cfg[ind], 'tone_powers', amp_comb)
            self.edit_config(self.drone_cfg[ind], 'tone_phis', phi_comb)
            self.edit_config(self.drone_cfg[ind], 'num_tones', len(freq_comb))

            # Save drone config
            self.drone_cfg[ind] = rfsoc_io.save_config(self.rfsoc_dirs[ind] / f"targ_config_drone_{self.timestamp}.yaml", self.drone_cfg[ind], self.save_cfg)

        if self.save_data:
            return targ_paths
        else:
            return targ_files

    @header
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

        com_to, _ = self._get_com_to(**kwargs)

        # Parse key word arguments
        write_tones = False
        save_data = self.save_data
        reset = True # Whether to turn off currently running timestreams
        turn_off = True # Whether to turn off timestream at end of data collection
        for key, value in kwargs.items():
            if key == 'write_tones':
                write_tones = value
            elif key == 'save_data':
                save_data = value
            elif key == 'reset':
                reset = value
            elif key == 'turn_off':
                turn_off = value

        # Write new tones
        if write_tones:
            self.write_config_comb(**kwargs)
            time.sleep(3)

        rfsoc_io.send_msg('INFO', f'Taking {t_sec} seconds of timestream data!', self.output)

        # Turn off all currently running timestreams
        if reset:
            ret = self.rfsoc.timestreamOn(on = False)
            time.sleep(1) # Wait to ensure that all timestreams were turned off

        stream_paths = []
        timestreams = []
        self.timestamp = str(time.time()).split('.')[0]
        if self.parallel and len(com_to) > 1:
            for board in self.board_list:
                avail = self.check_avail(com = board)
                # Turn on timestreams
                if reset: ret = self.rfsoc.timestreamOn(com_to = board, on = True)

                inds = [self.drone_list.index(com) for com in com_to if com.split('.')[0] == board]

                # Get total number of packets to capture
                N_packets = len(inds)*int(self.io_cfg['boards'][f'b{board}']['sampling_freq']*t_sec)

                # Take timestream
                data, auxs, ips, ports = self.streamer.take_timestream(N_packets)

                # Turn off timestream
                if turn_off: ret = self.rfsoc.timestreamOn(com_to = board, on = False)

                rfsoc_io.send_msg('INFO', f'Finished taking timestream data for board {board}!', self.output)

                # Sort the data by drone based on source IP address
                drone_ips = [self.drone_cfg[ind]['udp_source_ip'] for ind in inds]
                drone_data = [ [] for _ in range(len(inds))]
                for dat, aux, ip in zip(data, auxs, ips):
                    drone_ind = drone_ips.index(str(ip))
                    drone_data[drone_ind].append(np.append(aux[3], np.array(dat)))

                # Save the timestreams for each drone
                for data, ind in zip(drone_data, inds):
                    if not len(data) == 0:
                        # Convert data into I, Q
                        data = np.array(data)
                        I, Q = data[:,1::2].T, data[:,2::2].T
                        ts = data[:, 0].T

                        # Combine I and Q into complex S21 data
                        s21z = I + 1j * Q

                        # Combine array of times and complex S21z data
                        timestream_data = np.append([ts], s21z, axis = 0)

                        if save_data:
                            # Edit config with comb used for timestream
                            freq_comb, amp_comb, phi_comb = self._get_curr_comb(com_to[ind])
                            self.edit_config(self.drone_cfg[ind], 'tone_freqs', freq_comb)
                            self.edit_config(self.drone_cfg[ind], 'tone_powers', amp_comb)
                            self.edit_config(self.drone_cfg[ind], 'tone_phis', phi_comb)
                            self.edit_config(self.drone_cfg[ind], 'num_tones', len(freq_comb))
                            
                            self.drone_cfg[ind] = rfsoc_io.save_config(self.rfsoc_dirs[ind] / f"timestream_config_drone_{self.timestamp}.yaml", self.drone_cfg[ind], self.save_cfg)
                            stream_paths.append(self._save_timestream(ind, timestream_data))
                        else:
                            timestreams.append(timestream_data)
        else:
            for com in com_to:
                avail = self.check_avail(com = com)
                ind = self.drone_list.index(com)

                # Get number of packets per drone
                N_packets = int(self.io_cfg['boards'][f"b{com.split('.')[0]}"]['sampling_freq']*t_sec)

                # Turn on timestream
                if reset: ret = self.rfsoc.timestreamOn(com_to = com, on = True)

                # Take timestream
                data, aux, ips, ports = self.streamer.take_timestream(N_packets)

                # Turn off timestream
                if turn_off: ret = self.rfsoc.timestreamOn(com_to = com, on = False)

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
                    # Edit config with comb used for timestream
                    freq_comb, amp_comb, phi_comb = self._get_curr_comb(com)
                    self.edit_config(self.drone_cfg[ind], 'tone_freqs', freq_comb)
                    self.edit_config(self.drone_cfg[ind], 'tone_powers', amp_comb)
                    self.edit_config(self.drone_cfg[ind], 'tone_phis', phi_comb)
                    self.edit_config(self.drone_cfg[ind], 'num_tones', len(freq_comb))

                    self.drone_cfg[ind] = rfsoc_io.save_config(self.rfsoc_dirs[ind] / f"timestream_config_drone_{self.timestamp}.yaml", self.drone_cfg[ind], self.save_cfg)
                    stream_paths.append(self._save_timestream(ind, timestream_data))
                else:
                    timestreams.append(timestream_data)

        if save_data:
            # Save ext config
            self.ext_cfg = rfsoc_io.save_config(self.log_dir / f"timestream_config_ext_{self.timestamp}.yaml", self.ext_cfg, self.save_cfg)
            
            return stream_paths
        else:
            return timestreams

    ####################
    # Tuning Functions #
    ####################

    @header
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

        def _write_targ_comb(com, *args, **kwargs):
            avail = self.check_avail(com = com)
            # Write target comb using found resonators
            rtn1 = self.rfsoc.writeTargCombFromVnaSweep(com_to = com)
            rfsoc_io.send_msg('PCS', f'{rtn1.session}', self.output)

            # Write current comb to custom comb files
            rtn2 = self.rfsoc.createCustomCombFilesFromCurrentComb(com_to = com)
            rfsoc_io.send_msg('PCS', f'{rtn2.session}', self.output)

            return rtn1, rtn2

        # Get com_to list
        com_to, _ = self._get_com_to(**kwargs)
        write_targ_comb = True

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
            elif key == 'write_targ_comb':
                write_targ_comb = value

        # Take VNA sweep if new_sweep = True or if one does not already exist
        to_sweep = []
        for com in com_to:
            ind = self.drone_list.index(com)
            # Check if a VNA sweep was taken in the last day
            vna_file = rfsoc_io.get_most_recent_file(self.vna_dirs[ind], f"{self.io_cfg['save_file_names']['vna_sweep'][0]}*", self.output, time_past=24*3600)
            if not vna_file.exists() or new_sweep:
                to_sweep.append(com)
        if len(to_sweep) != 0:
            # Take VNA sweep(s) without saving configs (saved later)
            self.save_cfg = False
            kwargs['com_to'] = to_sweep
            self.take_vna_sweep(**kwargs)
            self.save_cfg = True

        rfsoc_io.send_msg('INFO', f"Finding resonators from VNA sweep for drones {com_to}!", output = self.output)
        for i, com in enumerate(com_to):
            # Find resonators from VNA sweep
            rtn = self.rfsoc.findVnaResonators(com_to = com, peak_prom_std = peak_prom_std[i], peak_prom_db = peak_prom_db[i], peak_dis = peak_dis[i],
                                                width_min = width_min[i], width_max = width_max[i], stitch = stitch[i], stitch_sw = stitch_sw[i],
                                                remove_cont = remove_cont[i], continuum_wn = continuum_wn[i], remove_noise = remove_noise[i], noise_wn = noise_wn[i])
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
        rfsoc_io.send_msg('INFO', f"Finished finding resonators from VNA sweep for drones {com_to}!", output = self.output)

        if write_targ_comb:
            rfsoc_io.send_msg('INFO', f"Writing target combs using found resonators for drones {com_to}!", output = self.output)
            self._run_parallel(_write_targ_comb, **kwargs)
            rfsoc_io.send_msg('INFO', f'Finished writing target combs for drones {com_to}!', output = self.output)

        found_nums = []
        found_freqs = []
        for com in com_to:
            # Parse com information 
            ind = self.drone_list.index(com)
            bid, drid = com.split('.')
            bip = self.io_cfg['boards'][f'b{bid}']['board_ip']

            # Get ssh key
            key = self.io_cfg['file_paths']['ssh_key']

            # Define directory where fonud resonators are saved
            vna_dir = self.drone_dir / f'drone{drid}' / 'vna'

            with rfsoc_io.get_connection(bip, key) as c:
                # Get file with found resonators
                res_file = rfsoc_io.get_most_recent_file_board(c, vna_dir, file_identifier = self.io_cfg["board_file_names"]["vna_res"])
                
                # Check that the files exist
                # --------------------------
                if not rfsoc_io.path_exists(c, res_file):
                    rfsoc_io.send_msg('ERROR', f"Could not find found resonators file for drone {com}.", self.output)
                    res_freq = None
                else:
                    res_freq = rfsoc_io.load_array_board(c, res_file, output = self.output)

            # Save resonator frequncies and number of found resonators to list
            res_freq = np.real(res_freq).tolist()
            found_num = len(res_freq)

            found_freqs.append(np.array(res_freq))
            found_nums.append(found_num)

            rfsoc_io.send_msg('DEBUG', f"Found detector frequencies for drone {com}: {np.array(res_freq)} Hz", self.output)
            rfsoc_io.send_msg('INFO', f"Found {found_num} detectors for drone {com}!", self.output)

            self.edit_config(self.drone_cfg[ind], 'found_num_detectors', found_num)
            self.edit_config(self.drone_cfg[ind], 'found_detector_freqs', res_freq)
            self.drone_cfg[ind] = rfsoc_io.save_config(self.rfsoc_dirs[ind] / f"vna_config_drone_{self.timestamp}.yaml", self.drone_cfg[ind], self.save_cfg)

        # Save config files
        self.ext_cfg = rfsoc_io.save_config(self.log_dir / f"vna_config_ext_{self.timestamp}.yaml", self.ext_cfg, self.save_cfg)
        return found_nums, found_freqs

    @header
    def find_detectors_fine(self, new_sweep = True, **kwargs):
        '''
        Find detectors and place tones at their minima. Length of bins is 500,000 Hz / N_steps.

        Parameters:
            new_sweep (bool): Whether or not to take a new target sweep
            stitch_bw (int): Number of bins to use on when stitching tones
        Returns:
            output (arr of floats): Returns an array of the found detector frequencies
        '''
        def _write_targ_comb(com, *args, **kwargs):
            # Write target comb using found resonators
            rtn1 = self.rfsoc.writeTargCombFromTargSweep(com_to = com)
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)

            # Write target comb to custom comb files
            rtn2 = self.rfsoc.createCustomCombFilesFromCurrentComb(com_to = com)
            rfsoc_io.send_msg('PCS', f'{rtn2.session}', self.output)                
            return rtn1, rtn2

        com_to, _ = self._get_com_to(**kwargs)

        # Evaluate kwargs
        stitch_bw = self._get_drone_args(com_to, ['tones', 'N_step'])
        write_tones = True

        # Parse key word arguments
        for key, value in kwargs.items():
            if key == 'stitch_bw':
                stitch_bw = self._parse_args(com_to, value)
            elif key == 'write_tones':
                write_tones = value

        # Take target sweep if new_sweep = True or if one does not already exist
        to_sweep = []
        for com in com_to:
            ind = self.drone_list.index(com)
            # Check if a target sweep was taken in the last day
            targ_file = rfsoc_io.get_most_recent_file(self.targ_dirs[ind], f"{self.io_cfg['save_file_names']['targ_sweep'][0]}*", self.output, time_past=24*3600)
            if not targ_file.exists() or new_sweep:
                to_sweep.append(com)
        if len(to_sweep) != 0:
            # Take target sweep(s) without saving config (saved later)
            self.save_cfg = False
            kwargs['com_to'] = to_sweep
            self.take_target_sweep(**kwargs)
            self.save_cfg = True

        rfsoc_io.send_msg('INFO', f"Finding resonators from target sweep for drones {com_to}!", output = self.output)
        for i, com in enumerate(com_to):
            # Find resonators from target sweep
            rtn = self.rfsoc.findTargResonators(com_to = com, stitch_bw = stitch_bw[i])
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
        rfsoc_io.send_msg('INFO', f"Finished finding resonators from target sweep for drones {com_to}!", output = self.output)

        if write_tones:
            rfsoc_io.send_msg('INFO', f"Writing target comb using found resonators for drones {com_to}!", output = self.output)
            self._run_parallel(_write_targ_comb, **kwargs)
            rfsoc_io.send_msg('INFO', f'Finished writing target combs for drones {com_to}!', output = self.output)

        found_nums = []
        found_freqs = []
        for com in com_to:
            # Parse com information 
            ind = self.drone_list.index(com)
            bid, drid = com.split('.')
            bip = self.io_cfg['boards'][f'b{bid}']['board_ip']

            # Get ssh key
            key = self.io_cfg['file_paths']['ssh_key']

            # Define directory where found resonators are saved
            targ_dir = self.drone_dir / f'drone{drid}' / 'targ'

            with rfsoc_io.get_connection(bip, key) as c:
                # Get file with found resonators
                res_file = rfsoc_io.get_most_recent_file_board(c, targ_dir, file_identifier = self.io_cfg["board_file_names"]["targ_res"])
                
                # Check that the files exist
                # --------------------------
                if not rfsoc_io.path_exists(c, res_file):
                    rfsoc_io.send_msg('ERROR', f"Could not find found resonators file for drone {com}.", self.output)
                    res_freq = None
                else:
                    res_freq = rfsoc_io.load_array_board(c, res_file, output = self.output)

            # Save resonator frequncies and number of found resonators to list
            res_freq = np.real(res_freq).tolist()
            found_num = len(res_freq)

            found_freqs.append(np.array(res_freq))
            found_nums.append(found_num)

            rfsoc_io.send_msg('DEBUG', f"Found detector frequencies for drone {com}: {np.array(res_freq)} Hz", self.output)
            rfsoc_io.send_msg('INFO', f"Found {found_num} detectors for drone {com}!", self.output)

            self.edit_config(self.drone_cfg[ind], 'found_num_detectors', found_num)
            self.edit_config(self.drone_cfg[ind], 'found_detector_freqs', res_freq)
            self.drone_cfg[ind] = rfsoc_io.save_config(self.rfsoc_dirs[ind] / f"targ_config_drone_{self.timestamp}.yaml", self.drone_cfg[ind], self.save_cfg)

        self.ext_cfg = rfsoc_io.save_config(self.log_dir / f"targ_config_ext_{self.timestamp}.yaml", self.ext_cfg, self.save_cfg)
        return found_nums, found_freqs

    #############################
    # Internal Helper Functions #
    #############################

    def _run_parallel(self, func, *args, **kwargs):
        '''
        Run subfunction in parallel or in series

        Parameters:
            func (func): Function to run
            args: Function args
            kwargs: Function kwargs
        '''

        com_to, boards = self._get_com_to(**kwargs)

        parallel = self.parallel
        for key, value in kwargs.items():
            if key == 'parallel':
                parallel = value
        
        rtn_list = []
        if parallel and len(com_to) > 1:
            for board in boards:
                rtn_list.append(func(board, *args, **kwargs))
        else:
            for com in com_to:
                rtn_list.append(func(com, *args, **kwargs))
        return rtn_list

    def _write_custom_comb(self, com, **kwargs):
        '''
        Write custom tone power(s), frequency(ies), and/or phase(s) of tones.
        Parameters:
            tone_freqs: Float, array or file path of array containing custom tone frequencies
            tone_powers: Float, array or file path of array containing custom tone powers
            tone_phis: Float, array or file path of array containing custom tone phases
        '''

        # Import waveform generation fuction from alcove_base.py in primecam_readout
        # --------------------------------------------------------------------------
        primecam_readout = Path(self.io_cfg['file_paths']['primecam_readout']) / 'alcove_commands'
        sys.path.append(str(primecam_readout))
        from alcove_base import generateWaveDdr4

        def _check_comb_type(key, value):
            # Convert from dB to normal units if necessary
            if key == 'tone_powers' and self.drone_cfg[ind]['tones']['dB']: value = utils.convert_from_dB(value)
            
            # Check if the tone frequencies are within the bandwidth of the RFSoC
            if key == 'tone_freqs' and np.max(np.abs(np.array(value) - self.drone_cfg[ind]['tones']['NCLO']*1e6))  > self.drone_cfg[ind]['tones']['full_bandwidth']*1e6/2:
                rfsoc_io.send_msg('WARNING', f"{value} contains frequencies outside of the RFSoC bandwidth. Not writing {key} custom comb!")
                return None

            return value
        
        def _reset_comb(comb_dict):
            for key, value in comb_dict.items():
                rfsoc_io.save_array_board(c, value['path'], np.array(value['comb']).tolist())

            # Return unmodified comb
            return comb_dict['tone_freqs']['comb'], comb_dict['tone_powers']['comb'], comb_dict['tone_phis']['comb']

        def _comb_peak(freqs, amps, phis):
            '''
            Function from tones.py in primecam_readout. Returns the maximum power produced by the comb. 
            '''
            x, _, _ = generateWaveDdr4(freqs, amps, phis)
            x.real, x.imag = x.real.astype("int16"), x.imag.astype("int16")
            return np.max(np.abs(x.real + 1j*x.imag))
        
        ind = self.drone_list.index(com)
        bid, drid = com.split('.')
        bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
        ssh_key = self.io_cfg['file_paths']['ssh_key']
        custom_comb_dir = self.drone_dir / f'drone{drid}' / 'custom_comb'
        gen_amps = False
        gen_phis = False

        # Evaluate passed key word arguments
        # ----------------------------------

        # Open connection to RFSoC board
        with rfsoc_io.get_connection(bip, ssh_key) as c:
            # Get current comb and paths to custom comb files
            # -----------------------------------------------
            freq_comb, amp_comb, phi_comb = self._get_curr_comb(com)

            freq_path = rfsoc_io.get_most_recent_file_board(c, custom_comb_dir, file_identifier = self.io_cfg["board_file_names"]["comb_freq"])
            amp_path = rfsoc_io.get_most_recent_file_board(c, custom_comb_dir, file_identifier = self.io_cfg["board_file_names"]["comb_amp"])
            phi_path = rfsoc_io.get_most_recent_file_board(c, custom_comb_dir, file_identifier = self.io_cfg["board_file_names"]["comb_phi"])


            # Combine comb, paths, and num_tones into single dictionary
            comb_dict = {"tone_freqs": {"path": freq_path, "comb": freq_comb, "num_tones": None},
                         "tone_powers": {"path": amp_path, "comb": amp_comb, "num_tones": None},
                         "tone_phis": {"path": phi_path, "comb": phi_comb, "num_tones": None}}

            # Parse comb values
            # -----------------
            for key, value in kwargs.items():
                # Get file path and comb corresponding to correct key
                try:
                    path = comb_dict[key]["path"]
                    if not rfsoc_io.path_exists(c, path):
                        rfsoc_io.send_msg('ERROR', f"Could not find {key} custom comb file at path {path}.", self.output)
                        raise FileNotFoundError
                    curr_comb = comb_dict[key]["comb"]
                    rfsoc_io.send_msg('DEBUG', f'Modifying {key} for drone {com}!', self.output)
                except:
                    comb_dict[key]['num_tones'] = len(curr_comb)
                    continue

                # Check if value is a string or file path
                try:
                    if value.lower() == 'gen':
                        if key == 'tone_powers': 
                            gen_amps = True
                        elif key == 'tone_phis':
                            gen_phis = True
                        comb_dict[key]['num_tones'] = float(0)
                        continue

                    value = Path(value)
                    # If value is a file path, ensure that it is a valid path
                    if not value.exists():
                        rfsoc_io.send_msg('WARNING', f"{value} is not a valid {key} custom comb file path for drone {com}!", self.output)
                        comb_dict[key]['num_tones'] = len(curr_comb)
                        continue

                    value = np.load(value)
                # If not file path, assume value is a number or an array of numbers
                except:
                    try:
                        # Assume an array is passed and check if its non-empty
                        if len(value) == 0:
                            rfsoc_io.send_msg('WARNING', f"'{value}' is an empty array, not writing to {key} custom comb file for drone {com}!", self.output)
                            comb_dict[key]['num_tones'] = len(curr_comb)
                            continue
                    except TypeError:
                        try:
                            # Assume a number is passed and use the same number for all tones
                            comb_dict[key]['num_tones'] = float(value) # Store the number in num_tones as a float for processing later since the number of tones in the comb may be unknown
                            continue
                        except:
                            # Invalid value passed; ignore and move on to next comb
                            rfsoc_io.send_msg('WARNING', f"{value} is not a valid file path, array, or number! Not writing {key} custom comb file for drone {com}.")
                            comb_dict[key]['num_tones'] = len(curr_comb)
                            continue
                value = _check_comb_type(key, value)
                if value is None: 
                    comb_dict[key]['num_tones'] = len(curr_comb)
                    continue

                # Copy the array with custom parameters onto the rfsoc board
                rfsoc_io.save_array_board(c, path, np.array(value).tolist())
                comb_dict[key]['num_tones'] = len(value)

            # If num_tones entry is None, comb was not modified and the number of tones is that of the current comb
            # -----------------------------------------------------------------------------------------------------
            for key, value in comb_dict.items():
                if value['num_tones'] is None: comb_dict[key]['num_tones'] = len(value['comb'])

            # Check that number of tones agree
            # --------------------------------
            tone_num = None
            for key, value in comb_dict.items():
                num_tones = value['num_tones']
                if type(num_tones) is float:
                    continue
                elif tone_num is None:
                    tone_num = num_tones
                elif num_tones != tone_num:
                    # Resetting back to current comb
                    rfsoc_io.send_msg('ERROR', f"The number of tones in the frequeny, power, and phase combs do not match. Reverting back to current comb!")
                    return _reset_comb(comb_dict)
                    

            # Write comb for num_tone entries that are floats (only single number was passed)
            # -------------------------------------------------------------------------------
            for key, value in comb_dict.items():
                comb_val = value['num_tones']
                if type(comb_val) is float:
                    if tone_num is None: 
                        tone_num = 1
                        rfsoc_io.send_msg('WARNING', f"Only numbers were passed as custom comb; assuming a single tone should be written! To avoid unexpected errors, please pass numbers as a list to write a single tone in the future.")
                    comb_val = comb_val * np.ones(tone_num)
                    comb_val = _check_comb_type(key, comb_val)
                    if comb_val is not None:
                        rfsoc_io.save_array_board(c, value['path'], np.array(comb_val).tolist())
                        comb_dict[key]['num_tones'] = len(comb_val)
                    else:
                        # Resetting back to current comb
                        rfsoc_io.send_msg('ERROR', f"The comb failed a check causing the number of tones in the frequeny, power, and phase combs to not match. Reverting back to current comb!")
                        return _reset_comb(comb_dict)

            # Generate phi comb - As done in primecam_readout
            # -----------------------------------------------

            # Define maximum RFSoC DAC power
            max_power = self.io_cfg['boards'][f'b{bid}']['max_power']
            tone_freqs = rfsoc_io.load_array_board(c, freq_path)
            tone_freqs_bb = np.array(tone_freqs) - self.drone_cfg[ind]['tones']['NCLO']*1e6

            # Determine if tone powers need to be generated
            # ---------------------------------------------
            factors = [1]
            rescale_step = self.drone_cfg[ind]['tones']['generation']['rescale_step']
            if gen_amps:
                tone_powers = np.ones(tone_num)*comb_max/np.sqrt(tone_num)
                factors = np.arange(0.36, 0, rescale_step)
            else:
                tone_powers = rfsoc_io.load_array_board(c, amp_path)
                if self.drone_cfg[ind]['tones']['generation']['rescale_power']: factors = np.arange(1, 0, rescale_step)
            
            # Generate tone phis and tone powers
            # ----------------------------------
            tone_phis = rfsoc_io.load_array_board(c, phi_path)

            # Iterate over tone power rescale factors
            for factor in factors:
                tone_powers = factor*tone_powers

                # Generate phis
                if gen_phis:
                    # Attempt to generate phis 'gen_attempts' number of times
                    for _ in range(self.drone_cfg[ind]['tones']['gen_attempts']):
                        tone_phis = np.random.uniform(-np.pi, np.pi, tone_num) # Randomly generate phase comb

                        # Check if max comb power is less than max DAC power
                        if _comb_peak(tone_freqs_bb, tone_powers, tone_phis) < max_power:
                            rfsoc_io.send_msg('DEBUG', f"The custom comb does not exceed the maximum DAC power with factor {factor}!", self.output)

                            # Save phi comb and power comb (if necessary)
                            rfsoc_io.save_array_board(c, phi_path, np.array(tone_phis).tolist())
                            if gen_amps or self.drone_cfg[ind]['tones']['generation']['rescale_power']: rfsoc_io.save_array_board(c, amp_path, np.array(tone_powers).tolist())
                            break
                    # Rescale tone power (if 'rescale_power' is True)
                    else:
                        if self.drone_cfg[ind]['tones']['generation']['rescale_power']: rfsoc_io.send_msg('DEBUG', f"Failed to generate comb with factor {factor}! Rescaling to lower tone power.", self.output)
                        continue
                    
                    # Break out of tone power loop if valid phi comb is found
                    break
            
                # Do not generate phis and check if comb exceeds maximum DAC power
                else:
                    comb_max = _comb_peak(tone_freqs_bb, tone_powers, tone_phis)
                    # Check if max comb power is less than max DAC power
                    if comb_max < max_power:
                        rfsoc_io.send_msg('DEBUG', f"The custom comb does not exceed the maximum DAC power with factor {factor}!", self.output)
                        # Save tone power comb (if necessary)
                        if gen_amps or self.drone_cfg[ind]['tones']['generation']['rescale_power']: rfsoc_io.save_array_board(c, amp_path, np.array(tone_powers).tolist())
                        break
                    # Rescale tone power (if 'rescale_power' is True)
                    elif self.drone_cfg[ind]['tones']['generation']['rescale_power']:
                        rfsoc_io.send_msg('DEBUG', f"The custom comb has points(s) with output power {comb_max} which exceeds the maximum DAC power of {max_power} when factor is {factor}! Rescaling to lower tone power.", self.output)
            # If loop finished successfully, no valid phi/amp comb was found and continuing with comb which exceeds the maximum DAC power at one or more points
            else:
                rfsoc_io.send_msg('WARNING', f"The custom comb has point(s) with output power which exceeds the maximum DAC power of {max_power}!", self.output)
                
            rfsoc_io.send_msg('INFO', f"Saved custom comb for drone {com}!", self.output)

            # Edit comb saved in drone config files
            # -------------------------------------
            # Frequency comb
            self.edit_config(self.drone_cfg[ind], 'tone_freqs', tone_freqs)

            # Amplitude comb
            if self.drone_cfg[ind]['tones']['dB']:
                # Convert tone powers to dB if original config file had dB tone powers
                tone_powers = utils.convert_to_dB(tone_powers).tolist()
            self.edit_config(self.drone_cfg[ind], 'tone_powers', tone_powers)

            # Phase comb
            self.edit_config(self.drone_cfg[ind], 'tone_phis', tone_phis)

            # Number of tones
            self.edit_config(self.drone_cfg[ind], 'num_tones', tone_num)
        return tone_freqs, tone_powers, tone_phis

    def _get_curr_comb(self, com):
        ''' 
        Get the current comb of RFSoC drone

        Parameters:
            com (str): Board.Drone ID
        '''

        # Parse com information 
        bid, drid = com.split('.')
        bip = self.io_cfg['boards'][f'b{bid}']['board_ip']

        # Get ssh key
        key = self.io_cfg['file_paths']['ssh_key']

        # Define directory where sweep combs are saved
        comb_dir = self.drone_dir / f'drone{drid}' / 'comb'

        # Get most recent comb frequency, amplitude, and phase files
        # ----------------------------------------------------------
        with rfsoc_io.get_connection(bip, key) as c:
            freq_file = rfsoc_io.get_most_recent_file_board(c, comb_dir, file_identifier = self.io_cfg["board_file_names"]["comb_freq"])
            amp_file = rfsoc_io.get_most_recent_file_board(c, comb_dir, file_identifier = self.io_cfg["board_file_names"]["comb_amp"])
            phi_file = rfsoc_io.get_most_recent_file_board(c, comb_dir, file_identifier = self.io_cfg["board_file_names"]["comb_phi"])

            # Check that the files exist
            # --------------------------
            if not rfsoc_io.path_exists(c, freq_file):
                rfsoc_io.send_msg('ERROR', f"Could not find tone frequencies of most recent comb for drone {com}.", self.output)
                comb_freq = None
            else:
                comb_freq = rfsoc_io.load_array_board(c, freq_file, output = self.output)

            if not rfsoc_io.path_exists(c, amp_file):
                rfsoc_io.send_msg('ERROR', f"Could not find tone powers of most recent comb for drone {com}.", self.output)
                comb_amp = None
            else:
                comb_amp = rfsoc_io.load_array_board(c, amp_file, output = self.output)

            if not rfsoc_io.path_exists(c, phi_file):
                rfsoc_io.send_msg('ERROR', f"Could not tone phis of most recent comb for drone {com}.", self.output)
                comb_phi = None
            else:
                comb_phi = rfsoc_io.load_array_board(c, phi_file, output = self.output)

        # Return loaded frequency, amplitude, and phase arrays
        return comb_freq, comb_amp, comb_phi

    def _save_timestream(self, ind, data, **kwargs):

        '''
        Save array of timestream data by breaking it into smaller files specified by max_size.
        '''
        from math import ceil

        # Get max file size in Megabytes
        max_file_size = self.io_cfg['io']['max_file_size']*1e6
        for key, value in kwargs.items():
            if key == 'max_file_size':
                max_file_size = value

        # Split timestream into multiple files if it exceeds the max file size
        tstream_size = sys.getsizeof(data) # Get file size in bytes
        tstream_len = np.shape(data)[1]
        trimmed_len = ceil(tstream_len/ceil(tstream_size/max_file_size))

        rfsoc_io.send_msg('DEBUG', f'Timestream size is {tstream_size/1e6} MB!', self.output)

        # Save Timestream
        fname = self.io_cfg["save_file_names"]["stream_fname"]
        fname = f'{fname}_{self.timestamp}'

        tstream_files = list([])
        timestream_dir = self.timestream_dirs[ind]
        for i, j in enumerate(range(0, tstream_len, trimmed_len)):
            tstream_file = timestream_dir / f'{fname}_{i+1:03}.npy'
            np.save(tstream_file , data[:, j:j+trimmed_len])
            tstream_files.append(tstream_file)
            rfsoc_io.send_msg('DEBUG', f'Successfully saved timestream {i+1}!', self.output)

        return tstream_files

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
            self.edit_config(self.drone_cfg[ind], key, arg)

    def _get_com_to(self, **kwargs):
        '''
        Parses a list of drone com_to and sets up these drones. The drone_list specified in the system config is used if no com_to is passed as a key word argument.
        If a com_to is passed as a key word argument, makes sure that the com_to is a list and that all drones are included in the system config drone_list.

        Parameters:
            com_to (list of str): String or list of strings specifying drone com_to
        Returns:
            com_to (list of str): List of drone com_to
            bids (list of str): List of boards in com_to
        '''

        # Set com_to to drone list specified in system config (make a copy so that it does not point to the class drone_list attribute)
        com_to = np.copy(self.drone_list)
        bids = self.board_list
        setup = True

        # Override com_to with that passed as key word argument (if any)
        for key, value in kwargs.items():
            if key == 'com_to':
                com_to = value

                # If com_to is not a list, make it a list
                if not isinstance(com_to, list): com_to = [com_to]
                
                # Get list of boards used
                bids = set()
                for drone in com_to[::-1]:
                    split_str = drone.split('.') # Split drone com_to into bid and drid
                    bids.add(split_str[0]) # Add bid to set of board ids

                    # Replace any board only com_to (e.g. '1') with bid.drid for all four drones
                    if len(split_str) == 1:
                        com_to.remove(drone)
                        for i in range(4):
                            com_to.append(drone + f'.{i + 1}')
                
                # Remove any duplicate entries from list
                com_to = sorted(list(set(com_to)))
                # Check that all drones are in initialized drone list
                extra_drones = set(com_to) - set(self.drone_list)
                if len(extra_drones) > 0: raise ValueError(f'The drones {sorted(list(extra_drones))} are not in system config drone list!')
            elif key == 'setup':
                setup = value

        # Set up drone with specified com_to list
        kwargs['restart'] = False
        if setup: self.setup_drones(**kwargs)
        
        return com_to, sorted(list(bids))
