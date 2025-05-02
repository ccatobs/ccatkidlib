#=================================#
# rfsoc_daq.py               2025 #
# Darshan Patel dp649@cornell.edu #
#=================================#

# Import Python modules
import os
import sys
import ast
import time
import pickle
import numpy as np
import multiprocessing as mp
mp.set_start_method('fork')

from math import floor
from pathlib import Path
from invoke import Responder

# Import local modules
from ccatkidlib.rfsoc.rfsoc_timestream import Streamer
from ccatkidlib.rfsoc_io import header
from ccatkidlib.style import Style
from ccatkidlib.analysis.vna import VNA
from ccatkidlib.analysis.target import Target

import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.pair as pair

class R:
    '''
    Class for data acquisition of Microwave Kinetic Inductance Detectors (MKIDs) using
    Radio Frequency System on a Chip (RFSoC).
    '''

    @utils.method_timer
    def __init__(self, cfg_path = f"{Path(__file__).parent}/system_config.yaml"):
        '''
        Constructor for R. Creates directories for data storage, configures logger, and starts
        RFSoC PCS agent.

        Parameters:
            cfg_path (str) : Path to system configuration file.
        '''
        # Load config files and setup logging
        # -----------------------------------

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
        self.log_dir = self.config_dirs[0].parent
        rfsoc_io.setup_logging(self.log_dir / self.io_cfg['io']['logging_fname'], self.io_cfg['io']['logging_level'], output = self.output)
        rfsoc_io.send_msg('INFO', f'{Style.INVERT}Date: {self.curr_date}; Session: {self.sess_id}{Style.DEFAULT}')

        # Add paths to primecam_readout modules, PCS clients, and ccatkidlib
        # ---------------------------------------------------------------------
        primecam_readout = Path(self.io_cfg['file_paths']['primecam_readout'])

        sys.path.append(str(primecam_readout))
        sys.path.append(self.io_cfg['file_paths']['pcs_dir'])
        sys.path.append(str(primecam_readout / 'alcove_commands'))

        rfsoc_io.send_msg('DEBUG', 'Finished appending file paths!', self.output)

        # Load local modules
        from ocs.ocs_client import OCSClient # Import PCS client module

        # Initialize PCS clients
        # ----------------------
        self.rfsoc = OCSClient(self.io_cfg['pcs_agents']['rfsoc_agent'], args=[])

        # Update Measurment Information 
        # -----------------------------
        if self.io_cfg['io']['influx_output']: self.rfsoc.updateMeasurement(measurement_name = f"{self.io_cfg['name']}, Date: {self.curr_date}, Session ID: {self.sess_id}" , measurement_desc = self.io_cfg['desc'])

        # Setup boards
        # ------------
        if self.io_cfg['init']['initialize_boards']: self.setup_boards()

        # Setup drones
        # ------------
        if self.io_cfg['init']['initialize_drones']: self.setup_drones()
        rfsoc_io.send_msg('INFO', f'Initialized RFSoC agent. Communicating with drones: {self.drone_list}!', self.output)

        # Initialize streamer attribute (As None, since it is only used when g3=False)
        # ----------------------------------------------------------------------------
        self.streamer = None

        # Set NCLO frequency RFSoC drones
        # -------------------------------------------------
        if self.io_cfg['init']['initialize_drones']:
            for com, cfg in zip(self.drone_list, self.drone_cfg):
                # Set NCLO
                rtn = self.rfsoc.setNCLO(com_to = com, f_lo = cfg['tones']['NCLO'], silent=True)
                rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
            rfsoc_io.send_msg('INFO', f"Set NCLO to {[cfg['tones']['NCLO'] for cfg in self.drone_cfg]} MHz for drones: {self.drone_list}!", self.output)

            # Set drone attenuations
            self.set_atten()

        # Get RFSoC temperature and storge space
        # --------------------------------------
        self.get_stats(com_to = self.drone_list, ADC_rms = False)

        # Save init configs
        # -----------------
        rfsoc_io.save_config(self.log_dir / f'init_config_ext_{self.timestamp}.yaml', self.ext_cfg, self.save_cfg)
        rfsoc_io.save_config(self.log_dir / f'init_config_io_{self.timestamp}.yaml', self.io_cfg, self.save_cfg)
        for rfsoc_dir, cfg in zip(self.config_dirs, self.drone_cfg):
            rfsoc_io.save_config(rfsoc_dir / f'init_config_drone_{self.timestamp}.yaml', cfg, self.save_cfg)

    #################
    # Setup Methods #
    #################

    @header
    @utils.method_timer
    def setup_boards(self, **kwargs):
        '''
        (Re)initialize RFSoC boards one at time and run queen commands to safely set accumulation length and start timestreaming.
        '''

        # Get which boards to initialize (only initialize boards with at least on running drone)
        # --------------------------------------------------------------------------------------
        kwargs['setup'] = False
        com_to, boards = self._get_com_to(**kwargs)
        NCLOs = []

        # Iterate over boards
        # -------------------
        for board in boards:
            # Create Fabric connection to board
            # ---------------------------------
            bip = self.io_cfg['boards'][f'b{board}']['board_ip']
            ssh_key = self.io_cfg['file_paths']['ssh_key']
            with rfsoc_io.get_connection(bip, ssh_key, output=self.output) as c:

                # Restart startup board service on board
                # --------------------------------------
                cmd = 'systemctl restart startup_board.service'
                rfsoc_io.send_msg('INFO', f'Initializing board {board}!', self.output)

                sudo_responder = Responder(pattern=r'\[sudo\] password:', response=f'xilinx\n') # Set up responder to run cmd with sudo
                stdout = c.sudo(cmd, hide=True, watchers=[sudo_responder], pty=True)
                rfsoc_io.send_msg('DEBUG', stdout, self.output)
                rfsoc_io.send_msg('INFO', f'Finished initializing board {board}!', self.output)

                # Restart all drones
                # ------------------
                self.setup_drones(com_to=[f'{board}.1', f'{board}.2', f'{board}.3', f'{board}.4'], restart = True)

                # Initialize drones in safe state for streaming (one at a time)
                # -------------------------------------------------------------
                # Set NCLOs
                rfsoc_io.send_msg('INFO', f'Setting NCLOs for board {board}!', self.output)
                for i in range(4):
                    com = f'{board}.{i+1}'
                    try:
                        ind = self.drone_list.index(com)
                        NCLO = self.drone_cfg[ind]['tones']['NCLO']
                        NCLOs.append(NCLO)
                    except:
                        NCLO = 500
                    rtn = self.rfsoc.setNCLO(com_to = com, f_lo = NCLO, silent=True)
                    rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
                    time.sleep(2)

                # Write VNA combs
                rfsoc_io.send_msg('INFO', f'Writing VNA combs for board {board}!', self.output)
                for i in range(4):
                    com = f'{board}.{i+1}'
                    rtn = self.rfsoc.writeNewVnaComb(com_to = com, silent=False)
                    rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
                    time.sleep(10)

                # Turn on timestreams
                rfsoc_io.send_msg('INFO', f'Turning timestreams on for board {board}!', self.output)
                for i in range(4):
                    com = f'{board}.{i+1}'
                    ret = self.rfsoc.timestreamOn(com_to = com, on = True, silent=False)
                    rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
                    time.sleep(10)
        ret = self.rfsoc.timestreamOn(com_to = None, on = False, silent=False) # Turn off all timestreams
        rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
        rfsoc_io.send_msg('INFO', f"Set NCLOs to {NCLOs} MHz for drones: {self.drone_list}!", self.output)

    def setup_drones(self, **kwargs):
        '''
        Setup (start, stop, or restart) drones specified by com_to.

        Parameters:
            com_to (list of str): List of drones to setup
            restart      (bool) : Whether to restart already running drones
        '''

        # Parse key word arguments
        # ------------------------
        com_to = self.drone_list
        restart = self.io_cfg['init']['restart']

        for key, value in kwargs.items():
            if key == 'restart':
                restart = value
            elif key == 'com_to':
                com_to = value

        # Edit master_drone_config.yaml in primecam_readout to match drone_list
        # ---------------------------------------------------------------------
        # Retrieve and load master drone config file
        master_drone_file = self.io_cfg['file_paths']['master_drone_list']
        master_drone = rfsoc_io.load_config(master_drone_file)

        # Loop through all boards
        for board in self.all_boards:
            # Loop through all drones
            for i in range(4):
                # Set to_run for drone (True if supposed to be running, False otherwise)
                com = f'{board}.{i+1}' # bid.drid
                master_drone[com]['ip'] = self.io_cfg['boards'][f'b{board}']['board_ip']
                master_drone[com]['to_run'] = com in com_to
        # Save edited config
        rfsoc_io.save_config(master_drone_file, master_drone)

        # Setup drones
        # ------------
        wait = False
        running_drones = []
        # Loop through all boards
        for board in self.all_boards:
            # Loop through all drones
            for i in range(4):
                com = f'{board}.{i+1}' # bid.drid
                rtn = self.rfsoc.action(com_to=com, action='status') # Get drone status

                # Parse OCS reply to get whether drone is currently running and if it supposed to be running
                # ------------------------------------------------------------------------------------------
                ip, to_run, running = ast.literal_eval(rtn.session['messages'][1][1].split(': ')[1])
                rfsoc_io.send_msg('PCS', f"{rtn.session}", self.output)

                # Start or Restart drones as appropriate
                # --------------------------------------
                if to_run and not running: # Start drones that are not running but should be running
                    rtn = self.rfsoc.action(com_to=com, action='start')
                    rfsoc_io.send_msg('INFO', f"Starting drone {com}...", self.output)
                    wait = True
                elif to_run and restart: # Restart drones if restart = True
                    rtn = self.rfsoc.action(com_to=com, action='restart')
                    rfsoc_io.send_msg('INFO', f"Restarting drone {com}...", self.output)
                    wait = True
                elif running:
                    running_drones.append(com)

        # Wait if any of the drones were started/restarted and make sure they are running
        # -------------------------------------------------------------------------------
        if wait:
            rfsoc_io.wait(25, output = self.output, desc = f"For drones to start")

            # Check that drones were properly started/restarted
            # -------------------------------------------------
            # Loop through all boards
            for board in self.all_boards:
                # Loop through all drones
                for i in range(4):
                    com = f'{board}.{i+1}' # bid.drid
                    rtn = self.rfsoc.action(com_to=com, action='status') # Get drone status

                    # Parse OCS reply to get whether drone is currently running and if it supposed to be running
                    # ------------------------------------------------------------------------------------------
                    ip, to_run, running = ast.literal_eval(rtn.session['messages'][1][1].split(': ')[1])
                    rfsoc_io.send_msg('PCS', f"{rtn.session}", self.output)

                    if running:
                        running_drones.append(com)
                    elif to_run and not running:
                        rfsoc_io.send_msg('ERROR', f'Failed to start drone {com}')

        rfsoc_io.send_msg('INFO', f"Drones {running_drones} are currently running!", self.output)

    def load_system_config(self, cfg_path = f"{Path(__file__).parent}/system_config.yaml"):
        '''
        Load the system and drone config files and setup file directory structure for saving data.

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

        self.all_boards = [board[1:] for board in self.io_cfg['boards']]

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
                # -----------------------------------------------------------------------
                drone_cfgs = self.drone_cfg[-1]
                num_drones = len(drone_cfgs)
                # Iterate through list of drone_cfs backwords (since elements are removed during the loop)
                for i, cfg in enumerate(drone_cfgs[::-1]):
                    # Remove drone config if drone not specified in drone list
                    if not (cfg['com_to'] in self.drone_list):
                        drone_cfgs.pop(num_drones - i - 1)

        # Flatten list of drone configs to match drone_list shape
        self.drone_cfg = [cfg for board in self.drone_cfg for cfg in board]

        # Assign commonly used parameters as class attributes
        # ---------------------------------------------------
        # Whether to run commands in parallel
        self.parallel_boards = self.io_cfg['runtime']['parallel_boards']
        self.parallel_drones = self.io_cfg['runtime']['parallel_drones']

        # IO parameters for saving and printing data
        self.output = self.io_cfg['io']['terminal_output']
        self.save_cfg = self.io_cfg['io']['save_config_copy']
        self.save_data = self.io_cfg['io']['save_data']

        # Commonly used file paths
        self.data_dir = Path(self.io_cfg['file_paths']['data_dir'])
        self.drone_dir = Path(self.io_cfg['file_paths']['drone_dir'])
        self.tmp_data_dir = Path(self.io_cfg['file_paths']['primecam_readout']) / 'tmp'
        self.tmp_dir = Path(__file__).parent / '..' / '..' / 'tmp'
        self.g3_dir = Path(self.io_cfg['file_paths']['g3_dir'])

        # Save session ID to config files
        # -------------------------------
        self._edit_config(self.ext_cfg, 'sess_id', self.sess_id)
        self._edit_config(self.io_cfg, 'sess_id', self.sess_id)
        for cfg in self.drone_cfg:
            self._edit_config(cfg, 'sess_id', self.sess_id)

        # Create file directory structure for saving data
        # -----------------------------------------------
        new_dir_paths = rfsoc_io.create_book(self.curr_date, self.sess_id, self.drone_list, self.data_dir, output = self.output)

        # Assign data directories as class attributes
        self.config_dirs, self.targ_dirs, self.timestream_dirs, self.vna_dirs = new_dir_paths

    @header
    @utils.method_timer
    def set_atten(self, drive = None, sense = None, **kwargs):
        '''
        Set drive/sense attenuations of RFSoC board frontend attenuations.

        Keyword Arguments:
            com_to  (list of str) : List of drones for which to set attenuation
            drive (list of float) : Values of drive (DAC) attenuations in dB (must be between 0 and 31.75)
            sense (list of float) : Values of sense (ADC) attenuations in dB (must be between 0 and 31.75)
        '''

        def _set_atten(com_to = None, direction = None, atten = None):
            '''
            Internal function for setting drone attenuations.

            Parameters:
                com_to    (list of str) : List of drone com_to for which attenuations should be set
                direction (list of str) : Which attenuator to set. Options are 'drive' for DAC and 'sense' for ADC
                atten   (list of float) : List of attenuations. Attenuation must be a float between 0 and 31.75
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
                self.rfsoc.setAtten(com_to = com, direction = direction, atten = att, silent=True)
            rfsoc_io.send_msg('INFO', f'Successfully set {direction} attenuation to {attens} for drones {com_to}!', self.output)

            return attens

        # Specify which attenuators to change
        com_to, _ = self._get_com_to(**kwargs)

        # Set drive attenuation
        drive_attens = _set_atten(com_to = com_to, direction = 'drive', atten = drive)

        # Set sense attenuation
        sense_attens = _set_atten(com_to = com_to, direction = 'sense', atten = sense)

        return drive_attens, sense_attens

    @header
    @utils.method_timer
    def write_config_comb(self, **kwargs):
        '''
        Write custom comb using comb parameters from config file. Config file parameters are
        superseceded by key word argument parameters. If no parameters are passed through the config file
        or as key word arguments, the most recent comb is used instead.

        Parameters:
            tone_freqs  (float | list | str) : Custom tone frequencies (Hz)
            tone_powers (float | list | str) : Custom tone powers
            tone_phis   (float | list | str) : Custom tone phases
        '''

        def _write_targ_comb(com, *args, **kwargs):
            '''
            Internal function for writing target comb using custom list.

            Parameters:
                com (str) : Drone for which custom comb should be written
            Returns:
                rtn : OCS reply object

            '''
            rtn = self.rfsoc.writeTargCombFromCustomList(com_to = com, silent=False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
            return rtn

        # Parse key word arguments
        # ------------------------
        com_to, _ = self._get_com_to(**kwargs)

        tone_freqs    = self._get_drone_args(com_to, ['tones', 'tone_freqs'])
        tone_powers   = self._get_drone_args(com_to, ['tones', 'tone_powers'])
        tone_phis     = self._get_drone_args(com_to, ['tones', 'tone_phis'])
        rescale_power = self._get_drone_args(com_to, ['tones', 'generation', 'rescale_power'])
        gen_attempts  = self._get_drone_args(com_to, ['tones', 'generation', 'gen_attempts'])

        # Evaluate kwargs
        for key, value in kwargs.items():
            if key == 'tone_freqs':
                tone_freqs = self._parse_args(com_to, value)
            elif key == 'tone_powers':
                tone_powers = self._parse_args(com_to, value)
            elif key == 'tone_phis':
                tone_phis = self._parse_args(com_to, value)
            elif key == 'rescale_power':
                rescale_power = self._parse_args(com_to, value)
                self._set_drone_args(com_to, ['tones', 'generation', 'rescale_power'], rescale_power)
            elif key == 'gen_attempts':
                gen_attempts = self._parse_args(com_to, value)
                self._set_drone_args(com_to, ['tones', 'generation', 'gen_attempts'], gen_attemptss)

        # Write custom comb for each drone
        # --------------------------------
        for com, rescale, attempts, freq, power, phi in zip(com_to, rescale_power, gen_attempts, tone_freqs, tone_powers, tone_phis):
            self._write_custom_comb(com, rescale_power = rescale, gen_attempts = attempts, tone_freqs = freq, tone_powers = power, tone_phis = phi)

        # Write sweep comb based on custom parameters
        # -------------------------------------------
        rtn = self._run_parallel(_write_targ_comb, **kwargs)
        rfsoc_io.send_msg('INFO', f'Sucessfully wrote custom comb for drones {com_to}!', self.output)
        return rtn

    ######################
    # Monitoring Methods #
    ######################

    @header
    @utils.method_timer
    def get_stats(self, space = True, temps = True, ADC_rms = True, **kwargs) -> None:
        '''
        Get RFSoC storage space, temperatures, and RMS power at drone ADCs.

        Parameters:
            space (bool): Whether to get RFSoC storage space
            temps (bool): Whether to get RFSoC temperatures
            ADC_rms (bool): Whether to get RMS power at ADCs
        '''
        if not self.rfsoc.feedMonitor.status().session['op_code'] == 2:
            self.rfsoc.feedMonitor.start()
            rfsoc_io.wait(60, desc="HK Feed Monitor Starting")

        if space: avail_spaces = self.get_avail_space(**kwargs)
        if temps: ps_temps, pl_temps = self.get_temps(**kwargs)
        if ADC_rms: rms_list = self.get_ADC_rms(**kwargs)

    def get_ADC_rms(self, **kwargs):
        '''
        Get the root mean squared (RMS) power at the analog to digital converter (ADC) for
        drones specified by com_to.

        Parameters:
            com_to     (list of str) : List of drones
        Returns:
            rms_list (list of float) : List of ADC rms values, sorted by bid.drid
        '''

        def _getSnapData(com, *args, **kwargs):
            '''
            Get Snap data at the analog to digital converter of the specified drone.

            Parameters:
                com (str) : Drone for which to get Snap data
            Returns:
                data (dict) : Data dictionary returned by getSnapData primecam_readout function (pickled)
            '''
            rtn = self.rfsoc.getSnapData(com_to = com, mux_sel = 0, silent = False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
            return rtn.session['messages'][1][1][:-3].split('}, ')

        com_to, _ = self._get_com_to(**kwargs)

        # Take data one board at a time (all drones transfers too much data through OCS)
        kwargs['parallel_boards'] = 1

        # Get snap data of drones
        rtns = self._run_parallel(_getSnapData, **kwargs)
        rtns = np.array([drone for drones in rtns for drone in drones])
        #rtns = np.array(self._run_parallel(_getSnapData, **kwargs).flatten()

        # Convert I, Q Snap data into ADC rms
        # -----------------------------------
        rms_list, inds = [], []
        for rtn in rtns:
            data_dic = pickle.loads(ast.literal_eval(rtn.split(': ')[-1])) # Load pickled data dictionary
            ind = self.drone_list.index(f"{data_dic['bid']}.{data_dic['drid']}") # Determine which drone the Snap data corresponds to
            inds.append(ind)

            # From primecam_readout alcove_base getADCrms function
            # ----------------------------------------------------
            I, Q = data_dic['data'] # Get I, Q Snap data
            z = I + 1j * Q # Convert to complex number
            rms = float(np.real(np.sqrt(np.mean(z * np.conj(z))))) # Calculate RMS value at the ADC
            rms_list.append(rms)

            self._edit_config(self.drone_cfg[ind], 'ADC_RMS', rms)  # Add ADC rms to drone config

        # Create list of ADC rms values for all drones, sort by drone bid.drid
        rms_list = [rms for _, rms in sorted(zip(inds, rms_list))]
        rfsoc_io.send_msg('INFO', f'RMS power at the ADC is {rms_list} for drones {com_to}!')
        return rms_list

    def get_avail_space(self, **kwargs):
        '''
        Check the available storage space on RFSoC board and clean the board drone directories if storage space is below specified threshold.

        Parameters:
            com (str): Bid of board to check available space
        '''
        def _get_space(bids):
            space_dict = self.rfsoc.feedMonitor.status().session['data']['drone_free_spaces_GB']['data']
            avail_spaces = []
            for bid in bids:
                avail_space = space_dict[f'spc_{bid}_1']
                avail_spaces.append(avail_space)
                self._edit_config(self.ext_cfg, ['boards', f'b{bid}', 'avail_space'], avail_space, append = True)
            rfsoc_io.send_msg('INFO', f'Storage space is {avail_spaces} GB for boards {bids}!', self.output)
            return avail_spaces

        com_to, bids = self._get_com_to(**kwargs)

        threshold = self.io_cfg['clean']['threshold']  # Threshold after which to send warnings (in GB)
        olderThanDaysAgo = self.io_cfg['clean']['olderThanDaysAgo'] # Days
        ftype = self.io_cfg['clean']['ftype']

        for key, value in kwargs.items():
            if key == 'threshold':
                threshold = value
            elif key == 'olderThanDaysAgo':
                olderThanDaysAgo = value
            elif key == 'ftype':
                ftype = value

        avail_spaces = _get_space(bids)

        # Clean boards if threshold is greater than zero
        if threshold > 0:
            boards_to_clean = [bid for avail_space, bid in zip(avail_spaces, bids) if avail_space < threshold]
            if len(boards_to_clean) > 0:
                drones_to_clean = [com for com in com_to if com.split('.')[0] in boards_to_clean]
                rfsoc_io.send_msg('INFO', f'Cleaning drones {drones_to_clean}...', self.output)
                self.rfsoc.cleanBoardDroneDirs(com_to=drones_to_clean, testing=False, leave_latest=True, ftype=ftype, olderThanDaysAgo=olderThanDaysAgo)
                rfsoc_io.send_msg('INFO', f'Finished cleaning drones {drones_to_clean}', self.output)
                avail_spaces = _get_space(bids)
        return avail_spaces

    def get_temps(self, **kwargs):
        '''
        Get RFSoC ps processor and pl fabric temperatures.
        '''
        def _get_temps(bids, loc):
            temps = []
            for bid in bids:
                temp = float(np.mean([temp_dict[f'temp_{bid}_{drid + 1}_{loc}'] for drid in range(4)]))
                temps.append(temp)
                self._edit_config(self.ext_cfg, ['boards', f'b{bid}', f'{loc}_temp'], temp, append = True)
            rfsoc_io.send_msg('INFO', f'{loc} temperatures are {temps} \xb0C for boards {bids}!', self.output)

        com_to, bids = self._get_com_to(**kwargs)
        temp_dict = self.rfsoc.feedMonitor.status().session['data']['drone_temperatures_C']['data']
        ps_temps = _get_temps(bids, 'ps')
        pl_temps = _get_temps(bids, 'pl')

        return ps_temps, pl_temps

    ############################
    # Data Acquisition Methods #
    ############################
    @header
    @utils.method_timer
    def take_vna_sweep(self, **kwargs):
        '''
        Take a vector network analyzer (VNA) sweep using RFSoC.

        Parameters:
            NCLO (int): Numerically controlled local oscillator frequency in MHz. Center frequency of VNA sweep.
        Returns:
            output (str): VNA sweep file path
        '''
        @utils.function_timer
        def _write_vna_comb(com, *args, **kwargs):
            rtn = self.rfsoc.writeNewVnaComb(com_to = com, silent = False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
            return rtn

        @utils.function_timer
        def _take_vna_sweep(com, *args, **kwargs):
            sweep_steps = args[0]
            rtn = self.rfsoc.vnaSweep(com_to = com, sweep_steps = sweep_steps, silent = False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
            return rtn

        # Get com_to
        com_to, boards = self._get_com_to(**kwargs)

        # Parse key word arguments
        write_comb = True
        sweep_steps = self._get_drone_args(com_to, ['tones', 'sweep_steps'])
        for key, value in kwargs.items():
            if key == 'NCLO':
                NCLO = self._parse_args(com_to, value)
                for com, N in zip(com_to, NCLO):
                    rtn = self.rfsoc.setNCLO(com_to = com, f_lo = N, silent=True)
                    rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)

                self._set_drone_args(com_to, "NCLO", NCLO)
                rfsoc_io.send_msg('INFO', f'Set NCLO to {NCLO} MHz for drones {com_to}!', self.output)
            elif key == 'write_comb':
                write_comb = value
            elif key == 'sweep_steps':
                sweep_steps = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "sweep_steps", sweep_steps)

        # Write VNA comb
        # --------------
        if write_comb:
            rfsoc_io.send_msg('INFO', 'Writing new VNA combs...', self.output)
            self._run_parallel(_write_vna_comb, **kwargs)
            rfsoc_io.send_msg('INFO', f'Successfully wrote new VNA combs for drones {com_to}!', self.output)
            time.sleep(1) # Wait before taking sweep (not waiting can affect sweep quality, unsure if this is still the case: need to test)

        # Take VNA sweep
        # --------------
        self.timestamp = str(time.time()).split('.')[0]
        rfsoc_io.send_msg('INFO', 'Taking VNA sweeps...', self.output)

        sweep_dict = self._group_args(com_to, sweep_steps)
        for sweep_steps, com in sweep_dict.items():
            args = [int(sweep_steps)]
            kwargs['com_to'] = com
            self._run_parallel(_take_vna_sweep, *args, **kwargs)
        kwargs['com_to'] = com_to

        rfsoc_io.send_msg('INFO', f'Finished taking VNA sweeps for drones {com_to} with timestamp {self.timestamp}!', self.output)
        self.get_stats(**kwargs)

        # Save VNA sweep data
        # -------------------
        vna_files, vna_paths = self._save_sweep(com_to, 'vna', self.vna_dirs)

        if self.save_data:
            return vna_paths
        else:
            return vna_files

    @header
    @utils.method_timer
    def take_target_sweep(self, **kwargs):
        '''
        Take a target sweep around the specified tones.

        Parameters:
            chan_bw (float): Bandwidth of sweep around each tone in MHz.
            sweep_steps (int): Number of points per tone
            write_comb (bool): Whether to write new comb or use current comb
            tone_freqs: Frequencies at which to place tones
            tone_powers: Readout power of tones
            tone_phis: Phase of tones
        Returns:
            output (str): File path of target sweep
        '''

        def _take_targ_sweep(com, *args, **kwargs):
            sweep_steps, chan_bw = args
            rtn = self.rfsoc.targetSweep(com_to = com, sweep_steps = sweep_steps, chan_bw = chan_bw, silent = False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
            return rtn

        com_to, boards = self._get_com_to(**kwargs)

        # Evaluate kwargs
        write_comb = False
        sweep_steps = self._get_drone_args(com_to, ['tones', 'sweep_steps'])
        chan_bw = self._get_drone_args(com_to, ['tones', 'chan_bw'])
        for key, value in kwargs.items():
            if key == 'write_comb':
                write_comb = value
            elif key == 'NCLO':
                NCLO = self._parse_args(com_to, value)
                for com, N in zip(com_to, NCLO):
                    rtn = self.rfsoc.setNCLO(com_to = com, f_lo = N, silent=True)
                    rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)

                self._set_drone_args(com_to, "NCLO", NCLO)
                rfsoc_io.send_msg('INFO', f'Set NCLO to {NCLO} MHz for drones {com_to}!', self.output)
            elif key == 'sweep_steps':
                sweep_steps = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "sweep_steps", sweep_steps)
            elif key == 'chan_bw':
                chan_bw = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "chan_bw", chan_bw)

        # Write new target sweep comb
        if write_comb:
            self.write_config_comb(**kwargs)
            time.sleep(1) # Wait before taking sweep (not waiting can affect sweep quality)

        # Get timestamp
        self.timestamp = str(time.time()).split('.')[0]
        rfsoc_io.send_msg('INFO', 'Taking target sweeps!', output = self.output)

        sweep_dict = self._group_args(com_to, sweep_steps, chan_bw)

        for arg, com in sweep_dict.items():
            sweep_steps, chan_bw = arg.split(', ')
            args = [int(sweep_steps), float(chan_bw)]
            kwargs['com_to'] = com
            self._run_parallel(_take_targ_sweep, *args, **kwargs)
        kwargs['com_to'] = com_to

        rfsoc_io.send_msg('INFO', f'Finished taking target sweeps for drones {com_to} with timestamp {self.timestamp}!', self.output)
        self.get_stats(**kwargs)

        # Save target sweep data
        # ----------------------
        targ_files, targ_paths = self._save_sweep(com_to, 'targ', self.targ_dirs)

        if self.save_data:
            return targ_paths
        else:
            return targ_files

    @header
    @utils.method_timer
    def take_timestream(self, t_sec, **kwargs):
        '''
        Take timestream data using RFSoC.

        Parameters:
            time (int): Length of timestream in seconds
            write_comb (bool): Whether to re-write comb
            tone_freqs: Frequencies at which to place tones
            tone_powers: Readout power of tones
            tone_phis: Phase of tones
        Return:
            output (list of str): Timestream file paths if save_data = True
        Returns:
            output: Return complex S21 timestream data in array
        '''

        def _take_timestream_python(com, *args, **kwargs):
            import gc
            from multiprocessing import Process, Queue, Manager, Value


            def _save_timestream_python(com_to, start_time, time_diff, packet_data, tstream_seg, tstream_outs):
                packs, addrs = zip(*packet_data)
                packs = [bytearray(pack) for pack in packs]
                ips = np.array([list(addr) for addr in addrs])[:,0]

                gc.collect()
                #ports = addrs[:, 1]

                diff = time_diff.value
                diff = diff if not diff == -1 else None
                data, auxs, time_diff.value = self.streamer.parse_packets(packs, timestamp=start_time, time_diff=diff)
                del packs, addrs
                gc.collect()


                drone_ips = self._get_drone_args(com_to, 'udp_source_ip')
                drone_data = [[] for _ in range(len(com_to))]
                for dat, aux, ip in zip(data, auxs, ips):
                    drone_ind = drone_ips.index(str(ip))
                    drone_data[drone_ind].append(np.append(np.int64(aux[2]), np.append(aux[3], np.array(dat).astype('int64'))))

                del data, auxs, ips
                gc.collect()

                tstream_out = []
                for com, data in zip(com_to, drone_data):
                    if not len(data) == 0:
                        # Convert data into I, Q
                        data = np.array(data).T.astype('int64')
                        gc.collect()

                        if self.save_data:
                            ind = self.drone_list.index(com)

                            # Save Timestream
                            # ---------------
                            fname = self.io_cfg["save_file_names"]["timestream"]
                            fname = f'{fname}_{self.timestamp}'

                            timestream_dir = self.timestream_dirs[ind]
                            tstream_file = timestream_dir / f'{fname}_{tstream_seg:03d}.npy'
                            np.save(tstream_file , data)
                            tstream_out.append(tstream_file)
                        else:
                            tstream_out.append(data)
                        #packet_counts = data[:, 0].T
                        #ts = data[:, 1].T
                        #I, Q = data[:,2::2].T, data[:,3::2].T
                    else:
                        tstream_out.append(None)
                del drone_data
                gc.collect()

                for out, outs in zip(tstream_out, tstream_outs):
                    outs.append(out)
                return tstream_out

            t_sec = args[0]
            finished = 0
            com_arr = [str(drone) for drone in ast.literal_eval(com)]

            rotation_time = self.io_cfg['timestream']['file_rotation_time']
            save_interval = self.io_cfg['timestream']['save_interval']

            for key, value in kwargs.items():
                if key == 'file_rotation_time':
                    rotation_time = value
                elif key == 'save_interval':
                    save_interval = value

            last_save = float(self.timestamp)
            packet_data = []
            saves = []

            q = Queue(maxsize=0)
            packet_capture = Process(target=self.streamer.capture_packets, args =(t_sec,), kwargs={'q': q})
            timer = Process(target=rfsoc_io.wait, args=(t_sec,), kwargs={'output': self.output, 'desc': f'Taking {t_sec} second timestream for {com}'})

            with Manager() as manager:
                tstream_outs = manager.list([manager.list([]) for _ in range(len(com_arr))])
                time_diff = Value('d', -1)
                tstream_seg = 0

                rtn = self.rfsoc.timestreamOn(com_to = com, on = True, silent=True)
                timer.start()
                start_time = time.time()
                time.sleep(0.25)
                packet_capture.start()

                while timer.is_alive() or finished == 1:
                    while True:
                        try:
                            packet_data.append(q.get_nowait())
                        except:
                            break
                    curr_time = time.time()
                    if (curr_time - last_save > save_interval) or not timer.is_alive():
                        last_save = curr_time
                        saves.append(Process(target=_save_timestream_python, args=(com_arr, start_time, time_diff, packet_data, tstream_seg, tstream_outs,)))
                        saves[-1].start()
                        tstream_seg += 1
                        packet_data.clear()
                    if not timer.is_alive: finished += 1

                packet_capture.terminate()
                time.sleep(0.1)
                packet_capture.close()

                rtn = self.rfsoc.timestreamOn(com_to = com, on = False, silent=True)
                rfsoc_io.send_msg('INFO', f"Finished taking {t_sec} seconds of timestream data with timestamp {self.timestamp}!")

                # Wait for all subprocesses to finish executing
                timer.join()
                for save in saves: save.join()

                num_files = int(np.round(rotation_time/save_interval))
                combine_ps = []

                for i, files in enumerate([list(out) for out in tstream_outs]):
                    p = Process(target=rfsoc_io.combine_npy, args=(files,num_files,i,tstream_outs,))
                    combine_ps.append(p)
                    p.start()

                for p in combine_ps: p.join()

                return list(tstream_outs)

        def _take_timestream_g3(com, *args, **kwargs):
            t_sec = args[0]
            rtn = self.rfsoc.timestreamOn(com_to = com, on = True, silent=True)
            rfsoc_io.wait(t_sec, output = self.output, desc = f'Taking {t_sec} second timestream for {com}')
            rtn = self.rfsoc.timestreamOn(com_to = com, on = False, silent=True)
            rfsoc_io.send_msg('INFO', f"Finished taking {t_sec} seconds of timestream data with timestamp {self.timestamp}!")
            return rtn

        def _save_timestream_g3(com_to):
            date = self.timestamp[0:5]

            fname = self.io_cfg["save_file_names"]["timestream"]
            fname = f'{fname}_{self.timestamp}'

            stream_paths = []
            for com in com_to:
                bid, drid = com.split('.')
                ind = self.drone_list.index(com)
                timestream_dir = self.timestream_dirs[ind]

                g3_tstream_dir = self.g3_dir / date / f'rfsoc{int(bid):02d}_drone{drid}'
                g3_file = rfsoc_io.get_most_recent_file(g3_tstream_dir, f'r{int(bid):02d}d{drid}*', output = self.output, time_past = np.inf)

                tstream_parts = g3_file.stem.split('_')
                tstream_name = '_'.join(tstream_parts[0:-1])
                try:
                    tstream_num = int(tstream_parts[-1])
                except:
                    tstream_num = -1

                g3_files = []
                while tstream_num >= 0 and rfsoc_io.get_creation_time(g3_file) - float(self.timestamp) > 0:
                    g3_files.append(g3_file)
                    tstream_num -= 1
                    g3_file = g3_tstream_dir / (tstream_name + f'_{tstream_num:03d}.g3')

                tstream_file = timestream_dir / f'{fname}.txt'
                with open(tstream_file, 'w') as file:
                    file.writelines(f"{g3_file}\n" for g3_file in g3_files[::-1])

                stream_paths.append(tstream_file)
            return stream_paths

        com_to, _ = self._get_com_to(**kwargs)

        # Parse key word arguments
        g3 = self.io_cfg['timestream']['g3'] # Whether to take g3 timestream
        write_comb = False
        save_data = self.save_data

        for key, value in kwargs.items():
            if key == 'write_comb':
                write_comb = value
            elif key == 'g3':
                g3 = value
            elif key == 'save_data':
                save_data = value

        # Write custom comb
        # -----------------
        if write_comb:
            self.write_config_comb(**kwargs)
            time.sleep(1)

        # Turn off all currently running timestreams
        # ------------------------------------------
        rtn = self.rfsoc.timestreamOn(on = False, silent=True)

        # Take timestreams
        # ----------------
        rfsoc_io.send_msg('INFO', f'Taking {t_sec} seconds of timestream data!', self.output)
        stream_out = None
        self.timestamp = str(time.time()).split('.')[0]
        args=[t_sec]
        if g3:
            self._run_parallel(_take_timestream_g3, *args, **kwargs)
            if self.save_data:
                time.sleep(1) 
                stream_out = _save_timestream_g3(com_to)
        else:
            if self.streamer is None:
                self.streamer = Streamer(self.io_cfg['timestream']['udp_ip'], self.io_cfg['timestream']['udp_port'])
                rfsoc_io.send_msg('INFO', f"Successfully initialized timestream object using address {self.io_cfg['timestream']['udp_ip']} and port {self.io_cfg['timestream']['udp_port']}!", output = self.output)
            stream_out = self._run_parallel(_take_timestream_python, *args, **kwargs)
            stream_out = [drone for parallel in stream_out for drone in parallel]

        # Get RMS power at the ADC
        # ------------------------
        self.get_stats(**kwargs)

        # Edit and save drone config with comb used for timestream
        # --------------------------------------------------------
        for com in com_to: self._save_curr_comb(com, self.io_cfg['save_file_names']['timestream'])

        # Save ext config
        # ---------------
        self.ext_cfg = rfsoc_io.save_config(self.log_dir / f"{self.io_cfg['save_file_names']['timestream']}_config_ext_{self.timestamp}.yaml", self.ext_cfg, self.save_cfg)

        return stream_out

    ##################
    # Tuning Methods #
    ##################

    @header
    @utils.method_timer
    def find_detectors(self, new_sweep = True, filter_freqs = True, **kwargs):
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
            '''
            Internal function for writing target comb using resonators found from VNA sweep.

            Parameters:
                com (str) : Drone for which target comb should be written
            Returns:
                rtn1 : OCS reply object from writeTargCombFromVnaSweep function
                rtn2 : OCS reply object from createCustomCombFilesFromCurrentComb function
            '''
            # Write target comb using found resonators
            rtn1 = self.rfsoc.writeTargCombFromVnaSweep(com_to = com, silent = False)
            rfsoc_io.send_msg('PCS', f'{rtn1.session}', self.output)

            return rtn1

        def _write_filtered_comb(com_to, vna_files):
            rtns = []
            for com, vna_file in zip(com_to, vna_files):
                vna = VNA(com = com, sweep_path = vna_file)
                good_resonators, _ = vna.filter_freqs()
                rtn = self.write_config_comb(tone_freqs = good_resonators, tone_powers = 'gen', tone_phis = 'gen', rescale_power = True, gen_attempts = 5)
                rtns.append(rtn)
            return rtns

        # Get com_to list
        com_to, boards = self._get_com_to(**kwargs)
        write_targ_comb = True

        # Evaluate kwargs
        peak_prom_std = self._get_drone_args(com_to, ['det_find', 'peak_prom_std'])
        peak_prom_db  = self._get_drone_args(com_to, ['det_find', 'peak_prom_db'])
        peak_dis      = self._get_drone_args(com_to, ['det_find', 'peak_dis'])
        width_min     = self._get_drone_args(com_to, ['det_find', 'width_min'])
        width_max     = self._get_drone_args(com_to, ['det_find', 'width_max'])
        stitch        = self._get_drone_args(com_to, ['det_find', 'stitch'])
        stitch_bw     = self._get_drone_args(com_to, ['tones', 'sweep_steps'])
        stitch_sw     = self._get_drone_args(com_to, ['det_find', 'stitch_sw'])
        remove_cont   = self._get_drone_args(com_to, ['det_find', 'remove_cont'])
        continuum_wn  = self._get_drone_args(com_to, ['det_find', 'continuum_wn'])
        remove_noise  = self._get_drone_args(com_to, ['det_find', 'remove_noise'])
        noise_wn      = self._get_drone_args(com_to, ['det_find', 'noise_wn'])

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
            elif key == 'sweep_steps':
                stitch_bw = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "sweep_steps", stitch_bw)
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
        old_files = []
        inds = []
        for i, com in enumerate(com_to):
            ind = self.drone_list.index(com)
            # Check if a VNA sweep was taken in the last day
            vna_file = rfsoc_io.get_most_recent_file(self.vna_dirs[ind], f"{self.io_cfg['save_file_names']['vna_sweep'][0]}*", output = self.output, time_past=24*3600)
            if not vna_file.exists() or new_sweep:
                to_sweep.append(str(com))
            else:
                inds.append(i)
                old_files.append(vna_file)
                vna = VNA(com_to=com, sweep_path=vna_file)
                stitch_bw[ind] = utils.dict_get(vna.drone_cfg, 'sweep_steps')

        if len(to_sweep) != 0:
            # Take VNA sweep(s) without saving configs (saved later)
            self.save_cfg = False
            kwargs['com_to'] = to_sweep
            vna_files = self.take_vna_sweep(**kwargs)
            for i, vna_file in zip(inds, old_files): vna_files.insert(i, vna_file) # Add newly old vna files into list of new vna files sorted by drone com_to
            self.save_cfg = True
        rfsoc_io.send_msg('INFO', f"Finding resonators from VNA sweep for drones {com_to}!", output = self.output)
        for i, com in enumerate(com_to):
            # Find resonators from VNA sweep
            rtn = self.rfsoc.findVnaResonators(com_to = com, peak_prom_std = peak_prom_std[i], peak_prom_db = peak_prom_db[i], peak_dis = peak_dis[i],
                                                width_min = width_min[i], width_max = width_max[i], stitch = stitch[i], stitch_bw = stitch_bw[i], stitch_sw = stitch_sw[i],
                                                remove_cont = remove_cont[i], continuum_wn = continuum_wn[i], remove_noise = remove_noise[i], noise_wn = noise_wn[i], silent=False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
        rfsoc_io.send_msg('INFO', f"Finished finding resonators from VNA sweep for drones {com_to}!", output = self.output)

        found_nums = self._save_resonators(com_to, 'vna')

        if write_targ_comb:
            rfsoc_io.send_msg('INFO', f"Writing target combs using found resonators for drones {com_to}!", output = self.output)
            rtn = _write_filtered_comb(com_to, vna_files) if filter_freqs else self._run_parallel(_write_targ_comb, **kwargs)
            rfsoc_io.send_msg('INFO', f'Finished writing target combs for drones {com_to}!', output = self.output)

            self.save_cfg = False
            for com in com_to: self._save_curr_comb(com, None)
            self.save_cfg = True

        return found_nums, vna_files

    @header
    @utils.method_timer
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
            '''
            Internal function for writing target comb using resonators found from target sweep.

            Parameters:
                com (str) : Drone for which target comb should be written
            Returns:
                rtn1 : OCS reply object from writeTargCombFromTargSweep function
                rtn2 : OCS reply object from createCustomCombFilesFromCurrentComb function
            '''
            # Write target comb using found resonators
            rtn1 = self.rfsoc.writeTargCombFromTargSweep(com_to = com, silent = False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)

            return rtn1

        com_to, boards = self._get_com_to(**kwargs)

        # Evaluate kwargs
        stitch_bw = self._get_drone_args(com_to, ['tones', 'sweep_steps'])
        write_targ_comb = True
        # Parse key word arguments
        for key, value in kwargs.items():
            if key == 'sweep_steps':
                stitch_bw = self._parse_args(com_to, value)
                self._set_drone_args(com_to, "sweep_steps", stitch_bw)
            elif key == 'write_targ_comb':
                write_targ_comb = value

        # Take target sweep if new_sweep = True if one does not already exist
        to_sweep = []
        targ_files = []
        for com in com_to:
            ind = self.drone_list.index(com)
            # Check if a target sweep was taken in the last day
            targ_file = rfsoc_io.get_most_recent_file(self.targ_dirs[ind], f"{self.io_cfg['save_file_names']['targ_sweep'][0]}*", output = self.output, time_past=24*3600)
            if not targ_file.exists() or new_sweep:
                to_sweep.append(com)
            else:
                targ_files.append(targ_file)
                target = Target(com_to=com, sweep_path=targ_file)
                stitch_bw[ind] = utils.dict_get(target.drone_cfg, 'sweep_steps')
        if len(to_sweep) != 0:
            # Take target sweep(s) without saving config (saved later)
            self.save_cfg = False
            kwargs['com_to'] = to_sweep
            targ_file = self.take_target_sweep(**kwargs)
            targ_files.extend(targ_file)
            self.save_cfg = True

        rfsoc_io.send_msg('INFO', f"Finding resonators from target sweep for drones {com_to}!", output = self.output)
        for i, com in enumerate(com_to):
            # Find resonators from target sweep
            rtn = self.rfsoc.findTargResonators(com_to = com, stitch_bw = stitch_bw[i], silent=False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}', self.output)
        rfsoc_io.send_msg('INFO', f"Finished finding resonators from target sweep for drones {com_to}!", output = self.output)

        found_nums = self._save_resonators(com_to, 'targ')

        if write_targ_comb:
            rfsoc_io.send_msg('INFO', f"Writing target comb using found resonators for drones {com_to}!", output = self.output)
            self._run_parallel(_write_targ_comb, **kwargs)
            rfsoc_io.send_msg('INFO', f'Finished writing target combs for drones {com_to}!', output = self.output)

            self.save_cfg = False
            for com in com_to: self._save_curr_comb(com, None)
            self.save_cfg = True

        return found_nums, targ_files

    ################
    # Save Methods #
    ################

    def _save_sweep(self, com_to, sweep, dirs):
        '''
        Internal function for saving VNA and target sweep files.

        Parameters:
            com_to (list of str) : List of drones
            sweep          (str) : Type of sweep: 'vna' or 'targ'
            dirs   (list of str) : List of file paths where the sweeps should be saved
        Returns:
            files (list of str)  : List of sweep file paths on RFSoC board(s)
            paths (list of str)  : List of file paths where sweeps were saved

        '''

        # Save sweep(s)
        # -------------
        files, paths = [], []
        ssh_key = self.io_cfg['file_paths']['ssh_key']

        fname = self.io_cfg["save_file_names"][f"{sweep}_sweep"]
        fname = f'{fname}_{self.timestamp}.npy'

        # Iterate over drones
        for com in com_to:
            # Create fabric connection to RFSoC board
            # ---------------------------------------
            ind = self.drone_list.index(com)
            path = dirs[ind] / fname

            bid, drid = com.split('.')
            data = None

            try:
                drone_id = f"*{bid}_{drid}*"
                file = rfsoc_io.get_most_recent_file(self.tmp_data_dir, file_identifier = self.io_cfg['board_file_names'][f'{sweep}']['s21'] + drone_id, output = self.output, time_past=time.time() - float(self.timestamp))

                # Save the data with name specified in system config file
                if self.save_data:
                    data = rfsoc_io.get_array(file, path, action = 'mv', load = False, output = self.output)
                    if data is None: raise FileNotFoundError
                else:
                    data = file if Path(file).exists() else None
            except FileNotFoundError:
                bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
                path = self.drone_dir / f'drone{drid}' / sweep

                with rfsoc_io.get_connection(bip, ssh_key, output = self.output) as c:
                    # Get most recent sweep file in sweep directory
                    file = rfsoc_io.get_most_recent_file_board(c, path, file_identifier = self.io_cfg['board_file_names'][f'{sweep}']['s21'], output = self.output, time_past=time.time() - float(self.timestamp))

                    # Save the data with name specified in system config file
                    if self.save_data:
                        data = rfsoc_io.get_array_board(c, bip, ssh_key, file, path, load = False, output = self.output)
                    else:
                        data = file if rfsoc_io.path_exists(c, file) else None

            # Add file to list of files if data array was successfully copied
            # ---------------------------------------------------------------
            if data is not None:
                files.append(file)
            else:
                files.append(None) # Append None if array was not successfully copied
                continue

            # Save current comb in drone config
            # ---------------------------------
            self._save_curr_comb(com, self.io_cfg['save_file_names'][f'{sweep}_sweep'])

            rfsoc_io.send_msg('DEBUG', f'Successfully copied {sweep} file from drone {com}!', self.output)
            paths.append(path)

        # Save ext config
        self.ext_cfg = rfsoc_io.save_config(self.log_dir / f"{self.io_cfg['save_file_names'][f'{sweep}_sweep']}_config_ext_{self.timestamp}.yaml", self.ext_cfg, self.save_cfg)

        return files, paths

    def _save_resonators(self, com_to, sweep):

        found_nums = []
        for com in com_to:
            # Parse com information
            ind = self.drone_list.index(com)
            bid, drid = com.split('.')

            # Define directory to temporary store resonator files
            res_dir = self.config_dirs[ind] / 'res'
            fname = f"{self.io_cfg['save_file_names'][f'{sweep}_sweep']}_res_{self.timestamp}.npy"


            try:
                drone_id = f"*{bid}_{drid}*"
                res_file = rfsoc_io.get_most_recent_file(self.tmp_data_dir, file_identifier = self.io_cfg["board_file_names"][f'{sweep}']['res'] + drone_id, output = self.output, time_past = time.time() - float(self.timestamp))
                res_path = rfsoc_io.get_array(res_file, res_dir / fname, action = 'mv', load = False, output = self.output)
                if res_path is None: raise FileNotFoundError
            except FileNotFoundError:
                bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
                ssh_key = self.io_cfg['file_paths']['ssh_key']
                path = self.drone_dir / f'drone{drid}' / f'{sweep}'

                with rfsoc_io.get_connection(bip, ssh_key, output = self.output) as c:
                    # Get file with found resonators
                    res_file = rfsoc_io.get_most_recent_file_board(c, path, file_identifier = self.io_cfg["board_file_names"][f'{sweep}']['res'], output = self.output, time_past = time.time() - float(self.timestamp))
                    res_path = rfsoc_io.get_array_board(c, bip, ssh_key, res_file, res_dir / fname, load = False, output = self.output)

            # Save resonator frequncies and number of found resonators to list
            try:
                found_freqs = np.load(res_dir / fname, mmap_mode='r')
                found_num = len(found_freqs)
            except:
                rfsoc_io.send_msg('ERROR', f"Failed to retrieve found resonators file!", self.output)
                found_num = None
            found_nums.append(found_num)

            rfsoc_io.send_msg('INFO', f"Found {found_num} detectors for drone {com}!", self.output)

            self._edit_config(self.drone_cfg[ind], 'found_num_detectors', found_num)
            self._edit_config(self.drone_cfg[ind], 'found_detector_freqs', res_path)
            self.drone_cfg[ind] = rfsoc_io.save_config(self.config_dirs[ind] / f"{self.io_cfg['save_file_names'][f'{sweep}_sweep']}_config_drone_{self.timestamp}.yaml", self.drone_cfg[ind], self.save_cfg)

        self.ext_cfg = rfsoc_io.save_config(self.log_dir / f"{self.io_cfg['save_file_names'][f'{sweep}_sweep']}_config_ext_{self.timestamp}.yaml", self.ext_cfg, self.save_cfg)
        return found_nums

    ########################
    # Arg Handling Methods #
    ########################

    def _group_args(self, com_to, *args):
        arg_list = []
        for arg in zip(*args):
            arg = ', '.join([str(a) for a in list(arg)])
            arg_list.append(arg)

        sweep_dict = {}
        for com, arg in zip(com_to, arg_list):
            com_list = sweep_dict.setdefault(arg, [])
            com_list.append(com)

        return sweep_dict

    def _parse_args(self, com_to, arg):
        '''
        Parse key word drone arguments

        Parameters:
            com_to       (list of str) : List of drones
            arg    (Any | list of Any) : Arugment to parse
        Returns:
            args         (list of Any) : List of parsed args
        '''

        args = None
        try: # Check if argument is a list
            if len(arg) == len(com_to): # Length of argument list should match number of drones
                args = arg
            else:
                rfsoc_io.send_msg('WARNING', f'{arg} is not a valid argument. Must be a single value or match the length of {com_to}!', self.output)
                return None # Return None if argument is invalid
        except:
            try: # Assume argument is a single value and create a list of arugments with length equal to the number of drones
                args = [arg] * len(com_to)
            except:
                rfsoc_io.send_msg('WARNING', f'{arg} is not a valid argument. Must be a single value or match the length of {com_to}!', self.output)
                return None # Return None if argument is invalid
        return args # Return parsed argument

    def _get_drone_args(self, com_to, key):
        '''
        Get drone arguments from drone config files

        Parameters:
            com_to (list of str) : List of drones
            key    (list of str) : List of dictionary key(s) of argument to retrieve from drone config files
        Returns:
            args   (list of Any) : List of values from drone config files corresponding to dictionary key
        '''

        inds = [self.drone_list.index(com) for com in com_to] # Get list of indicies corresponding to drones in com_to
        return [utils.dict_get(self.drone_cfg[ind], key) for ind in inds] # Get dictionary value corresponding to key for each drone config file

    def _set_drone_args(self, com_to, key, args):
        '''
        Set drone arguments in drone config files

        Parameters:
            com_to (list of str) : List of drones
            key            (str) : Dictionary key to set value of
            args   (list of Any) : List of values to set in drone config files
        Returns:
            rtn_list (list of bool) : List of returns from edit_config for each drone
        '''

        # Iterate over drones and args
        rtn_list = []
        for com, arg in zip(com_to, args):
            ind = self.drone_list.index(com) # Get index corresponding to drone in com_to
            rtn = self._edit_config(self.drone_cfg[ind], key, arg) # Set config value of specified key
            rtn_list.append(rtn)
        return rtn_list

    def _get_com_to(self, **kwargs):
        '''
        Parses a list of drone com_to and sets up these drones. The drone_list specified in the system config is used if no com_to is passed as a key word argument.
        If a com_to is passed as a key word argument, makes sure that the com_to is a list and that all drones are included in the system config drone_list.

        Parameters:
            com_to (list of str): String or list of strings specifying drone com_to
        Returns:
            com_to (list of str): List of drone com_to
            bids   (list of str): List of boards in com_to
        '''

        # Set com_to to drone list specified in system config (make a copy so that it does not point to the class drone_list attribute)
        com_to = np.copy(self.drone_list).tolist()
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
                for com in com_to[::-1]:
                    split_str = com.split('.') # Split drone com_to into bid and drid
                    bids.add(split_str[0]) # Add bid to set of board ids

                    # Replace any board only com_to (e.g. '1') with bid.drid for all four drones
                    if len(split_str) == 1:
                        com_to.remove(com)
                        for i in range(4):
                            com_to.append(com + f'.{i + 1}')

                # Remove any duplicate entries and sort list of drones
                com_to = sorted(list(set(com_to)))

                # Check that all drones are in initialized drone list
                extra_drones = set(com_to) - set(self.drone_list) # Get drones in com_to that are not in drone_list
                if len(extra_drones) > 0: raise ValueError(f'The drones {sorted(list(extra_drones))} are not in system config drone list!')
            elif key == 'setup':
                setup = value

        # Set up drone with specified com_to list
        kwargs['restart'] = False
        if setup: self.setup_drones(**kwargs)

        # Return list of drones and list of boards in use
        return com_to, sorted(list(bids))

    def _edit_config(self, cfg, key, value, append = False):
        '''
        Update key in specified configuration file with the specified value.

        Parameters:
            cfg    (dict) : Configuration file to update
            key     (str) : Key that should be updated
            value   (Any) : Value with which to update key
            append (bool) : Whether to append a new key, value pair to config file if key is not found
        Returns:
            done   (bool) : True if key was successfully created or updated.
        '''

        # Edit config file dictionary
        # ---------------------------
        done = utils.dict_set(cfg, key, value)

        # Check if key was successfully updated
        # -------------------------------------
        if done: # If matching key was updated
            rfsoc_io.send_msg('DEBUG', f'Updated key "{key}" with value "{value}" in config file"!')
        elif append: # If key was not found and append=True, add key value pair to dictionary
            cfg[key] = value
            done = True
            rfsoc_io.send_msg('DEBUG', f'Added key "{key}" with value "{value}" to config file!')
        else: # If key was not found and append=False
            rfsoc_io.send_msg('DEBUG', f'Failed to update key "{key}" with value "{value}" in config file!')
        return done

    #########################
    # Internal Comb Methods #
    #########################

    def _write_custom_comb(self, com, rescale_power = True, gen_attempts = 5, **kwargs):
        '''
        Write custom tone power(s), frequency(ies), and/or phase(s) of tones.
        Parameters:
            tone_freqs  (float | list of float | str) : Custom tone frequencies
            tone_powers (float | list of float | str) : Custom tone powers
            tone_phis   (float | list of float | str) : Custom tone phases
        Returns:
            tone_freqs  (list of float) : Comb frequencies used in written comb
            tone_powers (list of float) : Comb tone powers used in written comb
            tone_phis   (list of float) : Comb phases used in written comb
        '''

        import alcove_base

        def _check_comb_type(key, value):
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
            x, _, _ = alcove_base.generateWaveDdr4(freqs, amps, phis)
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
        with rfsoc_io.get_connection(bip, ssh_key, output = self.output) as c:
            # Get current comb and paths to custom comb files
            # -----------------------------------------------
            freq_comb, amp_comb, phi_comb = self._get_curr_comb(com, c = c)

            freq_path = rfsoc_io.get_most_recent_file_board(c, custom_comb_dir, file_identifier = self.io_cfg["board_file_names"]['targ']["comb_freq"], output = self.output, time_past=np.inf)
            amp_path  = rfsoc_io.get_most_recent_file_board(c, custom_comb_dir, file_identifier = self.io_cfg["board_file_names"]['targ']["comb_amp"], output = self.output, time_past=np.inf)
            phi_path  = rfsoc_io.get_most_recent_file_board(c, custom_comb_dir, file_identifier = self.io_cfg["board_file_names"]['targ']["comb_phi"], output = self.output, time_past=np.inf)

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
                if isinstance(value, (str, Path)):
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
                else:
                    try:
                        # Assume an array is passed and check if its non-empty
                        if len(value) == 0:
                            rfsoc_io.send_msg('WARNING', f"'{value}' is an empty array, not writing to {key} custom comb file for drone {com}!", self.output)
                            comb_dict[key]['num_tones'] = len(curr_comb)
                            continue
                    except TypeError:
                        if isinstance(value, (int, float)): # Assume a number is passed and use the same number for all tones
                            comb_dict[key]['num_tones'] = float(value) # Store the number in num_tones as a float for processing later since the number of tones in the comb may be unknown
                            continue
                        else:
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
            tone_freqs = rfsoc_io.get_array_board(c, bip, ssh_key, freq_path, self.tmp_dir, output = self.output)
            tone_freqs_bb = np.array(tone_freqs) - self.drone_cfg[ind]['tones']['NCLO']*1e6

            # Determine if tone powers need to be generated
            # ---------------------------------------------
            tone_powers = np.ones(tone_num)*comb_max/np.sqrt(tone_num) if gen_amps else rfsoc_io.get_array_board(c, bip, ssh_key, amp_path, self.tmp_dir, output = self.output)

            # Generate tone phis and tone powers
            # ----------------------------------
            # Generate phis (as done in primecam_readout tones.py genAmpsAndPhis)
            if gen_phis:
                best_peak = float('inf')  
                best_phis = None
                for _ in range(gen_attempts):
                    tone_phis = np.random.uniform(-np.pi, np.pi, tone_num)
                    comb_max = _comb_peak(tone_freqs_bb, tone_powers, tone_phis)
                    if comb_max < best_peak:
                        best_peak = comb_max
                        best_phis = tone_phis

                # Save phi comb and power comb (if necessary)
                rfsoc_io.save_array_board(c, phi_path, np.array(best_phis).tolist())
            else:
                tone_phis = rfsoc_io.get_array_board(c, bip, ssh_key, phi_path, self.tmp_dir, output = self.output)
                best_peak = _comb_peak(tone_freqs_bb, tone_powers, tone_phis)

            # Check if max comb power is less than max DAC power. Rescale power if specified
            if best_peak > max_power and rescale_power:
                rfsoc_io.send_msg('DEBUG', f"The custom comb has points(s) with output power {comb_max} which exceeds the maximum DAC power of {max_power} when factor is {factor}! Rescaling to lower tone power.", self.output)
                tone_powers = tone_powers*(max_power/best_peak)
            
            if gen_amps or rescale_power: rfsoc_io.save_array_board(c, amp_path, np.array(tone_powers).tolist())
            rfsoc_io.send_msg('INFO', f"Saved custom comb for drone {com}!", self.output)

            # Edit comb saved in drone config files
            # -------------------------------------
            # Frequency comb
            self._edit_config(self.drone_cfg[ind], 'tone_freqs', tone_freqs)

            # Amplitude comb
            self._edit_config(self.drone_cfg[ind], 'tone_powers', tone_powers)

            # Phase comb
            self._edit_config(self.drone_cfg[ind], 'tone_phis', tone_phis)

            # Number of tones
            self._edit_config(self.drone_cfg[ind], 'num_tones', tone_num)
        return tone_freqs, tone_powers, tone_phis

    def _get_curr_comb(self, com, **kwargs):
        '''
        Get the current comb of RFSoC drone

        Parameters:
            com      (str) : Drone bid.drid
            c (Connection) : Fabric connection object to RFSoC board
            load    (bool) : Whether to load the comb files
        Returns:
            comb_freq (list of float) : Current comb frequencies
            comb_amp  (list of float) : Current comb tone powers
            comp_phi  (list of float) : Current comb phases
        '''

        def _get_id(file):
            file = Path(file).stem
            return file.split('_')[-1]

        bid, drid = com.split('.')
        connection = None
        load = True
        for key, value in kwargs.items():
            if key == 'c':
                connection = value
            elif key == 'load':
                load = value

        names = ["comb_freq", "comb_amp", "comb_phi"]
        try:
            drone_id = f"*{bid}_{drid}*"
            # Get most recent comb file paths
            files = [rfsoc_io.get_most_recent_file(self.tmp_data_dir, file_identifier = [self.io_cfg["board_file_names"]['vna'][name] + drone_id, self.io_cfg["board_file_names"]['targ'][name] + drone_id], output=self.output, time_past = np.inf) for name in names]
            
            # Ensure that all comb files exist
            if any(not file.exists() for file in files): raise FileNotFoundError 
            
            # Ensure that all files are from the same comb (all have the id)
            ids = [_get_id(file) for file in files]
            if not len(set(ids)) == 1: raise FileNotFoundError
            if not load: return *tuple(files), False

            # Load most recent comb files
            combs = [rfsoc_io.get_array(file, self.tmp_dir, action='cp', output = self.output, timestamp=True) for file in files]
        except FileNotFoundError:
            # Parse com information
            bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
            ssh_key = self.io_cfg['file_paths']['ssh_key']

            # Create connection to board if Connection object not passed
            if connection is None: connection = rfsoc_io.get_connection(bip, ssh_key, output = self.output)

            # Define directory where sweep combs are saved
            comb_dir = self.drone_dir / f'drone{drid}' / 'comb'

            # Get most recent comb frequency, amplitude, and phase files
            # ----------------------------------------------------------
            with connection as c:
                # Get most recent comb file paths
                files = [rfsoc_io.get_most_recent_file_board(c, comb_dir, file_identifier = self.io_cfg["board_file_names"]['targ'][name], output=self.output, time_past = np.inf) for name in names]

                # ADD LOGIC FOR FILES NOT BEING ON BOARD EITHER
                if not load: return *tuple(files), True

                # Load most recent comb files
                combs = [rfsoc_io.get_array_board(c, bip, ssh_key, file, self.tmp_dir, output = self.output, timestamp=True) for file in files]

        # Return loaded frequency, amplitude, and phase arrays
        return *tuple(combs),

    #@utils.method_timer
    def _save_curr_comb(self, com, name, **kwargs):
        '''
        Save current comb files to data directory and add their file paths to drone config files.

        Parameters:
            com (str)  : Drone to save comb of
            name (str) : File name to use
            c (Connection) : Fabric connection object to RFSoC board
        '''
        bid, drid = com.split('.')
        ind = self.drone_list.index(com)

        # Define comb directory and file names to save to
        # -----------------------------------------------
        freq_name = self.tmp_dir
        amp_name  = self.tmp_dir
        phi_name  = self.tmp_dir

        timestamp = False
        if name is not None:
            timestamp = True

            comb_dir  = self.config_dirs[ind] / 'combs'
            freq_name = comb_dir / f'{name}_freq_comb_{self.timestamp}.npy'
            amp_name  = comb_dir / f'{name}_amp_comb_{self.timestamp}.npy'
            phi_name  = comb_dir / f'{name}_phi_comb_{self.timestamp}.npy'

        freq_file, amp_file, phi_file, from_board = self._get_curr_comb(com, load = False)

        if from_board:
            # Parse com information
            bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
            ssh_key = self.io_cfg['file_paths']['ssh_key']

            # Create connection to board if Connection object not passed
            connection = rfsoc_io.get_connection(bip, ssh_key, output = self.output)

            with connection as c:
                # Copy most recent comb files
                freq_path = rfsoc_io.get_array_board(c, bip, ssh_key, freq_file, freq_name, load = False, output = self.output, timestamp=timestamp)
                amp_path  = rfsoc_io.get_array_board(c, bip, ssh_key,  amp_file,  amp_name, load = False, output = self.output, timestamp=timestamp)
                phi_path  = rfsoc_io.get_array_board(c, bip, ssh_key,  phi_file,  phi_name, load = False, output = self.output, timestamp=timestamp)
        else:
            # Copy most recent comb files
            freq_path = rfsoc_io.get_array(freq_file, freq_name, action='cp', load = False, output = self.output, timestamp=timestamp)
            amp_path  = rfsoc_io.get_array(amp_file,  amp_name, action='cp', load = False, output = self.output, timestamp=timestamp)
            phi_path  = rfsoc_io.get_array(phi_file,  phi_name, action='cp', load = False, output = self.output, timestamp=timestamp)

        self._edit_config(self.drone_cfg[ind], 'tone_freqs', freq_path)
        self._edit_config(self.drone_cfg[ind], 'tone_powers', amp_path)
        self._edit_config(self.drone_cfg[ind], 'tone_phis', phi_path)

        if freq_path is not None: self._edit_config(self.drone_cfg[ind], 'num_tones', len(np.load(freq_path, mmap_mode='r')))

        # Save drone config
        self.drone_cfg[ind] = rfsoc_io.save_config(self.config_dirs[ind] / f"{name}_config_drone_{self.timestamp}.yaml", self.drone_cfg[ind], self.save_cfg)

    ########################
    # Other Helper Methods #
    ########################

    def _run_parallel(self, func, *args, **kwargs):
        '''
        Run subfunction in parallel or in series

        Parameters:
            func (func) : Function to run
        Returns:
            rtn_list (list of Any) : List of returns of func
        '''

        # Parse key word arguments
        # ------------------------
        com_to, boards = self._get_com_to(**kwargs)

        parallel_boards = self.parallel_boards
        parallel_drones = self.parallel_drones
        for key, value in kwargs.items():
            if key == 'parallel_boards':
                parallel_boards = value
            elif key == 'parallel_drones':
                parallel_drones = value

        if parallel_boards == -1: parallel_boards = len(boards)

        # Run func in parallel or series
        # ------------------------------

        # Calculate max number of arrays needed to accommodate drones
        # Need 4 arrays for 1 drone in parallel, 2 arrays for 2 & 3 drones in parallel, and 1 array for 4 drones in parallel
        num_drone_arr = 4 // parallel_drones
        num_drone_arr += 1 if 4 % parallel_drones > 0 else 0

        # Calculate max number of arrays needed to accommodate boards
        # Identical calculation as for drones with 4 replaced with number of boards
        num_board_arr = len(boards) // parallel_boards
        num_board_arr += 1 if len(boards) % parallel_boards > 0 else 0

        # Create array of com_to arrays
        com_arrs = [[] for _ in range(num_drone_arr*num_board_arr)]

        # Create array of sets to track which boards are in each com_to array
        board_arrs = [set() for _ in range(num_drone_arr*num_board_arr)]

        curr_bid = int(boards[0])

        drone_ind = -1 / parallel_drones
        high_ind = 0 # Highest index of com_arrs containing a com_to array with open space

        # Iterate over all drones in drone list
        for com in com_to:
            bid = int(com.split('.')[0])

            if bid > curr_bid:
                curr_bid = bid
                drone_ind = high_ind - 1 / parallel_drones

            drone_ind += 1 / parallel_drones

            board_set = board_arrs[floor(drone_ind)]

            # Check if number of boards exceeds the amount that should be run in parallel
            while len(board_set) >= parallel_boards and not bid in board_set:
                # Move on to next com_to array and check if it has open space
                high_ind += 1
                drone_ind = floor(drone_ind) + 1
                board_set = board_arrs[floor(drone_ind)]

            # Add com to com_to array and add board to set
            ind = floor(drone_ind)
            com_arrs[ind].append(float(com))
            board_arrs[ind].add(bid)

        s = Style()
        name = func.__name__

        rtn_list = []
        for com_to in com_arrs:
            rfsoc_io.send_msg('INFO', f'Running {s.func_name(name)} for drones {com_to}!')
            rtn_list.append(func(str(com_to), *args, **kwargs))

        return rtn_list

    ############
    # Shutdown #
    ############

    def close(self):
        if self.io_cfg['io']['influx_output']: self.rfsoc.updateMeasurement(measurement_name=None)