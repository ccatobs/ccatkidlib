""" Module used for data acquisition with a radio frequency system on a chip (RFSoC)

This module defines the ``R`` Class used for controlling and taking data with a
Xilinx ZCU111 radio frequency system on a chip (RFSoC). The data acquisition
methods are tailored for frequency-division multiplexed readout of microwave
resonators: kinetic inductance detectors (KIDs) in particular.

Authors:
    - Darshan Patel <dp649@cornell.edu>

Example:
    The RFSoC control object is initialized as:
        $ RC = R()

.. _Xilinx ZCU111 RFSoC:
   https://www.amd.com/en/products/adaptive-socs-and-fpgas/evaluation-boards/zcu111.html
"""

# Import Python modules
import sys
import ast
import json
import time
import numpy as np
import multiprocessing as mp
import polars as pl

from math import floor
from pathlib import Path
from invoke import Responder
from collections.abc import Callable, Iterable

# Import local modules
from ccatkidlib.rfsoc.rfsoc_timestream import Streamer
from ccatkidlib.rfsoc_io import header
from ccatkidlib.style import Style
from ccatkidlib.analysis.core.vna import VNA
from ccatkidlib.analysis.core.detector import Detector
from ccatkidlib.analysis.core.target import Target

import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.utils.pair as pair
import ccatkidlib.rfsoc.arg_utils as autils

if mp.get_start_method(allow_none=True) is None: mp.set_start_method('fork')
class R:
    ''' Class for controlling and taking data with a radio frequency system on a chip (RFSoC)

    Attributes:
        curr_date (str): Date of measurement
        timestamp (str): UNIX timestamp corresponding to most recently run DAQ method.
        measurement_name (str): Name of measurement
        measurement_desc (str): Description of measurement
        sess_id (str): Unique ID of measurement

        io_cfg (dict): Loaded IO configuration file
        ext_cfg (dict): Loaded external configuration file
        drone_cfg (list[dict]): List of loaded drone configuration files

        drone_list (list[str]): List of drones used for DAQ
        board_list (list[str]): List of boards used for DAQ
        drone_num (int): Number of drones used for DAQ
        board_num (int): Number of boards used for DAQ
        all_boards (list[str]): List of all RFSoC boards specified in IO configuration file

        parallel_boards (int): Number of boards to run in parallel. Positive integer or -1 to specify all boards
        parallel_drones (int): Number of drones to run in parallel. One of {1, 2, 3, 4}

        save_cfg (bool): Whether to save configuration files
        save_data (bool): Whether to save data files

        data_dir (str): Path to *ccatkidlib* data directory
        drone_dir (str): Path to *primecam_readout* drone directory on RFSoC board
        tmp_data_dir (str): Path to *primecam_readout* data directory: *primecam_readout/src/tmp* as of version ...
        tmp_dir (str): Path to *ccatkidlib* tmp data directory
        g3_dir (str): Path to *rfsoc-streamer* timestream g3 data directory

        config_dirs (list[str]): List of directories where *ccatkidlib* configuration files are saved for each drone
        targ_dirs (list[str]): List of directories where *ccatkidlib* target sweep data files are saved for each drone
        vna_dirs (list[str]): List of directories where *ccatkidlib* VNA sweep data files are saved for each drone
        timestream_dirs (list[str]): List of directories where *ccatkidlib* timestream data files are saved for each drone
        log_dir (str): Directory with *ccatkidlib* master log files

        rfsoc (ocs.OCSClient): OCS client for controlling RFSoC OCS agent
        streamer (ccatkidlib.Streamer): Streamer object used for collecting NumPy timestreams

        NCLOs (dict): Current NCLOs of each drone. Used by rfsoc-controller OCS agent
        drive_attens (dict): Current drive attenuations of each drone. Used by rfsoc-controller OCS agent
        sense_attens (dict): Current sense attenuations of each drone. Used by rfsoc-controller OCS agent
    '''

    @utils.method_timer
    def __init__(self, cfg_path: str = f"{Path(__file__).parent}/system_config.yaml", init_boards: bool | None = None, init_drones: bool | None = None, **kwargs) -> None:
        '''
        Constructor for R. Creates directories for data storage, configures logger, initializes RFSoC boards/drones, and starts RFSoC OCS agent.

        Args:
            cfg_path (str, optional): Path to system configuration file. Defaults to *system_config.yaml* in *ccatkidlib/rfsoc* directory
            init_boards (bool | None, optional): Whether to (re)initialize RFSoC boards. Defaults to None - Pulls from IO configuration file
            init_drones (bool | None, optional): Whether to (re)initialize RFSoC drones. Defaults to None - Pulls from IO configuration file

            kwargs: Key word arguments for passing system state information between control objects. Used by the rfsoc-controller OCS agent. See below:
            sess_id (str): Session ID of measurement
            measurement_name (str): Name of measurement
            measurement_desc (str): Description of measurement
            curr_date (str): Date of measurement
        '''

        # Load config files and setup logging
        # -----------------------------------
        # Current date in yyyy/mm/dd
        self.curr_date = time.strftime('%Y%m%d', time.gmtime())

        # Create a global timestamp used for file naming and pairing
        self.timestamp = str(time.time()).split('.')[0]

        # Load config files
        self._load_system_config(cfg_path)

        # Mainly for use with the rfsoc-controller PCS agent
        # Can use measurement information (sess_id, name, desc) from previous control object
        # ----------------------------------------------------------------------------------
        sess_id = None
        curr_date = None
        self.measurement_name = self.io_cfg['name']
        self.measurement_desc = self.io_cfg['desc']
        for key, value in kwargs.items():
            if key == 'sess_id':
                sess_id = value
            elif key == 'measurement_name':
                self.measurement_name = value
            elif key == 'measurement_desc':
                self.measurement_desc = value
            elif key == 'curr_date':
                curr_date = value

        # Create session id from first ten digits of current time
        new_measurement = sess_id is None or curr_date != self.curr_date or self.measurement_name != self.io_cfg['name'] or self.measurement_desc != self.io_cfg['desc']
        self.sess_id = str(time.time())[:10] if new_measurement else sess_id

        # Save session ID to config files
        # -------------------------------
        rfsoc_io.edit_config(self.ext_cfg, 'sess_id', self.sess_id)
        rfsoc_io.edit_config(self.io_cfg, 'sess_id', self.sess_id)
        for cfg in self.drone_cfg:
            rfsoc_io.edit_config(cfg, 'sess_id', self.sess_id)

        # Create directories
        # ------------------
        new_dir_paths = rfsoc_io.create_tree(self.drone_list, self.curr_date, self.sess_id, self.data_dir) # Create file directory structure for saving data
        self.config_dirs, self.targ_dirs, self.timestream_dirs, self.vna_dirs = new_dir_paths
        self.noise_files = rfsoc_io.create_tmp(self.drone_list, self.tmp_dir) # Create tmp directory and noise tone files

        # Setup logger
        self.log_dir = self.config_dirs[0].parent
        rfsoc_io.setup_logging(self.log_dir / self.io_cfg['io']['logging_fname'], self.io_cfg['io']['file_level'], self.io_cfg['io']['terminal_level'])
        rfsoc_io.send_msg('INFO', f'{Style.INVERT}Date: {self.curr_date}; Session: {self.sess_id}{Style.DEFAULT}')

        # Add paths to primecam_readout modules, PCS clients, and ccatkidlib
        # ---------------------------------------------------------------------
        primecam_readout = Path(self.io_cfg['file_paths']['primecam_readout'])

        sys.path.append(str(primecam_readout))
        sys.path.append(self.io_cfg['file_paths']['pcs_dir'])
        sys.path.append(str(primecam_readout / 'alcove_commands'))

        rfsoc_io.send_msg('DEBUG', f'Finished appending file paths! Paths in sys.path: {sys.path}')

        # Import OCSClient
        from ocs.ocs_client import OCSClient

        # Initialize PCS clients
        # ----------------------
        self.rfsoc = OCSClient(self.io_cfg['pcs_agents']['rfsoc_agent'], args=[])
        rfsoc_io.send_msg('INFO', f'Connected to RFSoC PCS agent!')

        # Setup boards
        # ------------
        init_boards = init_boards if init_boards is not None else self.io_cfg['init']['initialize_boards']
        if init_boards: self.setup_boards()

        # Setup drones
        # ------------
        init_drones = init_drones if init_drones is not None else self.io_cfg['init']['initialize_drones']
        if init_drones: self.setup_drones()
        rfsoc_io.send_msg('INFO', f'Communicating with drones: {self.drone_list}!')

        # Update Measurment Information
        # -----------------------------
        self._update_measurement(name = self.io_cfg['name'], desc = self.io_cfg['desc'])

        # Initialize streamer attribute (As None, since it is only used when g3=False)
        # ----------------------------------------------------------------------------
        self.streamer = None

        # Set NCLO frequency RFSoC drones
        # -------------------------------------------------
        self.NCLOs        = dict(zip(self.drone_list, [None] * self.drone_num))
        self.drive_attens = dict(zip(self.drone_list, [None] * self.drone_num))
        self.sense_attens = dict(zip(self.drone_list, [None] * self.drone_num))
        if init_drones:
            # Set NCLOs from drone config files
            self.set_NCLO(setup=False)

            # Set attenuations from drone config files
            self.set_atten(setup=False)

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
    def _update_measurement(self, name: str | None = None, desc: str = '', influx_output: bool | None = None) -> tuple[str, str] | None:
        '''
        Update measurement name and description

        Note:
            Only used when saving measurement name and description to an InfluxDB database.

        Args:
            name (str | None, optional): New name of measurement. Defaults to None.
            desc (str, optional): New description of measurement. Defaults to ''.
            influx_output (bool | None, optional): Whether to save measurement name and description to InfluxDB database. Defaults to None.

        Returns:
            return (tuple[str, str] | None): New name and description if update successful else None
        '''

        influx_output = influx_output if influx_output is not None else self.io_cfg['io']['influx_output']

        if influx_output:
            sess = self.rfsoc.updateMeasurement(measurement_name = f"{name}, Date: {self.curr_date}, Session ID: {self.sess_id}" if name else None, measurement_desc = desc).session
            success, _ = self._parse_ocs_session(sess)
            return name, desc if success else None
        else:
            return None

    def _load_system_config(self, cfg_path: str) -> None:
        '''
        Load the system (IO & external) and drone configuration files and setup file directory structure for saving data.

        Args:
            cfg_path (str): Path to system configuration file.
        '''

        # Load system configuration file
        # ------------------------------

        cfg = rfsoc_io.load_config(cfg_path)
        try:
            self.ext_cfg, self.io_cfg = cfg # Split config into external and IO config
        except ValueError:
            print("System config must contain two config files (an external config and an IO config)! Please reference the example system config file.")
            sys.exit()

        # Load drone list and drone configs
        # ---------------------------------
        self.drone_list = self.io_cfg['drone_list']
        self.board_list = None

        self.drone_list, bids = autils.get_com_to(self, com_to = self.drone_list, setup = False)
        self.board_list = bids

        self.drone_num = len(self.drone_list)
        self.board_num = len(self.board_list)

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
        self.save_cfg = self.io_cfg['io']['save_config_copy']
        self.save_data = self.io_cfg['io']['save_data']

        # Commonly used file paths
        self.data_dir = Path(self.io_cfg['file_paths']['data_dir'])
        self.drone_dir = Path(self.io_cfg['file_paths']['drone_dir'])
        self.tmp_data_dir = Path(self.io_cfg['file_paths']['primecam_readout']) / 'tmp'
        self.tmp_dir = Path(__file__).parent / '..' / '..' / 'tmp'
        self.g3_dir = Path(self.io_cfg['file_paths']['g3_dir'])

    @header
    @utils.method_timer
    def setup_boards(self, **kwargs) -> None:
        '''
        (Re)initialize RFSoC boards one at time

        Args:
            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Drones are used to determine which boards to initialize. Defaults to drone list in IO configuration file
        '''
        #TODO: Initialize boards in parallel?

        # Get which boards to initialize (only initialize boards with at least on running drone)
        # --------------------------------------------------------------------------------------
        kwargs['setup'] = False
        _, boards = autils.get_com_to(self, **kwargs)

        # Iterate over boards
        # -------------------
        for board in boards:
            # Create Fabric connection to board
            # ---------------------------------
            bip = self.io_cfg['boards'][f'b{board}']['board_ip']
            ssh_key = self.io_cfg['file_paths']['ssh_key']
            with rfsoc_io.get_connection(bip, ssh_key) as c:

                # Restart startup board service on board
                # --------------------------------------
                cmd = 'systemctl restart startup_board.service'
                rfsoc_io.send_msg('INFO', f'Initializing board {board}!')

                sudo_responder = Responder(pattern=r'\[sudo\] password:', response=f'xilinx\n') # Set up responder to run cmd with sudo
                stdout = c.sudo(cmd, hide=True, watchers=[sudo_responder], pty=True)
                rfsoc_io.send_msg('DEBUG', stdout)
                rfsoc_io.send_msg('INFO', f'Finished initializing board {board}!')

                # Restart all drones
                # ------------------
                self.setup_drones(com_to=[f'{board}.1', f'{board}.2', f'{board}.3', f'{board}.4'], restart = True)

    def setup_drones(self, **kwargs) -> None:
        '''
        (Re)initialize drones

        Args:
            kwargs. See below:
            com_to (str | list[str], optional): List of drones to setup. Defaults to drone list in IO configuration file
            restart (bool, optional): Whether to restart already running drones. Defaults to value in IO configuration file
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
                ip, to_run, running = json.loads(rtn.session['data']['data'])
                rfsoc_io.send_msg('PCS', f"{rtn.session}")

                # Start or Restart drones as appropriate
                # --------------------------------------
                if to_run and not running: # Start drones that are not running but should be running
                    rtn = self.rfsoc.action(com_to=com, action='start')
                    rfsoc_io.send_msg('INFO', f"Starting drone {com}...")
                    wait = True
                elif to_run and restart: # Restart drones if restart = True
                    rtn = self.rfsoc.action(com_to=com, action='restart')
                    rfsoc_io.send_msg('INFO', f"Restarting drone {com}...")
                    wait = True
                elif running:
                    running_drones.append(com)

        # Wait if any of the drones were started/restarted and make sure they are running
        # -------------------------------------------------------------------------------
        if wait:
            rfsoc_io.wait(25, desc = f"For drones to start")

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
                    ip, to_run, running = json.loads(rtn.session['data']['data'])
                    rfsoc_io.send_msg('PCS', f"{rtn.session}")

                    if to_run:
                        if running:
                            running_drones.append(com)
                        elif not running:
                            rfsoc_io.send_msg('ERROR', f'Failed to start drone {com}')

        # Log the currently running drones at INFO level if a change was made otherwise log to DEBUG
        rfsoc_io.send_msg('INFO' if wait else 'DEBUG' , f"Drones {sorted(running_drones)} are currently running!")

    @utils.method_timer
    def set_NCLO(self, NCLO: int | list[int] | None = None, **kwargs) -> list[float]:
        '''
        Set the numerically controlled local oscillator (NCLO) frequencies for each drone

        Note:
            If a single NCLO value is specified, it will be used for all drones.

        Args:
            NCLO (int | list[int] | None, optional): NCLO frequencies for each drone in MHz. Defaults to NCLOs in drone configuration files

            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Defaults to drone list in IO configuration file
        Returns:
            return (list[float]): Set NCLOs
        '''

        def _set_NCLO(com: str, *args, **kwargs):
            '''
            Internal function for setting NCLO frequencies in parallel

            Args:
                com (str): List of drones as string (e.g, "['1.2', '2.3', '2.4']")
                *args: NCLO frequency passed as a positional argument

            Returns:
                (ocs.OCSReply): OCS Reply object for RFSoC OCS agent setNCLO command
            '''
            NCLO = args[0]
            rtn = self.rfsoc.setNCLO(com_to = com, f_lo = NCLO, silent=True) # Set NCLO frequency
            rfsoc_io.send_msg('PCS', f'{rtn.session}')
            return rtn

        com_to, _ = autils.get_com_to(self, **kwargs) # Get com_to
        kwargs['setup'] = False

        # Get NCLO from drone config files
        # --------------------------------
        NCLOs = autils.get_drone_args(self, com_to, ['tones', 'NCLO']) # Get NCLOs from drone config files

        # Override config attenuations with those passed as method argument (if any)
        if NCLO is not None:
            LOs = autils.parse_args(self, com_to, NCLO) # Parse NCLOs passed as argument
            if LOs is None: return None # Return None if invalid NCLOs were passed

            NCLOs = list(map(int, LOs)) # Parse NCLOs passed as argument and cast as int

            autils.set_drone_args(self, com_to, "NCLO", NCLOs) # Update NCLOs in drone config files

        # Only set NCLOs that are different from the current ones
        # -------------------------------------------------------
        new_NCLOs = []
        new_com_to = []
        for com, NCLO in zip(com_to, NCLOs):
            # Check if NCLO to set matches current NCLO
            if not self.NCLOs.get(com, None) == NCLO:
                new_NCLOs.append(NCLO)
                new_com_to.append(com)
                self.NCLOs[com] = NCLO

        # Group drones with the same NCLO
        # --------------------------------
        NCLO_dict = autils.group_args(new_com_to, new_NCLOs)
        for NCLO, com in NCLO_dict.items():
            args = [int(NCLO)]
            kwargs['com_to'] = com
            self._run_parallel(_set_NCLO, *args, **kwargs)

        # Reset com_to and parallel_drones to what they were before
        kwargs['com_to'] = com_to
        rfsoc_io.send_msg('INFO', f'Successfully set NCLO to {list(NCLOs)} for drones {list(com_to)}!')

        return NCLOs

    @header
    @utils.method_timer
    def set_atten(self, drive: float | list[float] | None = None, sense: float | list[float] | None = None, **kwargs) -> tuple[list[float], list[float]]:
        '''
        Set drive/sense attenuations of frontend attenuators connected to RFSoC board

        Note:
            - Drive attenuations refer to attenuators on the digital-to-analog (DAC) side and sense attenuations refer to attenuators on the analog-to-digital (ADC) side
            - If a single drive/sense attenuation value is specified, it will be used for all drones.

        Args:
            drive (float | list[float] | None, optional): Drive attenuations for each drone in dB. Must be between 0 and 31.75. Defaults to drive attenuations in drone configuration files
            sense (float | list[float] | None, optional): Sense attenuations for each drone in dB. Must be between 0 and 31.75. Defaults to sense attenuations in drone configuration files

            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Defaults to drone list in IO configuration file
        Returns:
            return (tuple[list[float], list[float]]): Current drive and sense attenuations
        '''

        def _set_direction(com_to: list[str], direction: str, atten: list[float] = None) -> list[float] | None:
            '''
            Internal function for setting drone attenuations of a specific direction ('drive' or 'sense')

            Parameters:
                com_to (str | list[str]) : List of drones
                direction (str): Which attenuators to set. Must be one of {'drive', 'sense'}.
                atten (float | list[float] | None) : List of attenuation values. Attenuation must be between 0 and 31.75. Defaults to attenuation in drone configuration files
            Returns:
                return (list[float] | None) : The set attenuations. None if invalid attenuations were used
            '''

            def _set_atten(com: str, *args, **kwargs):
                '''
                Internal function for setting drone attenuations in parallel

                Args:
                    com (str): List of drones as string (e.g, "['1.2', '2.3', '2.4']")
                    *args: Attenuator direction and attenuation value passed as positional arguments

                Returns:
                    return (ocs.OCSReply): OCS Reply object for RFSoC OCS agent setAtten2025 command
                '''
                direction, att = args[0], args[1]
                rtn = self.rfsoc.setAtten2025(com_to = com, direction = direction, atten = att, silent=True)
                rfsoc_io.send_msg('PCS', f'{rtn.session}')
                return rtn

            # -----------------------
            # Get attenuations to set
            # -----------------------
            attens = autils.get_drone_args(self, com_to, ['atten', f'{direction}']) # Get attenuations from drone config files

            # Override config attenuations with those passed as method argument (if any)
            if atten is not None:
                attens = list(map(float, autils.parse_args(self, com_to, atten))) # Parse attenuations passed as argument and cast to int

                # Return None if invalid attenuations were passed
                if attens is None: return None

                autils.set_drone_args(self, com_to, direction, attens) # Update attenuations in drone config files

            # ----------------
            # Set attenuations
            # ----------------
            # Set attenuation of drones (can do max of one drone per board at a time since attenuations are set through serial communication)

            # Only set attenuations that are different from the current ones
            # --------------------------------------------------------------
            new_attens = []
            new_com_to = []
            new_dict = getattr(self, f'{direction}_attens') # Get current attenuations for the specified direction
            for com, atten in zip(com_to, attens):
                if not new_dict.get(com, None) == atten:
                    new_attens.append(atten)
                    new_com_to.append(com)
                    new_dict[com] = atten
            setattr(self, f'{direction}_attens', new_dict) # Update current attenuations for the specified direction

            # Change number of drones in parallel to one
            parallel_drones = self.parallel_drones
            self.parallel_drones = 1

            # Group drones with the same attenuations
            atten_dict = autils.group_args(new_com_to, new_attens)
            for att, com in atten_dict.items():
                args = [direction, float(att)]
                kwargs['com_to'] = com
                self._run_parallel(_set_atten, *args, **kwargs)

            # Reset com_to and parallel_drones to what they were before
            kwargs['com_to'] = com_to
            self.parallel_drones = parallel_drones
            #rfsoc_io.send_msg('INFO', f'Successfully set {direction} attenuation to {list(attens)} for drones {list(com_to)}!')

            return attens

        def _check_direction(com_to: list[str], direction: str, set_attens: list[float], new_attens: list[float]) -> None:
            '''
            Check if drive/sense attenuations were set successfully by comparing desired attenuations to current attenuations

            Args:
                com_to (list[str]): List of drones
                direction (str): Which attenuators to check. Must be one of {'drive', 'sense'}.
                set_attens (list[float]): Attenuation values attenuators should be set to
                new_attens (list[float]): Attenuation values attenuators are currently set to
            '''

            new_dict = getattr(self, f'{direction}_attens')
            for com, set_atten, new_atten in zip(com_to, set_attens, new_attens):
                if new_atten != set_atten:
                    rfsoc_io.send_msg('ERROR', f"Failed to set {direction} attenuation for drone {com} to {set_atten} dB. \
                                      Current attenuation is {new_atten if new_atten is not None else 'unknown'} dB.")
                    new_dict[com] = new_atten
            setattr(self, f'{direction}_attens', new_dict)

        # Specify which attenuators to change
        com_to, _ = autils.get_com_to(self, **kwargs)
        kwargs['setup'] = False

        # Set drive and sense attenuations
        set_drives, set_senses = _set_direction(com_to = com_to, direction = 'drive', atten = drive), \
                                 _set_direction(com_to = com_to, direction = 'sense', atten = sense)

        # Check that attenuations were set correctly
        new_drives, new_senses = self.get_atten(**kwargs)

        _check_direction(com_to, direction = 'drive', set_attens = set_drives, new_attens = new_drives)
        rfsoc_io.send_msg('INFO', f'Successfully set drive attenuations to {new_drives} for drones {com_to}.')

        _check_direction(com_to, direction = 'sense', set_attens = set_senses, new_attens = new_senses)
        rfsoc_io.send_msg('INFO', f'Successfully set sense attenuations to {new_senses} for drones {com_to}.')

        return new_drives, new_senses

    #===========================#
    # Monitoring Getter Methods #
    #===========================#

    @header
    @utils.method_timer
    def get_stats(self, space: bool = True, temps: bool = True, ADC_rms: bool = True, **kwargs) -> tuple[list[float], tuple[list[float], list[float]], list[float]]:
        '''
        Get RFSoC storage space, temperatures, and RMS power at drone analog-to-digital converters (ADCs).

        Args:
            space (bool, optional): Whether to get RFSoC storage space. Defaults to True
            temps (bool, optional): Whether to get RFSoC fabric and processor temperatures. Defaults to True
            ADC_rms (bool, optional): Whether to get root mean squared (RMS) power at ADCs. Defaults to True

            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Defaults to drone list in IO configuration file
        Returns:
            return tuple[list[float], tuple[list[float], list[float]], list[float]]: Stoarge space, temperatures, and RMS power at ADCs

        '''

        if kwargs['setup']: self.setup_drones(**kwargs) # Setup drones

        # Start RFSoC temperature and storage space feeds if not running
        # --------------------------------------------------------------
        if not self.rfsoc.feedMonitor.status().session['op_code'] == 2:
            self.rfsoc.feedMonitor.start()
            rfsoc_io.wait(60, desc="HK Feed Monitor Starting")

        kwargs['setup'] = False # Do not set up drones again
        if space: avail_spaces = self.get_avail_space(**kwargs) # Get storage space
        if temps: temp_lists = self.get_temps(**kwargs) # Get temperatures
        if ADC_rms: rms_list = self.get_ADC_rms(**kwargs) # Get RMS power at ADCs

        return avail_spaces, temp_lists, rms_list

    def get_ADC_rms(self, **kwargs) -> list[float]:
        '''
        Get root mean squared (RMS) power at the analog-to-digital converters (ADCs)

        Args:
            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Defaults to drone list in IO configuration file
        Returns:
            return (list[float]) : ADC RMS values (sorted by drone)
        '''

        def _getSnapData(com: str, *args, **kwargs) -> dict:
            '''
            Internal function for getting power at ADC in parallel

            Args:
                com (str): List of drones as string (e.g, "['1.2', '2.3', '2.4']")

            Returns:
                return (dict): Data dictionary returned by RFSoC OCS Agent ``getSnapData`` method
            '''
            rtn = self.rfsoc.getSnapData(com_to = com, mux_sel = 0, silent = False).session
            rfsoc_io.send_msg('PCS', f'{rtn}')
            return json.loads(rtn['data']['data'])[1:][0]

        com_to, _ = autils.get_com_to(self, **kwargs)
        kwargs['setup'] = False

        # Take data one board at a time (all drones transfers too much data through OCS)
        kwargs['parallel_boards'] = 1
        kwargs['parallel_drones'] = 2

        # Get snap data of drones
        rtns = self._run_parallel(_getSnapData, **kwargs)
        rtns = [drone for drones in rtns for drone in drones] # Flatten list of returns

        # Convert I, Q Snap data into ADC rms
        # -----------------------------------
        rms_list, inds = [], []
        for rtn in rtns:
            data_dic = rtn['data'] # Load pickled data dictionary
            ind = self.drone_list.index(f"{data_dic['bid']}.{data_dic['drid']}") # Determine which drone the Snap data corresponds to
            inds.append(ind)

            # From primecam_readout alcove_base getADCrms function
            # ----------------------------------------------------
            I, Q = data_dic['data'] # Get I, Q Snap data
            z = np.array(I) + 1j * np.array(Q) # Convert to complex number
            rms = float(np.real(np.sqrt(np.mean(z * np.conj(z))))) # Calculate RMS value at the ADC
            rms_list.append(rms)

            rfsoc_io.edit_config(self.drone_cfg[ind], 'ADC_RMS', rms)  # Add ADC rms to drone config

        # Create list of ADC rms values for all drones, sort by drone bid.drid
        rms_list = [rms for _, rms in sorted(zip(inds, rms_list))]
        rfsoc_io.send_msg('INFO', f'RMS power at the ADC is {rms_list} for drones {com_to}!')
        return rms_list

    def get_avail_space(self, **kwargs) -> list[float]:
        '''
        Check available storage space and clean drone directories on RFSoCs

        Args:
            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Defaults to drone list in IO configuration file
            threshold (float): Storage space threshold after which to automatically clean directories on RFSoCs
            olderThanDaysAgo (int | str): Number of days old a file must be in order to be deleted by automatic cleaning
            ftype (str): Type of file to be deleted by automatic cleaning
        Returns:
            return (list[float]): Available space left on each board (sorted by board)
        '''
        # Import clean_io from primecam_readout
        import clean_io

        def _get_space(bids) -> list[float]:
            '''
            Internal function for getting available storage space left on RFSoC boards

            Args:
                bids (list[str]): List of boards

            Returns:
                return (list[float]): Available storage space left on boards
            '''
            avail_spaces = []
            session_data = self.rfsoc.feedMonitor.status().session['data']
            for bid in bids:
                space_dict = session_data[f'drone_free_spaces_GB_board_{bid}']['data']
                avail_space = space_dict[f'spc_{bid}_1']
                avail_spaces.append(avail_space)
                rfsoc_io.edit_config(self.ext_cfg, ['boards', f'b{bid}', 'avail_space'], avail_space, append = True)
            rfsoc_io.send_msg('INFO', f'Storage space is {avail_spaces} GB for boards {bids}!')
            return avail_spaces

        com_to, bids = autils.get_com_to(self, **kwargs)
        kwargs['setup'] = False

        threshold = self.io_cfg['clean']['threshold']  # [GB] Threshold after which clean board drone directories
        olderThanDaysAgo = self.io_cfg['clean']['olderThanDaysAgo'] # [Days]
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
                rfsoc_io.send_msg('INFO', f'Cleaning drones {drones_to_clean}...')
                for com in drones_to_clean: self.rfsoc.cleanBoardDroneDirs(com_to=com, testing=False, leave_latest=True, olderThanDaysAgo=str(olderThanDaysAgo))
                rfsoc_io.send_msg('INFO', f'Finished cleaning drones {drones_to_clean}')
                avail_spaces = _get_space(bids)

                if self.io_cfg['clean']['tmp']:
                    # Clean primecam_readout tmp directory
                    clean_io.cleanQueenTmpDir(testing=False, ftype='', olderThanDaysAgo=str(olderThanDaysAgo))

                    # Clean ccatkidlib tmp directory
                    ignore_list = [str(Path(self.tmp_dir).resolve() / 'noise_tones.npy')]
                    clean_io._cleanDir(str(Path(self.tmp_dir).resolve()), testing=False, ignore_list=ignore_list, ftype=str(ftype), olderThanDaysAgo=str(olderThanDaysAgo))
        return avail_spaces

    def get_temps(self, **kwargs) -> tuple[list[float], list[float]]:
        '''
        Get fabric and processor temperatures of RFSoCs

        Args:
            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Defaults to drone list in IO configuration file
        Returns:
            return (tuple[list[float], list[float]]): Processor and fabric temperatures of RFSoCs
        '''
        def _get_temps(bids: list[str], loc: str) -> list[float]:
            '''
            Internal function for getting RFSoC temperatures using specified thermometer

            Args:
                bids (list[str]): List of boards
                loc (str): Location of thermometer. Must be one of {'ps', 'pl'}
            Returns:
                return (list[float]): List of temperatures
            '''
            temps = []
            for bid in bids:
                temp_dict = session_data[f'drone_temperatures_C_board_{bid}']['data']
                temp = float(np.mean([temp_dict[f'temp_{bid}_{drid + 1}_{loc}'] for drid in range(4)]))
                temps.append(temp)
                rfsoc_io.edit_config(self.ext_cfg, ['boards', f'b{bid}', f'{loc}_temp'], temp, append = True)
            rfsoc_io.send_msg('INFO', f'{loc} temperatures are {temps} \xb0C for boards {bids}!')
            return temps

        com_to, bids = autils.get_com_to(self, **kwargs)
        kwargs['setup'] = False
        session_data = self.rfsoc.feedMonitor.status().session['data']
        ps_temps = _get_temps(bids, 'ps')
        pl_temps = _get_temps(bids, 'pl')

        return ps_temps, pl_temps

    def get_atten(self, **kwargs) -> tuple[list[float], list[float]]:
        '''
        Get drive/sense attenuations of frontend attenuators connected to RFSoC board

        Note:
            - Drive attenuations refer to attenuators on the digital-to-analog (DAC) side and sense attenuations refer to attenuators on the analog-to-digital (ADC) side

        Args:
            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Defaults to drone list in IO configuration file
        Returns:
            return (tuple[list[float], list[float]]): Current drive and sense attenuations
        '''

        def _get_direction(com_to: str | list[str], direction: str) -> list[float]:
            '''
            Internal function for getting drone attenuations of a specific direction ('drive' or 'sense')

            Parameters:
                com_to (str | list[str]) : List of drones
                direction (str): Which attenuator values to get. Must be one of {'drive', 'sense'}.
            Returns:
                return (list[float] | None) : The current attenuations
            '''
            def _get_atten(com: str, *args, **kwargs):
                '''
                Internal function for getting drone attenuations in parallel

                Args:
                    com (str): List of drones as string (e.g, "['1.2', '2.3', '2.4']")
                    *args: Attenuator direction passed as a positional argument

                Returns:
                    return (ocs.OCSReply): OCS Reply object for RFSoC OCS agent getAtten command
                '''

                direction = args[0]
                rtn = self.rfsoc.getAtten(com_to = com, direction = direction, silent=False).session
                rfsoc_io.send_msg('PCS', f'{rtn}')
                return json.loads(rtn['data']['data'])[1:][0]

            # Get attenuations
            # ----------------
            # Get attenuation of drones (can do max of one drone per board at a time since attenuations are set through serial communication)

            # Change number of drones in parallel to one
            parallel_drones = self.parallel_drones
            self.parallel_drones = 1

            args = [direction]
            rtns = self._run_parallel(_get_atten, *args, **kwargs)
            rtns = [drone for drones in rtns for drone in drones]

            attens, inds = [], []
            for rtn in rtns:
                atten = rtn['data']
                com = rtn['channel'].split('_')[-2]
                ind = self.drone_list.index(com)
                if not isinstance(atten, float):
                    atten = None
                    rfsoc_io.send_msg('ERROR', f'Failed to get {direction} attenuation for drone {com}.')
                inds.append(ind)
                attens.append(atten)


            # Create list of attenuations for all drones, sort by drone bid.drid
            attens = [atten for _, atten in sorted(zip(inds, attens))]

            # Reset parallel_drones to what it was before
            self.parallel_drones = parallel_drones
            rfsoc_io.send_msg('INFO', f'Successfully got {direction} attenuations {attens} for drones {com_to}!')

            return attens

        # Specify which attenuators to change
        com_to, _ = autils.get_com_to(self, **kwargs)
        kwargs['setup'] = False

        # Set drive attenuation
        drive_attens = _get_direction(com_to = com_to, direction = 'drive')

        # Set sense attenuation
        sense_attens = _get_direction(com_to = com_to, direction = 'sense')

        return drive_attens, sense_attens

    #==========================#
    # Data Acquisition Methods #
    #==========================#
    @header
    @utils.method_timer
    def take_vna_sweep(self, **kwargs) -> list[str]:
        '''
        Take a vector network analyzer (VNA) style sweep covering the full 512 MHz available bandwidth

        Note:
            - The **ccatkidlib** data file paths are returned if save_data is True otherwise the **primecam_readout** data file paths are returned
            - If a single sweep_steps value is specified, it will be used for all drones.

        Args:
            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Defaults to drone list in IO configuration file
            write_comb (bool, optional): Whether to write a new VNA comb. Defaults to True
            sweep_steps (int | list[int], optional): Number of steps each tone should take during sweep. Defaults to value in drone configuration files

        Returns:
            return (list[str]): List of VNA sweep file paths
        '''
        @utils.function_timer
        def _write_vna_comb(com: str, *args, **kwargs):
            '''
            Internal function for writing new VNA combs in parallel

            Args:
                com (str): List of drones as string (e.g, "['1.2', '2.3', '2.4']")

            Returns:
                return (ocs.OCSReply): OCS Reply object for RFSoC OCS agent writeNewVnaComb command
            '''
            rtn = self.rfsoc.writeNewVnaComb(com_to = com, silent = False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}')
            return rtn

        @utils.function_timer
        def _take_vna_sweep(com: str, *args, **kwargs):
            '''
            Internal function for taking VNA sweeps in parallel

            Args:
                com (str): List of drones as string (e.g, "['1.2', '2.3', '2.4']")
                args: sweep_steps passed as a positional argument

            Returns:
                return (ocs.OCSReply): OCS Reply object for RFSoC OCS agent vnaSweep command
            '''
            sweep_steps = args[0]
            rtn = self.rfsoc.vnaSweep(com_to = com, sweep_steps = sweep_steps, silent = False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}')
            return rtn

        # Get com_to
        com_to, boards = autils.get_com_to(self, **kwargs)
        kwargs['setup'] = False # Do not set up drones again

        # TODO: Seperate sweep steps for VNA and target sweeps?
        # Parse key word arguments
        write_comb = True
        sweep_steps = autils.get_drone_args(self, com_to, ['tones', 'sweep_steps'])
        for key, value in kwargs.items():
            if key == 'write_comb':
                write_comb = value
            elif key == 'sweep_steps':
                sweep_steps = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "sweep_steps", sweep_steps)

        # Write VNA comb
        # --------------
        if write_comb:
            rfsoc_io.send_msg('INFO', 'Writing new VNA combs...')
            self._run_parallel(_write_vna_comb, **kwargs)
            rfsoc_io.send_msg('INFO', f'Successfully wrote new VNA combs for drones {com_to}!')
            time.sleep(1) # Wait before taking sweep (not waiting can affect sweep quality, unsure if this is still the case: need to test)

        # Take VNA sweep
        # --------------
        self.timestamp = str(time.time()).split('.')[0]
        rfsoc_io.send_msg('INFO', 'Taking VNA sweeps...')

        sweep_dict = autils.group_args(com_to, sweep_steps)
        for sweep_steps, com in sweep_dict.items():
            args = [int(sweep_steps)]
            kwargs['com_to'] = com
            self._run_parallel(_take_vna_sweep, *args, **kwargs)
        kwargs['com_to'] = com_to

        rfsoc_io.send_msg('INFO', f'Finished taking VNA sweeps for drones {com_to} with timestamp {self.timestamp}!')
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
    def take_target_sweep(self, **kwargs) -> list[str]:
        '''
        Take a vector network analyzer (VNA) style sweep with a specified set of tones.

        Note:
            - The **ccatkidlib** data file paths are returned if save_data is True otherwise the **primecam_readout** data file paths are returned
            - If a single sweep_steps or chan_bw value is specified, it will be used for all drones.

        Args:
            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Defaults to drone list in IO configuration file
            chan_bw (float | list[float], optional): Bandwidth of sweep around each tone in MHz. Defaults to value in drone configuration files
            sweep_steps (int | list[int], optional): Number of steps each tone should take during sweep. Defaults to value in drone configuration files

            tone_freqs (list[list[float]]: optional): Frequencies at which to place tones in Hz
            tone_powers (str | list[str] | float | list[float] | list[list[float]], optional): Tone powers in digital-to-analog (DAC) units (proportional to voltage). Use 'gen' to generate powers
            tone_phis (str | list[str] | float | list[float] | list[list[float]], optional): Tone phases in radians. Use 'gen' to generate phases
            write_comb (bool, optional): Whether to write a new target sweep comb. Defaults to True if any of ``tone_freqs``, ``tone_powers``, or ``tone_phis`` is specified

        Returns:
            return (list[str]): List of target sweep file paths
        '''

        def _take_targ_sweep(com: str, *args, **kwargs):
            '''
            Internal function for taking target sweeps in parallel

            Args:
                com (str): List of drones as string (e.g, "['1.2', '2.3', '2.4']")
                args: sweep_steps and chan_bw passed as positional arguments

            Returns:
                return (ocs.OCSReply): OCS Reply object for RFSoC OCS agent targetSweep command
            '''
            sweep_steps, chan_bw = args
            rtn = self.rfsoc.targetSweep(com_to = com, sweep_steps = sweep_steps, chan_bw = chan_bw, silent = False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}')
            return rtn

        com_to, boards = autils.get_com_to(self, **kwargs)
        kwargs['setup'] = False # Do not set up drones again

        # Evaluate kwargs
        write_comb = any(key in kwargs for key in ['tone_powers', 'tone_freqs', 'tone_phis'])
        sweep_steps = autils.get_drone_args(self, com_to, ['tones', 'sweep_steps'])
        chan_bw = autils.get_drone_args(self, com_to, ['tones', 'chan_bw'])
        for key, value in kwargs.items():
            if key == 'write_comb':
                write_comb = value
            elif key == 'sweep_steps':
                sweep_steps = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "sweep_steps", sweep_steps)
            elif key == 'chan_bw':
                chan_bw = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "chan_bw", chan_bw)

        # Write new target sweep comb
        if write_comb:
            self.write_config_comb(**kwargs)
            time.sleep(1) # Wait before taking sweep (not waiting can affect sweep quality)

        # Get timestamp
        self.timestamp = str(time.time()).split('.')[0]
        rfsoc_io.send_msg('INFO', 'Taking target sweeps!')

        sweep_dict = autils.group_args(com_to, sweep_steps, chan_bw)

        for arg, com in sweep_dict.items():
            sweep_steps, chan_bw = arg.split(', ')
            args = [int(sweep_steps), float(chan_bw)]
            kwargs['com_to'] = com
            self._run_parallel(_take_targ_sweep, *args, **kwargs)
        kwargs['com_to'] = com_to

        rfsoc_io.send_msg('INFO', f'Finished taking target sweeps for drones {com_to} with timestamp {self.timestamp}!')
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
    def take_timestream(self, t_sec: float, **kwargs) -> list[str]:
        '''
        Take timestream data with a specified set of tones

        Note:
            - Saving timestream data as g3 files requires integration with the *rfsoc-streamer* software
            - Use of the *rfsoc-streamer* software is recommended when streaming with large numbers of drones

        Args:
            t_sec (float): Length of timestream in seconds

            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Defaults to drone list in IO configuration file
            g3 (bool, optional): Whether to save timestream data as g3 files. Defaults to value in IO configuration file
            save_data (bool, optional): Whether to save data to ``ccatkidlib`` data directory. Defaults to value in IO configuration file

            tone_freqs (list[list[float]]: optional): Frequencies at which to place tones in Hz
            tone_powers (str | list[str] | float | list[float] | list[list[float]], optional): Tone powers in digital-to-analog (DAC) units (proportional to voltage). Use 'gen' to generate powers
            tone_phis (str | list[str] | float | list[float] | list[list[float]], optional): Tone phases in radians. Use 'gen' to generate phases
            write_comb (bool, optional): Whether to write a new target sweep comb. Defaults to True if any of ``tone_freqs``, ``tone_powers``, or ``tone_phis`` is specified
        Returns:
            return (list[str] | None): List of timestream file paths or None if ``save_data`` is False
        '''

        def _take_timestream_numpy(com: list[str], *args, **kwargs) -> list[str]:
            '''
            Internal function for capturing timestream data and saving as NumPy files

            Args:
                com (list[str]): List of drones

            Returns:
                list[str]: List of timestream NumPy file paths
            '''
            import gc
            from multiprocessing import Process, Queue, Manager, Value


            def _save_timestream_numpy(com_to, start_time, time_diff, packet_data, tstream_seg, tstream_outs):
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


                drone_ips = autils.get_drone_args(self, com_to, 'udp_source_ip')
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
            timer = Process(target=rfsoc_io.wait, args=(t_sec,), kwargs={'desc': f'Taking {t_sec} second timestream for {com}'})

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
                        saves.append(Process(target=_save_timestream_numpy, args=(com_arr, start_time, time_diff, packet_data, tstream_seg, tstream_outs,)))
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

        def _take_timestream_g3(com: list[str], *args, **kwargs):
            '''
            Internal function for taking timestreams when using the *rfsoc-streamer* to capture and save data

            Args:
                com (list[str]): List of drones
                *args: Length of timestream in seconds passed as a positional argument

            Returns:
                return (ocs.OCSReply): OCS Reply object for RFSoC OCS agent getAtten command
            '''
            t_sec = args[0]
            rtn = self.rfsoc.timestreamOn(com_to = com, on = True, silent=True)
            rfsoc_io.wait(t_sec, desc = f'Taking {t_sec} second timestream for {com}')
            rtn = self.rfsoc.timestreamOn(com_to = com, on = False, silent=True)
            rfsoc_io.send_msg('INFO', f"Finished taking {t_sec} seconds of timestream data with timestamp {self.timestamp}!")
            return rtn

        def _save_timestream_g3(com_to: list[str]) -> list[str]:
            '''
            Internal function for saving txt files with file paths of g3 timestream data

            Args:
                com_to (list[str]): List of drones

            Returns:
                return (list[str]): List of file paths of timestream txt files
            '''
            date = self.timestamp[0:5]

            fname = self.io_cfg["save_file_names"]["timestream"]
            fname = f'{fname}_{self.timestamp}'

            stream_paths = []
            for com in com_to:
                bid, drid = com.split('.')
                ind = self.drone_list.index(com)
                timestream_dir = self.timestream_dirs[ind]

                g3_tstream_dir = self.g3_dir / date / f'rfsoc{int(bid):02d}_drone{drid}'
                g3_file = rfsoc_io.get_most_recent_file(g3_tstream_dir, f'r{int(bid):02d}d{drid}*', time_past = np.inf)

                tstream_parts = g3_file.stem.split('_')
                tstream_name = '_'.join(tstream_parts[0:-1])
                try:
                    tstream_num = int(tstream_parts[-1])
                except ValueError:
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

        com_to, _ = autils.get_com_to(self, **kwargs)
        kwargs['setup'] = False # Do not set up drones again

        # Parse key word arguments
        g3 = self.io_cfg['timestream']['g3'] # Whether to take g3 timestream
        write_comb = any(key in kwargs for key in ['tone_powers', 'tone_freqs', 'tone_phis'])
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
        rfsoc_io.send_msg('INFO', f'Taking {t_sec} seconds of timestream data!')
        stream_out = None
        self.timestamp = str(time.time()).split('.')[0]
        args=[t_sec]
        if g3:
            self._run_parallel(_take_timestream_g3, *args, **kwargs)
            if save_data:
                time.sleep(1)
                stream_out = _save_timestream_g3(com_to)
        else:
            if self.streamer is None:
                self.streamer = Streamer(self.io_cfg['timestream']['udp_ip'], self.io_cfg['timestream']['udp_port'])
                rfsoc_io.send_msg('INFO', f"Successfully initialized timestream object using address {self.io_cfg['timestream']['udp_ip']} and port {self.io_cfg['timestream']['udp_port']}!")
            stream_out = self._run_parallel(_take_timestream_numpy, *args, **kwargs)
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

    #================#
    # Tuning Methods #
    #================#

    @header
    @utils.method_timer
    def write_config_comb(self, **kwargs):
        '''
        Write a custom set of radio frequency tones (comb)

        Args:
            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Defaults to drone list in IO configuration file

            tone_freqs (list[list[float]]: optional): Frequencies at which to place tones in Hz
            tone_powers (str | list[str] | float | list[float] | list[list[float]], optional): Tone powers in digital-to-analog (DAC) units (proportional to voltage). Use 'gen' to generate powers
            tone_phis (str | list[str] | float | list[float] | list[list[float]], optional): Tone phases in radians. Use 'gen' to generate phases

            rescale_power (bool | list[bool]): Whether to rescale tone powers if DAC maximum power is exceeded. Defaults to value in drone configuration files
            gen_attempts (int | list[int]): Number of attempts to randomize phases so that the resulting comb does not exceed DAC maximum power. Defaults to value in drone configuration files
        Returns:
            return (list[ocs.OCSReply]): List of OCSReply objects for RFSoC OCS agent writeTargCombFromCustomList command
        '''

        def _write_targ_comb(com, *args, **kwargs):
            '''
            Internal function for writing target comb using custom list in parallel

            Args:
                com (str): List of drones as string (e.g, "['1.2', '2.3', '2.4']")
            Returns:
                return (ocs.OCSReply): OCSReply object of RFSoC OCS agent for writeTargCombFromCustomList command
            '''
            rtn = self.rfsoc.writeTargCombFromCustomList(com_to = com, silent=False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}')
            return rtn

        # Parse key word arguments
        # ------------------------
        com_to, _ = autils.get_com_to(self, **kwargs)
        kwargs['setup'] = False # Do not set up drones again

        tone_freqs    = autils.get_drone_args(self, com_to, ['tones', 'custom', 'tone_freqs'])
        tone_powers   = autils.get_drone_args(self, com_to, ['tones', 'custom', 'tone_powers'])
        tone_phis     = autils.get_drone_args(self, com_to, ['tones', 'custom', 'tone_phis'])
        rescale_power = autils.get_drone_args(self, com_to, ['tones', 'custom', 'generation', 'rescale_power'])
        gen_attempts  = autils.get_drone_args(self, com_to, ['tones', 'custom', 'generation', 'gen_attempts'])

        # Evaluate kwargs
        for key, value in kwargs.items():
            if key == 'tone_freqs':
                tone_freqs = autils.parse_args(self, com_to, value)
            elif key == 'tone_powers':
                tone_powers = autils.parse_args(self, com_to, value)
            elif key == 'tone_phis':
                tone_phis = autils.parse_args(self, com_to, value)
            elif key == 'rescale_power':
                rescale_power = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, ['tones', 'custom', 'generation', 'rescale_power'], rescale_power)
            elif key == 'gen_attempts':
                gen_attempts = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, ['tones', 'custom', 'generation', 'gen_attempts'], gen_attempts)

        # Write custom comb for each drone
        # --------------------------------
        for com, rescale, attempts, freq, power, phi in zip(com_to, rescale_power, gen_attempts, tone_freqs, tone_powers, tone_phis):
            self._write_custom_comb(com, rescale_power = rescale, gen_attempts = attempts, tone_freqs = freq, tone_powers = power, tone_phis = phi)

        # Write sweep comb based on custom parameters
        # -------------------------------------------
        rtn = self._run_parallel(_write_targ_comb, **kwargs)
        rfsoc_io.send_msg('INFO', f'Sucessfully wrote custom comb for drones {com_to}!')
        return rtn

    @header
    @utils.method_timer
    def find_detectors(self, new_sweep: bool = True, **kwargs):
        '''
        Find detectors using a VNA sweep

        Note:
            - The larger of peak_prom_db or peak_prom_std is used if both are specified

        Args:
            new_sweep (bool, optional): Whether to take a new VNA sweep. Defaults to True

            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Defaults to drone list in IO configuration file
            sweep_steps (int | list[int], optional): Number of steps each tone should take during sweep. Defaults to value in drone configuration files
            write_targ_comb (bool, optional): Whether to write target comb using found detector frequencies. Defaults to True
            phase_filter (bool | list[bool], optional): Whether to perform phase filtering on found detectors. Defaults to value in drone configuration files
            filter_wn (int | list[int], optional): Number of points around tone to use for phase filtering. Defaults to value in drone configuration files

            peak_prom_std (float | list[float], optional): Peak height from surroundings, in noise std multiples. Defaults to value in drone configuration files
            peak_prom_db (float | list[float], optional): Peak height from surroundings, in dB. Defaults to value in drone configuration files
            peak_dis (int | list[int], optional): Min distance between peaks. Defaults to value in drone configuration files
            width_min (int | list[int], optional): Min peak width. Defaults to value in drone configuration files
            width_max (int | list[int], optional): Max peak width. Defaults to value in drone configuration files
            stitch (bool | list[bool], optional): Whether to stitch comb discontinuities. Defaults to value in drone configuration files
            stitch_sw (int | list[int], optional): Amount of points on each end to use for stitching. Defaults to value in drone configuration files
            remove_cont (bool | list[bool], optional): Whether to subtract the continuum. Defaults to value in drone configuration files
            continuum_wn (int | list[int], optional): Continuum filter cutoff frequency in Hz. Defaults to value in drone configuration files
            remove_noise (bool | list[bool], optional): Whether to subtract noise. Defaults to value in drone configuration files
            noise_wn (int | list[int], optional): Noise filter cutoff frequency in Hz. Defaults to value in drone configuration files
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
            rfsoc_io.send_msg('PCS', f'{rtn1.session}')

            return rtn1

        def _write_filtered_comb(com_to, vna_files, save_files, filter_wn):
            '''
            '''
            num_drones = len(com_to)

            found_nums = num_drones*[None]
            good_resonators = num_drones*[None]
            for i, (com, vna_file, save_file, win) in enumerate(zip(com_to, vna_files, save_files, filter_wn)):
                # Filter out fake resonators from found detectors
                # -----------------------------------------------
                vna = VNA(com_to = com, data_path = vna_file) # Create VNA object

                good_resonator, _ = vna.filter_det_f(win = win) # Filter out fake resonators
                good_resonator = good_resonator.real
                good_resonators[i] = good_resonator

                # Save filtered resonators to .npy file
                # -------------------------------------
                ind = self.drone_list.index(com)
                res_dir = self.config_dirs[ind] / 'res'
                fname = f"{self.io_cfg['save_file_names'][f'vna_sweep']}_res_filtered_{self.timestamp}.npy"
                save_path = res_dir / fname
                np.save(save_path, good_resonator)

                # Add filtered resonators file path and number to config file
                # -----------------------------------------------------------
                drone_cfg = self.drone_cfg[ind]
                found_num = len(good_resonator)
                found_nums[i] = found_num
                rfsoc_io.edit_config(drone_cfg, 'found_num_detectors', found_num)
                rfsoc_io.edit_config(drone_cfg, 'found_detector_freqs_filtered', str(save_path))
                rfsoc_io.send_msg('INFO', f'Kept {found_num} detectors after filtering for drone {com}!')

                drone_cfg = rfsoc_io.save_config(save_file, drone_cfg, self.save_cfg)

            # Write a new target comb using the filtered resonator frequencies
            # ----------------------------------------------------------------
            rtn = self.write_config_comb(com_to = com_to, tone_freqs = good_resonators, tone_powers = num_drones*['gen'], tone_phis = num_drones*['gen'], rescale_power = True, gen_attempts = 1)

            return rtn, found_nums

        # Get com_to list
        # ---------------
        com_to, boards = autils.get_com_to(self, **kwargs)
        kwargs['setup'] = False # Do not set up drones again
        write_targ_comb = True # TODO: Does this need to be a kwarg?

        # Get parameters from drone config files
        # -------------------------------------
        peak_prom_std = autils.get_drone_args(self, com_to, ['det_find', 'peak_prom_std'])
        peak_prom_db  = autils.get_drone_args(self, com_to, ['det_find', 'peak_prom_db'])
        peak_dis      = autils.get_drone_args(self, com_to, ['det_find', 'peak_dis'])
        width_min     = autils.get_drone_args(self, com_to, ['det_find', 'width_min'])
        width_max     = autils.get_drone_args(self, com_to, ['det_find', 'width_max'])
        stitch        = autils.get_drone_args(self, com_to, ['det_find', 'stitch'])
        stitch_bw     = autils.get_drone_args(self, com_to, ['tones', 'sweep_steps'])
        stitch_sw     = autils.get_drone_args(self, com_to, ['det_find', 'stitch_sw'])
        remove_cont   = autils.get_drone_args(self, com_to, ['det_find', 'remove_cont'])
        continuum_wn  = autils.get_drone_args(self, com_to, ['det_find', 'continuum_wn'])
        remove_noise  = autils.get_drone_args(self, com_to, ['det_find', 'remove_noise'])
        noise_wn      = autils.get_drone_args(self, com_to, ['det_find', 'noise_wn'])
        phase_filter  = autils.get_drone_args(self, com_to, ['det_find', 'phase_filter'])
        filter_wn     = autils.get_drone_args(self, com_to, ['det_find', 'filter_wn'])

        # Parse key word arguments
        # ------------------------
        for key, value in kwargs.items():
            if key == 'peak_prom_std':
                peak_prom_std = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "peak_prom_std", peak_prom_std)
            elif key == 'peak_prom_db':
                peak_prom_db = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "peak_prom_db", peak_prom_db)
            elif key == 'peak_dis':
                peak_dis = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "peak_dis", peak_dis)
            elif key == 'width_min':
                width_min = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "width_min", width_min)
            elif key == 'width_max':
                width_max = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "width_max", width_max)
            elif key == 'stitch':
                stitch = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "stitch", stitch)
            elif key == 'sweep_steps':
                stitch_bw = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "sweep_steps", stitch_bw)
            elif key == 'stitch_sw':
                stitch_sw = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "stitch_sw", stitch_sw)
            elif key == 'continuum_wn':
                continuum_wn = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "continuum_wn", continuum_wn)
            elif key == 'remove_cont':
                remove_cont = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "remove_cont", remove_cont)
            elif key == 'remove_noise':
                remove_noise = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "remove_noise", remove_noise)
            elif key == 'noise_wn':
                noise_wn = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "noise_wn", noise_wn)
            elif key == 'phase_filter':
                phase_filter = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "phase_filter", phase_filter)
            elif key == 'filter_wn':
                filter_wn = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, "filter_wn", filter_wn)
            elif key == 'write_targ_comb':
                write_targ_comb = value

        # Take VNA sweep(s) if new_sweep = True or if one does not already exist
        # ----------------------------------------------------------------------
        to_sweep = []
        vna_files = len(com_to)*[None]
        timestamps = len(com_to)*[-1]
        for i, com in enumerate(com_to):
            ind = self.drone_list.index(com)
            # Check if a VNA sweep was taken in the last day
            vna_file = rfsoc_io.get_most_recent_file(self.vna_dirs[ind], f"{self.io_cfg['save_file_names']['vna_sweep'][0]}*", time_past=24*3600)
            if not vna_file.exists() or new_sweep:
                to_sweep.append(str(com))
            else:
                vna = VNA(com_to=com, data_path=vna_file)
                stitch_bw[i] = utils.dict_get(vna.drone_cfg, 'sweep_steps')
                vna_files[i] = vna_file
                timestamps[i] = pair.get_timestamp(vna_file)

        if len(to_sweep) != 0:
            # Take VNA sweep(s) without saving configs (saved later)
            self.save_cfg = False
            kwargs['com_to'] = to_sweep
            kwargs['sweep_steps'] = stitch_bw
            new_files = self.take_vna_sweep(**kwargs)
            for i in range(len(com_to)):
                if vna_files[i] is None: vna_files[i] = new_files.pop(0) # Add old vna files into list of new vna files sorted by drone com_to
                if timestamps[i] == -1: timestamps[i] = self.timestamp
            self.save_cfg = True
            kwargs['com_to'] = com_to

        # Find resonators using VNA sweep(s)
        # ----------------------------------
        rfsoc_io.send_msg('INFO', f"Finding resonators from VNA sweep for drones {com_to}!")
        for i, com in enumerate(com_to):
            # Find resonators from VNA sweep
            rtn = self.rfsoc.findVnaResonators(com_to = com, peak_prom_std = peak_prom_std[i], peak_prom_db = peak_prom_db[i], peak_dis = peak_dis[i],
                                                width_min = width_min[i], width_max = width_max[i], stitch = stitch[i], stitch_bw = stitch_bw[i], stitch_sw = stitch_sw[i],
                                                remove_cont = remove_cont[i], continuum_wn = continuum_wn[i], remove_noise = remove_noise[i], noise_wn = noise_wn[i], silent=False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}')
        rfsoc_io.send_msg('INFO', f"Finished finding resonators from VNA sweep for drones {com_to}!")

        # Save found resonators files
        # ---------------------------
        found_nums, found_freqs, save_files = self._save_resonators(com_to, timestamps, 'vna')

        if write_targ_comb:
            rfsoc_io.send_msg('INFO', f"Writing target combs using found resonators for drones {com_to}!")
            filt_com_to, unfilt_com_to = np.array(com_to)[phase_filter], np.array(com_to)[np.logical_not(phase_filter)]

            if not len(filt_com_to) == 0: # Write target comb with phase filtered resonators
                rfsoc_io.send_msg('INFO', f"Filtering resonators for drones {filt_com_to}!")
                rtn, found_nums = _write_filtered_comb(filt_com_to, np.array(vna_files)[phase_filter], np.array(save_files)[phase_filter], np.array(filter_wn)[phase_filter]) # TODO: Might want to seperate filtering and comb writing

            if not len(unfilt_com_to) == 0: # Write target comb using all found resonators
                kwargs['com_to'] = unfilt_com_to
                rtn = self._run_parallel(_write_targ_comb, **kwargs)
            rfsoc_io.send_msg('INFO', f'Finished writing target combs for drones {com_to}!')

            # Saved target comb has no noise tones so overwrite noise tone files
            for com in com_to: np.save(self.noise_files[self.drone_list.index(com)], [])

            # Write current comb to drone config files without saving to disk
            self.save_cfg = False
            for com in com_to: self._save_curr_comb(com, None)
            self.save_cfg = True

        return found_nums, vna_files

    @header
    @utils.method_timer
    def tune_tone_placement(self, new_sweep: bool = True, **kwargs):
        '''
        Find detectors using a target sweep

        Args:
            new_sweep (bool, optional): Whether to take a new target sweep. Defaults to True

            kwargs. See below:
            com_to (str | list[str], optional): List of drones. Defaults to drone list in IO configuration file
            chan_bw (float | list[float], optional): Bandwidth of sweep around each tone in MHz. Defaults to value in drone configuration files
            sweep_steps (int | list[int], optional): Number of steps each tone should take during sweep. Defaults to value in drone configuration files
            write_targ_comb (bool, optional): Whether to write target comb using found detector frequencies. Defaults to True

            tone_freqs (list[list[float]]: optional): Frequencies at which to place tones in Hz
            tone_powers (str | list[str] | float | list[float] | list[list[float]], optional): Tone powers in digital-to-analog (DAC) units (proportional to voltage). Use 'gen' to generate powers
            tone_phis (str | list[str] | float | list[float] | list[list[float]], optional): Tone phases in radians. Use 'gen' to generate phases
            write_comb (bool, optional): Whether to write a new target sweep comb. Defaults to True if any of ``tone_freqs``, ``tone_powers``, or ``tone_phis`` is specified

        Returns:
            output (arr of floats): Returns an array of the found detector frequencies
        '''
        def _place_min(com_to, com_inds):
            # TODO: Group stitch_bw args and run in parallel
            rfsoc_io.send_msg('INFO', f"Finding resonators from target sweep for drones {com_to}!")
            for i, com in zip(com_inds, com_to):
                # Find resonators from target sweep
                rtn = self.rfsoc.findTargResonators(com_to = com, stitch_bw = stitch_bw[i], silent=False)
                rfsoc_io.send_msg('PCS', f'{rtn.session}')
            rfsoc_io.send_msg('INFO', f"Finished finding resonators from target sweep for drones {com_to}!")

            found_nums, found_freqs, save_files = self._save_resonators(com_to, np.array(timestamps, dtype=int)[com_inds], 'targ')
            return [[]]*len(com_to)

        def _place_grad(com_to, com_inds):
            num_drones = len(com_to)

            found_freqs = num_drones*[[]]
            found_nums = num_drones*[None]
            com_timestamps = np.array(timestamps, dtype=int)[com_inds]
            for i, (com_ind, com, timestamp) in enumerate(zip(com_inds, com_to, com_timestamps)):
                wn, trim_savgol_wn, diff_savgol_wn = window[com_ind], trim_savgol_window[com_ind], diff_savgol_window[com_ind]
                trim_savgol_k, diff_savgol_k = trim_savgol_order[com_ind], diff_savgol_order[com_ind]

                ind = self.drone_list.index(com)
                drone_cfg = self.drone_cfg[ind]

                # Need to create target object and set drone_cfg manually since it has not yet been saved to disk
                num_tones = utils.dict_get(drone_cfg, 'num_tones')
                targ = Target(com_to=com, data_path = targ_files[com_ind], tones = range(num_tones))
                targ.drone_cfg = drone_cfg

                det = Detector(com_to = com, targ=targ)
                
                found_freq = det.IQ_max_dist(trim_window = wn,
                                             trim_savgol_window=trim_savgol_wn,
                                             diff_savgol_window=diff_savgol_wn,
                                             trim_savgol_k=trim_savgol_k,
                                             diff_savgol_k=diff_savgol_k,
                                             max_workers=1).to_numpy().T[1]
                found_freqs[i] = found_freq
                # Save filtered resonators to .npy file
                # -------------------------------------
                res_dir = self.config_dirs[ind] / 'res'
                fname = f"{self.io_cfg['save_file_names'][f'targ_sweep']}_res_grad_{self.timestamp}.npy"
                save_path = res_dir / fname
                np.save(save_path, found_freq)

                # Add filtered resonators file path and number to config file
                # -----------------------------------------------------------
                found_num = len(found_freq)
                found_nums[i] = found_num
                rfsoc_io.edit_config(drone_cfg, 'found_num_detectors', found_num)
                rfsoc_io.edit_config(drone_cfg, 'found_detector_freqs', str(save_path))

                save_file = self.config_dirs[ind] / f"{self.io_cfg['save_file_names']['targ_sweep']}_config_drone_{self.timestamp}{f'_{timestamp}' if not timestamp == self.timestamp else ''}.yaml"
                drone_cfg = rfsoc_io.save_config(save_file, drone_cfg, self.save_cfg)
            return found_freqs

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
            rtn = self.rfsoc.writeTargCombFromTargSweep(com_to = com, silent = False)
            rfsoc_io.send_msg('PCS', f'{rtn.session}')

            return rtn

        com_to, boards = autils.get_com_to(self, **kwargs)
        kwargs['setup'] = False # Do not set up drones again

        # Evaluate kwargs
        write_targ_comb = True
        stitch_bw = autils.get_drone_args(self, com_to, ['tones', 'sweep_steps'])
        method = autils.get_drone_args(self, com_to, ['tune_placement', 'method'])
        window = autils.get_drone_args(self, com_to, ['tune_placement', 'grad', 'trim', 'window'])
        trim_savgol_window = autils.get_drone_args(self, com_to, ['tune_placement', 'grad', 'trim', 'savgol_window'])
        diff_savgol_window = autils.get_drone_args(self, com_to, ['tune_placement', 'grad', 'diff', 'savgol_window'])
        trim_savgol_order = autils.get_drone_args(self, com_to, ['tune_placement', 'grad', 'trim', 'savgol_order'])
        diff_savgol_order = autils.get_drone_args(self, com_to, ['tune_placement', 'grad', 'diff', 'savgol_order'])

        # Parse key word arguments
        for key, value in kwargs.items():
            if key == 'sweep_steps':
                stitch_bw = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, 'sweep_steps', stitch_bw)
            elif key == 'method':
                method = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, ['tune_placement', 'method'], method)
            elif key == 'window':
                window = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, ['tune_placement', 'grad', 'trim', 'window'], method)
            elif key == 'trim_savgol_window':
                trim_savgol_window = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, ['tune_placement', 'grad', 'trim', 'savgol_window'], method)
            elif key == 'diff_savgol_window':
                diff_savgol_window = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, ['tune_placement', 'grad', 'diff', 'savgol_window'], method)
            elif key == 'trim_savgol_order':
                trim_savgol_order = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, ['tune_placement', 'grad', 'trim', 'savgol_order'], method)
            elif key == 'diff_savgol_order':
                diff_savgol_order = autils.parse_args(self, com_to, value)
                autils.set_drone_args(self, com_to, ['tune_placement', 'grad', 'diff', 'savgol_order'], method)
            elif key == 'write_targ_comb':
                write_targ_comb = value

        # Take target sweep if new_sweep = True if one does not already exist
        to_sweep = []
        targ_files = len(com_to)*[None]
        timestamps = len(com_to)*[-1]
        for i, com in enumerate(com_to):
            ind = self.drone_list.index(com)
            # Check if a target sweep was taken in the last day
            targ_file = rfsoc_io.get_most_recent_file(self.targ_dirs[ind], f"{self.io_cfg['save_file_names']['targ_sweep'][0]}*", time_past=24*3600)
            if not targ_file.exists() or new_sweep:
                to_sweep.append(com)
            else:
                target = Target(com_to=com, data_path=targ_file)
                stitch_bw[i] = utils.dict_get(target.drone_cfg, 'sweep_steps')
                targ_files[i] = targ_file
                timestamps[i] = pair.get_timestamp(targ_file)

        if len(to_sweep) != 0:
            # Take target sweep(s) without saving config (saved later)
            self.save_cfg = False
            kwargs['com_to'] = to_sweep
            kwargs['sweep_steps'] = stitch_bw
            new_files = self.take_target_sweep(**kwargs)
            for i in range(len(com_to)):
                if targ_files[i] is None: targ_files[i] = new_files.pop(0) # Add old vna files into list of new vna files sorted by drone com_to
                if timestamps[i] == -1: timestamps[i] = self.timestamp
            self.save_cfg = True
            kwargs['com_to'] = com_to

        methods = {'min':  _place_min,
                   'grad': _place_grad}

        tone_freqs = [None]*len(com_to)
        for method_name, method_func in methods.items():
            method_inds = [i for i, com_method in enumerate(method) if com_method == method_name]
            method_com_to = np.array(com_to)[method_inds].tolist()
            if len(method_com_to) > 0:
                tone_freq = method_func(method_com_to, method_inds)
                for ind, freq in zip(method_inds, tone_freq): tone_freqs[ind] = freq
        if write_targ_comb:
            rfsoc_io.send_msg('INFO', "Writing target comb using tuned tone frequencies for drones %s!", com_to)

            untuned_com_to = [] # Drones with an invalid tuning method specified
            primecam_readout_com_to = [] # Drones with primecam_readout tuning method specified
            ccatkidlib_com_to, ccatkidlib_inds = [], [] # Drones with ccatkidlib tuning method specified
            for i, (com, freq) in enumerate(zip(com_to, tone_freqs)):
                if freq is None:
                    untuned_com_to.append(com)
                elif len(freq) == 0:
                    primecam_readout_com_to.append(com)
                else:
                    ccatkidlib_com_to.append(com), ccatkidlib_inds.append(i)

            if len(untuned_com_to) > 0:
                rfsoc_io.send_msg('WARNING', "Invalid tuning method specified for drones %s! Not writing target comb for these drones.", untuned_com_to)
            if len(primecam_readout_com_to) > 0:
                kwargs['com_to'] = primecam_readout_com_to
                self._run_parallel(_write_targ_comb, **kwargs)
            if len(ccatkidlib_com_to) > 0:
                kwargs['com_to'], kwargs['tone_freqs'] = ccatkidlib_com_to, [tone_freqs[ind] for ind in ccatkidlib_inds]
                for k, v in kwargs.items():
                    if isinstance(v, Iterable) and len(v) == len(com_to): kwargs[k] = [v[ind] for ind in ccatkidlib_inds]
                rtn = self.write_config_comb(**kwargs)

            rfsoc_io.send_msg('INFO', 'Finished writing target combs for drones %s!', primecam_readout_com_to + ccatkidlib_com_to)

            # Saved target comb has no noise tones so overwrite noise tone files
            for com in com_to: np.save(self.noise_files[self.drone_list.index(com)], [])

            # Write current comb to drone config files without saving to disk
            self.save_cfg = False
            for com in com_to: self._save_curr_comb(com, None)
            self.save_cfg = True
        return targ_files

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
            save_path = dirs[ind] / fname

            bid, drid = com.split('.')
            data = None

            try:
                drone_id = f"_{bid}_{drid}*"
                file = rfsoc_io.get_most_recent_file(self.tmp_data_dir, file_identifier = self.io_cfg['board_file_names'][f'{sweep}']['s21'] + drone_id, time_past=time.time() - float(self.timestamp))

                # Save the data with name specified in system config file
                if self.save_data:
                    data = rfsoc_io.get_array(file, save_path, action = 'mv', load = False)
                    if data is None:
                        rfsoc_io.send_msg('WARNING', f'Failed to find {sweep} sweep file for drone {com} in {self.tmp_data_dir}! Fetching file directly from board instead.')
                        raise FileNotFoundError
                else:
                    data = file if Path(file).exists() else None
            except FileNotFoundError:
                bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
                path = self.drone_dir / f'drone{drid}' / sweep

                with rfsoc_io.get_connection(bip, ssh_key) as c:
                    # Get most recent sweep file in sweep directory
                    file = rfsoc_io.get_most_recent_file_board(c, path, file_identifier = self.io_cfg['board_file_names'][f'{sweep}']['s21'], time_past=time.time() - float(self.timestamp))

                    # Save the data with name specified in system config file
                    if self.save_data:
                        data = rfsoc_io.get_array_board(c, bip, ssh_key, file, save_path, load = False)
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

            rfsoc_io.send_msg('DEBUG', f'Successfully copied {sweep} file from drone {com}!')
            paths.append(save_path)

        # Save ext config
        self.ext_cfg = rfsoc_io.save_config(self.log_dir / f"{self.io_cfg['save_file_names'][f'{sweep}_sweep']}_config_ext_{self.timestamp}.yaml", self.ext_cfg, self.save_cfg)

        return files, paths

    def _save_resonators(self, com_to, timestamps, sweep):

        found_nums = []
        found_freqs = []
        save_files = []
        for com, timestamp in zip(com_to, timestamps):
            # Parse com information
            ind = self.drone_list.index(com)
            bid, drid = com.split('.')

            # Define directory to store resonator files
            res_dir = self.config_dirs[ind] / 'res'
            fname = f"{self.io_cfg['save_file_names'][f'{sweep}_sweep']}_res_min_{self.timestamp}.npy"

            try:
                drone_id = f"_{bid}_{drid}*"
                res_file = rfsoc_io.get_most_recent_file(self.tmp_data_dir, file_identifier = self.io_cfg["board_file_names"][f'{sweep}']['res'] + drone_id, time_past = time.time() - float(self.timestamp) + 30)
                res_path = rfsoc_io.get_array(res_file, res_dir / fname, action = 'mv', load = False)
                if res_path is None: raise FileNotFoundError
            except FileNotFoundError:
                bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
                ssh_key = self.io_cfg['file_paths']['ssh_key']
                path = self.drone_dir / f'drone{drid}' / f'{sweep}'

                with rfsoc_io.get_connection(bip, ssh_key) as c:
                    # Get file with found resonators
                    res_file = rfsoc_io.get_most_recent_file_board(c, path, file_identifier = self.io_cfg["board_file_names"][f'{sweep}']['res'], time_past = time.time() - float(self.timestamp) + 30)
                    res_path = rfsoc_io.get_array_board(c, bip, ssh_key, res_file, res_dir / fname, load = False)

            # Save resonator frequncies and number of found resonators to list
            try:
                found_freq = np.load(res_dir / fname, mmap_mode='r')
                found_num = len(found_freq)
            except FileNotFoundError:
                rfsoc_io.send_msg('ERROR', f"Failed to retrieve found resonators file!")
                found_num = None
            found_nums.append(found_num)
            found_freqs.append(found_freq)

            rfsoc_io.send_msg('INFO', f"Found {found_num} detectors for drone {com}!")

            rfsoc_io.edit_config(self.drone_cfg[ind], 'found_num_detectors', found_num)
            rfsoc_io.edit_config(self.drone_cfg[ind], 'found_detector_freqs', res_path)

            save_file = self.config_dirs[ind] / f"{self.io_cfg['save_file_names'][f'{sweep}_sweep']}_config_drone_{self.timestamp}{f'_{timestamp}' if not timestamp == self.timestamp else ''}.yaml"
            save_files.append(save_file)
            self.drone_cfg[ind] = rfsoc_io.save_config(save_file, self.drone_cfg[ind], self.save_cfg)

        self.ext_cfg = rfsoc_io.save_config(self.log_dir / f"{self.io_cfg['save_file_names'][f'{sweep}_sweep']}_config_ext_{self.timestamp}.yaml", self.ext_cfg, self.save_cfg)
        return found_nums, found_freqs, save_files

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

        def _check_comb(key, value):
            # Check if the tone frequencies are within the bandwidth of the RFSoC
            if len(value) > 1024: value = value[:1024]
            if key == 'tone_freqs':
                value = np.array(value)
                if np.max(np.abs(value - self.drone_cfg[ind]['tones']['NCLO']*1e6))  > self.drone_cfg[ind]['tones']['full_bandwidth']*1e6/2:
                    rfsoc_io.send_msg('WARNING', f"{value} contains frequencies outside of the RFSoC bandwidth. Not writing {key} custom comb!")
                    return None

                # Ensure that all tones are spaced at least 500 Hz apart to prevent dropping tones (using 500 Hz for margin, should be able to go down to ~244 Hz)
                attempts = 0
                diffs = np.abs(np.diff(value, prepend=False))
                while any(diffs < 500):
                    value[diffs < 500] += 500
                    attempts += 1
                    diffs = np.abs(np.diff(value, prepend=False))
                    if attempts > 10:
                        return None
            return value

        def _reset_comb(ip, ssh_key, comb_dict):
            for key, value in comb_dict.items():
                rfsoc_io.save_array_board(ip, ssh_key, value['path'], utils.arr_to_list(value['comb']), self.tmp_dir)

            # Return unmodified comb
            return comb_dict['tone_freqs']['comb'], comb_dict['tone_powers']['comb'], comb_dict['tone_phis']['comb']

        def _comb_peak(freqs, amps, phis):
            '''
            Function from tones.py in primecam_readout. Returns the maximum power produced by the comb.
            '''
            x, _, _ = alcove_base.generateWaveDdr4(freqs, amps, phis)
            x.real, x.imag = x.real.astype("int16"), x.imag.astype("int16")
            return np.max(np.abs(x.real + 1j*x.imag))

        def _write_new_comb(ip, ssh_key, comb_dict, key, comb):
            comb_dict[key]['num_tones'] = len(comb)
            rfsoc_io.save_array_board(ip, ssh_key, comb_dict[key]['path'], utils.arr_to_list(comb), self.tmp_dir)
            comb_dict[key]['new_comb'] = comb
            return comb_dict

        ind = self.drone_list.index(com)
        bid, drid = com.split('.')
        bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
        ssh_key = self.io_cfg['file_paths']['ssh_key']
        custom_comb_dir = self.drone_dir / f'drone{drid}' / 'custom_comb'

        gen_amps = False
        gen_phis = False

        # Evaluate passed key word arguments
        # ----------------------------------
        connection = None
        for key, value in kwargs.items():
            if key == 'connection':
                connection = value

        if connection is None:  connection = rfsoc_io.get_connection(bip, ssh_key)
        # Open connection to RFSoC board
        with connection as c:
            # Get current comb and paths to custom comb files
            # -----------------------------------------------
            freq_comb, amp_comb, phi_comb = self._get_curr_comb(com, c = c)
            if freq_comb[0] < 0: freq_comb = np.array(freq_comb) + self.drone_cfg[ind]['tones']['NCLO']*1e6

            freq_path = rfsoc_io.get_most_recent_file_board(c, custom_comb_dir, file_identifier = self.io_cfg["board_file_names"]['targ']["comb_freq"], time_past=np.inf)
            amp_path  = rfsoc_io.get_most_recent_file_board(c, custom_comb_dir, file_identifier = self.io_cfg["board_file_names"]['targ']["comb_amp"], time_past=np.inf)
            phi_path  = rfsoc_io.get_most_recent_file_board(c, custom_comb_dir, file_identifier = self.io_cfg["board_file_names"]['targ']["comb_phi"], time_past=np.inf)

            # Combine comb, paths, and num_tones into single dictionary
            comb_dict = {"tone_freqs": {"path": freq_path, "comb": freq_comb, "new_comb": None, "num_tones": None},
                         "tone_powers": {"path": amp_path, "comb": amp_comb, "new_comb": None, "num_tones": None},
                         "tone_phis": {"path": phi_path, "comb": phi_comb, "new_comb": None, "num_tones": None}}

            # Parse comb values
            # -----------------
            for key, value in kwargs.items():
                # Get file path and comb corresponding to correct key
                try:
                    path = comb_dict[key]["path"]
                    if not rfsoc_io.path_exists(c, path):
                        rfsoc_io.send_msg('ERROR', f"Could not find {key} custom comb file at path {path}.")
                        raise FileNotFoundError
                    rfsoc_io.send_msg('DEBUG', f'Modifying {key} for drone {com}!')
                except:
                    comb_dict = _write_new_comb(bip, ssh_key, comb_dict, key, comb_dict[key]['comb'])
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
                        rfsoc_io.send_msg('WARNING', f"{value} is not a valid {key} custom comb file path for drone {com}!")
                        comb_dict = _write_new_comb(bip, ssh_key, comb_dict, key, comb_dict[key]['comb'])
                        continue

                    value = np.load(value)
                # If not file path, assume value is a number or an array of numbers
                else:
                    try:
                        # Assume an array is passed and check if its non-empty
                        if len(value) == 0:
                            rfsoc_io.send_msg('WARNING', f"'{value}' is an empty array, not writing to {key} custom comb file for drone {com}!")
                            comb_dict = _write_new_comb(bip, ssh_key, comb_dict, key, comb_dict[key]['comb'])
                            continue
                    except TypeError:
                        if np.issubdtype(type(value), (np.integer, np.floating)): # Assume a number is passed and use the same number for all tones
                            comb_dict[key]['num_tones'] = float(value) # Store the number in num_tones as a float for processing later since the number of tones in the comb may be unknown
                            continue
                        else:
                            # Invalid value passed; ignore and move on to next comb
                            rfsoc_io.send_msg('WARNING', f"{value} is not a valid file path, array, or number! Not writing {key} custom comb file for drone {com}.")
                            comb_dict = _write_new_comb(bip, ssh_key, comb_dict, key, comb_dict[key]['comb'])
                            continue

                value = _check_comb(key, value)
                if value is None:
                    comb_dict = _write_new_comb(bip, ssh_key, comb_dict, key, comb_dict[key]['comb'])
                    continue

                # Copy the array with custom parameters onto the rfsoc board
                comb_dict = _write_new_comb(bip, ssh_key, comb_dict, key, value)

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
                    return _reset_comb(bip, ssh_key, comb_dict)


            # Write comb for num_tone entries that are floats (only single number was passed)
            # -------------------------------------------------------------------------------
            for key, value in comb_dict.items():
                comb_val = value['num_tones']
                if type(comb_val) is float:
                    if tone_num is None:
                        tone_num = 1
                        rfsoc_io.send_msg('WARNING', f"Only numbers were passed as custom comb; assuming a single tone should be written! To avoid unexpected errors, please pass numbers as a list to write a single tone in the future.")
                    comb_val = comb_val * np.ones(tone_num)
                    comb_val = _check_comb(key, comb_val)
                    if comb_val is not None:
                        comb_dict = _write_new_comb(bip, ssh_key, comb_dict, key, comb_val)
                    else:
                        # Resetting back to current comb
                        rfsoc_io.send_msg('ERROR', f"The comb failed a check causing the number of tones in the frequeny, power, and phase combs to not match. Reverting back to current comb!")
                        return _reset_comb(bip, ssh_key, comb_dict)

            # Generate phi comb - As done in primecam_readout
            # -----------------------------------------------
            # Define maximum RFSoC DAC power
            max_power = self.io_cfg['boards'][f'b{bid}']['max_power']
            tone_freqs = comb_dict['tone_freqs']['new_comb']
            tone_freqs_bb = np.array(tone_freqs) - self.drone_cfg[ind]['tones']['NCLO']*1e6

            # Determine if tone powers need to be generated
            # ---------------------------------------------
            tone_powers = np.ones(tone_num)*max_power/np.sqrt(tone_num) if gen_amps else comb_dict['tone_powers']['new_comb']

            # Generate tone phis and tone powers
            # ----------------------------------
            # Generate phis (as done in primecam_readout tones.py genAmpsAndPhis)
            if gen_phis:
                best_peak = float('inf')
                tone_phis = None
                for _ in range(gen_attempts):
                    phis = np.random.uniform(-np.pi, np.pi, tone_num)
                    comb_max = _comb_peak(tone_freqs_bb, tone_powers, phis)
                    if comb_max < best_peak:
                        best_peak = comb_max
                        tone_phis = phis

                # Save phi comb and power comb (if necessary)
                rfsoc_io.save_array_board(bip, ssh_key, phi_path, utils.arr_to_list(tone_phis), self.tmp_dir)
            else:
                tone_phis = comb_dict['tone_phis']['new_comb']
                best_peak = _comb_peak(tone_freqs_bb, tone_powers, tone_phis)

            # Check if max comb power is less than max DAC power. Rescale power if specified
            if best_peak > max_power and rescale_power:
                rfsoc_io.send_msg('WARNING', f"The custom comb has points(s) with output power {best_peak} which exceeds the maximum DAC power of {max_power}! Rescaling to lower tone power.")
                tone_powers *= max_power/best_peak

            if gen_amps or rescale_power: rfsoc_io.save_array_board(bip, ssh_key, amp_path, utils.arr_to_list(tone_powers), self.tmp_dir)
            rfsoc_io.send_msg('INFO', f"Saved custom comb for drone {com}!")

            # Edit comb saved in drone config files
            # -------------------------------------
            # Frequency comb
            rfsoc_io.edit_config(self.drone_cfg[ind], ['tones', 'tone_freqs'], utils.arr_to_list(tone_freqs))

            # Amplitude comb
            rfsoc_io.edit_config(self.drone_cfg[ind], ['tones', 'tone_powers'], utils.arr_to_list(tone_powers))

            # Phase comb
            rfsoc_io.edit_config(self.drone_cfg[ind], ['tones', 'tone_phis'], utils.arr_to_list(tone_phis))

            # Number of tones
            rfsoc_io.edit_config(self.drone_cfg[ind], ['tones', 'num_tones'], int(tone_num))
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
            drone_id = f"_{bid}_{drid}*"
            # Get most recent comb file paths
            files = [rfsoc_io.get_most_recent_file(self.tmp_data_dir, file_identifier = [self.io_cfg["board_file_names"]['vna'][name] + drone_id, self.io_cfg["board_file_names"]['targ'][name] + drone_id], time_past = np.inf) for name in names]

            # Ensure that all comb files exist
            if any(not file.exists() for file in files): raise FileNotFoundError

            # Ensure that all files are from the same comb (all have the id)
            ids = [_get_id(file) for file in files]
            if not len(set(ids)) == 1: raise FileNotFoundError
            if not load: return *tuple(files), False

            # Load most recent comb files
            combs = [rfsoc_io.get_array(file, self.tmp_dir, action='cp', timestamp=True, load=True) for file in files]
        except FileNotFoundError:
            # Parse com information
            bip = self.io_cfg['boards'][f'b{bid}']['board_ip']
            ssh_key = self.io_cfg['file_paths']['ssh_key']

            # Create connection to board if Connection object not passed
            if connection is None: connection = rfsoc_io.get_connection(bip, ssh_key)

            # Define directory where sweep combs are saved
            comb_dir = self.drone_dir / f'drone{drid}' / 'comb'

            # Get most recent comb frequency, amplitude, and phase files
            # ----------------------------------------------------------
            with connection as c:
                # Get most recent comb file paths
                files = [rfsoc_io.get_most_recent_file_board(c, comb_dir, file_identifier = self.io_cfg["board_file_names"]['targ'][name], time_past = np.inf) for name in names]

                # ADD LOGIC FOR FILES NOT BEING ON BOARD EITHER
                if not load: return *tuple(files), True

                # Load most recent comb files
                combs = [rfsoc_io.get_array_board(c, bip, ssh_key, file, self.tmp_dir, timestamp=True, load=True) for file in files]

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
            connection = rfsoc_io.get_connection(bip, ssh_key)

            with connection as c:
                # Copy most recent comb files
                freq_path = rfsoc_io.get_array_board(c, bip, ssh_key, freq_file, freq_name, load = False, timestamp=timestamp)
                amp_path  = rfsoc_io.get_array_board(c, bip, ssh_key,  amp_file,  amp_name, load = False, timestamp=timestamp)
                phi_path  = rfsoc_io.get_array_board(c, bip, ssh_key,  phi_file,  phi_name, load = False, timestamp=timestamp)
        else:
            # Copy most recent comb files
            freq_path = rfsoc_io.get_array(freq_file, freq_name, action='cp', load = False, timestamp=timestamp)
            amp_path  = rfsoc_io.get_array(amp_file,  amp_name, action='cp', load = False, timestamp=timestamp)
            phi_path  = rfsoc_io.get_array(phi_file,  phi_name, action='cp', load = False, timestamp=timestamp)

        rfsoc_io.edit_config(self.drone_cfg[ind], ['tones', 'tone_freqs'], freq_path)
        rfsoc_io.edit_config(self.drone_cfg[ind], ['tones', 'tone_powers'], amp_path)
        rfsoc_io.edit_config(self.drone_cfg[ind], ['tones', 'tone_phis'], phi_path)

        if freq_path is not None: rfsoc_io.edit_config(self.drone_cfg[ind], ['tones', 'num_tones'], len(np.load(freq_path, mmap_mode='r')))

        # Save drone config
        self.drone_cfg[ind] = rfsoc_io.save_config(self.config_dirs[ind] / f"{name}_config_drone_{self.timestamp}.yaml", self.drone_cfg[ind], self.save_cfg)

    #================#
    # Helper Methods #
    #================#

    def _parse_ocs_session(self, sess: dict) -> tuple[bool, dict]:
        '''
        Parse session dictionary of OCS Reply object to determine if the OCS command was successful and to get the data.

        Args:
            sess (dict): .session attribute of ocs.ocs_client.OCSReply object

        Returns:
            tuple[bool, dict]: Whether the OCS command was successful, the data returned by the command ({} if no data)
        '''
        rfsoc_io.send_msg('PCS', f'{sess}')
        return sess['success'], sess['data']

    def _run_parallel(self, func: Callable, *args, parallel_boards: int | None = None, parallel_drones: int | None = None, **kwargs) -> list[any]:
        ''' Run the specified function ``func`` with drones specified by ``com_to`` in parallel.

        Args:
            func (Callable): Function to run in parallel
            parallel_boards (int | None, optional): Number of boards to run in parallel. Must be a positive integer or -1 to specify all boards. Defaults to value in IO configuration file
            parallel_drones (int | None, optional): Number of drones to run in parallel. Must be one of {1, 2, 3, 4}. Defaults to value in IO configuration file

            args: Positional arguments of ``func``
            kwargs. Key word arguments of `func``. Also, see below:
            com_to (str | list[str]): List drones to run ``func`` with

        Returns:
            return (list[any]): List of return values from ``func``
        '''

        # Parse key word arguments
        # ------------------------
        com_to, boards = autils.get_com_to(self, **kwargs)
        name = func.__name__

        if parallel_boards is None: parallel_boards = self.parallel_boards
        if parallel_drones is None: parallel_drones = self.parallel_drones

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

        com_arrs = [com_arr for com_arr in com_arrs if com_arr] # Remove empty com_to arrays

        s = Style()
        rtn_list = [None] * len(com_arrs)
        for i, com_arr in enumerate(com_arrs):
            rfsoc_io.send_msg('DEBUG', f'Running {s.func_name(name)} for drones {com_to}!')
            rtn_list[i] = func(str(com_arr), *args, **kwargs)

        return rtn_list
