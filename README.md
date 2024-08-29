# ccatkidlib
Data acquisition, tuning, and analysis library for CCAT microwave kinetic inductance detectors (MKIDs) using radio frequency system on chip (RFSoC). 

## Dependencies
The following are required for data acquisition with the RFSoC:
- [primecam_readout](https://github.com/TheJabur/primecam_readout) - Underlying software used to communicate with RFSoC
- [ocs](https://github.com/simonsobs/ocs?tab=readme-ov-file) - Commands are sent to RFSoC via a OCS agent that can be built locally from `queen_agent.py` in primecam_readout


## Initial Setup and Sample Usage 

Before one can begin acquiring data, system specific fields (e.g., IP addresses or file paths) in the configuration file must be properly set (see [Configuration Files](#configuration-files)). 

Once the configuration is set, the main data acquisiton object `R` can be imported from `rfsoc_daq.py` and intialized. All data acquisition and tuning is performed using methods of the `R` control object (see [DAQ](#data-acquisition-daq)). 

For example, the following script would be run to find detectors in a MKID array and take timestreams at the found frequencies. 

```python
# Import and intialize RFSoC control object
from rfsoc_daq.py import R
R = R()

# Roughly find detectors using VNA sweep
R.find_detectors()

# Refine detector frequencies using target sweep
R.find_detectors_fine()

# Take 60 second timestream at found frequencies
R.take_timestream(60)
```

The raw data products are saved in `/data` inside the following subdirectories:
- `/vna` - Stores VNA sweep files
- `/targ` - Stores target sweep files
- `/stream` - Stores timestream files
- `/rfsoc` - Stores log and config files

> [!TIP]
The location of `/data` and the filenames of the data products can be specified in the config file. 

## Configuration Files

The default configuration file for the `R` control object is `config.yaml`  in `/rfsoc`. Alternatively, the path of custom config file can be passed as an optional parameter to `R` during initialization.

> [!WARNING]
 Passing a custom config file to `R` that does not contain all the fields of the default config may cause errors. It is recommended to only *add* additional fields.

The default config file is split into two sections:
1. The first section contains fields that may change during (or frequently in-between) data acquisition sessions (e.g., RFSoC comb frequencies and powers). 
2. The second section contains fields that are not expected to change frequently (e.g., file paths, IP addresses, file names, etc.) 

Both sections of the config file are saved upon initialization of `R` in `/data/rfsoc`. Additionally, the first section of the config file is saved after every vna sweep, target sweep, and timestream that is run. Saving of config files can be disabled by setting `save_config_copy` to `False` in the config file.

> [!TIP]
Additional parameters can be added to the first section of the config file, which can be especially useful for tracking parameters that change during a data acquisition session (e.g., the temperature of a coldload).

 The `edit_main_config` method can be used If a field in the first section of the config file needs to be edited during a data acquisition session. The `reload_config` method can be used if the config file needs to be reloaded.

## Data Acquisition (DAQ)

All the code used for data acquisition (DAQ) can be found in `/rfsoc`. Data acquisition using the RFSoC is done through the control object `R` defined in `rfsoc_daq.py` and the module `rfsoc_io.py` provides functions for IO operations such as directory creation, file reading/writing, and logging.

There are three main data acquisition methods that can be run through `R`:
- `take_vna_sweep` - Takes a 500 MHz VNA sweep
- `take_target_sweep` - Takes a target sweep with the specified comb 
- `take_timestream(t)` - Takes a t second timestream with the specifed comb

As stated above, taking target sweeps and timestreams requires a comb to be specified. Every comb requires the following information:
- Tone frequencies - Frequencies at which to write tones
- Tone powers - Power of each tone *(can be different for each tone)*
- Tone phases - Phase of each tone (useful for preventing interference between tones)

The tone frequencies, tone powers, and tone phases are each specified by and stored in an array. There are three ways the comb arrays can be specified; here they are listed in order of priority:
1. Pass tone frequencies, powers, and/or phases as keyword argument(s) to `take_target_sweep` and `take_timestream(t)` using the keywords `tone_freqs`, `tone_powers`, and `tone_phis` respectively. These keywords can also be passed to other methods that use `take_target_sweep` or `take_timestream(t)`.
2. Edit the fields `tone_freqs`, `tone_powers`, and/or `tone_phis` in the configuration file. 
3. If the comb parameters are not specified through 1. or 2., the most recently used comb parameters will be used again. These parameters can be found in `/data/tmp`.

Along with data acquisition, basic detector locating and tuning can be done through `R`. Detectors can be located using the methods:
- `find_detectors` - Finds rough frequencies of detectors using VNA sweep and writes comb using found frequencies. Useful when frequencies are completly unknown to get initial comb.  
- `find_detectors_fine`- Finds frequencies of detectors using target sweep and writes comb using found frequencies. Useful to run after `find_detectors` or if comb with rough detector frequencies is specifed in some other manner.

> [!TIP]
> Both detector finding methods can be passed the keyword argument `new_sweep = False`. This can be useful to experiment with the detector finding algorithm input parameters without having to take new sweeps each time.

## File Output

All output files can be found in the `/data` directory whose location is specified in the configuration file. The following subdirectories are within `/data`:
- `/vna` - Stores VNA sweeps
- `/targ` - Stores target sweeps
- `/timestream` - Stores timestreams
- `/rfsoc` - Stores configuration files and logs
- `/tmp` - Stores most recently used comb parameters
- `/fig` - Stores output figures

All directories except `/tmp` have the following additional substructure:
- `/20240828` - Current date in yyyymmdd format
    - `/B1D1`- RFSoC board number, drone number
        - `/1724940799` - 10 digit UNIX timestamp when `R` was initialized

The output filenames for VNA sweeps, target sweeps, and timestreams can be specified in the config file, but the time the operation was performed is always appended to the filename. After each of these three operations, a snapshot of the configuration file is saved with the filename containing an identical timestamp as the output data filename.

The output data is stored as follows for the three operations.

### VNA Sweeps
```python
[[Sweep Frequencies],
 [Complex S21 data]]
```
### Target Sweeps
```python
[[Sweep Frequencies],
 [Complex S21 data]]
```
### Timestreams
```python
[[Tone 1 Complex S21 data],
 [Tone 2 Complex S21 data],
 .
 .
 .
 [Tone X Comlex S21 data]
]
```

## Analysis (Work in progress, Subject to change)

All of the code used for analysis can be found in `/analysis`. There are three main classes used for analysis and plotting:
- `Sweep` - Contains methods for analyzing stand alone target sweep data of a single MKID
- `Timestream` - Contains methods for analyzing stand alone timestream data of a single MKID
- `Resonator` - Represents a single MKID. Associated `Sweep` and `Timestream` objects can be added to `Resonator` to perform more complex data analysis.
