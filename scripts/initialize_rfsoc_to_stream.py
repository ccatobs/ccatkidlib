"""
    initialize_rfsoc_to_stream.py

    Test script to demonstrate functionality of rfsoc PCS agent.
    Once drone is set up, runs all the necessary Python code on the control computer
    to set up the RFSoC and stream packets over the network.

    Variables for the various rfsoc functions are generally set at the beginning
    of the script.
"""
import numpy as np

from ocs.ocs_client import OCSClient

# Initialize agents
rfsoc = OCSClient('queenagent', args=[])

# Set up variables
com_to = '1.1' # str
lo_for_vna = 700 # int
  # For VNA peak finding
prom_dB = 0.25 # float
width_min = 2 # int, number of bins
width_max = 500 # int, number of bins
distance = 80000 # int, number of bins
  # Wide target sweep
chan_bandwidth = 5 # float, MHz

# Set up NCLO
rfsoc.setNCLO(com_to=com_to,f_lo=lo_for_vna)
print("Finished NCLO Setup")

# Run VNA sweep and find resonators
rfsoc.writeNewVnaComb(com_to=com_to)
print("Wrote New VNA Comb")
rfsoc.vnaSweep(com_to=com_to) # default N_steps=500
print("Took VNA Sweep")
rfsoc.findVnaResonators(com_to=com_to,prom_dB=prom_dB,width_min=width_min,\
                        width_max=width_max, distance=distance)
print("Found VNA Resonators")

# Run wide target sweep
rfsoc.writeTargCombFromVnaSweep(com_to=com_to)
print("Wrote Target Comb from VNA Sweep")
rfsoc.targetSweep(com_to=com_to,chan_bandwidth=chan_bandwidth)
print("Took Target Sweep")
# Find frequencies with RFSoC
rfsoc.findTargResonators(com_to=com_to)
print("Found Targ Resonators")

# Two options: 1) could just write tones at this point with primecam_readout
#              2) could run analysis code to find point of highest sensitivity

# If writing with primecam_readout
rfsoc.writeTargCombFromTargSweep(com_to=com_to)
print("Wrote Targ Comb from Targ Sweep")

# If looking for point of highest sensitivity:
# Clean target sweep and fit for frequencies
# In phase space, look for max derivative of dphi/df
# Write custom tone at these locations
# None of these things are implemented yet here!
