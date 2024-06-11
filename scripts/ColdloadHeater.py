from ocs.ocs_client import OCSClient
from tqdm import tqdm
import time
import sys
import logging 

class ColdloadHeater:

    def __init__(self, ocs_client = 'LSA2FQZ', range = "low", percent = 0, wait_time = 1.20, log = False):
        # Initialize OCSClient
        self.percent = percent
        self.ls336 = OCSClient(ocs_client, args=[])
        self.ls336.acq.start()
        self.to_log = log
        self.stable = False

        # Initialize logger
        if self.to_log:
            self.send_msg(self.style("INITIALIZE COLDLOAD HEATER", "header"))

        # Check if a valid heater range is passed as an argument
        if(range.lower() != "low" and range.lower() != "medium" and range.lower() != "high"):
            self.send_msg(f"{str(range)} is an invalid heater range. Will default to heater range 'low'.")
            self.range = "low" # Reset to default
        else: 
            self.range = range.lower()

        # Check if a valid wait time is passed as an argument
        try:
            self.wait_time = float(wait_time)
        except:
            self.send_msg(str(wait_time)+ " is an invalid wait time. Will default to 1.20 s.")            
            self.wait_time = 1.20

        # Set heater range
        self.send_msg(f"Setting heater range to {str(self.range)}")
        self.ls336.set_heater_range(range=range)
        self.send_msg(f"Heater range is: {str(self.ls336.get_heater_attribute(attribute='heater_range').session['data'])}") 

        # Set heater to open loop mode
        self.send_msg("Setting heater to open loop mode")
        self.ls336.set_mode(mode='open loop')
        self.send_msg(f"Current mode: {str(self.ls336.get_heater_attribute(attribute='mode').session['data'])}")

    def set_power(self, percent=0):
        # Set heater power
        try:
            self.send_msg(f"Set coldload manual out to {percent} percent")
            self.ls336.set_manual_out(percent=percent)
            self.percent = percent
            self.send_msg(f"Current manual out: {str(self.ls336.get_heater_attribute(attribute='manual_out').session['data'])}")
        except:
            self.send_msg("Error setting manual out. Defaulting to 0%.")
            self.ls336.set_manual_out(percent=0)
            self.percent = 0
            self.send_msg(f"Current manual out: {str(self.ls336.get_heater_attribute(attribute='manual_out').session['data'])}")
        # Wait for heater temperature to stabilize
        time.sleep(self.wait_time)
    
        # Check temperature
        self.get_temperature()
    
    def get_power(self):
        return self.percent

    def get_stability(self):
        return self.stable

    def set_stability(self, stable):
        self.stable = stable
        return self.stable


    # Get Temperature
    def get_temperature(self):
        output = self.ls336.acq.status()
        time.sleep(1)
        temps = output.session.get('data').get('ls336_fields').get('data').get('Center_T')
        return temps

    def set_range(self, range):
        if(range.lower() != "low" and range.lower() != "medium" and range.lower() != "high"):
            self.send_message("Invalid heater range.")
            self.range = "low"
        else: 
            self.range = range.lower()
        self.ls336.set_heater_range(range=self.range)
    
    def get_range(self):
        return self.range

    def send_msg(self, msg):
        try:
            if self.to_log:
                logger = logging.getLogger(__name__)
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


