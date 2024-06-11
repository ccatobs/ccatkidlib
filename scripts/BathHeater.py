from ocs.ocs_client import OCSClient
from tqdm import tqdm
import time
import sys
import logging 

class BathHeater:

    def __init__(self, ocs_client = 'LSA1940', range = 10e-3, output = 0.1e-3, wait_time = 2, log = False):
        # Initialize OCSClient
        self.output = output
        self.ls372 = OCSClient(ocs_client)
        self.ls372.acq.start()

        # Set stability of bath heater
        self.stable = False
        
        self.to_log = log
        # Initialize logger
        if log:
            self.send_msg(self.style("INITIALIZE BATH HEATER", "header"))

        # # Check if a valid heater range is passed as an argument
        # if(range.lower() != "low" and range.lower() != "medium" and range.lower() != "high"):
        #     self.send_msg(f"{str(range)} is an invalid heater range. Will default to heater range 'low'.")
        #     self.range = "low" # Reset to default
        # else: 
        #     self.range = range.lower()

        # Check if a valid wait time is passed as an argument
        self.range = range

        try:
            self.wait_time = float(wait_time)
        except:
            self.send_msg(str(wait_time)+ " is an invalid wait time. Will default to 1.20 s.")            
            self.wait_time = 2

        # Set heater to open loop mode
        self.send_msg("Setting heater to open loop mode")
        self.ls372.set_output_mode(heater="sample", mode='Open Loop')
        #self.send_msg(f"Current mode: {self.ls372.get_heater_attribute(attribute='output_mode').session['data']}")

        # Set heater range
        self.send_msg(f"Setting heater range to {self.range} A")
        self.ls372.set_heater_range(heater='sample',range=range, wait=wait_time)
        #self.send_msg(f"Heater range is: {self.ls372.get_heater_attribute(attribute='heater_range').session['data']}") 

    def set_power(self, output=0):
        # Set heater power
        try:
            self.send_msg(f"Set sample manual out to {output} W")
            self.ls372.set_heater_output(heater='sample', output=output)
            self.output = output
            #self.send_msg(f"Current manual out: {self.ls372.get_heater_attribute(attribute='heater_output').session['data']}")
        except:
            self.send_msg("Error setting manual out. Defaulting to 0 W.")
            self.ls372.set_heater_output(heater='sample', output=0)
            self.output = 0
            #self.send_msg(f"Current manual out: {self.ls372.get_heater_attribute(attribute='heater_output').session['data']}")
        # Wait for heater temperature to stabilize
        time.sleep(self.wait_time)
    
        # Check temperature
        self.get_temperature()
    
    def get_power(self):
        return self.output

    def get_stability(self):
        return self.stable

    def set_stability(self, stable):
        self.stable = stable
        return self.stable

    # Get Temperature
    def get_temperature(self):
        out = self.ls372.acq.status()
        time.sleep(1)
        temps = out.session['data']['fields']['Channel_06']['T']
        return temps

    def set_range(self, range):
        try:
            self.ls372.set_heater_range(heater='sample',range=range, wait=self.wait_time)
            self.range = range
        except:
            pass
    
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


