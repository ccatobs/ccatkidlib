import time
from ocs.ocs_client import OCSClient
import numpy as np

coldload_agent = OCSClient('LSA291F')
logname = "auto.log"
stop_time = time.time() + 8000

while time.time()<stop_time:
    time.sleep(8)
    cl_temps = []
    for i in range(20):
        status,msg,session = coldload_agent.acq.status()
        time.sleep(0.1)
        coldload_temp = session['data']['fields']['Channel_2']['T']
        cl_temps.append(coldload_temp)

    cl_temps = np.array(cl_temps)
    coldload_temp = np.mean(cl_temps)
    timestamp = time.time()

    with open(logname, "a+") as file:
        file.write(f"{np.round(coldload_temp, 3)},\t{timestamp}\n")

