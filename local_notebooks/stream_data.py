import socket
import numpy as np
import sys
import os
sys.path.append('/usr/lib/python3/')

class rfsoc_udp_connection:

    def __init__(self):
        self.UDP_IP = "192.168.3.40"
        self.UDP_PORT = 4096	
        os.system("sudo fuser -k " + str(self.UDP_PORT) +  "/udp")
        
        self.sock = socket.socket(socket.AF_INET, # Internet
            socket.SOCK_DGRAM) # UDP
        self.sock.bind((self.UDP_IP, self.UDP_PORT)) 
        self.packetrate = 512e6/(2.0**20)


    def capturePacket(self, byteshift=-1):
        """
        Q - Remove byte rolling and conversion to separate function?
        """

        data = self.sock.recv(9000) # buffer size is 1024 bytes
        data = bytearray(data)
        data = np.roll(data,byteshift)
        return np.frombuffer(data, dtype="<i").astype("float")

    def getNpackets_2(self, N,byteshift=-1):
    
        return np.array([self.capturePacket(byteshift) for p in range(N)])
   
    def stream_data (self, t_sec):
        N_packet = int(self.packetrate*t_sec)
        return self.getNpackets_2(N_packet)
