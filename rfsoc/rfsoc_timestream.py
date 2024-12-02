import numpy as np
import socket
import time

class Streamer:
    '''
    Class for capturing UDP packets of timestreams taken with a radio frequency system on a chip (RFSoC).
    '''
    def __init__(self, udp_ip, udp_port):
        '''
        Initialize Streamer object by creating a socket and binding it to the timestream UDP IP & port.
        
        Parameters:
            udp_ip (str): UDP IP address
            udp_port (int): UPD port
        '''
        # Save global timestamp attribute
        self.timestamp = time.time()

        # Create socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Bind socket to timestream UDP IP & port
        self.sock.bind((udp_ip, udp_port))

    def capture_packets(self, N, buffer_size = 9000):
        '''
        Capture N UDP packets from the timestream. 

        Parameters:
            N (int): Number of packages to capture
            buffer_size (int): Maximum buffer size of each packet
        Returns:
            packs: Packet data (as a bytearray) of each captured packet 
            ips: Source IP of each captured packet
            ports: Source port of each captured packet
        '''
        packs, addrs = [], []

        self.timestamp = time.time()
        # Capture N packets from socket
        for i in range(int(N)):
            # Get single packet data and source address
            pack, addr = self.sock.recvfrom(buffer_size)
            packs.append(bytearray(pack))
            addrs.append(list(addr))

        addrs = np.array(addrs)
        ips = addrs[:, 0]
        ports = addrs[:, 1]
        return packs, ips, ports

    def parse_packets(self, packets):
        '''
        Parse the bytearray data contents of UPD packets.

        Parameters:
            packets: Array of packet bytearray data
        Returns:
            data: Array of timestream I and Q (main data)
            aux: Array of auxiliary data: packet info, channel count, packet count, and packet PTP timestamp
        '''
        data, aux = [], []

        time_diff = None
        # Parse all packets
        for pack in packets:
            # Parse (extra) packet info and number of active tones
            info, chan_count = np.frombuffer(pack, dtype = ">H", count = 2, offset = 8192).astype("float")

            # Parse packet count
            pack_count = np.frombuffer(pack, dtype = ">i", count = 1, offset = 8196).astype("int")[0]

            # Parse timestamp
            tstamp = self.get_utc(np.frombuffer(pack, dtype = ">I", offset = 8200, count = 3), -37)

            # Get the difference between PTP timestamp and actual time
            if time_diff is None:
                time_diff = tstamp - self.timestamp

            # Parse timestream data
            dat = np.frombuffer(pack, dtype = "<i", count = int(2*chan_count)).astype("float")
            data.append(dat)
            aux.append([info, int(chan_count), pack_count, tstamp - time_diff])
        return data, aux

    def take_timestream(self, N, offset = 30):
        '''
        Take a timestream of N packets.

        Parameters:
            N (int): Number of packets to capture
        
        Returns:
            I: Timestream I data
            Q: Timstream Q data
            aux: Auxiliary data: packet info, channel count, packet count, and packet PTP timestamp
            ips: Packet source IP addresses
            ports: Packet sourc ports
            offset: Number of packets to drop from beginning of timestream
        '''

        # Capture N packets
        packs, ips, ports = self.capture_packets(N)

        # Parse packet info
        data, aux = self.parse_packets(packs)

        return data[offset:], aux[offset:], ips[offset:], ports[offset:]

    ####################
    # Helper Functions #
    ####################

    def get_utc(self, d,offset=0):
        '''
        Convert PTP timestamp stored as three 4 byte integers into UTC.

        Parameters:
            d: Array of three 4 byte integers storing PTP timestamp
            offset: Offset between TAI and UTC
        Returns:
            UTC: PTP timestamp in UTC
        '''
        # offset from TAI to UTC is currently -37 seconds
        return int((d[-3] << 18) | (d[-2] >> 14)) + int((((d[-2] & 0x00003FFF) << 16) | (d[-1] >> 16))) * 1e-9 + offset