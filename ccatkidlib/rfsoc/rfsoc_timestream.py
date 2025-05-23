import numpy as np
import socket
import time

class Streamer:
    '''
    Class for capturing UDP packets of timestreams taken with a radio frequency system on a chip (RFSoC).
    '''
    def __init__(self, udp_ip, udp_port, timeout = 1):
        '''
        Initialize Streamer object by creating udp_ip, udp_port, and timestamp attributes.
        
        Parameters:
            udp_ip (str): UDP IP address
            udp_port (int): UPD port
        '''
        # Save global timestamp attribute
        self.timestamp = time.time()

        # Create socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Allow multiple clients to bind to socket
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind socket to timestream UDP IP & port
        self.sock.bind((udp_ip, udp_port))

        self.sock.settimeout(timeout)

    def capture_packets(self, t_sec, buffer_size = 9000, q = None):
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
        sock = self.sock

        # Flush buffer of any old packets
        for i in range(50):
            sock.recvfrom(buffer_size)

        timestamp = time.time()

        # Capture t_seconds seconds of packets from socket
        curr_time = time.time()
        if q is not None:
            while True:
                q.put_nowait(sock.recvfrom(buffer_size))
        else:
            packet_data = []
            for i in range(t_sec):
                packet_data.append(sock.recvfrom(buffer_size))
            
            #while curr_time - timestamp < t_sec:
            #    # Get single packet data and source address
            #    packet_data.append(sock.recvfrom(buffer_size))
            #    curr_time = time.time()
            packs, addrs = zip(*packet_data)
            packs = [bytearray(pack) for pack in packs]
            addrs = [list(addr) for addr in addrs]
            
            addrs = np.array(addrs)
            ips = addrs[:, 0]
            ports = addrs[:, 1]

            return packs, ips, ports

    def parse_packets(self, packets, timestamp = None, time_diff = None):
        '''
        Parse the bytearray data contents of UPD packets.

        Parameters:
            packets: Array of packet bytearray data
        Returns:
            data: Array of timestream I and Q (main data)
            aux: Array of auxiliary data: packet info, channel count, packet count, and packet PTP timestamp
        '''
        

        data, aux = [], []

        if timestamp is None: timestamp = self.timestamp
        # Parse all packets
        for pack in packets:
            # Parse (extra) packet info and number of active tones
            info, chan_count = np.frombuffer(pack, dtype = ">H", count = 2, offset = 8192).astype("int32")

            # Parse packet count
            pack_count = np.frombuffer(pack, dtype = ">i", count = 1, offset = 8196).astype("int32")[0]

            # Parse timestamp
            tstamp = self.get_utc(np.frombuffer(pack, dtype = ">I", offset = 8200, count = 3), -37)

            # Get the difference between PTP timestamp and actual time
            if time_diff is None:
                time_diff = tstamp - timestamp

            # Parse timestream data
            dat = np.frombuffer(pack, dtype = "<i", count = int(2*chan_count)).astype("int32")
            data.append(dat)
            aux.append([info, chan_count, pack_count, np.int64(100000*(tstamp - time_diff))])
        return data, aux, time_diff

    def take_timestream(self, t_sec, offset = 10):
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
        packs, ips, ports = self.capture_packets(t_sec)

        # Parse packet info
        data, aux, _ = self.parse_packets(packs)

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