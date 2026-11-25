"""UDP packet capture for |RFSoC| |KID| timestreams.

Defines the ``Streamer`` class, which binds a UDP socket and captures/parses the
time-ordered |I|/|Q| packets emitted by an |RFSoC| board.

.. codeauthor:: Darshan Patel <dp649@cornell.edu>
"""

import numpy as np
import socket
import time


class Streamer:
    """Captures UDP packets of |KID| timestreams from an |RFSoC| board.

    Attributes:
        timestamp (float): UNIX time at which the ``Streamer`` was created.
        sock (socket.socket): Bound UDP socket packets are read from.
    """

    def __init__(self, udp_ip: str, udp_port: int, timeout: float = 1) -> None:
        """Bind a UDP socket for timestream capture.

        Args:
            udp_ip (str): UDP IP address to bind to.
            udp_port (int): UDP port to bind to.
            timeout (float, optional): Socket receive timeout in seconds. Defaults to 1.
        """
        # Save global timestamp attribute
        self.timestamp = time.time()

        # Create socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Allow multiple clients to bind to socket
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind socket to timestream UDP IP & port
        self.sock.bind((udp_ip, udp_port))

        self.sock.settimeout(timeout)

    def capture_packets(
        self,
        t_sec: int,
        buffer_size: int = 9000,
        q=None,
    ) -> tuple[list[bytearray], np.ndarray, np.ndarray] | None:
        """Capture UDP packets from the timestream.

        Flushes any stale packets, then either streams packets into ``q`` indefinitely
        or collects a fixed number of packets and returns them.

        Args:
            t_sec (int): Number of packets to capture when ``q`` is ``None``.
            buffer_size (int, optional): Maximum bytes read per packet. Defaults to 9000.
            q (optional): If given, a queue each captured packet is pushed onto; the
                call then streams indefinitely and does not return. Defaults to ``None``.
        Returns:
            tuple[list[bytearray], np.ndarray, np.ndarray] | None: When ``q`` is
                ``None``, the captured packet bytearrays, their source IPs, and their
                source ports. Returns nothing when streaming into ``q``.
        """
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

            # while curr_time - timestamp < t_sec:
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

    def parse_packets(
        self,
        packets: list[bytearray],
        timestamp: float | None = None,
        time_diff: float | None = None,
    ) -> tuple[list[np.ndarray], list[list], float]:
        """Parse the bytearray contents of captured UDP packets.

        Args:
            packets (list[bytearray]): Raw packet bytearrays to parse.
            timestamp (float | None, optional): Reference UNIX time used to compute the
                PTP time offset. Defaults to the ``Streamer`` creation time.
            time_diff (float | None, optional): Precomputed offset between the packet PTP
                timestamp and wall-clock time. Defaults to computing it from the first packet.
        Returns:
            tuple[list[np.ndarray], list[list], float]: The per-packet timestream |I|/|Q|
                data, the per-packet auxiliary data (packet info, channel count, packet
                count, PTP timestamp in ns), and the PTP-to-wall-clock time offset.
        """

        data, aux = [], []

        if timestamp is None:
            timestamp = self.timestamp
        # Parse all packets
        for pack in packets:
            # Parse (extra) packet info and number of active tones
            info, chan_count = np.frombuffer(
                pack, dtype=">H", count=2, offset=8192
            ).astype("int32")

            # Parse packet count
            pack_count = np.frombuffer(pack, dtype=">i", count=1, offset=8196).astype(
                "int32"
            )[0]

            # Parse timestamp
            tstamp = self.get_utc(
                np.frombuffer(pack, dtype=">I", offset=8200, count=3), -37
            )

            # Get the difference between PTP timestamp and actual time
            if time_diff is None:
                time_diff = tstamp - timestamp

            # Parse timestream data
            dat = np.frombuffer(pack, dtype="<i", count=int(2 * chan_count)).astype(
                "int32"
            )
            data.append(dat)
            aux.append([info, chan_count, pack_count, np.int64(1e9 * tstamp)])
        return data, aux, time_diff

    def take_timestream(
        self,
        t_sec: int,
        offset: int = 10,
    ) -> tuple[list, list, np.ndarray, np.ndarray]:
        """Capture and parse a timestream, dropping leading warm-up packets.

        Args:
            t_sec (int): Number of packets to capture.
            offset (int, optional): Number of leading packets to drop. Defaults to 10.
        Returns:
            tuple[list, list, np.ndarray, np.ndarray]: The timestream data, auxiliary
                data, source IPs, and source ports, each with the first ``offset``
                packets removed.
        """

        # Capture N packets
        packs, ips, ports = self.capture_packets(t_sec)

        # Parse packet info
        data, aux, _ = self.parse_packets(packs)

        return data[offset:], aux[offset:], ips[offset:], ports[offset:]

    ####################
    # Helper Functions #
    ####################

    def get_utc(self, d: np.ndarray, offset: int = 0) -> float:
        """Convert a PTP timestamp stored as three 4-byte integers into UTC seconds.

        Args:
            d (np.ndarray): Three 4-byte integers holding the PTP timestamp.
            offset (int, optional): TAI-to-UTC offset in seconds. Defaults to 0.
        Returns:
            float: The PTP timestamp in UTC seconds.
        """
        # offset from TAI to UTC is currently -37 seconds
        return (
            int((d[-3] << 18) | (d[-2] >> 14))
            + int((((d[-2] & 0x00003FFF) << 16) | (d[-1] >> 16))) * 1e-9
            + offset
        )
