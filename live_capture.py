from scapy.all import AsyncSniffer, wrpcap, get_if_list
import tempfile
import os
import threading


def get_interfaces():
    return get_if_list()


class LiveCapture:

    def __init__(self, interface):
        self.interface = interface

    def capture(self, duration=5):

        temp_dir = tempfile.mkdtemp()

        pcap_path = os.path.join(
            temp_dir,
            "live_capture.pcap"
        )

        sniffer = AsyncSniffer(
            iface=self.interface,
            store=True
        )

        sniffer.start()

        threading.Event().wait(duration)

        packets = sniffer.stop()

        if not packets:
            return None

        wrpcap(pcap_path, packets)

        return pcap_path