import threading
import time
import io
import requests
import pandas as pd

from live_capture import LiveCapture


class MonitorWorker(threading.Thread):

    def __init__(self, interface, api_url, interval=5):
        super().__init__(daemon=True)

        self.interface = interface
        self.api_url = api_url
        self.interval = interval

        self.capture = LiveCapture(interface)

        self.running = False
        self.latest_df = None

    def stop(self):
        self.running = False

    def run(self):

        self.running = True

        print("===== Monitor Started =====", id(self))

        while self.running:

            try:

                print("Capturing packets...")

                pcap = self.capture.capture(self.interval)

                print("PCAP:", pcap)

                if pcap is None:
                    print("No packets captured.")
                    continue

                print("Sending PCAP to API...")

                with open(pcap, "rb") as f:

                    response = requests.post(
                        self.api_url,
                        files={
                            "file": (
                                "live_capture.pcap",
                                f,
                                "application/octet-stream"
                            )
                        },
                        timeout=300
                    )

                print("Response:", response.status_code)

                if response.status_code == 200:

                    self.latest_df = pd.read_csv(
                        io.StringIO(response.text)
                    )
                    print("Worker:", id(self))
                    print(self.latest_df.columns.tolist())
                    print(self.latest_df.head())
                    print("Worker id:", id(self))
                    print("latest_df inside worker:")
                    print(self.latest_df)

                    print("Flows received:", len(self.latest_df))

                else:
                    print(response.text)

            except Exception as e:

                print("ERROR:", e)

            time.sleep(1)