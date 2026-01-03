import pywifi
from pywifi import const
import time

def scan_wifi():
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]

    iface.scan()
    print("Scanning...")
    time.sleep(3)  # Wait for scan to complete

    results = iface.scan_results()
    for network in results:
        print(f"SSID: {network.ssid}, Signal: {network.signal} dBm, Channel: {network.freq} MHz")

if __name__ == "__main__":
    scan_wifi()
