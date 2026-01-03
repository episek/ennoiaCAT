import pywifi
from pywifi import const
import time
import pandas as pd
import streamlit as st

def freq_to_channel(freq):
    if freq == 2484:
        return 14
    elif 2412 <= freq <= 2472:
        return (freq - 2407) // 5  # 2.4 GHz
    elif 5180 <= freq <= 5825:
        return (freq - 5000) // 5  # 5 GHz
    elif 5955 <= freq <= 7115:
        return (freq - 5950) // 5 + 1  # 6 GHz
    else:
        return None

def classify_band(freq):
    if 2400 <= freq <= 2500:
        return "2.4 GHz"
    elif 5000 <= freq <= 5900:
        return "5 GHz"
    elif 5925 <= freq <= 7125:
        return "6 GHz"
    else:
        return "Unknown"

def scan_wifi():
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]  # Assume first adapter
    iface.scan()
    time.sleep(3)  # Allow time for scan to complete

    results = iface.scan_results()
    networks = []

    for network in results:
        ssid = network.ssid
        bssid = network.bssid
        signal = network.signal  # Approximate dBm
        freq = network.freq
        channel = freq_to_channel(freq)
        band = classify_band(freq)

        networks.append({
            "SSID": ssid,
            "BSSID": bssid,
            "Signal (dBm)": signal,
            "Frequency (MHz)": freq,
            "Channel": channel,
            "Band": band
        })

    df = pd.DataFrame(networks)
    return df.sort_values(by="Signal (dBm)", ascending=False)

def main():
    st.title("📡 WiFi Scanner for Windows (2.4/5/6 GHz)")

    if st.button("🔍 Scan WiFi Networks"):
        df = scan_wifi()
        if df.empty:
            st.warning("No WiFi networks found. Try again or check adapter.")
        else:
            st.success("Scan complete!")
            st.dataframe(df)

            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", data=csv, file_name="wifi_scan.csv", mime="text/csv")

if __name__ == "__main__":
    main()
