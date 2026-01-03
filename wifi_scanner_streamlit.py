import pywifi
from pywifi import const
import time
import pandas as pd
import streamlit as st

def freq_to_channel(freq):
    try:
        freq = int(freq/1e3)
        if freq == 2484:
            return 14
        elif 2412 <= freq <= 2472:
            return (freq - 2407) // 5
        elif 5180 <= freq <= 5825:
            return (freq - 5000) // 5
        elif 5955 <= freq <= 7115:
            return (freq - 5950) // 5 + 1
    except:
        pass
    return None

def classify_band(freq):
    if freq/1e3 is None:
        return "Unknown"
    elif 2400 <= freq <= 2500:
        return "2.4 GHz"
    elif 5000 <= freq <= 5900:
        return "5 GHz"
    elif 5925 <= freq <= 7125:
        return "6 GHz"
    else:
        return "Unknown"

def infer_bandwidth(channel, radio_type):
    try:
        ch = int(channel)
    except:
        return "Unknown"

    rt = radio_type.lower()
    if ch <= 14:
        return "20/40 MHz"
    elif 36 <= ch <= 48 or 149 <= ch <= 161:
        return "20/40/80 MHz"
    elif 52 <= ch <= 144:
        return "20/40/80 MHz"
    elif 1 <= ch <= 233:
        return "20/40/80/160 MHz"
    else:
        return "Unknown"

def scan_wifi():
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]
    iface.scan()
    time.sleep(3)
    results = iface.scan_results()

    networks = []

    for net in results:
        ssid = net.ssid or "<Hidden>"
        bssid = net.bssid
        signal = net.signal
        freq = net.freq

        channel = freq_to_channel(freq)
        band = classify_band(freq)

        # Estimate radio type based on band
        if band == "2.4 GHz":
            radio = "802.11b/g/n"
        elif band == "5 GHz":
            radio = "802.11a/n/ac"
        elif band == "6 GHz":
            radio = "802.11ax"
        else:
            radio = "Unknown"

        bw = infer_bandwidth(channel, radio)

        networks.append({
            "SSID": ssid,
            "BSSID": bssid,
            "Signal (dBm)": signal,
            "Frequency (KHz)": freq,
            "Channel": channel,
            "Band": band,
            "Radio Type (Estimated)": radio,
            "Bandwidth (Estimated)": bw
        })

    df = pd.DataFrame(networks).sort_values(by="Signal (dBm)", ascending=False)
    return df

def main():
    st.set_page_config(page_title="WiFi Scanner", layout="wide")
    st.title("📡 Full Windows WiFi Scanner (pywifi) with Bandwidth & Band Info")

    if st.button("🔍 Scan for WiFi Networks"):
        df = scan_wifi()
        if df.empty:
            st.warning("No networks found. Try again or check adapter.")
        else:
            st.success(f"Found {len(df)} networks.")
            st.dataframe(df)
            st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="wifi_scan.csv", mime="text/csv")

if __name__ == "__main__":
    main()
