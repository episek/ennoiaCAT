import subprocess
import re
import pandas as pd
import streamlit as st

def channel_to_freq(ch):
    try:
        ch = int(ch)
        if 1 <= ch <= 13:
            return 2407 + ch * 5
        elif ch == 14:
            return 2484
        elif 36 <= ch <= 165:
            return 5000 + ch * 5
        elif 1 <= ch <= 233:  # 6 GHz band (Wi-Fi 6E)
            return 5950 + ch * 5
    except:
        return None

def classify_band(freq):
    if freq is None:
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
        return "20/40 MHz" if "n" in rt else "20 MHz"
    elif 36 <= ch <= 48 or 149 <= ch <= 161:
        return "20/40/80 MHz" if "ac" in rt or "ax" in rt else "20 MHz"
    elif 52 <= ch <= 144:
        return "20/40/80 MHz" if "ac" in rt or "ax" in rt else "20 MHz"
    elif 1 <= ch <= 233:
        return "20/40/80/160 MHz" if "ax" in rt else "20 MHz"
    else:
        return "Unknown"

def parse_netsh_scan():
    result = subprocess.check_output("netsh wlan show networks mode=bssid", shell=True).decode(errors="ignore")
    lines = result.splitlines()

    networks = []
    current_ssid = ""
    bssid_count = 0
    net = {}

    for line in lines:
        line = line.strip()

        if line.startswith("SSID"):
            if "BSSID" not in line:
                current_ssid = line.split(":", 1)[1].strip()
                bssid_count = 0
        elif line.startswith("BSSID"):
            if bssid_count > 0 and "BSSID" in net:
                networks.append(net)
                net = {}
            net["SSID"] = current_ssid if current_ssid else "<Hidden>"
            net["BSSID"] = line.split(":", 1)[1].strip()
            bssid_count += 1
        elif line.startswith("Signal"):
            match = re.search(r"(\d+)\s*%", line)
            if match:
                percent = int(match.group(1))
                net["Signal (%)"] = percent
                net["Signal (dBm)"] = round((percent / 2) - 100)
        elif line.startswith("Channel"):
            ch = line.split(":", 1)[1].strip()
            net["Channel"] = ch
            net["Frequency (MHz)"] = channel_to_freq(ch)
            net["Band"] = classify_band(net["Frequency (MHz)"])
        elif line.startswith("Radio type"):
            net["Radio Type"] = line.split(":", 1)[1].strip()
        elif line.startswith("Authentication"):
            net["Authentication"] = line.split(":", 1)[1].strip()
        elif line.startswith("Encryption"):
            net["Encryption"] = line.split(":", 1)[1].strip()

    if net and "BSSID" in net:
        networks.append(net)

    # Infer bandwidth for each entry
    for net in networks:
        net["Bandwidth"] = infer_bandwidth(net.get("Channel", 0), net.get("Radio Type", ""))

    return pd.DataFrame(networks)

def main():
    st.title("📡 Windows WiFi Scanner (netsh) with Bandwidth Inference")

    if st.button("🔍 Scan WiFi Networks"):
        df = parse_netsh_scan()
        if df.empty:
            st.warning("No networks found.")
        else:
            st.success("Scan complete.")
            st.dataframe(df)
            st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="wifi_netsh_scan.csv", mime="text/csv")

if __name__ == "__main__":
    main()
