
import ennoia_client_lic as lic 
import argparse

parser = argparse.ArgumentParser(description="Mavenir License Client")
parser.add_argument(
    "--action",
    choices=["activate", "verify"],
    default="verify",
    help="Action to perform (default: verify)"
)
parser.add_argument("--key", help="Mavenir License key for activation")
args = parser.parse_args()


if args.action == "activate":
    if not args.key:
        print("❗ Please provide a license key with --key")
    else:
        success = lic.request_license(args.key)
elif args.action == "verify":
    success = lic.verify_license_file()
else:
    success = lic.verify_license_file()
  
if not success:
    print("❌ License verification failed. Please check your license key or contact support.")
    exit()
    
import json
import ast
import streamlit as st
from tinySA_config import TinySAHelper
from map_api import MapAPI
from types import SimpleNamespace
import pandas as pd
from timer import Timer, timed, fmt_seconds
import pywifi
from pywifi import const
import time
st.cache_data.clear()
st.cache_resource.clear()


    # Define option descriptions for reference
options_descriptions = {
    "plot": "plot rectangular",
    "scan": "scan by script",
    "start": "start frequency",
    "stop": "stop frequency",
    "center": "center frequency",
    "span": "span",
    "points": "scan points",
    "port": "specify port number",
    "device": "define device node",
    "verbose": "enable verbose output",
    "capture": "capture current display to file",
    "command": "send raw command",
    "save": "write output to CSV file"
}

st.set_page_config(page_title="Mavenir Systems", page_icon="🤖")
st.sidebar.image('mavenir_logo.png')
st.title("Mavenir Systems")
st.markdown(
    """ 
    Chat and Test with Mavenir Connect Platform ©. All rights reserved. 
    """
)

if not success:
    st.error("Mavenir License verification failed. Please check your license key or contact support.")
    st.stop()
else:
    st.success("Mavenir License verified successfully.")

# --- App logic starts here ---
selected_options = TinySAHelper.select_checkboxes()
st.success(f"You selected: {', '.join(selected_options) if selected_options else 'nothing'}")


# --- Caching the model and tokenizer ---

if "SLM" in selected_options:
    @st.cache_resource
    def load_model_and_tokenizer():
        return TinySAHelper.load_lora_model()

    st.write("\n⏳ Working in OFFLINE mode. Loading local model... (might take a minute)")
    tokenizer, peft_model, device = load_model_and_tokenizer()
    st.write(f"Device set to use {device}")
    map_api = MapAPI(peft_model, tokenizer)
else:
    st.write("\n⏳ Working in ONLINE mode.")  
    client, ai_model = TinySAHelper.load_OpenAI_model()
    map_api = MapAPI() 

helper = TinySAHelper()
system_prompt = helper.get_system_prompt()
few_shot_examples = helper.get_few_shot_examples()



@st.cache_data
def get_default_options():
    return map_api.get_defaults_opts()

def_dict = get_default_options()

few_shot_examples2 = map_api.get_few_shot_examples()

# --- Get and cache the TinySA port ---
if "tinySA_port" not in st.session_state:
    st.session_state.tinySA_port = helper.getport()

if "SLM" in selected_options:
    st.write(f"\n✅ Local SLM model {peft_model.config.name_or_path} loaded & device found! Let's get to work.\n")
else:
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = ai_model
    st.write(f"\n✅ Online LLM model {ai_model} loaded & device! Let's get to work.\n")
# Initialize TinySA device

st.write("Hi. I am Mavenir, your AI assistant. How can I help you today?")

st.write("Detected 5G NR Fujitsu Radio Unit!")

# Initialize session state

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask Mavenir:")


# ----------------- RU REPORT JSON (same as before, inside ru_report["..."]) -----------------
ru_report = {
    "metadata": {
        "vendor": "Fujitsu",
        "model": "FTRU-5G-RU4468",
        "site": "TX-DAL-Rooftop-Sector-A",
        "sector": "A",
        "timestamp": "2025-11-25T20:00:00-06:00",
        "generated_by": "Mavenir Agent v1.4"
    },
    "hardware": {
        "serial_number": "FTS-RU-A07C9912",
        "hw_version": "HW Rev. D",
        "sw_version": "R4.1.3-O-RU-FJ",
        "rf_chains": "4T4R",
        "band": "n78",
        "bandwidth_mhz": 100,
        "max_output_power_dbm": 40
    },
    "operational_status": {
        "admin_state": "unlocked",
        "operational_state": "enabled",
        "sync_state": "ptp_locked",
        "sfp_status": "warning_tx_temp_high",
        "front_haul": "active_10G",
        "timing_source": ["ptp", "synce"],
        "last_reboot_days": 12
    },
    "ptp_timing": {
        "ptp_lock": True,
        "grandmaster_id": "00:1D:C1:AA:22:11",
        "two_step": True,
        "mean_path_delay_ns": 420,
        "offset_from_master_ns": 35,
        "synce_state": "stable",
        "csr_status": "connected"
    },
    "rf_performance": {
        "downlink": {
            "dl_power_dbm": {
                "tx0": 39.2,
                "tx1": 38.7,
                "tx2": 39.1,
                "tx3": 39.2
            },
            "dl_power_imbalance_db": 1.5,
            "pa_health": "nominal",
            "evm_db": -36.4,
            "aclr_db": {
                "low_side": 48.1,
                "high_side": 47.9
            }
        },
        "uplink": {
            "ul_rssi_dbm": -83.5,
            "ul_noise_floor_dbm": -101.2,
            "rssi_imbalance_db": 1.0,
            "ul_sinr_db": 18.7
        }
    },
    "fronthaul": {
        "transport_type": "ecpri",
        "link_speed": "10G",
        "packet_drop_rate_pct": 0.003,
        "jitter_us": 4.2,
        "latency_rtt_us": 170,
        "oran_planes": {
            "c_plane": "stable",
            "u_plane": {
                "status": "active",
                "prbs": 273,
                "symbols_per_slot": 14
            },
            "s_plane": "connected",
            "m_plane": "healthy"
        }
    },
    "environmental": {
        "internal_temp_c": 47,
        "pa_temp_c": 52,
        "sfp_temp_c": 61,
        "voltage_v": 54.7,
        "current_a": 2.4,
        "fan_speed": "auto"
    },
    "alarms": {
        "active_alarms": [
            {
                "severity": "warning",
                "code": "SFP-TEMP-HI",
                "description": "SFP transmit laser temperature high"
            }
        ],
        "historical_alarms": [
            {
                "severity": "info",
                "code": "PTP-RELOCK",
                "count": 2
            },
            {
                "severity": "info",
                "code": "FH-JITTER-SPIKE",
                "count": 1
            }
        ]
    },
    "diagnostics": {
        "pcap_analysis": {
            "symbols_per_slot": 14,
            "prbs": 273,
            "bfp9_exponents_valid": True,
            "section_id_discontinuities": False,
            "ecpri_payload_mismatch": False
        },
        "issues_detected": [
            "slight_dl_power_imbalance",
            "sfp_temp_trending_high"
        ]
    },
    "recommendations": [
        "Monitor SFP temperature; replace if exceeding 65-67C.",
        "Inspect fiber connectors for cleanliness to reduce optical return loss.",
        "Verify grounding to prevent PTP wander.",
        "Track DL imbalance for PA gain drift.",
        "Run deeper analysis if PTP relocks recur."
    ]
}

# ----------------- HELPERS -----------------

def flatten_dict(d, parent_key="", sep="."):
    """
    Flatten nested dicts, e.g. {"a": {"b": 1}} -> {"a.b": 1}
    Good for showing as a 2-column table.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def dict_to_table(d, title=None):
    flat = flatten_dict(d)
    df = pd.DataFrame(list(flat.items()), columns=["Parameter", "Value"])
    if title:
        st.markdown(f"**{title}**")
    st.table(df)

# ----------------- SECTION RENDERERS (TABLE VERSION) -----------------

def show_metadata(report):
    dict_to_table(report["metadata"])

def show_hardware(report):
    dict_to_table(report["hardware"])

def show_operational_status(report):
    dict_to_table(report["operational_status"])

def show_ptp(report):
    dict_to_table(report["ptp_timing"])

def show_rf(report):
    st.markdown("### Downlink")
    dict_to_table(report["rf_performance"]["downlink"])
    st.markdown("### Uplink")
    dict_to_table(report["rf_performance"]["uplink"])

def show_fronthaul(report):
    dict_to_table(report["fronthaul"])

def show_environmental(report):
    dict_to_table(report["environmental"])

def show_alarms(report):
    st.markdown("### Active Alarms")
    active = report["alarms"]["active_alarms"]
    if active:
        df_active = pd.DataFrame(active)
        st.table(df_active)
    else:
        st.write("No active alarms.")

    st.markdown("### Historical Alarms")
    hist = report["alarms"]["historical_alarms"]
    if hist:
        df_hist = pd.DataFrame(hist)
        st.table(df_hist)
    else:
        st.write("No historical alarms.")

def show_rea(report):
    st.markdown("### PCAP Analysis")
    dict_to_table(report["diagnostics"]["pcap_analysis"])
    st.markdown("### Issues Detected")
    issues = report["diagnostics"]["issues_detected"]
    if issues:
        df_issues = pd.DataFrame(
            [{"Issue #": i + 1, "Description": issue} for i, issue in enumerate(issues)]
        )
        st.table(df_issues)
    else:
        st.write("No issues detected.")

def show_recommendations(report):
    recs = report["recommendations"]
    df_recs = pd.DataFrame(
        [{"#": i + 1, "Recommendation": r} for i, r in enumerate(recs)]
    )
    st.table(df_recs)


SECTION_FUNCS = {
    "Overview": show_metadata,
    "Hardware": show_hardware,
    "Operational": show_operational_status,
    "PTP / Timing": show_ptp,
    "RF Performance": show_rf,
    "Fronthaul": show_fronthaul,
    "Environment": show_environmental,
    "Alarms": show_alarms,
    "Diagnostics": show_rea,
    "Recommendations": show_recommendations,
}

# ----------------- UI LOGIC -----------------

if "current_section" not in st.session_state:
    st.session_state.current_section = "Overview"

st.title("RU Health & Status Report")

meta = ru_report["metadata"]
st.caption(
    f"**Vendor:** {meta['vendor']} | **Model:** {meta['model']} | "
    f"**Site:** {meta['site']} | **Sector:** {meta['sector']} | "
    f"**Generated:** {meta['timestamp']}"
)

st.markdown("---")

# ------------- TWO-COLUMN BUTTON LAYOUT -------------
section_names = list(SECTION_FUNCS.keys())

# split sections into two columns
left_sections = section_names[0:len(section_names)//2]
right_sections = section_names[len(section_names)//2:]

col_left, col_right = st.columns(2)

with col_left:
    for name in left_sections:
        if st.button(name, key=f"btn_{name}"):
            st.session_state.current_section = name

with col_right:
    for name in right_sections:
        if st.button(name, key=f"btn_{name}"):
            st.session_state.current_section = name

st.markdown("---")

current = st.session_state.current_section
st.header(current)
SECTION_FUNCS[current](ru_report)
























if prompt:
    t = Timer()
    t.start()

    # Store the user message in session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Construct the full prompt with context (system + user message)
        user_input = st.session_state.messages[-1]["content"]
        
        chat1 = [{"role": "system", "content": system_prompt}] + few_shot_examples + [{"role": "user", "content": user_input}]
        if "SLM" in selected_options:
            response = map_api.generate_response(chat1)
        else:
            openAImessage = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=chat1,
                temperature=0,
                max_tokens=200,
                frequency_penalty=1,
                stream=False
            )
            response = openAImessage.choices[0].message.content
        # Display the streamed response from the assistant
        st.markdown(response)
        
        # Save the assistant's response in session state
        st.session_state.messages.append({"role": "assistant", "content": response})


    system_prompt2 = map_api.get_system_prompt(def_dict,user_input)
    chat2 = [{"role": "system", "content": system_prompt2}] + few_shot_examples2 + [{"role": "user", "content": user_input}]
    if "SLM" in selected_options:
        api_str = map_api.generate_response(chat2)
    else:
        openAImessage = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=chat2,
                temperature=0,
                max_tokens=200,
                frequency_penalty=1,
                stream=False
            )
        api_str = openAImessage.choices[0].message.content
    #st.markdown(api_str)
    # Parse response safely into a dictionary
    def_dict["save"] = True
    print(f"\nSave output response:\n{def_dict}")
    api_dict = def_dict
    try:
        parsed = json.loads(api_str)
        if isinstance(parsed, dict):
            api_dict = parsed
            api_dict["save"] = True
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(api_str)
            if isinstance(parsed, dict):
                api_dict = parsed
                api_dict["save"] = True
        except Exception:
            print("Warning: Failed to parse response as a valid dictionary. Using default options.")

    print(f"\nParsed API options:\n{api_dict}")

    # Ensure it's a dict before using SimpleNamespace
    if isinstance(api_dict, dict):
        opt = SimpleNamespace(**api_dict)
        print(f"opt = {opt}")
        gcf = helper.configure_tinySA(opt)
        st.pyplot(gcf)
    else:
        st.error("API response is not a valid dictionary. Setting default options.")
 
    try:
        result = helper.read_signal_strength('max_signal_strengths.csv')
        if not result:
            st.error("Could not read signal strength data.")

        sstr, freq = result
        freq_mhz = [x / 1e6 for x in freq]
        print(f"\nSignal strengths: {sstr}")
        print(f"\nFrequencies: {freq_mhz}")
        
        operator_table = helper.get_operator_frequencies()
        if not operator_table:
            st.error("Operator table could not be loaded.")

        frequency_report_out = helper.analyze_signal_peaks(sstr, freq_mhz, operator_table)
        print(f"\nFrequency report: {frequency_report_out}")
        if not frequency_report_out:
            st.write("No strong trained frequency band seen.")

    except Exception as e:
        st.error(f"Failed to process request: {str(e)}")
    
    st.subheader("🗼 List of Available Cellular Networks")
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(frequency_report_out)

    # Display as a table in Streamlit
    st.dataframe(df)  # Interactive table
    
        
    
    


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
        try:
            freq = int(freq/1e3)
            if 2400 <= freq <= 2500:
                return "2.4 GHz"
            elif 5000 <= freq <= 5900:
                return "5 GHz"
            elif 5925 <= freq <= 7125:
                return "6 GHz"
            else:
                return "Unknown"
        except:
            pass
        return None

    def is_dfs_channel(channel):
        try:
            ch = int(channel)
        except:
            return False

        # Known DFS channel ranges for 5 GHz
        if 52 <= ch <= 64 or 100 <= ch <= 144:
            return True
        return False

    def infer_bandwidth(channel, radio_type):
        try:
            ch = int(channel)
        except:
            return "Unknown"

        rt = radio_type.lower()
        if ch <= 14:
            return "20/40 MHz"
        elif 36 <= ch <= 144 or 149 <= ch <= 165:
            if "ac" in rt or "ax" in rt:
                return "20/40/80/160 MHz"
            else:
                return "20/40 MHz"
        elif 1 <= ch <= 233:
            return "20/40/80/160 MHz" if "ax" in rt else "20 MHz"
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
                #"BSSID": bssid,
                "Signal (dBm)": signal,
                "Frequency (MHz)": freq,
                "Channel": channel,
                "Band": band,
                "Radio Type (Estimated)": radio,
                "Bandwidth (Estimated)": bw,
                "DFS Channel": "Yes" if is_dfs_channel(channel) else "No"
            })

        df = pd.DataFrame(networks).sort_values(by="Signal (dBm)", ascending=False)
        return df

    if any(x >= 2.39e9 for x in freq):

    # with st.expander("📡 WiFi Scanner"):
        # if st.button("🔍 Scan for WiFi Networks"):
        df = scan_wifi()
        st.subheader("📶 List of Available WiFi Networks")
        st.caption("Below are the scanned WiFi networks, including signal strength, frequency, and estimated bandwidth.")
        if df.empty:
            st.warning("No networks found.")
        else:
            st.success(f"Found {len(df)} networks.")
            st.dataframe(df)
            st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="wifi_scan.csv", mime="text/csv")

    t.stop()
    #print("elapsed:", fmt_seconds(t.elapsed()))
    st.write(f"elapsed: {fmt_seconds(t.elapsed())}")
    t.reset()  # reset to zero








