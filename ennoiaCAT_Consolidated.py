import ennoia_client_lic as lic
import argparse
import streamlit as st

# -----------------------------------------------------------------------------
# LICENSE ARGUMENTS
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Ennoia License Client")
parser.add_argument("--action", choices=["activate", "verify"], default="verify")
parser.add_argument("--key", help="License key for activation")
args, _unknown = parser.parse_known_args()

# try:
    # if args.action == "activate":
        # if not args.key:
            # print("Provide a license key with --key")
            # success = False
        # else:
            # success = lic.request_license(args.key)
    # else:
        # success = lic.verify_license_file()
# except Exception as e:
    # print(f"License check error: {e}")
    # success = False
success = True
# -----------------------------------------------------------------------------
# COMMON IMPORTS
# -----------------------------------------------------------------------------
import json
import ast
import time
import re
import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional imports
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

try:
    from streamlit_js_eval import streamlit_js_eval
    JS_EVAL_AVAILABLE = True
except ImportError:
    JS_EVAL_AVAILABLE = False

# Clear caches
st.cache_data.clear()
st.cache_resource.clear()

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Ennoia Equipment Controller", page_icon="🗼")

# -----------------------------------------------------------------------------
# LANGUAGE SELECTION (must be early for t() function)
# -----------------------------------------------------------------------------
language_map = {
    "🌐 Select language": None,
    "English": "en",
    "Français": "fr",
    "Español": "es",
    "Deutsch": "de",
    "עברית": "iw",      # Hebrew uses 'iw' in Google Translate API
    "हिन्दी": "hi",
    "العربية": "ar",
    "Русский": "ru",
    "中文": "zh-CN",    # Use zh-CN instead of zh-cn
    "日本語": "ja",
    "한국어": "ko"
}

selected_language = st.sidebar.selectbox("🌐 Language", list(language_map.keys()), index=0)
lang = language_map[selected_language]

# -----------------------------------------------------------------------------
# TRANSLATION HELPER FUNCTION
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def translate_text(text, target_lang):
    """Translate text to target language with caching."""
    if not text or not target_lang or target_lang == "en":
        return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        return text

def t(text):
    """Shorthand translation function for UI text."""
    if not TRANSLATOR_AVAILABLE or not lang or lang == "en":
        return text
    return translate_text(text, lang)

# -----------------------------------------------------------------------------
# EQUIPMENT SELECTION
# -----------------------------------------------------------------------------
st.sidebar.title("Equipment Selection")
equipment_type = st.sidebar.selectbox(
    "Select Equipment",
    [
        "Viavi OneAdvisor",
        "Keysight FieldFox",
        "Aukua XGA4250",
        "Cisco NCS540",
        "Rohde & Schwarz NRQ6",
        "tinySA",
        "ORAN PCAP Analyzer"
    ]
)

# Display appropriate logos (with file existence checks)
import os as _os
if _os.path.exists('ennoia.jpg'):
    st.sidebar.image('ennoia.jpg', width=200)
if equipment_type == "Viavi OneAdvisor":
    if _os.path.exists('viavi.png'):
        st.sidebar.image('viavi.png', width=200)
elif equipment_type == "Rohde & Schwarz NRQ6":
    if _os.path.exists('RS_logo.png'):
        st.sidebar.image('RS_logo.png', width=200)
elif equipment_type == "Aukua XGA4250":
    if _os.path.exists('aukua rgb high.jpg'):
        st.sidebar.image('aukua rgb high.jpg', width=200)
elif equipment_type == "Cisco NCS540":
    if _os.path.exists('cisco_logo.png'):
        st.sidebar.image('cisco_logo.png', width=200)
elif equipment_type == "ORAN PCAP Analyzer":
    if _os.path.exists('oran_logo.jpeg'):
        st.sidebar.image('oran_logo.jpeg', width=200)

st.title(f"🗼 Ennoia – {equipment_type} Agentic AI Control & Analysis")
st.caption(t("Natural-language controlled RF Spectrum Analyzer (OpenAI / SLM toggle)"))

# -----------------------------------------------------------------------------
# LICENSE CHECK
# -----------------------------------------------------------------------------
if not success:
    st.error(t("License verification failed."))
    st.stop()
else:
    st.success(t("Ennoia License verified successfully."))

# -----------------------------------------------------------------------------
# IMPORT EQUIPMENT-SPECIFIC MODULES
# -----------------------------------------------------------------------------
if equipment_type == "Viavi OneAdvisor":
    from map_api_vi import MapAPI
    from tinySA_config import TinySAHelper
    from timer import Timer, fmt_seconds
    from ennoia_viavi.system_api import OneAdvisorSystemAPI
    from ennoia_viavi.radio_api import OneAdvisorRadioAPI
    import pywifi
    helper_class = TinySAHelper

elif equipment_type == "Keysight FieldFox":
    from map_api import MapAPI
    from openai import OpenAI
    import pyvisa
    from openai_api_key_verifier import verify_api_key, check_model_access
    import os
    helper_class = None  # Keysight uses direct pyvisa

elif equipment_type == "Aukua XGA4250":
    from map_api import MapAPI
    from AK_config import AKHelper
    import pyvisa
    helper_class = AKHelper

elif equipment_type == "Cisco NCS540":
    from map_api import MapAPI
    from CS_config import CSHelper
    from ncs540_serial import NCS540Serial, list_serial_ports, list_eth_interfaces, use_selected_interfaces
    import pyvisa
    helper_class = CSHelper

elif equipment_type == "Rohde & Schwarz NRQ6":
    from map_api import MapAPI
    from RS_config import RSHelper
    from RsInstrument import RsInstrument
    from RSfunc import com_prep, com_check, meas_prep, measure, load_iq_csv, plot_time_domain, plot_fft, close
    import pyvisa
    helper_class = RSHelper

elif equipment_type == "tinySA":
    from map_api import MapAPI
    from tinySA_config import TinySAHelper
    import tinySA
    helper_class = TinySAHelper

elif equipment_type == "ORAN PCAP Analyzer":
    from map_api import MapAPI
    from ORAN_config import ORANHelper
    from ORAN_config import blind_interference_detection, generate_dmrs_type1_standard
    import requests
    helper_class = ORANHelper

# Display welcome message
st.markdown(t("Chat and Test with **Ennoia Technologies Connect Platform** ©. All rights reserved."))

# -----------------------------------------------------------------------------
# EQUIPMENT-SPECIFIC SETTINGS & CONNECTION
# -----------------------------------------------------------------------------

if equipment_type == "Viavi OneAdvisor":
    # Viavi-specific settings
    st.sidebar.subheader("🔌 Viavi OneAdvisor Connection")
    viavi_host = st.sidebar.text_input("OneAdvisor IP", value="192.168.1.100")

    st.sidebar.subheader("🔁 Sweep Settings")
    continuous_mode = st.sidebar.checkbox("Continuous Sweep", value=False)
    num_sweeps = st.sidebar.number_input("Sweeps (if continuous)", min_value=1, max_value=200, value=5, step=1)
    sweep_delay = st.sidebar.number_input("Delay between sweeps (s)", min_value=0.05, max_value=5.0, value=0.3, step=0.05)

    st.sidebar.subheader("📊 Display Mode")
    display_mode = st.sidebar.selectbox("Spectrum Display", ["Single sweep", "Multi-sweep average", "Waterfall"])

    WARMUP_SWEEPS = 2
    helper = TinySAHelper()

    def get_oneadvisor_identity(host):
        try:
            sys_api = OneAdvisorSystemAPI(host)
            sys_api.open()
            idn = sys_api.query("*IDN?")
            sys_api.close()
            if not idn:
                return None
            parts = [p.strip() for p in idn.split(",")]
            identity = {
                "vendor": parts[0] if len(parts) > 0 else "",
                "model": parts[1] if len(parts) > 1 else "",
                "serial": parts[2] if len(parts) > 2 else "",
                "firmware": parts[3] if len(parts) > 3 else "",
            }
            return identity
        except Exception as e:
            return {"error": str(e)}

elif equipment_type == "Keysight FieldFox":
    # Keysight-specific settings
    FIELD_FOX_IP = st.sidebar.text_input("FieldFox IP", value="192.168.1.100")

    # Connect to FieldFox
    try:
        rm = pyvisa.ResourceManager()
        inst = rm.open_resource(f"TCPIP0::{FIELD_FOX_IP}::inst0::INSTR", open_timeout=10000)
        inst.read_termination = '\n'
        inst.write_termination = '\n'
        inst.timeout = 5000
        idn = inst.query("*IDN?")
        st.sidebar.success(f"Connected to: {idn.strip()}")
        inst.write(":INSTrument:SELect 'SA'")
    except pyvisa.errors.VisaIOError as e:
        st.sidebar.warning(f"⏳ Waiting for FieldFox to connect... ({e})")
    except Exception as e:
        st.sidebar.warning(f"⏳ Connection error: {e}")

elif equipment_type == "Aukua XGA4250":
    # Aukua-specific settings
    ip = st.sidebar.text_input("Aukua IP", value="192.168.1.101")
    uribase = f"http://{ip}/api/v1/"
    helper = AKHelper()

elif equipment_type == "Cisco NCS540":
    # Cisco-specific settings
    ip = st.sidebar.text_input("Cisco IP", value="192.168.1.36")

    # Serial port selection
    serial_ports = list_serial_ports()
    if serial_ports:
        selected_serial = st.selectbox("Select Serial Port", serial_ports, index=0)
    else:
        st.error(t("❌ No USB serial ports found"))
        selected_serial = None

    if "conn" not in st.session_state:
        st.session_state.conn = None
    if "connected_port" not in st.session_state:
        st.session_state.connected_port = None

    username = st.text_input("Username", value="", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")

    if st.button("Connect Serial"):
        try:
            st.session_state.conn = NCS540Serial(port=selected_serial, username=username, password=password)
            st.session_state.connected_port = selected_serial
            st.session_state.conn.start_keep_alive(interval=60)
            st.success(t("✅ NCS540 Connected via") + f" {selected_serial} " + t("as") + f" {username}")
        except Exception as e:
            st.warning(t("⏳ Waiting for NCS540 to connect..."))

    if not st.session_state.conn:
        st.warning(t("⏳ Waiting for NCS540 to connect. Please press **Connect Serial** to continue."))
        st.stop()

    helper = CSHelper()

elif equipment_type == "Rohde & Schwarz NRQ6":
    # R&S-specific settings
    resource = st.sidebar.text_input("VISA Resource", value="TCPIP::nrq6-101528::hislip0")

    try:
        nrq = RsInstrument(resource, id_query=True, reset=True)
        st.sidebar.success(f"Connected to: {nrq.query('*IDN?')}")
        st.sidebar.write(f"Options: {nrq.query('*OPT?')}")
    except Exception as e:
        st.sidebar.warning(f"⏳ Waiting for NRQ6 to connect...")

    helper = RSHelper()

elif equipment_type == "tinySA":
    # tinySA-specific settings
    helper = TinySAHelper()
    if "tinySA_port" not in st.session_state:
        st.session_state.tinySA_port = helper.getport()

elif equipment_type == "ORAN PCAP Analyzer":
    # ORAN PCAP Analyzer settings
    st.sidebar.subheader("📡 ORAN Analysis Parameters")

    # Flask server URL configuration
    flask_host = st.sidebar.text_input("Flask Server Host", value="127.0.0.1")
    flask_port = st.sidebar.number_input("Flask Server Port", value=5002, min_value=1024, max_value=65535)

    st.sidebar.subheader("📊 5G NR Parameters")
    oran_n_id = st.sidebar.number_input("N_ID (Cell ID)", value=100, min_value=0, max_value=1007)
    oran_nscid = st.sidebar.selectbox("nSCID", [0, 1], index=0)
    oran_bandwidth = st.sidebar.selectbox("Bandwidth (MHz)", [5, 10, 15, 20, 50, 100], index=5)
    oran_scs = st.sidebar.selectbox("SCS (kHz)", [15, 30], index=1)
    oran_layers = st.sidebar.selectbox("Number of Layers", [1, 2, 4], index=2)
    oran_link = st.sidebar.selectbox("Link Direction", ["Uplink", "Downlink"], index=0)

    st.sidebar.subheader("📍 Frame Position")
    oran_subframe = st.sidebar.selectbox("Subframe (0-9)", list(range(10)), index=2)
    oran_slot = st.sidebar.selectbox("Slot (0-1)", [0, 1], index=0)

    st.sidebar.subheader("🔍 Detection Mode")
    detection_mode = st.sidebar.selectbox(
        "Interference Detection Method",
        ["DMRS-Based (Standard)", "AI-Based Blind Detection", "Both"],
        index=0
    )

    helper = ORANHelper()

    # Check Flask server status and auto-launch if needed
    st.sidebar.subheader("🔌 Flask Server Status")

    def _start_flask_server():
        """Launch the Flask server subprocess."""
        flask_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packet_oran_analysis_flask.py")
        if not os.path.exists(flask_script):
            st.sidebar.error("packet_oran_analysis_flask.py not found")
            return False
        st.session_state.flask_process = subprocess.Popen(
            [sys.executable, flask_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(3)
        return helper.check_flask_server()

    def _stop_flask_server():
        """Terminate the Flask server subprocess (or kill by port if not tracked)."""
        stopped = False
        if "flask_process" in st.session_state and st.session_state.flask_process is not None:
            try:
                st.session_state.flask_process.terminate()
                st.session_state.flask_process.wait(timeout=5)
            except Exception:
                pass
            st.session_state.flask_process = None
            stopped = True
        # Fallback: kill any process listening on the Flask port
        if not stopped or helper.check_flask_server():
            try:
                result = subprocess.run(
                    ["netstat", "-ano"], capture_output=True, text=True
                )
                for line in result.stdout.splitlines():
                    if f":{flask_port}" in line and "LISTENING" in line:
                        pid = line.strip().split()[-1]
                        subprocess.run(["taskkill", "/F", "/PID", pid],
                                       capture_output=True)
            except Exception:
                pass
            time.sleep(1)

    flask_running = helper.check_flask_server()

    if flask_running:
        st.sidebar.success("Flask server is running")
        if st.sidebar.button("Restart Flask Server"):
            _stop_flask_server()
            if _start_flask_server():
                st.sidebar.success("Flask server restarted successfully")
                st.session_state.confidence_reset_done = False
            else:
                st.sidebar.warning("Flask server is restarting, please wait...")
            st.rerun()
    else:
        # Auto-launch Flask server
        if "flask_process" not in st.session_state or st.session_state.flask_process is None or st.session_state.flask_process.poll() is not None:
            st.sidebar.info("Starting Flask server...")
            if _start_flask_server():
                st.sidebar.success("Flask server started automatically")
            else:
                st.sidebar.warning("Flask server is starting up, please wait...")
                st.rerun()
        else:
            st.sidebar.warning("Flask server is starting up, please wait...")
            time.sleep(2)
            st.rerun()

    # Reset AI confidence on first run of the session
    if "confidence_reset_done" not in st.session_state:
        if helper.check_flask_server():
            helper.reset_confidence()
        st.session_state.confidence_reset_done = True

    # File upload section
    st.sidebar.subheader("📁 PCAP File")
    uploaded_pcap = st.sidebar.file_uploader("Upload PCAP file", type=["pcap", "pcapng"])
    pcap_filepath = st.sidebar.text_input("Or enter PCAP file path", value="")

    if uploaded_pcap is not None:
        # Validate file size (max 500MB)
        max_file_size = 500 * 1024 * 1024  # 500 MB
        file_size = uploaded_pcap.size
        if file_size > max_file_size:
            st.sidebar.error(f"File too large: {file_size / 1024 / 1024:.1f} MB. Maximum allowed: 500 MB")
        else:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pcap') as tmp_file:
                tmp_file.write(uploaded_pcap.read())
                pcap_filepath = tmp_file.name
                st.sidebar.success(f"File uploaded: {uploaded_pcap.name} ({file_size / 1024 / 1024:.1f} MB)")

# -----------------------------------------------------------------------------
# AI MODEL SELECTION (COMMON)
# -----------------------------------------------------------------------------
if helper_class:
    selected_options = helper_class.select_checkboxes() if hasattr(helper_class, 'select_checkboxes') else []
    st.success(t("You selected:") + f" {', '.join(selected_options) if selected_options else t('nothing')}")

    if "SLM" in selected_options:
        @st.cache_resource
        def load_peft_model():
            return helper_class.load_lora_model()

        st.write(t("\n⏳ Working in OFFLINE mode. Loading local LoRA model..."))
        tokenizer, peft_model, device = load_peft_model()

        if peft_model is None:
            st.error(t("❌ Failed to load SLM model (GPU out of memory or model not found)"))
            st.warning(t("Falling back to template-based report generation"))
            map_api = None
        else:
            st.write(t("\n✅ Local SLM model") + f" {peft_model.config.name_or_path} " + t("loaded"))
            st.write(t("Device is set to use") + f" {device}! " + t("Let's get to work.\n"))

            if equipment_type == "Viavi OneAdvisor":
                map_api = MapAPI(backend="slm", injected_model=peft_model, injected_tokenizer=tokenizer, max_new_tokens=512, temperature=0.2)
            else:
                map_api = MapAPI(peft_model, tokenizer)
    else:
        @st.cache_resource
        def load_openai():
            client, ai_model = helper_class.load_OpenAI_model()
            st.session_state["openai_model"] = ai_model
            if equipment_type == "Viavi OneAdvisor":
                return MapAPI(backend="openai", openai_model=ai_model, max_new_tokens=512, temperature=0.2)
            else:
                return MapAPI()

        st.write(t("\n⏳ Working in ONLINE mode."))
        map_api = load_openai()
        st.write(t("\n✅ Online LLM model") + f" {st.session_state['openai_model']} " + t("loaded. Let's get to work.\n"))

# -----------------------------------------------------------------------------
# EQUIPMENT IDENTITY DISPLAY
# -----------------------------------------------------------------------------
if equipment_type == "Viavi OneAdvisor":
    st.sidebar.subheader("📟 OneAdvisor Identity")
    if viavi_host:
        ident = get_oneadvisor_identity(viavi_host)
        if ident and "error" not in ident:
            st.sidebar.success("Connected to OneAdvisor")
            st.sidebar.write(f"**Vendor:** {ident['vendor']}")
            st.sidebar.write(f"**Model:** {ident['model']}")
            st.sidebar.write(f"**Serial Number:** {ident['serial']}")
            st.sidebar.write(f"**Firmware:** {ident['firmware']}")
        else:
            st.sidebar.error(f"Could not read identity: {ident.get('error')}")

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "waterfall_data" not in st.session_state:
    st.session_state["waterfall_data"] = None
if "waterfall_freqs" not in st.session_state:
    st.session_state["waterfall_freqs"] = None
if "last_trace" not in st.session_state:
    st.session_state["last_trace"] = None
if "last_freqs" not in st.session_state:
    st.session_state["last_freqs"] = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (from Viavi baseline)
# -----------------------------------------------------------------------------
def parse_freq(val, default=None):
    """Converts values like 3500, '3.5 GHz', '3500 MHz', etc. → Hz."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            return default
        return parse_freq(val[0], default=default)

    s = str(val).strip().lower()
    m = re.match(r'^([0-9]*\.?[0-9]+)\s*(ghz|mhz|khz|hz)?$', s)
    if m:
        num = float(m.group(1))
        unit = m.group(2) or "hz"
        if unit == "ghz":
            return num * 1e9
        if unit == "mhz":
            return num * 1e6
        if unit == "khz":
            return num * 1e3
        return num

    try:
        return float(s)
    except Exception:
        return default

def extract_start_stop(text):
    """Extract start/stop frequencies from natural language."""
    if not text:
        return None, None

    txt = text.lower()
    txt = txt.replace("–", "-").replace("—", "-")

    def conv(num, unit):
        num = float(num)
        if unit == "g":
            return num * 1e9
        if unit == "m":
            return num * 1e6
        if unit == "k":
            return num * 1e3
        return num

    start_match = re.search(r"start(?:\s*freq(?:uency)?)?(?:\s*(?:to|at|from))?\s*([\d\.]+)\s*(g|m|k)?hz", txt)
    stop_match = re.search(r"stop(?:\s*freq(?:uency)?)?(?:\s*(?:to|at|from))?\s*([\d\.]+)\s*(g|m|k)?hz", txt)

    if start_match and stop_match:
        s = conv(start_match.group(1), start_match.group(2))
        e = conv(stop_match.group(1), stop_match.group(2))
        return (min(s, e), max(s, e))

    range_match = re.search(r"([\d\.]+)\s*(g|m|k)?hz\s*(?:to|-|up to|through|thru|until|and)\s*([\d\.]+)\s*(g|m|k)?hz", txt)
    if range_match:
        n1, u1, n2, u2 = (range_match.group(1), range_match.group(2), range_match.group(3), range_match.group(4))
        s = conv(n1, u1)
        e = conv(n2, u2)
        return (min(s, e), max(s, e))

    return None, None

def extract_numbers(input_string):
    """Extract numbers from string (for Keysight)."""
    pattern = r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'
    numbers = re.findall(pattern, input_string)
    return numbers

def auto_rbw(span_hz):
    """Basic rule-of-thumb RBW."""
    if span_hz <= 10e6:
        return 10e3
    if span_hz <= 100e6:
        return 100e3
    return 1e6

def find_peaks(y, max_peaks=5, min_dist=5):
    """Simple peak finder (no SciPy)."""
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            peaks.append((i, y[i]))
    peaks.sort(key=lambda p: p[1], reverse=True)
    selected = []
    for idx, val in peaks:
        if all(abs(idx - s[0]) >= min_dist for s in selected):
            selected.append((idx, val))
        if len(selected) >= max_peaks:
            break
    return selected

def bandpower_linear(freqs_hz, trace_dbm, f1_hz, f2_hz):
    """Approximate band power (dBm) by integrating over [f1,f2]."""
    freqs_hz = np.array(freqs_hz)
    trace_dbm = np.array(trace_dbm)

    if f2_hz <= f1_hz:
        return None

    mask = (freqs_hz >= f1_hz) & (freqs_hz <= f2_hz)
    if not np.any(mask):
        return None

    # Convert dBm to mW
    p_mw = 10 ** (trace_dbm[mask] / 10.0)
    f_band = freqs_hz[mask]

    # Approximate integral via trapezoidal rule
    power_mw = np.trapz(p_mw, f_band) / (f2_hz - f1_hz)
    if power_mw <= 0:
        return None
    return 10 * np.log10(power_mw)

def detect_5gnr_like_carriers(freqs_hz, trace_dbm, min_bw_mhz=5.0, rel_thresh_db=6.0):
    """
    Very simple 5G NR-like carrier detector:
    - Find strong peaks.
    - Estimate -3 dB bandwidth around each.
    - Mark carriers with BW >= min_bw_mhz in 600 MHz – 6 GHz.
    """
    freqs_hz = np.array(freqs_hz)
    trace_dbm = np.array(trace_dbm)
    peaks = find_peaks(trace_dbm, max_peaks=10, min_dist=10)
    carriers = []

    for idx, val in peaks:
        f0 = freqs_hz[idx]
        if f0 < 600e6 or f0 > 6e9:
            continue

        # -3 dB bandwidth estimation
        half_power = val - 3.0
        left = idx
        while left > 0 and trace_dbm[left] > half_power:
            left -= 1
        right = idx
        n = len(trace_dbm)
        while right < n - 1 and trace_dbm[right] > half_power:
            right += 1

        bw_hz = freqs_hz[right] - freqs_hz[left]
        bw_mhz = bw_hz / 1e6
        if bw_mhz < min_bw_mhz:
            continue

        carriers.append(
            {
                "Center (MHz)": f0 / 1e6,
                "Approx BW (MHz)": bw_mhz,
                "Peak Power (dBm)": float(val),
            }
        )
    return carriers

def wifi_band_from_freq(freq_hz):
    """Classify WiFi band from RF frequency."""
    f_mhz = freq_hz
    if 2400 <= f_mhz <= 2500:
        return "2.4 GHz"
    if 5000 <= f_mhz <= 5900:
        return "5 GHz"
    if 5925 <= f_mhz <= 7125:
        return "6 GHz"
    return None

def detect_wifi_like_carriers(freqs_hz, trace_dbm, rel_thresh_db=10.0):
    """
    Very simple WiFi-like detector:
    - Look for peaks in WiFi bands.
    - Report strongest candidates.
    """
    freqs_hz = np.array(freqs_hz)
    trace_dbm = np.array(trace_dbm)
    peaks = find_peaks(trace_dbm, max_peaks=20, min_dist=5)
    if not peaks:
        return []

    max_val = max(v for _, v in peaks)
    carriers = []
    for idx, val in peaks:
        if max_val - val > rel_thresh_db:
            continue
        f0 = freqs_hz[idx]
        band = wifi_band_from_freq(f0)
        if band is None:
            continue
        carriers.append(
            {
                "Center (MHz)": f0,
                "Band": band,
                "Peak Power (dBm)": float(val),
            }
        )
    return carriers

def classify_span_wifi_bands(freqs_hz):
    if not freqs_hz:
        return []
    f_min = min(freqs_hz)
    f_max = max(freqs_hz)
    bands = []
    # overlap checks in MHz
    if f_max >= 2400 and f_min <= 2505:
        bands.append("2.4 GHz")
    if f_max >= 5150 and f_min <= 5850:
        bands.append("5 GHz")
    if f_max >= 5925 and f_min <= 7125:
        bands.append("6 GHz")
    return bands

# -----------------------------------------------------------------------------
# MAIN CHAT HANDLER
# -----------------------------------------------------------------------------
prompt = st.chat_input(f"Ask Ennoia about {equipment_type}:")

if prompt:
    # Start timer for Viavi OneAdvisor
    if equipment_type == "Viavi OneAdvisor":
        from timer import Timer, fmt_seconds
        timer = Timer()
        timer.start()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    if equipment_type == "Viavi OneAdvisor":
        # Default frequencies
        DEFAULT_CENTER = 600e6  # 600 MHz
        DEFAULT_SPAN = 600e6    # 600 MHz
        DEFAULT_START = 300e6   # 300 MHz
        DEFAULT_STOP = 900e6    # 900 MHz

        chat1 = [
            {"role": "system", "content": f"You are Ennoia AI, an assistant for RF spectrum analysis using {equipment_type}. Default frequency range is 300 MHz to 900 MHz (600 MHz center, 600 MHz span). When asked about configuration, always mention these defaults."},
            {"role": "user", "content": prompt},
        ]
        if map_api is not None:
            response = map_api.generate_response(chat1)
        else:
            response = f"I'm the {equipment_type} assistant. The AI model is not currently loaded. Please check your configuration."

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # API JSON response for instrument config
        api_chat = [
            {"role": "system", "content": "Output ONLY a JSON object with optional keys: start, stop, center, span, rbw, vbw, ref_level. Use units like '600 MHz', '3.5 GHz' if helpful."},
            {"role": "user", "content": prompt},
        ]
        if map_api is not None:
            api_raw = map_api.generate_response(api_chat)
        else:
            api_raw = "{}"

        api_dict = {}
        try:
            api_dict = json.loads(api_raw)
        except Exception:
            try:
                api_dict = ast.literal_eval(api_raw)
            except Exception:
                api_dict = {}

        opt = SimpleNamespace(**api_dict)

        # Natural-language override for start/stop if missing
        nls, nle = extract_start_stop(prompt)
        if getattr(opt, "start", None) is None and nls is not None:
            opt.start = nls
        if getattr(opt, "stop", None) is None and nle is not None:
            opt.stop = nle

        # Parse frequencies with defaults
        start_hz = parse_freq(getattr(opt, "start", None), default=DEFAULT_START)
        stop_hz = parse_freq(getattr(opt, "stop", None), default=DEFAULT_STOP)
        center_hz = parse_freq(getattr(opt, "center", None))
        span_hz = parse_freq(getattr(opt, "span", None))

        # Calculate center/span if start/stop provided
        if center_hz is None and start_hz is not None and stop_hz is not None:
            center_hz = 0.5 * (float(start_hz) + float(stop_hz))
        if span_hz is None and start_hz is not None and stop_hz is not None:
            span_hz = abs(float(stop_hz) - float(start_hz))

        # Use defaults if still None
        if center_hz is None:
            center_hz = DEFAULT_CENTER
        if span_hz is None:
            span_hz = DEFAULT_SPAN

        # Calculate RBW, VBW, and ref_level
        rbw = parse_freq(getattr(opt, "rbw", None)) or auto_rbw(span_hz)
        vbw = parse_freq(getattr(opt, "vbw", None)) or rbw
        try:
            ref_level = float(getattr(opt, "ref_level", 0))
        except Exception:
            ref_level = 0.0

        # Display Configuration
        st.subheader(t("🗼 Spectrum Analyzer Configuration"))
        st.success(t("**Configuration for**") + f" {equipment_type}:")
        st.write(f"- **{t('Start Frequency')}:** {start_hz/1e6:.1f} MHz")
        st.write(f"- **{t('Stop Frequency')}:** {stop_hz/1e6:.1f} MHz")
        st.write(f"- **{t('Center Frequency')}:** {center_hz/1e6:.1f} MHz")
        st.write(f"- **{t('Span')}:** {span_hz/1e6:.1f} MHz")
        st.write(f"- **RBW:** {rbw/1e3:.1f} kHz")
        st.write(f"- **VBW:** {vbw/1e3:.1f} kHz")
        st.write(f"- **{t('Reference Level')}:** {ref_level} dBm")

        # Execute Viavi spectrum acquisition
        freqs = []
        trace = []

        if not viavi_host:
            st.error(t("Enter Viavi IP in sidebar."))
        else:
            try:
                # Discover radio SCPI port
                sys_api = OneAdvisorSystemAPI(viavi_host)
                sys_api.open()
                radio_port = sys_api.get_radio_scpi_port()
                sys_api.close()

                ra = OneAdvisorRadioAPI(viavi_host, scpi_port=radio_port)
                ra.open()

                # Explicitly enter Spectrum Analyzer mode
                ra.set_spectrum_mode("spectrumTuned")
                time.sleep(0.2)

                def configure_spectrum():
                    ra.configure_spectrum(
                        center_hz=center_hz,
                        span_hz=span_hz,
                        rbw_auto=False,
                        rbw_hz=rbw,
                        vbw_auto=False,
                        vbw_hz=vbw,
                        ref_level_dbm=ref_level,
                        atten_mode="Auto",
                    )
                    time.sleep(0.2)

                def one_sweep():
                    tr = ra.get_spectrum_trace()
                    if not tr:
                        return [], []
                    start_axis, stop_axis, _ = ra.get_spectrum_xaxis()
                    fq = np.linspace(start_axis, stop_axis, len(tr)).tolist()
                    tr = list(tr)
                    return fq, tr

                configure_spectrum()

                sweeps = num_sweeps if continuous_mode or display_mode in ["Multi-sweep average", "Waterfall"] else 1

                all_traces = []
                placeholder = st.empty()

                for i in range(sweeps):
                    # discard 1 quick trace before using the next one
                    _freqs_dummy, _trace_dummy = one_sweep()
                    time.sleep(0.05)

                    freqs, trace = one_sweep()
                    if not freqs:
                        st.warning(t("No trace received from Viavi."))
                        break

                    all_traces.append(trace)

                    # Store last trace in session state for later analysis/export
                    st.session_state["last_trace"] = trace
                    st.session_state["last_freqs"] = freqs

                    # Update waterfall buffer
                    wf = st.session_state["waterfall_data"]
                    if wf is None or st.session_state["waterfall_freqs"] is None or \
                            len(st.session_state["waterfall_freqs"]) != len(freqs):
                        wf = np.array(trace, dtype=float)[np.newaxis, :]
                    else:
                        wf = np.vstack([wf, np.array(trace, dtype=float)])
                    st.session_state["waterfall_data"] = wf
                    st.session_state["waterfall_freqs"] = freqs

                    freq_mhz = [f for f in freqs]

                    # PLOT according to display mode (but update live as we sweep)
                    fig, ax = plt.subplots()

                    if display_mode == "Multi-sweep average" and len(all_traces) > 1:
                        avg_trace = np.mean(np.array(all_traces), axis=0)
                        ax.plot(freq_mhz, avg_trace, label="Avg")
                        ax.plot(freq_mhz, trace, alpha=0.3, label="Last")
                        ax.legend()
                    elif display_mode == "Waterfall" and st.session_state["waterfall_data"] is not None:
                        wf_mhz = [f for f in st.session_state["waterfall_freqs"]]
                        wf_arr = st.session_state["waterfall_data"]
                        extent = [min(wf_mhz), max(wf_mhz), 0, wf_arr.shape[0]]
                        im = ax.imshow(
                            wf_arr,
                            aspect="auto",
                            origin="lower",
                            extent=extent,
                        )
                        ax.set_ylabel("Sweep index")
                        plt.colorbar(im, ax=ax, label="Power (dBm)")
                    else:
                        # Single sweep
                        ax.plot(freq_mhz, trace)

                    ax.set_xlabel("Frequency (MHz)")
                    ax.set_ylabel("Power (dBm)")
                    ax.set_title(f"{center_hz/1e9:.3f} GHz / Span {span_hz/1e6:.1f} MHz (Sweep {i+1}/{sweeps})")
                    ax.grid(True)

                    # Add peak markers only for non-waterfall plots
                    if display_mode != "Waterfall":
                        peaks = find_peaks(trace, max_peaks=5, min_dist=5)
                        for idx, val in peaks:
                            f_mhz = freq_mhz[idx]
                            ax.plot(f_mhz, val, "o", color="red")
                            ax.text(
                                f_mhz,
                                val,
                                f"{f_mhz:.1f} MHz\n{val:.1f} dBm",
                                fontsize=8,
                                ha="left",
                            )

                    placeholder.pyplot(fig)

                    if (continuous_mode or display_mode in ["Multi-sweep average", "Waterfall"]) and i < sweeps - 1:
                        time.sleep(float(sweep_delay))

                ra.close()

            except Exception as e:
                st.error(t("Viavi error:") + f" {e}")
                freqs, trace = [], []

        # -----------------------------------------------------------------------------
        # CELLULAR PEAK TABLE + 5G NR DETECTION
        # -----------------------------------------------------------------------------
        st.subheader(t("📶 Cellular Peaks & 5G NR-like Carriers"))

        if freqs and trace:
            freq_mhz = [f for f in freqs]
            peaks = find_peaks(trace, max_peaks=10, min_dist=5)
            rows = []
            for idx, val in peaks:
                rows.append(
                    {"Frequency (MHz)": freq_mhz[idx], "Power (dBm)": val}
                )
            if rows:
                st.write(t("Strong Peaks:"))
                df_peaks = pd.DataFrame(rows)
                st.dataframe(df_peaks)
            else:
                st.info(t("No prominent peaks detected."))

            nr_carriers = detect_5gnr_like_carriers(freqs, trace)
            if nr_carriers:
                st.write(t("5G NR-like Carriers (rough RF-only heuristic):"))
                df_nr = pd.DataFrame(nr_carriers)
                st.dataframe(df_nr)
            else:
                st.info(t("Checking for 5G NR carriers."))

            # CSV exports
            col1, col2 = st.columns(2)
            with col1:
                csv_trace = pd.DataFrame(
                    {"Frequency_Hz": freqs, "Power_dBm": trace}
                ).to_csv(index=False)
                st.download_button(
                    "📥 Download Last Trace CSV",
                    data=csv_trace,
                    file_name="viavi_trace.csv",
                    mime="text/csv",
                )
            with col2:
                if peaks:
                    csv_peaks = df_peaks.to_csv(index=False)
                    st.download_button(
                        "📥 Download Peaks CSV",
                        data=csv_peaks,
                        file_name="viavi_peaks.csv",
                        mime="text/csv",
                    )
        else:
            st.info(t("No spectrum to analyze."))

        # -----------------------------------------------------------------------------
        # OPERATOR TABLE MAPPING (reuse TinySAHelper logic)
        # -----------------------------------------------------------------------------
        st.subheader(t("🗼 Operator Table (by detected cellular bands)"))

        if freqs and trace:
            # Reuse freq_mhz if it already exists; otherwise compute
            try:
                freq_mhz
            except NameError:
                freq_mhz = [f / 1e6 for f in freqs]

            # Load operator_table.json via TinySAHelper
            operator_table = helper.get_operator_frequencies()

            if not operator_table:
                st.warning(t("operator_table.json not found or invalid – cannot map to operators."))
            else:
                # Use TinySAHelper's peak/band logic on Viavi trace
                try:
                    op_report = helper.analyze_signal_peaks(
                        sstr=trace,
                        freq_mhz=freq_mhz,
                        operator_table=operator_table,
                        window_size=5,
                        peak_height=-75,   # adjust threshold as needed
                        peak_distance=10   # MHz grouping threshold
                    )
                except Exception as e:
                    op_report = []
                    st.error(t("Operator analysis error:") + f" {e}")

                if op_report:
                    df_ops = pd.DataFrame(op_report)
                    st.dataframe(df_ops)
                    # Optional: allow CSV download
                    csv_ops = df_ops.to_csv(index=False)
                    st.download_button(
                        "📥 Download Operator Table Matches CSV",
                        data=csv_ops,
                        file_name="operator_matches.csv",
                        mime="text/csv",
                    )
                else:
                    st.info(t("No strong operator bands detected in the current span."))
        else:
            st.info(t("No RF data available for operator mapping."))

        # -----------------------------------------------------------------------------
        # BANDPOWER MEASUREMENT (USER-SELECTED)
        # -----------------------------------------------------------------------------
        st.subheader(t("📐 Bandpower Measurement"))

        if freqs and trace:
            col_bp1, col_bp2, col_bp3 = st.columns(3)
            with col_bp1:
                bp_f1_mhz = st.number_input("Band start (MHz)", value=freqs[0] / 1e6)
            with col_bp2:
                bp_f2_mhz = st.number_input("Band stop (MHz)", value=freqs[-1] / 1e6)
            with col_bp3:
                if st.button("Compute Bandpower"):
                    f1_hz = bp_f1_mhz * 1e6
                    f2_hz = bp_f2_mhz * 1e6
                    bp_dbm = bandpower_linear(freqs, trace, f1_hz, f2_hz)
                    if bp_dbm is None:
                        st.warning(t("No data in this band or invalid range."))
                    else:
                        st.success(t("Approx. bandpower in") + f" [{bp_f1_mhz:.3f}, {bp_f2_mhz:.3f}] MHz: {bp_dbm:.2f} dBm")
        else:
            st.info(t("Bandpower not available (no trace)."))

        st.subheader(t("📶 WiFi-like RF Carriers (from SA trace)"))

        if freqs and trace:
            wifi_rf = detect_wifi_like_carriers(freqs, trace)
            span_bands = classify_span_wifi_bands(freqs)

            if span_bands:
                st.info(t("Current span overlaps WiFi band(s):") + f" {', '.join(span_bands)}")

            if wifi_rf:
                df_wifi_rf = pd.DataFrame(wifi_rf)
                st.dataframe(df_wifi_rf)
            else:
                st.info(t("No obvious WiFi-like peaks in the current span."))
        else:
            st.info(t("No RF trace to analyze for WiFi-like carriers."))

        # -----------------------------------------------------------------------------
        # WIFI SCAN (PC INTERFACE)
        # -----------------------------------------------------------------------------
        st.subheader(t("📶 WiFi Scanner (PC interface)"))

        if freqs and any(f >= 2.39e3 for f in freqs):
            try:
                import pywifi
                wifi = pywifi.PyWiFi()
                iface = wifi.interfaces()[0]
                iface.scan()
                time.sleep(2)
                res = iface.scan_results()
                wifi_list = [
                    {
                        "SSID": r.ssid or "<Hidden>",
                        "Frequency (MHz)": r.freq,
                        "Signal (dBm)": r.signal,
                    }
                    for r in res
                ]
                if wifi_list:
                    st.dataframe(
                        pd.DataFrame(wifi_list).sort_values(
                            by="Signal (dBm)", ascending=False
                        )
                    )
                else:
                    st.info(t("No WiFi networks detected."))
            except Exception as e:
                st.error(t("WiFi scan failed:") + f" {e}")
        else:
            st.info(t("WiFi scan skipped (spectrum not in 2.4/5/6 GHz)."))

        # -----------------------------------------------------------------------------
        # TIMER
        # -----------------------------------------------------------------------------
        timer.stop()
        st.write(t("⏱️ Elapsed:"), fmt_seconds(timer.elapsed()))

    elif equipment_type == "Keysight FieldFox":
        # Keysight handling
        with st.chat_message("assistant"):
            SYSTEM_PROMPT = f"""You are an AI assistant for {equipment_type} spectrum analyzer that follows strict rules:
- Always respond concisely.
- Provide factual information only.
- Use bullet points when listing multiple items.
- Maintain a professional tone."""

            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error(t("❌ OPENAI_API_KEY environment variable is not set. Please set it to use AI features."))
                st.stop()
            client = OpenAI(api_key=api_key)

            stream = client.chat.completions.create(
                model=st.session_state.get("openai_model", "gpt-4o-mini"),
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                temperature=0,
                top_p=0.2,
                max_tokens=200,
                frequency_penalty=1,
                presence_penalty=1,
                stream=True,
            )
            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # Parse commands for FieldFox
        keywords = ["start", "stop", "center", "span", "rbw"]
        matches = [word for word in prompt.split() if word.lower() in [k.lower() for k in keywords]]
        startf = extract_numbers(prompt)

        if matches and startf:
            st.info(f"Processing FieldFox command: {matches}")

    elif equipment_type == "tinySA":
        # tinySA handling with full RAG integration from ennoiaCAT_RAG_INTW_LIC.py
        # Start timer
        from timer import Timer, fmt_seconds
        timer = Timer()
        timer.start()

        with st.chat_message("assistant"):
            # Stage 1: Conversational AI response
            user_input = st.session_state.messages[-1]["content"]
            system_prompt = helper.get_system_prompt()
            few_shot_examples = helper.get_few_shot_examples()

            chat1 = [{"role": "system", "content": system_prompt}] + few_shot_examples + [{"role": "user", "content": user_input}]

            if "SLM" in selected_options and map_api is not None:
                response = map_api.generate_response(chat1)
            elif "SLM" in selected_options and map_api is None:
                st.warning(t("SLM model not available. Using template response."))
                response = f"I'm the {equipment_type} assistant. The SLM model is not currently loaded. Please switch to OpenAI mode or check your configuration."
            else:
                from openai import OpenAI
                client, ai_model = helper_class.load_OpenAI_model()
                openAImessage = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=chat1,
                    temperature=0,
                    max_tokens=200,
                    frequency_penalty=1,
                    stream=False
                )
                response = openAImessage.choices[0].message.content

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Stage 2: API parameter extraction
        with st.status("Configuring tinySA parameters...", expanded=False) as status:
            def_dict = {
                "plot": True,  # Enable plotting by default
                "scan": False,
                "start": 300000000.0,
                "stop": 900000000.0,
                "points": 101,
                "port": False,
                "device": None,
                "verbose": False,
                "capture": False,
                "command": None,
                "save": None
            }

            st.write(t("Extracting scan parameters from query..."))
            if map_api is not None:
                few_shot_examples2 = map_api.get_few_shot_examples()
                system_prompt2 = map_api.get_system_prompt(def_dict, user_input)
            else:
                few_shot_examples2 = []
                system_prompt2 = "Extract scan parameters as JSON."
            chat2 = [{"role": "system", "content": system_prompt2}] + few_shot_examples2 + [{"role": "user", "content": user_input}]

            if "SLM" in selected_options and map_api is not None:
                api_str = map_api.generate_response(chat2)
            elif "SLM" in selected_options and map_api is None:
                api_str = "{}"
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
            status.update(label="Parameters extracted!", state="complete")

        # Parse API response
        def_dict["save"] = "max_signal_strengths.csv"  # Set filename for CSV output
        api_dict = def_dict
        try:
            parsed = json.loads(api_str)
            if isinstance(parsed, dict):
                api_dict = parsed
                api_dict["save"] = "max_signal_strengths.csv"
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(api_str)
                if isinstance(parsed, dict):
                    api_dict = parsed
                    api_dict["save"] = "max_signal_strengths.csv"
            except (ValueError, SyntaxError):
                logger.warning("Failed to parse response. Using default options.")

        logger.debug(f"Parsed API options: {api_dict}")

        # Parse frequency units (GHz, MHz, Hz) to Hz
        def parse_frequency(value):
            """Convert frequency with units to Hz"""
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                value = value.strip().lower()
                # Extract number and unit
                match = re.match(r'([\d.]+)\s*(ghz|mhz|hz)?', value)
                if match:
                    num = float(match.group(1))
                    unit = match.group(2) if match.group(2) else 'hz'
                    if unit == 'ghz':
                        return num * 1e9
                    elif unit == 'mhz':
                        return num * 1e6
                    else:
                        return num
            return float(value)

        # Apply frequency parsing to start and stop
        if isinstance(api_dict, dict):
            if 'start' in api_dict:
                api_dict['start'] = parse_frequency(api_dict['start'])
            if 'stop' in api_dict:
                api_dict['stop'] = parse_frequency(api_dict['stop'])
            if 'center' in api_dict:
                api_dict['center'] = parse_frequency(api_dict['center'])
            if 'span' in api_dict:
                api_dict['span'] = parse_frequency(api_dict['span'])

            # Validation: Check if user input mentions higher MHz/GHz but AI extracted lower value
            # This catches cases where AI extracts "240" from "2400"
            freq_mentions = re.findall(r'(\d+)\s*(ghz|mhz)', user_input.lower())
            if freq_mentions and len(freq_mentions) >= 1:
                # First frequency mention likely corresponds to start
                if api_dict.get('start'):
                    num_str, unit = freq_mentions[0]
                    mentioned_hz = float(num_str) * (1e9 if unit == 'ghz' else 1e6)
                    if abs(api_dict['start'] - mentioned_hz/10) < mentioned_hz/20:
                        logger.warning(f"Correcting start: AI extracted {api_dict['start']/1e6}MHz but user mentioned {mentioned_hz/1e6}MHz")
                        api_dict['start'] = mentioned_hz

                # Second frequency mention likely corresponds to stop
                if len(freq_mentions) >= 2 and api_dict.get('stop'):
                    num_str, unit = freq_mentions[1]
                    mentioned_hz = float(num_str) * (1e9 if unit == 'ghz' else 1e6)
                    if abs(api_dict['stop'] - mentioned_hz/10) < mentioned_hz/20:
                        logger.warning(f"Correcting stop: AI extracted {api_dict['stop']/1e6}MHz but user mentioned {mentioned_hz/1e6}MHz")
                        api_dict['stop'] = mentioned_hz

            logger.debug(f"Parsed frequencies (Hz): Start={api_dict.get('start')}, Stop={api_dict.get('stop')}")

        # Configure and run tinySA
        freq = None  # Initialize freq
        configured_start = None
        configured_stop = None

        if isinstance(api_dict, dict):
            opt = SimpleNamespace(**api_dict)
            logger.debug(f"opt = {opt}")

            # Store configured frequencies for WiFi check
            configured_start = getattr(opt, 'start', None)
            configured_stop = getattr(opt, 'stop', None)

            with st.status("Scanning spectrum with tinySA...", expanded=False) as status:
                st.write(f"Frequency range: {configured_start/1e6:.1f} - {configured_stop/1e6:.1f} MHz")
                st.write(f"Scan points: {opt.points}")
                gcf = helper.configure_tinySA(opt)
                status.update(label="Spectrum scan complete!", state="complete")

            # Only display plot if figure was created
            if gcf is not None:
                st.pyplot(gcf)
        else:
            st.error(t("API response is not a valid dictionary. Setting default options."))

        # Operator table analysis
        with st.status("Analyzing cellular frequencies...", expanded=False) as status:
            try:
                result = helper.read_signal_strength('max_signal_strengths.csv')
                if not result:
                    status.update(label="No signal data available", state="error")
                else:
                    sstr, freq = result
                    # Check if freq array has data
                    if not freq or len(freq) == 0:
                        status.update(label="No frequency data in CSV", state="error")
                    else:
                        freq_mhz = [x / 1e6 for x in freq]
                        logger.debug(f"Signal strengths: {sstr}")
                        logger.debug(f"Frequencies (MHz): {freq_mhz}")
                        logger.info(f"Frequency range: {min(freq)/1e9:.3f} GHz to {max(freq)/1e9:.3f} GHz")

                        operator_table = helper.get_operator_frequencies()
                        if not operator_table:
                            st.error(t("Operator table could not be loaded."))
                        else:
                            frequency_report_out = helper.analyze_signal_peaks(sstr, freq_mhz, operator_table)
                            logger.debug(f"Frequency report: {frequency_report_out}")

                            if not frequency_report_out:
                                status.update(label="No cellular frequencies detected", state="complete")
                            else:
                                status.update(label="Cellular analysis complete!", state="complete")
                                st.subheader(t("🗼 List of Available Cellular Networks"))
                                df = pd.DataFrame(frequency_report_out)
                                st.dataframe(df)

            except Exception as e:
                status.update(label=f"Analysis failed: {str(e)}", state="error")
                logger.error(f"Analysis failed: {e}", exc_info=True)

        # WiFi scanner
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
            except (TypeError, ValueError) as e:
                logger.debug(f"freq_to_channel conversion failed for {freq}: {e}")
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
            except (TypeError, ValueError) as e:
                logger.debug(f"classify_band conversion failed for {freq}: {e}")
            return None

        def is_dfs_channel(channel):
            try:
                ch = int(channel)
            except (TypeError, ValueError):
                return False
            if 52 <= ch <= 64 or 100 <= ch <= 144:
                return True
            return False

        def infer_bandwidth(channel, radio_type):
            try:
                ch = int(channel)
            except (TypeError, ValueError):
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
            import pywifi
            wifi = pywifi.PyWiFi()
            iface = wifi.interfaces()[0]
            iface.scan()
            time.sleep(2)  # Reduced from 3s to 2s for faster response
            results = iface.scan_results()

            networks = []
            for net in results:
                ssid = net.ssid or "<Hidden>"
                bssid = net.bssid
                signal = net.signal
                freq = net.freq

                channel = freq_to_channel(freq)
                band = classify_band(freq)

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

        # Check if configured frequency range includes WiFi bands (2.4 GHz or higher)
        should_scan_wifi = False
        if configured_start is not None and configured_stop is not None:
            # Check if range overlaps with WiFi bands (2.4 GHz+)
            if configured_stop >= 2.39e9:
                should_scan_wifi = True
                logger.info(f"WiFi scan triggered: stop freq {configured_stop/1e9:.3f} GHz >= 2.39 GHz")
        elif freq and any(x >= 2.39e9 for x in freq):
            should_scan_wifi = True
            logger.info(f"WiFi scan triggered: freq data includes 2.4+ GHz")

        if should_scan_wifi:
            with st.status("Scanning WiFi networks...", expanded=False) as status:
                try:
                    st.write(t("Detecting WiFi networks in 2.4/5/6 GHz bands..."))
                    df = scan_wifi()
                    if df.empty:
                        status.update(label="No WiFi networks found", state="complete")
                    else:
                        status.update(label=f"Found {len(df)} WiFi networks!", state="complete")

                    st.subheader(t("📶 List of Available WiFi Networks"))
                    if df.empty:
                        st.warning(t("No networks found."))
                    else:
                        st.success(t("Found") + f" {len(df)} " + t("networks."))
                        st.dataframe(df)
                        st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="wifi_scan.csv", mime="text/csv")
                except Exception as e:
                    status.update(label=f"WiFi scan failed: {str(e)}", state="error")
                    st.error(t("WiFi scan failed:") + f" {e}")

        # Timer display
        timer.stop()
        st.write(t("⏱️ Elapsed:") + f" {fmt_seconds(timer.elapsed())}")

    elif equipment_type == "ORAN PCAP Analyzer":
        # ORAN PCAP Analyzer handling
        from timer import Timer, fmt_seconds
        timer = Timer()
        timer.start()

        with st.chat_message("assistant"):
            # Stage 1: Conversational AI response
            system_prompt = helper.get_system_prompt()
            few_shot_examples = helper.get_few_shot_examples()
            chat1 = [{"role": "system", "content": system_prompt}] + few_shot_examples + [{"role": "user", "content": prompt}]

            if "SLM" in selected_options and map_api is not None:
                response = map_api.generate_response(chat1)
            elif "SLM" in selected_options and map_api is None:
                # SLM failed to load, provide fallback response
                st.warning(t("SLM model not available. Using template response."))
                response = "I'm the ORAN PCAP Analyzer assistant. The SLM model is not currently loaded. Please use the sidebar to upload a PCAP file and run analysis, or switch to OpenAI mode for conversational responses."
            else:
                from openai import OpenAI
                client, ai_model = helper_class.load_OpenAI_model()
                openAImessage = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=chat1,
                    temperature=0,
                    max_tokens=500,
                    frequency_penalty=1,
                    stream=False
                )
                response = openAImessage.choices[0].message.content

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Check for reset confidence command
        prompt_lower = prompt.lower()
        if "reset confidence" in prompt_lower or "reset the confidence" in prompt_lower:
            result = helper.reset_confidence()
            with st.chat_message("assistant"):
                if "error" in result:
                    st.error(f"Failed to reset confidence: {result['error']}")
                else:
                    st.success("AI confidence levels have been reset to 0.0 for all layers. Detection history has been cleared.")
                st.session_state.messages.append({"role": "assistant", "content": "AI confidence levels have been reset to 0.0 for all layers."})

        # Check for analysis commands
        should_analyze = any(word in prompt_lower for word in ["analyze", "process", "detect", "check", "run", "start"])

        if should_analyze and pcap_filepath:
            with st.status("Analyzing ORAN PCAP file...", expanded=True) as status:
                st.write(f"File: {pcap_filepath}")
                st.write(f"Parameters: N_ID={oran_n_id}, nSCID={oran_nscid}, BW={oran_bandwidth}MHz, SCS={oran_scs}kHz")
                st.write(f"Frame Position: Subframe={oran_subframe}, Slot={oran_slot}")
                st.write(f"Detection Mode: {detection_mode}")

                # Send to Flask server
                params = {
                    "N_ID": int(oran_n_id),
                    "nSCID": int(oran_nscid),
                    "bandwidth": oran_bandwidth,
                    "scs": oran_scs,
                    "layers": oran_layers,
                    "link": oran_link,
                    "subframe": oran_subframe,
                    "slot": oran_slot,
                    "detection_mode": detection_mode
                }

                result = helper.analyze_pcap(pcap_filepath, params)

                if "error" in result:
                    status.update(label=f"Analysis failed: {result['error']}", state="error")
                    st.error(result["error"])
                else:
                    status.update(label="Analysis complete!", state="complete")

                    # Parse and format the result as markdown
                    raw_message = result.get("message", "")
                    try:
                        # Try to parse JSON from the message
                        analysis_data = json.loads(raw_message)

                        # Get detection mode from response (fallback to local variable)
                        response_detection_mode = analysis_data.get('detection_mode', detection_mode)

                        # Format as markdown
                        md_output = f"""
## Detection Mode: {response_detection_mode}

### Analysis Summary

| Field | Value |
|-------|-------|
| **Status** | ✓ Success |
| **Message** | {analysis_data.get('message', 'Analysis completed')} |
| **Detection Method** | {response_detection_mode} |
| **Interference Detected** | {'Yes' if analysis_data.get('interference', 0) else 'No'} ({analysis_data.get('interference', 0)}) |
| **CSV File** | `{analysis_data.get('csv_file', 'N/A')}` |

### EVM Results (dB)

"""
                        # Check if we have both AI and DMRS results
                        ai_evm = analysis_data.get('ai_evm_db')
                        dmrs_evm = analysis_data.get('dmrs_evm_db')
                        detection_mode = analysis_data.get('detection_mode', '')

                        if ai_evm and dmrs_evm:
                            # Both mode - show both EVM results
                            md_output += """| Layer | AI-Based EVM (dB) | DMRS-Based EVM (dB) |
|-------|-------------------|---------------------|
"""
                            for i in range(len(ai_evm)):
                                md_output += f"| Layer {i} | {ai_evm[i]:.2f} | {dmrs_evm[i]:.2f} |\n"
                        elif ai_evm:
                            # AI-only mode
                            md_output += """| Layer | AI-Based EVM (dB) |
|-------|-------------------|
"""
                            for i, evm in enumerate(ai_evm):
                                md_output += f"| Layer {i} | {evm:.2f} |\n"
                        else:
                            # DMRS-only mode or fallback
                            evm_values = analysis_data.get('evm_db', [])
                            md_output += """| Layer | EVM (dB) |
|-------|----------|
"""
                            for i, evm in enumerate(evm_values):
                                md_output += f"| Layer {i} | {evm:.2f} |\n"

                        # Add layer details if available
                        layers_data = analysis_data.get('layers', {})
                        if layers_data:
                            if ai_evm and dmrs_evm:
                                # Both mode - show two separate tables
                                md_output += """
### DMRS-Based Layer Details

| Layer | Start PRB | End PRB | DMRS EVM (dB) |
|-------|-----------|---------|---------------|
"""
                                for layer_name, layer_data in layers_data.items():
                                    layer_num = layer_name.replace('layer_', '')
                                    dmrs_val = layer_data.get('dmrs_evm_db')
                                    dmrs_str = f"{dmrs_val:.2f}" if dmrs_val is not None else "N/A"
                                    has_interf = layer_data.get('has_interference', False)
                                    start_prb = layer_data.get('start_prb', 0) + 1 if has_interf else layer_data.get('start_prb', 0)
                                    end_prb = layer_data.get('end_prb', 'N/A')
                                    md_output += f"| Layer {layer_num} | {start_prb} | {end_prb} | {dmrs_str} |\n"

                                ai_confidence_list = analysis_data.get('ai_confidence', [])
                                md_output += """
### AI-Based Layer Details

| Layer | Start PRB | End PRB | AI EVM (dB) | Confidence (%) |
|-------|-----------|---------|-------------|----------------|
"""
                                for layer_name, layer_data in layers_data.items():
                                    layer_num = layer_name.replace('layer_', '')
                                    ai_val = layer_data.get('ai_evm_db')
                                    ai_str = f"{ai_val:.2f}" if ai_val is not None else "N/A"
                                    # Use AI-specific interference fields
                                    ai_has_interf = layer_data.get('ai_has_interference', False)
                                    ai_start = layer_data.get('ai_start_prb', 0)
                                    ai_end = layer_data.get('ai_end_prb', 'N/A')
                                    start_prb = ai_start + 1 if ai_has_interf else ai_start
                                    conf = layer_data.get('ai_confidence', ai_confidence_list[int(layer_num)] if int(layer_num) < len(ai_confidence_list) else 0.0)
                                    md_output += f"| Layer {layer_num} | {start_prb} | {ai_end} | {ai_str} | {conf:.1f} |\n"
                            elif ai_evm:
                                # AI-only mode layer details with confidence
                                ai_confidence_list = analysis_data.get('ai_confidence', [])
                                md_output += """
### Layer Details

| Layer | Start PRB | End PRB | EVM (dB) | Confidence (%) |
|-------|-----------|---------|----------|----------------|
"""
                                for layer_name, layer_data in layers_data.items():
                                    layer_num = layer_name.replace('layer_', '')
                                    has_interf = layer_data.get('has_interference', False)
                                    start_prb = layer_data.get('start_prb', 0) + 1 if has_interf else layer_data.get('start_prb', 0)
                                    end_prb = layer_data.get('end_prb', 'N/A')
                                    conf = layer_data.get('ai_confidence', ai_confidence_list[int(layer_num)] if int(layer_num) < len(ai_confidence_list) else 0.0)
                                    md_output += f"| Layer {layer_num} | {start_prb} | {end_prb} | {layer_data.get('evm_db', 0):.2f} | {conf:.1f} |\n"
                            else:
                                # DMRS-only mode layer details
                                md_output += """
### Layer Details

| Layer | Start PRB | End PRB | EVM (dB) |
|-------|-----------|---------|----------|
"""
                                for layer_name, layer_data in layers_data.items():
                                    layer_num = layer_name.replace('layer_', '')
                                    has_interf = layer_data.get('has_interference', False)
                                    start_prb = layer_data.get('start_prb', 0) + 1 if has_interf else layer_data.get('start_prb', 0)
                                    end_prb = layer_data.get('end_prb', 'N/A')
                                    md_output += f"| Layer {layer_num} | {start_prb} | {end_prb} | {layer_data.get('evm_db', 0):.2f} |\n"

                        st.markdown(md_output)

                        # Add note about interference if detected
                        if analysis_data.get('interference', 0):
                            st.warning(t("**Note:** Interference detected. Check layer details for affected PRB ranges."))

                    except json.JSONDecodeError:
                        # Fallback to plain text if not valid JSON
                        st.success(raw_message)

            # Display analysis results
            st.subheader(t("📊 Analysis Results"))

            # Display plots if available
            plots = helper.get_analysis_plots()
            if plots:
                # Check if we have both DMRS and AI plots (Both mode)
                has_dmrs_plots = "plot1.png" in plots or "plot2.png" in plots
                has_ai_plots = "plot3.png" in plots or "plot4.png" in plots

                if has_dmrs_plots and has_ai_plots:
                    # Both mode - show plots in two groups
                    st.markdown("#### " + t("DMRS-Based Detection Plots"))
                    dmrs_plots = {k: v for k, v in plots.items() if k in ["plot1.png", "plot2.png"]}
                    if dmrs_plots:
                        cols = st.columns(len(dmrs_plots))
                        plot_labels = {
                            "plot1.png": "DMRS Constellation",
                            "plot2.png": "DMRS SNR Detection"
                        }
                        for i, (name, path) in enumerate(dmrs_plots.items()):
                            with cols[i]:
                                st.image(path, caption=plot_labels.get(name, name))

                    st.markdown("#### " + t("AI-Based Detection Plots"))
                    ai_plots = {k: v for k, v in plots.items() if k in ["plot3.png", "plot4.png"]}
                    if ai_plots:
                        cols = st.columns(len(ai_plots))
                        plot_labels = {
                            "plot3.png": "AI Phase-Corrected Constellation",
                            "plot4.png": "AI Interference Detection"
                        }
                        for i, (name, path) in enumerate(ai_plots.items()):
                            with cols[i]:
                                st.image(path, caption=plot_labels.get(name, name))
                else:
                    # Single mode - show all plots in one row
                    cols = st.columns(len(plots))
                    plot_labels = {
                        "plot1.png": "Constellation",
                        "plot2.png": "Interference Detection"
                    }
                    for i, (name, path) in enumerate(plots.items()):
                        with cols[i]:
                            st.image(path, caption=plot_labels.get(name, name.replace(".png", "").replace("_", " ").title()))

            # Display interference detection results
            st.subheader(t("🔍 Interference Detection Results"))
            progress = helper.get_progress()
            st.info(f"Status: {progress.get('status', 'Unknown')}")

            # Load and display CSV results if available
            # File descriptions: (filename, description, mode)
            csv_files = [
                ("data_symbols.csv", "Equalized IQ symbols", "DMRS"),
                ("rx_frame_iq_cap.csv", "Raw RX frame IQ data", "Both"),
                ("evm_per_prb.csv", "EVM per PRB per layer (dB)", "AI"),
                ("snr_per_prb.csv", "SNR per PRB per layer (dB)", "DMRS"),
                ("snr_diff_per_prb.csv", "Interference level per PRB per layer (dB)", "AI"),
            ]
            # Check which CSV files are available
            available_csvs = [(f, d, m) for f, d, m in csv_files if helper.load_analysis_csv(f) is not None]
            if available_csvs:
                with st.expander(t("📂 Analysis CSV Files") + f" ({len(available_csvs)} available)", expanded=False):
                    for csv_file, description, mode in available_csvs:
                        df = helper.load_analysis_csv(csv_file)
                        if df is not None:
                            # Nested expander for each CSV file (collapsed by default)
                            with st.expander(f"📁 {csv_file} - {description} [{mode} mode]", expanded=False):
                                # For per-PRB files, show all rows (273 PRBs); otherwise limit to 100
                                if "per_prb" in csv_file:
                                    st.dataframe(df, height=400)  # Show all PRBs with scrollable view
                                else:
                                    st.dataframe(df.head(100))
                                st.download_button(
                                    f"📥 Download {csv_file}",
                                    data=df.to_csv(index=False),
                                    file_name=csv_file,
                                    mime="text/csv",
                                    key=f"download_{csv_file}"
                                )

        elif should_analyze and not pcap_filepath:
            st.warning(t("Please upload a PCAP file or enter a file path in the sidebar to start analysis."))

        # Timer display
        timer.stop()
        st.write(t("⏱️ Elapsed:") + f" {fmt_seconds(timer.elapsed())}")

    else:
        # Generic equipment handling
        if helper_class:
            system_prompt = helper.get_system_prompt() if hasattr(helper, 'get_system_prompt') else f"You are an AI assistant for {equipment_type}."
            few_shot_examples = helper.get_few_shot_examples() if hasattr(helper, 'get_few_shot_examples') else []

            chat1 = [{"role": "system", "content": system_prompt}] + few_shot_examples + [{"role": "user", "content": prompt}]

            if "SLM" in selected_options and map_api is not None:
                response = map_api.generate_response(chat1)
            elif "SLM" in selected_options and map_api is None:
                st.warning(t("SLM model not available. Using template response."))
                response = f"I'm the {equipment_type} assistant. The SLM model is not currently loaded. Please switch to OpenAI mode or check your configuration."
            else:
                from openai import OpenAI
                client, ai_model = helper_class.load_OpenAI_model()
                openAImessage = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=chat1,
                    temperature=0,
                    max_tokens=200,
                    frequency_penalty=1,
                    stream=False
                )
                response = openAImessage.choices[0].message.content

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

st.write("---")
st.caption(f"Ennoia {equipment_type} Controller - Powered by AI")
