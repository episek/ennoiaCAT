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
st.set_page_config(page_title="Ennoia Equipment Controller", page_icon="📡")

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
# SIDEBAR BRANDING & EQUIPMENT SELECTION
# -----------------------------------------------------------------------------
st.sidebar.title("Equipment Selection")

# Equipment options - only tinySA is enabled in this production version
ALL_EQUIPMENT = [
    "tinySA",
    "Viavi OneAdvisor",
    "Keysight FieldFox",
    "Aukua XGA4250",
    "Cisco NCS540",
    "Rohde & Schwarz NRQ6",
    "ORAN PCAP Analyzer"
]
ENABLED_EQUIPMENT = ["tinySA"]  # Only tinySA enabled in this version

# Custom CSS to style disabled options (greyed out appearance)
st.markdown("""
<style>
/* Grey out disabled equipment options in sidebar selectbox dropdown */
section[data-testid="stSidebar"] div[data-baseweb="select"] ul[role="listbox"] li:not(:first-child) {
    color: #999999 !important;
    opacity: 0.6;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] ul[role="listbox"] li:first-child {
    color: inherit !important;
    font-weight: 600;
}
/* Also style the popover/menu options */
div[data-baseweb="popover"] ul[role="listbox"] li:not(:first-child) {
    color: #888888 !important;
    opacity: 0.5;
}
div[data-baseweb="popover"] ul[role="listbox"] li:first-child {
    color: inherit !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Default to tinySA (index 0)
equipment_type = st.sidebar.selectbox(
    "Select Equipment",
    ALL_EQUIPMENT,
    index=0
)

# Check if selected equipment is enabled
if equipment_type not in ENABLED_EQUIPMENT:
    st.sidebar.error(f"⚠️ {equipment_type} not available")
    st.error(f"**{equipment_type}** is not available in this version. Please select **tinySA**.")
    st.stop()

import os as _os
if _os.path.exists('ennoia.jpg'):
    st.sidebar.image('ennoia.jpg', width=200)

st.title(f"📡 Ennoia – {equipment_type} Agentic AI Control & Analysis")
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
# IMPORT TINYSA MODULES
# -----------------------------------------------------------------------------
from map_api import MapAPI
from tinySA_config import TinySAHelper
import tinySA
from timer import Timer, fmt_seconds

helper_class = TinySAHelper
helper = TinySAHelper()

# Display welcome message
st.markdown(t("Chat and Test with **Ennoia Technologies Connect Platform** ©. All rights reserved."))

# -----------------------------------------------------------------------------
# TINYSA CONNECTION STATUS
# -----------------------------------------------------------------------------
st.sidebar.subheader("🔌 tinySA Connection")
if "tinySA_port" not in st.session_state:
    try:
        st.session_state.tinySA_port = helper.getport()
        st.sidebar.success(f"Connected: {st.session_state.tinySA_port}")
    except Exception as e:
        st.session_state.tinySA_port = None
        st.sidebar.warning(f"tinySA not detected: {e}")

if st.session_state.tinySA_port:
    st.sidebar.info(f"Port: {st.session_state.tinySA_port}")

# -----------------------------------------------------------------------------
# AI MODEL SELECTION
# -----------------------------------------------------------------------------
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
        map_api = MapAPI(peft_model, tokenizer)
else:
    @st.cache_resource
    def load_openai():
        client, ai_model = helper_class.load_OpenAI_model()
        st.session_state["openai_model"] = ai_model
        return MapAPI()

    st.write(t("\n⏳ Working in ONLINE mode."))
    map_api = load_openai()
    st.write(t("\n✅ Online LLM model") + f" {st.session_state['openai_model']} " + t("loaded. Let's get to work.\n"))

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
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

# -----------------------------------------------------------------------------
# WIFI HELPER FUNCTIONS
# -----------------------------------------------------------------------------
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
    time.sleep(2)
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

# -----------------------------------------------------------------------------
# MAIN CHAT HANDLER
# -----------------------------------------------------------------------------
prompt = st.chat_input("Ask Ennoia about tinySA spectrum analysis:")

if prompt:
    # Start timer
    timer = Timer()
    timer.start()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

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
            response = "I'm the tinySA assistant. The SLM model is not currently loaded. Please switch to OpenAI mode or check your configuration."
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
            "plot": True,
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
    def_dict["save"] = "max_signal_strengths.csv"
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
        freq_mentions = re.findall(r'(\d+)\s*(ghz|mhz)', user_input.lower())
        if freq_mentions and len(freq_mentions) >= 1:
            if api_dict.get('start'):
                num_str, unit = freq_mentions[0]
                mentioned_hz = float(num_str) * (1e9 if unit == 'ghz' else 1e6)
                if abs(api_dict['start'] - mentioned_hz/10) < mentioned_hz/20:
                    logger.warning(f"Correcting start: AI extracted {api_dict['start']/1e6}MHz but user mentioned {mentioned_hz/1e6}MHz")
                    api_dict['start'] = mentioned_hz

            if len(freq_mentions) >= 2 and api_dict.get('stop'):
                num_str, unit = freq_mentions[1]
                mentioned_hz = float(num_str) * (1e9 if unit == 'ghz' else 1e6)
                if abs(api_dict['stop'] - mentioned_hz/10) < mentioned_hz/20:
                    logger.warning(f"Correcting stop: AI extracted {api_dict['stop']/1e6}MHz but user mentioned {mentioned_hz/1e6}MHz")
                    api_dict['stop'] = mentioned_hz

        logger.debug(f"Parsed frequencies (Hz): Start={api_dict.get('start')}, Stop={api_dict.get('stop')}")

    # Configure and run tinySA
    freq = None
    configured_start = None
    configured_stop = None

    if isinstance(api_dict, dict):
        opt = SimpleNamespace(**api_dict)
        logger.debug(f"opt = {opt}")

        configured_start = getattr(opt, 'start', None)
        configured_stop = getattr(opt, 'stop', None)

        with st.status("Scanning spectrum with tinySA...", expanded=False) as status:
            st.write(f"Frequency range: {configured_start/1e6:.1f} - {configured_stop/1e6:.1f} MHz")
            st.write(f"Scan points: {opt.points}")
            gcf = helper.configure_tinySA(opt)
            status.update(label="Spectrum scan complete!", state="complete")

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
    should_scan_wifi = False
    if configured_start is not None and configured_stop is not None:
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

st.write("---")
st.caption("Ennoia tinySA Controller - Powered by AI")
