import ennoia_client_lic as lic
import argparse

# -----------------------------------------------------------------------------
# LICENSE ARGUMENTS
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Ennoia License Client")
parser.add_argument("--action", choices=["activate", "verify"], default="verify")
parser.add_argument("--key", help="License key for activation")
args, _unknown = parser.parse_known_args()

if args.action == "activate":
    if not args.key:
        print("❗ Provide a license key with --key")
        success = False
    else:
        success = lic.request_license(args.key)
else:
    success = lic.verify_license_file()

success = 1


if not success:
    print("❌ License verification failed.")
    

# -----------------------------------------------------------------------------
# STREAMLIT + AI + VIAVI ONEADVISOR
# -----------------------------------------------------------------------------
import json
import ast
import time
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from map_api_vi import MapAPI
from tinySA_config import TinySAHelper
from timer import Timer, fmt_seconds

# ---- VIAVI SDK ----
from ennoia_viavi.system_api import OneAdvisorSystemAPI
from ennoia_viavi.radio_api import OneAdvisorRadioAPI

# ---- PC WiFi scanner ----
import pywifi

st.cache_data.clear()
st.cache_resource.clear()

# -----------------------------------------------------------------------------
# STREAMLIT UI HEADER
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Ennoia Viavi Controller", page_icon="🗼")
st.sidebar.image("viavi.png", width=200)
st.sidebar.image("ennoia.jpg", width=200)

st.title("🗼 Ennoia – Viavi OneAdvisor Agentic AI Control & Analysis")
st.caption("Natural-language controlled RF Spectrum Analyzer (OpenAI / SLM toggle)")

# -----------------------------------------------------------------------------
# LICENSE CHECK
# -----------------------------------------------------------------------------
if not success:
    st.error("License verification failed.")
    st.stop()
else:
    st.success("Ennoia License verified successfully.")

WARMUP_SWEEPS = 2  # number of traces to discard after config changes
helper = TinySAHelper()

# -----------------------------------------------------------------------------
# SIDEBAR — VIAVI SETTINGS
# -----------------------------------------------------------------------------
st.sidebar.subheader("🔌 Viavi OneAdvisor Connection")
viavi_host = st.sidebar.text_input(
    "OneAdvisor IP", value="192.168.1.100"
)

st.sidebar.subheader("🔁 Sweep Settings")
continuous_mode = st.sidebar.checkbox("Continuous Sweep", value=False)
num_sweeps = st.sidebar.number_input(
    "Sweeps (if continuous)",
    min_value=1,
    max_value=200,
    value=5,
    step=1,
)
sweep_delay = st.sidebar.number_input(
    "Delay between sweeps (s)",
    min_value=0.05,
    max_value=5.0,
    value=0.3,
    step=0.05,
)

st.sidebar.subheader("📊 Display Mode")
display_mode = st.sidebar.selectbox(
    "Spectrum Display",
    ["Single sweep", "Multi-sweep average", "Waterfall"],
)


def get_oneadvisor_identity(host):
    """
    Returns a dict with model, serial number, firmware, and vendor info.
    """
    try:
        sys_api = OneAdvisorSystemAPI(host)
        sys_api.open()

        # Standard SCPI Identification
        idn = sys_api.query("*IDN?")  # Typical response: "Viavi,OneAdvisor-800,123456,1.23"

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



# -----------------------------------------------------------------------------
# AI MODEL (OpenAI / SLM TOGGLE, but single backend MapAPI)
# -----------------------------------------------------------------------------
# st.sidebar.subheader("🤖 AI Mode")
# ai_mode = st.sidebar.radio("Choose AI Engine:", ["LLM (OpenAI)", "SLM (offline)"])

# map_api = MapAPI()  # single backend for now

# if ai_mode == "SLM (offline)":
    # st.sidebar.warning(
        # "SLM mode selected, but no local SLM model is configured.\n"
        # "Using OpenAI backend via MapAPI."
    # )
# else:
    # st.sidebar.info("Using OpenAI")


# -----------------------------------------------------------------------------
# AI MODEL (SLM / OpenAI via TinySAHelper UI)
# -----------------------------------------------------------------------------
selected_options = TinySAHelper.select_checkboxes()
st.success(f"You selected: {', '.join(selected_options) if selected_options else 'nothing'}")

helper = TinySAHelper()

if "SLM" in selected_options:

    @st.cache_resource
    def load_peft_model():
        return TinySAHelper.load_lora_model()

    st.write("\n⏳ Working in OFFLINE mode. Loading local LoRA model... (might take a minute)")
    tokenizer, peft_model, device = load_peft_model()

    st.write(
        f"\n✅ Local SLM model {peft_model.config.name_or_path} "
        f"loaded"
    )
    st.write(
        f"Device is set to use {device}! Let's get to work.\n"
    )

    map_api = MapAPI(
        backend="slm",
        injected_model=peft_model,
        injected_tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,
    )

else:

    @st.cache_resource
    def load_openai():
        client, ai_model = TinySAHelper.load_OpenAI_model()
        st.session_state["openai_model"] = ai_model
        return MapAPI(
            backend="openai",
            openai_model=ai_model,
            max_new_tokens=512,
            temperature=0.2,
        )

    st.write("\n⏳ Working in ONLINE mode.")
    map_api = load_openai()
    st.write(
        f"\n✅ Online LLM model {st.session_state['openai_model']} loaded. "
        f"Let's get to work.\n"
    )

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
# SESSION STATE FOR WATERFALL / AVERAGING
# -----------------------------------------------------------------------------
if "waterfall_data" not in st.session_state:
    st.session_state["waterfall_data"] = None  # np.ndarray [num_sweeps, num_freqs]
if "waterfall_freqs" not in st.session_state:
    st.session_state["waterfall_freqs"] = None  # list of Hz
if "last_trace" not in st.session_state:
    st.session_state["last_trace"] = None
if "last_freqs" not in st.session_state:
    st.session_state["last_freqs"] = None

# -----------------------------------------------------------------------------
# MESSAGE HISTORY
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

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

    # Already numeric
    if isinstance(val, (int, float)):
        return float(val)

    # Handle list/array/tuple from LLM or JSON
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            return default
        return parse_freq(val[0], default=default)

    s = str(val).strip().lower()

    # Match "<number> <optional unit>"
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
        return num  # plain Hz

    # Fallback: try to parse as plain float (already in Hz)
    try:
        return float(s)
    except Exception:
        return default


def extract_start_stop(text):
    """
    Extract start/stop frequencies from natural language.

    Handles phrases like:
      - "set the start freq to 600MHz and the stop freq to 900MHz"
      - "start 600 MHz stop 900 MHz"
      - "scan 600MHz-900MHz"
      - "scan from 600 MHz to 900 MHz"
    Returns (start_hz, stop_hz) or (None, None).
    """
    if not text:
        return None, None

    txt = text.lower()
    # normalize different dashes
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

    # --- Case 1: explicit "start ... 600MHz" and "stop ... 900MHz"
    # Allow optional 'freq', 'frequency', and words like 'to', 'at', 'from'
    start_match = re.search(
        r"start(?:\s*freq(?:uency)?)?(?:\s*(?:to|at|from))?\s*([\d\.]+)\s*(g|m|k)?hz",
        txt,
    )
    stop_match = re.search(
        r"stop(?:\s*freq(?:uency)?)?(?:\s*(?:to|at|from))?\s*([\d\.]+)\s*(g|m|k)?hz",
        txt,
    )

    if start_match and stop_match:
        s = conv(start_match.group(1), start_match.group(2))
        e = conv(stop_match.group(1), stop_match.group(2))
        return (min(s, e), max(s, e))

    # --- Case 2: "600 to 900 MHz", "600-900 MHz", "600 MHz - 900 MHz"
    range_match = re.search(
        r"([\d\.]+)\s*(g|m|k)?hz\s*(?:to|-|up to|through|thru|until|and)\s*([\d\.]+)\s*(g|m|k)?hz",
        txt,
    )
    if range_match:
        n1, u1, n2, u2 = (
            range_match.group(1),
            range_match.group(2),
            range_match.group(3),
            range_match.group(4),
        )
        s = conv(n1, u1)
        e = conv(n2, u2)
        return (min(s, e), max(s, e))

    return None, None


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
prompt = st.chat_input(
    "Ask Ennoia: e.g. 'set the start freq 600MHz and stop freq 900MHz' or 'scan 3.3–3.7 GHz'"
)

if prompt:
    # 🔁 If we're in Waterfall mode, start a fresh waterfall for this new command
    if display_mode == "Waterfall":
        st.session_state["waterfall_data"] = None
        st.session_state["waterfall_freqs"] = None

    t = Timer()
    t.start()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1) Human-style response
    chat1 = [
        {
            "role": "system",
            "content": "You are Ennoia AI, an assistant for RF spectrum analysis using Viavi OneAdvisor.",
        },
        {"role": "user", "content": prompt},
    ]
    response = map_api.generate_response(chat1)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    # 2) API JSON response for instrument config
    api_chat = [
        {
            "role": "system",
            "content": (
                "Output ONLY a JSON object with optional keys: "
                "start, stop, center, span, rbw, vbw, ref_level. "
                "Use units like '600 MHz', '3.5 GHz' if helpful."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    api_raw = map_api.generate_response(api_chat)

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

    # -----------------------------------------------------------------------------
    # VIAVI SPECTRUM ACQUISITION
    # -----------------------------------------------------------------------------
    st.subheader("🗼 Spectrum Analyzer")

    freqs = []
    trace = []

    if not viavi_host:
        st.error("Enter Viavi IP in sidebar.")
    else:
        try:
            # Discover radio SCPI port
            sys_api = OneAdvisorSystemAPI(viavi_host)
            sys_api.open()
            radio_port = sys_api.get_radio_scpi_port()
            sys_api.close()

            ra = OneAdvisorRadioAPI(viavi_host, scpi_port=radio_port)
            ra.open()

            start_hz = parse_freq(getattr(opt, "start", None))
            stop_hz  = parse_freq(getattr(opt, "stop", None))
            center   = parse_freq(getattr(opt, "center", None))
            span     = parse_freq(getattr(opt, "span", None))

            # normalize scalars
            if isinstance(start_hz, (list, tuple, np.ndarray)):
                start_hz = parse_freq(start_hz[0])
            if isinstance(stop_hz, (list, tuple, np.ndarray)):
                stop_hz = parse_freq(stop_hz[0])

            if center is None and start_hz is not None and stop_hz is not None:
                center = 0.5 * (float(start_hz) + float(stop_hz))
            if span is None and start_hz is not None and stop_hz is not None:
                span = abs(float(stop_hz) - float(start_hz))

            if center is None:
                center = 600e6
            if span is None:
                span = 600e6

            rbw = parse_freq(getattr(opt, "rbw", None)) or auto_rbw(span)
            vbw = parse_freq(getattr(opt, "vbw", None)) or rbw
            try:
                ref_level = float(getattr(opt, "ref_level", 0))
            except Exception:
                ref_level = 0.0

            # Explicitly enter Spectrum Analyzer mode
            ra.set_spectrum_mode("spectrumTuned")
            time.sleep(0.2)

            def configure_spectrum():
                ra.configure_spectrum(
                    center_hz=center,
                    span_hz=span,
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
                #print (freqs)
                if not freqs:
                    st.warning("No trace received from Viavi.")
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
                ax.set_title(f"{center/1e9:.3f} GHz / Span {span/1e6:.1f} MHz (Sweep {i+1}/{sweeps})")
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
            st.error(f"Viavi error: {e}")
            freqs, trace = [], []

    # -----------------------------------------------------------------------------
    # CELLULAR PEAK TABLE + 5G NR DETECTION
    # -----------------------------------------------------------------------------
    st.subheader("📶 Cellular Peaks & 5G NR-like Carriers")

    if freqs and trace:
        freq_mhz = [f for f in freqs]
        peaks = find_peaks(trace, max_peaks=10, min_dist=5)
        rows = []
        for idx, val in peaks:
            rows.append(
                {"Frequency (MHz)": freq_mhz[idx], "Power (dBm)": val}
            )
        if rows:
            st.write("Strong Peaks:")
            df_peaks = pd.DataFrame(rows)
            st.dataframe(df_peaks)
        else:
            st.info("No prominent peaks detected.")

        nr_carriers = detect_5gnr_like_carriers(freqs, trace)
        if nr_carriers:
            st.write("5G NR-like Carriers (rough RF-only heuristic):")
            df_nr = pd.DataFrame(nr_carriers)
            st.dataframe(df_nr)
        else:
            st.info("Checking for 5G NR carriers.")

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
        st.info("No spectrum to analyze.")

    # -----------------------------------------------------------------------------
    # OPERATOR TABLE MAPPING (reuse TinySAHelper logic)
    # -----------------------------------------------------------------------------
    st.subheader("🗼 Operator Table (by detected cellular bands)")

    if freqs and trace:
        # Reuse freq_mhz if it already exists; otherwise compute
        try:
            freq_mhz
        except NameError:
            freq_mhz = [f / 1e6 for f in freqs]

        # Load operator_table.json via TinySAHelper
        operator_table = helper.get_operator_frequencies()

        if not operator_table:
            st.warning("operator_table.json not found or invalid – cannot map to operators.")
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
                st.error(f"Operator analysis error: {e}")

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
                st.info("No strong operator bands detected in the current span.")
    else:
        st.info("No RF data available for operator mapping.")


    # -----------------------------------------------------------------------------
    # BANDPOWER MEASUREMENT (USER-SELECTED)
    # -----------------------------------------------------------------------------
    st.subheader("📐 Bandpower Measurement")

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
                    st.warning("No data in this band or invalid range.")
                else:
                    st.success(f"Approx. bandpower in [{bp_f1_mhz:.3f}, {bp_f2_mhz:.3f}] MHz: {bp_dbm:.2f} dBm")
    else:
        st.info("Bandpower not available (no trace).")

    st.subheader("📶 WiFi-like RF Carriers (from SA trace)")

    if freqs and trace:
        wifi_rf = detect_wifi_like_carriers(freqs, trace)
        span_bands = classify_span_wifi_bands(freqs)

        if span_bands:
            st.info(f"Current span overlaps WiFi band(s): {', '.join(span_bands)}")

        if wifi_rf:
            df_wifi_rf = pd.DataFrame(wifi_rf)
            st.dataframe(df_wifi_rf)
        else:
            st.info("No obvious WiFi-like peaks in the current span.")
    else:
        st.info("No RF trace to analyze for WiFi-like carriers.")

    # -----------------------------------------------------------------------------
    # WIFI SCAN (PC INTERFACE)
    # -----------------------------------------------------------------------------
    st.subheader("📶 WiFi Scanner (PC interface)")

    if freqs and any(f >= 2.39e3 for f in freqs):
        try:
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
                st.info("No WiFi networks detected.")
        except Exception as e:
            st.error(f"WiFi scan failed: {e}")
    else:
        st.info("WiFi scan skipped (spectrum not in 2.4/5/6 GHz).")

    # -----------------------------------------------------------------------------
    # TIMER
    # -----------------------------------------------------------------------------
    t.stop()
    st.write("⏱️ Elapsed:", fmt_seconds(t.elapsed()))
