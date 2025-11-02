# ennoia_agentic_app.py
"""
Ennoia Agentic App — SSH/SFTP backend flow (no HTTP)
- Streamlit (Windows) SSHes into EC2, runs the location builder, SFTPs JSON back,
  waits for the local file, loads it, then runs analysis.

Parts:
1) ConfigurationAgent → model/load, options, device setup (SLM/LLM)
2) LocationAgent      → detect location → SSH to EC2 to build → SFTP → load
3) AnalysisAgent      → cellular+Wi-Fi analysis with agentic steps

Env vars you can set on Windows (PowerShell setx):
  setx ENNOIA_SSH_HOST "ec2-XX-XX-XX-XX.compute.amazonaws.com"
  setx ENNOIA_SSH_USER "ec2-user"
  setx ENNOIA_SSH_KEY  "C:\\Keys\\ennoia-ec2.pem"
  setx ENNOIA_REMOTE_PY "python3"
  setx ENNOIA_REMOTE_SCRIPT "/home/ec2-user/operator_table_service.py"
  setx ENNOIA_REMOTE_OUT "/tmp/ennoia_tables"
  setx ENNOIA_LOCAL_OUT  "C:\\ennoia\\tables"
"""

from __future__ import annotations
import argparse
import ast
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import requests

import pandas as pd
import streamlit as st

# === Your existing libs ===
import ennoia_client_lic as lic
from tinySA_config import TinySAHelper
from map_api import MapAPI
from timer import Timer, fmt_seconds

# Optional Wi-Fi scan
try:
    import pywifi
    from pywifi import const  # noqa: F401
    PYWIFI_AVAILABLE = True
except Exception:
    PYWIFI_AVAILABLE = False

# Optional SSH/SFTP (Paramiko)
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except Exception:
    PARAMIKO_AVAILABLE = False

# ---------- CLI: license activate/verify ----------
parser = argparse.ArgumentParser(description="Ennoia License Client")
parser.add_argument("--action", choices=["activate", "verify"], default="verify",
                    help="Action to perform (default: verify)")
parser.add_argument("--key", help="Ennoia License key for activation")
args, _ = parser.parse_known_args()

if args.action == "activate":
    if not args.key:
        print("❗ Please provide a license key with --key")
        success = False
    else:
        success = lic.request_license(args.key)
else:
    success = lic.verify_license_file()

# ---------- Streamlit page setup ----------
st.set_page_config(page_title="Ennoia Technologies", page_icon="🤖")
st.sidebar.image('aws_logo1.png')
st.sidebar.image('ennoia.jpg')
st.title("Ennoia Technologies")
st.markdown("Rapid Edge Analysis (REA) with Ennoia Connect Platform ©. All rights reserved.")

success = True
if not success:
    st.error("Ennoia License verification failed. Please check your license key or contact support.")
    st.stop()
else:
    st.success("Ennoia License verified successfully.")

# Clear caches on hot-reload
st.cache_data.clear()
st.cache_resource.clear()

# =====================================================================================
#                                      Core Layer
# =====================================================================================

@dataclass
class ModelProvider:
    """Abstracts SLM (offline) vs LLM (online) usage for agents."""
    mode: str  # 'SLM' or 'LLM'
    helper: TinySAHelper

    # Lazily filled
    tokenizer: Optional[object] = None
    peft_model: Optional[object] = None
    device: Optional[str] = None
    client: Optional[object] = None
    ai_model: Optional[str] = None
    map_api: Optional[MapAPI] = None

    def init(self):
        if self.mode == 'SLM':
            st.write("\n⏳ Working in OFFLINE mode. Loading local model…")

            @st.cache_resource(show_spinner=False)
            def _load_local():
                return self.helper.load_lora_model()

            @st.cache_resource
            def load_model_and_tokenizer():
                return TinySAHelper.load_lora_model()


            #self.tokenizer, self.peft_model, self.device = _load_local()
            self.tokenizer, self.peft_model, self.device = load_model_and_tokenizer()
            st.write(f"Device set to use {self.device}")
            #map_api = MapAPI(peft_model, tokenizer)
            self.map_api = MapAPI(self.peft_model, self.tokenizer)
            st.write(f"Device set to use {self.device}")
            st.success(f"Local SLM model {getattr(self.peft_model, 'config', SimpleNamespace(name_or_path='local')).name_or_path} loaded!")
        else:
            st.write("\n⏳ Working in ONLINE mode.")
            # must be a static/class method in your TinySAConfig
            self.client, self.ai_model = TinySAHelper.load_OpenAI_model()
            self.map_api = MapAPI()
            st.success(f"Online LLM model {self.ai_model} ready!")

    def chat(self, messages: List[Dict], *, temperature=0, max_tokens=200, frequency_penalty=1) -> str:
        if self.mode == 'SLM':
            return self.map_api.generate_response(messages)
        else:
            resp = self.client.chat.completions.create(
                model=self.ai_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                stream=False,
            )
            return resp.choices[0].message.content

# =====================================================================================
#                                 Agent 1: Connectivity
# =====================================================================================
class ConfigurationAgent:
    def __init__(self, helper: TinySAHelper, provider: ModelProvider):
        self.helper = helper
        self.provider = provider

    @st.cache_data(show_spinner=False)
    def _defaults_from_map_api(_map_api: MapAPI):
        return _map_api.get_defaults_opts()

    def run(self):
        st.header("1) Rapid Edge Analysis: Connectivity Agent")
        selected_options = TinySAHelper.select_checkboxes()
        st.success(f"You selected: {', '.join(selected_options) if selected_options else 'nothing'}")

        mode = 'SLM' if 'SLM' in selected_options else 'LLM'
        self.provider.mode = mode
        self.provider.init()

        def_dict = ConfigurationAgent._defaults_from_map_api(self.provider.map_api)

        if "tinySA_port" not in st.session_state:
            st.session_state.tinySA_port = self.helper.getport()

        st.write("Hi. I am Ennoia, your AI assistant. How can I help you today?")
        return selected_options, def_dict

# =====================================================================================
#                            SSH/SFTP helpers (no HTTP path) — HARDENED
# =====================================================================================
import os, re, json, time
import paramiko

def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def _out_filename(city: str, country: str) -> str:
    return f"operator_table_{_norm_name(city)}_{_norm_name(country)}.json"

def _load_pkey_any(pem_path: str):
    last_exc = None
    for Key in (paramiko.RSAKey, paramiko.Ed25519Key, paramiko.ECDSAKey):
        try:
            return Key.from_private_key_file(pem_path)
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"Failed to load private key '{pem_path}': {last_exc}")

def _open_sftp_with_retries(ssh: paramiko.SSHClient, retries: int = 3, base_sleep: float = 0.8):
    last = None
    for i in range(retries):
        try:
            return ssh.open_sftp()
        except Exception as e:
            last = e
            time.sleep(base_sleep + i * 0.4)
    raise RuntimeError(f"File transfer failed after {retries} attempts: {last}")

def _adopt_operator_table(local_json_path: str, *, city: str, region: str):
    with open(local_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Single source of truth
    st.session_state["operator_table_df"] = df
    st.session_state["operator_table_source"] = local_json_path
    st.session_state["active_city"] = city
    st.session_state["active_region"] = region

    # Bump revision to invalidate any caches keyed on it
    st.session_state["operator_table_rev"] = st.session_state.get("operator_table_rev", 0) + 1

    st.success(f"✅ Loaded {len(df)} rows from {os.path.basename(local_json_path)} "
               f"for {city}, {region} (rev {st.session_state['operator_table_rev']})")

    # Optional: clear Streamlit caches if you used @st.cache_* on any loader
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass

    # Force UI to render with the new table
    st.rerun()

def run_remote_build_and_fetch(
    *,
    host: str,
    username: str,
    pem_path: str,
    remote_python: str,
    remote_script: str,
    location: str,
    remote_out_dir: str = "/tmp/ennoia_tables",
    local_out_dir: str | None = None,
    timeout_s: int = 300,
) -> Optional[str]:
    """
    1) SSH to EC2
    2) Run: <remote_python> <remote_script> --location "<location>" --out_dir <remote_out_dir>
    3) Prefer the absolute JSON path printed by the remote script (stdout)
    4) SFTP download to local_out_dir
    5) Return local path (str) or None on failure
    """
    try:
        import streamlit as st  # ensure we’re in Streamlit runtime
    except Exception:
        pass

    try:
        pkey = _load_pkey_any(pem_path)
    except Exception as e:
        st.error(f"Key load failed: {e}")
        return None

    local_out_dir = local_out_dir or os.getenv("ENNOIA_LOCAL_OUT", ".")
    os.makedirs(local_out_dir, exist_ok=True)

    # Build the expected filename as a fallback (remote will also print absolute path)
    parts = [p.strip() for p in (location or "").split(",") if p.strip()]
    if len(parts) < 1:
        st.error("Location must be 'City, Country' (e.g., 'Dublin, Ireland').")
        return None
    city = parts[0]
    country = parts[-1] if len(parts) > 1 else ""
    fallback_remote_file = f"{remote_out_dir.rstrip('/')}/{_out_filename(city, country)}"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    with st.status("🔐 Connecting to AWS from Edge for Deep Packet Inspection ...", expanded=False) as status:
        try:
            client.connect(
                hostname=host,
                username=username,
                pkey=pkey,
                port=22,
                look_for_keys=False,
                allow_agent=False,
                timeout=30,
                auth_timeout=30,
                banner_timeout=200,
            )
        except Exception as e:
            status.update(label=f"SSH connection failed: {e}", state="error")
            return None

        status.update(label="🚀 Running remote build script…")
        # Ensure remote dir exists, then run builder. Use json.dumps for safe quoting.
        cmd = (
            f'mkdir -p {json.dumps(remote_out_dir)} && '
            f'{json.dumps(remote_python)} {json.dumps(remote_script)} '
            f'--location {json.dumps(location)} '
            f'--out_dir {json.dumps(remote_out_dir)}'
        )
        stdin, stdout, stderr = client.exec_command(cmd)

        # Read streams AFTER process exits to avoid race conditions
        rc = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", "ignore").strip()
        err = stderr.read().decode("utf-8", "ignore").strip()
        if err:
            st.caption(f"Remote stderr:\n{err[:2000]}")

        if rc != 0:
            status.update(label=f"Remote process failed (rc={rc})", state="error")
            client.close()
            return None

        # Prefer the path printed by the remote script (last non-empty stdout line)
        printed_path = ""
        for line in reversed(out.splitlines()):
            if line.strip():
                printed_path = line.strip()
                break

        if not printed_path.endswith(".json"):
            # Fall back to the deterministic name if stdout didn’t carry the path
            remote_json_path = fallback_remote_file
        else:
            remote_json_path = printed_path

        # Open SFTP with retries (avoids “unpack requires a buffer of 4 bytes”)
        try:
            sftp = _open_sftp_with_retries(client, retries=3)
        except Exception as e:
            status.update(label=f"File open failed: {e}", state="error")
            client.close()
            return None

        status.update(label="📦 Waiting for remote JSON…")
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                sftp.stat(remote_json_path)
                break
            except IOError:
                time.sleep(1.2)
        else:
            status.update(label=f"Timed out waiting for {os.path.basename(remote_json_path)}", state="error")
            sftp.close(); client.close()
            return None

        status.update(label="⬇️ Receiving from AWS to Edge…")
        local_path = os.path.join(local_out_dir, os.path.basename(remote_json_path))
        try:
            sftp.get(remote_json_path, local_path)
            status.update(label=f"✅ Downloaded: {os.path.basename(local_path)}", state="complete")
        except Exception as e:
            status.update(label=f"SFTP get failed: {e}", state="error")
            sftp.close(); client.close()
            return None
        finally:
            try: sftp.close()
            except Exception: pass
            client.close()

    return local_path

# =====================================================================================
#                                 Agent 3: Location
# =====================================================================================

DALLAS_HOME = ("Dallas", "TX")

class LocationAgent:
    """Detect location, build via SSH on EC2, SFTP JSON back, block until available."""

    def __init__(self, helper: TinySAHelper):
        self.helper = helper

    def _get_location_ip(self) -> Optional[Tuple[str, str]]:
        try:
            import urllib.request, json as _json
            with urllib.request.urlopen("https://ipinfo.io/json", timeout=3) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
                city = data.get("city")
                country_code = (data.get("country") or "").upper()
                iso2_to_name = {"IE": "Ireland", "GB": "United Kingdom", "UK": "United Kingdom", "US": "United States"}
                country = iso2_to_name.get(country_code, country_code)
                if city and country:
                    return city, country
        except Exception:
            return None
        return None

    def _manual_location_input(self) -> Tuple[str, str]:
        """Atomic location input: apply changes only when 'Set location' is clicked."""
        # Defaults (one-time)
        st.session_state.setdefault("active_city", "Dallas")
        st.session_state.setdefault("active_region", "TX")  # state or country
        st.session_state.setdefault("location_changed", False)

        st.caption(f"Active location: **{st.session_state['active_city']}, {st.session_state['active_region']}**")

        # Inputs are non-reactive; they only take effect on submit
        with st.form("location_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            new_city   = col1.text_input("City", value=st.session_state["active_city"], key="loc_city_input")
            new_region = col2.text_input("State/Country", value=st.session_state["active_region"], key="loc_region_input")
            apply_btn  = st.form_submit_button("Set location")

        if apply_btn:
            st.session_state["active_city"] = new_city.strip()
            st.session_state["active_region"] = new_region.strip()
            st.session_state["location_changed"] = True
            st.success(f"Location set to **{st.session_state['active_city']}, {st.session_state['active_region']}**")

        # Always return the current *active* pair
        return st.session_state["active_city"], st.session_state["active_region"]

    def detect_and_gate(self) -> Optional[pd.DataFrame]:
        st.header("2) Rapid Edge Analysis: Configuration Agent")

        # SSH settings panel (read defaults from env)
        with st.expander("🔧 Remote AWS settings", expanded=False):
            ssh_host = st.text_input("Public IP / DNS", value=os.getenv("ENNOIA_SSH_HOST", ""))
            ssh_user = st.text_input("SSH Username", value=os.getenv("ENNOIA_SSH_USER", "ec2-user"))
            ssh_key  = st.text_input("Path to .pem (Windows)", value=os.getenv("ENNOIA_SSH_KEY", r"C:\Users\rices\ennoiaCAT\AWS=Hackathon.pem"))
            remote_python = st.text_input("Remote Python", value=os.getenv("ENNOIA_REMOTE_PY", "python3"))
            remote_script = st.text_input("Remote script path", value=os.getenv("ENNOIA_REMOTE_SCRIPT", "/home/ec2-user/operator_table_service.py"))
            remote_out_dir = st.text_input("Remote output dir", value=os.getenv("ENNOIA_REMOTE_OUT", "/tmp/ennoia_tables"))
            local_out_dir = st.text_input("Local output dir", value=os.getenv("ENNOIA_LOCAL_OUT", r"C:\Users\rices\ennoiaCAT"))

        # Detect/enter location
        method = st.radio("Location method", ["Auto (IP)", "Manual"], horizontal=True)
        if method == "Auto (IP)":
            detected = self._get_location_ip() or self._manual_location_input()
        else:
            detected = self._manual_location_input()

        # Optional GPS reverse-geocode
        with st.expander("Optional: GPS coordinates → reverse-geocode"):
            lat = st.text_input("Latitude", "")
            lon = st.text_input("Longitude", "")
            if lat and lon:
                try:
                    latf, lonf = float(lat), float(lon)
                    try:
                        from geopy.geocoders import Nominatim
                        geolocator = Nominatim(user_agent="ennoia_agentic_app")
                        loc = geolocator.reverse((latf, lonf), timeout=5, language="en")
                        if loc:
                            address = loc.raw.get("address", {})
                            dcity = address.get("city") or address.get("town") or address.get("village") or address.get("county")
                            dcountry = address.get("country")
                            if dcity and dcountry:
                                detected = (dcity, dcountry)
                                st.info(f"Reverse-geocoded: {dcity}, {dcountry}")
                    except Exception:
                        st.caption("geopy not available or reverse-geocode failed; continuing with IP/manual.")
                except Exception:
                    st.warning("GPS values not valid")

        city, state = detected
        st.write(f"**Detected location:** {city}, {state}")

        # Auto-run remote build if SSH config looks valid, otherwise show button
        uploaded_df: Optional[pd.DataFrame] = None
        location_str = f"{city}, {state}"

        def _try_run():
            if not all([ssh_host, ssh_user, ssh_key, remote_python, remote_script, remote_out_dir, local_out_dir]):
                st.warning("SSH settings incomplete. Please fill in the SSH panel or upload a table manually.")
                st.session_state.setdefault("carrier_guard_active", True)
                return None
            local_path = run_remote_build_and_fetch(
                host=ssh_host,
                username=ssh_user,
                pem_path=ssh_key,
                remote_python=remote_python,
                remote_script=remote_script,
                location=location_str,
                remote_out_dir=remote_out_dir,
                local_out_dir=local_out_dir,
                timeout_s=300,
            )
            return local_path

        # Auto-run once per new location (no extra click)
        key_loc = f"last_loc_{_norm_name(location_str)}"
        already_ran = st.session_state.get(key_loc, False)
        local_path = None
        if not already_ran:
            local_path = _try_run()
            st.session_state[key_loc] = True

        # Manual trigger if needed
        if st.button("Build operator table from AWS now"):
            local_path = _try_run()

        # If file arrived, adopt it atomically (updates session + reruns)
        if local_path and os.path.exists(local_path):
            _adopt_operator_table(local_path, city=city, region=state)
            # _adopt_operator_table() calls st.rerun(), so code after this won’t run on success.

        # Fallback: manual upload
        if st.session_state.get("carrier_guard_active", False):
            st.warning("📍 Could not auto-generate the operator table for this location. "
                       "Please upload a Carrier Table (CSV or JSON) to continue.")
            up = st.file_uploader("Upload Carrier Table (CSV or JSON)", type=["csv", "json"])
            if up is not None:
                try:
                    if up.name.lower().endswith('.json'):
                        uploaded_df = pd.DataFrame(json.load(io.TextIOWrapper(up)))
                    else:
                        uploaded_df = pd.read_csv(up)
                    # Persist a normalized JSON copy, then adopt
                    try:
                        c = _norm_name(city); s = _norm_name(state)
                        fname = f"operator_table_{c}_{s}.json"
                        os.makedirs(local_out_dir or ".", exist_ok=True)
                        local_json = os.path.join(local_out_dir or ".", fname)
                        uploaded_df.to_json(local_json, orient='records', indent=2)
                    except Exception:
                        local_json = None

                    if local_json and os.path.exists(local_json):
                        _adopt_operator_table(local_json, city=city, region=state)
                    else:
                        # Adopt directly from DataFrame (no file path available)
                        st.session_state["operator_table_df"] = uploaded_df
                        st.session_state["operator_table_source"] = "<manual upload>"
                        st.session_state["active_city"] = city
                        st.session_state["active_region"] = state
                        st.session_state["operator_table_rev"] = st.session_state.get("operator_table_rev", 0) + 1
                        st.session_state["carrier_guard_active"] = False
                        try:
                            st.cache_data.clear()
                            st.cache_resource.clear()
                        except Exception:
                            pass
                        st.success(f"✅ Adopted manual table for {city}, {state} (rev {st.session_state['operator_table_rev']})")
                        st.rerun()

                except Exception as e:
                    st.error(f"Failed to parse table: {e}")

        if st.session_state.get("carrier_guard_active", False):
            st.stop()

        # Always return the current session copy (if any)
        return st.session_state.get("operator_table_df")

# =====================================================================================
#                                 Agent 2: Analysis
# =====================================================================================
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


class AnalysisAgent:
    def __init__(self, helper: TinySAHelper, provider: ModelProvider):
        self.helper = helper
        self.provider = provider

    @staticmethod
    def _sweep_overlaps_wifi(freqs) -> bool:
        """
        Return True if the sweep overlaps standard Wi-Fi bands.
        Accepts a sequence of frequencies in Hz, MHz, or GHz (numeric).
        Heuristic:
          - if max <= 10     -> treat as GHz
          - elif max <= 10000-> treat as MHz
          - else             -> treat as Hz
        """
        try:
            vals = [float(x) for x in freqs]
            vmin = min(vals); vmax = max(vals)
        except Exception:
            return False

        # Normalize to Hz by heuristic
        if vmax <= 10:
            vmin *= 1e9; vmax *= 1e9
        elif vmax <= 10000:
            vmin *= 1e6; vmax *= 1e6

        # Wi-Fi bands (Hz)
        wifi_bands = [
            (2.400e9, 2.500e9),   # 2.4 GHz
            (5.150e9, 5.925e9),   # 5 GHz
            (5.925e9, 7.125e9),   # 6 GHz
        ]
        for lo, hi in wifi_bands:
            if not (vmax < lo or vmin > hi):  # overlap
                return True
        return False

    # ========= Robust Wi-Fi helpers =========
    @staticmethod
    def _wifi_freq_to_mhz(v) -> Optional[int]:
        try:
            f = float(v)
        except Exception:
            return None
        # pywifi usually returns MHz; some drivers return Hz
        if f > 1e6 and f < 1e5:  # guard impossible; keep standard
            pass
        if f > 1e6 and f < 1e7:
            return int(round(f))           # already MHz (e.g., 2412)
        if f > 1e9:                        # Hz (e.g., 2.412e9)
            return int(round(f / 1e6))
        if f > 100 and f < 10000:          # MHz but float
            return int(round(f))
        return None

    @staticmethod
    def _wifi_channel_from_mhz(mhz: int) -> Optional[int]:
        if mhz is None:
            return None
        # 2.4 GHz
        if 2412 <= mhz <= 2472:
            return int((mhz - 2407) / 5)   # 1..13
        if mhz == 2484:
            return 14
        # 5 GHz (common center freqs)
        if 5180 <= mhz <= 5825:
            return int((mhz - 5000) / 5)   # standard derivation
        # 6 GHz (Wi-Fi 6E/7 SUB-1)
        if 5955 <= mhz <= 7115:
            # Channel 1 at 5955 with 5 MHz step
            return int(((mhz - 5955) / 5) + 1)
        return None

    @staticmethod
    def _wifi_band_from_mhz(mhz: int) -> str:
        if mhz is None: return "Unknown"
        if 2400 <= mhz <= 2500: return "2.4 GHz"
        if 5150 <= mhz <= 5925: return "5 GHz"
        if 5925 <= mhz <= 7125: return "6 GHz"
        return "Unknown"

    @staticmethod
    def _wifi_is_dfs(ch: Optional[int]) -> str:
        if ch is None: return "No"
        try:
            c = int(ch)
        except Exception:
            return "No"
        return "Yes" if (52 <= c <= 64) or (100 <= c <= 144) else "No"

    @staticmethod
    def _wifi_radio_type(band: str) -> str:
        b = (band or "").lower()
        if b == "2.4 ghz": return "802.11b/g/n"
        if b == "5 ghz":   return "802.11a/n/ac/ax"
        if b == "6 ghz":   return "802.11ax/802.11be"
        return "Unknown"

    @staticmethod
    def _wifi_bw_estimate(ch: Optional[int], radio: str) -> str:
        # conservative guess
        if ch is None:
            return "Unknown"
        r = (radio or "").lower()
        if "2.4" in r or "b/g/n" in r:
            return "20/40 MHz"
        if "ac" in r or "ax" in r or "be" in r:
            return "20/40/80/160 MHz"
        return "20 MHz"

    def _scan_wifi_table(self) -> pd.DataFrame:
        # Safe wrapper; if pywifi is missing or fails, return empty with all columns.
        cols = ["SSID","Signal (dBm)","Frequency (MHz)","Channel","Band","Radio","Bandwidth","DFS"]
        if not PYWIFI_AVAILABLE:
            return pd.DataFrame(columns=cols)

        try:
            import pywifi
            wifi = pywifi.PyWiFi()
            ifaces = wifi.interfaces()
            if not ifaces:
                return pd.DataFrame(columns=cols)
            iface = ifaces[0]
            iface.scan()
            time.sleep(3)
            results = iface.scan_results()
        except Exception:
            return pd.DataFrame(columns=cols)

        rows = []
        for net in results:
            ssid = getattr(net, "ssid", "") or "<Hidden>"
            signal = getattr(net, "signal", None)
            raw_freq = getattr(net, "freq", None)

            mhz = self._wifi_freq_to_mhz(raw_freq)
            ch  = self._wifi_channel_from_mhz(mhz)
            band = self._wifi_band_from_mhz(mhz)
            radio = self._wifi_radio_type(band)
            bw = self._wifi_bw_estimate(ch, radio)
            dfs = self._wifi_is_dfs(ch)

            rows.append({
                "SSID": ssid,
                "Signal (dBm)": signal if signal is not None else "",
                "Frequency (MHz)": mhz if mhz is not None else "",
                "Channel": ch if ch is not None else "",
                "Band": band,
                "Radio": radio,
                "Bandwidth": bw,
                "DFS": dfs,
            })

        df = pd.DataFrame(rows, columns=cols)
        # Sort strongest first if available
        if "Signal (dBm)" in df.columns and not df["Signal (dBm)"].isna().all():
            df = df.sort_values(by="Signal (dBm)", ascending=False, na_position="last")
        return df

    # ---------- Agentic analysis pipeline ----------
    def run(self, def_dict: Dict, override_operator_table: Optional[pd.DataFrame]):
        st.header("3) Rapid Edge Analysis: Spectrum and PCAP Analysis Agents")

        # Chat stage 1
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        prompt = st.chat_input("Ask Ennoia:")

        if prompt:
            t = Timer(); t.start()
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            system_prompt = self.helper.get_system_prompt()
            few_shots = self.helper.get_few_shot_examples()
            chat1 = ([{"role": "system", "content": system_prompt}] +
                     few_shots +
                     [{"role": "user", "content": prompt}])

            with st.chat_message("assistant"):
                response = self.provider.chat(chat1)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            # Stage 2: structured API options
            sys2 = self.provider.map_api.get_system_prompt(def_dict, prompt)
            few2 = self.provider.map_api.get_few_shot_examples()
            chat2 = ([{"role": "system", "content": sys2}] +
                     few2 +
                     [{"role": "user", "content": prompt}])

            api_str = self.provider.chat(chat2)
            api_dict = dict(def_dict); api_dict["save"] = True
            try:
                parsed = json.loads(api_str)
                if isinstance(parsed, dict):
                    api_dict = parsed; api_dict["save"] = True
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(api_str)
                    if isinstance(parsed, dict):
                        api_dict = parsed; api_dict["save"] = True
                except Exception:
                    print("Failed to parse model options. Using defaults.")

            opt = SimpleNamespace(**api_dict)
            gcf = self.helper.configure_tinySA(opt)
            st.pyplot(gcf)

            # Cellular peak analysis
            try:
                result = self.helper.read_signal_strength('max_signal_strengths.csv')
                if not result:
                    st.error("Could not read signal strength data.")
                else:
                    sstr, freq = result
                    freq_mhz = [x / 1e6 for x in freq]

                    # Operator table (override from LocationAgent if present)
                    if override_operator_table is not None:
                        operator_table_df = override_operator_table
                    else:
                        operator_table_df = self.helper.get_operator_frequencies()

                    if operator_table_df is None or (hasattr(operator_table_df, 'empty') and operator_table_df.empty):
                        st.error("Operator table could not be loaded.")
                    else:
                        # Device capability filter
                        st.subheader("📱 Device / UE Capability")
                        devices = {
                            "(none)": {},
                            "Sierra EM9191": {
                                "nr_bands": ["n1","n2","n3","n5","n7","n8","n12","n20","n28","n38","n40","n41","n66","n71","n77","n78","n79"],
                                "lte_bands": ["1","2","3","4","5","7","8","12","13","14","17","20","25","26","28","29","30","38","39","40","41","42","46","48","66","71"],
                                "max_bw_mhz": 100,
                                "mimo": "4x4"
                            },
                            "Quectel EC25": {
                                "nr_bands": [],
                                "lte_bands": ["1","3","5","7","8","20","28","38","40","41"],
                                "max_bw_mhz": 20,
                                "mimo": "2x2"
                            },
                            "Pixel 7 Pro (Global)": {
                                "nr_bands": ["n1","n2","n3","n5","n7","n8","n12","n20","n28","n40","n41","n66","n71","n77","n78"],
                                "lte_bands": ["1","2","3","4","5","7","8","12","17","20","28","38","40","41","66","71"],
                                "max_bw_mhz": 100,
                                "mimo": "4x4"
                            },
                            "iPhone 15 Pro / Pro Max (Global)": {
                                "nr_bands": ["n1","n2","n3","n5","n7","n8","n12","n14","n20","n25","n26","n28","n38","n40","n41","n48","n53","n66","n70","n71","n77","n78","n79"],
                                "lte_bands": ["1","2","3","4","5","7","8","12","13","14","17","18","19","20","25","26","28","29","30","32","38","39","40","41","42","46","48","66","71"],
                                "max_bw_mhz": 100,
                                "mimo": "4x4"
                            },
                            "Galaxy S24 Ultra (Global)": {
                                "nr_bands": ["n1","n2","n3","n5","n7","n8","n20","n28","n38","n40","n41","n66","n71","n75","n76","n77","n78","n79"],
                                "lte_bands": ["1","2","3","4","5","7","8","12","13","17","18","19","20","26","28","32","38","39","40","41","42","46","48","66","71"],
                                "max_bw_mhz": 100,
                                "mimo": "4x4"
                            }
                        }
                        sel = st.selectbox("Select a device profile", list(devices.keys()), index=0)
                        device_caps = devices.get(sel) or None

                        operator_rows = operator_table_df.to_dict('records') if hasattr(operator_table_df, 'to_dict') else operator_table_df

                        if device_caps:
                            import re as _re
                            def extract_bands(text: str):
                                text = text or ""
                                nums = _re.findall(r"Band\s*(\d+)", text)
                                nrs = _re.findall(r"(n\d+)", text)
                                out = []; out += nums; out += nrs
                                return list(dict.fromkeys(out))
                            nr_supported = set(map(str.lower, device_caps.get("nr_bands", [])))
                            lte_supported = set(device_caps.get("lte_bands", []))
                            filtered = []
                            for r in operator_rows:
                                bands = extract_bands(r.get("3GPP Band", ""))
                                keep = False
                                for b in bands:
                                    if b.startswith('n'):
                                        if b.lower() in nr_supported:
                                            keep = True
                                    else:
                                        if b in lte_supported:
                                            keep = True
                                if keep:
                                    filtered.append(r)
                            if not filtered:
                                st.warning("No bands from the local table are supported by the selected device.")
                            else:
                                st.caption(f"Filtered by {sel}: {len(filtered)} rows")
                            operator_rows = filtered or operator_rows

                        st.subheader("🗼 Cellular Analysis")
                        frequency_report_out = self.helper.analyze_signal_peaks(sstr, freq_mhz, operator_rows)
                        st.session_state.step3_done = True
                        if not frequency_report_out:
                            st.info("No strong trained cellular band seen.")
                        else:
                            df = pd.DataFrame(frequency_report_out)
                            st.dataframe(df, use_container_width=True)

                    # Wi-Fi scan when sweep overlaps Wi-Fi ranges
                    # if self._sweep_overlaps_wifi(freq):
                        # st.subheader("📶 Wi-Fi Networks (local scan)")
                        # if not PYWIFI_AVAILABLE:
                            # st.warning("pywifi not available in this environment; skipping Wi-Fi scan.")
                        # else:
                            # dfw = self._scan_wifi_table()
                            # if dfw.empty:
                                # st.warning("No Wi-Fi networks found or pywifi not available.")
                            # else:
                                # st.dataframe(dfw, use_container_width=True)
                                # st.download_button("📥 Download Wi-Fi CSV", dfw.to_csv(index=False), "wifi_scan.csv", "text/csv")
                                
                            # # dfw = self._scan_wifi()
                            # # if dfw.empty:
                                # # st.warning("No Wi-Fi networks found.")
                            # # else:
                                # # st.success(f"Found {len(dfw)} networks.")
                                # # st.dataframe(dfw, use_container_width=True)
                                # # st.download_button("📥 Download Wi-Fi CSV", data=dfw.to_csv(index=False),
                                                   # # file_name="wifi_scan.csv", mime="text/csv")
                    if any(x >= 2.39e9 for x in freq):
                        df = scan_wifi()
                        st.subheader("📶 List of Available WiFi Networks")
                        st.caption("Below are the scanned WiFi networks, including signal strength, frequency, and estimated bandwidth.")
                        if df.empty:
                            st.warning("No networks found.")
                        else:
                            st.success(f"Found {len(df)} networks.")
                            st.dataframe(df)
                            st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="wifi_scan.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Failed to process analysis: {e}")

            t.stop()
            st.write(f"elapsed: {fmt_seconds(t.elapsed())}")

# Step 4: O-RAN U-Plane PCAP → CSV Converter
def oran_pcap_to_csv_step():
    #st.title("📡 O-RAN U-Plane PCAP to CSV Converter")
    #st.header("4) REA PCAP Agent")

    uploaded_file = st.file_uploader("Transfer O-RAN Fronthaul PCAP", type=["pcap", "pcapng"])
    if uploaded_file:
        with open("temp.pcap", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Transferring and processing..."):
            try:
                res = requests.post("http://localhost:8010/upload", files={"pcap": open("temp.pcap", "rb")})
                if res.ok:
                    data = res.json()
                    csv_path = data["output_file"]
                    st.success("✅ Converted successfully!")
                    df = pd.read_csv(csv_path)

                    st.dataframe(df.head())

                    port_id = st.selectbox("Select Port", sorted(df['Port'].unique()))
                    subframe = st.selectbox("Select Subframe", sorted(df['Subframe'].unique()))
                    slot = st.selectbox("Select Slot", sorted(df['Slot'].unique()))
                    symbol = st.selectbox("Select Symbol", sorted(df['Symbol'].unique()))

                    filtered = df[
                        (df["Port"] == port_id) &
                        (df["Subframe"] == subframe) &
                        (df["Slot"] == slot) &
                        (df["Symbol"] == symbol)
                    ]

                    st.line_chart({"I": filtered["Real"], "Q": filtered["Imag"]})
                    # Load the CSV file
                    df = pd.read_csv(csv_path, header=None)

                    # Drop the first row
                    df = df.iloc[1:, :]

                    # Drop the first 5 columns
                    df = df.iloc[:, 5:]

                    # Save to a new CSV
                    df.to_csv(csv_path, index=False, header=False)
                    #import zipfile

                    #with zipfile.ZipFile('csv.zip', 'w') as zipf:
                    #    zipf.write(csv_path)  # Replace with your file name

                else:
                    st.error(f"❌ Failed to process PCAP: {res.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

def nav_buttons():
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("⬅️ Back", use_container_width=True, disabled=st.session_state.current_step == 1):
            st.session_state.current_step -= 1
            st.rerun()
    with col2:
        # Gate the Next button when on step 3
        next_disabled = (
            (st.session_state.current_step == 3 and not st.session_state.step3_done)
        )
        label = "➡️ Next" if not next_disabled else "➡️ Next (finish Step 3 first)"
        if st.button(label, use_container_width=True, disabled=next_disabled):
            st.session_state.current_step = 4
            st.rerun()
            
ENNOIA_SSH_HOST="98.84.113.163"
ENNOIA_SSH_USER="ec2-user"
ENNOIA_SSH_KEY=r"C:\Users\rices\ennoiaCAT\AWS-Hackathon.pem"
ENNOIA_REMOTE_IN="/home/ec2-user/ennoiaCAT/csv_files"
ENNOIA_LOCAL_OUT=r"C:\Users\rices\ennoiaCAT"
ENNOIA_REMOTE_OUT= "/home/ec2-user/ennoiaCAT/csv_files/out"
def sftp_upload_atomic(local_path: str, remote_dir: str) -> str:
    import os, time, paramiko

    base = os.path.basename(local_path)
    remote_tmp = f"{remote_dir}/{base}.part"
    remote_final = f"{remote_dir}/{base}"
    ready_tmp = f"{remote_final}.ready.part"
    ready_final = f"{remote_final}.ready"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=ENNOIA_SSH_HOST,
        username=ENNOIA_SSH_USER,
        key_filename=ENNOIA_SSH_KEY,
        timeout=20
    )

    sftp = ssh.open_sftp()
    try:
        # Ensure target directory exists
        try:
            sftp.listdir(remote_dir)
        except IOError:
            sftp.mkdir(remote_dir)

        # ✅ Step 1: remove old file and flag if they exist
        for path in [remote_final, ready_final, remote_tmp, ready_tmp]:
            try:
                sftp.remove(path)
                print(f"🧹 Removed old file: {path}")
            except IOError:
                pass  # file didn’t exist, fine

        # ✅ Step 2: upload as .part, then rename atomically
        print(f"⬆️ Transferring {local_path} → {remote_final}")
        sftp.put(local_path, remote_tmp)
        sftp.rename(remote_tmp, remote_final)

        # ✅ Step 3: optional short delay for stability
        time.sleep(1)

        # ✅ Step 4: create .ready flag (atomic rename)
        with sftp.file(ready_tmp, "w") as flag:
            flag.write("ready\n")
        sftp.rename(ready_tmp, ready_final)

        print(f"✅ Transferred {remote_final} and created {ready_final}")
    finally:
        sftp.close()
        ssh.close()

    return remote_final

def trigger_flask(remote_path: str):
    payload = {"path": remote_path, "token": PROCESS_TOKEN}
    r = requests.post(FLASK_URL, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

# Example integration after you generate/write the CSV:
def send_csv_to_ec2_and_process(local_csv_path: str):
    st.info("Transferring from Edge to AWS…")
    remote_path = sftp_upload_atomic(local_csv_path, ENNOIA_REMOTE_IN)
    st.success(f"Successfully transferred to {remote_path}")

    # st.info("Triggering Flask processing…")
    # resp = trigger_flask(remote_path)
    # st.success(f"Flask accepted job: {resp}")

def sftp_connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=ENNOIA_SSH_HOST,
        username=ENNOIA_SSH_USER,
        key_filename=ENNOIA_SSH_KEY,
        timeout=25
    )
    return ssh, ssh.open_sftp()

def wait_remote_file(sftp, path, stable_seconds=1.0, timeout=600):
    """Wait for remote file to exist and its size to stop changing."""
    end = time.time() + timeout
    last = -1
    last_change = time.time()
    while time.time() < end:
        try:
            attr = sftp.stat(path)
            size = attr.st_size
            if size != last:
                last = size
                last_change = time.time()
            else:
                if time.time() - last_change >= stable_seconds:
                    return True
        except IOError:
            pass
        time.sleep(0.3)
    return False

def sftp_upload_atomic_with_ready(local_csv_path: str) -> str:
    """Upload CSV as .part → rename, then create .ready to trigger EC2 watcher.
       Returns the *base filename* used on EC2 (e.g., data.csv)."""
    base = os.path.basename(local_csv_path)
    remote_tmp   = f"{ENNOIA_REMOTE_IN}/{base}.part"
    remote_final = f"{ENNOIA_REMOTE_IN}/{base}"
    ready_tmp    = f"{remote_final}.ready.part"
    ready_final  = f"{remote_final}.ready"

    ssh, sftp = sftp_connect()
    try:
        # ensure inbox exists; clean leftovers for same-name re-runs
        try: sftp.listdir(ENNOIA_REMOTE_IN)
        except IOError: sftp.mkdir(ENNOIA_REMOTE_IN)
        for p in (remote_final, remote_tmp, ready_final, ready_tmp):
            try: sftp.remove(p)
            except IOError: pass

        # atomic upload
        sftp.put(local_csv_path, remote_tmp)
        sftp.rename(remote_tmp, remote_final)

        # create .ready atomically
        f = sftp.file(ready_tmp, "w")
        f.write("ready\n")
        f.close()
        sftp.rename(ready_tmp, ready_final)
    finally:
        sftp.close(); ssh.close()

    return base

def fetch_report_when_ready(base_filename: str, local_dir: str = ".") -> str:
    """Waits for <base>.report.md.done on EC2, then downloads <base>.report.md."""
    remote_report = f"{ENNOIA_REMOTE_OUT}/{base_filename}.report.md"
    remote_done   = remote_report + ".done"
    local_report  = os.path.join(local_dir, f"{base_filename}.report.md")

    ssh, sftp = sftp_connect()
    try:
        # wait for done flag written by EC2 watcher after saving genreport
        ok = wait_remote_file(sftp, remote_done, stable_seconds=0.5, timeout=900)
        if not ok:
            raise TimeoutError("Timed out waiting for report completion on AWS")

        # (optional) ensure report file itself is stable
        ok = wait_remote_file(sftp, remote_report, stable_seconds=0.5, timeout=30)
        if not ok:
            raise TimeoutError("Report file did not stabilize on AWS")

        # download report
        sftp.get(remote_report, local_report)

        # optional: clear the .done flag so the next run is clean
        try: sftp.remove(remote_done)
        except IOError: pass
    finally:
        sftp.close(); ssh.close()

    return local_report
# =====================================================================================
#                                         App
# =====================================================================================
st.session_state.step3_done = False

def main():
    helper = TinySAHelper()
    provider = ModelProvider(mode='LLM', helper=helper)

    # 1) Configuration
    st.session_state.current_step = 1
    config_agent = ConfigurationAgent(helper, provider)
    selected_options, def_dict = config_agent.run()

    # 2) Location: SSH build & SFTP fetch → override operator table
    st.session_state.current_step = 2
    location_agent = LocationAgent(helper)
    override_operator_table = location_agent.detect_and_gate()

    # 3) Analysis
    st.session_state.current_step = 3
    analysis_agent = AnalysisAgent(helper, provider)
    analysis_agent.run(def_dict, override_operator_table)
    #nav_buttons()
    #print (st.session_state.current_step)
    #4) PCAP Analysis
    #if st.session_state.step3_done == True:
    oran_pcap_to_csv_step()
    
    # ───────────── Start button gate ─────────────
    with st.form("ec2_start_form", clear_on_submit=False):
        st.caption("Click **Start** to transfer the trimmed output to AWS and trigger processing.")
        start_clicked = st.form_submit_button("Start", use_container_width=True)

    if start_clicked:
        # Prevent accidental double-submits
        if st.session_state.get("ec2_upload_in_progress"):
            st.info("Transfer already in progress…")
            return

        st.session_state["ec2_upload_in_progress"] = True
        send_csv_to_ec2_and_process(r"outputs\temp.pcap.csv")
        st.session_state["ec2_upload_in_progress"] = False
        st.success("🚀 AWS transfer initiated.")
    
    #st.info("Waiting for EC2 report…")
    try:
        local_report_path = fetch_report_when_ready("temp.pcap.csv", local_dir=".")
        st.success("Report received from AWS ✅")

        # Show the report text
        with open(local_report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        #st.text(report_text)
        st.markdown(report_text, unsafe_allow_html=True)

        # Offer download
        with open(local_report_path, "rb") as f:
            st.download_button(
                "Download report.md",
                data=f,
                file_name=os.path.basename(local_report_path),
                mime="text/markdown"
            )
    except Exception as e:
        st.error(f"Failed to fetch report: {e}")
    

if __name__ == "__main__":
    main()
