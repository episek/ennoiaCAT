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
st.sidebar.image('ennoia.jpg')
st.title("Ennoia Technologies")
st.markdown("""
Chat and Test with Ennoia Connect Platform ©. All rights reserved.
""")

if not success:
    st.error("Ennoia License verification failed. Please check your license key or contact support.")
    st.stop()
else:
    st.success("Ennoia License verified successfully.")

# Clear caches on hot-reload to avoid stale states during development
st.cache_data.clear()
st.cache_resource.clear()

# ---- session-state init (place once, before the UI) ----
if "selected_device" not in st.session_state:
    # try to restore last choice from disk
    try:
        with open("last_device.json", "r", encoding="utf-8") as f:
            st.session_state["selected_device"] = json.load(f).get("selected_device", "(none)")
    except Exception:
        st.session_state["selected_device"] = "(none)"

st.session_state.setdefault("device_caps", {})
st.session_state.setdefault("operator_table_df", None)

# ---------- Helper: explode operator table to one row per Operator × Band ----------
import re as _re

def explode_operators_and_bands(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a table where a row may include multiple operators and combined 3GPP band text
    into legacy format: one row per Operator × Band with Band Type and normalized band.
    """
    if df_in is None or (isinstance(df_in, pd.DataFrame) and df_in.empty):
        return df_in

    df = df_in.copy()
    for col in ["3GPP Band","Operators","Uplink Frequency (MHz)","Downlink Frequency (MHz)","Bandwidth","Technology"]:
        if col not in df.columns:
            df[col] = ""

    rows = []
    for _, r in df.iterrows():
        ops_val = r.get("Operators", "")
        if isinstance(ops_val, list):
            ops = [str(x).strip() for x in ops_val if str(x).strip()]
        else:
            ops = [o.strip() for o in str(ops_val).split(",") if o.strip()]
        if not ops:
            ops = ["<Unknown>"]

        gpp = str(r.get("3GPP Band",""))
        lte_list = _re.findall(r"\b(?:Band|B)\s*(\d+)\b", gpp, flags=_re.I)
        nr_list  = [b.lower() for b in _re.findall(r"\b(n\d+)\b", gpp, flags=_re.I)]

        base = r.to_dict()
        for op in ops:
            for b in lte_list:
                row = dict(base)
                row["Operator"] = op
                row["Band Type"] = "LTE"
                row["Band"] = str(b)
                row["3GPP Band (norm)"] = f"Band {b}"
                rows.append(row)
            for nb in nr_list:
                row = dict(base)
                row["Operator"] = op
                row["Band Type"] = "NR"
                row["Band"] = nb
                row["3GPP Band (norm)"] = nb
                rows.append(row)
        if not lte_list and not nr_list:
            for op in ops:
                row = dict(base)
                row["Operator"] = op
                row["Band Type"] = "Unknown"
                row["Band"] = gpp
                row["3GPP Band (norm)"] = gpp
                rows.append(row)

    out = pd.DataFrame(rows)
    preferred = [
        "Operator","Band","Band Type","3GPP Band (norm)","3GPP Band",
        "Uplink Frequency (MHz)","Downlink Frequency (MHz)","Bandwidth","Technology",
    ]
    tail = [c for c in out.columns if c not in preferred]
    return out[[c for c in preferred if c in out.columns] + tail]

import re
import pandas as pd

import re

def explode_analysis_results_unified(rows):
    """
    Expand analyzer output to one row per Operator×Band with a single 'Band' column.
    Accepts:
      - list[dict] (preferred)
      - pandas.DataFrame (will be converted to list-of-dicts)
      - None / empty
    """
    # ---- normalize input safely ----
    if rows is None:
        return []
    if isinstance(rows, pd.DataFrame):
        if rows.empty:
            return []
        rows_list = rows.to_dict("records")
    elif isinstance(rows, list):
        if len(rows) == 0:
            return []
        rows_list = rows
    else:
        # unknown type -> nothing to do
        return []

    out = []
    for r in rows_list:
        base = dict(r)
        gpp_text = str(r.get("3GPP Band") or r.get("Matched Band") or r.get("Band") or "")

        # Parse bands
        ltes = re.findall(r"\b(?:Band|B)\s*(\d+)\b", gpp_text, flags=re.I)
        nrs  = [b.lower() for b in re.findall(r"\b(n\d+)\b", gpp_text, flags=re.I)]

        # Parse operators
        ops_val = r.get("Operators", r.get("Operator", ""))
        if isinstance(ops_val, list):
            ops = [str(x).strip() for x in ops_val if str(x).strip()]
        else:
            ops = [o.strip() for o in str(ops_val).split(",") if o.strip()]
        if not ops:
            ops = ["<Unknown>"]

        peak_mhz  = r.get("Peak Frequency (MHz)") or r.get("Peak (MHz)") or r.get("PeakFreqMHz")
        power_dbm = r.get("Power (dBm)") or r.get("Peak Power (dBm)") or r.get("PowerdBm")
        conf      = r.get("Confidence") or r.get("Score") or r.get("Match Score")
        notes     = r.get("Notes") or r.get("Comment")
        rng       = r.get("Frequency Range (MHz)") or r.get("Range (MHz)")

        # LTE rows
        for op in ops:
            for b in ltes:
                row = dict(base)
                row["Operator"] = op
                row["Band Type"] = "LTE"
                row["Band"] = str(b)
                row["3GPP Band (norm)"] = f"Band {b}"
                row["Peak (MHz)"] = peak_mhz
                row["Power (dBm)"] = power_dbm
                row["Confidence"] = conf
                row["Range (MHz)"] = rng
                row["Notes"] = notes
                out.append(row)

        # NR rows
        for op in ops:
            for nb in nrs:
                row = dict(base)
                row["Operator"] = op
                row["Band Type"] = "NR"
                row["Band"] = nb
                row["3GPP Band (norm)"] = nb
                row["Peak (MHz)"] = peak_mhz
                row["Power (dBm)"] = power_dbm
                row["Confidence"] = conf
                row["Range (MHz)"] = rng
                row["Notes"] = notes
                out.append(row)

        # Generic if no bands parsed
        if not ltes and not nrs:
            for op in ops:
                row = dict(base)
                row["Operator"] = op
                row["Band Type"] = "Unknown"
                row["Band"] = gpp_text.strip()
                row["3GPP Band (norm)"] = gpp_text
                row["Peak (MHz)"] = peak_mhz
                row["Power (dBm)"] = power_dbm
                row["Confidence"] = conf
                row["Range (MHz)"] = rng
                row["Notes"] = notes
                out.append(row)

    # Order columns
    preferred = [
        "Operator","Band","Band Type","3GPP Band (norm)","3GPP Band",
        "Peak (MHz)","Power (dBm)","Confidence","Range (MHz)","Notes",
    ]
    if not out:
        return []
    cols = list(out[0].keys())
    tail = [c for c in cols if c not in preferred]
    final_cols = [c for c in preferred if c in cols] + tail
    return [{k: rec.get(k, "") for k in final_cols} for rec in out]    
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
            st.write("\n⏳ Working in OFFLINE mode. Loading local model... (might take a minute)")

            @st.cache_resource(show_spinner=False)
            def _load_local():
                return self.helper.load_lora_model()

            self.tokenizer, self.peft_model, self.device = _load_local()
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
            # Local generation via MapAPI
            return self.map_api.generate_response(messages)
        else:
            # OpenAI
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
#                                 Agent 1: Configuration
# =====================================================================================
class ConfigurationAgent:
    def __init__(self, helper: TinySAHelper, provider: ModelProvider):
        self.helper = helper
        self.provider = provider
        self.system_prompt = helper.get_system_prompt()
        self.few_shot_examples = helper.get_few_shot_examples()

    @st.cache_data(show_spinner=False)
    def _defaults_from_map_api(_map_api: MapAPI):
        return _map_api.get_defaults_opts()

    def run(self):
        st.header("1) Configuration")
        # UI: select options
        selected_options = TinySAHelper.select_checkboxes()
        st.success(f"You selected: {', '.join(selected_options) if selected_options else 'nothing'}")

        # Model init
        mode = 'SLM' if 'SLM' in selected_options else 'LLM'
        self.provider.mode = mode
        self.provider.init()

        # Cache default options from MapAPI
        def_dict = ConfigurationAgent._defaults_from_map_api(self.provider.map_api)

        # Cache TinySA port
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
    raise RuntimeError(f"SFTP open failed after {retries} attempts: {last}")

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

    with st.status("🔐 Connecting to EC2 via SSH…", expanded=False) as status:
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
            status.update(label=f"SFTP open failed: {e}", state="error")
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

        status.update(label="⬇️ Downloading JSON via SFTP…")
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
        """Try simple IP-based location via public IP APIs (city, country name)."""
        try:
            import urllib.request, json as _json  # noqa: E401
            # ipinfo-style endpoint (no key for coarse city/region). If blocked, will raise.
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
        st.info("Location detection failed or disabled. Please enter your location to continue.")
        col1, col2 = st.columns(2)
        with col1:
            city = st.text_input("City", value="Dallas")
        with col2:
            state = st.text_input("State/Country", value="TX")
        return city.strip(), state.strip()

    def detect_and_gate(self) -> Optional[pd.DataFrame]:
        st.header("2) Location & Carrier Table")

        # SSH settings panel (read defaults from env)
        with st.expander("🔧 Remote builder (SSH→EC2) settings", expanded=False):
            ssh_host = st.text_input("EC2 Public IP / DNS", value=os.getenv("ENNOIA_SSH_HOST", ""))
            ssh_user = st.text_input("SSH Username", value=os.getenv("ENNOIA_SSH_USER", "ec2-user"))
            ssh_key  = st.text_input("Path to .pem (Windows)", value=os.getenv("ENNOIA_SSH_KEY", r"C:\Keys\ennoia-ec2.pem"))
            remote_python = st.text_input("Remote Python", value=os.getenv("ENNOIA_REMOTE_PY", "python3"))
            remote_script = st.text_input("Remote script path", value=os.getenv("ENNOIA_REMOTE_SCRIPT", "/home/ec2-user/operator_table_service.py"))
            remote_out_dir = st.text_input("Remote output dir", value=os.getenv("ENNOIA_REMOTE_OUT", "/tmp/ennoia_tables"))
            local_out_dir = st.text_input("Local output dir", value=os.getenv("ENNOIA_LOCAL_OUT", r"C:\ennoia\tables"))

        # Detect/enter location
        method = st.radio("Location method", ["Auto (IP)", "Manual"], horizontal=True)
        if method == "Auto (IP)":
            detected = self._get_location_ip() or self._location_from_wifi()
            if detected is None:
                st.warning("Auto location failed. Please enter manually.")
                detected = self._manual_location_input()
        else:
            detected = self._manual_location_input()

        # Optional GPS reverse-geocode override
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
        if not already_ran:
            local_path = _try_run()
            st.session_state[key_loc] = True
        else:
            local_path = None

        # Manual trigger if needed
        if st.button("Build operator table via SSH now"):
            local_path = _try_run()

        # If file arrived, load it
        if local_path and os.path.exists(local_path):
            try:
                uploaded_df = pd.read_json(local_path)
                st.success(f"Loaded carrier table from {local_path}")
                st.session_state["carrier_guard_active"] = False
            except Exception as e:
                st.error(f"Failed to parse downloaded JSON: {e}")
                st.session_state.setdefault("carrier_guard_active", True)

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
                    # Optionally cache locally with normalized name
                    try:
                        c = _norm_name(city); s = _norm_name(state)
                        fname = f"operator_table_{c}_{s}.json"
                        os.makedirs(local_out_dir or ".", exist_ok=True)
                        uploaded_df.to_json(os.path.join(local_out_dir or ".", fname), orient='records', indent=2)
                        st.caption(f"Saved to {os.path.join(local_out_dir or '.', fname)}")
                    except Exception:
                        pass
                    st.success("Carrier table uploaded.")
                    st.session_state["carrier_guard_active"] = False
                except Exception as e:
                    st.error(f"Failed to parse carrier table: {e}")

        if st.session_state.get("carrier_guard_active", False):
            st.stop()

        return uploaded_df  # None => analyzer will use Dallas/defaults (if implemented on your side)

# =====================================================================================
#                                 Agent 2: Analysis
# =====================================================================================


class AnalysisAgent:
    def __init__(self, helper: TinySAHelper, provider: ModelProvider):
        self.helper = helper
        self.provider = provider

    # ---------- Wi-Fi helpers ----------
    @staticmethod
    def _freq_to_channel(freq_mhz: float) -> Optional[int]:
        try:
            freq = int(freq_mhz)
            if freq == 2484:
                return 14
            elif 2412 <= freq <= 2472:
                return (freq - 2407) // 5
            elif 5180 <= freq <= 5825:
                return (freq - 5000) // 5
            elif 5955 <= freq <= 7115:
                return (freq - 5950) // 5 + 1
        except Exception:
            pass
        return None

    @staticmethod
    def _classify_band(freq_mhz: float) -> Optional[str]:
        try:
            freq = int(freq_mhz)
            if 2400 <= freq <= 2500:
                return "2.4 GHz"
            elif 5000 <= freq <= 5900:
                return "5 GHz"
            elif 5925 <= freq <= 7125:
                return "6 GHz"
            else:
                return "Unknown"
        except Exception:
            pass
        return None

    @staticmethod
    def _is_dfs_channel(ch: Optional[int]) -> bool:
        if ch is None:
            return False
        try:
            ch = int(ch)
        except Exception:
            return False
        return (52 <= ch <= 64) or (100 <= ch <= 144)

    @staticmethod
    def _infer_bw(ch: Optional[int], radio_type: str) -> str:
        if ch is None:
            return "Unknown"
        try:
            ch = int(ch)
        except Exception:
            return "Unknown"
        rt = (radio_type or "").lower()
        if ch <= 14:
            return "20/40 MHz"
        elif 36 <= ch <= 144 or 149 <= ch <= 165:
            if ("ac" in rt) or ("ax" in rt):
                return "20/40/80/160 MHz"
            else:
                return "20/40 MHz"
        elif 1 <= ch <= 233:
            return "20/40/80/160 MHz" if "ax" in rt else "20 MHz"
        else:
            return "Unknown"

    def _sweep_overlaps_wifi(self, freqs) -> bool:
        """Return True if sweep overlaps standard Wi‑Fi bands. Handles Hz or MHz inputs."""
        try:
            vals = [float(x) for x in freqs]
            vmin = min(vals); vmax = max(vals)
        except Exception:
            return False
        # Heuristic: if values look like MHz, convert to Hz
        if vmax < 10000:  # very likely MHz
            vmin *= 1e6; vmax *= 1e6
        # Wi‑Fi bands (Hz)
        bands = [
            (2.400e9, 2.500e9),   # 2.4 GHz
            (5.150e9, 5.925e9),   # 5 GHz typical
            (5.925e9, 7.125e9),   # 6 GHz
        ]
        for lo, hi in bands:
            if not (vmax < lo or vmin > hi):
                return True
        return False

    def _scan_wifi(self) -> pd.DataFrame:
        if not PYWIFI_AVAILABLE:
            return pd.DataFrame(columns=[
                "SSID", "Signal (dBm)", "Frequency (MHz)", "Channel", "Band",
                "Radio Type (Estimated)", "Bandwidth (Estimated)", "DFS Channel"
            ])
        wifi = pywifi.PyWiFi()
        iface = wifi.interfaces()[0]
        iface.scan()
        time.sleep(3)
        results = iface.scan_results()

        networks = []
        for net in results:
            ssid = net.ssid or "<Hidden>"
            signal = getattr(net, 'signal', None)
            freq = getattr(net, 'freq', None)
            ch = self._freq_to_channel(freq)
            band = self._classify_band(freq)
            if band == "2.4 GHz":
                radio = "802.11b/g/n"
            elif band == "5 GHz":
                radio = "802.11a/n/ac"
            elif band == "6 GHz":
                radio = "802.11ax"
            else:
                radio = "Unknown"
            bw = self._infer_bw(ch, radio)
            networks.append({
                "SSID": ssid,
                "Signal (dBm)": signal,
                "Frequency (MHz)": freq,
                "Channel": ch,
                "Band": band,
                "Radio Type (Estimated)": radio,
                "Bandwidth (Estimated)": bw,
                "DFS Channel": "Yes" if self._is_dfs_channel(ch) else "No",
            })
        df = pd.DataFrame(networks).sort_values(by="Signal (dBm)", ascending=False)
        return df

    # ---------- Agentic analysis pipeline ----------
    def run(self, def_dict: Dict, override_operator_table: Optional[pd.DataFrame]):
        st.header("3) Analysis (Agentic)")

        # ---- init chat history & input ----
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        prompt = st.chat_input("Ask Ennoia:")

        # ────────────────────────────────────────────────
        # 🧩 KEEP OPERATOR TABLE STABLE ACROSS RERUNS
        # ────────────────────────────────────────────────
        if "operator_table_df" not in st.session_state:
            st.session_state.operator_table_df = (
                override_operator_table
                if override_operator_table is not None
                else self.helper.get_operator_frequencies()
            )
        else:
            if override_operator_table is not None:
                st.session_state.operator_table_df = override_operator_table
        operator_table_df = st.session_state.operator_table_df


        st.markdown("### Carrier Table")

        # ---- Operator table toggle (define once, stable key) ----
        if operator_table_df is None or not isinstance(operator_table_df, pd.DataFrame) or operator_table_df.empty:
            st.warning("No operator table loaded.")
        else:
            # Initialize default once
            st.session_state.setdefault("legacy_operator_rows", True)

            st.checkbox(
                "Show legacy per-operator rows",
                value=st.session_state["legacy_operator_rows"],
                key="legacy_operator_rows"   # stable, reused key
            )

            legacy_mode_tbl = bool(st.session_state["legacy_operator_rows"])

            # Build the table to display
            table_for_display = (
                explode_operators_and_bands(operator_table_df)  # your helper that explodes to per-operator rows
                if legacy_mode_tbl else operator_table_df
            )

            st.dataframe(table_for_display, use_container_width=True)

        # Make analysis use the legacy style too (so results look like before)
        operator_rows = table_for_display.to_dict("records")

        # Optional: export buttons
        st.download_button(
            "📥 Download current table (CSV)",
            data=table_for_display.to_csv(index=False),
            file_name="operator_table_legacy.csv",
            mime="text/csv"
        )
        st.download_button(
            "📥 Download current table (JSON)",
            data=table_for_display.to_json(orient="records", indent=2),
            file_name="operator_table_legacy.json",
            mime="application/json"
        )




        # ────────────────────────────────────────────────
        # 📱 UE SELECTOR (OUTSIDE `if prompt:` SO IT ALWAYS SHOWS)
        # ────────────────────────────────────────────────
        if "selected_device" not in st.session_state:
            # restore last selection if present
            try:
                with open("last_device.json", "r", encoding="utf-8") as f:
                    st.session_state["selected_device"] = json.load(f).get("selected_device", "(none)")
            except Exception:
                st.session_state["selected_device"] = "(none)"
        st.session_state.setdefault("device_caps", {})

        st.subheader("📱 Device / UE Capability")
        devices = {
            "(none)": {},
            "Sierra EM9191": {
                "nr_bands": ["n1","n2","n3","n5","n7","n8","n12","n20","n28","n38","n40","n41","n66","n71","n77","n78","n79"],
                "lte_bands": ["1","2","3","4","5","7","8","12","13","14","17","20","25","26","28","29","30","38","39","40","41","42","46","48","66","71"],
                "max_bw_mhz": 100, "mimo": "4x4"
            },
            "Quectel EC25": {
                "nr_bands": [],
                "lte_bands": ["1","3","5","7","8","20","28","38","40","41"],
                "max_bw_mhz": 20, "mimo": "2x2"
            },
            "Pixel 7 Pro (Global)": {
                "nr_bands": ["n1","n2","n3","n5","n7","n8","n12","n20","n28","n40","n41","n66","n71","n77","n78"],
                "lte_bands": ["1","2","3","4","5","7","8","12","17","20","28","38","40","41","66","71"],
                "max_bw_mhz": 100, "mimo": "4x4"
            },
            "iPhone 16 Pro": {
                "nr_bands": ["n1","n2","n3","n5","n7","n8","n12","n20","n25","n28","n30","n38","n40","n41","n48","n66","n70","n71","n77","n78","n79","n258","n260","n261"],
                "lte_bands": ["1","2","3","4","5","7","8","12","13","17","18","19","20","25","26","28","29","30","32","38","39","40","41","42","46","48","66","71"],
                "max_bw_mhz": 100, "mimo": "4x4"
            },
            "Galaxy S24 Ultra": {
                "nr_bands": ["n1","n2","n3","n5","n7","n8","n12","n20","n25","n28","n30","n38","n40","n41","n48","n66","n70","n71","n77","n78","n79","n258","n260","n261"],
                "lte_bands": ["1","2","3","4","5","7","8","12","13","17","18","19","20","25","26","28","29","30","32","38","39","40","41","42","46","48","66","71"],
                "max_bw_mhz": 100, "mimo": "4x4"
            }
        }

        def _on_device_change():
            sel = st.session_state.get("device_select_box", "(none)")
            st.session_state["selected_device"] = sel
            st.session_state["device_caps"] = devices.get(sel, {})
            try:
                with open("last_device.json", "w", encoding="utf-8") as f:
                    json.dump({"selected_device": sel}, f)
            except Exception:
                pass
            st.rerun()

        sel = st.selectbox(
            "Select a device profile",
            list(devices.keys()),
            index=list(devices.keys()).index(st.session_state.get("selected_device", "(none)")),
            key="device_select_box",
            on_change=_on_device_change
        )
        device_caps = st.session_state.get("device_caps") or None

        # ────────────────────────────────────────────────
        # 🔁 ALWAYS RENDER FROM CACHED SWEEP (EVEN WITHOUT PROMPT)
        # ────────────────────────────────────────────────
        def _extract_and_filter(operator_table_df, device_caps):
            # table → rows
            operator_rows = (operator_table_df.to_dict("records")
                             if hasattr(operator_table_df, "to_dict") else operator_table_df)
            # Robust UE filter on the (possibly) exploded table
            if device_caps:
                nr_supported  = {b.lower() for b in device_caps.get("nr_bands", [])}
                lte_supported = {str(b) for b in device_caps.get("lte_bands", [])}

                filtered = []
                for r in operator_rows:
                    # Prefer normalized fields if present
                    lte = str(r.get("LTE Band", "")).strip()
                    nr  = str(r.get("NR Band", "")).lower().strip()
                    keep = False
                    if lte and lte in lte_supported:
                        keep = True
                    if nr and nr in nr_supported:
                        keep = True

                    # Fallback to original 3GPP Band parsing if normalized fields are missing
                    if not keep:
                        gpp = str(r.get("3GPP Band", ""))
                        ltes = re.findall(r"\b(?:Band|B)\s*(\d+)\b", gpp, flags=re.I)
                        nrs  = [b.lower() for b in re.findall(r"\b(n\d+)\b", gpp, flags=re.I)]
                        if any(b in lte_supported for b in ltes) or any(b in nr_supported for b in nrs):
                            keep = True

                    if keep:
                        filtered.append(r)

                if not filtered:
                    st.warning(f"No bands from the table are supported by: {st.session_state.get('selected_device','(none)')}")
                else:
                    st.caption(f"Filtered by {st.session_state.get('selected_device','(none)')}: {len(filtered)} rows")

                operator_rows = filtered or operator_rows
            return operator_rows

        def _render_from_cache():
            cache = st.session_state.get("last_scan")
            if not cache:
                return
            sstr = cache["sstr"]; freq = cache["freq"]
            freq_mhz = [x / 1e6 for x in freq]
            operator_rows = _extract_and_filter(operator_table_df, device_caps)
            st.subheader("🗼 Cellular Analysis")
            try:
                out = self.helper.analyze_signal_peaks(sstr, freq_mhz, operator_rows)
                if not out:
                    st.info("No strong trained cellular band seen.")
                else:
                    # out -> returned by self.helper.analyze_signal_peaks(...)
                    legacy_rows = explode_analysis_results_unified(out)
                    if not legacy_rows:
                        st.info("No strong trained cellular band seen.")
                    else:
                        df = pd.DataFrame(legacy_rows)
                        sort_cols = [c for c in ["Operator","Band Type","Band","Peak (MHz)","Power (dBm)"] if c in df.columns]
                        if sort_cols:
                            df = df.sort_values(by=sort_cols, ascending=[True, True, True, True, False], na_position="last")
                        st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Failed to render cached analysis: {e}")
            # Optional: redraw previous plot if cached
            if st.session_state.get("last_plot_fig") is not None:
                st.pyplot(st.session_state["last_plot_fig"])

        # render (if a sweep was already done earlier)
        _render_from_cache()

        # ────────────────────────────────────────────────
        # HEAVY PATH (ONLY WHEN USER SENDS A PROMPT)
        # ────────────────────────────────────────────────
        if prompt:
            t = Timer(); t.start()
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Stage 1: free-form chat
            system_prompt = self.helper.get_system_prompt()
            few_shots = self.helper.get_few_shot_examples()
            chat1 = ([{"role": "system", "content": system_prompt}] +
                     few_shots +
                     [{"role": "user", "content": prompt}])
            with st.chat_message("assistant"):
                response = self.provider.chat(chat1)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            # Stage 2: structured options
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
                    st.warning("Failed to parse model options. Using defaults.")

            # Configure tinySA and plot
            opt = SimpleNamespace(**api_dict)
            gcf = self.helper.configure_tinySA(opt)
            st.pyplot(gcf)
            # cache the plot so UE changes can re-display it
            st.session_state["last_plot_fig"] = gcf

            # ---- Cellular peak analysis (NEW SWEEP) ----
            try:
                result = self.helper.read_signal_strength('max_signal_strengths.csv')
                if not result:
                    st.error("Could not read signal strength data.")
                else:
                    sstr, freq = result
                    # cache sweep for future reruns
                    st.session_state["last_scan"] = {"sstr": sstr, "freq": freq, "timestamp": time.time()}

                    freq_mhz = [x / 1e6 for x in freq]
                    operator_rows = _extract_and_filter(operator_table_df, device_caps)

                    st.subheader("🗼 Cellular Analysis")
                    out = self.helper.analyze_signal_peaks(sstr, freq_mhz, operator_rows)
                    if not out:
                        st.info("No strong trained cellular band seen.")
                    else:
                        # out -> returned by self.helper.analyze_signal_peaks(...)
                        legacy_rows = explode_analysis_results_unified(out)
                        if not legacy_rows:
                            st.info("No strong trained cellular band seen.")
                        else:
                            df = pd.DataFrame(legacy_rows)
                            sort_cols = [c for c in ["Operator","Band Type","Band","Peak (MHz)","Power (dBm)"] if c in df.columns]
                            if sort_cols:
                                df = df.sort_values(by=sort_cols, ascending=[True, True, True, True, False], na_position="last")
                            st.dataframe(df, use_container_width=True, hide_index=True)
                    # Wi-Fi (based on sweep)
                    if self._sweep_overlaps_wifi(freq):
                        st.subheader("📶 Wi-Fi Networks (local scan)")
                        if not PYWIFI_AVAILABLE:
                            st.warning("pywifi not available in this environment; skipping Wi-Fi scan.")
                        else:
                            dfw = self._scan_wifi()
                            if dfw.empty:
                                st.warning("No Wi-Fi networks found.")
                            else:
                                st.success(f"Found {len(dfw)} networks.")
                                st.dataframe(dfw, use_container_width=True)
                                st.download_button("📥 Download Wi-Fi CSV", data=dfw.to_csv(index=False),
                                                   file_name="wifi_scan.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Failed to process analysis: {e}")

            t.stop()
            st.write(f"elapsed: {fmt_seconds(t.elapsed())}")

# =====================================================================================
#                                         App
# =====================================================================================

def main():
    helper = TinySAHelper()
    provider = ModelProvider(mode='LLM', helper=helper)

    # Agent 1: Configuration
    config_agent = ConfigurationAgent(helper, provider)
    selected_options, def_dict = config_agent.run()

    # Agent 3: Location gating + optional override table
    location_agent = LocationAgent(helper)
    override_operator_table = location_agent.detect_and_gate()
    
    # If LocationAgent provided a new table, adopt it; else cached; else default
    if st.session_state["operator_table_df"] is None:
        st.session_state["operator_table_df"] = (
            override_operator_table
            if override_operator_table is not None
            else self.helper.get_operator_frequencies()
        )
    else:
        if override_operator_table is not None:
            st.session_state["operator_table_df"] = override_operator_table

    operator_table_df = st.session_state["operator_table_df"]

    # Agent 2: Analysis
    analysis_agent = AnalysisAgent(helper, provider)
    analysis_agent.run(def_dict, operator_table_df)

if __name__ == "__main__":
    main()
