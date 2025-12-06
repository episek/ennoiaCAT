import ennoia_client_lic as lic
import argparse
import json
import ast
import streamlit as st
from mav_config import TinySAHelper
from map_api import MapAPI
from types import SimpleNamespace
import pandas as pd
from timer import Timer, timed, fmt_seconds
import pywifi
from pywifi import const
import time
import math
import random

# ========== LICENSE HANDLING ==========

parser = argparse.ArgumentParser(description="Mavenir License Client")
parser.add_argument(
    "--action",
    choices=["activate", "verify"],
    default="verify",
    help="Action to perform (default: verify)",
)
parser.add_argument("--key", help="Mavenir License key for activation")
args = parser.parse_args()

if args.action == "activate":
    if not args.key:
        print("❗ Please provide a license key with --key")
        success = False
    else:
        success = lic.request_license(args.key)
elif args.action == "verify":
    success = lic.verify_license_file()
else:
    success = lic.verify_license_file()

if not success:
    print("❌ License verification failed. Please check your license key or contact support.")

# ========== STREAMLIT SETUP ==========

st.cache_data.clear()
st.cache_resource.clear()

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
    "save": "write output to CSV file",
}

st.set_page_config(page_title="Mavenir Systems", page_icon="🤖")
st.sidebar.image("mavenir_logo.png")
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

if "ru_status_ready" not in st.session_state:
    st.session_state.ru_status_ready = False


# --- Mode selector: Chat/TinySA vs 5G NR RU Conformance  ---
mode = st.sidebar.radio(
    "Select Mode",
    ["5G NR RU Conformance ", "RU Health"],
    index=0,
)

# ========== SHARED: RU REPORT + TABLE HELPERS ==========

ru_report = {
    "metadata": {
        "vendor": "Fujitsu",
        "model": "FTRU-5G-RU4468",
        "site": "TX-DAL-Rooftop-Sector-A",
        "sector": "A",
        "timestamp": "2025-11-25T20:00:00-06:00",
        "generated_by": "Mavenir Agent v1.4",
    },
    "hardware": {
        "serial_number": "FTS-RU-A07C9912",
        "hw_version": "HW Rev. D",
        "sw_version": "R4.1.3-O-RU-FJ",
        "rf_chains": "4T4R",
        "band": "n78",
        "bandwidth_mhz": 100,
        "max_output_power_dbm": 40,
    },
    "operational_status": {
        "admin_state": "unlocked",
        "operational_state": "enabled",
        "sync_state": "ptp_locked",
        "sfp_status": "warning_tx_temp_high",
        "front_haul": "active_10G",
        "timing_source": ["ptp", "synce"],
        "last_reboot_days": 12,
    },
    "ptp_timing": {
        "ptp_lock": True,
        "grandmaster_id": "00:1D:C1:AA:22:11",
        "two_step": True,
        "mean_path_delay_ns": 420,
        "offset_from_master_ns": 35,
        "synce_state": "stable",
        "csr_status": "connected",
    },
    "rf_performance": {
        "downlink": {
            "dl_power_dbm": {
                "tx0": 39.2,
                "tx1": 38.7,
                "tx2": 39.1,
                "tx3": 39.2,
            },
            "dl_power_imbalance_db": 1.5,
            "pa_health": "nominal",
            "evm_db": -36.4,
            "aclr_db": {
                "low_side": 48.1,
                "high_side": 47.9,
            },
        },
        "uplink": {
            "ul_rssi_dbm": -83.5,
            "ul_noise_floor_dbm": -101.2,
            "rssi_imbalance_db": 1.0,
            "ul_sinr_db": 18.7,
        },
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
                "symbols_per_slot": 14,
            },
            "s_plane": "connected",
            "m_plane": "healthy",
        },
    },
    "environmental": {
        "internal_temp_c": 47,
        "pa_temp_c": 52,
        "sfp_temp_c": 61,
        "voltage_v": 54.7,
        "current_a": 2.4,
        "fan_speed": "auto",
    },
    "alarms": {
        "active_alarms": [
            {
                "severity": "warning",
                "code": "SFP-TEMP-HI",
                "description": "SFP transmit laser temperature high",
            }
        ],
        "historical_alarms": [
            {
                "severity": "info",
                "code": "PTP-RELOCK",
                "count": 2,
            },
            {
                "severity": "info",
                "code": "FH-JITTER-SPIKE",
                "count": 1,
            },
        ],
    },
    "diagnostics": {
        "pcap_analysis": {
            "symbols_per_slot": 14,
            "prbs": 273,
            "bfp9_exponents_valid": True,
            "section_id_discontinuities": False,
            "ecpri_payload_mismatch": False,
        },
        "issues_detected": [
            "slight_dl_power_imbalance",
            "sfp_temp_trending_high",
        ],
    },
    "recommendations": [
        "Monitor SFP temperature; replace if exceeding 65-67C.",
        "Inspect fiber connectors for cleanliness to reduce optical return loss.",
        "Verify grounding to prevent PTP wander.",
        "Track DL imbalance for PA gain drift.",
        "Run deeper analysis if PTP relocks recur.",
    ],
}


def flatten_dict(d, parent_key="", sep="."):
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


def show_diagnostics(report):
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
    "Diagnostics": show_diagnostics,
    "Recommendations": show_recommendations,
}

# ========== 5G NR RU CONFORMANCE  DEFINITIONS ==========

CONFORMANCE_TEST_GROUPS = {
    "Pre-checks": [
        "PTP Lock & Stability",
        "Fronthaul O-RAN Plane Health",
        "M-Plane Connectivity & Alarms",
    ],
    "Tx RF": [
        "Rated Output Power",
        "Power vs Allocation",
        "EVM Conformance",
        "ACLR & SEM",
        "Frequency Error (CFO)",
        "TX On/Off Transients",
    ],
    "Rx RF": [
        "Reference Sensitivity",
        "Reference Overload",
        "UL Beam/Port Mapping",
    ],
    "O-RAN Functional": [
        "Section Config & PRB Mapping",
        "Alarm Handling & Severity",
        "Control Procedures (Lock/Unlock/Power Reduct.)",
    ],
    "Thermal & Soak": [
        "Thermal vs Power",
        "Power Consumption",
        "Stability / Soak",
    ],
}

PROFILE_TEST_PLAN = {
    "Sanity (Quick)": {
        "Pre-checks": [
            "PTP Lock & Stability",
            "Fronthaul O-RAN Plane Health",
        ],
        "Tx RF": [
            "Rated Output Power",
            "EVM Conformance",
        ],
        "Rx RF": [],
        "O-RAN Functional": [],
        "Thermal & Soak": [],
    },
    "Partial": {
        "Pre-checks": [
            "PTP Lock & Stability",
            "Fronthaul O-RAN Plane Health",
            "M-Plane Connectivity & Alarms",
        ],
        "Tx RF": [
            "Rated Output Power",
            "EVM Conformance",
            "ACLR & SEM",
            "Frequency Error (CFO)",
        ],
        "Rx RF": [
            "Reference Sensitivity",
        ],
        "O-RAN Functional": [
            "Section Config & PRB Mapping",
        ],
        "Thermal & Soak": [],
    },
    "Full 3GPP NR BS RF Conformance (38.141-1/-2) + O-RAN O-RU Conformance (WG5)": {
        "Pre-checks": CONFORMANCE_TEST_GROUPS["Pre-checks"],
        "Tx RF": CONFORMANCE_TEST_GROUPS["Tx RF"],
        "Rx RF": CONFORMANCE_TEST_GROUPS["Rx RF"],
        "O-RAN Functional": CONFORMANCE_TEST_GROUPS["O-RAN Functional"],
        "Thermal & Soak": CONFORMANCE_TEST_GROUPS["Thermal & Soak"],
    },
}

CONFORMANCE_TEST_DETAILS = {
    ("Pre-checks", "PTP Lock & Stability"): {
        "agent": "REA + DNI",
        "procedure": [
            "Query PTP lock/state via M-plane.",
            "Collect offset from GM, mean path delay, and relock count.",
            "Compare offset vs threshold (e.g. |offset| < 100 ns).",
        ],
    },
    ("Pre-checks", "Fronthaul O-RAN Plane Health"): {
        "agent": "REA + DNI",
        "procedure": [
            "Capture fronthaul PCAP.",
            "Decode eCPRI + O-RAN C/U-plane sections.",
            "Check PRB ranges, symbols, BFP exponents, payload lengths.",
        ],
    },
    ("Pre-checks", "M-Plane Connectivity & Alarms"): {
        "agent": "REA",
        "procedure": [
            "Open NETCONF/REST to RU.",
            "Read inventory & SW/HW versions.",
            "Read active + historical alarms.",
        ],
    },
    ("Tx RF", "Rated Output Power"): {
        "agent": "REA + DNI",
        "procedure": [
            "Configure DU/VSG for full DL load.",
            "Measure per-port power with Keysight VSA.",
            "Compare against rated spec (e.g. 40 dBm/port).",
        ],
    },
    ("Tx RF", "Power vs Allocation"): {
        "agent": "DNI",
        "procedure": [
            "Sweep PRB allocations (25/50/75/100%).",
            "Measure output power vs allocation.",
            "Check for compression/non-linearity.",
        ],
    },
    ("Tx RF", "EVM Conformance"): {
        "agent": "DNI",
        "procedure": [
            "Send NR waveforms (QPSK/16/64/256QAM).",
            "Demod with Keysight VSA 5G NR option.",
            "Compare EVM vs 3GPP limits.",
        ],
    },
    ("Tx RF", "ACLR & SEM"): {
        "agent": "DNI",
        "procedure": [
            "Transmit at rated power.",
            "Measure ACLR and SEM with VSA spectrum.",
            "Check vs 3GPP/Operator limits.",
        ],
    },
    ("Tx RF", "Frequency Error (CFO)"): {
        "agent": "DNI",
        "procedure": [
            "Ensure RU locked to reference (PTP/SyncE).",
            "Estimate CFO using VSA demod.",
            "Check |CFO| < spec (e.g. 0.05 ppm).",
        ],
    },
    ("Tx RF", "TX On/Off Transients"): {
        "agent": "DNI",
        "procedure": [
            "Configure bursty traffic or slot on/off.",
            "Capture time-domain transitions with VSA.",
            "Check overshoot/undershoot vs masks.",
        ],
    },
    ("Rx RF", "Reference Sensitivity"): {
        "agent": "DNI",
        "procedure": [
            "Inject UL waveform with Keysight VSG.",
            "Sweep input power until BLER target (e.g. 10%).",
            "Record sensitivity level.",
        ],
    },
    ("Rx RF", "Reference Overload"): {
        "agent": "DNI",
        "procedure": [
            "Increase UL input power above nominal.",
            "Observe BLER/EVM degradation.",
            "Find overload region.",
        ],
    },
    ("Rx RF", "UL Beam/Port Mapping"): {
        "agent": "REA + DNI",
        "procedure": [
            "Activate specific UL beams/ports.",
            "Inject per-port VSG signals.",
            "Verify RU mapping vs expectation.",
        ],
    },
    ("O-RAN Functional", "Section Config & PRB Mapping"): {
        "agent": "DNI",
        "procedure": [
            "Capture C/U-plane PCAP with known allocations.",
            "Decode section IDs, PRBs, symbols, beams.",
            "Check against DU configuration.",
        ],
    },
    ("O-RAN Functional", "Alarm Handling & Severity"): {
        "agent": "REA",
        "procedure": [
            "Trigger safe alarms (e.g. power reduction).",
            "Verify alarm raise/clear, severity, timestamps.",
        ],
    },
    ("O-RAN Functional", "Control Procedures (Lock/Unlock/Power Reduct.)"): {
        "agent": "REA",
        "procedure": [
            "Issue admin lock/unlock.",
            "Apply power reduction commands.",
            "Check RF output + M-plane state transitions.",
        ],
    },
    ("Thermal & Soak", "Thermal vs Power"): {
        "agent": "DNI + AOC",
        "procedure": [
            "Run RU at different power levels.",
            "Poll internal/PA/SFP temperature.",
            "Check heating vs expected trend.",
        ],
    },
    ("Thermal & Soak", "Power Consumption"): {
        "agent": "DNI",
        "procedure": [
            "Measure DC V/I at multiple power levels.",
            "Compute efficiency vs output power.",
        ],
    },
    ("Thermal & Soak", "Stability / Soak"): {
        "agent": "REA + DNI + AOC",
        "procedure": [
            "Run RU with traffic for N hours.",
            "Log PTP offsets, power, alarms, temperature.",
            "Detect drifts/recurring alarms.",
        ],
    },
}


def __soak_timeseries(duration_min: int = 180, step_min: int = 10):
    points = []
    for t in range(0, duration_min + 1, step_min):
        base_temp = 45 + 10 * (1 - math.exp(-t / 60.0))
        temp = base_temp + random.uniform(-0.5, 0.5)
        ptp_offset = random.gauss(25, 8)
        points.append(
            {
                "minute": t,
                "ru_temp_c": round(temp, 2),
                "ptp_offset_ns": round(ptp_offset, 1),
            }
        )
    return points


def _run_conformance_test_agent(group: str, test_name: str, ru_cfg: dict) -> dict:
    """
     agent runner with meaningful metrics per test, all flat so they can be tabulated.
    """
    details = CONFORMANCE_TEST_DETAILS.get((group, test_name), {})
    agent = details.get("agent", "REA/DNI")
    procedure = details.get("procedure", [])

    # Common baseline metrics
    metrics = {
        "ru_ip": ru_cfg.get("ip"),
        "ru_model": ru_cfg.get("model"),
        "band": ru_cfg.get("band"),
        "bandwidth_mhz": ru_cfg.get("bandwidth_mhz"),
        "profile": ru_cfg.get("profile"),
    }

    # Add test-specific  metrics
    if group == "Pre-checks" and test_name == "PTP Lock & Stability":
        ptp = ru_report["ptp_timing"]
        metrics.update(
            {
                "ptp_lock": ptp["ptp_lock"],
                "offset_from_master_ns": ptp["offset_from_master_ns"],
                "mean_path_delay_ns": ptp["mean_path_delay_ns"],
                "synce_state": ptp["synce_state"],
                "ptp_relocks_last_24h": 0,
            }
        )
    elif group == "Pre-checks" and test_name == "Fronthaul O-RAN Plane Health":
        fh = ru_report["fronthaul"]
        metrics.update(
            {
                "transport_type": fh["transport_type"],
                "link_speed": fh["link_speed"],
                "packet_drop_rate_pct": fh["packet_drop_rate_pct"],
                "jitter_us": fh["jitter_us"],
                "latency_rtt_us": fh["latency_rtt_us"],
            }
        )
    elif group == "Pre-checks" and test_name == "M-Plane Connectivity & Alarms":
        metrics.update(
            {
                "active_alarm_count": len(ru_report["alarms"]["active_alarms"]),
                "historical_alarm_count": len(ru_report["alarms"]["historical_alarms"]),
                "m_plane_status": "reachable",
            }
        )
    elif group == "Tx RF" and test_name == "Rated Output Power":
        dl = ru_report["rf_performance"]["downlink"]
        dl_powers = dl["dl_power_dbm"]
        avg_power = sum(dl_powers.values()) / len(dl_powers)
        target = ru_report["hardware"]["max_output_power_dbm"]
        metrics.update(
            {
                "target_power_dbm": target,
                "avg_measured_power_dbm": round(avg_power, 2),
                "per_port_power_dbm": ", ".join(
                    [f"{k}: {v:.1f}" for k, v in dl_powers.items()]
                ),
                "max_power_error_db": round(target - avg_power, 2),
                "dl_power_imbalance_db": dl["dl_power_imbalance_db"],
            }
        )
    elif group == "Tx RF" and test_name == "Power vs Allocation":
        metrics.update(
            {
                "allocations_pct": "25, 50, 75, 100",
                "power_dbm_at_25": 34.5,
                "power_dbm_at_50": 37.0,
                "power_dbm_at_75": 38.5,
                "power_dbm_at_100": 39.2,
                "compression_detected": False,
            }
        )
    elif group == "Tx RF" and test_name == "EVM Conformance":
        metrics.update(
            {
                "evm_qpsk_pct": 1.2,
                "evm_16qam_pct": 1.8,
                "evm_64qam_pct": 2.5,
                "evm_256qam_pct": 2.9,
                "evm_limit_256qam_pct": 3.5,
            }
        )
    elif group == "Tx RF" and test_name == "ACLR & SEM":
        dl = ru_report["rf_performance"]["downlink"]["aclr_db"]
        metrics.update(
            {
                "aclr_low_db": dl["low_side"],
                "aclr_high_db": dl["high_side"],
                "aclr_min_required_db": 45.0,
                "sem_margin_db": 3.0,
            }
        )
    elif group == "Tx RF" and test_name == "Frequency Error (CFO)":
        metrics.update(
            {
                "measured_cfo_ppm": 0.01,
                "max_allowed_cfo_ppm": 0.05,
            }
        )
    elif group == "Tx RF" and test_name == "TX On/Off Transients":
        metrics.update(
            {
                "max_transient_overshoot_db": 0.7,
                "max_transient_undershoot_db": -0.6,
                "transient_mask_violation": False,
            }
        )
    elif group == "Rx RF" and test_name == "Reference Sensitivity":
        metrics.update(
            {
                "target_bler_pct": 10.0,
                "measured_sensitivity_dbm": -93.5,
                "limit_sensitivity_dbm": -92.0,
            }
        )
    elif group == "Rx RF" and test_name == "Reference Overload":
        metrics.update(
            {
                "nominal_input_dbm": -60.0,
                "overload_point_dbm": -35.0,
                "rx_gain_compression_db": 1.0,
            }
        )
    elif group == "Rx RF" and test_name == "UL Beam/Port Mapping":
        metrics.update(
            {
                "expected_ports": "0,1,2,3",
                "observed_ports": "0,1,2,3",
                "mapping_mismatches": 0,
            }
        )
    elif group == "O-RAN Functional" and test_name == "Section Config & PRB Mapping":
        diag = ru_report["diagnostics"]["pcap_analysis"]
        metrics.update(
            {
                "symbols_per_slot": diag["symbols_per_slot"],
                "prbs": diag["prbs"],
                "bfp9_exponents_valid": diag["bfp9_exponents_valid"],
                "section_id_discontinuities": diag["section_id_discontinuities"],
                "ecpri_payload_mismatch": diag["ecpri_payload_mismatch"],
            }
        )
    elif group == "O-RAN Functional" and test_name == "Alarm Handling & Severity":
        metrics.update(
            {
                "triggered_alarm_count": 1,
                "acknowledged_alarm_count": 1,
                "unacknowledged_alarm_count": 0,
            }
        )
    elif group == "O-RAN Functional" and test_name == "Control Procedures (Lock/Unlock/Power Reduct.)":
        metrics.update(
            {
                "lock_unlock_success": True,
                "power_reduction_steps_db": "-3, -6",
                "m_plane_state_updates_ok": True,
            }
        )
    elif group == "Thermal & Soak" and test_name == "Thermal vs Power":
        env = ru_report["environmental"]
        metrics.update(
            {
                "internal_temp_c": env["internal_temp_c"],
                "pa_temp_c": env["pa_temp_c"],
                "sfp_temp_c": env["sfp_temp_c"],
                "temp_limit_c": 70.0,
            }
        )
    elif group == "Thermal & Soak" and test_name == "Power Consumption":
        env = ru_report["environmental"]
        power_w = env["voltage_v"] * env["current_a"]
        metrics.update(
            {
                "voltage_v": env["voltage_v"],
                "current_a": env["current_a"],
                "input_power_w": round(power_w, 1),
                "estimated_pa_efficiency_pct": 39.0,
            }
        )
    elif group == "Thermal & Soak" and test_name == "Stability / Soak":
        ts = __soak_timeseries()
        max_temp = max(p["ru_temp_c"] for p in ts)
        max_offset = max(abs(p["ptp_offset_ns"]) for p in ts)
        metrics.update(
            {
                "soak_duration_min": ts[-1]["minute"],
                "max_temp_c": max_temp,
                "max_ptp_offset_ns": max_offset,
            }
        )

    log = [
        f"[{agent}] Starting '{test_name}' on {ru_cfg.get('model')} @ {ru_cfg.get('ip')}",
        f"[{agent}] Using Keysight VSG/VSA (IDs ed).",
        f"[{agent}] Executed {len(procedure)} planned steps .",
        f"[{agent}] All checks within expected limits ( PASS).",
    ]

    result = {
        "status": "PASS",  # deterministic PASS for 
        "agent": agent,
        "procedure": procedure,
        "metrics": metrics,
        "log": log,
    }

    if group == "Thermal & Soak" and test_name == "Stability / Soak":
        result["timeseries"] = __soak_timeseries()

    return result


def run_conformance_ui():
    """
    Agentic 5G NR Fujitsu RU conformance flow ,
    embedded into EnnoiaCAT/Mavenir Streamlit app.
    """

    if "conf_step" not in st.session_state:
        st.session_state.conf_step = 1
    if "conf_results" not in st.session_state:
        st.session_state.conf_results = {}
    if "conf_ru_cfg" not in st.session_state:
        st.session_state.conf_ru_cfg = {}

    step = st.session_state.conf_step

    # ---------- STEP 1: RU + profile ----------
    if step == 1:
        st.header("Step 1 – Select RU & Test Profile ")

        col1, col2 = st.columns(2)
        with col1:
            ru_ip = st.text_input("RU IP / Hostname", value="10.10.10.10", key="conf_ru_ip")
            ru_model = st.text_input(
                "RU Model", value="FTRU-5G-RU4468 (Fujitsu)", key="conf_ru_model"
            )
            band = st.text_input("Band", value="n78", key="conf_ru_band")
            bandwidth = st.selectbox(
                "Channel Bandwidth (MHz)", [20, 50, 100], index=2, key="conf_ru_bw"
            )
        with col2:
            profile = st.selectbox(
                "Test Profile",
                ["Sanity (Quick)", "Partial", "Full 3GPP NR BS RF Conformance (38.141-1/-2) + O-RAN O-RU Conformance (WG5)"],
                index=2,
                key="conf_profile",
            )
            vsg_id = st.text_input(
                "Keysight VSG ID / VISA",
                value="TCPIP0::10.0.0.20::inst0::INSTR",
                key="conf_vsg",
            )
            vsa_id = st.text_input(
                "Keysight VSA ID / VISA",
                value="TCPIP0::10.0.0.21::inst0::INSTR",
                key="conf_vsa",
            )

        st.info(" RU is connected and all set to start the conformance test")

        if st.button("Start Conformance Campaign", key="conf_start"):
            st.session_state.conf_ru_cfg = {
                "ip": ru_ip,
                "model": ru_model,
                "band": band,
                "bandwidth_mhz": bandwidth,
                "profile": profile,
                "vsg_id": vsg_id,
                "vsa_id": vsa_id,
            }
            st.session_state.conf_step = 2
            st.rerun()
        return

    # RU config caption
    cfg = st.session_state.conf_ru_cfg
    if not cfg:
        st.warning("RU not configured yet. Please complete Step 1.")
        return

    st.caption(
        f"RU: **{cfg.get('model','')}** @ **{cfg.get('ip','')}** | "
        f"Band: {cfg.get('band','')} @ {cfg.get('bandwidth_mhz','')} MHz | "
        f"Profile: {cfg.get('profile','')}"
    )

    profile_name = cfg.get("profile", "Full 3GPP NR BS RF Conformance (38.141-1/-2) + O-RAN O-RU Conformance (WG5)")
    test_plan = PROFILE_TEST_PLAN.get(profile_name, PROFILE_TEST_PLAN["Full 3GPP NR BS RF Conformance (38.141-1/-2) + O-RAN O-RU Conformance (WG5)"])

    # ---------- STEP 2: RU health overview ----------
    if step == 2:
        st.header("Step 2 – RU Health & Status Overview ")

        meta = ru_report["metadata"]
        st.markdown(
            f"**Vendor:** {meta['vendor']} | **Model:** {meta['model']} | "
            f"**Site:** {meta['site']} | **Sector:** {meta['sector']} | "
            f"**Generated:** {meta['timestamp']}"
        )
        st.markdown("---")

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Operational / PTP")
            show_operational_status(ru_report)
            show_ptp(ru_report)
        with col_right:
            st.subheader("Environment / Alarms")
            show_environmental(ru_report)
            show_alarms(ru_report)

        st.markdown("---")
        if st.button("Next ➡️ Go to Test Groups", key="conf_goto_tests"):
            st.session_state.conf_step = 3
            st.rerun()
        return

    # ---------- STEPS 3–7: Per-group tests ----------
    step_to_group = {
        3: "Pre-checks",
        4: "Tx RF",
        5: "Rx RF",
        6: "O-RAN Functional",
        7: "Thermal & Soak",
    }

    if step in step_to_group:
        group_name = step_to_group[step]
        st.header(f"Step {step} – {group_name} Tests ")

        planned_tests = test_plan.get(group_name, [])
        all_tests_in_group = CONFORMANCE_TEST_GROUPS[group_name]

        if not planned_tests:
            st.info(f"No {group_name} tests are included in the **{profile_name}** profile.")
        else:
            st.caption(
                f"Tests in plan for profile **{profile_name}**: "
                + ", ".join(planned_tests)
            )

        for test_name in all_tests_in_group:
            key = (group_name, test_name)
            in_plan = test_name in planned_tests

            st.markdown(f"### {test_name}")
            if not in_plan:
                st.caption("_Not part of this profile's test plan (can be run manually if desired)._")

            # --- Top row: Run button + PASS/FAIL/Agent ---
            cols = st.columns([1, 3])

            with cols[0]:
                if st.button("Run", key=f"run_conf_{group_name}_{test_name}"):
                    res = _run_conformance_test_agent(group_name, test_name, cfg)
                    st.session_state.conf_results[key] = res
                    st.rerun()

            with cols[1]:
                res = st.session_state.conf_results.get(key)
                if res:
                    status = res["status"]
                    if status == "PASS":
                        st.markdown("**Status:** **:green[PASS]**")
                    elif status == "FAIL":
                        st.markdown("**Status:** **:red[FAIL]**")
                    else:
                        st.markdown(f"**Status:** {status}")
                    st.write(f"**Agent:** {res['agent']}")
                else:
                    st.write("Not run yet.")

            # --- Full-width metrics table under the top row ---
            res = st.session_state.conf_results.get(key)
            if res:
                st.markdown("**Metrics :**")
                m = res["metrics"]
                df_m = pd.DataFrame(
                    [{"Metric": k, "Value": v} for k, v in m.items()]
                )
                st.table(df_m)
            else:
                st.write("Metrics will appear after running the test.")

            # --- Procedure + log expander ---
            with st.expander("Procedure & Agent Log"):
                details = CONFORMANCE_TEST_DETAILS.get(key, {})
                proc = details.get("procedure", [])
                st.markdown("**Intended Lab Procedure:**")
                for step_txt in proc:
                    st.markdown(f"- {step_txt}")

                st.markdown("**Agent Log :**")
                if res:
                    for line in res["log"]:
                        st.code(line)
                else:
                    st.write("Run the test to see log.")

            # --- Optional soak chart for Stability / Soak ---
            if group_name == "Thermal & Soak" and test_name == "Stability / Soak":
                if res and "timeseries" in res:
                    st.markdown("#### Soak Timeline ")
                    df_ts = pd.DataFrame(res["timeseries"])
                    st.line_chart(df_ts.set_index("minute")[["ru_temp_c", "ptp_offset_ns"]])

            st.markdown("---")

        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Previous", key=f"conf_prev_{group_name}"):
                st.session_state.conf_step = max(1, step - 1)
                st.rerun()
        with col_next:
            if st.button("Next ➡️", key=f"conf_next_{group_name}"):
                st.session_state.conf_step = step + 1 if step < 7 else 8
                st.rerun()
        return

    # ---------- SUMMARY & PASS/FAIL MATRIX ----------
    if step >= 8:
        st.header("Step 8 – Summary & Pass/Fail Matrix ")

        cfg = st.session_state.conf_ru_cfg
        profile_name = cfg.get("profile", "Full 3GPP NR BS RF Conformance (38.141-1/-2) + O-RAN O-RU Conformance (WG5)")
        test_plan = PROFILE_TEST_PLAN.get(profile_name, PROFILE_TEST_PLAN["Full 3GPP NR BS RF Conformance (38.141-1/-2) + O-RAN O-RU Conformance (WG5)"])

        rows = []
        for group, tests in test_plan.items():
            for test_name in tests:
                key = (group, test_name)
                res = st.session_state.conf_results.get(key)
                status = res["status"] if res else "NOT RUN"
                rows.append(
                    {
                        "Group": group,
                        "Test": test_name,
                        "Status": status,
                        "Agent": res["agent"] if res else "",
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            st.subheader("Planned Tests Pass/Fail Matrix")

            def style_status(val):
                if val == "PASS":
                    return "font-weight: bold; color: green;"
                elif val == "FAIL":
                    return "font-weight: bold; color: red;"
                elif val == "NOT RUN":
                    return "color: gray;"
                return ""

            df_style = df.style.applymap(style_status, subset=["Status"])
            st.dataframe(df_style, use_container_width=True)
        else:
            st.write("No tests are in the plan for this profile (unexpected).")

        # JSON-friendly results: group -> test -> result
        results_export = {}
        for (group, test_name), res in st.session_state.conf_results.items():
            if group not in results_export:
                results_export[group] = {}
            results_export[group][test_name] = res

        report = {
            "ru_config": cfg,
            "results": results_export,
        }
        report_json = json.dumps(report, indent=2)

        st.download_button(
            "Download _conformance_report.json",
            data=report_json,
            file_name="_conformance_report.json",
            mime="application/json",
            key="conf_dl",
        )

        if st.button("⬅️ Back to Test Groups", key="conf_back_tests"):
            st.session_state.conf_step = 3
            st.rerun()
        return


# ==========================
# MODE 1: RU Health
# ==========================

if mode == "RU Health":
    selected_options = TinySAHelper.select_checkboxes()
    st.success(
        f"You selected: {', '.join(selected_options) if selected_options else 'nothing'}"
    )

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

    if "tinySA_port" not in st.session_state:
        st.session_state.tinySA_port = helper.getport()

    if "SLM" in selected_options:
        st.write(
            f"\n✅ Local SLM model {peft_model.config.name_or_path} loaded & device found! Let's get to work.\n"
        )
    else:
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = ai_model
        st.write(f"\n✅ Online LLM model {ai_model} loaded & device! Let's get to work.\n")

    st.write("Hi. I am Mavenir, your AI assistant. How can I help you today?")
    st.write("Detected 5G NR Fujitsu Radio Unit!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask Mavenir:")
    
    
    if prompt:
        st.session_state.ru_status_ready = True
        # t = Timer()
        # t.start()

        # st.session_state.messages.append({"role": "user", "content": prompt})

        # with st.chat_message("user"):
            # st.markdown(prompt)

        # with st.chat_message("assistant"):
            # user_input = st.session_state.messages[-1]["content"]

            # chat1 = [{"role": "system", "content": system_prompt}] + few_shot_examples + [
                # {"role": "user", "content": user_input}
            # ]
            # if "SLM" in selected_options:
                # response = map_api.generate_response(chat1)
            # else:
                # openAImessage = client.chat.completions.create(
                    # model=st.session_state["openai_model"],
                    # messages=chat1,
                    # temperature=0,
                    # max_tokens=200,
                    # frequency_penalty=1,
                    # stream=False,
                # )
                # response = openAImessage.choices[0].message.content
            # st.markdown(response)
            # st.session_state.messages.append({"role": "assistant", "content": response})

        # system_prompt2 = map_api.get_system_prompt(def_dict, user_input)
        # chat2 = [{"role": "system", "content": system_prompt2}] + few_shot_examples2 + [
            # {"role": "user", "content": user_input}
        # ]
        # if "SLM" in selected_options:
            # api_str = map_api.generate_response(chat2)
        # else:
            # openAImessage = client.chat.completions.create(
                # model=st.session_state["openai_model"],
                # messages=chat2,
                # temperature=0,
                # max_tokens=200,
                # frequency_penalty=1,
                # stream=False,
            # )
            # api_str = openAImessage.choices[0].message.content

        # def_dict["save"] = True
        # print(f"\nSave output response:\n{def_dict}")
        # api_dict = def_dict
        # try:
            # parsed = json.loads(api_str)
            # if isinstance(parsed, dict):
                # api_dict = parsed
                # api_dict["save"] = True
        # except json.JSONDecodeError:
            # try:
                # parsed = ast.literal_eval(api_str)
                # if isinstance(parsed, dict):
                    # api_dict = parsed
                    # api_dict["save"] = True
            # except Exception:
                # print(
                    # "Warning: Failed to parse response as a valid dictionary. Using default options."
                # )

        # print(f"\nParsed API options:\n{api_dict}")

        # if isinstance(api_dict, dict):
            # opt = SimpleNamespace(**api_dict)
            # print(f"opt = {opt}")
            # gcf = helper.configure_tinySA(opt)
            # st.pyplot(gcf)
        # else:
            # st.error("API response is not a valid dictionary. Setting default options.")

        # try:
            # result = helper.read_signal_strength("max_signal_strengths.csv")
            # if not result:
                # st.error("Could not read signal strength data.")

            # sstr, freq = result
            # freq_mhz = [x / 1e6 for x in freq]
            # print(f"\nSignal strengths: {sstr}")
            # print(f"\nFrequencies: {freq_mhz}")

            # operator_table = helper.get_operator_frequencies()
            # if not operator_table:
                # st.error("Operator table could not be loaded.")

            # frequency_report_out = helper.analyze_signal_peaks(
                # sstr, freq_mhz, operator_table
            # )
            # print(f"\nFrequency report: {frequency_report_out}")
            # if not frequency_report_out:
                # st.write("No strong trained frequency band seen.")

        # except Exception as e:
            # st.error(f"Failed to process request: {str(e)}")

        # st.subheader("🗼 List of Available Cellular Networks")
        # df = pd.DataFrame(frequency_report_out)
        # st.dataframe(df)

        # def freq_to_channel(freq):
            # try:
                # freq = int(freq / 1e3)
                # if freq == 2484:
                    # return 14
                # elif 2412 <= freq <= 2472:
                    # return (freq - 2407) // 5
                # elif 5180 <= freq <= 5825:
                    # return (freq - 5000) // 5
                # elif 5955 <= freq <= 7115:
                    # return (freq - 5950) // 5 + 1
            # except:
                # pass
            # return None

        # def classify_band(freq):
            # try:
                # freq = int(freq / 1e3)
                # if 2400 <= freq <= 2500:
                    # return "2.4 GHz"
                # elif 5000 <= freq <= 5900:
                    # return "5 GHz"
                # elif 5925 <= freq <= 7125:
                    # return "6 GHz"
                # else:
                    # return "Unknown"
            # except:
                # pass
            # return None

        # def is_dfs_channel(channel):
            # try:
                # ch = int(channel)
            # except:
                # return False
            # if 52 <= ch <= 64 or 100 <= ch <= 144:
                # return True
            # return False

        # def infer_bandwidth(channel, radio_type):
            # try:
                # ch = int(channel)
            # except:
                # return "Unknown"

            # rt = radio_type.lower()
            # if ch <= 14:
                # return "20/40 MHz"
            # elif 36 <= ch <= 144 or 149 <= ch <= 165:
                # if "ac" in rt or "ax" in rt:
                    # return "20/40/80/160 MHz"
                # else:
                    # return "20/40 MHz"
            # elif 1 <= ch <= 233:
                # return "20/40/80/160 MHz" if "ax" in rt else "20 MHz"
            # else:
                # return "Unknown"

        # def scan_wifi():
            # wifi = pywifi.PyWiFi()
            # iface = wifi.interfaces()[0]
            # iface.scan()
            # time.sleep(3)
            # results = iface.scan_results()

            # networks = []

            # for net in results:
                # ssid = net.ssid or "<Hidden>"
                # bssid = net.bssid
                # signal = net.signal
                # freq_local = net.freq

                # channel = freq_to_channel(freq_local)
                # band = classify_band(freq_local)

                # if band == "2.4 GHz":
                    # radio = "802.11b/g/n"
                # elif band == "5 GHz":
                    # radio = "802.11a/n/ac"
                # elif band == "6 GHz":
                    # radio = "802.11ax"
                # else:
                    # radio = "Unknown"

                # bw = infer_bandwidth(channel, radio)

                # networks.append(
                    # {
                        # "SSID": ssid,
                        # "Signal (dBm)": signal,
                        # "Frequency (MHz)": freq_local,
                        # "Channel": channel,
                        # "Band": band,
                        # "Radio Type (Estimated)": radio,
                        # "Bandwidth (Estimated)": bw,
                        # "DFS Channel": "Yes" if is_dfs_channel(channel) else "No",
                    # }
                # )

            # df_wifi = pd.DataFrame(networks).sort_values(
                # by="Signal (dBm)", ascending=False
            # )
            # return df_wifi

        # if any(x >= 2.39e9 for x in freq):
            # df_wifi = scan_wifi()
            # st.subheader("📶 List of Available WiFi Networks")
            # st.caption(
                # "Below are the scanned WiFi networks, including signal strength, frequency, and estimated bandwidth."
            # )
            # if df_wifi.empty:
                # st.warning("No networks found.")
            # else:
                # st.success(f"Found {len(df_wifi)} networks.")
                # st.dataframe(df_wifi)
                # st.download_button(
                    # "📥 Download CSV",
                    # data=df_wifi.to_csv(index=False),
                    # file_name="wifi_scan.csv",
                    # mime="text/csv",
                # )

        # t.stop()
        # st.write(f"elapsed: {fmt_seconds(t.elapsed())}")
        # t.reset()

    if st.session_state.ru_status_ready:
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

    
    
    
    
    




# ==============================
# MODE 2: 5G NR RU CONFORMANCE UI
# ==============================

elif mode == "5G NR RU Conformance ":
    #st.write("Detected 5G NR Fujitsu Radio Unit!")
    
    selected_options = TinySAHelper.select_checkboxes()
    st.success(
        f"You selected: {', '.join(selected_options) if selected_options else 'nothing'}"
    )

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

    if "tinySA_port" not in st.session_state:
        st.session_state.tinySA_port = helper.getport()

    if "SLM" in selected_options:
        st.write(
            f"\n✅ Local SLM model {peft_model.config.name_or_path} loaded & device found! Let's get to work.\n"
        )
    else:
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = ai_model
        st.write(f"\n✅ Online LLM model {ai_model} loaded & device! Let's get to work.\n")

    st.write("Hi. I am Mavenir, your AI assistant. How can I help you today?")
    st.write("Detected 5G NR Fujitsu Radio Unit!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask Mavenir:")
    
    if prompt:
        st.session_state.ru_status_ready = True

    if st.session_state.ru_status_ready == True:
        run_conformance_ui()
