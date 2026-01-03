import streamlit as st
import pandas as pd

st.set_page_config(page_title="5G NR RU Report", layout="wide")

# ----------------- RU REPORT JSON (same as before, inside ru_report["..."]) -----------------
ru_report = {
    "metadata": {
        "vendor": "Fujitsu",
        "model": "FTRU-5G-RU4468",
        "site": "TX-DAL-Rooftop-Sector-A",
        "sector": "A",
        "timestamp": "2025-11-25T20:00:00-06:00",
        "generated_by": "Ennoia REA Agent v1.4"
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
    "rea_diagnostics": {
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
        "Run DNI deeper analysis if PTP relocks recur."
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
    dict_to_table(report["rea_diagnostics"]["pcap_analysis"])
    st.markdown("### Issues Detected")
    issues = report["rea_diagnostics"]["issues_detected"]
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
    "REA Diagnostics": show_rea,
    "Recommendations": show_recommendations,
}

# ----------------- UI LOGIC -----------------

if "current_section" not in st.session_state:
    st.session_state.current_section = "Overview"

st.title("5G NR Fujitsu RU Health & Status Report")

meta = ru_report["metadata"]
st.caption(
    f"**Vendor:** {meta['vendor']} | **Model:** {meta['model']} | "
    f"**Site:** {meta['site']} | **Sector:** {meta['sector']} | "
    f"**Generated:** {meta['timestamp']}"
)

st.markdown("---")

# Button row
section_names = list(SECTION_FUNCS.keys())
cols = st.columns(len(section_names))

for col, name in zip(cols, section_names):
    if col.button(name, key=f"btn_{name}"):
        st.session_state.current_section = name

st.markdown("---")

current = st.session_state.current_section
st.header(current)
SECTION_FUNCS[current](ru_report)
