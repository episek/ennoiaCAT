#!/usr/bin/env python

# --- Make sure we can import ennoia_viavi even when run from this folder ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # C:\Users\epise\ennoiaCAT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------

import streamlit as st

from ennoia_viavi.streamlit_ui import (
    connect_and_get_radio_client,
    render_spectrum_panel,
    render_wifi_panel,
    render_nr5g_panel,
)


def main() -> None:
    st.set_page_config(
        page_title="VIAVI OneAdvisor – Spectrum & 5G NR",
        layout="wide",
    )

    st.title("VIAVI OneAdvisor – Spectrum & 5G NR Dashboard")

    # ---- Sidebar: connection parameters ----
    st.sidebar.header("Connection")

    host = st.sidebar.text_input(
        "Instrument IP Address",
        value="192.168.1.100",
        key="sidebar_host"
    )
    discovery_port = st.sidebar.number_input(
        "System SCPI Port",
        value=5025,
        step=1,
        key="sidebar_disc_port"
    )
    timeout = st.sidebar.number_input(
        "Timeout (s)",
        value=5.0,
        step=0.5,
        key="sidebar_timeout"
    )

    if "radio_connected" not in st.session_state:
        st.session_state.radio_connected = False
        st.session_state.host = None

    if st.sidebar.button("Connect", key="sidebar_connect_btn"):
        ra = connect_and_get_radio_client(
            host=host,
            discovery_port=int(discovery_port),
            timeout=float(timeout)
        )
        if ra is not None:
            st.session_state.radio_connected = True
            st.session_state.host = host
            ra.close()
        else:
            st.session_state.radio_connected = False
            st.session_state.host = None

    if not st.session_state.radio_connected:
        st.info("Enter OneAdvisor IP and click **Connect**.")
        return

    host = st.session_state.host

    # Tabs
    tab_spectrum, tab_wifi, tab_nr5g = st.tabs(
        ["Spectrum", "Wi-Fi Scan", "5G NR"]
    )

    # ---- SPECTRUM TAB ----
    with tab_spectrum:
        ra = connect_and_get_radio_client(host)
        if ra is not None:
            try:
                # Pass custom keys to avoid UID collision
                render_spectrum_panel(
                    ra,
                    default_center_ghz=3.5,
                    default_span_mhz=100.0,
                )
            finally:
                ra.close()

    # ---- WI-FI TAB ----
    with tab_wifi:
        ra = connect_and_get_radio_client(host)
        if ra is not None:
            try:
                render_wifi_panel(ra)
            finally:
                ra.close()

    # ---- 5G NR TAB ----
    with tab_nr5g:
        ra = connect_and_get_radio_client(host)
        if ra is not None:
            try:
                render_nr5g_panel(ra)
            finally:
                ra.close()


if __name__ == "__main__":
    main()
