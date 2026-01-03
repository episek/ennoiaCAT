#!/usr/bin/env python

"""
Standalone Streamlit demo for VIAVI OneAdvisor:

- Connect via System SCPI
- Discover Radio SCPI port
- Tabs:
    - Spectrum Analyzer
    - Wi-Fi Scan
    - 5G NR Quick Summary
"""

import sys
from pathlib import Path

# Make sure we can import ennoia_viavi when running from ennoiaCAT
ROOT = Path(__file__).resolve().parent  # C:\Users\epise\ennoiaCAT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import streamlit as st
import matplotlib.pyplot as plt

from ennoia_viavi.system_api import OneAdvisorSystemAPI
from ennoia_viavi.radio_api import OneAdvisorRadioAPI
from ennoia_viavi.plotting import plot_spectrum_trace, plot_wifi_scan


def connect_viavi(host: str, discovery_port: int, timeout: float):
    """
    Connect to System SCPI, discover Radio SCPI port, return (radio_port, idn).
    Show Streamlit errors if something fails.
    """
    try:
        sys_api = OneAdvisorSystemAPI(host, port=discovery_port, timeout=timeout)
        sys_api.open()
    except Exception as e:
        st.error(f"Failed to connect to System SCPI ({host}:{discovery_port}): {e}")
        return None, None

    try:
        idn = sys_api.idn()
        st.info(f"System IDN: {idn}")
        radio_port = sys_api.get_radio_scpi_port()
        st.success(f"Discovered Radio SCPI port: {radio_port}")
    except Exception as e:
        st.error(f"Failed during :PRTM:LIST? discovery: {e}")
        radio_port = None
        idn = None
    finally:
        sys_api.close()

    return radio_port, idn


def spectrum_tab(host: str, radio_port: int, timeout: float):
    st.subheader("Spectrum Analyzer")

    col1, col2, col3 = st.columns(3)
    with col1:
        center_ghz = st.number_input(
            "Center Frequency (GHz)",
            value=3.5,
            step=0.1,
            format="%.3f",
            key="spec_center_ghz",
        )
    with col2:
        span_mhz = st.number_input(
            "Span (MHz)",
            value=100.0,
            step=10.0,
            format="%.1f",
            key="spec_span_mhz",
        )
    with col3:
        ref_level = st.number_input(
            "Reference Level (dBm)",
            value=0.0,
            step=5.0,
            key="spec_ref_level",
        )

    col4, col5 = st.columns(2)
    with col4:
        rbw_khz = st.number_input(
            "RBW (kHz)",
            value=100.0,
            step=10.0,
            format="%.1f",
            key="spec_rbw_khz",
        )
    with col5:
        vbw_khz = st.number_input(
            "VBW (kHz)",
            value=100.0,
            step=10.0,
            format="%.1f",
            key="spec_vbw_khz",
        )

    if st.button("Acquire Spectrum Trace", key="spec_acquire_btn"):
        try:
            ra = OneAdvisorRadioAPI(host, scpi_port=radio_port, timeout=timeout)
            ra.open()

            ra.set_spectrum_mode("spectrumTuned")
            ra.configure_spectrum(
                center_hz=center_ghz * 1e9,
                span_hz=span_mhz * 1e6,
                rbw_auto=False,
                rbw_hz=rbw_khz * 1e3,
                vbw_auto=False,
                vbw_hz=vbw_khz * 1e3,
                ref_level_dbm=ref_level,
                atten_mode="Auto",
            )

            time.sleep(0.3)
            trace = ra.get_spectrum_trace()
            #st.write(trace)
            start_mhz, stop_mhz, _ = ra.get_spectrum_xaxis()

            if not trace:
                st.warning("No trace data received.")
                return

            st.write(
                f"Points: {len(trace)}, Start: {start_mhz/1e3:.3f} GHz, "
                f"Stop: {stop_mhz/1e3:.3f} GHz"
            )

            fig, ax = plt.subplots()
            plot_spectrum_trace(
                trace_dbm=trace,
                start_mhz=start_mhz,
                stop_mhz=stop_mhz,
                ax=ax,
                title=f"Spectrum @ {center_ghz:.3f} GHz, {span_mhz:.1f} MHz span",
            )
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error acquiring spectrum trace: {e}")
        finally:
            try:
                ra.close()
            except Exception:
                pass


def wifi_tab(host: str, radio_port: int, timeout: float):
    st.subheader("Wi-Fi Band Scan (Spectrum Mode)")

    band = st.selectbox(
        "Band",
        options=["2.4", "5", "6"],
        index=0,
        key="wifi_band",
    )
    span_mhz = st.number_input(
        "Span (MHz)",
        value=100.0,
        step=20.0,
        format="%.1f",
        key="wifi_span_mhz",
    )
    rbw_khz = st.number_input(
        "RBW (kHz)",
        value=100.0,
        step=10.0,
        format="%.1f",
        key="wifi_rbw_khz",
    )
    ref_level = st.number_input(
        "Reference Level (dBm)",
        value=-10.0,
        step=5.0,
        key="wifi_ref_level",
    )

    if st.button("Scan Wi-Fi Band", key="wifi_scan_btn"):
        try:
            ra = OneAdvisorRadioAPI(host, scpi_port=radio_port, timeout=timeout)
            ra.open()

            trace, center_hz, span_hz = ra.scan_wifi_band(
                band=band,
                span_hz=span_mhz * 1e6,
                rbw_hz=rbw_khz * 1e3,
                ref_level_dbm=ref_level,
                settle_time_s=0.3,
            )

            if not trace:
                st.warning("No trace data received.")
                return

            fig, ax = plt.subplots()
            label = {"2.4": "2.4 GHz Wi-Fi", "5": "5 GHz Wi-Fi", "6": "6 GHz Wi-Fi"}[
                band
            ]
            plot_wifi_scan(
                trace_dbm=trace,
                center_hz=center_hz,
                span_hz=span_hz,
                band_label=label,
                ax=ax,
            )
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error scanning Wi-Fi band: {e}")
        finally:
            try:
                ra.close()
            except Exception:
                pass


def nr5g_tab(host: str, radio_port: int, timeout: float):
    st.subheader("5G NR Quick Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        center_ghz = st.number_input(
            "Center Frequency (GHz)",
            value=3.5,
            step=0.1,
            format="%.3f",
            key="nr_center_ghz",
        )
    with col2:
        band = st.selectbox(
            "Band",
            options=["FR1", "FR2"],
            index=0,
            key="nr_band",
        )
    with col3:
        ref_level = st.number_input(
            "Reference Level (dBm)",
            value=0.0,
            step=5.0,
            key="nr_ref_level",
        )

    col4, col5 = st.columns(2)
    with col4:
        chan_standard = st.number_input(
            "Channel Standard (band code)",
            value=700,
            step=1,
            key="nr_chan_standard",
        )
    with col5:
        chan_num = st.number_input(
            "Channel Number",
            value=1,
            step=1,
            key="nr_chan_num",
        )

    if st.button("Run 5G NR Quick Summary", key="nr_run_btn"):
        try:
            ra = OneAdvisorRadioAPI(host, scpi_port=radio_port, timeout=timeout)
            ra.open()

            with st.spinner("Measuring 5G NR..."):
                summary = ra.nr5g_quick_summary(
                    center_hz=center_ghz * 1e9,
                    band=band,
                    chan_standard=int(chan_standard),
                    chan_num=int(chan_num),
                    ref_level_dbm=ref_level,
                )

            st.success("Measurement complete.")
            st.json(summary)

        except Exception as e:
            st.error(f"Error running 5G NR summary: {e}")
        finally:
            try:
                ra.close()
            except Exception:
                pass


def main():
    st.set_page_config(
        page_title="VIAVI OneAdvisor – Spectrum & 5G NR",
        layout="wide",
    )

    st.title("VIAVI OneAdvisor – Spectrum & 5G NR Dashboard")

    # Sidebar: connection
    st.sidebar.header("Connection")

    host = st.sidebar.text_input(
        "Instrument IP Address",
        value="192.168.1.100",
        key="sidebar_host",
    )
    discovery_port = st.sidebar.number_input(
        "System SCPI Port",
        value=5025,
        step=1,
        key="sidebar_disc_port",
    )
    timeout = st.sidebar.number_input(
        "Timeout (s)",
        value=5.0,
        step=0.5,
        key="sidebar_timeout",
    )

    if "radio_port" not in st.session_state:
        st.session_state.radio_port = None
        st.session_state.connected_host = None

    if st.sidebar.button("Connect", key="sidebar_connect_btn"):
        radio_port, idn = connect_viavi(
            host=host,
            discovery_port=int(discovery_port),
            timeout=float(timeout),
        )
        if radio_port is not None:
            st.session_state.radio_port = radio_port
            st.session_state.connected_host = host

    if st.session_state.radio_port is None or st.session_state.connected_host is None:
        st.info("Enter IP and click **Connect** to start.")
        return

    conn_host = st.session_state.connected_host
    radio_port = st.session_state.radio_port

    tab_spec, tab_wifi, tab_nr = st.tabs(["Spectrum", "Wi-Fi", "5G NR"])

    with tab_spec:
        spectrum_tab(conn_host, radio_port, float(timeout))

    with tab_wifi:
        wifi_tab(conn_host, radio_port, float(timeout))

    with tab_nr:
        nr5g_tab(conn_host, radio_port, float(timeout))


if __name__ == "__main__":
    main()
