"""
streamlit_ui.py

Reusable Streamlit UI helpers for VIAVI OneAdvisor:

- connect_and_get_radio_client()
- render_spectrum_panel()
- render_wifi_panel()
- render_nr5g_panel()

You can import these into your own Streamlit app and mix them
with your existing Ennoia agentic UI.
"""

from typing import Optional

import streamlit as st

from .system_api import OneAdvisorSystemAPI
from .radio_api import OneAdvisorRadioAPI
from .plotting import plot_spectrum_trace, plot_wifi_scan


# -------------------- connection helper --------------------


def connect_and_get_radio_client(
    host: str, discovery_port: int = 5025, timeout: float = 5.0
) -> Optional[OneAdvisorRadioAPI]:
    """
    Convenience helper: connect to system SCPI, discover radio port,
    and return a OneAdvisorRadioAPI instance (already opened).

    Returns None on failure (and shows an error via Streamlit).
    """
    try:
        sys_api = OneAdvisorSystemAPI(host, port=discovery_port, timeout=timeout)
        sys_api.open()
    except Exception as e:
        st.error(f"Failed to connect to system SCPI ({host}:{discovery_port}): {e}")
        return None

    try:
        idn = sys_api.idn()
        st.info(f"System: {idn}")
        radio_port = sys_api.get_radio_scpi_port()
    except Exception as e:
        sys_api.close()
        st.error(f"Failed to discover Radio SCPI port via :PRTM:LIST?: {e}")
        return None
    finally:
        sys_api.close()

    try:
        ra = OneAdvisorRadioAPI(host, scpi_port=radio_port, timeout=timeout)
        ra.open()
        st.success(f"Connected to Radio SCPI on port {radio_port}")
        st.caption(f"Radio IDN: {ra.idn()}")
        return ra
    except Exception as e:
        st.error(f"Failed to connect to Radio SCPI on port {radio_port}: {e}")
        return None


# -------------------- spectrum panel --------------------


def render_spectrum_panel(
    ra: OneAdvisorRadioAPI,
    default_center_ghz: float = 3.5,
    default_span_mhz: float = 100.0,
) -> None:
    """
    Render a Spectrum Analyzer control panel with a single trace plot.
    """
    st.subheader("Spectrum Analyzer")

    col1, col2, col3 = st.columns(3)
    with col1:
        center_ghz = st.number_input(
            "Center Frequency (GHz)", value=default_center_ghz, step=0.1, format="%.3f"
        )
    with col2:
        span_mhz = st.number_input(
            "Span (MHz)", value=default_span_mhz, step=10.0, format="%.1f"
        )
    with col3:
        ref_level = st.number_input("Reference Level (dBm)", value=0.0, step=5.0)

    col_rbw, col_vbw = st.columns(2)
    with col_rbw:
        rbw_khz = st.number_input(
            "RBW (kHz)", value=100.0, step=10.0, format="%.1f"
        )
    with col_vbw:
        vbw_khz = st.number_input(
            "VBW (kHz)", value=100.0, step=10.0, format="%.1f"
        )

    if st.button("Acquire Spectrum Trace"):
        try:
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

            # Let it settle very briefly
            import time

            time.sleep(0.3)

            trace = ra.get_spectrum_trace()
            start_hz, stop_hz, _ = ra.get_spectrum_xaxis()

            if not trace:
                st.warning("No trace data received.")
                return

            st.write(f"Points: {len(trace)}, Start: {start_hz/1e9:.3f} GHz, "
                     f"Stop: {stop_hz/1e9:.3f} GHz")

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            plot_spectrum_trace(
                trace_dbm=trace,
                start_hz=start_hz,
                stop_hz=stop_hz,
                ax=ax,
                title=f"Spectrum @ {center_ghz:.3f} GHz, {span_mhz:.1f} MHz span",
            )
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error acquiring spectrum trace: {e}")


# -------------------- Wi-Fi panel --------------------


def render_wifi_panel(ra: OneAdvisorRadioAPI) -> None:
    """
    Render a simple Wi-Fi scan panel for 2.4 / 5 / 6 GHz bands.
    """
    st.subheader("Wi-Fi Band Scan (Spectrum Mode)")

    band = st.selectbox("Band", options=["2.4", "5", "6"], index=0)
    span_mhz = st.number_input("Span (MHz)", value=100.0, step=20.0, format="%.1f")
    rbw_khz = st.number_input("RBW (kHz)", value=100.0, step=10.0, format="%.1f")
    ref_level = st.number_input("Reference Level (dBm)", value=-10.0, step=5.0)

    if st.button("Scan Wi-Fi Band"):
        try:
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

            import matplotlib.pyplot as plt

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


# -------------------- 5G NR panel --------------------


def render_nr5g_panel(ra: OneAdvisorRadioAPI) -> None:
    """
    Render a 5G NR quick summary panel (OBW, ACLR, SEM, EVM).
    """
    st.subheader("5G NR Quick Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        center_ghz = st.number_input(
            "Center Frequency (GHz)", value=3.5, step=0.1, format="%.3f"
        )
    with col2:
        band = st.selectbox("Band", options=["FR1", "FR2"], index=0)
    with col3:
        ref_level = st.number_input("Reference Level (dBm)", value=0.0, step=5.0)

    col4, col5 = st.columns(2)
    with col4:
        chan_standard = st.number_input(
            "Channel Standard (band code)", value=700, step=1
        )
    with col5:
        chan_num = st.number_input("Channel Number", value=1, step=1)

    if st.button("Run 5G NR Quick Summary"):
        with st.spinner("Measuring..."):
            try:
                summary = ra.nr5g_quick_summary(
                    center_hz=center_ghz * 1e9,
                    band=band,
                    chan_standard=int(chan_standard),
                    chan_num=int(chan_num),
                    ref_level_dbm=ref_level,
                )
            except Exception as e:
                st.error(f"Error running 5G NR quick summary: {e}")
                return

        st.success("Measurement complete.")
        st.json(summary)
