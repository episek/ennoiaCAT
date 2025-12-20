"""
Instrument Adapter Classes
Provides a unified interface for different instrument types
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import streamlit as st


class InstrumentAdapter(ABC):
    """Base class for all instrument adapters"""

    def __init__(self, connection_info: Dict[str, Any]):
        self.connection_info = connection_info
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the instrument"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the instrument"""
        pass

    @abstractmethod
    def get_helper_class(self):
        """Get the helper/config class for this instrument"""
        pass

    @abstractmethod
    def render_ui(self):
        """Render instrument-specific UI elements"""
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """Get the display name for this instrument"""
        pass


class TinySAAdapter(InstrumentAdapter):
    """Adapter for TinySA Spectrum Analyzer"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.helper = None

    def connect(self) -> bool:
        try:
            # Try importing without initializing heavy dependencies
            import sys
            import importlib.util

            # Check if tinySA_config exists
            spec = importlib.util.find_spec("tinySA_config")
            if spec is None:
                st.error("TinySA configuration module not found")
                return False

            # Import the module
            from tinySA_config import TinySAHelper
            self.helper = TinySAHelper()
            self.is_connected = True
            return True
        except ImportError as e:
            st.error(f"Failed to import TinySA module: {e}")
            st.info("Some features may require PyTorch. Install with: pip install torch")
            return False
        except Exception as e:
            st.error(f"Failed to connect to TinySA: {e}")
            st.info("Try connecting without AI features enabled")
            return False

    def disconnect(self) -> bool:
        self.helper = None
        self.is_connected = False
        return True

    def get_helper_class(self):
        return self.helper

    def render_ui(self):
        st.sidebar.image('ennoia.jpg', width=200)
        st.title("TinySA Spectrum Analyzer")
        st.caption(f"Connected to {self.connection_info.get('port', 'Unknown')}")

    def get_display_name(self) -> str:
        return f"TinySA on {self.connection_info.get('port', 'Unknown')}"


class ViaviAdapter(InstrumentAdapter):
    """Adapter for Viavi OneAdvisor"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.system_api = None
        self.radio_api = None
        self.helper = None

    def connect(self) -> bool:
        try:
            from ennoia_viavi.system_api import OneAdvisorSystemAPI
            from ennoia_viavi.radio_api import OneAdvisorRadioAPI
            from viavi_config import ViaviHelper

            ip = self.connection_info.get('ip')
            self.system_api = OneAdvisorSystemAPI(ip)
            self.system_api.open()

            radio_port = self.system_api.get_radio_scpi_port()
            self.radio_api = OneAdvisorRadioAPI(ip, scpi_port=radio_port)
            self.radio_api.open()

            self.helper = ViaviHelper()
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Viavi: {e}")
            return False

    def disconnect(self) -> bool:
        try:
            if self.radio_api:
                self.radio_api.close()
            if self.system_api:
                self.system_api.close()
            self.is_connected = False
            return True
        except Exception:
            return False

    def get_helper_class(self):
        return self.helper

    def render_ui(self):
        import time
        import json
        import ast
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from types import SimpleNamespace

        st.sidebar.image("viavi.png", width=200)
        st.sidebar.image("ennoia.jpg", width=200)
        st.title("🗼 Viavi OneAdvisor Agentic AI Control & Analysis")
        st.caption(f"Connected to {self.connection_info.get('ip', 'Unknown')}")

        # Sidebar settings
        st.sidebar.subheader("🔁 Sweep Settings")
        continuous_mode = st.sidebar.checkbox("Continuous Sweep", value=True)
        num_sweeps = st.sidebar.number_input(
            "Number of sweeps",
            min_value=1,
            max_value=200,
            value=10,
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
            index=1,  # Default to Multi-sweep average
        )

        # Session state for waterfall / averaging
        if "waterfall_data" not in st.session_state:
            st.session_state["waterfall_data"] = None
        if "waterfall_freqs" not in st.session_state:
            st.session_state["waterfall_freqs"] = None
        if "last_trace" not in st.session_state:
            st.session_state["last_trace"] = None
        if "last_freqs" not in st.session_state:
            st.session_state["last_freqs"] = None

        # Check if we have the necessary objects in session_state for spectrum acquisition
        if 'viavi_user_prompt' in st.session_state and st.session_state['viavi_user_prompt']:
            prompt = st.session_state['viavi_user_prompt']
            st.session_state['viavi_user_prompt'] = None  # Clear it

            # Check if we're using SLM or LLM mode
            use_slm = "SLM (Offline)" in st.session_state.get('ai_options', [])

            # Create system prompt - simpler for SLM, more detailed for OpenAI
            if use_slm:
                system_content = (
                    "Extract start and stop frequencies from the request. "
                    "Output JSON only: {\"start\": \"VALUE\", \"stop\": \"VALUE\"}\n"
                    "Example: 'scan 600 to 900 MHz' -> {\"start\": \"600 MHz\", \"stop\": \"900 MHz\"}"
                )
            else:
                system_content = (
                    "You are a helpful assistant that converts natural language into JSON configuration.\n"
                    "Extract frequency parameters from the user's request and output ONLY a valid JSON object.\n"
                    "Available keys: start, stop, center, span, rbw, vbw, ref_level.\n"
                    "- Frequencies can be in Hz, MHz, or GHz (e.g., '600 MHz', '3.5 GHz')\n"
                    "- Only include keys that are mentioned in the request\n"
                    "- Do NOT add explanations, just output the JSON object\n\n"
                    "Example input: 'scan from 600 MHz to 900 MHz'\n"
                    "Example output: {\"start\": \"600 MHz\", \"stop\": \"900 MHz\"}\n\n"
                    "Example input: 'scan 3.3 to 3.7 GHz'\n"
                    "Example output: {\"start\": \"3.3 GHz\", \"stop\": \"3.7 GHz\"}"
                )

            # Get API options from AI
            api_chat = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

            try:
                st.info(f"Processing with {'SLM' if use_slm else 'OpenAI'} mode")

                if use_slm:
                    # Use local SLM model
                    if 'map_api' in st.session_state and st.session_state['map_api']:
                        map_api = st.session_state['map_api']
                        st.write("Using SLM model for spectrum configuration...")

                        # Check if the map_api has the necessary attributes
                        if not hasattr(map_api, 'tokenizer') or not hasattr(map_api, 'model'):
                            st.error("SLM model not properly initialized. Missing tokenizer or model.")
                            st.info("Falling back to OpenAI API...")
                            # Fall back to OpenAI
                            from map_api_vi import MapAPI
                            temp_api = MapAPI()
                            api_raw = temp_api.generate_response(api_chat)
                        else:
                            api_raw = map_api.generate_response(api_chat)
                            st.write(f"SLM Response: {api_raw}")
                    else:
                        st.error("SLM model not loaded. Please enable SLM in AI options.")
                        st.info("Falling back to OpenAI API...")
                        from map_api_vi import MapAPI
                        temp_api = MapAPI()
                        api_raw = temp_api.generate_response(api_chat)
                else:
                    # Use OpenAI client
                    if 'openai_client' in st.session_state and st.session_state['openai_client']:
                        client = st.session_state['openai_client']
                        model = st.session_state.get('openai_model', 'gpt-4o-mini')

                        openAImessage = client.chat.completions.create(
                            model=model,
                            messages=api_chat,
                            temperature=0,
                            max_tokens=200,
                            frequency_penalty=1,
                            stream=False
                        )
                        api_raw = openAImessage.choices[0].message.content
                        st.write(f"OpenAI Response: {api_raw}")
                    else:
                        # Fallback: use map_api_vi which is designed for OpenAI
                        from map_api_vi import MapAPI
                        map_api = MapAPI()
                        api_raw = map_api.generate_response(api_chat)

                api_dict = {}

                # Try to parse the response
                try:
                    # Clean the response - sometimes models add markdown formatting
                    cleaned_response = api_raw.strip()
                    if cleaned_response.startswith("```json"):
                        cleaned_response = cleaned_response[7:]
                    if cleaned_response.startswith("```"):
                        cleaned_response = cleaned_response[3:]
                    if cleaned_response.endswith("```"):
                        cleaned_response = cleaned_response[:-3]
                    cleaned_response = cleaned_response.strip()

                    # Try to extract JSON from text (look for { ... })
                    import re
                    json_match = re.search(r'\{[^{}]*\}', cleaned_response, re.DOTALL)
                    if json_match:
                        cleaned_response = json_match.group(0)

                    api_dict = json.loads(cleaned_response)
                    st.success(f"✓ Parsed config: {api_dict}")

                except json.JSONDecodeError as e:
                    st.warning(f"JSON parsing failed, trying ast.literal_eval...")
                    try:
                        # Try extracting JSON again with a more greedy pattern
                        json_match = re.search(r'\{.*\}', api_raw, re.DOTALL)
                        if json_match:
                            api_dict = ast.literal_eval(json_match.group(0))
                            st.success(f"✓ Parsed config (via ast): {api_dict}")
                        else:
                            raise ValueError("No JSON object found in response")
                    except Exception as e2:
                        st.warning(f"Structured parsing failed. Using natural language parser...")
                        st.info(f"Raw AI response: {api_raw[:200]}...")
                        # Don't fail - we'll use the natural language parser below
                        api_dict = {}

                opt = SimpleNamespace(**api_dict) if api_dict else SimpleNamespace()

                # Natural-language override for start/stop if missing or if JSON parsing failed
                from viavi_config import ViaviHelper
                nls, nle = ViaviHelper.extract_start_stop(prompt)

                # Apply natural language extraction if we don't have values from JSON
                if getattr(opt, "start", None) is None and nls is not None:
                    opt.start = nls
                    st.info(f"Extracted start frequency from text: {nls / 1e6:.1f} MHz")
                if getattr(opt, "stop", None) is None and nle is not None:
                    opt.stop = nle
                    st.info(f"Extracted stop frequency from text: {nle / 1e6:.1f} MHz")

                # Check if we have at least some configuration to work with
                if not hasattr(opt, "start") and not hasattr(opt, "stop") and not hasattr(opt, "center"):
                    st.warning("Could not extract frequency configuration from the request.")
                    st.info("Using default scan: 300 MHz to 900 MHz")
                    # Set default frequencies
                    opt.start = 300e6  # 300 MHz in Hz
                    opt.stop = 900e6   # 900 MHz in Hz

                # Perform spectrum acquisition
                self._acquire_spectrum(opt, continuous_mode, num_sweeps, sweep_delay, display_mode)

            except Exception as e:
                st.error(f"Failed to process spectrum command: {e}")
                import traceback
                st.error(traceback.format_exc())

    def _acquire_spectrum(self, opt, continuous_mode, num_sweeps, sweep_delay, display_mode):
        """Acquire and display spectrum from Viavi OneAdvisor"""
        import time
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from viavi_config import ViaviHelper

        st.subheader("🗼 Spectrum Analyzer")

        freqs = []
        trace = []

        try:
            ra = self.radio_api

            start_hz = ViaviHelper.parse_freq(getattr(opt, "start", None))
            stop_hz  = ViaviHelper.parse_freq(getattr(opt, "stop", None))
            center   = ViaviHelper.parse_freq(getattr(opt, "center", None))
            span     = ViaviHelper.parse_freq(getattr(opt, "span", None))

            st.write(f"DEBUG - Parsed start: {start_hz}, stop: {stop_hz}")

            # normalize scalars
            if isinstance(start_hz, (list, tuple, np.ndarray)):
                start_hz = ViaviHelper.parse_freq(start_hz[0])
            if isinstance(stop_hz, (list, tuple, np.ndarray)):
                stop_hz = ViaviHelper.parse_freq(stop_hz[0])

            if center is None and start_hz is not None and stop_hz is not None:
                center = 0.5 * (float(start_hz) + float(stop_hz))
            if span is None and start_hz is not None and stop_hz is not None:
                span = abs(float(stop_hz) - float(start_hz))

            if center is None:
                center = 600e6
            if span is None:
                span = 600e6

            st.info(f"Configuring spectrum: Center={center/1e6:.1f} MHz, Span={span/1e6:.1f} MHz")
            st.info(f"Frequency range: {(center-span/2)/1e6:.1f} - {(center+span/2)/1e6:.1f} MHz")

            rbw = ViaviHelper.parse_freq(getattr(opt, "rbw", None)) or ViaviHelper.auto_rbw(span)
            vbw = ViaviHelper.parse_freq(getattr(opt, "vbw", None)) or rbw
            try:
                ref_level = float(getattr(opt, "ref_level", 0))
            except Exception:
                ref_level = 0.0

            # Store these for use in one_sweep function
            config_center = center
            config_span = span

            # Explicitly enter Spectrum Analyzer mode
            ra.set_spectrum_mode("spectrumTuned")
            time.sleep(0.2)

            def configure_spectrum():
                ra.configure_spectrum(
                    center_hz=config_center,
                    span_hz=config_span,
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

                # Try to get frequency axis from instrument
                try:
                    start_axis, stop_axis, _ = ra.get_spectrum_xaxis()
                    st.write(f"DEBUG - get_spectrum_xaxis returned: start={start_axis}, stop={stop_axis}")
                except Exception as e:
                    st.error(f"Error getting spectrum xaxis: {e}")
                    start_axis = 0
                    stop_axis = 0

                # If instrument returns invalid axis, calculate from center/span
                if abs(start_axis) < 1 and abs(stop_axis) < 1:  # More robust check for ~0
                    st.warning("⚠️ Instrument returned invalid frequency axis, calculating from configuration")
                    start_axis = config_center - (config_span / 2)
                    stop_axis = config_center + (config_span / 2)
                    st.write(f"✓ Using calculated range: {start_axis/1e6:.1f} to {stop_axis/1e6:.1f} MHz")

                fq = np.linspace(start_axis, stop_axis, len(tr)).tolist()
                tr = list(tr)

                # Debug: show actual frequency range
                if fq:
                    st.write(f"DEBUG - Final frequency axis: {min(fq)/1e6:.1f} to {max(fq)/1e6:.1f} MHz ({len(fq)} points)")
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

                freq_mhz = [f / 1e6 for f in freqs]

                # PLOT according to display mode
                fig, ax = plt.subplots()

                if display_mode == "Multi-sweep average" and len(all_traces) > 1:
                    avg_trace = np.mean(np.array(all_traces), axis=0)
                    ax.plot(freq_mhz, avg_trace, label="Avg")
                    ax.plot(freq_mhz, trace, alpha=0.3, label="Last")
                    ax.legend()
                elif display_mode == "Waterfall" and st.session_state["waterfall_data"] is not None:
                    wf_mhz = [f / 1e6 for f in st.session_state["waterfall_freqs"]]
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
                    peaks = ViaviHelper.find_peaks(trace, max_peaks=5, min_dist=5)
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

            # Display analysis tables
            self._display_analysis(freqs, trace)

        except Exception as e:
            st.error(f"Viavi error: {e}")

    def _display_analysis(self, freqs, trace):
        """Display spectrum analysis tables and data"""
        import pandas as pd
        from viavi_config import ViaviHelper

        if not freqs or not trace:
            return

        freq_mhz = [f / 1e6 for f in freqs]

        # Cellular Peaks & 5G NR Detection
        st.subheader("📶 Cellular Peaks & 5G NR-like Carriers")

        peaks = ViaviHelper.find_peaks(trace, max_peaks=10, min_dist=5)
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

        nr_carriers = ViaviHelper.detect_5gnr_like_carriers(freqs, trace)
        if nr_carriers:
            st.write("5G NR-like Carriers (rough RF-only heuristic):")
            df_nr = pd.DataFrame(nr_carriers)
            st.dataframe(df_nr)

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
            if rows:
                csv_peaks = df_peaks.to_csv(index=False)
                st.download_button(
                    "📥 Download Peaks CSV",
                    data=csv_peaks,
                    file_name="viavi_peaks.csv",
                    mime="text/csv",
                )

        # Operator Table Mapping
        st.subheader("🗼 Operator Table (by detected cellular bands)")

        operator_table = self.helper.get_operator_frequencies()
        if operator_table:
            try:
                op_report = self.helper.analyze_signal_peaks(
                    sstr=trace,
                    freq_mhz=freq_mhz,
                    operator_table=operator_table,
                    window_size=5,
                    peak_height=-75,
                    peak_distance=10
                )
            except Exception as e:
                op_report = []
                st.error(f"Operator analysis error: {e}")

            if op_report:
                df_ops = pd.DataFrame(op_report)
                st.dataframe(df_ops)
                csv_ops = df_ops.to_csv(index=False)
                st.download_button(
                    "📥 Download Operator Table Matches CSV",
                    data=csv_ops,
                    file_name="operator_matches.csv",
                    mime="text/csv",
                )
            else:
                st.info("No strong operator bands detected in the current span.")

        # Bandpower Measurement
        st.subheader("📐 Bandpower Measurement")

        col_bp1, col_bp2, col_bp3 = st.columns(3)
        with col_bp1:
            bp_f1_mhz = st.number_input("Band start (MHz)", value=freqs[0] / 1e6)
        with col_bp2:
            bp_f2_mhz = st.number_input("Band stop (MHz)", value=freqs[-1] / 1e6)
        with col_bp3:
            if st.button("Compute Bandpower"):
                f1_hz = bp_f1_mhz * 1e6
                f2_hz = bp_f2_mhz * 1e6
                bp_dbm = ViaviHelper.bandpower_linear(freqs, trace, f1_hz, f2_hz)
                if bp_dbm is None:
                    st.warning("No data in this band or invalid range.")
                else:
                    st.success(f"Approx. bandpower in [{bp_f1_mhz:.3f}, {bp_f2_mhz:.3f}] MHz: {bp_dbm:.2f} dBm")

        # WiFi Detection
        st.subheader("📶 WiFi-like RF Carriers (from SA trace)")

        wifi_rf = ViaviHelper.detect_wifi_like_carriers(freqs, trace)
        span_bands = ViaviHelper.classify_span_wifi_bands(freqs)

        if span_bands:
            st.info(f"Current span overlaps WiFi band(s): {', '.join(span_bands)}")

        if wifi_rf:
            df_wifi_rf = pd.DataFrame(wifi_rf)
            st.dataframe(df_wifi_rf)
        else:
            st.info("No obvious WiFi-like peaks in the current span.")

    def get_display_name(self) -> str:
        return f"Viavi OneAdvisor @ {self.connection_info.get('ip', 'Unknown')}"


class MavenirAdapter(InstrumentAdapter):
    """Adapter for Mavenir 5G NR Radio Unit"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.helper = None

    def connect(self) -> bool:
        try:
            from mav_config import TinySAHelper
            self.helper = TinySAHelper()
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Mavenir RU: {e}")
            return False

    def disconnect(self) -> bool:
        self.helper = None
        self.is_connected = False
        return True

    def get_helper_class(self):
        return self.helper

    def render_ui(self):
        st.sidebar.image('mavenir_logo.png', width=200)
        st.title("Mavenir 5G NR Radio Unit")
        st.caption(f"Connected to {self.connection_info.get('ip', 'Unknown')}")

    def get_display_name(self) -> str:
        return f"Mavenir 5G NR RU @ {self.connection_info.get('ip', 'Unknown')}"


class CiscoAdapter(InstrumentAdapter):
    """Adapter for Cisco NCS540"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.helper = None
        self.conn = None

    def connect(self) -> bool:
        try:
            from CS_config import CSHelper
            self.helper = CSHelper()
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Cisco NCS540: {e}")
            return False

    def disconnect(self) -> bool:
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
        self.helper = None
        self.is_connected = False
        return True

    def get_helper_class(self):
        return self.helper

    def render_ui(self):
        st.sidebar.image('cisco_logo.png', width=200)
        st.sidebar.image('ennoia_white_black_hi-def.png', width=200)
        st.title("Cisco NCS540")
        st.caption(f"Connected via {self.connection_info.get('port', 'Unknown')}")

    def get_display_name(self) -> str:
        return f"Cisco NCS540 on {self.connection_info.get('port', 'Unknown')}"


class KeysightAdapter(InstrumentAdapter):
    """Adapter for Keysight FieldFox"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.inst = None

    def connect(self) -> bool:
        try:
            import pyvisa
            rm = pyvisa.ResourceManager()
            resource = self.connection_info.get('resource')
            self.inst = rm.open_resource(resource)
            self.inst.read_termination = '\n'
            self.inst.write_termination = '\n'
            self.inst.timeout = 5000
            self.inst.write(":INSTrument:SELect 'SA'")
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Keysight: {e}")
            return False

    def disconnect(self) -> bool:
        try:
            if self.inst:
                self.inst.close()
            self.is_connected = False
            return True
        except Exception:
            return False

    def get_helper_class(self):
        return self.inst

    def render_ui(self):
        st.sidebar.image('ennoia.jpg', width=200)
        st.title("Keysight FieldFox Spectrum Analyzer")
        st.caption(f"Connected to {self.connection_info.get('ip', 'Unknown')}")

    def get_display_name(self) -> str:
        return f"Keysight FieldFox @ {self.connection_info.get('ip', 'Unknown')}"


class RohdeSchwarzAdapter(InstrumentAdapter):
    """Adapter for Rohde & Schwarz NRQ6"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.inst = None
        self.helper = None

    def connect(self) -> bool:
        try:
            from RsInstrument import RsInstrument
            from RS_config import RSHelper

            resource = self.connection_info.get('resource')
            self.inst = RsInstrument(resource, id_query=True, reset=True)
            self.helper = RSHelper()
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Rohde & Schwarz: {e}")
            return False

    def disconnect(self) -> bool:
        try:
            if self.inst:
                self.inst.close()
            self.helper = None
            self.is_connected = False
            return True
        except Exception:
            return False

    def get_helper_class(self):
        return {
            'instrument': self.inst,
            'helper': self.helper
        }

    def render_ui(self):
        st.sidebar.image('RS_logo.png', width=200)
        st.title("Rohde & Schwarz NRQ6")
        st.caption(f"Connected via {self.connection_info.get('resource', 'Unknown')}")

    def get_display_name(self) -> str:
        return f"Rohde & Schwarz via {self.connection_info.get('resource', 'Unknown')}"


class AukuaAdapter(InstrumentAdapter):
    """Adapter for Aukua Systems"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.helper = None

    def connect(self) -> bool:
        try:
            from AK_config import AKHelper
            self.helper = AKHelper()
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Aukua: {e}")
            return False

    def disconnect(self) -> bool:
        self.helper = None
        self.is_connected = False
        return True

    def get_helper_class(self):
        return self.helper

    def render_ui(self):
        st.sidebar.image('aukua rgb high.jpg', width=200)
        st.sidebar.image('ennoia_white_black_hi-def.png', width=200)
        st.title("Aukua Systems")
        st.caption(f"Connected via {self.connection_info.get('resource', 'Unknown')}")

    def get_display_name(self) -> str:
        return f"Aukua via {self.connection_info.get('resource', 'Unknown')}"


class AdapterFactory:
    """Factory for creating instrument adapters"""

    @staticmethod
    def create_adapter(instrument_type, connection_info: Dict[str, Any]) -> Optional[InstrumentAdapter]:
        """Create an adapter for the given instrument type"""
        from instrument_detector import InstrumentType

        adapter_map = {
            InstrumentType.TINYSA: TinySAAdapter,
            InstrumentType.VIAVI: ViaviAdapter,
            InstrumentType.MAVENIR_RU: MavenirAdapter,
            InstrumentType.CISCO_NCS540: CiscoAdapter,
            InstrumentType.KEYSIGHT: KeysightAdapter,
            InstrumentType.ROHDE_SCHWARZ: RohdeSchwarzAdapter,
            InstrumentType.AUKUA: AukuaAdapter,
        }

        adapter_class = adapter_map.get(instrument_type)
        if adapter_class:
            return adapter_class(connection_info)
        return None
