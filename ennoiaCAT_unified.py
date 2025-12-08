"""
EnnoiaCAT Unified - Multi-Instrument Test Platform
Automatically detects and supports multiple test instruments
"""
# import ennoia_client_lic as lic  # Disabled for now
import argparse
import json
import ast
import streamlit as st
from types import SimpleNamespace
import pandas as pd
from timer import Timer, fmt_seconds
from instrument_detector import InstrumentDetector, InstrumentType
from instrument_adapters import AdapterFactory

# License verification - DISABLED
# parser = argparse.ArgumentParser(description="Ennoia License Client")
# parser.add_argument(
#     "--action",
#     choices=["activate", "verify"],
#     default="verify",
#     help="Action to perform (default: verify)"
# )
# parser.add_argument("--key", help="Ennoia License key for activation")
# args = parser.parse_args()

# if args.action == "activate":
#     if not args.key:
#         print("❗ Please provide a license key with --key")
#         success = False
#     else:
#         success = lic.request_license(args.key)
# elif args.action == "verify":
#     success = lic.verify_license_file()
# else:
#     success = lic.verify_license_file()

# if not success:
#     print("❌ License verification failed. Please check your license key or contact support.")

# Bypass license check for testing
success = True
print("WARNING: License verification disabled for testing")

# Streamlit configuration
st.set_page_config(page_title="Ennoia Technologies", page_icon="🤖")

# Initialize session state
if "detected_instruments" not in st.session_state:
    st.session_state.detected_instruments = []
if "selected_instrument" not in st.session_state:
    st.session_state.selected_instrument = None
if "instrument_adapter" not in st.session_state:
    st.session_state.instrument_adapter = None
if "detection_done" not in st.session_state:
    st.session_state.detection_done = False

# Main header
st.sidebar.image('ennoia.jpg')
st.title("Ennoia Technologies")
st.markdown(
    """
    Multi-Instrument Test Platform with AI Assistant ©. All rights reserved.
    """
)

# License status
if not success:
    st.error("Ennoia License verification failed. Please check your license key or contact support.")
    #st.stop()
else:
    st.warning("⚠️ License verification disabled for testing")

# Instrument Detection Section
st.header("🔍 Instrument Detection")

# Add IP configuration expander
with st.expander("⚙️ Network Instrument Configuration (Optional)"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Viavi IPs")
        viavi_ip = st.text_input("Viavi IP Address", value="192.168.1.100", key="viavi_ip")
        viavi_ips = [viavi_ip] if viavi_ip else None

    with col2:
        st.subheader("Keysight IPs")
        keysight_ip = st.text_input("Keysight IP Address", value="192.168.1.100", key="keysight_ip")
        keysight_ips = [keysight_ip] if keysight_ip else None

    with col3:
        st.subheader("Mavenir RU IPs")
        mavenir_ip = st.text_input("Mavenir IP Address", value="10.10.10.10", key="mavenir_ip")
        mavenir_ips = [mavenir_ip] if mavenir_ip else None

# Detection button
if st.button("🔍 Detect Instruments", type="primary"):
    with st.spinner("Detecting instruments..."):
        detector = InstrumentDetector()
        st.session_state.detected_instruments = detector.detect_all(
            viavi_ips=viavi_ips,
            keysight_ips=keysight_ips,
            mavenir_ips=mavenir_ips
        )
        st.session_state.detection_done = True

# Display detected instruments
if st.session_state.detection_done:
    if st.session_state.detected_instruments:
        st.success(f"✅ Found {len(st.session_state.detected_instruments)} instrument(s)")

        # Create selection dropdown
        instrument_options = {
            inst.display_name: inst
            for inst in st.session_state.detected_instruments
        }

        selected_name = st.selectbox(
            "Select Instrument to Use:",
            options=["-- Select an instrument --"] + list(instrument_options.keys()),
            key="instrument_selector"
        )

        if selected_name != "-- Select an instrument --":
            selected_inst = instrument_options[selected_name]

            # Display instrument info
            st.info(f"**Type:** {selected_inst.instrument_type.value}")
            st.info(f"**Connection:** {selected_inst.connection_info}")

            # Connect button
            if st.button("🔌 Connect to Instrument"):
                with st.spinner("Connecting..."):
                    adapter = AdapterFactory.create_adapter(
                        selected_inst.instrument_type,
                        selected_inst.connection_info
                    )

                    if adapter and adapter.connect():
                        st.session_state.selected_instrument = selected_inst
                        st.session_state.instrument_adapter = adapter
                        st.success(f"✅ Connected to {selected_name}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to connect to instrument")
    else:
        st.warning("⚠️ No instruments detected. Please check connections and try again.")

        # Manual selection option
        st.subheader("Manual Instrument Selection")
        st.info("If auto-detection failed, you can manually select an instrument type:")

        manual_type = st.selectbox(
            "Instrument Type:",
            options=["-- Select --"] + [itype.value for itype in InstrumentType],
            key="manual_type"
        )

        if manual_type != "-- Select --":
            st.warning("⚠️ Manual configuration not fully implemented yet. Please ensure instrument is properly connected for auto-detection.")

# Main application section (only show if connected)
if st.session_state.instrument_adapter and st.session_state.instrument_adapter.is_connected:
    st.markdown("---")

    # Render instrument-specific UI
    adapter = st.session_state.instrument_adapter
    adapter.render_ui()

    # Common AI configuration section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Configuration")

    # AI mode selection (common to all instruments that support it)
    ai_options = st.sidebar.multiselect(
        "Select AI Options:",
        ["SLM (Offline)", "Online LLM"],
        default=["Online LLM"],
        key="ai_options"
    )

    # Initialize AI based on selection
    use_slm = "SLM (Offline)" in ai_options

    # Load appropriate AI model
    if use_slm:
        try:
            # Check if torch is available
            try:
                import torch
                torch_available = True
            except ImportError:
                torch_available = False
                st.error("PyTorch not installed. SLM mode requires PyTorch.")
                st.info("Install with: pip install torch torchvision torchaudio")
                st.info("Falling back to online LLM mode")
                use_slm = False

            if torch_available:
                from tinySA_config import TinySAHelper

                @st.cache_resource
                def load_model_and_tokenizer():
                    return TinySAHelper.load_lora_model()

                st.write("⏳ Loading offline model...")
                tokenizer, peft_model, device = load_model_and_tokenizer()
                st.write(f"✅ Local model loaded on {device}")

                from map_api import MapAPI
                map_api = MapAPI(peft_model, tokenizer)

        except Exception as e:
            st.error(f"Failed to load SLM: {e}")
            st.info("Falling back to online LLM")
            use_slm = False

    if not use_slm:
        try:
            from tinySA_config import TinySAHelper
            client, ai_model = TinySAHelper.load_OpenAI_model()

            from map_api import MapAPI
            map_api = MapAPI()

            if "openai_model" not in st.session_state:
                st.session_state["openai_model"] = ai_model
            st.write(f"✅ Online LLM model {ai_model} loaded")

        except Exception as e:
            st.error(f"Failed to load OpenAI: {e}")
            st.warning("AI features may not be available")
            # Don't stop - continue without AI
            map_api = None

    # Get helper class from adapter
    helper = adapter.get_helper_class()

    # TinySA-specific initialization
    if st.session_state.selected_instrument.instrument_type.value == "TinySA Spectrum Analyzer":
        if "tinySA_port" not in st.session_state:
            try:
                st.session_state.tinySA_port = helper.getport()
            except Exception as e:
                st.warning(f"Could not auto-detect TinySA port: {e}")

    # Try to get system prompt and examples (instrument-specific)
    try:
        if hasattr(helper, 'get_system_prompt'):
            system_prompt = helper.get_system_prompt()
        else:
            system_prompt = "You are an AI assistant for test and measurement equipment."

        if hasattr(helper, 'get_few_shot_examples'):
            few_shot_examples = helper.get_few_shot_examples()
        else:
            few_shot_examples = []
    except Exception:
        system_prompt = "You are an AI assistant for test and measurement equipment."
        few_shot_examples = []

    # Chat interface
    st.markdown("---")
    st.subheader("💬 AI Assistant")
    st.write("Hi. I am Ennoia, your AI assistant. How can I help you today?")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask Ennoia:")

    if prompt:
        t = Timer()
        t.start()

        # Store user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Generate response
            user_input = st.session_state.messages[-1]["content"]

            chat1 = [{"role": "system", "content": system_prompt}] + few_shot_examples + [
                {"role": "user", "content": user_input}
            ]

            if use_slm:
                response = map_api.generate_response(chat1)
            else:
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

        # Additional instrument-specific processing
        # TinySA-specific plotting and signal analysis
        if st.session_state.selected_instrument and \
           st.session_state.selected_instrument.instrument_type.value == "TinySA Spectrum Analyzer":

            try:
                import json
                import ast
                from types import SimpleNamespace

                # Get default options from map_api
                @st.cache_data
                def get_default_options():
                    return map_api.get_defaults_opts()

                def_dict = get_default_options()
                few_shot_examples2 = map_api.get_few_shot_examples()

                # Get API options from AI
                system_prompt2 = map_api.get_system_prompt(def_dict, user_input)
                chat2 = [{"role": "system", "content": system_prompt2}] + few_shot_examples2 + [
                    {"role": "user", "content": user_input}
                ]

                if use_slm:
                    api_str = map_api.generate_response(chat2)
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

                # Parse response into options dictionary
                def_dict["save"] = True
                api_dict = def_dict
                try:
                    parsed = json.loads(api_str)
                    if isinstance(parsed, dict):
                        api_dict = parsed
                        api_dict["save"] = True
                except json.JSONDecodeError:
                    try:
                        parsed = ast.literal_eval(api_str)
                        if isinstance(parsed, dict):
                            api_dict = parsed
                            api_dict["save"] = True
                    except Exception:
                        st.warning("Using default TinySA options")

                # Configure TinySA and plot
                if isinstance(api_dict, dict):
                    opt = SimpleNamespace(**api_dict)
                    gcf = helper.configure_tinySA(opt)
                    if gcf:
                        st.pyplot(gcf)

                # Read and analyze signal strength
                result = helper.read_signal_strength('max_signal_strengths.csv')
                if result:
                    sstr, freq = result
                    freq_mhz = [x / 1e6 for x in freq]

                    operator_table = helper.get_operator_frequencies()
                    if operator_table:
                        frequency_report_out = helper.analyze_signal_peaks(sstr, freq_mhz, operator_table)

                        if frequency_report_out:
                            # Display frequency analysis table
                            df = pd.DataFrame(frequency_report_out)
                            st.dataframe(df)
                        else:
                            st.write("No strong trained frequency band seen.")
                    else:
                        st.warning("Operator table could not be loaded.")
                else:
                    st.warning("Could not read signal strength data.")

            except Exception as e:
                st.error(f"TinySA processing error: {str(e)}")

        t.stop()
        st.write(f"⏱️ Elapsed: {fmt_seconds(t.elapsed())}")
        t.reset()

else:
    # Show instructions when not connected
    if not st.session_state.detection_done:
        st.info("👆 Click 'Detect Instruments' above to get started")
    else:
        st.info("👆 Select and connect to an instrument above to continue")

# Disconnect button (in sidebar if connected)
if st.session_state.instrument_adapter and st.session_state.instrument_adapter.is_connected:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔌 Disconnect Instrument", type="secondary"):
        if st.session_state.instrument_adapter.disconnect():
            st.session_state.instrument_adapter = None
            st.session_state.selected_instrument = None
            st.success("✅ Disconnected")
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("EnnoiaCAT Unified v1.0")
st.sidebar.caption("© Ennoia Technologies. All rights reserved.")
