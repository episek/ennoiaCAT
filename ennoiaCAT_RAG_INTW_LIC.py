
import ennoia_client_lic as lic 
import argparse

parser = argparse.ArgumentParser(description="Ennoia License Client")
parser.add_argument(
    "--action",
    choices=["activate", "verify"],
    default="verify",
    help="Action to perform (default: verify)"
)
parser.add_argument("--key", help="Ennoia License key for activation")
args = parser.parse_args()


if args.action == "activate":
    if not args.key:
        print("❗ Please provide a license key with --key")
        success = False
    else:
        lic.request_license(args.key)
        success = True
elif args.action == "verify":
    lic.verify_license_file()
    success = True
else:
    lic.verify_license_file()
    success = True

# Development mode: bypass license check
success = 1
  
if not success:
    print("❌ License verification failed. Please check your license key or contact support.")
    exit()
    
import json
import ast
import streamlit as st
from tinySA_config import TinySAHelper
from map_api import MapAPI
from types import SimpleNamespace
import pandas as pd
import pywifi
import time


    # Define option descriptions for reference
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
    "save": "write output to CSV file"
}

st.set_page_config(page_title="Ennoia Technologies", page_icon="🤖")
st.sidebar.image('ennoia.jpg')
st.title("Ennoia Technologies")
st.markdown(
    """ 
    Chat and Test with Ennoia Connect Platform ©. All rights reserved. 
    """
)

if not success:
    st.error("Ennoia License verification failed. Please check your license key or contact support.")
    st.stop()
else:
    st.success("Ennoia License verified successfully.")

# --- App logic starts here ---
selected_options = TinySAHelper.select_checkboxes()
st.success(f"You selected: {', '.join(selected_options) if selected_options else 'nothing'}")


# --- Caching the model and tokenizer ---

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

# --- Get and cache the TinySA port ---
if "tinySA_port" not in st.session_state:
    st.session_state.tinySA_port = helper.getport()

if "SLM" in selected_options:
    st.write(f"\n✅ Local SLM model {peft_model.config.name_or_path} loaded & device found! Let's get to work.\n")
else:
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = ai_model
    st.write(f"\n✅ Online LLM model {ai_model} loaded & device! Let's get to work.\n")
# Initialize TinySA device

st.write("Hi. I am Ennoia, your AI assistant. How can I help you today?")

# Initialize session state

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask Ennoia:")

if prompt:
    # Store the user message in session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Construct the full prompt with context (system + user message)
        user_input = st.session_state.messages[-1]["content"]
        
        chat1 = [{"role": "system", "content": system_prompt}] + few_shot_examples + [{"role": "user", "content": user_input}]
        if "SLM" in selected_options:
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
        # Display the streamed response from the assistant
        st.markdown(response)
        
        # Save the assistant's response in session state
        st.session_state.messages.append({"role": "assistant", "content": response})


    system_prompt2 = map_api.get_system_prompt(def_dict,user_input)
    chat2 = [{"role": "system", "content": system_prompt2}] + few_shot_examples2 + [{"role": "user", "content": user_input}]
    if "SLM" in selected_options:
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
    #st.markdown(api_str)
    # Parse response safely into a dictionary
    def_dict["save"] = True
    print(f"\nSave output response:\n{def_dict}")
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
        except (ValueError, SyntaxError):
            print("Warning: Failed to parse response as a valid dictionary. Using default options.")

    print(f"\nParsed API options:\n{api_dict}")

    # Ensure it's a dict before using SimpleNamespace
    if isinstance(api_dict, dict):
        opt = SimpleNamespace(**api_dict)
        print(f"opt = {opt}")
        gcf = helper.configure_tinySA(opt)
        st.pyplot(gcf)
    else:
        st.error("API response is not a valid dictionary. Setting default options.")
 
    try:
        result = helper.read_signal_strength('max_signal_strengths.csv')
        if not result:
            st.error("Could not read signal strength data.")

        sstr, freq = result
        freq_mhz = [x / 1e6 for x in freq]
        print(f"\nSignal strengths: {sstr}")
        print(f"\nFrequencies: {freq_mhz}")
        
        operator_table = helper.get_operator_frequencies()
        if not operator_table:
            st.error("Operator table could not be loaded.")

        frequency_report_out = helper.analyze_signal_peaks(sstr, freq_mhz, operator_table)
        print(f"\nFrequency report: {frequency_report_out}")
        if not frequency_report_out:
            st.write("No strong trained frequency band seen.")

    except Exception as e:
        st.error(f"Failed to process request: {str(e)}")
    
    st.subheader("🗼 List of Available Cellular Networks")
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(frequency_report_out)

    # Display as a table in Streamlit
    st.dataframe(df)  # Interactive table
    
    
    
    


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
        except (ValueError, TypeError, ZeroDivisionError):
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
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        return None

    def is_dfs_channel(channel):
        try:
            ch = int(channel)
        except (ValueError, TypeError):
            return False

        # Known DFS channel ranges for 5 GHz
        if 52 <= ch <= 64 or 100 <= ch <= 144:
            return True
        return False

    def infer_bandwidth(channel, radio_type):
        try:
            ch = int(channel)
        except (ValueError, TypeError):
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

    if any(x >= 2.39e9 for x in freq):

    # with st.expander("📡 WiFi Scanner"):
        # if st.button("🔍 Scan for WiFi Networks"):
        df = scan_wifi()
        st.subheader("📶 List of Available WiFi Networks")
        st.caption("Below are the scanned WiFi networks, including signal strength, frequency, and estimated bandwidth.")
        if df.empty:
            st.warning("No networks found.")
        else:
            st.success(f"Found {len(df)} networks.")
            st.dataframe(df)
            st.download_button("📥 Download CSV", data=df.to_csv(index=False), file_name="wifi_scan.csv", mime="text/csv")








