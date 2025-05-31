"""
# GitHub examples repository path: not known yet

This Python example shows how to capture IQ sample files using the R&S NRQ6 as receiver.
The script will perform two captures and in particular record the timing.

Preconditions:
- Installed RsInstrument Python module from pypi.org
- Installed VISA e.g. R&S Visa 7.2.x or newer

Tested with:
- NRQ6, FW V02.40.23032501
- Python 3.12
- RsInstrument 1.60.0

Author: R&S Product Management AE 1GP3 / PJ
Updated on 10.04.2024
Version: v1.0

Technical support -> https://www.rohde-schwarz.com/support

Before running, please always check your setup!
This example does not claim to be complete. All information have been
compiled with care. However, errors can’t be ruled out.

"""

# --> Import necessary packets
import json
import ast
import streamlit as st
from RS_config import RSHelper
from map_api import MapAPI
from types import SimpleNamespace
import pandas as pd


from deep_translator import GoogleTranslator
from RsInstrument import *
from RSfunc import com_prep, com_check, meas_prep, measure, load_iq_csv, plot_time_domain, plot_fft, close

import socket
import subprocess
import os
import zipfile
import re
#from RohdeSchwarz import NRQ
import pyvisa

language_map = {
    "🌐 Select language": None,
    "English": "en",
    "Français": "fr",
    "Español": "es",
    "Deutsch": "de",
    "עברית": "he",
    "हिन्दी": "hi",
    "العربية": "ar",
    "Русский": "ru",
    "中文": "zh-cn",
    "日本語": "ja",
    "한국어": "ko"
}

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



rm = pyvisa.ResourceManager()
print(rm.list_resources())

resource = 'TCPIP::nrq6-101528::hislip0'
# Define variables
#resource = 'TCPIP::192.168.1.100::hislip0'  # VISA resource string for the device
#resource = 'TCPIP0::192.168.1.100::inst0::INSTR'  # VISA resource string for the device
#resource = 'TCPIP::192.168.1.100::INSTR'  # VISA resource string for the device
#resource = "TCPIP0::nrq6-101528::inst0::INSTR"


# Define the device handle
#nrq = RsInstrument(resource, True, True, options="SelectVisa='rs'")
try:
    nrq = RsInstrument(resource, id_query=True, reset=True)
    print(nrq.query('*IDN?'))
    print(nrq.query("*OPT?"))
except Exception as e:
    print(f"Failed to connect: {e}")

#inst = NRQ(resource)


#  Main program begins here

st.set_page_config(page_title="Ennoia Technologies", page_icon="🤖")
#st.sidebar.image('ennoia.jpg')
st.sidebar.image('RS_logo.png')
#st.title("Ennoia Technologies")
st.title("Rohde & Schwarz GmbH")

# Inject CSS to change cursor over dropdown
# st.markdown(f"""
    # <style>
    # /* Apply cursor pointer to the selectbox */
    # div[data-testid="stSelectbox"] {{
        # cursor: pointer;
    # }}
    # </style>
# """, unsafe_allow_html=True)

# 🔽 Dropdown with full names
selected_language = st.selectbox("🌐 Select your language", list(language_map.keys()), index=0)
lang = language_map[selected_language]
if lang:
    text = (
        """ 
        Chat and Test with **Ennoia Technologies Connect Platform** ©. All rights reserved. 
        """
    )
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    st.markdown(translated)


    # --- App logic starts here ---
    selected_options = RSHelper.select_checkboxes()
    st.success(f"You selected: {', '.join(selected_options) if selected_options else 'nothing'}")


    # --- Caching the model and tokenizer ---

    if "SLM" in selected_options:
        @st.cache_resource
        def load_model_and_tokenizer():
            return RSHelper.load_lora_model()

        st.write("\n⏳ Working in OFFLINE mode. Loading local model... (might take a minute)")
        tokenizer, peft_model, device = load_model_and_tokenizer()
        st.write(f"Device set to use {device}")
        map_api = MapAPI(peft_model, tokenizer)
    else:
        st.write("\n⏳ Working in ONLINE mode.")  
        client, ai_model = RSHelper.load_OpenAI_model()
        map_api = MapAPI() 

    helper = RSHelper()
    system_prompt = helper.get_system_prompt()
    few_shot_examples = helper.get_few_shot_examples()



    @st.cache_data
    def get_default_options():
        return map_api.get_defaults_opts()

    def_dict = get_default_options()

    few_shot_examples2 = map_api.get_few_shot_examples()

    # --- Get and cache the RS port ---
    if "RS_port" not in st.session_state:
        st.session_state.RS_port = helper.getport()

    if "SLM" in selected_options:
        #st.write(f"\n✅ Local SLM model {peft_model.config.name_or_path} loaded & device found! Let's get to work.\n")
        text = f"\n✅ Local SLM model {peft_model.config.name_or_path} loaded & device found! Let's get to work.\n"
        translated = GoogleTranslator(source='auto', target=lang).translate(text)
        st.write(translated)
    else:
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = ai_model
        #st.write(f"\n✅ Online LLM model {ai_model} loaded & device! Let's get to work.\n")
        text = f"\n✅ Online LLM model {ai_model} loaded & device! Let's get to work.\n"
        translated = GoogleTranslator(source='auto', target=lang).translate(text)
        st.write(translated)
    # Initialize RS device

    #st.write("Hi. I am Ennoia, your AI assistant. How can I help you today?")
    text = f"Hi. I am Ennoia, your AI assistant. How can I help you today?"
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    st.write(translated)

    translated = com_prep(nrq,lang)
    st.write(translated)
    
    translated = com_check(nrq,lang)
    st.write(translated)


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
            except Exception:
                print("Warning: Failed to parse response as a valid dictionary. Using default options.")

        print(f"\nParsed API options:\n{api_dict}")

        # Ensure it's a dict before using SimpleNamespace
        if isinstance(api_dict, dict):
            opt = SimpleNamespace(**api_dict)
            print(f"opt = {opt}")
            gcf = helper.configure_RS(opt)
            #st.pyplot(gcf)
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

        except Exception as e:
            st.error(f"Failed to process request: {str(e)}")
          
      
        # Convert to Pandas DataFrame
        df = pd.DataFrame(frequency_report_out)

        # Display as a table in Streamlit
        st.dataframe(df)  # Interactive table
        
    meas_prep(nrq)
    translated,iq_pair = measure(nrq)
    st.write(translated)
    
    # === Streamlit UI ===
    #st.set_page_config(page_title="IQ Signal FFT Viewer", layout="centered")
    #st.title("📡 IQ Signal Viewer (.csv)")

    #uploaded_file = st.file_uploader("Upload I/Q CSV File", type="csv")
    #sample_rate = st.number_input("Sample Rate (Hz)", value=122_880_000, step=1_000_000)
    
    uploaded_file = "iq_capture.csv"  # Change to your file location
    sample_rate = 122880000

    if uploaded_file and os.path.exists(uploaded_file):
        iq = load_iq_csv(uploaded_file)
        st.subheader("🕒 Time-Domain Plot")
        st.pyplot(plot_time_domain(iq))

        st.subheader("🔊 Frequency-Domain Plot (FFT)")
        translated, plotout = plot_fft(iq, sample_rate)
        st.write(translated)
        # Display the FFT plot
        st.pyplot(plotout)
    else:
        st.info("Please upload a CSV file with 'I' and 'Q' columns.")

    # Close the instrument connection
    close(nrq)
    text = "Ask me more"
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    st.write(translated)
    