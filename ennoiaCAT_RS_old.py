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
import struct
import csv
import matplotlib.pyplot as plt
import numpy as np

from RsInstrument import *
from time import time, sleep
from deep_translator import GoogleTranslator

import socket
import subprocess
import os
import zipfile
import re
#from RohdeSchwarz import NRQ

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


import pyvisa

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

# Define all the subroutines

def com_prep():
    """Preparation of the communication (termination, etc...)"""
    manu = nrq.visa_manufacturer
    text = f"VISA Manufacturer: {manu}"
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    st.write(translated)
    nrq.visa_timeout = 3000  # Timeout for VISA Read Operations
    nrq.opc_timeout = 3000  # Timeout for opc-synchronised operations
    nrq.instrument_status_checking = True  # Error check after each command, can be True or False
    nrq.clear_status()  # Clear status register
    nrq.logger.log_to_console = False  # Route SCPI logging feature to console on (True) or off (False)
    nrq.logger.mode = LoggingMode.Off  # Switch On or Off SCPI logging


def com_check():
    """Test the device connection, request ID as well as installed options"""
    idn_response = nrq.query('*IDN?')
    text = f"Hello, I am {idn_response}"
    translated_intro = GoogleTranslator(source='auto', target=lang).translate(text)
    st.write(f"{translated_intro} {idn_response}")    
    query_response = nrq.query('*OPT?')
    text =f"and I have the following options available: \n {query_response}"
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    st.write(translated)
    # idn_response = nrq.query('*IDN?')
    # st.write({nrq.query("SENSe:FUNCtion?")})
    # nrq.clear_status()
    # sleep(0.5)  # Give it time to settle before next query

def meas_prep():
    """Prepare the devise for the measurement"""
    #inst.set_mode_iq()
    nrq.write('SENSe:FUNCtion "XTIMe:VOLT:IQ"')  # Change sensor mode to I/Q
    #inst.set_center(2e09)
    nrq.write('SENSe:FREQuency:CENTer 2e09')  # Center Frequency to 2 GHz
    #inst.set_bw_res_manual()
    nrq.write('SENSE:BANDwidth:RESolution:TYPE:AUTO:STATe OFF')  # Change bandwidth setting to manual state
    #inst.set_bw_res_normal()
    nrq.write('SENSE:BANDwidth:RESolution:TYPE NORMal')  # Flat filter type
    #inst.set_bw_res(1e8)
    nrq.write('SENSE:BANDwidth:RES 1e8''')  # Analysis bandwidth is 100 MHz now
    #inst.set_trace_length(15e5)
    nrq.write('SENSE:TRACe:IQ:RLENgth 15e5')  # IQ trace length is 1.5 million samples now
    nrq.write('')
    # cf = nrq.query('SENSe:FREQuency:CENTer?')
    # bw = nrq.query('SENSe:BANDwidth:RESolution?')
    # trace = nrq.query('SENSE:TRACe:IQ:RLENgth?')
    # sf = nrq.query('SENSe:BANDwidth:SRATe:CUV?')
    # text = (
        # f"Current setup parameters:\n"
        # f"Center Frequency is {cf} Hz,\n"
        # f"Analysis bandwidth is {bw} Hz,\n"
        # f"Trace length is {trace} Sa,\n"
        # f"Sample Rate is {sf} Sa/s,\n"
    # )
    # translated = GoogleTranslator(source='auto', target=lang).translate(text)
    # st.write(translated)    
    nrq.write('FORM:DATA REAL,64')


def collect_iq_samples(nrq, trace_length):
    # Set up IQ measurement
    nrq.write("CONF:IQ")  # Configure instrument for IQ acquisition

    # Set the IQ bandwidth (if needed)
    # nrq.write("SENS:IQ:BAND 20e6")  # For example: 20 MHz bandwidth

    # Set the number of IQ samples to capture
    nrq.write("SENS:IQ:POIN 10000")  # Number of points

    # Trigger a new acquisition
    nrq.write("INIT:IMM; *WAI")  # Start measurement and wait until complete

    # Fetch the IQ data as binary block
    iq_raw = nrq.query_bin_block("FETCh:WAVeform:IQ:TRACe?", datatype=RsInstrument.DataType.Float32, read_termination='')

    # Process the binary IQ data
    # IQ samples are returned as [I0, Q0, I1, Q1, ..., In, Qn]
    I_samples = iq_raw[::2]
    Q_samples = iq_raw[1::2]
    complex_iq = [complex(i, q) for i, q in zip(I_samples, Q_samples)]

    print(f"Collected {len(complex_iq)} IQ samples")
    # Convert flat list to list of (I, Q) pairs
    iq_pairs = list(zip(I_samples, Q_samples))
    return iq_pairs

def save_iq_as_csv(float_list, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        iq_pairs = zip(float_list[::2], float_list[1::2])  # I/Q
        writer.writerow(['I', 'Q'])
        writer.writerows(iq_pairs)

def save_float_list_as_bin(float_list, filename):
    with open(filename, 'wb') as f:
        for value in float_list:
            f.write(struct.pack('f', value))  # float32 format

def bin_to_csv(bin_file, csv_file):
    if not os.path.exists(bin_file):
        print(f"[ERROR] File not found: {bin_file}")
        return

    with open(bin_file, 'rb') as f:
        bin_data = f.read()
    # Number of float32 samples
    num_samples = len(bin_data) // 4
    print(f"[INFO] Total float32 values: {num_samples} ({num_samples//2} I/Q pairs)")

    # Unpack as float32 (IEEE 754), little-endian
    float_data = struct.unpack('<' + 'd' * (len(bin_data)//8), bin_data)

    # Group into I/Q pairs
    iq_pairs = zip(float_data[::2], float_data[1::2])  # (I, Q)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['I', 'Q'])
        writer.writerows(iq_pairs)

    print(f"[✅] Saved CSV: {csv_file}")
    return(iq_pairs)

def measure():
    """Perform measurement and timing calculation, print results"""
    start = time()  # Capture (system) start time
    nrq.write('INITiate:IMMediate')  # Initiates a single trigger measurement
    nrq.visa_timeout = 10000  # Extend Visa timeout to avoid errors
    #output = nrq.query_bin_or_ascii_float_list('FETCh1?')  # Get the measurement in binary format
    output = nrq.query_bin_or_ascii_float_list('FETCh1?')  # Get the measurement in binary format
    nrq.visa_timeout = 3000  # Change back timeout to standard value
    inter = time()  # Capture system time after I/Q data has been received
    duration = inter-start  # And calculate process time
    text = f"After {round(duration, 1)} seconds {len(output)} I/Q samples have been recorded."
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    st.write(translated)        
    #st.write(f'After {round(duration, 1)} seconds {len(output)} I/Q samples have been recorded.')
    # Perform 2nd take
    #nrq.write('INITiate:IMMediate')
    #nrq.visa_timeout = 10000
    #output = nrq.query_bin_or_ascii_float_list('FETCh1?')
    #output = nrq.query_bin_or_ascii_float_list
    #nrq.visa_timeout = 3000
    

    # Connect to instrument
    #nrq = RsInstrument("TCPIP0::192.168.1.100::5025::SOCKET", id_query=True, reset=False)

    # Optional: configure frequency, RBW, etc. before capture
    # nrq.write("SENSE:FREQ:CENT 1.5e9")  # example center freq
    # nrq.write("SENSE:BAND:RES 1e6")     # example RBW

    # Capture and save I/Q data
    save_float_list_as_bin(output, "iq_capture.bin")
    #iq_data = collect_iq_samples(nrq, trace_length=10000)
    iq_pairs = bin_to_csv("iq_capture.bin", "iq_capture.csv")
    #bin_to_csv(output, "iq_capture.csv")
    #save_iq_as_csv(output, "iq_capture.csv")

    #instr.close()
    
    
    end = time()
    duration = end - start
    text = f"After {round(duration, 1)} seconds both records have been taken, with the last one {len(output)} I/Q samples have been recorded."
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    st.write(translated)        
    #st.write(f'After {round(duration, 1)} seconds both records have been taken,'
    #      f'with the last one {len(output)} I/Q samples have been recorded.')
    return(iq_pairs)


def load_iq_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    iq = df['I'].to_numpy() + 1j * df['Q'].to_numpy()
    return iq

def plot_time_domain(iq):
    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    ax[0].plot(np.real(iq), label='I')
    ax[1].plot(np.imag(iq), label='Q', color='orange')
    ax[0].set_ylabel('Amplitude')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_xlabel('Sample Index')
    ax[0].set_title('Time Domain Signal')
    ax[0].grid(True)
    ax[1].grid(True)
    return fig

def plot_fft(iq, sample_rate):
    N = len(iq)
    fft_data = np.fft.fftshift(np.fft.fft(iq))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/sample_rate))
    magnitude_db = 20 * np.log10(np.abs(fft_data) + 1e-12)
    magnitude_db[N // 2] = magnitude_db[N // 2]-60
    
    text = f"Detected high DC component of > 60dB above noise floor. Suppressed the DC Component"
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    st.write(translated)        

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs / 1e6, magnitude_db)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('FFT Spectrum')
    ax.grid(True)
    return fig


def close():
    """Close the VISA session"""
    nrq.close()



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






    com_prep()
    com_check()


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
        
    meas_prep()
    iq_pair = measure()

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
        st.pyplot(plot_fft(iq, sample_rate))
    else:
        st.info("Please upload a CSV file with 'I' and 'Q' columns.")


    close()
    text = "Program successfully ended."
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    st.write(translated)
    