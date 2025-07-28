from __future__ import print_function
import requests  # http://docs.python-requests.org/
import time
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
    else:
        success = lic.request_license(args.key)
elif args.action == "verify":
    success = lic.verify_license_file()
else:
    success = lic.verify_license_file()
  
if not success:
    print("❌ License verification failed. Please check your license key or contact support.")
    exit()


# --> Import necessary packets
import json
import ast
import streamlit as st
from AK_config import AKHelper
from map_api import MapAPI
from types import SimpleNamespace
import pandas as pd


from deep_translator import GoogleTranslator
#from RsInstrument import *
#from RSfunc import com_prep, com_check, meas_prep, measure, load_iq_csv, plot_time_domain, plot_fft, close
from streamlit_js_eval import streamlit_js_eval
import socket
import subprocess
import os
import zipfile
import re
#from RohdeSchwarz import NRQ
import pyvisa
import matplotlib.pyplot as plt


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
    "save": "write output to CSV file",
    "run": "run the PCAP file"
}

st.set_page_config(page_title="Ennoia Technologies", page_icon="🤖")
st.sidebar.image('aukua rgb high.jpg')
st.sidebar.image('ennoia_white_black_hi-def.png')
st.title("Aukua Systems and")
st.title("Ennoia Technologies")


if not success:
    st.error("Ennoia License verification failed. Please check your license key or contact support.")
    st.stop()
else:
    st.success("Ennoia License verified successfully.")


rm = pyvisa.ResourceManager()
print(rm.list_resources())

#resource = 'TCPIP::nrq6-101528::hislip0'
# Define variables
#resource = 'TCPIP::192.168.1.100::hislip0'  # VISA resource string for the device
#resource = 'TCPIP0::192.168.1.100::inst0::INSTR'  # VISA resource string for the device
#resource = 'TCPIP::192.168.1.100::INSTR'  # VISA resource string for the device
#resource = "TCPIP0::nrq6-101528::inst0::INSTR"



#inst = NRQ(resource)


#  Main program begins here


# Get screen width
screen_width = streamlit_js_eval(js_expressions="screen.width", key="get_width")

# Dynamically inject font size CSS
if screen_width:
    font_size = max(min(int(screen_width / 60), 20), 10)

    # Apply font size globally
    st.markdown(f"""
        <style>
        html, body, [class*="st-"] {{
            font-size: {font_size}px !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    # Optional: Set global matplotlib font size
    plt.rcParams.update({'font.size': font_size})



# Inject CSS to change cursor over dropdown
# st.markdown(f"""
    # <style>
    # /* Apply cursor pointer to the selectbox */
    # div[data-testid="stSelectbox"] {{
        # cursor: pointer;
    # }}
    # </style>
# """, unsafe_allow_html=True)

# Define the device handle
try:

    ip = "192.168.1.101"  # IP address of the system
    uribase = "http://"+ip+"/api/v1/"


except Exception as e:
    #st.error(f"Failed to connect: {e}")
    print(f"Failed to connect: {e}")

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
    selected_options = AKHelper.select_checkboxes()
    st.success(f"You selected: {', '.join(selected_options) if selected_options else 'nothing'}")


    # --- Caching the model and tokenizer ---

    if "SLM" in selected_options:
        @st.cache_resource
        def load_model_and_tokenizer():
            return AKHelper.load_lora_model()

        text=("\n⏳ Working in OFFLINE mode. Loading local model... (might take a minute)")
        translated = GoogleTranslator(source='auto', target=lang).translate(text)
        st.write(translated)
        tokenizer, peft_model, device = load_model_and_tokenizer()
        text=(f"Device set to use {device}")
        translated = GoogleTranslator(source='auto', target=lang).translate(text)
        st.write(translated)
        map_api = MapAPI(peft_model, tokenizer)
    else:
        text=("\n⏳ Working in ONLINE mode.")  
        translated = GoogleTranslator(source='auto', target=lang).translate(text)
        st.write(translated)
        client, ai_model = AKHelper.load_OpenAI_model()
        map_api = MapAPI() 

    helper = AKHelper()
    system_prompt = helper.get_system_prompt()
    few_shot_examples = helper.get_few_shot_examples()

    text = (f"Found Aukua Systems Model XGA4250 at IP Address {ip}")
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    st.write(translated)
   

    # @st.cache_data
    # def get_default_options():
        # return map_api.get_defaults_opts()

    def get_defaults_opts():
         # Define default options dictionary
        opts = {
            "plot": False,
            "scan": False,
            "start": 300000000.0,
            "stop": 900000000.0,
            "points": 101,
            "port": False,
            "device": None,
            "verbose": False,
            "capture": False,
            "command": None,
            "save": None
        }
        return opts



    def_dict = get_defaults_opts()

    few_shot_examples2 = map_api.get_few_shot_examples()

    # --- Get and cache the AK port ---
    if "Aukua_port" not in st.session_state:
        st.session_state.AK_port = helper.getport()

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
    # Initialize AK device

    #st.write("Hi. I am Ennoia, your AI assistant. How can I help you today?")
    text = f"Hi. I am Ennoia, your AI assistant. How can I help you today?"
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    st.write(translated)
    d = 0
    # translated = com_prep(nrq,lang)
    # st.write(translated)
    
    # translated = com_check(nrq,lang)
    # st.write(translated)


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

        # # Ensure it's a dict before using SimpleNamespace
        if isinstance(api_dict, dict):
            opt = SimpleNamespace(**api_dict)
            print(f"opt = {opt}")
            helper.configure_AK(opt,lang,d)
            if opt.port:
                if (d == 0):
                    text = ("PCAP Porting to XGA4250 Successfully Completed!")
                    translated = GoogleTranslator(source='auto', target=lang).translate(text)
                    st.write(translated)
        else:
            st.error("API response is not a valid dictionary. Setting default options.")
     

    