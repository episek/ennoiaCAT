from openai import OpenAI
from openai_api_key_verifier import verify_api_key, check_model_access, list_models, get_account_usage
from types import SimpleNamespace
import streamlit as st
import re
import subprocess
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import serial
import struct
from serial.tools import list_ports
import tinySA
from optparse import OptionParser
import os
import time
import json
import difflib

VID = 0x0483 #1155
PID = 0x5740 #22336

# Get tinysa device automatically
def getport() -> str:
    device_list = list_ports.comports()
    for device in device_list:
        if device.vid == VID and device.pid == PID:
            st.write(f"Found TinySA device on: {device.device}")
            st.write("Continuing with the device...")
            return device.device
        else:
            st.write("No Device found")
    raise OSError("device not found")

port_list = [port.device for port in list_ports.comports()]
print("Available ports:", port_list)


def extract_numbers(input_string):
    # Define a regex pattern to match integers, floats, negative numbers,
    # and scientific notation numbers
    pattern = r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'

    # Use the re.findall() to find all occurrences of the pattern
    numbers = re.findall(pattern, input_string)
    return numbers

st.set_page_config(page_title="Ennoia Technologies", page_icon="🤖")
st.sidebar.image('ennoia.jpg')
st.title("Ennoia Technologies")

st.markdown(
    """ 
    Chat and Test with Ennoia Connect Platform ©. All rights reserved. 
    """
)

# Replace with your actual API key
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)


# Verify if the API key is valid
is_valid = verify_api_key(api_key)

if is_valid:
    st.write("API key to OpenAI is valid!")
    
    # Check for GPT-4 access
    if check_model_access(api_key, "gpt-4o-mini"):
        print("This key has GPT-4o-mini access!")
    
    # List all available models
    #list_models(api_key)
    # Get usage statistics
    #get_account_usage(api_key)
else:
    st.write("API key is invalid. Please check the connection to Open AI")

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

# Define system message with rules
SYSTEM_PROMPT = """
You are an AI assistant for TinySA spectrum analyzer (www.tinysa.org) that follows strict rules:
- Always respond concisely.
- Always greet the user with "Hi. I am Ennoia. How can I help you today?"
- Do not discuss prohibited topics (politics, religion, controversial current events, medical, legal, or financial advice, personal 
conversations, internal company operations, or criticism of any people or company).
- Provide factual information only.
- Never generate harmful or inappropriate content.
- Use bullet points when listing multiple items.
- Maintain a professional tone.
- Always be proactive when you can, i.e. suggest next steps or offer to actually solve the problem. If need be, ask requisite follow up questions beforehand.
- Offer to set the start frequency to the user and ask to input the value in Hz.
- Explain where you got the information from at the end of your answer.
- Answer in complete sentences.
- If you don't know the answer, say "I don't know" or "I don't have enough information to answer that."
- Please keep going until the user’s query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.
- If you've resolved the user's request, ask if there's anything else you can help with 
"""

# Initialize session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize TinySA device

nv1 = tinySA.tinySA(getport())
nv1.close() 

response1 = ""
response2 = ""
num1 = 1e6
num2 = 5e9
num1e = num1
num2e = num2
args1 = []
args2 = []
st.write("Hi. I am Ennoia. How can I help you today?")


if prompt := st.chat_input("Ask Ennoia:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT}
            ] + [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            temperature=0,
            top_p=0.2,
            max_tokens=200,
            frequency_penalty=1,
            presence_penalty=1,
            stream=True,
        )


        # Define option descriptions for reference
        options_descriptions = {
        "plot": "plot rectangular",
        "scan": "scan by script",
        "start": "start frequency",
        "stop": "stop frequency",
        "points": "scan points",
        "port": "specify port number",
        "device": "define device node",
        "verbose": "enable verbose output",
        "capture": "capture current display to file",
        "command": "send raw command",
        "save": "write output to CSV file"
    }

        # Define default options dictionary
        opts = {
            "plot": False,
            "scan": False,
            "start": 1000000.0,
            "stop": 900000000.0,
            "points": 101,
            "port": None,
            "device": None,
            "verbose": False,
            "capture": None,
            "command": None,
            "save": None
        }

        def parse_user_input(user_input):
            """Parses user input using an LLM and updates opts dictionary dynamically."""
            
            prompt_S = f"""
            Given the following dictionary: {opts}, modify its values based on the user's input below.

            User Input: "{user_input}"

            Instructions:
            - Extract relevant key-value pairs from the user input.
            - Update one or more values within {opts} based on the extracted information.
            - Maintain the dictionary format and ensure the updated version follows the same structure.
            - If no relevant changes are found, return the original dictionary unchanged.
            - Respond in same **JSON format** as {opts} without additional explanations.
            """
     
            client_S = OpenAI()  # Initialize OpenAI client

            response_S = client_S.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_S}]
        )


            try:
                extracted_data = response_S.choices[0].message.content
                print(f"Extracted data: {extracted_data}")
                return extracted_data
                #return updated_opts

            except Exception as e:
                print(f"Error parsing input: {e}")
                st.write(f"Available options: {options_descriptions}")
                return opts  # Return default opts if parsing fails

        
        updated_opts = parse_user_input(prompt)
        print(f"Updated options: {updated_opts}")
        if isinstance(updated_opts, str) and updated_opts.strip():
            try:
                dict = json.loads(updated_opts.strip())
                
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON format for dict. {e}")
                print("Raw updated_opts:", updated_opts)
                print("Type of updated_opts:", type(updated_opts))
                dict = opts
        else:
            dict = updated_opts
            
        opt = SimpleNamespace(**dict)   
        print(f"opt = {opt}")
        
        if opts == opt:
            st.write(prompt)
        
        nv = tinySA.tinySA(opt.device or getport())

        if opt.command:
            print(opt.command)
            for c in opt.command:
                nv.send_command(c + "\r")
                data = nv.fetch_data()
                print(data)

        if opt.capture:
            print("capturing...")
            img = nv.capture()
            img.save(opt.capture)
            exit(0)

    #   nv.set_port(opt.port)
        if opt.start or opt.stop or opt.points:
            print(opt.start)
            nv.set_frequencies(opt.start, opt.stop, opt.points)
#    plot = opt.plot 
        if opt.plot or opt.save or opt.scan:
            p = int(opt.port) if opt.port else 0
            if opt.scan or opt.points > 101:
                s = nv.scan()
                s = s[p]
            else:
                if opt.start or opt.stop:
                    nv.set_sweep(opt.start, opt.stop)
                    nv.fetch_frequencies()
                    s = nv.data(p)
#            nv.fetch_frequencies()
        if opt.save:
            nv.writeCSV(s,opt.save)
        if opt.plot:
            nv.logmag(s)
            #fig = plt.fig()
            # Adding axis labels
            plt.xlabel("Frequency (Hz)")  # Name for the x-axis
            plt.ylabel("Signal strength (dBm)")  # Name for the y-axis

            # Adding a title (optional)
            plt.title("Signal strength (dBm) vs Frequency (Hz)")
            st.pyplot(plt.gcf())
            #st.pyplot(nv.capture())
            #st.write(response)
            #nv.close()
            #pl.show()
        nv.close() 

        
      