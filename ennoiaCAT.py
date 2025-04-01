from openai import OpenAI
from openai_api_key_verifier import verify_api_key, check_model_access, list_models, get_account_usage
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
- Provide factual information only.
- Never generate harmful or inappropriate content.
- Use bullet points when listing multiple items.
- Maintain a professional tone.
- Always be proactive when you can, i.e. suggest next steps or offer to actually solve the problem. If need be, ask requisite follow up questions beforehand.
- Offer to set the start frequency to the user and ask to input the value in Hz.
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

        
#    st.session_state.messages.append({"role": "assistant", "content": response})

 # Getting Answers
   # response = None
    #print("Checking for a Valid Query...")
    #valid_q = get_gemini_response(prompt)
  #  print("LLM Response for valid query:", prompt)
    # idx = get_most_sim_idx(model.encode([prompt]), q_embs)
   # if prompt:
        # check if the valid_q is in query list
   # print(" Generating Ennoia response...")
#        match = re.search(r"[-+]?\d*\.\d+e[-+]?\d+", prompt)
        #match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", prompt)
 

    startf = extract_numbers(prompt)
    if startf:
            #start = float(match.group())
        print(startf)
        # Call another script
        #fname = r"C:\Users\rices\OneDrive\startup\tinySA\tinySA.py"
        fname = "tinySA.py"
        for num in startf:
            string2 = float(num)
        args = ['-p', '-S', num]
        combined_string = f"{fname} {args}"
        print(combined_string)
        #subprocess.run(['python', fname] + args)

        #from streamlit.runtime.scriptrunner import add_script_run_ctx,get_script_run_ctx
        #from subprocess import Popen
        #ctx = get_script_run_ctx()
        ##Some code##
        
        
        parser = OptionParser(usage="%prog: [options]")
        parser.add_option("-p", "--plot", dest="plot",
                      action="store_true", default=False,
                      help="plot rectanglar", metavar="PLOT")
        parser.add_option("-c", "--scan", dest="scan",
                      action="store_true", default=False,
                      help="scan by script", metavar="SCAN")
        parser.add_option("-S", "--start", dest="start",
                      type="float", default=1e6,
                      help="start frequency", metavar="START")
        parser.add_option("-E", "--stop", dest="stop",
                      type="float", default=900e6,
                      help="stop frequency", metavar="STOP")
        parser.add_option("-N", "--points", dest="points",
                      type="int", default=101,
                      help="scan points", metavar="POINTS")
        parser.add_option("-P", "--port", type="int", dest="port",
                      help="port", metavar="PORT")
        parser.add_option("-d", "--dev", dest="device",
                      help="device node", metavar="DEV")
        parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="verbose output")
        parser.add_option("-C", "--capture", dest="capture",
                      help="capture current display to FILE", metavar="FILE")
        parser.add_option("-e", dest="command", action="append",
                      help="send raw command", metavar="COMMAND")
        parser.add_option("-o", dest="save",
                      help="write CSV file", metavar="SAVE")
        #args = ["-S", string2]
        (opt, args1) = parser.parse_args(args)

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
        
        #process = Popen(['python',fname] + args, shell=True)
        #add_script_run_ctx(process,ctx)
        nv.close()
        response = (f"{r"Setting start frequency to "}{num} {" Hz"}")
        print(response)
        st.write(response)
        #print(stream)
    else:
        response = st.write_stream(stream)
        #print(response) 
        #subprocess.run(['python', combined_string], capture_output=True, text=True)
        #with open(combined_string) as file:
        #    exec(file.read())
        # Print the output of the called script
        #print(result.stdout)
    #else:
    #    print("no match found") 
   #     response = {}
   #     response["answer"] = "text" 
        #prompt
        #get_agent_response(prompt)
   # else:
   #     print("Checking with LLM Agent for Response...")
   #     response = {}
   #     response["answer"] = " text " 
        #get_agent_response(prompt)
        # We can use the below response as a fail safe response for a Rule Based Chatbot
        # response["answer"] = "I don't have exposure to enough data to answer this question."

    # Display Assistant Response
    #display_content(response,"assistant")
    # Add response message to chat history
    #response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})