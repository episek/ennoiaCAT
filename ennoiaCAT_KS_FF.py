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
import time
from Keysight import FieldFox

VID = 0x0483 #1155
PID = 0x5740 #22336

import pyvisa

# Replace with the IP address of your FieldFox
#FIELD_FOX_IP = "192.168.1.100"
rm = pyvisa.ResourceManager()
resources = rm.list_resources()
for res in resources:
    try:
        eth_inst = rm.open_resource(res)
        print(res, eth_inst.query("*IDN?").strip())
        eth_inst = rm.open_resource(res)
        parts = res.split(':')
        ip_address = parts[2]  # assuming IP is the 3th element
        print("IP Address:", ip_address)
    except:
        continue

visa_address = "TCPIP0::192.168.1.100::inst0::INSTR"
#inst = FieldFox(visa_address)
inst = FieldFox(res)
#inst.open()

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

try:
    rm = pyvisa.ResourceManager()
    ff = rm.open_resource(visa_address)
    ff.read_termination = '\n'
    ff.write_termination = '\n'
    ff.timeout = 5000


    # Test connection with *IDN? query
    idn = inst.query("*IDN?")
    st.write("Connected to:", idn.strip())

except Exception as e:
    print("⚠️ FieldFox is not connected or unreachable.")
    print("Error:", e)

# Set instrument to Spectrum Analyzer mode
inst.set_mode_sa()          # Set to Spectrum Analyzer mode

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

# Define system message with rules
SYSTEM_PROMPT = """
You are an AI assistant for Keysight Fieldfox spectrum analyzer that follows strict rules:
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

# Initialize KS Fieldfox device

#nv1 = tinySA.tinySA(getport())
#nv1.close() 

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
        
    keywords = ["start","stop","center","span","rbw"]
    #matches = [word for word in keywords if word in prompt.lower()]
    matches = [word for word in prompt.split() if word.lower() in [k.lower() for k in keywords]]
    startf = extract_numbers(prompt)
    if matches and startf:
        print(matches)
        if startf:
                #start = float(match.group())
            print(startf)
            # Call another script
            #fname = r"C:\Users\rices\OneDrive\startup\tinySA\tinySA.py"
            #fname = "tinySA.py"
            args1 = []
            args2 = []
            response1 = ""
            response2 = ""
            num_elements = len(startf)
            print(num_elements)
            i = 0
            for num in startf:
                #string2 = float(num)
                if "start" in matches[i]:
                    num1 = float(startf[i])
                    str1e = f"{num1:.2e}" 
                    #args1 = ['-p', '-S', str1e]
                    #inst.write(f":FREQuency:STARt {num1:.0f}")
                    inst.set_start(num1)
                    response1 = (f"{r"Setting start frequency to "}{str1e} {" Hz"}")
                if "stop" in matches[i]:
                    num2 = float(startf[i])
                    str2e = f"{num2:.2e}"
                    #args2 = ['-p', '-E', str2e]
                    #inst.write(f":FREQuency:STOP {num2:.0f}")
                    inst.set_stop(num2)
                    response2 = (f"{r"Setting stop frequency to "}{str2e} {" Hz"}")
                if "center" in matches[i]:
                    num1 = float(startf[i])
                if "span" in matches[i]:
                    num1e = num1 - float(startf[i])/2
                    num2e = num1 + float(startf[i])/2
                    num1 = num1e
                    num2 = num2e
                    str1e = f"{num1e:.2e}" 
                    str2e = f"{num2e:.2e}" 
                    #args1 = ['-p', '-S', str1e]
                    #args2 = ['-p', '-E', str2e]
                    #inst.write(f":FREQuency:STARt {num1:.0f}")
                    inst.set_start(num1)
                    #inst.write(f":FREQuency:STOP {num2:.0f}")
                    inst.set_stop(num2)
                    response1 = (f"{r"Setting start frequency to "}{str1e} {" Hz"}")
                    response2 = (f"{r"Setting stop frequency to "}{str2e} {" Hz"}")
                i+=1
            
            # # Trigger a single sweep
            inst.single_sweep()
            # inst.write(":INITiate:IMMediate; *WAI")
            # inst.write(":INIT:CONT OFF")    # Pause continuous updates
            # inst.write(":INIT:IMM")         # Manual sweep
            # inst.query("*OPC?")             # Wait until it's done            
            # Wait for the measurement to complete
            #print(inst.read()) # This will read the *OPC? response


            # Read the number of points in the trace
            number_of_points = inst.get_points()
            # inst.write("SENS:SWE:POIN?")
            # number_of_points = inst.read()
            print("Number of points in trace:", number_of_points)


            #inst.write("INIT:CONT ON")  # Restart continuous mode
            inst.restart_continuous()

        
            #raw_data = inst.query(":TRACe:DATA?")
            raw_data = inst.fetch_trace()

            trace = [float(x) for x in raw_data.strip().split(",")]

            # Build frequency array
            #import numpy as np
            freq = np.linspace(num1, num2, len(trace))

            # Plot the data
            #import matplotlib.pyplot as plt

            plt.plot(freq / 1e6, trace)
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Amplitude (dBm)")
            plt.title("Spectrum from FieldFox")
            st.pyplot(plt.gcf())
            plt.grid(True)
            plt.show()

            inst.close()
            
            #process = Popen(['python',fname] + args, shell=True)
            #add_script_run_ctx(process,ctx)
            #nv.close()
            #response = (f"{r"Setting start frequency to "}{num} {" Hz"}")
            print(response1)
            print(response2)
            time.sleep(1)
            st.write(response1)
            st.write(response2)
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
    if matches:
        st.session_state.messages.append({"role": "assistant", "content": response1})
        st.session_state.messages.append({"role": "assistant", "content": response2})