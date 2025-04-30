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
import os

from dotenv import load_dotenv
from scraper import scrape_website
from embedder import create_vectorstore_from_text
from retriever import load_vectorstore, retrieve_relevant_documents

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from accelerate.utils import infer_auto_device_map, get_balanced_memory

import torch
import threading
from transformers import TextIteratorStreamer

import torch
import threading
from transformers import TextIteratorStreamer

import threading
from transformers import TextIteratorStreamer

def query_local_llm_stream_with_context(user_input, model, tokenizer, system_prompt, max_new_tokens=200, temperature=0.7):
    # Ensure the context is only appended once
    full_prompt = f"{system_prompt}\nUser: {user_input}\nAssistant:"

    encoded = tokenizer(full_prompt, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    # ✅ Set clean decoding for token streaming
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True, 
        skip_special_tokens=True,
        decode_kwargs={"clean_up_tokenization_spaces": True}  # Keep spaces and punctuation clean
    )

    # Generation settings with streaming
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,  # Use sampling for randomness
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id
    }

    # Start generation in a background thread
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream output and ensure clean formatting
    response = ""
    for token in streamer:
        # Clean up by removing extra "Assistant:" or "User:" tags
        token_clean = token.replace("Assistant:", "").replace("User:", "").strip()

        # Stop if the model mistakenly generates a new user prompt
        if "user:" in token_clean.lower():
            break

        response += token_clean + " "  # Add space between tokens for readability
        yield token_clean + " "  # Stream token with spacing

    return response.strip()  # Return the cleaned final response


def parse_user_input(user_input, opts,pipe):
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
    
    # Usage
    response_S = query_local_llm(prompt_S, pipe)

    try:
        # Ensure the response is in the correct format (assumes response is a string in JSON format)
        extracted_data = response_S.strip()  # Strip unnecessary spaces/newlines

        # Attempt to parse the response as JSON
        updated_opts = json.loads(extracted_data)
        print(f"Updated dictionary: {updated_opts}")

        # Ensure the structure is valid and matches opts' expected format
        if isinstance(updated_opts, dict):
            return updated_opts
        else:
            print("The LLM response does not contain a valid dictionary.")
            return opts  # Return the original opts if the response is not valid

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print(f"Original LLM response: {response_S}")
        return opts  # Return the original opts if there's an issue with parsing

    except Exception as e:
        print(f"Error parsing input: {e}")
        return opts  # Return original opts if other errors occur


def query_local_llm(prompt, pipe):
    full_prompt = f"User: {prompt}\nAssistant:"

    response = ""

    # Generate response iteratively
    for chunk in pipe(full_prompt, max_new_tokens=200, temperature=0.7, return_full_text=False):
        chunk_text = chunk["generated_text"]

        # Remove any repeated prompt in the output
        response_part = chunk_text.replace(full_prompt, "").strip()
        yield response_part  # ✅ Streams correct output dynamically

        response += response_part  # ✅ Accumulate full response

    return response.strip()  # ✅ Ensure clean final response


@st.cache_resource
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        offload_folder="offload_dir" if device == "cpu" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    return tokenizer, model

#os.environ["CUDA_VISIBLE_DEVICES"] = "" # Set to empty string to disable GPU
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


st.set_page_config(page_title="Ennoia Technologies", page_icon="🤖")
st.sidebar.image('ennoia.jpg')
st.title("Ennoia Technologies")

st.markdown(
    """ 
    Chat and Test with Ennoia Connect Platform ©. All rights reserved. 
    """
)

os.environ["STREAMLIT_WATCH_FILES"] = "false"
load_dotenv()

# STEP 1: Scrape website (only once, or cache)
text = scrape_website("https://www.tinysa.org/wiki")

# Join the list of texts into a single string
text = "".join(text)

if not text.strip():
    print("❌ No text found on the website! Please check the URL.")
    exit()


# STEP 2: Create vectorstore
vectorstore = create_vectorstore_from_text(text)

# STEP 3: Load vectorstore
vectorstore = load_vectorstore()

# STEP 4: Load Local Model
st.write("\n⏳ Working in OFFLINE mode. Loading local model... (might take a minute)")

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
st.write(f"Device set to use {device}")

# Load model to the correct device

tokenizer, model = load_model()


# """ bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,        # Enable 4-bit quantization
#     bnb_4bit_compute_dtype=torch.float16,  # Half-precision computation
#     bnb_4bit_use_double_quant=True,        # Further compression
#      llm_int8_enable_fp32_cpu_offload=True # 🔥 Important: allow CPU/GPU split with FP32 CPU offload
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     torch_dtype=torch.float16,  # Use float16 for model weights
#     device_map="auto", # Automatically split layers across GPU/CPU
#     trust_remote_code=True
# )
#  """
print(model.device)  # Make sure it's on the correct device (CUDA/CPU)


# # Measure and print memory balance
# max_memory = get_balanced_memory(
#     model,
#     no_split_module_classes=model._no_split_modules,
#     dtype=torch.float16,
#     low_zero=False
# )

# print("\n📊 Suggested device memory usage per device:")
# print(max_memory)

# Create the text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)

# """ # 🔥 NEW: generate function manually without `pipeline`
# def generate(prompt, max_new_tokens=200, temperature=0.7):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     output = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         temperature=temperature,
#         do_sample=True,  # if you want more randomness
#     )
#     return tokenizer.decode(output[0], skip_special_tokens=True) """

st.write(f"\n✅ Local model {model.config.name_or_path} loaded! Let's get to work.\n")


# Define system message with rules
# SYSTEM_PROMPT = """
# You are Ennoia, an AI assistant for the TinySA spectrum analyzer (www.tinysa.org). Follow these strict guidelines when responding:

# - Always begin with: "Hi. I am Ennoia. How can I help you today?"
# - Keep responses concise and professional.
# - Use bullet points for lists or steps.
# - Only provide factual, verifiable information.
# - Avoid prohibited topics: politics, religion, controversial current events, medical, legal, or financial advice, personal conversations, or company criticism.
# - Never generate harmful, inappropriate, or misleading content.
# - If the answer is unknown or unclear, say: "I don't know" or "I don't have enough information to answer that."
# - Always offer proactive help: suggest next steps, follow-up questions, or solutions.
# - Explain the source of your information at the end of your response.
# - Only end your turn if the user's query is fully resolved. Otherwise, continue helping.
# - After resolving the issue, ask: "Is there anything else I can help with?"

# Answer in complete sentences and remain focused on assisting with the TinySA spectrum analyzer.
# """
SYSTEM_PROMPT = """
 You are Ennoia, an AI assistant for the TinySA spectrum analyzer (www.tinysa.org).
 Answer in complete sentences and remain focused on assisting with the TinySA spectrum analyzer.
 Explain the source of your information at the end of your response.
 """

# Initialize session state

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize TinySA device

nv1 = tinySA.tinySA(getport())
nv1.close() 

st.write("Hi. I am Ennoia, your AI assistant. How can I help you today?")

prompt = st.chat_input("Ask Ennoia:")

if prompt:
    # Store the user message in session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Construct the full prompt with context (system + user message)
        user_message = st.session_state.messages[-1]["content"]
        
        print(f"User message: {user_message}")  # Debugging output
        # Query the model with the system prompt and user input
        response_generator = query_local_llm_stream_with_context(user_message, model, tokenizer, SYSTEM_PROMPT)
        
        # Initialize an empty string to store the final response
        full_response = ""

        # Process tokens one by one
        for token in response_generator:
            token_clean = token.strip()  # Ensure no leading/trailing spaces on individual tokens
            full_response += token_clean + " "  # Add token with a space for separation

        # Finalize the cleaned response
        final_response = full_response.strip()  # Strip any leading/trailing whitespace after full generation

        print(f"Final response: {final_response}")  # Debugging output
        # Display the streamed response from the assistant
        st.markdown(final_response)
        
        # Save the assistant's response in session state
        st.session_state.messages.append({"role": "assistant", "content": final_response})


        
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
        "plot": True,
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

    
    updated_opts = parse_user_input(prompt,opts,pipe)
    print(f"Updated options: {updated_opts}")
    if isinstance(updated_opts, str) and updated_opts.strip():
        try:
            dict = json.loads(updated_opts.strip())
            print("Dict loaded successfully")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format for dict. {e}")
            print("Raw updated_opts:", updated_opts)
            print("Type of updated_opts:", type(updated_opts))
            dict = opts
    else:
        dict = updated_opts
        print("No valid JSON found, using default opts")
        #response = st.write_stream(stream)
        
    opt = SimpleNamespace(**dict)   
    print(f"opt = {opt}")
    
    #opts_json = json.dumps(opts)
    #updated_opts_json = json.dumps(updated_opts, sort_keys=True, indent=4)

    #if opts_json == json.loads(updated_opts):
    #    response = st.writestream(stream)
    #    print("Matched input")
    #else:
    #    print(f"Did not match input. {opts_json} vs {(updated_opts)}")
    
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


      