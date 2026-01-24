# CS_config.py
from __future__ import print_function
import requests  # http://docs.python-requests.org/
import time
import os
import json
import torch
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from serial.tools import list_ports
#from RsInstrument import *
import matplotlib.pyplot as plt
import streamlit as st
import csv
import numpy as np
from scipy.signal import find_peaks
import struct
import pandas as pd
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    GoogleTranslator = None
#import cgi
import sys
import tempfile
from PIL import Image
try:
    from scapy.all import rdpcap, sendp, sendpfast, get_if_list, get_if_addr, get_if_hwaddr
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    rdpcap = sendp = sendpfast = get_if_list = get_if_addr = get_if_hwaddr = None
import statistics
#import matplotlib.pyplot as plt
try:
    import pyshark
    PYSHARK_AVAILABLE = True
except ImportError:
    PYSHARK_AVAILABLE = False
    pyshark = None
import psutil, socket
#from streamlit_autorefresh import st_autorefresh

class CSHelper:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        
    def get_system_prompt(self):
        # Using LLaMA 2-style formatting
        system_prompt = (
            "You are Ennoia, an AI assistant specifically for the Cisco NCS540 CSR  (https://www.cisco.com/).\n"
            "Your role is to help users configure, troubleshoot, and operate the Cisco NCS540 CSR only.\n"
            "Assume every question is about Cisco Systems.\n"
            "If the user mention pcap or PCAP then reply with: 'I am ready to assist you with PCAP related commands.'\n"
            "If the user asks anything unrelated to NCS540 or PCAP, reply with: 'I can only assist with queries related to the Cisco Systems NCS540.'\n"
            "Do not provide generic RF advice. Never ask follow-up questions. Repeat all numeric values exactly.\n"
            "Answer only with complete sentences. Do not cut the sentence when displaying it.\n"
        )

        return system_prompt

        
    def get_few_shot_examples(self):
        # === Few-Shot Examples ===
        few_shot_examples = [
            {"role": "user", "content": "How do I set the NCS540"},
            {"role": "assistant", "content": "To set the NCS540, connect to the Cisco Console through the COM port"},
            
            {"role": "user", "content": "What is the maximum rate of the NCS540"},
            {"role": "assistant", "content": "The maximum rate per port of the NCS540 is 28Gbps"},
            
            {"role": "user", "content": "How many data ports the NCS540 has"},
            {"role": "assistant", "content": "The NCS540 has four SFP28 data ports with rates up to 28Gbps"},
            
            {"role": "user", "content": "How to load a PCAP file"},
            {"role": "assistant", "content": "To load a PCAP file run the Cisco_playback python file provided in the Console"},
            
            {"role": "user", "content": "How to capture the recieved PCAP file"},
            {"role": "assistant", "content": 'In roder to capture the received PCAP file, please run the provided Cisco_capture file in the Console'},
            
            {"role": "user", "content": "What is the command to fetch the IQ data?"},
            {"role": "assistant", "content": "You can fetch the IQ data using: FETCh1?"},
            
            {"role": "user", "content": "How do I save the captured IQ data to a file?"},
            {"role": "assistant", "content": "To save the captured IQ data, you can use a script to write it to a binary file."},
            
            {"role": "user", "content": "port pcap file"},
            {"role": "assistant", "content": "The pcap file porting will start soon"},
            
            {"role": "user", "content": "capture pcap file"},
            {"role": "assistant", "content": "The pcap file capture will start soon"},
            
            {
                "role": "user",
                "content": "How do I configure the Cisco Systems Protocol Analyzer?"
            },
            {"role": "assistant", "content": "To configure the Protocol Analyzer, connect to the Cisco Console through the COM port and follow the setup instructions provided in the interface"},
            {
                "role": "user",
                "content": "How do I configure the NCS540"
            },
            {"role": "assistant", "content": "To configure the NCS540, connect to the Cisco Console through the COM port and follow the setup instructions provided in the interface"},
            {
                "role": "user",
                "content": "How do I configure the Cisco"
            },
            {"role": "assistant", "content": "To configure the Cisco unit, connect to the Cisco Console through the COM port and follow the setup instructions provided in the interface"}
            
           # {
                # "role": "assistant",
                # "content": (
                    # "To configure the R&S NRQ6 power sensor:\n\n"
                    # "1. **Connect the sensor** via USB or LAN and install VISA drivers.\n"
                    # "2. **Reset the device**: `*RST`\n"
                    # "3. **Set frequency**: `SENSE:FREQ 2e9` (for 2 GHz)\n"
                    # "4. **Enable averaging**: `SENSE:AVER:COUNT 10`\n"
                    # "5. **Set bandwidth**: `SENSE:BAND:RES 1e6` (1 MHz)\n"
                    # "6. **Trigger mode**: `TRIG:SOUR IMM` (immediate)\n"
                    # "7. **Start measurement**: `INIT:IMM`\n"
                    # "8. **Fetch result**: `FETCH:POWER:AC?` or `FETCH:POWER:PEAK?`\n\n"
                    # "You can also use Python with PyVISA or the R&S Power Viewer Plus software for easier setup and control."
                # )
            # }
        ]
        return few_shot_examples      
        
    @staticmethod
    def select_checkboxes():
        st.markdown("### Select your model type")

        # Initialize state
        if "submitted" not in st.session_state:
            st.session_state.submitted = False

        if not st.session_state.submitted:
            # Show checkboxes
            option1 = st.checkbox("Online LLM", key="opt_a")
            option2 = st.checkbox("Local SLM", key="opt_b")

            # Submit button
            if st.button("Submit"):
                st.session_state.submitted = True
                st.session_state.selected = []
                if st.session_state.opt_a:
                    st.session_state.selected.append("LLM")
                if st.session_state.opt_b:
                    st.session_state.selected.append("SLM")
                st.rerun()

            # Return early: don't continue until submitted
            st.stop()

        # After submission
        return st.session_state.selected

    @staticmethod
    def load_lora_model(base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", lora_path="./tinyllama_CS_lora"):
        """
        Loads a base language model with LoRA weights and returns the tokenizer and the merged model.

        Args:
            base_model_name (str): The name or path of the base model.
            lora_path (str): Path to the directory containing LoRA weights.

        Returns:
            tokenizer: HuggingFace tokenizer object.
            model: The merged model ready for inference.
            device: The device on which the model is loaded (CPU or GPU).
        """
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            local_files_only=True,
            trust_remote_code=False
        )
        tokenizer.pad_token = tokenizer.unk_token  # Recommended for LLaMA models
        tokenizer.use_default_system_prompt = False  # Avoid auto-formatting if needed

        # Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": device},
            local_files_only=True,
            trust_remote_code=False
        )
        print(f"Base model loaded on {device}.")

        # Load and Merge LoRA
        peft_model = PeftModel.from_pretrained(base_model, lora_path)
        peft_model = peft_model.merge_and_unload()
        peft_model.eval().to(device)

        return tokenizer, peft_model, device

    
    
    def query_local_llm_fast(self, user_input, model, tokenizer, system_prompt, max_new_tokens=50):
        full_prompt = f"{system_prompt}\nUser: {user_input}\nAssistant:"
        encoded = tokenizer(full_prompt, return_tensors="pt", truncation=True, padding=True)
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic, no randomness
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response = decoded.split("Assistant:")[-1].strip()
        return response

    def query_local_llm_stream_with_context(self, user_input, model, tokenizer, system_prompt, max_new_tokens=50):
        full_prompt = f"{system_prompt}\nUser: {user_input}\nAssistant:"
        encoded = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        with torch.no_grad():
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                decode_kwargs={"clean_up_tokenization_spaces": True}
            )
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "streamer": streamer,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id
            }

            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            response = ""
            for token in streamer:
                token_clean = token.replace("Assistant:", "").replace("User:", "").strip()
                if "user:" in token_clean.lower():
                    break
                response += token_clean + " "
                yield token_clean + " "
        return response.strip()

    def load_model(self, device):
        compute_dtype = torch.float16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True,
            trust_remote_code=False
        )
        tokenizer.pad_token = tokenizer.eos_token

        offload_path = "offload_dir"
        os.makedirs(offload_path, exist_ok=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            offload_folder=offload_path,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False
        )
        try:
            peft_model = PeftModel.from_pretrained(base_model, "./tinyllama_CS_lora", offload_folder=offload_path)
            peft_model = peft_model.merge_and_unload()
        except (KeyError, OSError) as e:
            print(f"❗ Error applying LoRA adapter: {e}")
            peft_model = base_model  # Fallback

        return tokenizer, peft_model, base_model

    @staticmethod
    def load_OpenAI_model():

        from openai import OpenAI
        from openai_api_key_verifier import verify_api_key, check_model_access, list_models, get_account_usage  
        # Replace with your actual API key
        api_key = os.getenv("OPENAI_API_KEY")
        # Verify if the API key is valid
        is_valid = verify_api_key(api_key)
        ai_model = "gpt-4o-mini"
        if is_valid and check_model_access(api_key, ai_model):
            st.success("API key to OpenAI is valid!")
        else:
            st.error("API key is invalid. Please check the connection to Open AI")

        client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))
        
        return client, ai_model

    def getport(self):
        # VID = 0x0483
        # PID = 0x5740
        # timeout_seconds = 60
        # start_time = time.time()

        # while True:
            # device_list = list_ports.comports()
            # for deviceV in device_list:
                # if deviceV.vid == VID and deviceV.pid == PID:
                    # st.success(f"Device found: {deviceV.device}")
                    # return deviceV.device

            # if time.time() - start_time > timeout_seconds:
                # raise OSError("RS device not found after waiting 60 seconds")

            # st.write("Waiting for RS device to be connected...")
            time.sleep(10)



    def configure_CS(self, opt, lang, d=0):

        def t(text):
            """Translate text to target language."""
            if not TRANSLATOR_AVAILABLE or not lang or lang == "en":
                return text
            try:
                return GoogleTranslator(source='auto', target=lang).translate(text)
            except Exception:
                return text

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

            st.write(t("[✅] Saved CSV:") + f" {csv_file}")
            with st.spinner("🔄 Processing, please wait..."):
                time.sleep(3)  # simulate long task
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
            time_domain_power = 10*np.log10(np.sum(np.abs(iq)**2) / len(iq) ) # Power in time domain
            #st.write(f"Time Domain Power: {time_domain_power}")
            return fig

        def plot_fft(iq, sample_rate, centerFreq=0, lang="en"):
            N = len(iq)
            fft_data = np.fft.fftshift(np.fft.fft(iq))
            freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/sample_rate))
            magnitude_db = 10 * np.log10(np.abs(fft_data)**2/N + 1e-17)
            magnitude_db[N // 2] = magnitude_db[N // 2]-60
            
            # Verify power consistency
            freq_domain_power = 10*np.log10(np.sum(np.abs(fft_data)**2)/N)  # Power in frequency domain
            #st.write(f"Freq Domain Power: {freq_domain_power}")
            text = t("Detected high DC component of > 60dB above noise floor. Suppressed the DC Component")
            #st.write(text)        

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot((freqs + centerFreq) / 1e6, magnitude_db)
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Magnitude (dB)')
            ax.set_title('FFT Spectrum')
            ax.grid(True)
            return (translated, fig)


        def setup_power(nv, freq_hz):
            nv.write("SENSe:FUNCtion 'POWer:AVG'")  # Set to power mode
            nv.write(f"SENSe:FREQuency:CENTer {freq_hz}")
            nv.write("UNIT:POWer DBM")   # or WATt
            nv.write("SENSe:BANDwidth:RESolution 5E5")  # 1 MHz RBW
            nv.write("INITiate:CONTinuous OFF")        # Manual trigger
            nv.write("TRIGger:SOURce IMM")             # Immediate trigger
            time.sleep(0.05)

        def measure_power(nv, freq_hz):
            setup_power(nv, freq_hz)
            nv.write("INITiate:IMMediate")
            time.sleep(0.1)
            power_str = nv.query("FETCH:POWer:AVG?").strip()
            return float(power_str)

        def power_sweep(nv, center_freq_hz=750e6, span_hz=300e6, step_hz=1e6):
            freqs = np.arange(center_freq_hz - span_hz / 2,
                              center_freq_hz + span_hz / 2 + step_hz, step_hz)
            powers = []
            for f in freqs:
                p = measure_power(nv, f)
                powers.append(p)
            return freqs, powers

        def plot_spectrum(freqs, powers):
            plt.rcParams['font.family'] = 'Arial'  # or 'sans-serif', 'Times New Roman', etc.
            plt.rcParams['font.size'] = 10         # global font size (can be dynamic)  
            fig, ax = plt.subplots()
            ax.plot(freqs / 1e6, powers)
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Power (dBm)")
            ax.set_title("Power vs. Frequency")
            ax.grid(True)
            return fig

        def pcap_playback():
            #ip = "192.168.1.101"  # IP address of the system
            #uribase = "http://"+ip+"/api/v1/"
            st.write(t("Started Loading the PCAP file"))

            #pcapfile = "oran_uplane_output_interf_det_awgn_new_DL.pcap"
            uploaded_file = None
            #uploaded_file = st.file_uploader("Drag and drop or browse a Fronthaul PCAP file", type=["pcap", "pcapng"])
            #time.sleep(15)
            if uploaded_file is not None:
                # Save to Streamlit's directory
                save_path = os.path.join(os.getcwd(), uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(t("Saved file to:") + f" {save_path}")

                # Prepare payload for Flask
                pcapfile = {"filepath": save_path}
                
            else: 
                #pcapfile = "oran_uplane_output_interf_det_awgn_new_DL_UL.pcap"
                pcapfile = "test_ip36to37_9000B.pcap"
                st.write(t("Selected Fronthaul File:") + f" {pcapfile}")
            

            # # ---------------------------------------------------------------------
            # # Upload the PCAP file; remove any PCAP file that's there first.
            # requests.post(uribase+"players/1/remove/").raise_for_status()
            # response = requests.post(uribase+"players/1/upload/",
                                     # files={"pcap": open(pcapfile, "rb")})
            # response.raise_for_status()  # Raise an exception if request failed

            # # ---------------------------------------------------------------------
            # # Start playing the file.
            # response = requests.post(uribase+"players/1/start/")
            # response.raise_for_status()  # Raise an exception if request failed

            # # ---------------------------------------------------------------------
            # # Wait for the playback to finish.
            # while True:
                # time.sleep(5)
                # response = requests.get(uribase+"players/1/")
                # response.raise_for_status()
                # if response.json()['state'] != "PLAYING":
                    # break

            st.write(t("Done!"))

        # === Helper: list interfaces with IPs ===
        def list_interfaces_with_ips():
            print("\n=== Available Interfaces ===")
            for idx, iface in enumerate(get_if_list()):
                try:
                    ip = get_if_addr(iface)
                except Exception:
                    ip = "No IP assigned"
                print(f"[{idx}] {iface:25} IP: {ip}")
            print()

        def choose_interface(prompt):
            list_interfaces_with_ips()
            choice = int(input(f"Select interface index for {prompt}: "))
            return get_if_list()[choice]

        def capture_job():
            print(f"[INFO] Starting pyshark capture on {iface_in} ...")
            cap.sniff(timeout=25)
            #cap.sniff(packet_count=1000)
            print(f"[INFO] Capture finished, {len(cap)} packets saved to {pcap_file_out}")

        def list_eth_interfaces():
            eths = {}
            addrs = psutil.net_if_addrs()
            for iface, details in addrs.items():
                for addr in details:
                    if addr.family == socket.AF_INET:
                        if not addr.address.startswith("127.") and not addr.address.startswith("169.254."):
                            eths[iface] = addr.address
            return eths

        # Helper to select IP and interface
        def pick_ip(role: str, key: str):
            ip_list = []
            iface_by_ip = {}
            
            for iface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        ip = addr.address
                        ip_list.append(ip)
                        iface_by_ip[ip] = iface

            selected_ip = st.selectbox(f"{role} IP Address", ["-- Select --"] + ip_list, key=key)
            if selected_ip == "-- Select --":
                return None, None
            return selected_ip, iface_by_ip[selected_ip]
    
        def run_pcap_replay_with_capture(pcap_file_in, pcap_file_out, iface_out, iface_in, src_ip):
            packets = rdpcap(pcap_file_in)
            for pkt in packets:
                pkt.time = time.time()

            capture_result = {}

            def run_capture():
                try:
                    cap = pyshark.LiveCapture(
                        interface=iface_in,
                        output_file=pcap_file_out,
                        custom_parameters=["-B", "64"],
                        bpf_filter=f"src host {src_ip}"
                    )
                    cap.sniff(timeout=25)
                    capture_result["count"] = len(cap)
                except Exception as e:
                    capture_result["error"] = str(e)

            cap_thread = threading.Thread(target=run_capture)
            cap_thread.start()

            time.sleep(1)
            t_start = time.time()
            sendp(packets, iface=iface_out, inter=0.0005, loop=0, verbose=False)
            t_end = time.time()
            cap_thread.join()

            return {
                "duration": t_end - t_start,
                "pkts_sent": len(packets),
                "bytes_sent": sum(len(pkt) for pkt in packets),
                **capture_result
            }

    # resource = 'TCPIP::nrq6-101528::hislip0'
        # nv = RsInstrument(resource, id_query=True, reset=True)
        # nv.visa_timeout = 20000  # Timeout in milliseconds
        # gcf ={}
        # if opt.command:
            # for c in opt.command:
                # nv.send_command(c + "\r")
                # data = nv.fetch_data()

        if opt.port:
            if (d == 0):
                pcap_playback()
        
        else:
            if opt.capture:
                # === Select NICs ===
                #iface_out = choose_interface("Output (send PCAP to Gi0/0/0/1)")
                #iface_in  = choose_interface("Capture (connected to either Gi0/0/0/3 or Gi0/0/0/5)")



                st.title("PCAP Replay + Capture")

                # Step 1: Interface Selection
                #st.markdown("### Step 1: Select Interfaces")

                #src_ip_out, iface_out = pick_ip("Output Interface (Send)", "ip_out_key")
                #src_ip_in, iface_in = pick_ip("Capture Interface (Receive)", "ip_in_key")
                
                iface_out = st.session_state.iface_out
                iface_in = st.session_state.iface_in
                src_ip_out = st.session_state.selected_ip_out
                src_ip_in = st.session_state.selected_ip_in
                
                #src_mac_out = get_if_hwaddr(iface_out)
                src_mac_out = f"02:11:22:33:44:55"

                if src_ip_out and iface_out and src_ip_in and iface_in:
                    st.success(t("✅ Interfaces Selected"))
                    st.write(t("Send via:") + f" {iface_out} ({src_ip_out})")
                    st.write(t("Capture on:") + f" {iface_in} ({src_ip_in})")

                    pcap_file_out = st.text_input("Playback PCAP File", "oran_uplane_DLUL_cisco.pcap")
                    pcap_file_in = st.text_input("Captured Output File", "capture_output.pcapng")

                    #if st.button("🚀 Replay & Capture"):
                    #with st.spinner("Sending and capturing..."):
                    print(iface_out)
                    print(iface_in)
                    print(src_ip_out)
                    print(pcap_file_out)
                    print(pcap_file_in)

                    #run first flask_pcap_backend.py
                    res = requests.post("http://localhost:8050/replay_and_capture", json={
                        "iface_out": iface_out,
                        "iface_in": iface_in,
                        "src_mac": src_mac_out,
                        "pcap_file_in": pcap_file_in,
                        "pcap_file_out": pcap_file_out
                    })

                    if res.ok:
                        result = res.json()
                        #st.success("✅ Capture Successfully Completed!")
                        st.success(t("✅ Successfully Captured on Ennoia ECP!"))
                        st.write(t("Sent:") + f" {result['pkts_sent']} pkts | {result['bytes_sent']/1e6:.2f} MB")
                        st.write(t("Captured in:") + f" {result['output_file']}")
                    else:
                        st.error(t("❌ Error:") + f" {res.status_code}")

                else:
                    st.info(t("Please select both Output and Capture interfaces to proceed."))
            

                # # === Stats ===
                # pkts_sent = len(packets)
                # pkts_recv = len(sniffer)
                # loss = pkts_sent - pkts_recv
                # loss_pct = (loss / pkts_sent * 100) if pkts_sent else 0
                # throughput_bps = (bytes_sent * 8) / duration if duration > 0 else 0
                # throughput_mbps = throughput_bps / 1e6
                # throughput_gbps = throughput_bps / 1e9

                # st.write("\n=== TEST RESULTS ===")
                # st.write(f"Packets Sent     : {pkts_sent}")
                # st.write(f"Packets Received : {pkts_recv}")
                # st.write(f"Packet Loss      : {loss} ({loss_pct:.2f}%)")
                # st.write(f"Test Duration    : {duration:.3f} sec")
                # st.write(f"Throughput       : {throughput_mbps:.2f} Mbps ({throughput_gbps:.3f} Gbps)")


                # print(params)  # Print out the filename
                # with open(params['filename'], 'wb') as f:
                    # for chunk in response.iter_content(100000):
                        # f.write(chunk)
                # text = (f"Data Transfer to {params['filename']} was successfully done!")
                # translated = GoogleTranslator(source='auto', target=lang).translate(text)
                # st.write(translated)
                
                streamlit_directory = os.getcwd()
                filename = pcap_file_in
                full_path = os.path.join(streamlit_directory, filename)
                st.write("Sending this file to SiMon PCAP Analyzer:", full_path)
                payload = {"filepath": full_path}
                
                try:                    
                    response = requests.post("http://localhost:5002/upload", json=payload)
                    st.success(f"SiMon says: {response.text}")
                except Exception as e:
                    st.error(f"http://localhost:5002 is not set: {e}")

                image = Image.open("plot2.png")
                st.image(image, caption="Interference Detection", use_container_width=True)
                    
                image = Image.open("plot1.png")
                st.image(image, caption="Equalized Rx Constellation", use_container_width=True)

                while True:
                    status = requests.get("http://localhost:5002/progress").json()
                    st.write(f"Analyzer Status: {status['status']}")
                    if "Completed" in status['status'] or "Error" in status['status'] or "Idle" in status['status']:
                        break
                    time.sleep(10)
                    
        return

