# RS_config.py
import os
import json
import torch
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from serial.tools import list_ports
from RsInstrument import *
import matplotlib.pyplot as plt
import time
import streamlit as st
import csv
import numpy as np
from scipy.signal import find_peaks
import struct
import pandas as pd


class RSHelper:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        
    def get_system_prompt(self):
        # Using LLaMA 2-style formatting
        system_prompt = (
            "You are Ennoia, an AI assistant specifically for the R&S NRQ6 frequency selective power sensor  (https://www.rohde-schwarz.com/).\n"
            "Your role is to help users configure, troubleshoot, and operate the R&S power sensor only.\n"
            "Assume every question is about Rohde & Schwarz.\n"
            "If the user asks anything unrelated, reply with: 'I can only assist with queries related to the R&S power sensor.'\n"
            "Do not provide generic RF advice. Never ask follow-up questions. Repeat all numeric values exactly.\n"
        )

        return system_prompt

        
    def get_few_shot_examples(self):
        # === Few-Shot Examples ===
        few_shot_examples = [
            {"role": "user", "content": "How do I set the center frequency to 2 GHz?"},
            {"role": "assistant", "content": "To set the center frequency to 2 GHz, use the command: SENSe:FREQuency:CENTer 2e09"},
            
            {"role": "user", "content": "What is the maximum trace length I can set?"},
            {"role": "assistant", "content": "The maximum trace length you can set is 15 million samples using: SENSE:TRACe:IQ:RLENgth 15e6"},
            
            {"role": "user", "content": "Can you help me with the bandwidth settings?"},
            {"role": "assistant", "content": "Yes, you can set the bandwidth to 100 MHz using: SENSE:BANDwidth:RES 1e8"},
            
            {"role": "user", "content": "What is the command to initiate a measurement?"},
            {"role": "assistant", "content": "To initiate a measurement, use: INITiate:IMMediate"},
            
            {"role": "user", "content": "How do I change the sensor mode to I/Q?"},
            {"role": "assistant", "content": 'To change the sensor mode to I/Q, use: SENSe:FUNCtion "XTIMe:VOLT:IQ"'},
            
            {"role": "user", "content": "What is the command to fetch the IQ data?"},
            {"role": "assistant", "content": "You can fetch the IQ data using: FETCh1?"},
            
            {"role": "user", "content": "How do I save the captured IQ data to a file?"},
            {"role": "assistant", "content": "To save the captured IQ data, you can use a script to write it to a binary file."},
            
            {
                "role": "user",
                "content": "How do I configure the R&S NRQ6 power sensor?"
            },
            {
                "role": "assistant",
                "content": (
                    "To configure the R&S NRQ6 power sensor:\n\n"
                    "1. **Connect the sensor** via USB or LAN and install VISA drivers.\n"
                    "2. **Reset the device**: `*RST`\n"
                    "3. **Set frequency**: `SENSE:FREQ 2e9` (for 2 GHz)\n"
                    "4. **Enable averaging**: `SENSE:AVER:COUNT 10`\n"
                    "5. **Set bandwidth**: `SENSE:BAND:RES 1e6` (1 MHz)\n"
                    "6. **Trigger mode**: `TRIG:SOUR IMM` (immediate)\n"
                    "7. **Start measurement**: `INIT:IMM`\n"
                    "8. **Fetch result**: `FETCH:POWER:AC?` or `FETCH:POWER:PEAK?`\n\n"
                    "You can also use Python with PyVISA or the R&S Power Viewer Plus software for easier setup and control."
                )
            }
        ]
        return few_shot_examples      
        
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

    def load_lora_model(base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", lora_path="./tinyllama_RS_lora"):
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
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.unk_token  # Recommended for LLaMA models
        tokenizer.use_default_system_prompt = False  # Avoid auto-formatting if needed

        # Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": device}
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
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        offload_path = "offload_dir"
        os.makedirs(offload_path, exist_ok=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            offload_folder=offload_path,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True
        )
        try:
            peft_model = PeftModel.from_pretrained(base_model, "./tinyllama_RS_lora", offload_folder=offload_path)
            peft_model = peft_model.merge_and_unload()
        except (KeyError, OSError) as e:
            print(f"❗ Error applying LoRA adapter: {e}")
            peft_model = base_model  # Fallback

        return tokenizer, peft_model, base_model

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



    def configure_RS(self, opt):
        
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

            st.write(f"[✅] Saved CSV: {csv_file}")
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

        def plot_fft(iq, sample_rate, centerFreq=0):
            N = len(iq)
            fft_data = np.fft.fftshift(np.fft.fft(iq))
            freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/sample_rate))
            magnitude_db = 20 * np.log10(np.abs(fft_data) + 1e-12)
            magnitude_db[N // 2] = magnitude_db[N // 2]-60
            
            #text = f"Detected high DC component of > 60dB above noise floor. Suppressed the DC Component"
            #translated = GoogleTranslator(source='auto', target=lang).translate(text)
            #st.write(translated)        

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot((freqs +centerFreq) / 1e6, magnitude_db)
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Magnitude (dB)')
            ax.set_title('FFT Spectrum')
            ax.grid(True)
            return fig


        resource = 'TCPIP::nrq6-101528::hislip0'
        nv = RsInstrument(resource, id_query=True, reset=True)
        gcf ={}
        if opt.command:
            for c in opt.command:
                nv.send_command(c + "\r")
                data = nv.fetch_data()

        if opt.capture:
            img = nv.capture()
            img.save(opt.capture)
            return

        if opt.start or opt.stop or opt.points:
            #nv.set_frequencies(opt.start, opt.stop, opt.points)
            center = (opt.start + opt.stop) // 2
            res = (opt.stop - opt.start)
            if (res > 1e8):
                res = 1e8
            #inst.set_mode_iq()
            nv.write('SENSe:FUNCtion "XTIMe:VOLT:IQ"')  # Change sensor mode to I/Q
            #inst.set_center(2e09)
            nv.write(f"SENSe:FREQuency:CENTer {center}")  # Center Frequency to 2 GHz
            #inst.set_bw_res_manual()
            nv.write('SENSE:BANDwidth:RESolution:TYPE:AUTO:STATe OFF')  # Change bandwidth setting to manual state
            #inst.set_bw_res_normal()
            nv.write('SENSE:BANDwidth:RESolution:TYPE NORMal')  # Flat filter type
            #inst.set_bw_res(1e8)
            nv.write(f"SENSE:BANDwidth:RES {res}")  # Analysis bandwidth is 100 MHz now
            #inst.set_trace_length(15e5)
            nv.write('SENSE:TRACe:IQ:RLENgth 15e5')  # IQ trace length is 1.5 million samples now
            nv.write('')
            nv.write('FORM:DATA REAL,64')

           # nv.write('INITiate:IMMediate')  # Initiates a single trigger measurement
           # nv.visa_timeout = 10000  # Extend Visa timeout to avoid errors
           # output = nv.query_bin_or_ascii_float_list('FETCh1?')  # Get the measurement in binary format
           # nv.visa_timeout = 3000  # Change back timeout to standard value

            #save_float_list_as_bin(output, "iq_capture.bin")
            #iq_data = collect_iq_samples(nrq, trace_length=10000)
            #iq_pairs = bin_to_csv("iq_capture.bin", "iq_capture.csv")
            #nv.set_frequencies(opt.start, opt.stop, opt.points)

        if opt.plot or opt.save or opt.scan:
            # p = int(opt.port) if opt.port else 0
            # if opt.scan or opt.points > 101:
                # s = nv.scan()
                # s = s[p]
            # else:
                # if opt.start or opt.stop:
                    # nv.set_sweep(opt.start, opt.stop)
                    # nv.fetch_frequencies()
                    # s = nv.data(p)
            nv.write('INITiate:IMMediate')  # Initiates a single trigger measurement
            nv.visa_timeout = 10000  # Extend Visa timeout to avoid errors
            output = nv.query_bin_or_ascii_float_list('FETCh1?')  # Get the measurement in binary format
            nv.visa_timeout = 3000  # Change back timeout to standard value

            save_float_list_as_bin(output, "iq_capture.bin")
            #iq_data = collect_iq_samples(nrq, trace_length=10000)
            iq_pairs = bin_to_csv("iq_capture.bin", "iq_capture.csv")

        if opt.plot:
            # nv.logmag(s)
            # # Adding axis labels
            # plt.xlabel("Frequency (Hz)")  # Name for the x-axis
            # plt.ylabel("Signal strength (dBm)")  # Name for the y-axis

            # # Adding a title (optional)
            # plt.title("Signal strength (dBm) vs Frequency (Hz)")
            
            uploaded_file = "iq_capture.csv"  # Change to your file location
            sample_rate = res*1.2288

            if uploaded_file and os.path.exists(uploaded_file):
                iq = load_iq_csv(uploaded_file)
                st.subheader("🕒 Time-Domain Plot")
                st.pyplot(plot_time_domain(iq))

                st.subheader("🔊 Frequency-Domain Plot (FFT)")
                center = (opt.start + opt.stop) // 2
                st.pyplot(plot_fft(iq, sample_rate,center))
            else:
                st.info("Please upload a CSV file with 'I' and 'Q' columns.")            
            
            gcf = 0
        nv.close()
        return gcf
    
    
    def find_max_signal_strength_to_csv(self,file_list, output_filename="max_signal_strengths.csv", min_strength=-80):
        max_strengths = {}
        
        for filename in file_list:
            try:
                with open(filename, 'r') as file:
                    for line in file:
                        if not line.strip():
                            continue
                        
                        try:
                            freq, strength = map(float, line.split(','))
                            freq = int(freq)
                            # Only consider strengths >= min_strength
                            if strength >= min_strength:
                                if freq not in max_strengths or strength > max_strengths[freq]:
                                    max_strengths[freq] = strength
                                
                        except ValueError:
                            print(f"Skipping invalid line in {filename}: {line.strip()}")
                            continue
                        
            except FileNotFoundError:
                print(f"Could not find file: {filename}")
                continue
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        # Sort and save to CSV
        sorted_results = sorted(max_strengths.items())
        
        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Frequency', 'Signal_Strength'])  # Header
            writer.writerows(sorted_results)
        
        return dict(sorted_results)
    

    def read_signal_strength(self,filename):
        frequencies = []
        strengths = []
        filelist = ["output.csv"]

        # Generate CSVs from external scripts
        self.find_max_signal_strength_to_csv(filelist, output_filename="max_signal_strengths.csv", min_strength=-80)

        try:
            with open(filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    freq = int(float(row['Frequency']))
                    strength = float(row['Signal_Strength'])
                    frequencies.append(freq)
                    strengths.append(strength)
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return None
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            return None

        return strengths, frequencies


    def get_operator_frequencies(self):
        file_path = 'operator_table.json'
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                print(f"Loaded JSON from {file_path}")
                return data
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return {}
        else:
            print(f"File not found: {file_path}")
            return {}


    def analyze_signal_peaks(self,sstr, freq_mhz, operator_table, window_size=5, peak_height=-75, peak_distance=10):
        """
        Analyze signal peaks, group them by 3GPP band, and reduce the number of peaks.

        Parameters:
        - sstr: List or array of signal strengths.
        - freq_mhz: List or array of frequencies (in MHz) corresponding to `sstr`.
        - operator_table: List of dictionaries containing operator band info.
        - window_size: Number of samples before and after peak to average.
        - peak_height: Minimum signal strength (in dBm) to consider a peak.
        - peak_distance: Minimum distance (in MHz) between peaks to consider them as distinct.

        Returns:
        - List of dictionaries with band match information.
        """
        peaks, _ = find_peaks(sstr, height=peak_height)
        grouped_peaks = []

        for peak in peaks:
            freq = freq_mhz[peak]
            closest_band = None
            min_diff = float('inf')

            # Find the closest operator band to the peak frequency
            for band in operator_table:
                try:
                    uplink = [int(x.strip()) for x in band['Uplink Frequency (MHz)'].split(' - ')]
                    downlink = [int(x.strip()) for x in band['Downlink Frequency (MHz)'].split(' - ')]
                except Exception:
                    continue  # Skip malformed entries

                if uplink[0] <= freq <= uplink[1] or downlink[0] <= freq <= downlink[1]:
                    diff = min(
                        abs(uplink[0] - freq), abs(uplink[1] - freq),
                        abs(downlink[0] - freq), abs(downlink[1] - freq)
                    )
                    if diff < min_diff:
                        min_diff = diff
                        closest_band = band

            if closest_band:
                # Group peaks by band and distance
                found_group = False
                for group in grouped_peaks:
                    # Check if the peak is within the distance of an existing group
                    if abs(group['frequency'] - freq) <= peak_distance:
                        group['peaks'].append(peak)
                        found_group = True
                        break

                if not found_group:
                    # Create a new group for this peak
                    grouped_peaks.append({
                        'band': closest_band,
                        'frequency': freq,
                        'peaks': [peak]
                    })

        # Process the grouped peaks and compute their average strength
        result = []
        for group in grouped_peaks:
            band = group['band']
            all_peaks = group['peaks']
            # Get the average signal strength for all peaks in this group
            avg_strength = round(float(np.mean([sstr[peak] for peak in all_peaks])), 2)

            # Use the first and last peak's frequencies to define the frequency range
            start_idx = max(all_peaks[0] - window_size, 0)
            end_idx = min(all_peaks[-1] + window_size, len(sstr))
            freq_range = f"{int(freq_mhz[start_idx])} - {int(freq_mhz[end_idx - 1])}"

            result.append({
                "operator": band.get('Operators', 'Unknown'),
                "strength": avg_strength,
                "technology": band.get('Technology', 'Unknown'),
                "service": "Mobile",
                "frequency_range": freq_range,
                "band_3GPP": band.get('3GPP Band', 'Unknown'),
                "source": "Internal Database"
            })

        return result