# tinySA_config.py
import os
import json
import torch
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from serial.tools import list_ports
import tinySA
import matplotlib.pyplot as plt
import time
import streamlit as st
import csv
import numpy as np
from scipy.signal import find_peaks
from dotenv import load_dotenv


class TinySAHelper:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        
    def get_system_prompt(self):
        # Using LLaMA 2-style formatting
        system_prompt = (
            "You are Ennoia, an AI assistant specifically for the TinySA spectrum analyzer (www.tinysa.org).\n"
            "Your role is to help users configure, troubleshoot, and operate the TinySA only.\n"
            "Assume every question is about TinySA.\n"
            "If the user asks anything unrelated, reply with: 'I can only assist with queries related to the TinySA spectrum analyzer.'\n"
            "Do not provide generic RF advice. Never ask follow-up questions. Repeat all numeric values exactly.\n"
            "Answer only with complete sentences. Do not cut the sentence when displaying it.\n"
            "The information will be provided in a numbered format\n"
        )

        return system_prompt

        
    def get_few_shot_examples(self):
        # === Few-Shot Examples ===
        few_shot_examples = [
            {
                "role": "user",
                "content": ("Set the start frequency to 300 MHz")
            },
            {
                "role": "assistant",
                "content": (
                    "To set the start frequency to 300 MHz on the TinySA, press the “Start” button, then use "
                    "the navigation keys or rotary encoder to adjust the value to 300 MHz. Confirm the value by pressing 'OK'."
                )
            },
            {
                "role": "user",
                "content": ("Set the stop frequency to 900 MHz")
            },
            {
                "role": "assistant",
                "content": (
                    "To set the stop frequency to 900 MHz on the TinySA, press the “Stop” button, then use the rotary "
                    "knob or arrow keys to change the value to 900 MHz. Confirm by pressing 'OK'."
                )
            },
            {
                "role": "user",
                "content": ("Set the start frequency to 300 MHz and stop frequency to 900 MHz")
            },
            {
                "role": "assistant",
                "content": (
                    "To set the start frequency to 300 MHz and stop frequency to 900 MHz on the TinySA, press the “Start” button, then use the rotary "
                    "knob or arrow keys to change the value to 300 MHz. Confirm by pressing 'OK'. Then press the “Stop” button and adjust the value to 900 MHz. Confirm by pressing 'OK'."
                )
            }, 
                        {
                "role": "user",
                "content": ("How to configure tinySA spectrum analyzer")
            },
            {
                "role": "assistant",
                "content": (
                "tinySA Configuration (Quick Guide)\n\n"
                "Power on via USB or switch.\n"
                "Set Start and Stop frequencies (e.g., 100M to 500M).\n"
                "Choose Points (e.g., 101).\n"
                "Select Input: Low (≤350 MHz) or High (≤960 MHz / 5.3 GHz on Ultra).\n"
                "Use Markers to find peaks, Traces to view signal curves.\n"
                "Enable TG (Tracking Generator) for filters or cable tests.\n"
                "Run Level Cal under Cal for amplitude accuracy.\n"
                "Use Save/Recall to store or load settings.\n"
                "For more, visit tinySA.org."
                )
            }, 
            {
                "role": "user",
                "content": ("This is a random query not related to tinySA spectrum analyzer"
                )
            },
            {
                "role": "assistant",
                "content": (
                "I can only assist with queries related to the TinySA spectrum analyzer."
                )
            }, 
            
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

    def load_lora_model(base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", lora_path="./tinyllama_tinysa_lora"):
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
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)
        tokenizer.pad_token = tokenizer.unk_token  # Recommended for LLaMA models
        tokenizer.use_default_system_prompt = False  # Avoid auto-formatting if needed

        # Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": device},
            local_files_only=True
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
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token

        offload_path = "offload_dir"
        os.makedirs(offload_path, exist_ok=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            offload_folder=offload_path,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        try:
            peft_model = PeftModel.from_pretrained(base_model, "./tinyllama_tinysa_lora", offload_folder=offload_path)
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
        VID = 0x0483
        PID = 0x5740
        timeout_seconds = 60
        start_time = time.time()

        while True:
            device_list = list_ports.comports()
            for deviceV in device_list:
                if deviceV.vid == VID and deviceV.pid == PID:
                    st.success(f"Device found: Fujitsu Radio Unit")
                    return deviceV.device

            if time.time() - start_time > timeout_seconds:
                raise OSError("TinySA device not found after waiting 60 seconds")

            st.write("Waiting for TinySA device to be connected...")
            time.sleep(10)



    def configure_tinySA(self, opt):
        nv = tinySA.tinySA(opt.device or self.getport())
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
            nv.set_frequencies(opt.start, opt.stop, opt.points)

        if opt.plot or opt.save or opt.scan:
            p = int(opt.port) if opt.port else 0
            if opt.scan or opt.points > 101:
                s = nv.scan()
                s = s[p]
            else:
                if opt.start or opt.stop:
                    nv.set_sweep(opt.start, opt.stop)
                    nv.fetch_frequencies()
                    s = nv.send_scan(opt.start, opt.stop)
                    s = nv.data(p)
                    nv.resume()

        if opt.save:
            nv.writeCSV(s, opt.save)

        if opt.plot:
            nv.logmag(s)
            # Adding axis labels
            plt.xlabel("Frequency (Hz)")  # Name for the x-axis
            plt.ylabel("Signal strength (dBm)")  # Name for the y-axis

            # Adding a title (optional)
            plt.title("Signal strength (dBm) vs Frequency (Hz)")
            gcf = plt.gcf()
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