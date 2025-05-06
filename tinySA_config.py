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
                    st.success(f"Device found: {deviceV.device}")
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
                    s = nv.data(p)

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
