"""
ORAN PCAP Analyzer Configuration Helper
Provides helper functions for integrating ORAN PCAP analysis with Streamlit
via Flask backend (since pyshark cannot run directly in Streamlit).
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# Flask server configuration
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5002
FLASK_URL = f"http://{FLASK_HOST}:{FLASK_PORT}"

class ORANHelper:
    """Helper class for ORAN PCAP Analysis integration"""

    def __init__(self):
        self.flask_url = FLASK_URL
        self.analysis_results = None

    @staticmethod
    def select_checkboxes():
        """Display model selection checkboxes"""
        st.sidebar.subheader("Model Selection")
        slm_option = st.sidebar.checkbox("Use SLM (Local Model)", value=False)
        openai_option = st.sidebar.checkbox("Use OpenAI (Cloud)", value=True)

        selected = []
        if slm_option:
            selected.append("SLM")
        if openai_option:
            selected.append("OpenAI")
        return selected

    @staticmethod
    def load_OpenAI_model():
        """Load OpenAI model for chat functionality"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)
        model = "gpt-4o-mini"
        return client, model

    @staticmethod
    def load_lora_model():
        """Load local TinyLlama model for offline operation"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            import gc

            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

            # Aggressively clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()

            # Check available GPU memory
            if torch.cuda.is_available():
                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                free_mem_gb = free_mem / (1024**3)
                print(f"Available GPU memory: {free_mem_gb:.2f} GB")

                if free_mem_gb < 1.0:
                    print("Not enough GPU memory, using CPU instead")
                    device = "cpu"
                else:
                    device = "cuda"
            else:
                device = "cpu"

            print(f"Loading SLM model: {model_name} on {device}...")

            # Try local first, fall back to download
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            except Exception:
                print("Model not cached locally, downloading...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)

            if device == "cuda":
                try:
                    # Try 4-bit quantization first
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
                    # Don't call .to() when using device_map="auto"
                    print(f"SLM model loaded successfully with 4-bit quantization")
                    return tokenizer, model, device
                except Exception as cuda_error:
                    print(f"CUDA 4-bit load failed: {cuda_error}, trying float16...")
                    torch.cuda.empty_cache()
                    gc.collect()

                    try:
                        # Try float16 without quantization
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            low_cpu_mem_usage=True
                        )
                        print(f"SLM model loaded successfully with float16")
                        return tokenizer, model, device
                    except Exception as fp16_error:
                        print(f"CUDA float16 load failed: {fp16_error}, falling back to CPU")
                        torch.cuda.empty_cache()
                        gc.collect()
                        device = "cpu"

            # CPU loading - explicitly disable features that cause meta tensor issues
            print("Loading model on CPU (this may take a moment)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                device_map=None,  # Explicitly disable device_map
                trust_remote_code=False
            )
            # Model should already be on CPU, but move explicitly if needed
            if next(model.parameters()).device.type != device:
                model = model.to(device)

            print(f"SLM model loaded successfully on {device}")
            return tokenizer, model, device
        except Exception as e:
            st.error(f"Failed to load local model: {e}")
            import traceback
            traceback.print_exc()
            return None, None, "cpu"

    def get_system_prompt(self):
        """Return the system prompt for ORAN analysis"""
        return """You are an expert O-RAN fronthaul packet analyzer following the O-RAN Alliance specifications.
You help users analyze PCAP files containing O-RAN fronthaul data and detect interference patterns.

Your capabilities include:
1. Analyzing BFP9 compressed IQ data from O-RAN U-plane packets
2. Detecting interference using DMRS-based channel estimation
3. Performing blind interference detection using AI-based EVM analysis
4. Generating reports on signal quality, EVM, and interference regions

When users ask about PCAP analysis, provide helpful guidance on:
- Required parameters (N_ID, nSCID, bandwidth, SCS)
- How to interpret interference detection results
- Understanding PRB-level and layer-level interference reports

Always be precise about technical details and provide actionable recommendations."""

    def get_few_shot_examples(self):
        """Return few-shot examples for the AI model"""
        return [
            {
                "role": "user",
                "content": "What parameters do I need to analyze an O-RAN PCAP file?"
            },
            {
                "role": "assistant",
                "content": """To analyze an O-RAN PCAP file, you'll need the following parameters:

1. **N_ID (Cell ID)**: The physical cell identity (0-1007), typically 100 for testing
2. **nSCID**: Scrambling ID (0 or 1), typically 0
3. **Bandwidth**: 5, 10, 15, 20, 50, or 100 MHz
4. **SCS (Subcarrier Spacing)**: 15 or 30 kHz
5. **Number of Layers**: 1, 2, or 4 for MIMO configurations
6. **Link Direction**: Uplink or Downlink

The analyzer will process the BFP9 compressed IQ data and perform:
- 4x4 MIMO equalization using DMRS
- SNR estimation per subcarrier
- Interference detection per PRB and layer"""
            },
            {
                "role": "user",
                "content": "How does interference detection work?"
            },
            {
                "role": "assistant",
                "content": """The interference detection works in two modes:

**1. DMRS-Based Detection (Standard)**
- Uses known DMRS pilot sequences for channel estimation
- Computes SNR per subcarrier by comparing received vs expected DMRS
- Detects interference by finding PRB regions where SNR drops significantly (>10dB)

**2. AI-Based Blind Detection**
- Does not require knowledge of DMRS values
- Uses EVM (Error Vector Magnitude) analysis on QPSK constellation
- Applies phase correction and normalization
- Detects interference regions by analyzing EVM spikes per PRB

Both methods output:
- Layer-by-layer interference regions (PRB start/end)
- Constellation plots showing equalization quality
- Interference level plots per PRB"""
            }
        ]

    def check_flask_server(self):
        """Check if the Flask server is running"""
        try:
            response = requests.get(f"{self.flask_url}/progress", timeout=2)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except Exception:
            return False

    def get_progress(self):
        """Get analysis progress from Flask server"""
        try:
            response = requests.get(f"{self.flask_url}/progress", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"status": "Unknown"}
        except Exception as e:
            return {"status": f"Error: {str(e)}"}

    def analyze_pcap(self, filepath, params=None):
        """
        Send PCAP file path to Flask server for analysis

        Args:
            filepath: Full path to the PCAP file
            params: Optional dict with analysis parameters
                   (N_ID, nSCID, bandwidth, scs, layers, link_direction)

        Returns:
            dict with analysis results or error message
        """
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filepath}"}

        if not self.check_flask_server():
            return {"error": "Flask server not running. Please start packet_oran_analysis_det_st.py first."}

        try:
            payload = {"filepath": filepath}
            if params:
                payload.update(params)

            response = requests.post(
                f"{self.flask_url}/upload",
                json=payload,
                timeout=300  # 5 minute timeout for large files
            )

            if response.status_code == 200:
                self.analysis_results = response.text
                return {"success": True, "message": response.text}
            else:
                return {"error": f"Server error: {response.status_code} - {response.text}"}

        except requests.exceptions.Timeout:
            return {"error": "Analysis timed out. The PCAP file may be too large."}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

    def get_analysis_plots(self, plot_dir="."):
        """
        Get paths to generated analysis plots

        Args:
            plot_dir: Directory where plots are saved

        Returns:
            dict with plot file paths organized by detection mode
            - DMRS mode: plot1.png (constellation), plot2.png (SNR detection)
            - AI mode: plot1.png (constellation), plot2.png (interference detection)
            - Both mode: plot1.png, plot2.png (DMRS), plot3.png, plot4.png (AI)
        """
        plots = {}
        # Check for all possible plot files
        plot_files = ["plot1.png", "plot2.png", "plot3.png", "plot4.png"]

        for pf in plot_files:
            path = os.path.join(plot_dir, pf)
            if os.path.exists(path):
                plots[pf] = path

        return plots

    def load_analysis_csv(self, csv_path):
        """Load analysis results from CSV file"""
        try:
            if os.path.exists(csv_path):
                return pd.read_csv(csv_path, header=None)
            return None
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return None


# Blind detection functions (from gen_full_frame_channel_awgn_ai.py)
def gen_QAM_ref(M):
    """Generate M-QAM reference constellation"""
    k = int(np.sqrt(M))
    iq = np.arange(-k + 1, k, 2)
    I, Q = np.meshgrid(iq, iq)
    const_M = I.flatten() + 1j * Q.flatten()
    const_M /= np.sqrt((np.abs(const_M)**2).mean())
    return const_M

def correct_qpsk_phase_drift(iq_data):
    """Estimate and correct constant phase rotation using 4th-power QPSK method"""
    angle_est = np.angle(np.mean(iq_data ** 4)) / 4
    corrected = iq_data * np.exp(-1j * angle_est)
    return corrected

def evm_calc(rx_symbol, M):
    """Calculate EVM for received symbols against M-QAM constellation"""
    const_M = gen_QAM_ref(M)
    min_dist = np.zeros(len(rx_symbol))

    for i in range(len(rx_symbol)):
        distance = np.abs(rx_symbol[i] - const_M.flatten())
        min_dist[i] = (np.min(distance))**2

    evm = np.sqrt(np.sum(min_dist)/len(rx_symbol))
    return evm

def blind_interference_detection(rx_frame, numREs, numLayers, PRB=12):
    """
    Perform blind interference detection using EVM analysis

    Args:
        rx_frame: Received frame data (slots, symbols, layers, REs)
        numREs: Number of resource elements
        numLayers: Number of MIMO layers
        PRB: PRB size (default 12)

    Returns:
        dict with interference regions per layer
    """
    M1 = 4  # QPSK modulation for DMRS

    evm_per_layer = {}
    for layer in range(numLayers):
        evm_per_layer[layer] = np.zeros(numREs)

    # Calculate EVM per RE for each layer
    for i in range(numREs):
        if i % 2 != 0:
            for layer in range(numLayers):
                evm_per_layer[layer][i] = evm_per_layer[layer][i-1]
            continue

        # Layers 0,1 on even REs
        for layer in [0, 1]:
            if layer < numLayers:
                data = rx_frame[:, 2, layer, i]  # DMRS symbol 2
                corrected = correct_qpsk_phase_drift(data) * np.exp(-1j * np.pi/4)
                corrected = correct_qpsk_phase_drift(corrected) * np.exp(-1j * np.pi/4)
                corrected /= np.sqrt((np.abs(corrected)**2).mean())
                evm_per_layer[layer][i] = evm_calc(corrected.flatten(), M1) + 1e-12

        # Layers 2,3 on odd REs
        if i + 1 < numREs:
            for layer in [2, 3]:
                if layer < numLayers:
                    data = rx_frame[:, 2, layer, i+1]
                    corrected = correct_qpsk_phase_drift(data) * np.exp(-1j * np.pi/4)
                    corrected /= np.sqrt((np.abs(corrected)**2).mean())
                    evm_per_layer[layer][i] = evm_calc(corrected.flatten(), M1) + 1e-12

    # Convert to dB and compute per-PRB averages
    evm_db_PRB = {}
    num_prbs = numREs // PRB

    for layer in range(numLayers):
        evm_prb = np.zeros(num_prbs)
        for p in range(num_prbs):
            for j in range(PRB):
                evm_prb[p] += evm_per_layer[layer][PRB*p + j]
        evm_db_PRB[layer] = 20 * np.log10(evm_prb / PRB + 1e-12)

    # Detect interference regions
    upper_threshold = 10.0
    lower_threshold = -10.0

    interference_regions = {}
    for layer in range(numLayers):
        evm_diff = np.diff(evm_db_PRB[layer])
        snr_diff = -evm_diff  # Invert for SNR-like interpretation

        regions = detect_snr_drop_regions(snr_diff, lower_threshold, upper_threshold)
        interference_regions[layer] = regions

    return interference_regions, evm_db_PRB

def detect_snr_drop_regions(diff_vector, lower_thresh=-10.0, upper_thresh=10.0):
    """Detect regions where SNR drops below threshold and recovers"""
    drop_regions = []
    in_drop = False
    start_idx = 0

    for i, val in enumerate(diff_vector):
        if not in_drop and val < lower_thresh:
            in_drop = True
            start_idx = i
        elif in_drop and val > upper_thresh:
            drop_regions.append((start_idx, i))
            in_drop = False

    if in_drop:
        drop_regions.append((start_idx, len(diff_vector)))

    return drop_regions


# DMRS generation function
def generate_dmrs_type1_standard(N_ID, nSCID, n, l, numREs, layer):
    """Generate DMRS Type 1 sequence for 5G NR"""
    assert layer in [0, 1, 2, 3], "Only 4 layers supported"
    ref_layer = layer if layer in [0, 2] else layer - 1
    lam = 0 if ref_layer in [0, 1] else 1
    M = 2 * numREs
    Nc = 1600
    c_init = int((2**17 * (n//2 + 1) * (2*N_ID + 1)) + 2*nSCID + l + (2**14 * lam)) % (2**31)
    x = np.zeros(Nc + M, dtype=int)
    x[:31] = [1] + [0]*30
    for i in range(31, Nc + M):
        x[i] = (x[i-28] + x[i-31]) % 2
    c = np.zeros(M, dtype=int)
    for i in range(M):
        c[i] = (x[Nc + i] + ((c_init >> (i % 31)) & 1)) % 2
    bpsk = 1 - 2 * c
    dmrs_seq = (bpsk[0::2] + 1j * bpsk[1::2]) / np.sqrt(2)
    k_offset = 0 if layer in [0, 1] else 1
    re_indices = np.arange(k_offset, numREs, 2)
    grid = np.zeros(numREs, dtype=complex)
    grid[re_indices] = dmrs_seq[:len(re_indices)]
    if layer in [1, 3]:
        toggle = np.ones(len(re_indices), dtype=complex)
        toggle[1::2] = -1
        grid[re_indices] *= toggle
    return grid
