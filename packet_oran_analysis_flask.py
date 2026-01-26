"""
O-RAN PCAP Analysis Flask Server
Uses scapy for PCAP reading and BFP9 decoding for IQ extraction.
All analysis is performed from the converted CSV data.
Equalization matches packet_oran_analysis_det_st.py exactly.

Usage:
    python packet_oran_analysis_flask.py

Servers:
    Port 5000 - Report display (app1)
    Port 5001 - Plot display (app2)
    Port 5002 - Upload/Analysis API:
        POST /upload - JSON with filepath to analyze
        GET /progress - Current analysis status
        GET /report - Generated report HTML
        GET /plots - Returns plot files as base64
"""

import os
import csv
import time
import datetime
import io
import base64
from threading import Thread, Event

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from flask import Flask, request, jsonify, render_template_string
import markdown2

# OpenAI for report generation
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set. Report generation will use fallback.")

# SLM (Local Model) support - TinyLlama (Offline, GPU)
SLM_MODEL = None
SLM_TOKENIZER = None
SLM_DEVICE = "cuda"

# Local path to TinyLlama model (offline mode)
SLM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "TinyLlama-1.1B-Chat-v1.0")
# Alternative: use HuggingFace cache if model was previously downloaded
SLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_slm_model():
    """Load TinyLlama model for offline report generation (GPU with CPU fallback)"""
    global SLM_MODEL, SLM_TOKENIZER, SLM_DEVICE

    if SLM_MODEL is not None:
        return True  # Already loaded

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        import gc

        # Try local path first (fully offline)
        if os.path.exists(SLM_MODEL_PATH):
            model_source = SLM_MODEL_PATH
            print(f"Loading SLM from local path: {model_source}")
        else:
            model_source = SLM_MODEL_NAME
            print(f"Loading SLM from cache: {model_source}")

        # Aggressively clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

            free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_mem_gb = free_mem / (1024**3)
            print(f"Available GPU memory: {free_mem_gb:.2f} GB")

            if free_mem_gb < 1.0:
                print("Not enough GPU memory, using CPU")
                SLM_DEVICE = "cpu"
            else:
                SLM_DEVICE = "cuda"
        else:
            SLM_DEVICE = "cpu"

        print(f"Loading SLM model on {SLM_DEVICE}...")
        SLM_TOKENIZER = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=True
        )

        if SLM_DEVICE == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                SLM_MODEL = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    quantization_config=quantization_config,
                    device_map="auto",
                    local_files_only=True
                )
                print(f"SLM model loaded on GPU ({torch.cuda.get_device_name(0)})")
            except Exception as cuda_error:
                print(f"CUDA failed: {cuda_error}, falling back to CPU")
                torch.cuda.empty_cache()
                gc.collect()
                SLM_DEVICE = "cpu"
                SLM_MODEL = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    torch_dtype=torch.float32,
                    local_files_only=True
                )
                SLM_MODEL.to(SLM_DEVICE)
                print("SLM model loaded on CPU")
        else:
            SLM_MODEL = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.float32,
                local_files_only=True
            )
            SLM_MODEL.to(SLM_DEVICE)
            print("SLM model loaded on CPU")

        return True
    except Exception as e:
        print(f"Failed to load SLM model: {e}")
        print("Make sure TinyLlama model is cached or placed in: " + SLM_MODEL_PATH)
        import traceback
        traceback.print_exc()
        return False


def generate_with_slm(prompt, max_new_tokens=256, analysis_data=None):
    """Generate text using TinyLlama model on GPU (CUDA)"""
    global SLM_MODEL, SLM_TOKENIZER, SLM_DEVICE

    if SLM_MODEL is None:
        if not load_slm_model():
            return None

    try:
        import torch

        # Build a short prompt with key analysis data
        if analysis_data:
            short_prompt = f"""Generate a brief O-RAN analysis report:
- File: {analysis_data.get('filename', 'Unknown')}
- Interference: {'Detected' if analysis_data.get('interference', 0) else 'None'}
- EVM: {analysis_data.get('evm', 'N/A')} dB
Keep response under 100 words."""
        else:
            short_prompt = "Generate a brief O-RAN fronthaul analysis summary. Keep it under 100 words."

        # Format prompt for TinyLlama chat format
        formatted_prompt = f"<|system|>\nYou are an O-RAN expert.</s>\n<|user|>\n{short_prompt}</s>\n<|assistant|>\n"

        print("SLM: Tokenizing prompt...")
        inputs = SLM_TOKENIZER(formatted_prompt, return_tensors="pt").to(SLM_DEVICE)

        print(f"SLM: Generating response on GPU ({SLM_DEVICE})...")
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Use automatic mixed precision for faster GPU inference
                outputs = SLM_MODEL.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    num_beams=1,
                    pad_token_id=SLM_TOKENIZER.eos_token_id
                )

        print("SLM: Decoding response...")
        response = SLM_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part (after assistant tag)
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()

        print("SLM: Generation complete")
        return response
    except Exception as e:
        print(f"SLM generation error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Try to import scapy for PCAP reading
try:
    from scapy.all import rdpcap
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("WARNING: scapy not available. Install with: pip install scapy")

# Global state
progress_status = {"status": "Idle"}
uploaded_filename = None
file_uploaded_event = Event()
current_detection_mode = "DMRS-Based (Standard)"  # Track current mode for plot display
genReport = None
prompt = ""
interf = 0
layer_interf_start = [0, 0, 0, 0]
layer_interf_end = [272, 272, 272, 272]  # Default to full range (no interference)
layer_has_interf = [False, False, False, False]  # Per-layer interference flag
evma_db = [0, 0, 0, 0]

# AI-based detection results (used in "Both" mode)
ai_interf = 0
ai_interf_start = [0, 0, 0, 0]
ai_interf_end = [272, 272, 272, 272]
ai_has_interf = [False, False, False, False]
ai_evm_results_global = [0, 0, 0, 0]

# Frame parameters (defaults for 100MHz, 30kHz SCS)
numSlots = 20
numSymbols = 14
numLayers = 4
numREs = 3276
numPRBs = 273
#N_ID = 100
#nSCID = 0

# Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)


# -----------------------------------------------------------------------------
# BFP9 DECODING FUNCTIONS (from pcap_utils.py)
# -----------------------------------------------------------------------------

def decode_bfp9_payload(payload: bytes, start_offset: int = 0, prb_count: int = 273):
    """
    Decode BFP9 (Block Floating Point 9-bit) compressed IQ data.
    """
    iq_samples = []
    i = start_offset
    prb_size = 28  # 1 byte header + 27 bytes data

    for prb_idx in range(prb_count):
        if i + prb_size > len(payload):
            break

        exponent = payload[i] & 0x0F
        scale = 2 ** (-exponent)
        i += 1

        prb_data = payload[i:i+27]
        if len(prb_data) < 27:
            break

        bits = int.from_bytes(prb_data, byteorder='big')

        for k in range(12):
            shift = (11 - k) * 18
            sample_bits = (bits >> shift) & 0x3FFFF

            i_part = (sample_bits >> 9) & 0x1FF
            q_part = sample_bits & 0x1FF

            if i_part >= 256:
                i_part -= 512
            if q_part >= 256:
                q_part -= 512

            iq = complex(i_part * scale, q_part * scale)
            iq_samples.append(iq)

        i += 27

    return iq_samples


def extract_oran_uplane(pcap_file, start_offset=0, mav=0):
    """Extract O-RAN U-plane data from PCAP file using scapy."""
    if not SCAPY_AVAILABLE:
        raise ImportError("scapy is required for PCAP parsing")

    pkts = rdpcap(pcap_file)
    results = []

    for pkt in pkts:
        try:
            raw = bytes(pkt["Raw"])
            ecpri_hdr_len = 0 if mav else 4
            header_offset = ecpri_hdr_len

            frame_id = raw[header_offset + 5]
            subframe_id = raw[header_offset + 6] >> 4
            slot_id = raw[header_offset + 7] >> 6
            symbol_id = raw[header_offset + 7] & 0x0F
            port_id = raw[header_offset + 1]

            payload = raw[header_offset + 12:]
            iq_samples = decode_bfp9_payload(payload, start_offset)

            results.append({
                "frame_id": frame_id,
                "port_id": port_id,
                "subframe": subframe_id,
                "slot": slot_id,
                "symbol": symbol_id,
                "iq": iq_samples
            })

        except Exception as e:
            print(f"[WARN] Skipping packet: {e}")

    return results


def pcap_to_csv(pcap_file, output_csv, mav=0):
    """Convert PCAP file to CSV with IQ data."""
    parsed = extract_oran_uplane(pcap_file, mav=mav)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Port", "Subframe", "Slot", "Symbol", "IQ_Index", "Real", "Imag"])
        for entry in parsed:
            for i, iq in enumerate(entry["iq"]):
                writer.writerow([
                    entry["port_id"], entry["subframe"], entry["slot"],
                    entry["symbol"], i, iq.real, iq.imag
                ])

    return output_csv


# -----------------------------------------------------------------------------
# CSV LOADING FUNCTIONS (matching packet_oran_analysis_det_st.py)
# -----------------------------------------------------------------------------

def load_frame_from_csv(csv_path, num_slots=20, num_symbols=14, num_layers=4, num_res=3276):
    """
    Load frame from CSV file in the format used by packet_oran_analysis_det_st.py.
    CSV format: 2 columns (I, Q) per row, no header.
    Order: slot -> symbol -> layer -> RE
    """
    iq_data = np.loadtxt(csv_path, delimiter=',')

    frame = np.zeros((num_slots, num_symbols, num_layers, num_res), dtype=np.complex64)

    row_idx = 0
    for slot in range(num_slots):
        for symbol in range(num_symbols):
            for layer in range(num_layers):
                for re in range(num_res):
                    if row_idx < len(iq_data):
                        i_val = iq_data[row_idx, 0]
                        q_val = iq_data[row_idx, 1]
                        frame[slot, symbol, layer, re] = i_val + 1j * q_val
                        row_idx += 1

    return frame


def build_frame_from_pcap_csv(csv_file, target_subframe, target_slot, num_slots=20, num_symbols=14, num_layers=4, num_res=3276):
    """
    Build rx_frame from PCAP-converted CSV for a specific subframe/slot.
    """
    df = pd.read_csv(csv_file)

    # Filter for target subframe and slot
    df_filtered = df[(df['Subframe'] == target_subframe) & (df['Slot'] == target_slot)]

    if df_filtered.empty:
        print(f"Warning: No data for subframe={target_subframe}, slot={target_slot}")
        return None

    # Initialize frame
    frame = np.zeros((num_slots, num_symbols, num_layers, num_res), dtype=np.complex64)

    # Map slot index
    slot_idx = target_subframe * 2 + target_slot  # For 30kHz SCS
    if slot_idx >= num_slots:
        slot_idx = slot_idx % num_slots

    # Fill frame
    for _, row in df_filtered.iterrows():
        port = int(row['Port']) % num_layers
        symbol = int(row['Symbol']) % num_symbols
        iq_idx = int(row['IQ_Index']) % num_res
        iq_val = complex(row['Real'], row['Imag'])

        frame[slot_idx, symbol, port, iq_idx] = iq_val

    return frame, slot_idx


# -----------------------------------------------------------------------------
# DMRS GENERATION (matching packet_oran_analysis_det_st.py exactly)
# -----------------------------------------------------------------------------

def generate_dmrs_type1_standard(N_ID, nSCID, n, l, numREs, layer):
    """Generate DMRS Type 1 sequence for 5G NR (3GPP TS 38.211 Gold sequence)."""
    assert layer in [0, 1, 2, 3], "Only 4 layers supported"
    ref_layer = layer if layer in [0, 2] else layer - 1
    lam = 0 if ref_layer in [0, 1] else 1
    M = 2 * numREs
    Nc = 1600
    c_init = int((2**17 * (n//2 + 1) * (2*N_ID + 1)) + 2*nSCID + l + (2**14 * lam)) % (2**31)

    # Gold sequence generation per 3GPP TS 38.211
    # x1 sequence - fixed initialization [1, 0, 0, ..., 0]
    x1 = np.zeros(Nc + M + 31, dtype=int)
    x1[0] = 1
    for i in range(31, Nc + M + 31):
        x1[i] = (x1[i-28] + x1[i-31]) % 2

    # x2 sequence - initialized from c_init (this is where N_ID affects the sequence)
    x2 = np.zeros(Nc + M + 31, dtype=int)
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for i in range(31, Nc + M + 31):
        x2[i] = (x2[i-28] + x2[i-31]) % 2

    # Gold sequence: c(n) = (x1(n+Nc) + x2(n+Nc)) mod 2
    c = np.zeros(M, dtype=int)
    for i in range(M):
        c[i] = (x1[i + Nc] + x2[i + Nc]) % 2

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


# -----------------------------------------------------------------------------
# EQUALIZATION (matching packet_oran_analysis_det_st.py exactly)
# -----------------------------------------------------------------------------

def equalize_frame_mimo_4x4(rx_frame, N_ID, nSCID, numREs=3276, dmrs_symbols=[2, 11], start_slot=0):
    """
    Per-layer SISO equalization using layer-specific DMRS positions.

    This isolates layers 0,1 (EVEN DMRS subcarriers) from layers 2,3 (ODD DMRS subcarriers)
    so that interference on one pair doesn't affect the other.

    Each layer estimates its own scalar channel using only its DMRS positions:
    - Layers 0,1: Channel estimated using EVEN subcarrier DMRS
    - Layers 2,3: Channel estimated using ODD subcarrier DMRS
    """
    numSlots, numSymbols, numLayers, _ = rx_frame.shape
    eq_frame = np.zeros_like(rx_frame, dtype=complex)

    # Only process start_slot (other slots have no data and would dilute channel estimate)
    slot = start_slot
    for layer in range(numLayers):
        # Determine DMRS subcarrier positions for this layer
        k_offset = 0 if layer in [0, 1] else 1
        dmrs_indices = np.arange(k_offset, numREs, 2)

        # Estimate channel from DMRS symbols
        H_est = np.zeros(numREs, dtype=complex)
        H_count = np.zeros(numREs)

        for dmrs_sym in dmrs_symbols:
            # Generate reference DMRS for this layer
            ref_dmrs = generate_dmrs_type1_standard(N_ID, nSCID, slot, dmrs_sym, numREs, layer)
            rx_dmrs = rx_frame[slot, dmrs_sym, layer, :]

            # Estimate channel only at DMRS positions (where ref_dmrs is non-zero)
            for k in dmrs_indices:
                if ref_dmrs[k] != 0:
                    H_est[k] += rx_dmrs[k] / ref_dmrs[k]
                    H_count[k] += 1

        # Average channel estimate across DMRS symbols
        for k in dmrs_indices:
            if H_count[k] > 0:
                H_est[k] /= H_count[k]
            else:
                H_est[k] = 1.0

        # Interpolate channel to non-DMRS positions (use nearest DMRS estimate)
        for k in range(numREs):
            if k not in dmrs_indices:
                # Find nearest DMRS position
                if k_offset == 0:  # Layers 0,1 - DMRS at even positions
                    nearest = k - 1 if k % 2 == 1 else k
                else:  # Layers 2,3 - DMRS at odd positions
                    nearest = k + 1 if k % 2 == 0 else k
                if nearest < numREs:
                    H_est[k] = H_est[nearest]
                else:
                    H_est[k] = H_est[k - 1]

        # Equalize all symbols for this layer
        for symbol in range(numSymbols):
            eq_frame[slot, symbol, layer, :] = rx_frame[slot, symbol, layer, :] / (H_est + 1e-12)

    return eq_frame


# -----------------------------------------------------------------------------
# SIGNAL PROCESSING (matching packet_oran_analysis_det_st.py exactly)
# -----------------------------------------------------------------------------

def compute_evm(ref_symbols, rx_symbols):
    """
    Compute Error Vector Magnitude (EVM) in dB.
    EXACT COPY from packet_oran_analysis_det_st.py
    """
    error_vector = rx_symbols - ref_symbols
    evm_rms = np.sqrt(np.mean(np.abs(error_vector)**2))
    ref_rms = np.sqrt(np.mean(np.abs(ref_symbols)**2))
    evm_db = 20 * np.log10(evm_rms / ref_rms)
    return evm_db


def generate_256qam_constellation():
    """
    Generate normalized 256-QAM reference constellation.
    256-QAM has 16x16 = 256 points at positions ±1, ±3, ±5, ..., ±15
    """
    # 256-QAM: 16 levels per axis
    levels = np.arange(-15, 16, 2)  # [-15, -13, -11, ..., 13, 15]
    I, Q = np.meshgrid(levels, levels)
    constellation = I.flatten() + 1j * Q.flatten()

    # Normalize to unit average power
    avg_power = np.mean(np.abs(constellation)**2)
    constellation_normalized = constellation / np.sqrt(avg_power)

    return constellation_normalized


def compute_evm_blind_256qam(rx_symbols):
    """
    Compute EVM using blind 256-QAM detection.
    For each received symbol, find the nearest 256-QAM constellation point
    and calculate the error vector magnitude.

    Args:
        rx_symbols: Equalized received symbols (1D complex array)

    Returns:
        EVM in dB
    """
    # Generate normalized 256-QAM reference
    qam256_ref = generate_256qam_constellation()

    # Normalize received symbols to match 256-QAM power
    rx_power = np.mean(np.abs(rx_symbols)**2)
    rx_normalized = rx_symbols / np.sqrt(rx_power)

    # Find nearest constellation point for each received symbol
    nearest_points = np.zeros_like(rx_normalized)
    for i, rx_sym in enumerate(rx_normalized):
        distances = np.abs(rx_sym - qam256_ref)
        nearest_idx = np.argmin(distances)
        nearest_points[i] = qam256_ref[nearest_idx]

    # Calculate EVM
    error_vector = rx_normalized - nearest_points
    evm_rms = np.sqrt(np.mean(np.abs(error_vector)**2))
    ref_rms = np.sqrt(np.mean(np.abs(nearest_points)**2))
    evm_db = 20 * np.log10(evm_rms / ref_rms + 1e-12)

    return evm_db


# -----------------------------------------------------------------------------
# AI-BASED BLIND INTERFERENCE DETECTION (from gen_full_frame_channel_awgn_ai.py)
# No N_ID or nSCID required - only needs SCS=30kHz, BW=100MHz
# -----------------------------------------------------------------------------

def correct_qpsk_phase_drift(iq_data):
    """
    Estimate and correct constant phase rotation using 4th-power QPSK method.
    DMRS symbols are QPSK modulated, so we use 4th power to remove modulation.
    """
    angle_est = np.angle(np.mean(iq_data ** 4)) / 4
    corrected = iq_data * np.exp(-1j * angle_est)
    return corrected


def gen_QAM_ref(M):
    """
    Generate M-QAM reference constellation (normalized to unit average power).
    For DMRS blind detection, use M=4 (QPSK).
    """
    k = int(np.sqrt(M))
    iq = np.arange(-k + 1, k, 2)
    I, Q = np.meshgrid(iq, iq)
    const_M = I.flatten() + 1j * Q.flatten()
    const_M /= np.sqrt((np.abs(const_M)**2).mean())
    return const_M


def evm_calc_blind(rx_symbol, M):
    """
    Calculate EVM for received symbols against M-QAM constellation.
    Returns linear EVM (not dB).
    """
    const_M = gen_QAM_ref(M)
    min_dist = np.zeros(len(rx_symbol))

    for i in range(len(rx_symbol)):
        distance = np.abs(rx_symbol[i] - const_M.flatten())
        min_dist[i] = (np.min(distance))**2

    evm = np.sqrt(np.sum(min_dist) / len(rx_symbol))
    return evm


def blind_interference_detection(rx_frame, start_slot, numLayers=4, numREs=3276, PRB=12):
    """
    AI-based blind interference detection using QPSK EVM analysis.
    Does NOT require N_ID or nSCID - only processes raw received DMRS symbols.

    Based on gen_full_frame_channel_awgn_ai.py approach:
    - Uses DMRS symbol 2 (QPSK modulated)
    - Layers 0,1 on even REs, layers 2,3 on odd REs
    - Phase correction using 4th-power method
    - EVM calculation against QPSK constellation
    - Interference detected via EVM diff per PRB

    Args:
        rx_frame: Raw received frame (slots, symbols, layers, REs)
        start_slot: Slot to analyze
        numLayers: Number of MIMO layers (default 4)
        numREs: Number of resource elements (default 3276 for 100MHz)
        PRB: PRB size (default 12)

    Returns:
        drop_regions_per_layer: List of (start, end) PRB tuples per layer
        evm_db_PRB: EVM in dB per PRB per layer (shape: numLayers x numPRBs)
    """
    M1 = 4  # QPSK for DMRS
    num_prbs = numREs // PRB

    # Initialize EVM arrays per PRB per layer
    evm_PRB = np.zeros((numLayers, num_prbs))

    # Process each PRB
    for prb_idx in range(num_prbs):
        re_start = prb_idx * PRB
        re_end = re_start + PRB

        # Layers 0,1: Even REs on DMRS symbol 2
        for layer in [0, 1]:
            if layer < numLayers:
                # Get all even REs in this PRB from DMRS symbol 2
                even_res = np.arange(re_start, re_end, 2)
                data = rx_frame[start_slot, 2, layer, even_res]

                if len(data) > 0 and np.any(data != 0):
                    corrected = correct_qpsk_phase_drift(data) * np.exp(-1j * np.pi/4)
                    corrected = correct_qpsk_phase_drift(corrected) * np.exp(-1j * np.pi/4)
                    corrected /= np.sqrt((np.abs(corrected)**2).mean() + 1e-12)
                    evm_PRB[layer, prb_idx] = evm_calc_blind(corrected.flatten(), M1) + 1e-12
                else:
                    evm_PRB[layer, prb_idx] = 1e-12

        # Layers 2,3: Odd REs on DMRS symbol 2
        for layer in [2, 3]:
            if layer < numLayers:
                # Get all odd REs in this PRB from DMRS symbol 2
                odd_res = np.arange(re_start + 1, re_end, 2)
                if len(odd_res) > 0 and odd_res[-1] >= numREs:
                    odd_res = odd_res[:-1]
                if len(odd_res) == 0:
                    evm_PRB[layer, prb_idx] = 1e-12
                    continue
                data = rx_frame[start_slot, 2, layer, odd_res]

                if len(data) > 0 and np.any(data != 0):
                    # Apply double phase correction (same as layers 0,1)
                    corrected = correct_qpsk_phase_drift(data) * np.exp(-1j * np.pi/4)
                    corrected = correct_qpsk_phase_drift(corrected) * np.exp(-1j * np.pi/4)
                    corrected /= np.sqrt((np.abs(corrected)**2).mean() + 1e-12)
                    evm_PRB[layer, prb_idx] = evm_calc_blind(corrected.flatten(), M1) + 1e-12
                else:
                    evm_PRB[layer, prb_idx] = 1e-12

    # Convert to dB and clamp to reasonable range
    evm_db_PRB = 20 * np.log10(evm_PRB + 1e-12)
    evm_db_PRB = np.clip(evm_db_PRB, -40.0, 10.0)  # Clamp EVM to reasonable range

    # Apply smoothing to EVM before diff to reduce single-PRB noise
    from scipy.ndimage import median_filter
    evm_db_smoothed = np.zeros_like(evm_db_PRB)
    for layer in range(numLayers):
        evm_db_smoothed[layer] = median_filter(evm_db_PRB[layer], size=3)

    # Compute diff (inverted to SNR-like interpretation)
    evm_diff_layers = np.diff(evm_db_smoothed, axis=1)
    snr_diff_layers = -evm_diff_layers

    # Remove 1-PRB positive/negative pairs (false detections)
    for layer in range(numLayers):
        for i in range(len(snr_diff_layers[layer]) - 1):
            if (snr_diff_layers[layer, i] > 10.0 and snr_diff_layers[layer, i+1] < -10.0) or \
               (snr_diff_layers[layer, i] < -10.0 and snr_diff_layers[layer, i+1] > 10.0):
                snr_diff_layers[layer, i] = 0.0
                snr_diff_layers[layer, i+1] = 0.0

    # Thresholds for blind detection
    upper_threshold = 10.0
    lower_threshold = -10.0

    # Detect interference regions per layer
    drop_regions_per_layer = []
    for layer in range(numLayers):
        regions = detect_snr_drop_regions(snr_diff_layers[layer], lower_threshold, upper_threshold)
        # Filter out very short regions (1-2 PRBs) as they're likely false detections
        filtered_regions = [(start, end) for start, end in regions if (end - start) >= 3]
        drop_regions_per_layer.append(filtered_regions)

    return drop_regions_per_layer, evm_db_PRB, snr_diff_layers


def compute_snr_per_prb_dmrs(eq_frame, N_ID, nSCID, start_slot, numLayers=4, numREs=3276, PRB=12):
    """
    Compute SNR per PRB using DMRS reference (for when tx_frame is not available).
    Follows the same approach as *_det_new.py but uses DMRS as reference.

    Key: Layers 0,1 use EVEN DMRS subcarriers, Layers 2,3 use ODD DMRS subcarriers.

    Args:
        eq_frame: Equalized frame (slots, symbols, layers, REs)
        N_ID: Cell ID for DMRS generation
        nSCID: Scrambling ID for DMRS generation
        start_slot: Slot to analyze
        numLayers: Number of MIMO layers
        numREs: Number of resource elements
        PRB: PRB size (default 12)

    Returns:
        snr_prb: SNR in dB per PRB per layer (shape: numLayers x num_prbs)
    """
    num_prbs = numREs // PRB
    dmrs_symbols = [2, 11]  # Use both DMRS symbols like *_det_new.py

    # Initialize SNR per PRB per layer
    snr_prb = np.zeros((numLayers, num_prbs))

    for layer in range(numLayers):
        # Determine DMRS subcarrier offset based on layer
        # Layers 0,1: even subcarriers (k_offset=0)
        # Layers 2,3: odd subcarriers (k_offset=1)
        k_offset = 0 if layer in [0, 1] else 1

        for prb_idx in range(num_prbs):
            re_start = prb_idx * PRB
            re_end = re_start + PRB

            # Get DMRS positions within this PRB
            dmrs_positions = np.arange(re_start + k_offset, re_end, 2)

            signal_power = 0.0
            noise_power = 0.0
            num_samples = 0

            for dmrs_sym in dmrs_symbols:
                # Generate expected DMRS reference
                ref_dmrs_full = generate_dmrs_type1_standard(N_ID, nSCID, start_slot, dmrs_sym, numREs, layer)
                ref_vals = ref_dmrs_full[dmrs_positions]

                # Get equalized DMRS
                eq_vals = eq_frame[start_slot, dmrs_sym, layer, dmrs_positions]

                # Only use non-zero reference positions
                mask = ref_vals != 0
                if np.any(mask):
                    signal_power += np.sum(np.abs(ref_vals[mask])**2)
                    noise_power += np.sum(np.abs(eq_vals[mask] - ref_vals[mask])**2)
                    num_samples += np.sum(mask)

            # Compute SNR for this PRB
            if num_samples > 0 and noise_power > 0:
                avg_signal = signal_power / num_samples
                avg_noise = noise_power / num_samples
                snr_linear = avg_signal / (avg_noise + 1e-12)
                snr_db = 10 * np.log10(snr_linear + 1e-12)
                # Clamp SNR to reasonable range to avoid -120 dB outliers
                snr_prb[layer, prb_idx] = np.clip(snr_db, -40.0, 60.0)
            else:
                snr_prb[layer, prb_idx] = 40.0  # High SNR if no noise (good signal)

    return snr_prb


def average_snr_per_subcarrier(rx_frame, tx_frame, start_slot, dmrs_symbols=[2, 11]):
    """
    Calculate average SNR (in dB) per subcarrier for each layer,
    averaged across all slots and DMRS symbols.
    EXACT COPY from packet_oran_analysis_det_st.py
    """
    numSlots, numSymbols, numLayers, numREs = rx_frame.shape
    signal_power = np.zeros((numLayers, numREs))
    noise_power = np.zeros((numLayers, numREs))
    num_samples = 0

    for slot in range(numSlots):
        for symbol in dmrs_symbols:
            for layer in range(numLayers):
                x = tx_frame[slot, symbol, layer, :]
                y = rx_frame[slot, symbol, layer, :]
                signal_power[layer, :] += np.abs(x)**2 + 1e-12
                noise_power[layer, :] += np.abs(y - x)**2

            num_samples += 1

    signal_power /= num_samples
    noise_power /= num_samples

    with np.errstate(divide='ignore', invalid='ignore'):
        snr_linear = np.where(noise_power > 0, signal_power / noise_power, np.inf)
        snr_dB = 10 * np.log10(snr_linear)

    return snr_dB


def compute_snr_per_subcarrier_dmrs(eq_frame, N_ID, nSCID, start_slot, num_layers=4, numREs=3276, dmrs_symbols=[2, 11]):
    """
    Calculate SNR per subcarrier using DMRS symbols 2 and 11.
    Compares equalized DMRS against generated reference DMRS.
    This is the EXACT approach from packet_oran_analysis_det_st.py.

    Args:
        eq_frame: Equalized frame (slots, symbols, layers, REs)
        N_ID: Cell ID for DMRS generation
        nSCID: Scrambling ID for DMRS generation
        start_slot: Slot to analyze
        num_layers: Number of MIMO layers
        numREs: Number of resource elements
        dmrs_symbols: DMRS symbol indices [2, 11]

    Returns:
        snr_dB: Array of shape (num_layers, numREs) with SNR in dB
    """
    numSlots = eq_frame.shape[0]
    signal_power = np.zeros((num_layers, numREs))
    noise_power = np.zeros((num_layers, numREs))
    num_samples = 0

    for slot in range(numSlots):
        for symbol in dmrs_symbols:
            for layer in range(num_layers):
                # Generate expected DMRS (reference)
                ref_dmrs = generate_dmrs_type1_standard(N_ID, nSCID, slot, symbol, numREs, layer)
                # Get equalized received DMRS
                rx_dmrs = eq_frame[slot, symbol, layer, :]

                # Calculate signal and noise power
                signal_power[layer, :] += np.abs(ref_dmrs)**2 + 1e-12
                noise_power[layer, :] += np.abs(rx_dmrs - ref_dmrs)**2

            num_samples += 1

    signal_power /= num_samples
    noise_power /= num_samples

    with np.errstate(divide='ignore', invalid='ignore'):
        snr_linear = np.where(noise_power > 0, signal_power / noise_power, np.inf)
        snr_dB = 10 * np.log10(snr_linear)

    return snr_dB


def moving_average_layers(snr_map, window_size):
    """Apply moving average filter."""
    return uniform_filter1d(snr_map, size=window_size, axis=1, mode='nearest')


def average_snr_per_prb(snr_vector, prb_size=12):
    """Average SNR per PRB."""
    num_prbs = len(snr_vector) // prb_size
    snr_prbs = np.zeros(num_prbs)
    for prb in range(num_prbs):
        start = prb * prb_size
        end = start + prb_size
        snr_prbs[prb] = np.nanmean(snr_vector[start:end])
    return snr_prbs


def detect_snr_drop_regions(diff_vector, lower_thresh=-3.0, upper_thresh=3.0):
    """
    Returns a list of (start_idx, end_idx) regions where ΔSNR dropped below lower_thresh
    and recovered after rising above upper_thresh. If not recovered, extend to end.
    If it starts below threshold without prior rise, begin at 0.
    EXACT COPY from packet_oran_analysis_det_st.py
    """
    drop_regions = []
    in_drop = False
    start_idx = 0

    for i, val in enumerate(diff_vector):
        if not in_drop and val < lower_thresh:
            # start of drop
            in_drop = True
            start_idx = i
        elif in_drop and val > upper_thresh:
            # recovery from drop
            drop_regions.append((start_idx, i))
            in_drop = False

    # Handle drop continuing to end
    if in_drop:
        drop_regions.append((start_idx, len(diff_vector)))

    return drop_regions


# -----------------------------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# -----------------------------------------------------------------------------

def analyze_capture(rx_frame, tx_frame, start_slot, N_ID_val, nSCID_val, num_layers=4, num_res=3276, detection_mode="DMRS-Based (Standard)", has_tx_reference=True):
    """
    Main analysis function supporting different detection modes:
    - "DMRS-Based (Standard)": Uses N_ID/nSCID for equalization and SNR-based detection
    - "AI-Based Blind Detection": Uses QPSK EVM analysis without N_ID/nSCID
    - "Both": Runs both methods, returns AI-based results

    Args:
        has_tx_reference: If False, use DMRS-based SNR/EVM calculation (no tx_frame_iq.csv)
    """
    global interf, layer_interf_start, layer_interf_end, layer_has_interf, evma_db, numLayers
    global ai_interf_start, ai_interf_end, ai_has_interf, ai_interf, ai_evm_results_global

    # Calculate number of PRBs from REs (12 REs per PRB)
    num_prbs = num_res // 12  # 273 for 100MHz
    max_prb = num_prbs - 1    # 272 for 100MHz (0-indexed)

    layer_interf_start = [0, 0, 0, 0]
    layer_interf_end = [max_prb, max_prb, max_prb, max_prb]  # Full range = no interference
    layer_has_interf = [False, False, False, False]
    # Initialize AI interference results (will be set in AI and Both modes)
    ai_interf_start = [0, 0, 0, 0]
    ai_interf_end = [max_prb, max_prb, max_prb, max_prb]
    ai_has_interf = [False, False, False, False]
    ai_interf = 0
    ai_evm_results_global = [0.0] * num_layers
    interf = 0
    numLayers = num_layers
    eq_frame_mimo = None
    evm_results = [0.0] * num_layers

    print(f"\n{'='*60}")
    print(f"Detection Mode: {detection_mode}")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # AI-BASED BLIND INTERFERENCE DETECTION
    # Does NOT require N_ID or nSCID - only SCS=30kHz, BW=100MHz
    # -------------------------------------------------------------------------
    if detection_mode in ["AI-Based Blind Detection", "Both"]:
        print("\n=== AI-Based Blind Interference Detection ===")
        print("(No N_ID/nSCID required - using QPSK EVM analysis on DMRS symbol 2)")

        try:
            # IMPORTANT: Use rx_frame (received frame), NOT tx_frame
            # The blind detection analyzes the raw received DMRS symbols
            # It does NOT need any reference - just phase-corrects and calculates EVM
            blind_regions, evm_db_PRB, blind_snr_diff = blind_interference_detection(
                rx_frame, start_slot, numLayers, num_res
            )

            # Plot blind detection results (plot3.png for Both mode, plot2.png for AI-only)
            plt.figure(figsize=(12, 6))
            blind_upper_threshold = 10.0
            blind_lower_threshold = -10.0
            blind_jumps = (blind_snr_diff > blind_upper_threshold) | (blind_snr_diff < blind_lower_threshold)

            for layer in range(numLayers):
                plt.subplot(2, 2, layer + 1)
                plt.plot(blind_snr_diff[layer], label='Interference Level (dB)')
                plt.plot(np.where(blind_jumps[layer])[0], blind_snr_diff[layer][blind_jumps[layer]], 'ro', label='Interference Edge Detected')
                plt.axhline(blind_upper_threshold, color='gray', linestyle='--', linewidth=0.8)
                plt.axhline(blind_lower_threshold, color='gray', linestyle='--', linewidth=0.8)
                plt.title(f'Layer {layer} - AI-Based Blind Detection')
                plt.xlabel('PRB Index')
                plt.ylabel('Interference Level')
                plt.grid(True)
                plt.legend(loc='lower left', fontsize=8)

            plt.tight_layout()
            # Save as plot2.png for AI-only mode, plot4.png for Both mode
            if detection_mode == "AI-Based Blind Detection":
                plt.savefig("plot2.png")
            else:
                plt.savefig("plot4.png")  # AI interference detection for Both mode
            plt.close()

            # Save EVM per PRB per layer to CSV (AI-based blind detection)
            print("Saving EVM per PRB per layer to evm_per_prb.csv...")
            evm_df = pd.DataFrame(
                evm_db_PRB.T,  # Transpose so rows=PRBs, cols=Layers
                columns=[f'Layer{i}_EVM_dB' for i in range(numLayers)]
            )
            evm_df.index.name = 'PRB'
            evm_df.to_csv('evm_per_prb.csv', float_format='%.2f')
            print(f"Saved EVM per PRB to evm_per_prb.csv ({evm_db_PRB.shape[1]} PRBs x {numLayers} layers)")

            # Save SNR diff (interference level) per PRB per layer to CSV
            print("Saving SNR diff per PRB per layer to snr_diff_per_prb.csv...")
            snr_diff_df = pd.DataFrame(
                blind_snr_diff.T,  # Transpose so rows=PRBs, cols=Layers
                columns=[f'Layer{i}_SNR_Diff_dB' for i in range(numLayers)]
            )
            snr_diff_df.index.name = 'PRB'
            snr_diff_df.to_csv('snr_diff_per_prb.csv', float_format='%.2f')
            print(f"Saved SNR diff per PRB to snr_diff_per_prb.csv ({blind_snr_diff.shape[1]} PRBs x {numLayers} layers)")

            # Extract EVM results from blind detection (RMS average per layer)
            # Correct method: convert dB to linear power, average, convert back to dB
            # EVM_total_dB = 10 * log10(mean(10^(EVM_dB/10)))
            for layer in range(numLayers):
                layer_evm_db = evm_db_PRB[layer]
                evm_linear_power = 10 ** (layer_evm_db / 10)
                mean_power = np.mean(evm_linear_power)
                evm_results[layer] = float(10 * np.log10(mean_power + 1e-12))
            evma_db = evm_results
            print(f"AI-Based EVM results (from per-PRB): {evm_results}")

            # If AI-Based mode (not "Both"), use blind detection results for interference
            if detection_mode == "AI-Based Blind Detection":
                for layer in range(numLayers):
                    print(f"Layer {layer} - AI-Based Blind Detection Regions:")
                    regions = blind_regions[layer]
                    if not regions:
                        print("  No Interference Detected.")
                        # Keep defaults: start=0, end=max_prb (full clean range)
                        layer_has_interf[layer] = False
                    else:
                        for start, end in regions:
                            print(f"  Interference Detected from PRB {start+1} to {end}")
                            layer_interf_start[layer] = start
                            layer_interf_end[layer] = end
                            layer_has_interf[layer] = True
                            interf = 1
                    print()

                # Create constellation plot for AI mode (plot1.png)
                # Apply phase correction and gain normalization (matching gen_full_frame_channel_awgn_ai.py)
                plt.figure(figsize=(12, 6))
                for layer in range(numLayers):
                    plt.subplot(2, 2, layer + 1)

                    # Layers 0,1 use even REs, layers 2,3 use odd REs
                    if layer in [0, 1]:
                        dmrs_data = rx_frame[start_slot, 2, layer, ::2]  # Even REs
                    else:
                        dmrs_data = rx_frame[start_slot, 2, layer, 1::2]  # Odd REs

                    # Apply phase correction twice (as in gen_full_frame_channel_awgn_ai.py)
                    corrected = correct_qpsk_phase_drift(dmrs_data) * np.exp(-1j * np.pi/4)
                    corrected = correct_qpsk_phase_drift(corrected) * np.exp(-1j * np.pi/4)

                    # Normalize gain to unit power
                    corrected /= np.sqrt((np.abs(corrected)**2).mean() + 1e-12)

                    plt.plot(np.real(corrected), np.imag(corrected), '.', alpha=0.5)
                    plt.title(f'Phase-Corrected DMRS - Layer {layer} - EVM: {evm_results[layer]:.2f} dB')
                    plt.xlabel('I')
                    plt.ylabel('Q')
                    plt.grid(True)
                    plt.axis('equal')
                    plt.xlim(-2, 2)
                    plt.ylim(-2, 2)
                plt.tight_layout()
                plt.savefig("plot1.png")
                plt.close()

                # For AI-only mode, set AI interference variables (same as layer_interf)
                ai_interf_start = layer_interf_start.copy()
                ai_interf_end = layer_interf_end.copy()
                ai_has_interf = layer_has_interf.copy()

                # Return early for AI-only mode (no DMRS results)
                return None, evm_results, evm_results, None

            # For "Both" mode, store AI detection results separately
            ai_interf_start = [0, 0, 0, 0]
            ai_interf_end = [max_prb, max_prb, max_prb, max_prb]
            ai_has_interf = [False, False, False, False]
            ai_interf = 0
            for layer in range(numLayers):
                print(f"Layer {layer} - AI-Based Blind Detection Regions:")
                regions = blind_regions[layer]
                if not regions:
                    print("  No Interference Detected.")
                    # Keep defaults: start=0, end=max_prb (full clean range)
                else:
                    for start, end in regions:
                        print(f"  Interference Detected from PRB {start+1} to {end}")
                        ai_interf_start[layer] = start
                        ai_interf_end[layer] = end
                        ai_has_interf[layer] = True
                        ai_interf = 1
                print()

            # Store AI-based EVM results for "Both" mode (before DMRS overwrites)
            ai_evm_results = evm_results.copy()
            ai_evm_results_global[:] = evm_results  # Store in global for report generation
            print(f"AI-Based EVM results stored: {ai_evm_results}")

            # Create AI constellation plot for Both mode (plot3.png)
            # This matches plot1.png from AI-only mode
            plt.figure(figsize=(12, 6))
            for layer in range(numLayers):
                plt.subplot(2, 2, layer + 1)
                # Layers 0,1 use even REs, layers 2,3 use odd REs
                if layer in [0, 1]:
                    dmrs_data = rx_frame[start_slot, 2, layer, ::2]
                else:
                    dmrs_data = rx_frame[start_slot, 2, layer, 1::2]
                # Apply phase correction (same as AI-only mode)
                corrected = correct_qpsk_phase_drift(dmrs_data) * np.exp(-1j * np.pi/4)
                corrected = correct_qpsk_phase_drift(corrected) * np.exp(-1j * np.pi/4)
                corrected /= np.sqrt((np.abs(corrected)**2).mean() + 1e-12)
                plt.plot(np.real(corrected), np.imag(corrected), '.', alpha=0.5)
                plt.title(f'Phase-Corrected DMRS - Layer {layer} - EVM: {ai_evm_results[layer]:.2f} dB')
                plt.xlabel('I')
                plt.ylabel('Q')
                plt.grid(True)
                plt.axis('equal')
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)
            plt.tight_layout()
            plt.savefig("plot3.png")  # AI constellation for Both mode
            plt.close()

        except Exception as e:
            print(f"Blind detection error: {e}")
            import traceback
            traceback.print_exc()
            ai_evm_results = [0.0] * numLayers
            ai_interf_start = [0, 0, 0, 0]
            ai_interf_end = [max_prb, max_prb, max_prb, max_prb]
            ai_has_interf = [False, False, False, False]
            ai_interf = 0
            ai_evm_results_global[:] = [0.0] * numLayers

    # Initialize AI results if not in AI mode
    if detection_mode == "DMRS-Based (Standard)":
        ai_evm_results = None
        ai_interf_start = [0, 0, 0, 0]
        ai_interf_end = [max_prb, max_prb, max_prb, max_prb]
        ai_has_interf = [False, False, False, False]
        ai_interf = 0
        ai_evm_results_global[:] = [0.0] * numLayers

    # -------------------------------------------------------------------------
    # DMRS-BASED INTERFERENCE DETECTION
    # Requires N_ID and nSCID for equalization
    # -------------------------------------------------------------------------
    if detection_mode in ["DMRS-Based (Standard)", "Both"]:
        print("\n=== DMRS-Based Interference Detection ===")
        print(f"(Using N_ID={N_ID_val}, nSCID={nSCID_val})")
        print(f"Reference CSV available: {has_tx_reference}")

        # Frame selection logic
        if has_tx_reference:
            # Original logic when CSV files are available
            if (start_slot >= 4) and (start_slot <= 5):
                rx_framec1 = rx_frame
                print(f"Using rx_frame for equalization (start_slot={start_slot})")
            else:
                rx_framec1 = tx_frame
                print(f"Using tx_frame for equalization (start_slot={start_slot})")
        else:
            # No CSV reference - always use rx_frame (from PCAP)
            rx_framec1 = rx_frame
            print(f"No CSV reference - using rx_frame from PCAP for equalization")

        print(f"Equalizing frame with N_ID={N_ID_val}, nSCID={nSCID_val}, start_slot={start_slot}")
        eq_frame_mimo = equalize_frame_mimo_4x4(rx_framec1, N_ID_val, nSCID_val, numREs=num_res, start_slot=start_slot)

        # Save SNR per PRB per layer to CSV (right after equalization)
        # Uses DMRS symbol 2 for SNR calculation
        print("Saving SNR per PRB per layer to snr_per_prb.csv...")
        snr_prb_for_csv = compute_snr_per_prb_dmrs(
            eq_frame_mimo, N_ID_val, nSCID_val, start_slot, numLayers, num_res
        )
        # Format: rows = PRBs (0-272), columns = Layer0, Layer1, Layer2, Layer3
        snr_df = pd.DataFrame(
            snr_prb_for_csv.T,  # Transpose so rows=PRBs, cols=Layers
            columns=[f'Layer{i}_SNR_dB' for i in range(numLayers)]
        )
        snr_df.index.name = 'PRB'
        snr_df.to_csv('snr_per_prb.csv', float_format='%.2f')
        print(f"Saved SNR per PRB to snr_per_prb.csv (273 PRBs x {numLayers} layers)")

        # Calculate EVM for each layer
        evm_results = []
        if has_tx_reference:
            # Original EVM calculation using TX reference
            for layer in range(numLayers):
                tx_syms = tx_frame[start_slot, :1, layer, :num_res].flatten()
                rx_syms = eq_frame_mimo[start_slot, :1, layer, :num_res].flatten()
                mask = tx_syms != 0  # Only use non-zero REs (ignore DMRS-zeroed REs)
                evm = compute_evm(tx_syms[mask], rx_syms[mask])
                evm_results.append(evm)
            print(f"EVM calculated using TX reference CSV")
        else:
            # DMRS-based EVM calculation (no TX reference)
            # Compare equalized DMRS against generated DMRS reference
            dmrs_symbols = [2, 11]
            for layer in range(numLayers):
                evm_per_dmrs = []
                for dmrs_sym in dmrs_symbols:
                    # Generate expected DMRS
                    ref_dmrs = generate_dmrs_type1_standard(N_ID_val, nSCID_val, start_slot, dmrs_sym, num_res, layer)
                    # Get equalized DMRS
                    eq_dmrs = eq_frame_mimo[start_slot, dmrs_sym, layer, :]
                    # Only use non-zero DMRS positions
                    mask = ref_dmrs != 0
                    if np.any(mask):
                        evm = compute_evm(ref_dmrs[mask], eq_dmrs[mask])
                        evm_per_dmrs.append(evm)
                # Average EVM across DMRS symbols
                evm_results.append(np.mean(evm_per_dmrs) if evm_per_dmrs else 0.0)
            print(f"EVM calculated using DMRS reference (no TX CSV)")

        evma_db = evm_results
        dmrs_evm_results = evm_results.copy()
        print(f"DMRS-Based EVM results: {dmrs_evm_results}")

        # Visualize constellation per layer
        plt.figure(figsize=(12, 6))
        for layer in range(numLayers):
            data_symbols = eq_frame_mimo[start_slot, :, layer, :].flatten()
            plt.subplot(2, 2, layer + 1)
            plt.plot(np.real(data_symbols[:3276]), np.imag(data_symbols[:3276]), '.', alpha=0.5)
            plt.title(f'4x4 Equalized Constellation - Layer {layer} - EVM: {evm_results[layer]:.2f} dB')
            plt.xlabel('I')
            plt.ylabel('Q')
            plt.grid(True)
        plt.tight_layout()
        plt.savefig("plot1.png")
        plt.close()

        # Save equalized symbols to CSV
        iq_array = np.column_stack((
            np.real(data_symbols).astype(np.float32),
            np.imag(data_symbols).astype(np.float32)
        ))
        pd.DataFrame(iq_array).to_csv('data_symbols.csv', index=False, header=False, float_format='%.7f')

        # Save TX reference symbols to CSV (only if TX reference is available)
        if has_tx_reference:
            tx_data_symbols = tx_frame[start_slot, :, layer, :].flatten()
            iq_array = np.column_stack((
                np.real(tx_data_symbols).astype(np.float32),
                np.imag(tx_data_symbols).astype(np.float32)
            ))
            pd.DataFrame(iq_array).to_csv('tx_data_symbols.csv', index=False, header=False, float_format='%.7f')

        # Interference detection
        if has_tx_reference:
            # Original: Uses tx_frame as reference (comparing equalized RX vs TX)
            print("Computing SNR per subcarrier (eq_frame vs tx_frame)...")
            snr_avg = average_snr_per_subcarrier(eq_frame_mimo, tx_frame, start_slot)

            print(f"SNR Layer 0 sample: {snr_avg[0, :10]}")

            # Apply smoothing
            smoothed_snr = moving_average_layers(snr_avg, window_size=96)
            print(f"Smoothed SNR Layer 0 sample: {smoothed_snr[0, :10]}")

            # Compute average SNR per PRB for each layer
            snr_prb_all_layers = np.array([average_snr_per_prb(snr_avg[layer]) for layer in range(numLayers)])

            # Replace inf/nan values with interpolated values to avoid discontinuities
            for layer in range(numLayers):
                snr_layer = snr_prb_all_layers[layer]
                valid_mask = np.isfinite(snr_layer)
                if not np.all(valid_mask) and np.any(valid_mask):
                    valid_indices = np.where(valid_mask)[0]
                    invalid_indices = np.where(~valid_mask)[0]
                    snr_layer[invalid_indices] = np.interp(invalid_indices, valid_indices, snr_layer[valid_indices])
                elif not np.any(valid_mask):
                    snr_layer[:] = 0
                snr_prb_all_layers[layer] = snr_layer

            # Apply median filter to SNR per PRB to reduce single-PRB noise
            from scipy.ndimage import median_filter
            snr_prb_smoothed = np.zeros_like(snr_prb_all_layers)
            for layer in range(numLayers):
                snr_prb_smoothed[layer] = median_filter(snr_prb_all_layers[layer], size=5)

            # Compute SNR difference between adjacent PRBs
            snr_diff_layers = np.diff(snr_prb_smoothed, axis=1)
            snr_diff_layers = np.nan_to_num(snr_diff_layers, nan=0.0, posinf=0.0, neginf=0.0)

            # Thresholds for SNR-based detection
            upper_threshold = 15.0
            lower_threshold = -15.0

            # Identify jumps and notches (interference edges)
            jumps = (snr_diff_layers > upper_threshold) | (snr_diff_layers < lower_threshold)

            # Plot interference detection
            plt.figure(figsize=(12, 6))
            for layer in range(numLayers):
                plt.subplot(2, 2, layer + 1)
                plt.plot(snr_diff_layers[layer], label='SNR Diff (dB)')
                plt.plot(np.where(jumps[layer])[0], snr_diff_layers[layer][jumps[layer]], 'ro', label='Interference Edge')
                plt.axhline(upper_threshold, color='gray', linestyle='--', linewidth=0.8)
                plt.axhline(lower_threshold, color='gray', linestyle='--', linewidth=0.8)
                plt.title(f'Layer {layer} - DMRS-Based Detection')
                plt.xlabel('PRB Index')
                plt.ylabel('SNR Diff (dB)')
                plt.grid(True)
                plt.legend(loc='lower left', fontsize=8)

            plt.tight_layout()
            plt.savefig("plot2.png")
            plt.close()

            # Apply region detection per layer
            drop_regions_per_layer = []
            for layer in range(numLayers):
                regions = detect_snr_drop_regions(snr_diff_layers[layer], lower_threshold, upper_threshold)
                filtered_regions = [(start, end) for start, end in regions if (end - start) >= 3]
                drop_regions_per_layer.append(filtered_regions)

        else:
            # No TX reference: Use DMRS-based SNR calculation, then follow *_det_new.py approach
            # Layers 0,1 use EVEN DMRS subcarriers, Layers 2,3 use ODD DMRS subcarriers
            print("Using DMRS-based SNR calculation (same approach as *_det_new.py)...")
            print(f"  Layers 0,1: EVEN DMRS subcarriers")
            print(f"  Layers 2,3: ODD DMRS subcarriers")

            # Compute SNR per PRB using DMRS reference (replaces average_snr_per_subcarrier)
            snr_prb_all_layers = compute_snr_per_prb_dmrs(
                eq_frame_mimo, N_ID_val, nSCID_val, start_slot, numLayers, num_res
            )

            print(f"SNR per PRB Layer 0 sample: {snr_prb_all_layers[0, :10]}")

            # Apply median filter to remove single-PRB spikes before computing diff
            from scipy.ndimage import median_filter
            snr_prb_filtered = np.zeros_like(snr_prb_all_layers)
            for layer in range(numLayers):
                snr_prb_filtered[layer] = median_filter(snr_prb_all_layers[layer], size=3)

            # Compute SNR difference between adjacent PRBs
            snr_diff_layers = np.diff(snr_prb_filtered, axis=1)  # shape: (4, 272)

            # Remove 1-PRB positive/negative pairs (false detections)
            # A pair is when diff[i] and diff[i+1] have opposite signs and both exceed threshold
            for layer in range(numLayers):
                for i in range(len(snr_diff_layers[layer]) - 1):
                    if (snr_diff_layers[layer, i] > 15.0 and snr_diff_layers[layer, i+1] < -15.0) or \
                       (snr_diff_layers[layer, i] < -15.0 and snr_diff_layers[layer, i+1] > 15.0):
                        # This is a 1-PRB spike, zero out both diffs
                        snr_diff_layers[layer, i] = 0.0
                        snr_diff_layers[layer, i+1] = 0.0

            # Thresholds
            upper_threshold = 15.0  # dB
            lower_threshold = -15.0  # dB

            # Identify jumps and notches
            jumps = (snr_diff_layers > upper_threshold) | (snr_diff_layers < lower_threshold)

            # Plot interference detection - EXACT from *_det_new.py
            plt.figure(figsize=(12, 6))
            for layer in range(numLayers):
                plt.subplot(2, 2, layer + 1)
                plt.plot(snr_diff_layers[layer], label='Interference Level (dB)')
                plt.plot(np.where(jumps[layer])[0], snr_diff_layers[layer][jumps[layer]], 'ro', label='Interference Edge Detected')
                plt.axhline(upper_threshold, color='gray', linestyle='--', linewidth=0.8)
                plt.axhline(lower_threshold, color='gray', linestyle='--', linewidth=0.8)
                plt.title(f'Layer {layer} - Interference Detection')
                plt.xlabel('PRB Index')
                plt.ylabel('Interference Level')
                plt.grid(True)
                plt.legend(loc='lower left')

            plt.tight_layout()
            plt.savefig("plot2.png")
            plt.close()

            # Apply region detection per layer - EXACT from *_det_new.py
            drop_regions_per_layer = [detect_snr_drop_regions(snr_diff_layers[layer], lower_threshold, upper_threshold) for layer in range(numLayers)]

        # Report interference regions
        for layer in range(numLayers):
            print(f"Layer {layer} - DMRS-Based Detection Regions:")
            regions = drop_regions_per_layer[layer]
            if not regions:
                print("  No Interference Detected.")
                # Keep defaults: start=0, end=max_prb (full clean range)
                layer_has_interf[layer] = False
            else:
                for start, end in regions:
                    print(f"  Interference Detected from PRB {start+1} to {end}")
                    layer_interf_start[layer] = start
                    layer_interf_end[layer] = end
                    layer_has_interf[layer] = True
                    interf = 1
            print()

    # Return with both AI and DMRS EVM results
    # ai_evm_results: from blind detection (None if DMRS-only mode)
    # dmrs_evm_results: from DMRS-based detection
    return eq_frame_mimo, evm_results, ai_evm_results, dmrs_evm_results


# -----------------------------------------------------------------------------
# FLASK ROUTES
# -----------------------------------------------------------------------------

@app.route('/')
def index():
    """Home page with status"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>O-RAN PCAP Analyzer</title>
        <style>
            body { font-family: Arial; padding: 40px; background: #0e1a40; color: white; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 20px; background: #1a2a5a; margin: 20px 0; border-radius: 5px; }
            code { background: #2a3a6a; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>O-RAN PCAP Analyzer</h1>
            <p>Flask server using scapy + BFP9 decoding (matching packet_oran_analysis_det_st.py)</p>
            <div class="status">
                <h3>Server Status: Running</h3>
                <p>Port: 5002</p>
                <p>scapy available: {{ scapy }}</p>
            </div>
            <h3>API Endpoints:</h3>
            <ul>
                <li><code>POST /upload</code> - Submit PCAP file for analysis</li>
                <li><code>GET /progress</code> - Get analysis progress</li>
                <li><code>GET /report</code> - Get generated report</li>
            </ul>
        </div>
    </body>
    </html>
    """, scapy=SCAPY_AVAILABLE)


@app.route('/progress', methods=['GET'])
def get_progress():
    """Return current analysis progress"""
    global progress_status
    return jsonify(progress_status)


@app.route('/upload', methods=['POST'])
def upload():
    """Process PCAP file for analysis"""
    global uploaded_filename, genReport, progress_status, interf
    global layer_interf_start, layer_interf_end, N_ID, nSCID, numREs, numPRBs

    progress_status = {"status": "Processing started..."}

    try:
        data = request.get_json(force=True)
        filepath = data.get("filepath")

        if not filepath or not os.path.exists(filepath):
            progress_status = {"status": "Error: Invalid file path"}
            return jsonify({"error": "Invalid file path"}), 400

        # Path traversal protection - normalize and validate path
        filepath = os.path.normpath(os.path.abspath(filepath))
        if ".." in filepath or not filepath.endswith(('.pcap', '.pcapng')):
            progress_status = {"status": "Error: Invalid file path or extension"}
            return jsonify({"error": "Invalid file path or extension"}), 400

        # File size check to prevent DoS (max 500MB)
        max_file_size = 500 * 1024 * 1024  # 500 MB
        file_size = os.path.getsize(filepath)
        if file_size > max_file_size:
            progress_status = {"status": f"Error: File too large ({file_size / 1024 / 1024:.1f} MB > 500 MB limit)"}
            return jsonify({"error": f"File too large. Max size: 500 MB"}), 400

        if not SCAPY_AVAILABLE:
            progress_status = {"status": "Error: scapy not installed"}
            return jsonify({"error": "scapy required. pip install scapy"}), 500

        # Get parameters (ensure integers for DMRS generation)
        N_ID = int(data.get("N_ID", 100))
        nSCID = int(data.get("nSCID", 0))
        bandwidth = data.get("bandwidth", 100)
        scs = data.get("scs", 30)
        layers = data.get("layers", 4)
        link = data.get("link", "Uplink")
        subframe = data.get("subframe", 0)
        slot_in_subframe = data.get("slot", 0)

        # Model selection for report generation: "OpenAI", "SLM", or list like ["OpenAI", "SLM"]
        model_selection = data.get("model_selection", ["OpenAI"])
        if isinstance(model_selection, str):
            model_selection = [model_selection]

        # Detection mode: "DMRS-Based (Standard)", "AI-Based Blind Detection", or "Both"
        detection_mode = data.get("detection_mode", "DMRS-Based (Standard)")

        global current_detection_mode
        current_detection_mode = detection_mode  # Update global for plot display

        uploaded_filename = filepath
        progress_status = {"status": f"Processing: {os.path.basename(filepath)}"}

        print(f"=" * 60)
        print(f"Analysis Parameters:")
        print(f"  File: {filepath}")
        print(f"  N_ID={N_ID}, nSCID={nSCID}, BW={bandwidth}MHz, SCS={scs}kHz")
        print(f"  Subframe={subframe}, Slot={slot_in_subframe}, Layers={layers}")
        print(f"  Detection Mode: {detection_mode}")
        print(f"  Model Selection: {model_selection}")
        print(f"=" * 60)

        # Determine PRBs and REs
        if scs == 30:
            numPRBs = {5: 14, 10: 27, 15: 41, 20: 54, 50: 135, 100: 273}.get(bandwidth, 273)
        else:
            numPRBs = {5: 28, 10: 55, 15: 83, 20: 110, 50: 275, 100: 550}.get(bandwidth, 275)
        numREs = numPRBs * 12

        # Calculate start_slot and packet range
        slots_per_subframe = 2 if scs == 30 else 1
        start_slot = subframe * slots_per_subframe + slot_in_subframe

        start_packet = start_slot * numSymbols * layers
        end_packet = start_packet + numSymbols * layers
        if link == "Uplink":
            start_packet += 224
            end_packet += 224

        print(f"  start_slot={start_slot} (subframe={subframe} * 2 + slot={slot_in_subframe})")
        print(f"  packet range={start_packet}-{end_packet}")
        print(f"  Frame selection: {'rx_frame' if (start_slot >= 4 and start_slot <= 5) else 'tx_frame'}")

        # Step 1: Convert PCAP to CSV
        progress_status = {"status": "Converting PCAP to CSV (BFP9 decoding)..."}
        flask_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = filepath + ".csv"

        try:
            pcap_to_csv(filepath, csv_file, mav=0)
            print(f"CSV saved to: {csv_file}")
        except Exception as e:
            print(f"PCAP conversion error: {e}")
            progress_status = {"status": f"PCAP conversion error: {str(e)}"}
            return jsonify({"error": f"PCAP conversion failed: {str(e)}"}), 500

        # Step 2: Load reference frames (rx_frame_iq.csv and tx_frame_iq.csv)
        # Both files should be in the same directory as this Flask script
        progress_status = {"status": "Loading reference frames..."}

        rx_csv_path = os.path.join(flask_dir, "rx_frame_iq.csv")
        tx_csv_path = os.path.join(flask_dir, "tx_frame_iq.csv")

        print(f"Flask directory: {flask_dir}")
        print(f"Looking for rx_frame_iq.csv: {rx_csv_path} - {'Found' if os.path.exists(rx_csv_path) else 'Not found'}")
        print(f"Looking for tx_frame_iq.csv: {tx_csv_path} - {'Found' if os.path.exists(tx_csv_path) else 'Not found'}")

        # Track if we have reference CSV files (needed for DSP mode)
        has_tx_reference = os.path.exists(tx_csv_path)
        has_rx_reference = os.path.exists(rx_csv_path)

        try:
            if has_rx_reference:
                print(f"Loading rx_frame from: {rx_csv_path}")
                rx_frame = load_frame_from_csv(rx_csv_path, numSlots, numSymbols, layers, numREs)
            else:
                print(f"rx_frame_iq.csv not found, building from PCAP CSV")
                result = build_frame_from_pcap_csv(csv_file, subframe, slot_in_subframe, numSlots, numSymbols, layers, numREs)
                if result is None:
                    raise ValueError("No data found for specified subframe/slot")
                rx_frame, start_slot = result

            if has_tx_reference:
                print(f"Loading tx_frame from: {tx_csv_path}")
                full_tx_frame = load_frame_from_csv(tx_csv_path, numSlots, numSymbols, layers, numREs)
            else:
                print(f"tx_frame_iq.csv not found, using rx_frame as reference (DMRS-based mode)")
                full_tx_frame = rx_frame.copy()

            # CRITICAL: Match original packet_oran_analysis_det_new.py behavior
            # Only copy start_slot from full frames - this is how the original works!
            # Lines 1113-1128: tx_frame is zeroed, only start_slot is copied from full_frame
            # This ensures SNR calculation only uses the slot of interest
            full_rx_frame = rx_frame.copy()
            rx_frame = np.zeros((numSlots, numSymbols, layers, numREs), dtype=np.complex64)
            tx_frame = np.zeros((numSlots, numSymbols, layers, numREs), dtype=np.complex64)
            rx_frame[start_slot, :, :, :] = full_rx_frame[start_slot, :, :, :]
            tx_frame[start_slot, :, :, :] = full_tx_frame[start_slot, :, :, :]
            print(f"Copied only start_slot={start_slot} from full frames (matching original behavior)")

        except Exception as e:
            print(f"Frame loading error: {e}")
            import traceback
            traceback.print_exc()
            progress_status = {"status": f"Frame loading error: {str(e)}"}
            return jsonify({"error": f"Frame loading failed: {str(e)}"}), 500

        # Step 3: Run analysis
        progress_status = {"status": f"Analyzing slot {start_slot} ({detection_mode})..."}

        # Clean up old plot files to prevent showing stale plots from previous mode
        for old_plot in ["plot1.png", "plot2.png", "plot3.png", "plot4.png"]:
            if os.path.exists(old_plot):
                os.remove(old_plot)
                print(f"Removed old plot file: {old_plot}")

        try:
            eq_frame, evm_results, ai_evm_results, dmrs_evm_results = analyze_capture(
                rx_frame, tx_frame, start_slot, N_ID, nSCID, layers, numREs, detection_mode, has_tx_reference
            )
        except Exception as e:
            print(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
            progress_status = {"status": f"Analysis error: {str(e)}"}
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

        # Step 4: Generate report - matching packet_oran_analysis_det_st.py
        progress_status = {"status": "Generating report..."}
        report_result = generate_report("", os.path.basename(filepath), model_selection)
        genReport = report_result[0]  # First element is the report text

        file_uploaded_event.set()

        if interf == 1:
            progress_status = {"status": f"Analysis complete ({detection_mode}) - Interference detected"}
        else:
            progress_status = {"status": f"Analysis complete ({detection_mode}) - No interference detected"}

        # Convert numpy types for JSON
        evm_results_native = [float(x) for x in evm_results]

        # Prepare AI and DMRS EVM results for response
        ai_evm_native = [float(x) for x in ai_evm_results] if ai_evm_results is not None else None
        dmrs_evm_native = [float(x) for x in dmrs_evm_results] if dmrs_evm_results is not None else None

        # Build layer data with DMRS results (used for DMRS-only and Both modes)
        layers_dict = {}
        for i in range(layers):
            layer_data = {
                "start_prb": int(layer_interf_start[i]),
                "end_prb": int(layer_interf_end[i]),
                "has_interference": layer_has_interf[i],
                "evm_db": float(evm_results[i]) if i < len(evm_results) else 0.0,
                "dmrs_evm_db": float(dmrs_evm_results[i]) if dmrs_evm_results is not None and i < len(dmrs_evm_results) else None,
            }
            # Add AI-specific results for Both mode
            if ai_evm_results is not None:
                layer_data["ai_evm_db"] = float(ai_evm_results[i]) if i < len(ai_evm_results) else None
            if ai_interf_start is not None:
                layer_data["ai_start_prb"] = int(ai_interf_start[i])
                layer_data["ai_end_prb"] = int(ai_interf_end[i])
                layer_data["ai_has_interference"] = ai_has_interf[i]
            layers_dict[f"layer_{i}"] = layer_data

        return jsonify({
            "success": True,
            "message": f"Analysis complete for {os.path.basename(filepath)}",
            "csv_file": csv_file,
            "detection_mode": detection_mode,
            "interference": int(interf),
            "evm_db": evm_results_native,
            "ai_evm_db": ai_evm_native,
            "dmrs_evm_db": dmrs_evm_native,
            "evm_per_prb_csv": "evm_per_prb.csv",
            "snr_per_prb_csv": "snr_per_prb.csv",
            "snr_diff_per_prb_csv": "snr_diff_per_prb.csv",
            "layers": layers_dict
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        progress_status = {"status": f"Error: {str(e)}"}
        return jsonify({"error": str(e)}), 500


@app.route('/report', methods=['GET'])
def get_report():
    """Return generated report"""
    global genReport
    if genReport:
        html = markdown2.markdown(genReport, extras=["tables"])
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>O-RAN Analysis Report</title>
            <style>
                body { font-family: Arial; padding: 40px; background: #f0f8ff; }
                table { border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ccc; padding: 10px; }
                th { background: #e0e8f0; }
            </style>
        </head>
        <body>{{ report | safe }}</body>
        </html>
        """, report=html)
    return jsonify({"error": "No report available"})


@app.route('/plots/<filename>', methods=['GET'])
def get_plot(filename):
    """Return plot as base64. In 'Both' mode, return AI-based plots instead."""
    # In "Both" mode, map plot1/plot2 to AI plots (plot3/plot4)
    actual_filename = filename
    if current_detection_mode == "Both":
        if filename == "plot1.png":
            actual_filename = "plot3.png"  # AI constellation
        elif filename == "plot2.png":
            actual_filename = "plot4.png"  # AI interference detection

    if filename in ["plot1.png", "plot2.png"]:
        if os.path.exists(actual_filename):
            with open(actual_filename, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            return jsonify({"image": encoded})
    return jsonify({"error": "Plot not found"})


def align_markdown_table(md_table: str) -> str:
    """Align markdown table columns - EXACT COPY from packet_oran_analysis_det_st.py"""
    lines = [line.strip() for line in md_table.strip().split("\n")]
    rows = [line.split("|")[1:-1] for line in lines if "|" in line]
    col_widths = [max(len(cell.strip()) for cell in col) for col in zip(*rows)]
    aligned = []
    for i, row in enumerate(rows):
        padded = [" " + cell.strip().ljust(w) + " " for cell, w in zip(row, col_widths)]
        line = "|" + "|".join(padded) + "|"
        if i == 1:  # separator row
            sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
            aligned.append(sep)
        else:
            aligned.append(line)
    return "\n".join(aligned)


def df_to_markdown_table(df):
    """Convert DataFrame to markdown table - EXACT COPY from packet_oran_analysis_det_st.py"""
    # Build header
    headers = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["-" * len(col) for col in df.columns]) + " |"

    # Build each row
    rows = "\n".join(
        "| " + " | ".join(str(cell) for cell in row) + " |" for row in df.values
    )

    return f"{headers}\n{separator}\n{rows}"


def generate_report(data, fname, model_selection=None):
    """Generates a report using OpenAI or SLM based on model_selection"""
    global prompt, progress_status
    global ai_interf, ai_interf_start, ai_interf_end, ai_has_interf, ai_evm_results_global, current_detection_mode

    if model_selection is None:
        model_selection = ["OpenAI"]

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    safe_fname = f"`{fname}`"
    safe_now = f"`{now}`"
    safe_interf = f"`{interf}`"

    if (interf == 0):
        report_header_template = pd.DataFrame({
            "Item"  : ["Filename", "Date", "Status", "Issues"],
            "Value" : [{safe_fname}, {safe_now},  "✅", "No Issues"]
        })
        progress_status = {"status": "Analyzer Successfully Finished the Analysis - No Issues Found"}
    else:
        report_header_template = pd.DataFrame({
            "Item"  : ["Filename", "Date", "Status", "Issues"],
            "Value" : [{safe_fname}, {safe_now},  "❌", "Found Interference"]
        })
        progress_status = {"status": "Analyzer Successfully Finished the Analysis - Interference Found"}

    markdown_header = df_to_markdown_table(report_header_template)

    if (interf == 0):
        link_dir = "Downlink"
        interf_l0 = "None"
        interf_l1 = "None"
        interf_l2 = "None"
        interf_l3 = "None"
    else:
        link_dir = "Uplink"
        if not layer_has_interf[0]:
            interf_l0 = "None"
        else:
            interf_l0 = f"Detected in PRBs {layer_interf_start[0]+1}-{layer_interf_end[0]}"
        if not layer_has_interf[1]:
            interf_l1 = "None"
        else:
            interf_l1 = f"Detected in PRBs {layer_interf_start[1]+1}-{layer_interf_end[1]}"
        if not layer_has_interf[2]:
            interf_l2 = "None"
        else:
            interf_l2 = f"Detected in PRBs {layer_interf_start[2]+1}-{layer_interf_end[2]}"
        if not layer_has_interf[3]:
            interf_l3 = "None"
        else:
            interf_l3 = f"Detected in PRBs {layer_interf_start[3]+1}-{layer_interf_end[3]}"

    scs_str = "30 KHz"

    data_summary_template = """
    | **Variable**                       | **Value**     | **Description**                                             |
    |                                    |               |                                                             |
    | **Sub-carrier spacing (KHz)**      | {scs_str}     | Defines spacing between sub-carriers.                       |
    | **Number Of Antennas**             | 4             | Represents antennas configured in this setup.               |
    | **Max Frames**                     | 1             | Indicates the max number of frames transmitted.             |
    | **DL Direction**                   | {link_dir}    | Specifies the traffic direction.                            |
    | **U-Plane Packet Type**            | U-plane       | Denotes user plane packets carrying user data.              |
    | **Number Of PRBs**                 | 273           | Count of Physical Resource Blocks for transmission.         |
    | **Bandwidth Frequency (MHz)**      | 98.28 MHz     | Total bandwidth allocated for transmission.                 |
    | **Interference**                   | {safe_interf} | Indicates if interference was found in the packet.          |
    | **Interference - L0**              | {interf_l0}   | Indicates if interference was found in layer 0 the packet.  |
    | **Interference - L1**              | {interf_l1}   | Indicates if interference was found in layer 1 the packet.  |
    | **Interference - L2**              | {interf_l2}   | Indicates if interference was found in layer 2 the packet.  |
    | **Interference - L3**              | {interf_l3}   | Indicates if interference was found in layer 3 the packet.  |
    """

    data_summary_filled = data_summary_template.format(safe_interf=safe_interf, link_dir=link_dir, scs_str=scs_str, interf_l0=interf_l0, interf_l1=interf_l1, interf_l2=interf_l2, interf_l3=interf_l3)

    aligned_summary = align_markdown_table(data_summary_filled)

    # Build AI-Based Detection Results section for "Both" mode
    ai_section = ""
    if current_detection_mode == "Both":
        # Format AI interference results
        ai_interf_l0 = "None" if not ai_has_interf[0] else f"Detected in PRBs {ai_interf_start[0]+1}-{ai_interf_end[0]}"
        ai_interf_l1 = "None" if not ai_has_interf[1] else f"Detected in PRBs {ai_interf_start[1]+1}-{ai_interf_end[1]}"
        ai_interf_l2 = "None" if not ai_has_interf[2] else f"Detected in PRBs {ai_interf_start[2]+1}-{ai_interf_end[2]}"
        ai_interf_l3 = "None" if not ai_has_interf[3] else f"Detected in PRBs {ai_interf_start[3]+1}-{ai_interf_end[3]}"

        ai_summary_template = """
    | **AI Detection Variable**          | **Value**     | **Description**                                             |
    |                                    |               |                                                             |
    | **Detection Method**               | Blind QPSK    | AI-based blind detection without N_ID/nSCID parameters.     |
    | **AI Interference Flag**           | `{ai_interf}` | AI-detected interference (1=found, 0=none).                 |
    | **AI Interference - L0**           | {ai_l0}       | AI-detected interference in layer 0.                        |
    | **AI Interference - L1**           | {ai_l1}       | AI-detected interference in layer 1.                        |
    | **AI Interference - L2**           | {ai_l2}       | AI-detected interference in layer 2.                        |
    | **AI Interference - L3**           | {ai_l3}       | AI-detected interference in layer 3.                        |
    | **AI EVM - L0 (dB)**               | {ai_evm0:.2f} | AI-based EVM measurement for layer 0.                       |
    | **AI EVM - L1 (dB)**               | {ai_evm1:.2f} | AI-based EVM measurement for layer 1.                       |
    | **AI EVM - L2 (dB)**               | {ai_evm2:.2f} | AI-based EVM measurement for layer 2.                       |
    | **AI EVM - L3 (dB)**               | {ai_evm3:.2f} | AI-based EVM measurement for layer 3.                       |
    """
        ai_summary_filled = ai_summary_template.format(
            ai_interf=ai_interf,
            ai_l0=ai_interf_l0, ai_l1=ai_interf_l1, ai_l2=ai_interf_l2, ai_l3=ai_interf_l3,
            ai_evm0=ai_evm_results_global[0], ai_evm1=ai_evm_results_global[1],
            ai_evm2=ai_evm_results_global[2], ai_evm3=ai_evm_results_global[3]
        )
        aligned_ai_summary = align_markdown_table(ai_summary_filled)
        ai_section = f"""

    ## AI-Based Blind Detection Results

    {aligned_ai_summary}
    """

    # Determine summary section title based on detection mode
    if current_detection_mode == "AI-Based Blind Detection":
        summary_title = "Fronthaul Data Summary (AI-Based)"
    elif current_detection_mode == "Both":
        summary_title = "Fronthaul Data Summary (DMRS-Based)"
    else:
        summary_title = "Fronthaul Data Summary (DMRS-Based)"

    prompt = f"""
    You are an O-RAN fronthaul packet analyzer following the O-RAN fronthaul specification from www.o-ran.org.
    Generate a professional report summarizing the following structured O-RAN fronthaul data:


    The report provides a high level overview of the contents of the fronthaul data.

    The report will contain a header mentioning the following:
    - The filename based on {fname}
    - Date and time of the report generation using the format {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    - A green checkmark emoji if the fronthaul data has no issues, or a red cross emoji if there are issues.
    - A list of issues if any are found in the fronthaul data or say "None" if no issues are found.
    - Elaborate on each line of the fronthaul data summary table in the detailed analysis
    - Make sure all the lines of the data_summary_template exist starting from the Sub-carrier Spacing {scs_str}
    - Add a conclusion sentence in the end-user

    You have to use the following specific format for the report:

    ## Report Header

    {markdown_header}


    ## {summary_title}

    {aligned_summary}
    {ai_section}

    ## Detailed Analysis
    **1. Sub-carrier Spacing** - 30 KHz reflects granularity, influencing latency and spectral efficiency.

    **2. Number of Antennas** - A value of 2 indicates a dual-antenna configuration, which may enhance signal reliability and increase throughput capabilities compared to single-antenna setups.

    """

    sections = prompt.split("## ")
    header_section = sections[1] if len(sections) >= 1 else ""
    summary_section = sections[2] if len(sections) >= 2 else ""
    analysis_section = sections[3] if len(sections) >= 3 else ""

    # Generate report based on model selection
    report_content = None

    # Try SLM first if selected
    if "SLM" in model_selection:
        print("Generating report using SLM (TinyLlama)...")
        progress_status = {"status": "Generating report using SLM (TinyLlama)..."}

        # Pass condensed analysis data to SLM
        analysis_data = {
            'filename': fname,
            'interference': interf,
            'evm': evma_db[0] if evma_db else 'N/A',
            'layers_affected': sum(1 for i in range(4) if layer_has_interf[i])
        }

        slm_response = generate_with_slm(prompt, max_new_tokens=256, analysis_data=analysis_data)
        if slm_response:
            # Combine SLM response with the formatted template
            report_content = f"{prompt}\n\n## AI Analysis (TinyLlama)\n{slm_response}"
            print("SLM report generation successful")

    # Try OpenAI if selected and SLM didn't produce output
    if report_content is None and "OpenAI" in model_selection:
        if OPENAI_API_KEY:
            print("Generating report using OpenAI...")
            progress_status = {"status": "Generating report using OpenAI..."}
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                report_content = response.choices[0].message.content
                print("OpenAI report generation successful")
            except Exception as e:
                print(f"OpenAI API error: {e}")
        else:
            print("OpenAI selected but API key not available")

    # Fallback: return the formatted prompt as the report
    if report_content is None:
        print("Using fallback report (no AI generation)")
        report_content = prompt

    return (report_content, sections, header_section, summary_section, analysis_section)


# -----------------------------------------------------------------------------
# APP1: Report Display (Port 5000)
# -----------------------------------------------------------------------------

app1 = Flask("Report")

@app1.route("/")
def report_page():
    """Display the analysis report."""
    global genReport

    print("[App1] Waiting for file upload...")
    file_uploaded_event.wait()
    print(f"[App1] Detected uploaded file: {uploaded_filename}")
    print(f"[App1] Current detection mode: {current_detection_mode}")

    if genReport:
        html_report = markdown2.markdown(genReport, extras=["tables"])

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>O-RAN Analysis Report</title>
            <style>
                body { font-family: Arial; padding: 40px; background: #f0f8ff; }
                table, td, th { border: 1px solid #ccc; border-collapse: collapse; padding: 10px; }
                th { background: #e0e8f0; }
            </style>
        </head>
        <body>
            <div class="report-content">
                {{ html_report | safe }}
            </div>
            <div class="footer">
                <p><strong>Prepared by:</strong> Ennoia's AI O-RAN Fronthaul Packet Analyzer</p>
            </div>
        </body>
        </html>
        """
        return render_template_string(html_template, html_report=html_report)
    return "No report available yet"


# -----------------------------------------------------------------------------
# APP2: Plot Display (Port 5001)
# -----------------------------------------------------------------------------

app2 = Flask("Plots")

def create_plot():
    """Create the analysis plots based on detection mode."""
    print("[App2] Waiting for file upload...")
    file_uploaded_event.wait()
    print(f"[App2] Detected uploaded file: {uploaded_filename}")
    print(f"[App2] Current detection mode: {current_detection_mode}")

    try:
        # In "Both" mode, display AI plots (plot3.png and plot4.png)
        if current_detection_mode == "Both":
            if os.path.exists("plot3.png") and os.path.exists("plot4.png"):
                from PIL import Image
                img1 = Image.open("plot3.png")
                img2 = Image.open("plot4.png")
                total_width = max(img1.width, img2.width)
                total_height = img1.height + img2.height
                combined = Image.new('RGB', (total_width, total_height), 'white')
                combined.paste(img1, (0, 0))
                combined.paste(img2, (0, img1.height))
                buf = io.BytesIO()
                combined.save(buf, format='PNG')
                buf.seek(0)
                plot_data = base64.b64encode(buf.read()).decode('utf-8')
                print("[App2] Returning AI-based plots (plot3+plot4) for 'Both' mode")
                return plot_data
            else:
                print("[App2] AI plots not found for 'Both' mode, falling back")

        # In "AI-Based Blind Detection" mode, display AI plots (plot1.png and plot2.png)
        if current_detection_mode == "AI-Based Blind Detection":
            if os.path.exists("plot1.png") and os.path.exists("plot2.png"):
                from PIL import Image
                img1 = Image.open("plot1.png")
                img2 = Image.open("plot2.png")
                total_width = max(img1.width, img2.width)
                total_height = img1.height + img2.height
                combined = Image.new('RGB', (total_width, total_height), 'white')
                combined.paste(img1, (0, 0))
                combined.paste(img2, (0, img1.height))
                buf = io.BytesIO()
                combined.save(buf, format='PNG')
                buf.seek(0)
                plot_data = base64.b64encode(buf.read()).decode('utf-8')
                print("[App2] Returning AI-based plots (plot1+plot2) for 'AI-Based' mode")
                return plot_data
            else:
                print("[App2] AI plots not found for 'AI-Based' mode, falling back")

        # Default: Load data_symbols.csv for DMRS-based or fallback
        data = np.loadtxt('data_symbols.csv', delimiter=',')
        iq_complex = data[:, 0] + 1j * data[:, 1]

        # Normalize
        iq_norm = np.linalg.norm(iq_complex) / np.sqrt(len(iq_complex))
        iq_normalized = iq_complex / iq_norm

        i_samples = np.real(iq_normalized)
        q_samples = np.imag(iq_normalized)

        # Create 2x2 plot
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Constellation
        axs[0, 0].plot(i_samples[:3276], q_samples[:3276], '.', alpha=0.5)
        axs[0, 0].set_title('Equalized Constellation')
        axs[0, 0].set_xlabel('I')
        axs[0, 0].set_ylabel('Q')
        axs[0, 0].grid(True)

        # Plot 2: I samples over index
        axs[0, 1].plot(i_samples[:3276])
        axs[0, 1].set_title('I Samples')
        axs[0, 1].set_xlabel('Sample Index')
        axs[0, 1].set_ylabel('Amplitude')
        axs[0, 1].grid(True)

        # Plot 3: Q samples over index
        axs[1, 0].plot(q_samples[:3276])
        axs[1, 0].set_title('Q Samples')
        axs[1, 0].set_xlabel('Sample Index')
        axs[1, 0].set_ylabel('Amplitude')
        axs[1, 0].grid(True)

        # Plot 4: Magnitude
        magnitude = np.abs(iq_normalized[:3276])
        axs[1, 1].plot(magnitude)
        axs[1, 1].set_title('Signal Magnitude')
        axs[1, 1].set_xlabel('Sample Index')
        axs[1, 1].set_ylabel('Magnitude')
        axs[1, 1].grid(True)

        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return plot_data

    except Exception as e:
        print(f"[App2] Error creating plot: {e}")
        return None


@app2.route("/")
def plots_page():
    """Display the analysis plots"""
    plot_data = create_plot()
    if plot_data:
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>O-RAN Analysis Plots</title>
        </head>
        <body style="background-color: #add8e6; text-align: center;">
            <h1>U-plane Data Performance</h1>
            <img src="data:image/png;base64,{{ plot }}" alt="Plot Grid">
        </body>
        </html>
        """
        return render_template_string(html, plot=plot_data)
    return "No plots available yet"


# -----------------------------------------------------------------------------
# Thread runner function
# -----------------------------------------------------------------------------

def run_app(flask_app, port):
    """Run a Flask app on the specified port"""
    flask_app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("O-RAN PCAP Analysis Flask Server")
    print("(Matching packet_oran_analysis_det_st.py)")
    print("=" * 60)
    print(f"scapy available: {SCAPY_AVAILABLE}")
    print("Starting servers:")
    print("  - Port 5000: Report display (app1)")
    print("  - Port 5001: Plot display (app2)")
    print("  - Port 5002: Upload/Analysis API (upload_app)")
    print("=" * 60)

    # Start all three servers in parallel threads
    Thread(target=run_app, args=(app, 5002), daemon=True).start()
    Thread(target=run_app, args=(app2, 5001), daemon=True).start()
    Thread(target=run_app, args=(app1, 5000), daemon=True).start()

    # Keep the main thread alive
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
