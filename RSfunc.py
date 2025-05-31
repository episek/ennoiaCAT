 
 
 #Define all the subroutines
from deep_translator import GoogleTranslator
import struct
import csv
from time import time, sleep
from RsInstrument import LoggingMode
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def com_prep(nrq, lang='en'):
    """Preparation of the communication (termination, etc...)"""
    manu = nrq.visa_manufacturer
    text = f"VISA Manufacturer: {manu}"
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    #st.write(translated)
    nrq.visa_timeout = 3000  # Timeout for VISA Read Operations
    nrq.opc_timeout = 3000  # Timeout for opc-synchronised operations
    nrq.instrument_status_checking = True  # Error check after each command, can be True or False
    nrq.clear_status()  # Clear status register
    nrq.logger.log_to_console = False  # Route SCPI logging feature to console on (True) or off (False)
    nrq.logger.mode = LoggingMode.Off  # Switch On or Off SCPI logging
    return translated 


def com_check(nrq, lang='en'):
    """Test the device connection, request ID as well as installed options"""
    idn_response = nrq.query('*IDN?')
    text = f"Hello, I am {idn_response}"
    translated_intro = GoogleTranslator(source='auto', target=lang).translate(text)
    str1 = translated_intro + idn_response
    query_response = nrq.query('*OPT?')
    text =f" \n and I have the following options available: \n {query_response}".replace("\\n", "\n")
    str2 = GoogleTranslator(source='auto', target=lang).translate(text)
    #st.write(translated)
    # idn_response = nrq.query('*IDN?')
    # st.write({nrq.query("SENSe:FUNCtion?")})
    # nrq.clear_status()
    # sleep(0.5)  # Give it time to settle before next query
    text = str1 + str2
    parts = text.split(',')
    translated = ', '.join(parts)
    return translated 

def meas_prep(nrq):
    """Prepare the devise for the measurement"""
    #inst.set_mode_iq()
    nrq.write('SENSe:FUNCtion "XTIMe:VOLT:IQ"')  # Change sensor mode to I/Q
    #inst.set_center(2e09)
    nrq.write('SENSe:FREQuency:CENTer 2e09')  # Center Frequency to 2 GHz
    #inst.set_bw_res_manual()
    nrq.write('SENSE:BANDwidth:RESolution:TYPE:AUTO:STATe OFF')  # Change bandwidth setting to manual state
    #inst.set_bw_res_normal()
    nrq.write('SENSE:BANDwidth:RESolution:TYPE NORMal')  # Flat filter type
    #inst.set_bw_res(1e8)
    nrq.write('SENSE:BANDwidth:RES 1e8''')  # Analysis bandwidth is 100 MHz now
    #inst.set_trace_length(15e5)
    nrq.write('SENSE:TRACe:IQ:RLENgth 15e5')  # IQ trace length is 1.5 million samples now
    nrq.write('')
    # cf = nrq.query('SENSe:FREQuency:CENTer?')
    # bw = nrq.query('SENSe:BANDwidth:RESolution?')
    # trace = nrq.query('SENSE:TRACe:IQ:RLENgth?')
    # sf = nrq.query('SENSe:BANDwidth:SRATe:CUV?')
    # text = (
        # f"Current setup parameters:\n"
        # f"Center Frequency is {cf} Hz,\n"
        # f"Analysis bandwidth is {bw} Hz,\n"
        # f"Trace length is {trace} Sa,\n"
        # f"Sample Rate is {sf} Sa/s,\n"
    # )
    # translated = GoogleTranslator(source='auto', target=lang).translate(text)
    # st.write(translated)    
    nrq.write('FORM:DATA REAL,64')


def collect_iq_samples(nrq, trace_length):
    # Set up IQ measurement
    nrq.write("CONF:IQ")  # Configure instrument for IQ acquisition

    # Set the IQ bandwidth (if needed)
    # nrq.write("SENS:IQ:BAND 20e6")  # For example: 20 MHz bandwidth

    # Set the number of IQ samples to capture
    nrq.write("SENS:IQ:POIN 10000")  # Number of points

    # Trigger a new acquisition
    nrq.write("INIT:IMM; *WAI")  # Start measurement and wait until complete

    # Fetch the IQ data as binary block
    iq_raw = nrq.query_bin_block("FETCh:WAVeform:IQ:TRACe?", datatype=RsInstrument.DataType.Float32, read_termination='')

    # Process the binary IQ data
    # IQ samples are returned as [I0, Q0, I1, Q1, ..., In, Qn]
    I_samples = iq_raw[::2]
    Q_samples = iq_raw[1::2]
    complex_iq = [complex(i, q) for i, q in zip(I_samples, Q_samples)]

    print(f"Collected {len(complex_iq)} IQ samples")
    # Convert flat list to list of (I, Q) pairs
    iq_pairs = list(zip(I_samples, Q_samples))
    return iq_pairs

def save_iq_as_csv(float_list, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        iq_pairs = zip(float_list[::2], float_list[1::2])  # I/Q
        writer.writerow(['I', 'Q'])
        writer.writerows(iq_pairs)

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

    print(f"[✅] Saved CSV: {csv_file}")
    return(iq_pairs)

def measure(nrq, lang='en'):
    """Perform measurement and timing calculation, print results"""
    start = time()  # Capture (system) start time
    nrq.write('INITiate:IMMediate')  # Initiates a single trigger measurement
    nrq.visa_timeout = 10000  # Extend Visa timeout to avoid errors
    #output = nrq.query_bin_or_ascii_float_list('FETCh1?')  # Get the measurement in binary format
    output = nrq.query_bin_or_ascii_float_list('FETCh1?')  # Get the measurement in binary format
    nrq.visa_timeout = 3000  # Change back timeout to standard value
    inter = time()  # Capture system time after I/Q data has been received
    duration = inter-start  # And calculate process time
    text = f"After {round(duration, 1)} seconds {len(output)} I/Q samples have been recorded."
    str1 = GoogleTranslator(source='auto', target=lang).translate(text)
    st.write(str1)        
    #st.write(f'After {round(duration, 1)} seconds {len(output)} I/Q samples have been recorded.')
    # Perform 2nd take
    #nrq.write('INITiate:IMMediate')
    #nrq.visa_timeout = 10000
    #output = nrq.query_bin_or_ascii_float_list('FETCh1?')
    #output = nrq.query_bin_or_ascii_float_list
    #nrq.visa_timeout = 3000
    

    # Connect to instrument
    #nrq = RsInstrument("TCPIP0::192.168.1.100::5025::SOCKET", id_query=True, reset=False)

    # Optional: configure frequency, RBW, etc. before capture
    # nrq.write("SENSE:FREQ:CENT 1.5e9")  # example center freq
    # nrq.write("SENSE:BAND:RES 1e6")     # example RBW

    # Capture and save I/Q data
    save_float_list_as_bin(output, "iq_capture.bin")
    #iq_data = collect_iq_samples(nrq, trace_length=10000)
    iq_pairs = bin_to_csv("iq_capture.bin", "iq_capture.csv")
    #bin_to_csv(output, "iq_capture.csv")
    #save_iq_as_csv(output, "iq_capture.csv")

    #instr.close()
    
    
    end = time()
    duration = end - start
    # text = f"After {round(duration, 1)} seconds both records have been taken, with the last one {len(output)} I/Q samples have been recorded."
    text = f"After {round(duration, 1)} seconds, {len(output)} I/Q samples have been saved."
    str2 = GoogleTranslator(source='auto', target=lang).translate(text)
    # text = str1 + str2
    # parts = text.split(',')
    # translated = ', '.join(parts)
    st.write(str2)        
    #st.write(f'After {round(duration, 1)} seconds both records have been taken,'
    #      f'with the last one {len(output)} I/Q samples have been recorded.')
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

def plot_fft(iq, sample_rate,lang='en'): 
    N = len(iq)
    fft_data = np.fft.fftshift(np.fft.fft(iq))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/sample_rate))
    magnitude_db = 20 * np.log10(np.abs(fft_data) + 1e-12)
    magnitude_db[N // 2] = magnitude_db[N // 2]-60
    
    text = f"Detected high DC component of > 60dB above noise floor. Suppressed the DC Component"
    translated = GoogleTranslator(source='auto', target=lang).translate(text)
    #st.write(translated)        

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs / 1e6, magnitude_db)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('FFT Spectrum')
    ax.grid(True)
    return translated,fig


def close(nrq):
    """Close the VISA session"""
    nrq.close()