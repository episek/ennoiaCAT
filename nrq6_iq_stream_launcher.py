import os
import socket
import subprocess
import time
from pathlib import Path
from RsInstrument import *

# === USER CONFIGURATION ===
visa_resource = 'TCPIP::nrq6-101528::hislip0'
udp_port = 50000
stream_tool_exe = 'NRQ6_IQ_Streaming_Tool.exe'
convert_exe = 'ConvertIQ_13.exe'
expected_wv_file = 'iq_capture.wv'  # or 'iq_capture.iqw'
output_csv = 'iq_capture.csv'

def get_local_ip(nrq6_host='nrq6-101528'):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((socket.gethostbyname(nrq6_host), 80))
        return s.getsockname()[0]

def configure_nrq6_streaming(nrq6, pc_ip, port):
    print(f"[INFO] Configuring NRQ6 to stream IQ to {pc_ip}:{port}")
    scpi_cmd = f'SYST:COMM:LAN:IQ:DEST "{pc_ip}",{port}'
    nrq6.write(scpi_cmd)
    print(f"[✅] NRQ6 configured")

def launch_stream_tool(tool_path):
    print(f"[INFO] Launching R&S Streaming Tool: {tool_path}")
    subprocess.Popen([tool_path], shell=True)

def wait_for_wv_file(path, timeout=120):
    print(f"[WAIT] Waiting for file: {path}")
    for _ in range(timeout):
        if Path(path).exists():
            print(f"[✅] File detected: {path}")
            return True
        time.sleep(1)
    print(f"[❌] Timeout waiting for file: {path}")
    return False

def convert_wv_to_csv(wv_path, csv_path, exe_name="ConvertIQ_13.exe"):
    exe_path = os.path.join(os.path.dirname(__file__), exe_name)
    cmd = [exe_path, '-i', wv_path, '-o', csv_path, '-f', 'CSV']
    print(f"[INFO] Running conversion: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"[✅] CSV created: {csv_path}")
    else:
        print(f"[❌] Conversion failed:\n{result.stderr}")

if __name__ == "__main__":
    try:
        local_ip = get_local_ip()
        print(f"[INFO] Detected PC IP: {local_ip}")

        nrq6 = RsInstrument(visa_resource, id_query=True, reset=False)
        configure_nrq6_streaming(nrq6, local_ip, udp_port)
        nrq6.close()

        if not os.path.exists(stream_tool_exe):
            raise FileNotFoundError(f"{stream_tool_exe} not found")
        
        launch_stream_tool(stream_tool_exe)

        input("\n[🕹️] After capture completes and you stop streaming, press Enter to continue...")

        if wait_for_wv_file(expected_wv_file):
            convert_wv_to_csv(expected_wv_file, output_csv, exe_name=convert_exe)

    except Exception as e:
        print(f"[❌ ERROR] {e}")
