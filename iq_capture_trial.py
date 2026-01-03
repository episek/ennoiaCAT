import socket
import subprocess
import time
import os
import zipfile
import re

def extract_ip_from_resource(resource):
    match = re.search(r'TCPIP::([^:]+)::', resource)
    return match.group(1) if match else None

def capture_iq_bin_file(output_path, port=50000, duration_sec=10):
    print(f"[INFO] Listening for IQ data on UDP port {port} for {duration_sec} seconds...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))
    sock.settimeout(1.0)

    with open(output_path, 'wb') as f:
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            try:
                data, _ = sock.recvfrom(65536)
                f.write(data)
            except socket.timeout:
                continue

    print(f"[INFO] IQ data saved to: {output_path}")

def convert_bin_to_csv(bin_path, csv_path, exe_name="ConvertIQ_13.exe"):
    exe_path = os.path.join(os.path.dirname(__file__), exe_name)
    cmd = [exe_path, '-i', bin_path, '-o', csv_path, '-f', 'CSV']
    
    print(f"[INFO] Running converter: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"[INFO] Conversion successful: {csv_path}")
        return True
    else:
        print(f"[ERROR] Conversion failed:\n{result.stderr}")
        return False

def zip_file(file_path):
    zip_path = file_path + ".zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname=os.path.basename(file_path))
    print(f"[INFO] Compressed to: {zip_path}")
    return zip_path

if __name__ == "__main__":
    # Step 1: Get IP from VISA resource string
    visa_resource = 'TCPIP::nrq6-101528::hislip0'
    ip_address = extract_ip_from_resource(visa_resource)
    print(f"[INFO] Using NRQ6 IP address: {ip_address}")

    # Step 2: Define file names
    base_name = "iq_capture"
    bin_file = base_name + ".bin"
    csv_file = base_name + ".csv"

    # Step 3: Capture IQ stream and convert
    capture_iq_bin_file(bin_file, port=50000, duration_sec=10)
    if convert_bin_to_csv(bin_file, csv_file):
        zip_file(csv_file)
