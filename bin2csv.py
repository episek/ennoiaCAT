import struct
import csv
import os

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

bin_to_csv("iq_capture.bin", "iq_capture1.csv")