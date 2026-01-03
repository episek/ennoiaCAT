from scapy.all import rdpcap, sendp, sendpfast, get_if_list, get_if_addr, get_if_hwaddr
import time, statistics, threading
import matplotlib.pyplot as plt
import pyshark
import sys

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


iface_out = sys.argv[1]
iface_in = sys.argv[2]
src_ip = sys.argv[3]
pcap_file_in = sys.argv[4]
pcap_file_out = sys.argv[5]


# === Select NICs ===
# iface_out = choose_interface("Output (send PCAP to Gi0/0/0/1)")
# iface_in  = choose_interface("Capture (connected to either Gi0/0/0/3 or Gi0/0/0/5)")

# pcap_file_in  = "oran_uplane_DLUL_cisco.pcap"    # input PCAP file to replay
# #pcap_file_in  = "test_ip36to37_9000B.pcap"    # input PCAP file to replay
# #pcap_file_in  = "test_ip36to37_1.pcap"    # input PCAP file to replay
# pcap_file_out = "capture_output.pcapng" # pyshark saves here

# === Load PCAP to replay ===
packets = rdpcap(pcap_file_in)
print(f"[INFO] Loaded {len(packets)} packets from {pcap_file_in}")

# Tag send times for latency check
send_timestamps = []
for pkt in packets:
    pkt.time = time.time()

# Automatically fetch IP of the output NIC you selected
src_ip = get_if_addr(iface_out)
print(f"[INFO] Using source IP {src_ip} for BPF filter")

# Setup pyshark capture with filter
cap = pyshark.LiveCapture(
    interface=iface_in,
    output_file=pcap_file_out,
    custom_parameters=["-B", "64"],
    bpf_filter=f"src host {src_ip}"
)

def capture_job():
    print(f"[INFO] Starting pyshark capture on {iface_in} ...")
    cap.sniff(timeout=25)
    #cap.sniff(packet_count=1000)
    print(f"[INFO] Capture finished, {len(cap)} packets saved to {pcap_file_out}")

# === Run capture in parallel with sending ===
capture_thread = threading.Thread(target=capture_job)
capture_thread.start()

time.sleep(1)  # give tshark time to start

print(f"[INFO] Sending packets on {iface_out}...")
t_start = time.time()
bytes_sent = sum(len(pkt) for pkt in packets)
sendp(packets, iface=iface_out, inter=0.0005, loop=0, verbose=1)
#sendpfast(packets, iface=iface_out, file_cache=True, pps=0)
t_end = time.time()
duration = t_end - t_start
print(f"[INFO] Finished sending in {duration:.3f} seconds")

capture_thread.join()

# === Reload captured packets for analysis ===
sniffer = rdpcap(pcap_file_out)
print(f"[INFO] Reloaded {len(sniffer)} packets from {pcap_file_out}")

# === Stats ===
pkts_sent = len(packets)
pkts_recv = len(sniffer)
loss = pkts_sent - pkts_recv
loss_pct = (loss / pkts_sent * 100) if pkts_sent else 0
throughput_bps = (bytes_sent * 8) / duration if duration > 0 else 0
throughput_mbps = throughput_bps / 1e6
throughput_gbps = throughput_bps / 1e9

print("\n=== TEST RESULTS ===")
print(f"Packets Sent     : {pkts_sent}")
print(f"Packets Received : {pkts_recv}")
print(f"Packet Loss      : {loss} ({loss_pct:.2f}%)")
print(f"Test Duration    : {duration:.3f} sec")
print(f"Throughput       : {throughput_mbps:.2f} Mbps ({throughput_gbps:.3f} Gbps)")

# === Latency + Jitter (only if same length) ===
if pkts_recv and pkts_sent == pkts_recv:
    recv_times = [pkt.time for pkt in sniffer]
    latencies = [(recv_times[i] - packets[i].time) * 1000 for i in range(pkts_recv)]
    jitters = [latencies[i+1] - latencies[i] for i in range(len(latencies)-1)]

    print(f"Avg Latency      : {statistics.mean(latencies):.3f} ms")
    print(f"Min Latency      : {min(latencies):.3f} ms")
    print(f"Max Latency      : {max(latencies):.3f} ms")
    print(f"Avg Jitter       : {statistics.mean(jitters):.3f} ms")

    # Plot latency per packet
    plt.figure(figsize=(10,4))
    plt.plot(latencies, marker='o', linestyle='-', markersize=2)
    plt.title("Latency per Packet")
    plt.xlabel("Packet Index")
    plt.ylabel("Latency (ms)")
    plt.grid(True)
    plt.show()

    # Plot jitter histogram
    plt.figure(figsize=(6,4))
    plt.hist(jitters, bins=30, color="skyblue", edgecolor="black")
    plt.title("Jitter Distribution")
    plt.xlabel("Jitter (ms)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# === Show a few captured packets ===
print("\n=== First 10 Captured Packets ===")
for pkt in sniffer[:10]:
    print(f"[CAPTURE] {pkt.summary()}")
