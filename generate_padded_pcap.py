from scapy.all import rdpcap, wrpcap, Raw

pcap_file_in  = "test_ip36to37.pcap"
pcap_file_out = "test_ip36to37_1500B.pcap"

target_size = 1500  # target total packet size (bytes)

# Load input PCAP
packets = rdpcap(pcap_file_in)
print(f"[INFO] Loaded {len(packets)} packets from {pcap_file_in}")

new_packets = []
for pkt in packets:
    cur_len = len(pkt)
    if cur_len < target_size:
        pad_len = target_size - cur_len
        pkt = pkt / Raw(b"X" * pad_len)  # add padding
    elif cur_len > target_size:
        # Trim payload if bigger than 1500B
        pkt = pkt.copy()
        pkt = pkt[:target_size]
    new_packets.append(pkt)

# Save new PCAP
wrpcap(pcap_file_out, new_packets)
print(f"[INFO] Wrote {len(new_packets)} packets to {pcap_file_out} (each ~{target_size} bytes)")
