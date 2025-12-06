from scapy.all import Ether, IP, UDP, wrpcap

packets = []
for i in range(100000):
    pkt = Ether(src="02:11:22:33:44:55", dst="00:E0:4C:27:77:50") \
          / IP(src="192.168.1.36", dst="192.168.2.37") \
          / UDP(sport=12345, dport=54321) \
          / f"TEST_FRAME_{i}"
    packets.append(pkt)

wrpcap("test_ip36to37_100000p.pcap", packets)
print("Wrote test_ip36to37.pcap with 100000 packets")
