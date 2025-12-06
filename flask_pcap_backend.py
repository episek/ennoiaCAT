from flask import Flask, request, jsonify
from scapy.all import rdpcap, sendp
import pyshark
import threading, time, asyncio

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Flask server is up and running!"

@app.route("/replay_and_capture", methods=["POST"])
def replay_and_capture():
    data = request.json
    iface_out = data["iface_out"]
    iface_in = data["iface_in"]
    src_mac = data["src_mac"]
    pcap_file_in = data["pcap_file_in"]
    pcap_file_out = data["pcap_file_out"]

    print("[INFO] Input params:")
    print(iface_out, iface_in, src_mac, pcap_file_in, pcap_file_out)

    # === Load PCAP packets (FIXED INPUT FILE)
    packets = rdpcap(pcap_file_out)
    for pkt in packets:
        pkt.time = time.time()

    bytes_sent = sum(len(pkt) for pkt in packets)

    # === Setup pyshark capture
    def capture_job():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cap = pyshark.LiveCapture(
            interface=iface_in,
            output_file=pcap_file_in,
            bpf_filter=f"ether src {src_mac}",
            custom_parameters=["-B", "64"]
        )
        cap.sniff(timeout=50)
        loop.close()

    capture_thread = threading.Thread(target=capture_job)
    capture_thread.start()
    time.sleep(1)

    # === Send PCAP packets
    start = time.time()
    sendp(packets, iface=iface_out, inter=0.001, loop=0, verbose=1)
    end = time.time()
    duration = end - start

    capture_thread.join()

    return jsonify({
        "status": "done",
        "pkts_sent": len(packets),
        "bytes_sent": bytes_sent,
        "duration": duration,
        "output_file": pcap_file_in
    })

if __name__ == "__main__":
    app.run(port=8050)
