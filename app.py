from flask import Flask, request, jsonify
from pcap_utils import extract_oran_uplane
import os
import csv

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_pcap():
    file = request.files.get("pcap")
    if not file:
        return jsonify({"error": "No PCAP file uploaded"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    try:
        parsed = extract_oran_uplane(save_path)
        output_csv = os.path.join(OUTPUT_FOLDER, file.filename + ".csv")
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Port", "Subframe", "Slot", "Symbol", "IQ_Index", "Real", "Imag"])
            for entry in parsed:
                for i, iq in enumerate(entry["iq"]):
                    writer.writerow([
                        entry["port_id"], entry["subframe"], entry["slot"],
                        entry["symbol"], i, iq.real, iq.imag
                    ])
        return jsonify({"status": "ok", "output_file": output_csv})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8010)
