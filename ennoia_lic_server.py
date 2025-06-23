from flask import Flask, request, jsonify
from datetime import datetime, timezone
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
import base64, json, os

import os

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "license_log.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)
print(f"📓 License logs will be saved to: {os.path.abspath(LOG_FILE)}")
with open(LOG_FILE, "w", encoding="utf-8") as f:
    pass  # Creates an empty file

app = Flask(__name__)

# In-memory license database
licenses = {
    "Ennoia1": {"fingerprint": None, "expires": "2030-01-01"},
    "Ennoia2": {"fingerprint": None, "expires": "2029-12-31"},
    "Guest": {"fingerprint": None, "expires": "2031-06-15"}
}

# RSA key pair
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
public_key = private_key.public_key()

# Log file path
LOG_FILE = "license_log.jsonl"

def log_event(event: dict):
    print("📣 Logging event:", event)  # Debug print
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
        f.flush()



@app.route('/verify', methods=['POST'])
def verify():
    data = request.json
    key = data.get('license_key')
    fingerprint = data.get('fingerprint')
    client_ip = request.remote_addr

    license_info = licenses.get(key)
    event = {
        "license_key": key,
        "fingerprint": fingerprint,
        "client_ip": client_ip,
        "action": "verify"
    }

    if not license_info:
        event["status"] = "rejected"
        event["reason"] = "Invalid license key"
        log_event(event)
        return jsonify({"valid": False, "reason": event["reason"]}), 403

    if datetime.now(timezone.utc).isoformat() > datetime.strptime(license_info['expires'], "%Y-%m-%d"):
        event["status"] = "rejected"
        event["reason"] = "License expired"
        log_event(event)
        return jsonify({"valid": False, "reason": event["reason"]}), 403

    if license_info['fingerprint'] is None:
        licenses[key]['fingerprint'] = fingerprint
        event["status"] = "bound"
    elif fingerprint != license_info['fingerprint']:
        event["status"] = "rejected"
        event["reason"] = "Fingerprint mismatch"
        log_event(event)
        return jsonify({"valid": False, "reason": event["reason"]}), 403
    else:
        event["status"] = "granted"

    # Sign license payload
    license_payload = {
        "license_key": key,
        "fingerprint": fingerprint,
        "expires": license_info['expires']
    }
    payload_bytes = json.dumps(license_payload).encode()
    signature = private_key.sign(
        payload_bytes,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )

    event["result"] = "signed"
    log_event(event)

    return jsonify({
        "valid": True,
        "license": license_payload,
        "signature": base64.b64encode(signature).decode()
    })

@app.route('/public-key', methods=['GET'])
def get_public_key():
    client_ip = request.remote_addr
    event = {
        "action": "get_public_key",
        "client_ip": client_ip,
        "status": "served"
    }
    log_event(event)

    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return {"public_key": pem.decode()}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)