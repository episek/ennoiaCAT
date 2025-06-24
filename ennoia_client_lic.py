import requests, json, base64, hashlib, uuid
from datetime import datetime
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature

SERVER_URL = "http://localhost:9000"

def get_fingerprint():
    raw = f"{uuid.getnode()}"
    return hashlib.sha256(raw.encode()).hexdigest()

def request_license(license_key):
    payload = {
        "license_key": license_key,
        "fingerprint": get_fingerprint()
    }
    try:
        r = requests.post(f"{SERVER_URL}/verify", json=payload)
        r.raise_for_status()
        with open("license.json", "w") as f:
            json.dump(r.json(), f)
        print("✅ License file saved.")
    except Exception as e:
        print("❌ License request failed:", e)

def verify_license_file():
    try:
        with open("license.json", "r") as f:
            data = json.load(f)
        print("🔍 Verifying license...\n")
        success = 0
        license_data = data["license"]
        signature = base64.b64decode(data["signature"])
        payload = json.dumps(license_data).encode()

        print("License Data = ",license_data)
        print("\nSignature = ",signature)
        print("\nPayload = ", payload)
        
        # Expiration check
        if datetime.utcnow() > datetime.strptime(license_data["expires"], "%Y-%m-%d"):
            print("❌ License expired.")
            return success
        else:
            print("✅ License is still valid until", license_data["expires"])

        # Fingerprint check
        if license_data["fingerprint"] != get_fingerprint():
            print("❌ License not valid for this machine.")
            return success
        else:
            print("✅ License fingerprint matches this machine.")

        # Load public key
        r = requests.get(f"{SERVER_URL}/public-key")
        public_key = serialization.load_pem_public_key(r.json()["public_key"].encode())
        print("🔑 Public key loaded successfully.")
       
        try:
            public_key.verify(
                signature,
                payload,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            print("✅ Signature is valid.")
        except InvalidSignature:
            print("❌ Signature is invalid.")

        print("✅ License is valid and untampered.")
        success = 1
        return success
    except Exception as e:
        print("❌ License verification failed:", e)
        success = 0
        return success 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["activate", "verify"])
    parser.add_argument("--key", help="License key for activation")
    args = parser.parse_args()

    if args.action == "activate":
        if not args.key:
            print("❗ Please provide a license key with --key")
        else:
            request_license(args.key)
    elif args.action == "verify":
        verify_license_file()