#!/usr/bin/env python3
"""
Ennoia License Key Generator
Run ONCE to generate RSA key pair for license signing.
KEEP private_key.pem SECRET - never share or commit to git!
"""

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import os

def generate_keys(output_dir="."):
    """Generate RSA key pair for license signing."""

    # Generate 2048-bit RSA key pair
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    public_key = private_key.public_key()

    # Save private key (PEM format)
    private_path = os.path.join(output_dir, "private_key.pem")
    with open(private_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    print(f"Private key saved to: {private_path}")
    print("WARNING: Keep this file SECRET! Never commit to git!")

    # Save public key (PEM format)
    public_path = os.path.join(output_dir, "public_key.pem")
    with open(public_path, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
    print(f"Public key saved to: {public_path}")

    # Also output public key as Python string for embedding
    public_key_str = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()

    embed_path = os.path.join(output_dir, "public_key_embed.py")
    with open(embed_path, "w") as f:
        f.write('# Auto-generated - embed this in ennoia_core.py\n')
        f.write('PUBLIC_KEY = """' + public_key_str + '"""\n')
    print(f"Embeddable public key saved to: {embed_path}")

    return private_path, public_path

if __name__ == "__main__":
    print("Generating RSA key pair for Ennoia licensing...")
    generate_keys()
    print("\nDone! Now you can use generate_license.py to create licenses.")
