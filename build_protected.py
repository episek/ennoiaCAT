#!/usr/bin/env python3
"""
Build Script for Ennoia Protected Distribution

This script creates the protected distribution package:
1. Obfuscates ennoia_core.py with PyArmor
2. Packages all necessary files
3. Creates a ready-to-distribute folder

Usage:
    python build_protected.py

Output:
    dist/ folder containing all distribution files
"""

import os
import shutil
import subprocess
import sys

# Configuration
SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(SOURCE_DIR, "dist")
PROTECTED_DIR = os.path.join(SOURCE_DIR, "dist_protected")

# Files to include in distribution
DISTRIBUTION_FILES = [
    # Core application
    "ennoiaCAT_Consolidated.py",
    "tinySA.py",
    "tinySA_config.py",
    "map_api.py",
    "timer.py",
    "ennoia_client_lic.py",

    # Data files
    "operator_table.json",
    "tinySA_train.json",

    # SLM Training
    "train_tinySA.py",

    # Documentation
    "README.md",
    "DOCUMENTATION.md",
    "LICENSING_GUIDE.md",
    "requirements_tinysa.txt",

    # Assets
    "ennoia.jpg",
]

# Directories to include
DISTRIBUTION_DIRS = [
    "tests",
]


def run_command(cmd, cwd=None):
    """Run a shell command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True


def clean_dist():
    """Clean existing distribution directory."""
    if os.path.exists(DIST_DIR):
        print(f"Cleaning {DIST_DIR}...")
        shutil.rmtree(DIST_DIR)
    os.makedirs(DIST_DIR)


def obfuscate_core():
    """Obfuscate ennoia_core.py with PyArmor."""
    print("\nObfuscating ennoia_core.py...")

    # Clean protected directory
    if os.path.exists(PROTECTED_DIR):
        shutil.rmtree(PROTECTED_DIR)

    # Run PyArmor
    cmd = ["pyarmor", "gen", "-O", PROTECTED_DIR, "ennoia_core.py"]
    if not run_command(cmd, cwd=SOURCE_DIR):
        print("PyArmor obfuscation failed!")
        return False

    return True


def copy_files():
    """Copy distribution files."""
    print("\nCopying distribution files...")

    # Copy protected core
    shutil.copy(
        os.path.join(PROTECTED_DIR, "ennoia_core.py"),
        os.path.join(DIST_DIR, "ennoia_core.py")
    )

    # Copy PyArmor runtime
    runtime_src = None
    for item in os.listdir(PROTECTED_DIR):
        if item.startswith("pyarmor_runtime"):
            runtime_src = os.path.join(PROTECTED_DIR, item)
            break

    if runtime_src:
        shutil.copytree(
            runtime_src,
            os.path.join(DIST_DIR, os.path.basename(runtime_src))
        )

    # Copy other files
    for filename in DISTRIBUTION_FILES:
        src = os.path.join(SOURCE_DIR, filename)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(DIST_DIR, filename))
            print(f"  Copied: {filename}")
        else:
            print(f"  Warning: {filename} not found")

    # Copy directories
    for dirname in DISTRIBUTION_DIRS:
        src = os.path.join(SOURCE_DIR, dirname)
        if os.path.exists(src):
            shutil.copytree(src, os.path.join(DIST_DIR, dirname))
            print(f"  Copied: {dirname}/")


def create_activation_script():
    """Create activation helper script."""
    script = '''#!/usr/bin/env python3
"""
Ennoia License Activation Helper

Run this script to get your machine information for license activation.
Send the displayed information to Ennoia to receive your license.
"""

try:
    from ennoia_core import get_device_info, get_license_info

    print("=" * 60)
    print("Ennoia tinySA Controller - License Activation")
    print("=" * 60)

    print("\\nDevice Information:")
    info = get_device_info()
    print(f"  Machine ID:    {info['machine_id']}")
    print(f"  tinySA Serial: {info['tinysa_serial']}")
    print(f"  tinySA Port:   {info['tinysa_port']}")

    print("\\n" + "-" * 60)
    print("Please send the above Machine ID and tinySA Serial to Ennoia")
    print("to receive your license file.")
    print("-" * 60)

    print("\\nLicense Status:")
    lic_info = get_license_info()
    if "error" in lic_info:
        print(f"  Status: NOT ACTIVATED")
        print(f"  Message: {lic_info['error']}")
    else:
        print(f"  Customer: {lic_info.get('customer', 'N/A')}")
        print(f"  Expires:  {lic_info.get('expires', 'N/A')}")
        print(f"  Status:   VALID")

except ImportError:
    print("Error: ennoia_core module not found.")
    print("Please ensure the application is properly installed.")
'''

    with open(os.path.join(DIST_DIR, "activate.py"), "w") as f:
        f.write(script)
    print("  Created: activate.py")


def main():
    print("=" * 60)
    print("Ennoia Protected Distribution Builder")
    print("=" * 60)

    # Check PyArmor is installed
    try:
        subprocess.run(["pyarmor", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: PyArmor is not installed.")
        print("Install with: pip install pyarmor")
        return 1

    # Build steps
    clean_dist()

    if not obfuscate_core():
        return 1

    copy_files()
    create_activation_script()

    print("\n" + "=" * 60)
    print("Build complete!")
    print("=" * 60)
    print(f"\nDistribution folder: {DIST_DIR}")
    print("\nTo distribute:")
    print("1. Zip the 'dist' folder")
    print("2. Send to customer along with license.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
