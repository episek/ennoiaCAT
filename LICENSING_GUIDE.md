# Ennoia tinySA Licensing System

This document explains the node-locked licensing system for Ennoia tinySA Controller.

---

## Table of Contents

1. [Overview](#overview)
2. [For Ennoia Administrators](#for-ennoia-administrators)
3. [For End Users](#for-end-users)
4. [License Tiers](#license-tiers)
5. [Troubleshooting](#troubleshooting)
6. [Technical Details](#technical-details)

---

## Overview

The Ennoia licensing system uses **node-locked licensing** that binds a license to:
- A specific **machine** (identified by hardware fingerprint)
- A specific **tinySA device** (identified by serial number)

This ensures that each license can only be used on the authorized computer with the authorized tinySA device.

### Architecture

```
+----------------------------------+
|  ennoia_core (Protected)         |
|  - License validation            |
|  - Machine ID generation         |
|  - tinySA serial detection       |
|  - getport() function            |
+----------------------------------+
              |
              v
+----------------------------------+
|  tinySA.py (Open Source)         |
|  - Device commands               |
|  - Scan functions                |
|  - Data parsing                  |
+----------------------------------+
              |
              v
+----------------------------------+
|  ennoiaCAT_Consolidated.py       |
|  - Streamlit UI                  |
|  - AI integration                |
+----------------------------------+
```

---

## For Ennoia Administrators

### Initial Setup (One Time)

1. **Generate RSA Key Pair**

   Run once to create signing keys:
   ```bash
   cd licensing
   python generate_keys.py
   ```

   This creates:
   - `private_key.pem` - **KEEP SECRET, NEVER SHARE**
   - `public_key.pem` - Embedded in the app
   - `public_key_embed.py` - For embedding in code

2. **Protect the Private Key**

   - Store `private_key.pem` securely
   - Never commit to git
   - Back up in a secure location
   - This key signs all licenses

### Generating Licenses for Customers

1. **Get Customer's Machine Info**

   Ask the customer to run:
   ```bash
   python ennoia_core.py
   ```

   They will see output like:
   ```
   Device Information:
     Machine ID:    efbacac31246e8d5f8ce113bba6c1c98
     tinySA Serial: c916bf5052784944
     tinySA Port:   COM3
   ```

2. **Generate the License**

   ```bash
   cd licensing
   python generate_license.py \
       --customer "Company Name" \
       --email "user@company.com" \
       --machine-id "efbacac31246e8d5f8ce113bba6c1c98" \
       --tinysa-serial "c916bf5052784944" \
       --days 365 \
       --tier professional \
       --output customer_license.json
   ```

3. **Send License to Customer**

   Email the generated `customer_license.json` file to the customer with activation instructions.

### License Generation Options

| Option | Description | Example |
|--------|-------------|---------|
| `--customer` | Customer/company name | "Acme Corp" |
| `--email` | Customer email | "user@acme.com" |
| `--machine-id` | Machine fingerprint | "abc123..." |
| `--tinysa-serial` | Device serial number | "TSA-12345" |
| `--days` | Validity period | 60, 365 |
| `--tier` | License tier | trial, standard, professional, perpetual |
| `--output` | Output file path | license.json |

### Renewing Licenses

1. Generate a new license with updated expiration date
2. Send the new `license.json` to the customer
3. Customer replaces their old license file

### Building Protected Distribution

1. **Obfuscate the core module**
   ```bash
   mkdir -p dist_protected
   pyarmor gen -O dist_protected ennoia_core.py
   ```

2. **Package for distribution**
   ```
   dist/
   ├── ennoia_core.py           (obfuscated)
   ├── pyarmor_runtime_000000/  (runtime)
   ├── tinySA.py                (open source)
   ├── tinySA_config.py         (open source)
   ├── ennoiaCAT_Consolidated.py
   ├── map_api.py
   └── requirements.txt
   ```

---

## For End Users

### Activation Steps

1. **Get Your License**

   Contact Ennoia to purchase a license. Provide:
   - Your company/name
   - Your email address

2. **Get Your Machine Info**

   Connect your tinySA device and run:
   ```bash
   python ennoia_core.py
   ```

   Send the displayed **Machine ID** and **tinySA Serial** to Ennoia.

3. **Receive License File**

   Ennoia will send you a `license.json` file.

4. **Activate**

   Place `license.json` in the application directory:
   ```
   ennoiaCAT/
   ├── license.json        <-- Place here
   ├── ennoia_core.py
   ├── tinySA.py
   └── ...
   ```

5. **Run the Application**

   ```bash
   streamlit run ennoiaCAT_Consolidated.py
   ```

### Verifying Your License

Run:
```bash
python ennoia_core.py
```

If valid, you'll see:
```
License Status:
  Customer:  Your Company
  Expires:   2027-01-25
  Status:    VALID
```

### License Renewal

1. Contact Ennoia before your license expires
2. Receive a new `license.json` file
3. Replace the old file with the new one
4. No reinstallation needed

---

## License Tiers

| Tier | Duration | Features |
|------|----------|----------|
| **Trial** | 60 days | Basic scan, spectrum view |
| **Standard** | 1 year | + WiFi scan, CSV export |
| **Professional** | 1 year | + Cellular ID, advanced analysis, SLM mode |
| **Perpetual** | Lifetime | All features, never expires |

---

## Troubleshooting

### "License file not found"

**Cause**: No `license.json` in the application directory.

**Solution**:
1. Ensure `license.json` is in the same folder as `ennoia_core.py`
2. Check file name is exactly `license.json` (case-sensitive on Linux/Mac)

### "License not valid for this machine"

**Cause**: The license was generated for a different computer.

**Solution**:
1. Run `python ennoia_core.py` to get your current Machine ID
2. Contact Ennoia with the new Machine ID
3. Request a new license for this machine

### "License not valid for this tinySA device"

**Cause**: The license was generated for a different tinySA.

**Solution**:
1. Connect the correct tinySA device
2. Or contact Ennoia to update the license for your new device

### "License expired"

**Cause**: The license validity period has ended.

**Solution**:
1. Contact Ennoia to renew your license
2. Replace the old `license.json` with the renewed one

### "Invalid license signature"

**Cause**: The license file has been modified or corrupted.

**Solution**:
1. Re-download the license file from Ennoia
2. Ensure the file wasn't modified by email clients or editors
3. Request a new license if the problem persists

---

## Technical Details

### Machine ID Generation

The machine fingerprint is generated from:
- MAC address (network adapter)
- Hashed with SHA-256
- First 32 characters used

### tinySA Serial Detection

The tinySA serial is obtained by:
1. Detecting the device via USB VID/PID (0x0483:0x5740)
2. Sending the `info` command
3. Parsing the serial from the response

### License File Format

```json
{
  "license": {
    "customer": "Company Name",
    "email": "user@company.com",
    "machine_id": "efbacac31246e8d5f8ce113bba6c1c98",
    "tinysa_serial": "c916bf5052784944",
    "tier": "professional",
    "issued": "2026-01-25",
    "expires": "2027-01-25",
    "features": ["basic_scan", "spectrum_view", ...]
  },
  "signature": "BASE64_RSA_SIGNATURE..."
}
```

### Security

- Licenses are signed with RSA-2048
- Signature prevents tampering
- Core module is obfuscated with PyArmor
- Cannot be bypassed without the valid license file

---

## Support

For licensing issues, contact:
- GitHub: https://github.com/rajagopalsridhar/ennoiaCAT/issues

---

*Ennoia Technologies Connect Platform - All rights reserved*
