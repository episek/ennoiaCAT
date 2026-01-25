# Ennoia tinySA Controller - Documentation

## Overview

Ennoia tinySA Controller is an AI-powered spectrum analyzer interface for the tinySA device. Control your tinySA using natural language commands with either cloud-based AI (OpenAI) or a locally-trained Small Language Model (SLM) for offline operation.

### Features

- Natural language control of tinySA spectrum analyzer
- Automatic cellular network detection and identification
- WiFi network scanning (2.4/5/6 GHz bands)
- Multi-language support (11 languages)
- Offline mode with local SLM
- CSV export of scan data

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_tinysa.txt
```

### 2. Set Up API Key (for Online Mode)

Create a `.env` file:
```env
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Connect tinySA

- Connect tinySA to USB port
- Device will be auto-detected (VID: 0x0483, PID: 0x5740)

### 4. Run the App

```bash
streamlit run ennoiaCAT_Consolidated.py
```

Open `http://localhost:8501` in your browser.

---

## Using the Application

### Step 1: Select AI Mode

- **Online LLM** - Uses OpenAI GPT-4o-mini (requires internet + API key)
- **Local SLM** - Uses TinyLlama (offline, requires training first)

### Step 2: Enter Commands

Type natural language commands in the chat:

```
"Scan from 300 MHz to 900 MHz"
"Show me the cellular bands"
"Check WiFi signals at 2.4 GHz"
"Set the frequency range to 88 MHz to 108 MHz for FM radio"
```

### Step 3: View Results

- Spectrum plot displays automatically
- Detected cellular networks shown in table
- WiFi networks listed with signal strength

---

## Example Commands

### Frequency Scanning

| Command | Action |
|---------|--------|
| "Scan 300 to 900 MHz" | Sets frequency range and scans |
| "Start at 100 MHz, stop at 500 MHz" | Sets start/stop frequencies |
| "Scan the FM radio band" | Scans 88-108 MHz |
| "Check the 2.4 GHz WiFi band" | Scans 2400-2500 MHz |
| "Scan cellular LTE bands" | Scans 700-900 MHz |

### Device Control

| Command | Action |
|---------|--------|
| "Set RBW to 100 kHz" | Adjusts resolution bandwidth |
| "Take a screenshot" | Captures tinySA display |
| "What is the peak signal?" | Finds maximum signal |

### Analysis

| Command | Action |
|---------|--------|
| "What operators are active?" | Identifies cellular carriers |
| "Find WiFi interference" | Scans and lists WiFi networks |
| "Check for signals at 433 MHz" | Scans ISM band |

---

## Running Tests

```bash
# Run all tinySA tests
python -m pytest tests/unit/test_tinysa.py -v

# Run with verbose output
python -m pytest tests/ -v --tb=short
```

### Test Coverage

- Device detection (VID/PID matching)
- Serial command generation
- Frequency parsing and conversion
- WiFi channel mapping
- CSV export/import
- API key validation

---

## SLM Training Guide

The Small Language Model allows offline operation without an internet connection or API key.

### Prerequisites

```bash
pip install torch transformers peft datasets
```

For GPU (faster training):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Training Data Structure

Edit `tinySA_train.json`:

```json
[
  {
    "instruction": "User's command or question",
    "input": "Additional context (optional)",
    "response": "AI's response"
  }
]
```

### Adding Natural Language Commands

#### Basic Commands

```json
{
  "instruction": "Set start frequency to 300 MHz",
  "input": "",
  "response": "Setting start frequency to 300 MHz. Navigate to Frequency menu and select Start, then enter 300M."
}
```

#### Casual Variations

```json
{
  "instruction": "start at 300 meg",
  "input": "",
  "response": "Setting start frequency to 300 MHz."
}
```

#### Question Format

```json
{
  "instruction": "How do I scan for FM radio stations?",
  "input": "",
  "response": "To scan FM radio, set the frequency range from 88 MHz to 108 MHz. Set Start to 88M and Stop to 108M. Each peak represents a radio station."
}
```

#### Contextual Commands

```json
{
  "instruction": "I want to check my WiFi signal",
  "input": "My router is 2.4 GHz",
  "response": "To check 2.4 GHz WiFi, scan from 2400 MHz to 2500 MHz. Your router's signal will appear as a peak. Channels 1, 6, and 11 are most common at 2412, 2437, and 2462 MHz."
}
```

#### Error Handling

```json
{
  "instruction": "Scan at 10 GHz",
  "input": "",
  "response": "The tinySA supports frequencies up to 960 MHz (or 5.3 GHz on Ultra model). Please choose a frequency within the supported range."
}
```

### Training Categories to Include

1. **Frequency Commands** (50+ examples)
   - Setting start/stop frequencies
   - Center/span mode
   - Common bands (FM, cellular, WiFi, ISM)

2. **Device Operations** (30+ examples)
   - RBW settings
   - Reference level
   - Markers
   - Screenshots

3. **Analysis Questions** (30+ examples)
   - Signal identification
   - Peak finding
   - Interference detection

4. **Troubleshooting** (20+ examples)
   - Device not found
   - No signal detected
   - Calibration

5. **Educational** (20+ examples)
   - What is RBW?
   - How does a spectrum analyzer work?
   - Band explanations

### Run Training

```bash
python train_tinySA.py
```

Training takes 5-15 minutes depending on hardware.

### Training Output

```
tinyllama_tinysa_lora/
├── adapter_model.safetensors   # Trained weights (~4.5 MB)
├── adapter_config.json         # LoRA config
├── tokenizer.json              # Tokenizer
└── final_eval_results.json     # Training metrics
```

### Evaluate Training Quality

Check `final_eval_results.json`:

| eval_loss | Quality | Action |
|-----------|---------|--------|
| < 1.0 | Excellent | Ready to use |
| 1.0 - 2.0 | Good | Acceptable |
| 2.0 - 3.0 | Fair | Add more training data |
| > 3.0 | Poor | Review data quality |

### Improving Model Quality

1. **Add more examples** - Aim for 150+ total
2. **Vary phrasing** - Same intent, different words
3. **Include typos** - "freq" instead of "frequency"
4. **Add slang** - "meg" for MHz, "gig" for GHz
5. **Real questions** - Use actual user queries

---

## File Structure

```
ennoiaCAT/
├── ennoiaCAT_Consolidated.py   # Main application
├── tinySA.py                   # tinySA driver
├── tinySA_config.py            # tinySA helper class
├── map_api.py                  # LLM API wrapper
├── timer.py                    # Timing utilities
├── ennoia_client_lic.py        # License client
├── operator_table.json         # Cellular operator data
├── requirements_tinysa.txt     # Dependencies
│
├── Training:
│   ├── train_tinySA.py         # Training script
│   └── tinySA_train.json       # Training data
│
└── tests/
    ├── conftest.py             # Test fixtures
    └── unit/test_tinysa.py     # Unit tests (29 tests)
```

---

## Troubleshooting

### "tinySA device not found"

1. Check USB connection
2. Verify in Device Manager (Windows) or `lsusb` (Linux)
3. Install CH340 USB driver if needed
4. Try different USB port

### "SLM model not loaded"

1. Run training first: `python train_tinySA.py`
2. Check `tinyllama_tinysa_lora/` directory exists
3. Ensure 4GB+ RAM available
4. Try CPU mode if GPU fails

### "API key invalid"

1. Create `.env` file with `OPENAI_API_KEY=sk-...`
2. Check key hasn't expired
3. Verify API credits available

### "Hebrew/Chinese not translating"

- Hebrew uses code `iw` (not `he`)
- Chinese uses `zh-CN` (case sensitive)
- Requires internet for translation

### "Dimension mismatch error"

- tinySA may have returned empty data
- Check device connection
- Verify frequency range is valid

---

## Hardware Requirements

### Minimum
- CPU: Dual-core 2GHz
- RAM: 4GB
- Storage: 500MB
- USB 2.0 port

### Recommended (for SLM)
- CPU: Quad-core 3GHz
- RAM: 8GB
- GPU: NVIDIA with 4GB+ VRAM (optional)
- Storage: 5GB

---

## Dependencies

```
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
pyserial>=3.5
openai>=1.0.0
deep-translator>=1.11.4
pywifi>=1.1.12
Pillow>=10.0.0
torch>=2.0.0 (for SLM)
transformers>=4.30.0 (for SLM)
peft>=0.5.0 (for SLM)
```

---

## License

Ennoia Technologies Connect Platform © All rights reserved.

---

## Support

GitHub Issues: https://github.com/rajagopalsridhar/ennoiaCAT/issues
