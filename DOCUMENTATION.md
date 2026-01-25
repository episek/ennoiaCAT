# Ennoia Equipment Controller - Documentation

## Overview

Ennoia Equipment Controller is an AI-powered platform for controlling and analyzing RF test equipment using natural language commands. It supports both cloud-based LLM (OpenAI) and local SLM (TinyLlama) for offline operation.

## Supported Equipment

| Equipment | Manufacturer | Use Case |
|-----------|--------------|----------|
| tinySA | tinysa.org | Spectrum analysis (100kHz - 5.3GHz) |
| Viavi OneAdvisor | Viavi Solutions | 5G NR field testing |
| Keysight FieldFox | Keysight | RF spectrum analysis |
| Aukua XGA4250 | Aukua | Network timing analysis |
| Cisco NCS540 | Cisco | Router diagnostics |
| Rohde & Schwarz NRQ6 | R&S | Power measurements |
| ORAN PCAP Analyzer | - | O-RAN fronthaul analysis |

---

## Installation

### Prerequisites

- Python 3.10+
- Git
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/rajagopalsridhar/ennoiaCAT.git
cd ennoiaCAT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=sk-your-api-key-here
FLASK_HOST=127.0.0.1
FLASK_PORT=5002
```

---

## Running the Application

### Start the Streamlit App

```bash
streamlit run ennoiaCAT_Consolidated.py
```

The app will open at `http://localhost:8501`

### Using the App

1. **Select Language** - Choose from 11 supported languages
2. **Select Equipment** - Pick your test equipment from the dropdown
3. **Choose AI Model**:
   - **Online LLM** - Uses OpenAI GPT-4o-mini (requires internet)
   - **Local SLM** - Uses TinyLlama (offline, requires training)
4. **Enter Commands** - Type natural language commands in the chat

### Example Commands

```
"Scan the spectrum from 300 MHz to 900 MHz"
"Set the start frequency to 2.4 GHz"
"Show me the WiFi channels"
"What is the signal strength at 700 MHz?"
```

---

## Testing

### Run All Tests

```bash
# Run complete test suite
python -m pytest tests/ -v

# Run only unit tests
python -m pytest tests/unit/ -v

# Run with coverage
python -m pytest tests/ -v --cov=. --cov-report=html
```

### Test Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── unit/
│   ├── test_instrument_adapters.py
│   ├── test_instrument_detector.py
│   ├── test_llm_models.py
│   └── test_slm_models.py
└── integration/
    ├── test_llm_instrument_integration.py
    └── test_viavi_tinysa_live.py
```

---

## SLM Training Guide

### Overview

The Small Language Model (SLM) uses TinyLlama 1.1B fine-tuned with LoRA (Low-Rank Adaptation) for device-specific commands. This allows offline operation without requiring an OpenAI API key.

### Prerequisites

```bash
pip install torch transformers peft datasets bitsandbytes
```

For GPU training (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Training Data Format

The training data is stored in `tinySA_train.json` with this structure:

```json
[
  {
    "instruction": "The task or question",
    "input": "Additional context (can be empty)",
    "response": "The expected response"
  }
]
```

### Example Training Entries

```json
[
  {
    "instruction": "Set the start frequency to 300 MHz",
    "input": "",
    "response": "I am setting the start frequency to 300 MHz on the tinySA. Navigate to Frequency > Start and enter 300M."
  },
  {
    "instruction": "How do I scan for WiFi signals?",
    "input": "I want to see 2.4 GHz networks",
    "response": "To scan WiFi signals, set the frequency range from 2.4 GHz to 2.5 GHz. Set Start to 2400M and Stop to 2500M. The peaks you see represent WiFi access points on different channels."
  },
  {
    "instruction": "What is the signal strength at the marker?",
    "input": "Marker is at 850 MHz",
    "response": "The signal strength at 850 MHz is displayed next to the marker on the screen. This frequency is in the cellular band, likely showing LTE Band 5 signals."
  }
]
```

### Enriching Training Data

To make commands more natural, add variations:

#### 1. **Add Synonyms and Variations**

```json
// Formal
{"instruction": "Configure the start frequency to 100 MHz", ...}

// Casual
{"instruction": "Set start to 100 meg", ...}

// Question form
{"instruction": "How do I change the start frequency to 100 MHz?", ...}

// Command form
{"instruction": "Start frequency 100 MHz", ...}
```

#### 2. **Add Context-Rich Examples**

```json
{
  "instruction": "I want to check if there's interference on my WiFi",
  "input": "My router is on channel 6",
  "response": "To check for WiFi interference on channel 6, scan from 2.4 GHz to 2.5 GHz. Channel 6 is centered at 2437 MHz. Look for other signals near this frequency that might cause interference."
}
```

#### 3. **Add Error Handling Examples**

```json
{
  "instruction": "Set frequency to 10 GHz",
  "input": "",
  "response": "The tinySA Ultra supports frequencies up to 5.3 GHz. Please set a frequency within the supported range (100 kHz to 5.3 GHz)."
}
```

#### 4. **Add Multi-Step Procedures**

```json
{
  "instruction": "How do I measure the bandwidth of a signal?",
  "input": "I see a peak at 433 MHz",
  "response": "To measure bandwidth: 1) Place marker on the peak at 433 MHz. 2) Use the marker delta function to find the -3dB points. 3) The bandwidth is the difference between the two -3dB frequencies. For ISM band signals at 433 MHz, typical bandwidth is 1.7 MHz."
}
```

### Running Training

```bash
# Ensure you have the base model downloaded
# (happens automatically on first run)

# Run training
python train_tinySA.py
```

Training output:
- `tinyllama_tinysa_lora/adapter_model.safetensors` - Fine-tuned weights
- `tinyllama_tinysa_lora/adapter_config.json` - LoRA configuration
- `tinyllama_tinysa_lora/final_eval_results.json` - Training metrics

### Training Parameters

Edit `train_tinySA.py` to adjust:

```python
# Number of training epochs (more = better learning, but slower)
num_train_epochs=3

# Learning rate (lower = more stable, higher = faster learning)
learning_rate=2e-4

# Batch size (higher = faster, but needs more memory)
per_device_train_batch_size=4

# LoRA rank (higher = more capacity, but larger model)
r=8
lora_alpha=16
```

### Evaluating the Model

After training, check `final_eval_results.json`:

```json
{
  "eval_loss": 1.23,  // Lower is better
  "eval_accuracy": 0.85
}
```

| Loss | Quality |
|------|---------|
| < 1.0 | Excellent |
| 1.0 - 2.0 | Good |
| 2.0 - 3.0 | Needs improvement |
| > 3.0 | Poor - review data |

### Tips for Better Training

1. **More data is better** - Aim for 100+ examples
2. **Diverse phrasing** - Same command, different words
3. **Consistent format** - Keep response style uniform
4. **Domain-specific** - Focus on your equipment's terminology
5. **Real scenarios** - Use actual user questions as training data

---

## Project Structure

```
ennoiaCAT/
├── ennoiaCAT_Consolidated.py    # Main Streamlit application
├── map_api.py                   # LLM API wrapper
├── timer.py                     # Timing utilities
├── ennoia_client_lic.py         # License management
│
├── Equipment Configs:
│   ├── tinySA.py, tinySA_config.py
│   ├── viavi_config.py, ennoia_viavi/
│   ├── AK_config.py (Aukua)
│   ├── CS_config.py (Cisco)
│   ├── RS_config.py (Rohde & Schwarz)
│   └── ORAN_config.py
│
├── Training:
│   ├── train_tinySA.py          # SLM training script
│   └── tinySA_train.json        # Training data
│
├── Data:
│   └── operator_table.json      # Cellular operator frequencies
│
└── Tests:
    └── tests/
```

---

## Troubleshooting

### "Device not found"
- Ensure equipment is connected via USB
- Check USB drivers are installed
- Verify device shows in Device Manager

### "API key invalid"
- Check `.env` file has correct `OPENAI_API_KEY`
- Verify API key has not expired
- Ensure you have API credits

### "SLM model not loaded"
- Run `python train_tinySA.py` first
- Check `tinyllama_tinysa_lora/` directory exists
- Ensure enough RAM (4GB+) or GPU memory

### "Translation not working"
- Hebrew uses code `iw` (not `he`)
- Chinese uses `zh-CN` (case sensitive)
- Check internet connection for translation API

---

## License

Ennoia Technologies Connect Platform © All rights reserved.

---

## Support

- GitHub Issues: https://github.com/rajagopalsridhar/ennoiaCAT/issues
- Documentation: This file
