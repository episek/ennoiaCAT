# EnnoiaCAT Unified - Multi-Instrument Test Platform

## Overview

EnnoiaCAT Unified is an intelligent test platform that automatically detects and supports multiple test instruments through a single, unified interface. The system uses AI assistance to help users interact with their test equipment using natural language.

## Supported Instruments

The platform supports the following instrument types:

### 1. **TinySA Spectrum Analyzer**
- **Connection**: USB (Auto-detected via VID:0x0483, PID:0x5740)
- **Config Module**: `tinySA_config.py`
- **Features**: RF spectrum analysis, signal strength measurement

### 2. **Viavi OneAdvisor**
- **Connection**: Network (IP-based, default: 192.168.1.100)
- **Config Module**: `ennoia_viavi/system_api.py`
- **Features**: Professional RF spectrum analysis, cellular testing
- **Network Setup Required**:
  - **IMPORTANT**: Set your laptop's IP address to **192.168.1.100** on the network interface connected to the Viavi device
  - Ensure laptop and Viavi are on the same network subnet
  - Disable any VPN or firewall that might block SCPI communication on port 5025

### 3. **Mavenir 5G NR Radio Unit**
- **Connection**: Network (NETCONF, default: 10.10.10.10)
- **Config Module**: `mav_config.py`
- **Features**: 5G NR conformance testing, RU health monitoring

### 4. **Cisco NCS540**
- **Connection**: Serial (USB-to-Serial adapters)
- **Config Module**: `ncs540_serial.py`
- **Features**: Network equipment testing, CLI access

### 5. **Keysight FieldFox**
- **Connection**: Network (VISA/SCPI, default: 192.168.1.100)
- **Config Module**: `KS_config.py`
- **Features**: Spectrum analysis, network analysis

### 6. **Rohde & Schwarz NRQ6**
- **Connection**: Network (VISA/HiSLIP)
- **Config Module**: `RS_config.py`
- **Features**: Power measurement, IQ capture

### 7. **Aukua Systems**
- **Connection**: Network (VISA)
- **Config Module**: `AK_config.py`
- **Features**: Specialized test equipment

## Architecture

### Core Components

```
ennoiaCAT_unified.py          # Main application
├── instrument_detector.py     # Auto-detection logic
├── instrument_adapters.py     # Unified instrument interface
└── Individual config modules  # Instrument-specific implementations
```

### How It Works

1. **Detection**: The system scans for connected instruments via:
   - USB ports (for TinySA, Cisco serial)
   - Network ports (for IP-based instruments)
   - VISA resources (for SCPI instruments)

2. **Selection**: User selects from detected instruments via dropdown

3. **Connection**: Appropriate adapter initializes the instrument

4. **Operation**: Unified AI interface allows natural language interaction

## Installation

### Prerequisites

```bash
# Python packages
pip install streamlit pandas numpy matplotlib
pip install pyserial pyvisa pyvisa-py
pip install openai
pip install ennoia_client_lic

# Optional: For specific instruments
pip install RsInstrument  # Rohde & Schwarz
pip install pywifi         # WiFi scanning
pip install deep-translator # Multi-language support
```

### VISA Installation

For network instruments (Keysight, R&S, etc.):
- Install [NI-VISA](https://www.ni.com/en-us/support/downloads/drivers/download.ni-visa.html) or
- Install [Keysight IO Libraries](https://www.keysight.com/us/en/lib/software-detail/computer-software/io-libraries-suite-downloads-2175637.html)

## Usage

### Running the Unified Application

```bash
streamlit run ennoiaCAT_unified.py
```

### Workflow

1. **License Verification** (automatic on startup)
   ```bash
   # Activate license (first time)
   python ennoiaCAT_unified.py --action activate --key YOUR_LICENSE_KEY

   # Verify license (default)
   python ennoiaCAT_unified.py --action verify
   ```

2. **Configure Network IPs** (optional, in sidebar)
   - Expand "Network Instrument Configuration"
   - Enter IP addresses for network-based instruments
   - Default IPs are pre-filled

3. **Detect Instruments**
   - Click "🔍 Detect Instruments" button
   - System scans all connection types
   - Displays found instruments

4. **Select & Connect**
   - Choose instrument from dropdown
   - Click "🔌 Connect to Instrument"
   - Instrument-specific UI loads

5. **Configure AI** (in sidebar)
   - Choose "SLM (Offline)" for local AI model
   - Choose "Online LLM" for OpenAI GPT
   - Or select both

6. **Interact**
   - Type natural language queries in chat
   - AI assistant responds and controls instrument
   - Results displayed in real-time

### Example Interactions

```
User: "Scan from 300 MHz to 900 MHz"
AI: "I'll configure the spectrum analyzer to scan from 300 MHz to 900 MHz..."
[Instrument performs scan and displays results]

User: "What's the peak frequency?"
AI: "The peak signal is at 433.92 MHz with -45 dBm power"

User: "Show me the signal strength"
[AI displays signal strength chart and analysis]
```

## Instrument-Specific Features

### TinySA
- Frequency scanning
- Signal strength analysis
- Operator frequency mapping
- CSV export

### Viavi OneAdvisor
- Advanced spectrum analysis
- 5G NR carrier detection
- WiFi analysis
- Waterfall displays
- Multi-sweep averaging

### Mavenir 5G NR RU
- RU health monitoring
- RF performance metrics
- PTP timing analysis
- Conformance test campaigns
- O-RAN plane monitoring

### Cisco NCS540
- CLI command execution
- Bridge domain configuration
- PCAP analysis
- Serial console access

### Keysight FieldFox
- Spectrum analyzer mode
- Network analyzer mode
- SCPI command interface

### Rohde & Schwarz NRQ6
- Power measurement
- IQ data capture
- Multi-language support

## Configuration Files

Each instrument type requires its own configuration module:

- `tinySA_config.py` - TinySA helper
- `ennoia_viavi/` - Viavi system & radio APIs
- `mav_config.py` - Mavenir configuration
- `CS_config.py` - Cisco configuration
- `RS_config.py` - Rohde & Schwarz configuration
- `AK_config.py` - Aukua configuration
- `KS_config.py` - Keysight configuration

## Troubleshooting

### No Instruments Detected

**USB Instruments (TinySA, Cisco)**:
- Check USB connection
- Verify device drivers installed
- Check Device Manager (Windows) or `lsusb` (Linux)

**Network Instruments**:
- **Configure your laptop's IP address first** (especially for Viavi):
  - Viavi requires laptop IP: 192.168.1.100
  - Keysight typically uses: 192.168.1.x range
  - Mavenir uses: 10.10.10.x range
- Verify IP address configuration
- Check network connectivity (`ping <ip>`)
- Ensure firewall allows connections (SCPI port 5025 for Viavi)
- Verify instrument is powered on
- Disable VPN if connection fails

**VISA Instruments**:
- Install VISA runtime (NI-VISA or Keysight IO)
- Run `pyvisa-info` to check installation
- Test with NI MAX or Keysight Connection Expert

### Connection Fails

1. **Check instrument power and cables**
2. **Verify IP addresses** in network configuration
3. **Check firewall settings**
4. **Ensure VISA backends** are properly installed
5. **Review instrument manuals** for connection requirements

### AI Features Not Working

**Offline SLM**:
- Ensure model files are in correct location
- Check PyTorch installation
- Verify sufficient RAM/VRAM

**Online LLM**:
- Set `OPENAI_API_KEY` environment variable
- Verify API key is valid
- Check internet connectivity

## Migration from Legacy Files

If you were using individual `ennoiaCAT_*.py` files:

1. **Keep your existing files** - they still work independently
2. **Use `ennoiaCAT_unified.py`** for multi-instrument setups
3. **All configuration modules** are compatible
4. **No data loss** - same underlying functionality

## File Structure

```
ennoiaCAT/
├── ennoiaCAT_unified.py          # New unified application
├── instrument_detector.py         # Auto-detection system
├── instrument_adapters.py         # Adapter pattern implementation
│
├── Legacy individual files:
├── ennoiaCAT_RAG_INT_LICt.py     # TinySA baseline
├── ennoiaCAT_VI.py                # Viavi
├── ennoiaCAT_MAV_RCT.py           # Mavenir
├── ennoiaCAT_CIS.py               # Cisco
├── ennoiaCAT_RS.py                # Rohde & Schwarz
├── ennoiaCAT_KS.py                # Keysight
├── ennoiaCAT_AUK.py               # Aukua
│
├── Configuration modules:
├── tinySA_config.py
├── CS_config.py
├── RS_config.py
├── AK_config.py
├── KS_config.py
├── mav_config.py
├── map_api.py
│
└── Support modules:
    ├── ennoia_client_lic.py
    ├── timer.py
    ├── pcap_utils.py
    └── ...
```

## Advanced Features

### Custom IP Ranges

To scan multiple IPs:

```python
# Edit ennoiaCAT_unified.py
viavi_ips = ["192.168.1.100", "192.168.1.101", "10.0.0.20"]
keysight_ips = ["192.168.2.100", "192.168.2.101"]
```

### Adding New Instruments

1. Create config module: `myinstrument_config.py`
2. Add to `InstrumentType` enum in `instrument_detector.py`
3. Create detection method in `InstrumentDetector`
4. Create adapter class in `instrument_adapters.py`
5. Add to `AdapterFactory`

## Support

For issues or questions:
- Check instrument manuals for connection details
- Verify all dependencies are installed
- Review logs for error messages
- Contact Ennoia Technologies support

## License

This software requires a valid Ennoia license.
© Ennoia Technologies. All rights reserved.
