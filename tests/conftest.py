"""
Pytest configuration and shared fixtures for EnnoiaCAT tests
"""
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_serial_port():
    """Mock serial port for testing"""
    mock_port = Mock()
    mock_port.device = "COM3"
    mock_port.vid = 0x0483
    mock_port.pid = 0x5740
    mock_port.description = "TinySA Spectrum Analyzer"
    return mock_port


@pytest.fixture
def mock_tinysa_device():
    """Mock TinySA device info"""
    return {
        'port': 'COM3',
        'vid': 0x0483,
        'pid': 0x5740,
        'description': 'TinySA Spectrum Analyzer'
    }


@pytest.fixture
def mock_viavi_device():
    """Mock Viavi device info"""
    return {
        'ip': '192.168.1.100',
        'protocol': 'SCPI',
        'port': 5025
    }


@pytest.fixture
def mock_keysight_device():
    """Mock Keysight device info"""
    return {
        'ip': '192.168.1.101',
        'resource': 'TCPIP0::192.168.1.101::inst0::INSTR',
        'idn': 'Keysight Technologies,N9918A,MY12345678,A.01.23'
    }


@pytest.fixture
def mock_rohde_schwarz_device():
    """Mock Rohde & Schwarz device info"""
    return {
        'resource': 'TCPIP0::192.168.1.102::hislip0::INSTR',
        'idn': 'Rohde&Schwarz,NRQ6,123456,1.0.0'
    }


@pytest.fixture
def mock_mavenir_device():
    """Mock Mavenir RU device info"""
    return {
        'ip': '10.10.10.10',
        'protocol': 'NETCONF',
        'port': 830
    }


@pytest.fixture
def mock_cisco_device():
    """Mock Cisco NCS540 device info"""
    return {
        'port': 'COM4',
        'vid': 0x0403,
        'pid': 0x6001,
        'description': 'USB Serial Converter'
    }


@pytest.fixture
def mock_aukua_device():
    """Mock Aukua device info"""
    return {
        'resource': 'TCPIP0::192.168.1.103::inst0::INSTR',
        'idn': 'AUKUA,System-100,SN123456,1.0.0'
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="This is a test response from the LLM model."
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_transformers_model():
    """Mock transformers model for SLM testing"""
    mock_model = MagicMock()
    mock_model.generate.return_value = [[1, 2, 3, 4, 5]]  # Mock token IDs
    return mock_model


@pytest.fixture
def mock_transformers_tokenizer():
    """Mock transformers tokenizer"""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.decode.return_value = "Generated response from SLM"
    mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]
    return mock_tokenizer


@pytest.fixture
def mock_pyvisa_resource_manager():
    """Mock PyVISA resource manager"""
    mock_rm = MagicMock()
    mock_rm.list_resources.return_value = [
        'TCPIP0::192.168.1.100::inst0::INSTR',
        'TCPIP0::192.168.1.101::hislip0::INSTR'
    ]

    mock_inst = MagicMock()
    mock_inst.query.return_value = "Keysight Technologies,N9918A,MY12345678,A.01.23"
    mock_rm.open_resource.return_value = mock_inst

    return mock_rm


@pytest.fixture
def sample_spectrum_data():
    """Sample spectrum analyzer data for testing"""
    import numpy as np
    frequencies = np.linspace(100e6, 1e9, 101)  # 100 MHz to 1 GHz, 101 points
    power_levels = -80 + 20 * np.random.rand(101)  # Random power levels
    return {
        'frequencies': frequencies,
        'power_levels': power_levels
    }


@pytest.fixture
def mock_streamlit():
    """Mock streamlit for testing UI components"""
    with patch('streamlit.sidebar') as mock_sidebar, \
         patch('streamlit.title') as mock_title, \
         patch('streamlit.caption') as mock_caption, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.success') as mock_success, \
         patch('streamlit.info') as mock_info:

        yield {
            'sidebar': mock_sidebar,
            'title': mock_title,
            'caption': mock_caption,
            'error': mock_error,
            'success': mock_success,
            'info': mock_info
        }


@pytest.fixture
def env_with_openai_key(monkeypatch):
    """Set up environment with OpenAI API key"""
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test-key-12345')


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file"""
    config_file = tmp_path / "test_config.json"
    config_data = {
        "instrument": "TinySA",
        "port": "COM3",
        "settings": {
            "start_freq": 100e6,
            "stop_freq": 1e9
        }
    }
    import json
    config_file.write_text(json.dumps(config_data))
    return str(config_file)


@pytest.fixture
def temp_pcap_file(tmp_path):
    """Create a temporary PCAP file for testing"""
    pcap_file = tmp_path / "test.pcap"
    # Write minimal PCAP header (24 bytes)
    pcap_header = bytes([
        0xd4, 0xc3, 0xb2, 0xa1,  # Magic number (little endian)
        0x02, 0x00, 0x04, 0x00,  # Version major/minor
        0x00, 0x00, 0x00, 0x00,  # Timezone
        0x00, 0x00, 0x00, 0x00,  # Sigfigs
        0xff, 0xff, 0x00, 0x00,  # Snaplen
        0x01, 0x00, 0x00, 0x00   # Network (Ethernet)
    ])
    pcap_file.write_bytes(pcap_header)
    return str(pcap_file)


@pytest.fixture
def mock_flask_response():
    """Create a mock Flask response"""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "success": True,
        "evm_results": {0: -25.0, 1: -26.0, 2: -24.5, 3: -25.5},
        "interference_detected": False
    }
    response.text = "Analysis completed successfully"
    return response


@pytest.fixture
def sample_iq_data():
    """Generate sample IQ data for testing"""
    import numpy as np
    np.random.seed(42)
    num_samples = 3276  # Standard PRB count for 100MHz

    # Generate QPSK-like constellation with noise
    qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    indices = np.random.randint(0, 4, num_samples)
    iq_data = qpsk_symbols[indices]

    # Add small noise
    noise = 0.05 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    iq_data += noise

    return iq_data


@pytest.fixture
def env_production_config(monkeypatch):
    """Set up production-like environment variables"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-production-key-1234567890")
    monkeypatch.setenv("FLASK_HOST", "0.0.0.0")
    monkeypatch.setenv("FLASK_PORT", "5002")
    monkeypatch.setenv("REPLAY_CAPTURE_URL", "http://production-server:8050")
    monkeypatch.setenv("SIMON_ANALYZER_URL", "http://production-server:5002")


@pytest.fixture
def mock_requests():
    """Mock requests module for network tests"""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        yield {
            'get': mock_get,
            'post': mock_post
        }


@pytest.fixture
def sample_evm_data():
    """Generate sample EVM data across PRBs"""
    import numpy as np
    num_prbs = 273
    num_layers = 4

    # Generate baseline EVM around -25 dB
    evm_data = np.random.normal(-25, 2, (num_layers, num_prbs))

    return evm_data


@pytest.fixture
def sample_snr_data():
    """Generate sample SNR data across PRBs"""
    import numpy as np
    num_prbs = 273
    num_layers = 4

    # Generate baseline SNR around 30 dB
    snr_data = np.random.normal(30, 3, (num_layers, num_prbs))

    return snr_data
