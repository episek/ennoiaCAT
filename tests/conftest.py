"""
Pytest configuration and shared fixtures for Ennoia tinySA tests
"""
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_serial_port():
    """Mock serial port for tinySA testing"""
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
def sample_csv_data(tmp_path):
    """Create a sample signal strength CSV file"""
    csv_file = tmp_path / "max_signal_strengths.csv"
    csv_content = """frequency,signal_strength
300000000,-50.2
400000000,-45.1
500000000,-48.7
600000000,-52.3
700000000,-47.8
800000000,-49.5
900000000,-51.1
"""
    csv_file.write_text(csv_content)
    return str(csv_file)


@pytest.fixture
def mock_serial_connection():
    """Mock serial connection for tinySA"""
    mock_serial = MagicMock()
    mock_serial.write = MagicMock()
    mock_serial.read = MagicMock(return_value=b'\n')
    mock_serial.readline = MagicMock(return_value=b'-50.2\n')
    mock_serial.close = MagicMock()
    return mock_serial


@pytest.fixture
def sample_operator_table():
    """Sample operator frequency table"""
    return [
        {
            "operator": "Carrier A",
            "band": "Band 7",
            "start_mhz": 2620,
            "end_mhz": 2690,
            "technology": "LTE"
        },
        {
            "operator": "Carrier B",
            "band": "Band 3",
            "start_mhz": 1805,
            "end_mhz": 1880,
            "technology": "LTE"
        },
        {
            "operator": "Carrier C",
            "band": "n78",
            "start_mhz": 3300,
            "end_mhz": 3800,
            "technology": "5G NR"
        }
    ]


@pytest.fixture
def mock_wifi_scan_results():
    """Mock WiFi scan results"""
    return [
        {
            "SSID": "TestNetwork1",
            "Signal (dBm)": -45,
            "Frequency (MHz)": 2437,
            "Channel": 6,
            "Band": "2.4 GHz"
        },
        {
            "SSID": "TestNetwork2",
            "Signal (dBm)": -55,
            "Frequency (MHz)": 5180,
            "Channel": 36,
            "Band": "5 GHz"
        }
    ]
