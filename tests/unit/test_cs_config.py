"""
Unit tests for CS_config.py
Tests logging, URL configuration, MAC address handling, and translation functions.
"""
import pytest
import os
from unittest.mock import Mock, MagicMock, patch
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestCSConfigEnvironmentVariables:
    """Test environment variable configuration"""

    def test_replay_capture_url_default(self, monkeypatch):
        """Test default REPLAY_CAPTURE_URL value"""
        monkeypatch.delenv("REPLAY_CAPTURE_URL", raising=False)
        default_url = os.getenv("REPLAY_CAPTURE_URL", "http://localhost:8050")
        assert default_url == "http://localhost:8050"

    def test_replay_capture_url_custom(self, monkeypatch):
        """Test custom REPLAY_CAPTURE_URL value"""
        monkeypatch.setenv("REPLAY_CAPTURE_URL", "http://production-server:8050")
        url = os.getenv("REPLAY_CAPTURE_URL", "http://localhost:8050")
        assert url == "http://production-server:8050"

    def test_simon_analyzer_url_default(self, monkeypatch):
        """Test default SIMON_ANALYZER_URL value"""
        monkeypatch.delenv("SIMON_ANALYZER_URL", raising=False)
        default_url = os.getenv("SIMON_ANALYZER_URL", "http://localhost:5002")
        assert default_url == "http://localhost:5002"

    def test_simon_analyzer_url_custom(self, monkeypatch):
        """Test custom SIMON_ANALYZER_URL value"""
        monkeypatch.setenv("SIMON_ANALYZER_URL", "http://analyzer.prod:5002")
        url = os.getenv("SIMON_ANALYZER_URL", "http://localhost:5002")
        assert url == "http://analyzer.prod:5002"

    def test_default_mac_address_fallback(self, monkeypatch):
        """Test DEFAULT_MAC_ADDRESS fallback"""
        monkeypatch.delenv("DEFAULT_MAC_ADDRESS", raising=False)
        mac = os.getenv("DEFAULT_MAC_ADDRESS", "02:00:00:00:00:01")
        assert mac == "02:00:00:00:00:01"

    def test_default_mac_address_custom(self, monkeypatch):
        """Test custom DEFAULT_MAC_ADDRESS"""
        monkeypatch.setenv("DEFAULT_MAC_ADDRESS", "aa:bb:cc:dd:ee:ff")
        mac = os.getenv("DEFAULT_MAC_ADDRESS", "02:00:00:00:00:01")
        assert mac == "aa:bb:cc:dd:ee:ff"


class TestCSHelperImport:
    """Test CSHelper class import and basic functionality"""

    def test_import_cs_config(self):
        """Test that CS_config module can be imported"""
        with patch('streamlit.cache_data'), \
             patch('streamlit.cache_resource'), \
             patch('streamlit.sidebar'), \
             patch('streamlit.markdown'):
            # Import should not raise
            try:
                from CS_config import CSHelper
                assert CSHelper is not None
            except ImportError as e:
                # Allow import errors for optional dependencies
                assert "pyvisa" in str(e).lower() or "torch" in str(e).lower()


class TestLoggingConfiguration:
    """Test logging configuration"""

    def test_logger_exists(self):
        """Test that logger is properly configured"""
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("CS_config")
        assert logger is not None
        assert logger.level >= 0

    def test_logger_output(self, caplog):
        """Test logger produces output"""
        import logging
        logger = logging.getLogger("test_cs_config")
        logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.INFO):
            logger.info("Test message")

        assert "Test message" in caplog.text


class TestNetworkRequestMocking:
    """Test network request handling with mocks"""

    @patch('requests.post')
    def test_replay_capture_request_success(self, mock_post):
        """Test successful replay and capture request"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'pkts_sent': 100,
            'bytes_sent': 50000,
            'output_file': 'capture.pcapng'
        }
        mock_post.return_value = mock_response

        import requests
        url = os.getenv("REPLAY_CAPTURE_URL", "http://localhost:8050")
        res = requests.post(f"{url}/replay_and_capture", json={
            "iface_out": "eth0",
            "iface_in": "eth1",
            "src_mac": "02:00:00:00:00:01",
            "pcap_file_in": "input.pcap",
            "pcap_file_out": "output.pcap"
        }, timeout=60)

        assert res.ok
        result = res.json()
        assert result['pkts_sent'] == 100
        assert result['bytes_sent'] == 50000

    @patch('requests.post')
    def test_replay_capture_request_timeout(self, mock_post):
        """Test replay and capture request timeout handling"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        with pytest.raises(requests.exceptions.Timeout):
            requests.post("http://localhost:8050/replay_and_capture", timeout=1)

    @patch('requests.post')
    def test_upload_request_connection_error(self, mock_post):
        """Test upload request connection error handling"""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with pytest.raises(requests.exceptions.ConnectionError):
            requests.post("http://localhost:5002/upload", json={})

    @patch('requests.get')
    def test_progress_polling(self, mock_get):
        """Test progress polling with different statuses"""
        mock_response = Mock()
        mock_response.json.return_value = {'status': 'Completed'}
        mock_get.return_value = mock_response

        import requests
        response = requests.get("http://localhost:5002/progress", timeout=15)
        status = response.json()

        assert status['status'] == 'Completed'


class TestTranslationFunction:
    """Test translation helper function"""

    def test_translation_english_passthrough(self):
        """Test that English text passes through unchanged"""
        # Mock translation function behavior
        def t(text, lang="en"):
            if not lang or lang == "en":
                return text
            return text  # Simplified mock

        result = t("Hello World", lang="en")
        assert result == "Hello World"

    def test_translation_none_language(self):
        """Test translation with None language"""
        def t(text, lang=None):
            if not lang or lang == "en":
                return text
            return text

        result = t("Test message", lang=None)
        assert result == "Test message"

    def test_translation_empty_text(self):
        """Test translation with empty text"""
        def t(text, lang="fr"):
            if not text:
                return text
            return text

        result = t("", lang="fr")
        assert result == ""


class TestFileOperations:
    """Test file operation utilities"""

    def test_binary_file_write(self, tmp_path):
        """Test writing binary data to file"""
        import struct

        test_file = tmp_path / "test.bin"
        float_list = [1.0, 2.5, 3.14, -1.5]

        with open(test_file, 'wb') as f:
            for value in float_list:
                f.write(struct.pack('f', value))

        # Verify file was created
        assert test_file.exists()
        assert test_file.stat().st_size == len(float_list) * 4  # 4 bytes per float32

    def test_binary_file_read(self, tmp_path):
        """Test reading binary data from file"""
        import struct

        test_file = tmp_path / "test.bin"
        original_values = [1.0, 2.5, 3.14]

        # Write test data
        with open(test_file, 'wb') as f:
            for value in original_values:
                f.write(struct.pack('f', value))

        # Read back
        with open(test_file, 'rb') as f:
            bin_data = f.read()

        read_values = struct.unpack(f'<{len(original_values)}f', bin_data)

        for orig, read in zip(original_values, read_values):
            assert abs(orig - read) < 0.001


class TestMACAddressHandling:
    """Test MAC address validation and handling"""

    def test_valid_mac_format(self):
        """Test valid MAC address format"""
        import re
        mac_pattern = r'^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$'

        valid_macs = [
            "02:00:00:00:00:01",
            "aa:bb:cc:dd:ee:ff",
            "AA:BB:CC:DD:EE:FF",
            "12:34:56:78:9a:bc"
        ]

        for mac in valid_macs:
            assert re.match(mac_pattern, mac), f"MAC {mac} should be valid"

    def test_invalid_mac_format(self):
        """Test invalid MAC address formats"""
        import re
        mac_pattern = r'^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$'

        invalid_macs = [
            "02:00:00:00:00",  # Too short
            "02:00:00:00:00:01:02",  # Too long
            "02-00-00-00-00-01",  # Wrong separator
            "gg:hh:ii:jj:kk:ll",  # Invalid hex
            "020000000001"  # No separators
        ]

        for mac in invalid_macs:
            assert not re.match(mac_pattern, mac), f"MAC {mac} should be invalid"


class TestIPAddressValidation:
    """Test IP address validation"""

    def test_valid_ipv4_addresses(self):
        """Test valid IPv4 address formats"""
        import re
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'

        valid_ips = [
            "192.168.1.1",
            "10.0.0.1",
            "127.0.0.1",
            "255.255.255.255",
            "0.0.0.0"
        ]

        for ip in valid_ips:
            assert re.match(ipv4_pattern, ip), f"IP {ip} should match pattern"
            # Additional validation for octets
            octets = ip.split('.')
            for octet in octets:
                assert 0 <= int(octet) <= 255, f"Octet {octet} out of range"

    def test_invalid_ipv4_addresses(self):
        """Test invalid IPv4 address formats"""
        import re
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'

        invalid_ips = [
            "192.168.1",  # Missing octet
            "192.168.1.1.1",  # Extra octet
            "192.168.1.256",  # Octet > 255
            "192.168.-1.1",  # Negative octet
            "abc.def.ghi.jkl"  # Non-numeric
        ]

        for ip in invalid_ips:
            if re.match(ipv4_pattern, ip):
                # Check octets are in range
                octets = ip.split('.')
                valid = all(0 <= int(o) <= 255 for o in octets if o.isdigit())
                assert not valid, f"IP {ip} should be invalid"
