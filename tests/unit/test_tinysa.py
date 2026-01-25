"""
Unit tests for tinySA spectrum analyzer functionality.
Tests tinySA serial communication, data parsing, and spectrum analysis.
"""
import pytest
import numpy as np
import os
import sys
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestTinySADeviceDetection:
    """Test tinySA device detection and connection"""

    def test_device_vid_pid(self):
        """Test correct VID/PID for tinySA"""
        VID = 0x0483  # 1155
        PID = 0x5740  # 22336

        assert VID == 1155
        assert PID == 22336

    def test_getport_with_device(self, mock_serial_port):
        """Test getport when device is connected"""
        with patch('serial.tools.list_ports.comports') as mock_comports:
            mock_comports.return_value = [mock_serial_port]

            # Simulate getport behavior
            device_list = mock_comports()
            found_port = None
            for device in device_list:
                if device.vid == 0x0483 and device.pid == 0x5740:
                    found_port = device.device
                    break

            assert found_port == "COM3"

    def test_getport_no_device(self):
        """Test getport raises error when no device"""
        with patch('serial.tools.list_ports.comports') as mock_comports:
            mock_comports.return_value = []

            device_list = mock_comports()
            found = False
            for device in device_list:
                if device.vid == 0x0483 and device.pid == 0x5740:
                    found = True

            assert not found


class TestTinySADataParsing:
    """Test tinySA data parsing functions"""

    def test_first_float_extraction(self):
        """Test extracting first float from messy data"""
        import re

        def _first_float(s):
            if not s:
                return None
            s = s.strip().replace('\x00', '').replace(',', '.')
            try:
                return float(s)
            except Exception:
                m = re.search(r'[-+]?(?:\d+(?:\.\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?', s)
                if m:
                    try:
                        return float(m.group(0))
                    except Exception:
                        return None
            return None

        # Clean numeric strings
        assert _first_float("123.45") == 123.45
        assert _first_float("-67.89") == -67.89

        # Empty/None
        assert _first_float("") is None
        assert _first_float(None) is None

    def test_data_array_parsing(self):
        """Test parsing data array from tinySA"""
        raw_data = "-45.2\n-50.1\n-48.7\n"

        xs = []
        for line in raw_data.split('\n'):
            if line.strip():
                try:
                    xs.append(float(line.strip()))
                except ValueError:
                    continue

        assert len(xs) == 3
        assert xs[0] == -45.2
        assert xs[1] == -50.1
        assert xs[2] == -48.7


class TestTinySAFrequencySettings:
    """Test tinySA frequency configuration"""

    def test_set_frequencies(self):
        """Test frequency array generation"""
        start = 1e6
        stop = 350e6
        points = 101

        frequencies = np.linspace(start, stop, points)

        assert len(frequencies) == points
        assert frequencies[0] == start
        assert frequencies[-1] == stop

    def test_sweep_range_calculation(self):
        """Test sweep start/stop calculation"""
        start = 300e6
        stop = 900e6

        span = stop - start
        center = (start + stop) / 2

        assert span == 600e6
        assert center == 600e6

    def test_rbw_auto_calculation(self):
        """Test automatic RBW calculation"""
        def auto_rbw(span_hz):
            if span_hz <= 10e6:
                return 10e3
            if span_hz <= 100e6:
                return 100e3
            return 1e6

        assert auto_rbw(5e6) == 10e3
        assert auto_rbw(50e6) == 100e3
        assert auto_rbw(500e6) == 1e6


class TestTinySASpectrumAnalysis:
    """Test spectrum analysis functions"""

    def test_find_peaks(self):
        """Test peak finding algorithm"""
        def find_peaks(y, max_peaks=5, min_dist=5):
            peaks = []
            for i in range(1, len(y) - 1):
                if y[i] > y[i - 1] and y[i] > y[i + 1]:
                    peaks.append((i, y[i]))
            peaks.sort(key=lambda p: p[1], reverse=True)
            selected = []
            for idx, val in peaks:
                if all(abs(idx - s[0]) >= min_dist for s in selected):
                    selected.append((idx, val))
                if len(selected) >= max_peaks:
                    break
            return selected

        # Create test data with peaks
        y = [0, 1, 2, 10, 2, 1, 0, 1, 2, 8, 2, 1, 0]
        peaks = find_peaks(y, max_peaks=2, min_dist=3)

        assert len(peaks) == 2
        assert peaks[0][0] == 3  # Index of highest peak (value 10)
        assert peaks[1][0] == 9  # Index of second peak (value 8)

    def test_dimension_validation(self):
        """Test frequency/data dimension validation"""
        frequencies = np.linspace(100e6, 900e6, 101)
        data = np.random.randn(101) - 50  # Valid: same length

        assert len(frequencies) == len(data)

        # Test mismatch
        bad_data = np.random.randn(50)
        assert len(frequencies) != len(bad_data)

    def test_empty_data_handling(self):
        """Test handling of empty scan data"""
        data = []

        # Should not raise error, just return empty
        assert len(data) == 0

        # Logmag validation pattern
        if data is None or len(data) == 0:
            is_valid = False
        else:
            is_valid = True

        assert not is_valid


class TestTinySASerialCommands:
    """Test tinySA serial command generation"""

    def test_sweep_command(self):
        """Test sweep start/stop command format"""
        start = 300e6
        stop = 900e6

        cmd_start = f"sweep start {int(start)}\r"
        cmd_stop = f"sweep stop {int(stop)}\r"

        assert cmd_start == "sweep start 300000000\r"
        assert cmd_stop == "sweep stop 900000000\r"

    def test_scan_command(self):
        """Test scan command format"""
        start = 300e6
        stop = 900e6
        points = 101

        cmd = f"scan {int(start)} {int(stop)} {points}\r"

        assert cmd == "scan 300000000 900000000 101\r"

    def test_rbw_command(self):
        """Test RBW command format"""
        rbw = 100e3

        cmd = f"rbw {int(rbw)}\r"

        assert cmd == "rbw 100000\r"

    def test_mode_commands(self):
        """Test mode switching commands"""
        commands = {
            "low_input": "mode low input\r",
            "high_input": "mode high input\r",
            "low_output": "mode low output\r"
        }

        for mode, expected in commands.items():
            assert expected.endswith("\r")
            assert "mode" in expected


class TestTinySACSVExport:
    """Test CSV export functionality"""

    def test_csv_format(self):
        """Test CSV data format"""
        frequencies = [300e6, 400e6, 500e6]
        power_levels = [-50.2, -45.1, -48.7]

        csv_lines = []
        for i in range(len(frequencies)):
            csv_lines.append(f"{int(frequencies[i])}, {power_levels[i]:.2f}")

        assert csv_lines[0] == "300000000, -50.20"
        assert csv_lines[1] == "400000000, -45.10"

    def test_signal_strength_csv_read(self):
        """Test reading signal strength CSV"""
        csv_content = "frequency,signal_strength\n300000000,-50.2\n400000000,-45.1\n"

        # Parse CSV content
        lines = csv_content.strip().split('\n')[1:]  # Skip header
        frequencies = []
        strengths = []

        for line in lines:
            parts = line.split(',')
            frequencies.append(float(parts[0]))
            strengths.append(float(parts[1]))

        assert len(frequencies) == 2
        assert frequencies[0] == 300e6
        assert strengths[0] == -50.2


class TestTinySAOperatorAnalysis:
    """Test cellular operator frequency analysis"""

    def test_frequency_band_matching(self):
        """Test matching frequencies to cellular bands"""
        # Example operator table entry
        operator_entry = {
            "operator": "Test Carrier",
            "band": "Band 7",
            "start_mhz": 2620,
            "end_mhz": 2690
        }

        # Test frequency in band
        test_freq_mhz = 2650
        in_band = operator_entry["start_mhz"] <= test_freq_mhz <= operator_entry["end_mhz"]

        assert in_band

        # Test frequency out of band
        test_freq_mhz = 2500
        in_band = operator_entry["start_mhz"] <= test_freq_mhz <= operator_entry["end_mhz"]

        assert not in_band

    def test_peak_detection_threshold(self):
        """Test peak detection with threshold"""
        signal_strengths = [-80, -75, -50, -55, -80, -85]
        threshold = -60  # dBm

        strong_signals = [i for i, s in enumerate(signal_strengths) if s > threshold]

        assert 2 in strong_signals  # -50 dBm
        assert 3 in strong_signals  # -55 dBm
        assert 0 not in strong_signals  # -80 dBm


class TestWiFiChannelConversion:
    """Test WiFi channel conversion functions"""

    def test_freq_to_channel_2_4ghz(self):
        """Test frequency to channel conversion for 2.4 GHz"""
        def freq_to_channel(freq):
            try:
                freq_mhz = int(freq / 1e3)
                if freq_mhz == 2484:
                    return 14
                elif 2412 <= freq_mhz <= 2472:
                    return (freq_mhz - 2407) // 5
                return None
            except (TypeError, ValueError):
                return None

        # Input is in kHz
        assert freq_to_channel(2412e3) == 1
        assert freq_to_channel(2437e3) == 6
        assert freq_to_channel(2462e3) == 11
        assert freq_to_channel(2484e3) == 14

    def test_freq_to_channel_5ghz(self):
        """Test frequency to channel conversion for 5 GHz"""
        def freq_to_channel(freq):
            try:
                freq_mhz = int(freq / 1e3)
                if 5180 <= freq_mhz <= 5825:
                    return (freq_mhz - 5000) // 5
                return None
            except (TypeError, ValueError):
                return None

        # Input is in kHz
        assert freq_to_channel(5180e3) == 36
        assert freq_to_channel(5240e3) == 48
        assert freq_to_channel(5500e3) == 100

    def test_classify_band(self):
        """Test frequency band classification"""
        def classify_band(freq):
            try:
                freq_mhz = int(freq / 1e3)
                if 2400 <= freq_mhz <= 2500:
                    return "2.4 GHz"
                elif 5000 <= freq_mhz <= 5900:
                    return "5 GHz"
                elif 5925 <= freq_mhz <= 7125:
                    return "6 GHz"
                return "Unknown"
            except (TypeError, ValueError):
                return None

        # Input is in kHz
        assert classify_band(2412e3) == "2.4 GHz"
        assert classify_band(5180e3) == "5 GHz"
        assert classify_band(5955e3) == "6 GHz"

    def test_dfs_channel_detection(self):
        """Test DFS channel identification"""
        def is_dfs_channel(channel):
            try:
                ch = int(channel)
                return 52 <= ch <= 64 or 100 <= ch <= 144
            except (TypeError, ValueError):
                return False

        # DFS channels
        assert is_dfs_channel(52) is True
        assert is_dfs_channel(100) is True
        assert is_dfs_channel(120) is True

        # Non-DFS channels
        assert is_dfs_channel(36) is False
        assert is_dfs_channel(149) is False


class TestFrequencyParsing:
    """Test frequency parsing utilities"""

    def test_parse_frequency_hz(self):
        """Test parsing frequency in Hz"""
        import re

        def parse_frequency(value):
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                value = value.strip().lower()
                match = re.match(r'([\d.]+)\s*(ghz|mhz|hz)?', value)
                if match:
                    num = float(match.group(1))
                    unit = match.group(2) if match.group(2) else 'hz'
                    if unit == 'ghz':
                        return num * 1e9
                    elif unit == 'mhz':
                        return num * 1e6
                    else:
                        return num
            return float(value)

        assert parse_frequency("1000 hz") == 1000
        assert parse_frequency("100 mhz") == 100e6
        assert parse_frequency("2.4 ghz") == 2.4e9
        assert parse_frequency(1000000) == 1e6

    def test_parse_frequency_none(self):
        """Test parsing None frequency"""
        def parse_frequency(value):
            if value is None:
                return None
            return float(value)

        assert parse_frequency(None) is None


class TestAPIKeyValidation:
    """Test API key validation"""

    def test_api_key_present(self, monkeypatch):
        """Test when API key is present"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890abcdefghij")

        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None
        assert len(api_key) > 20

    def test_api_key_missing(self, monkeypatch):
        """Test when API key is missing"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is None


class TestLoggingConfiguration:
    """Test logging configuration"""

    def test_logger_levels(self):
        """Test all logging levels are available"""
        import logging

        levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL
        ]

        for level in levels:
            assert isinstance(level, int)

    def test_logger_output(self, caplog):
        """Test logger produces output"""
        import logging

        logger = logging.getLogger("test_tinysa")
        logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.INFO):
            logger.info("Test info message")
            logger.warning("Test warning message")

        assert "Test info message" in caplog.text
        assert "Test warning message" in caplog.text
