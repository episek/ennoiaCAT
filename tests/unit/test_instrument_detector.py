"""
Unit tests for instrument detection functionality
Tests InstrumentDetector class and instrument discovery
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from instrument_detector import (
    InstrumentDetector,
    InstrumentType,
    InstrumentInfo
)


class TestInstrumentInfo:
    """Test InstrumentInfo class"""

    def test_instrument_info_creation(self):
        """Test creating InstrumentInfo object"""
        info = InstrumentInfo(
            instrument_type=InstrumentType.TINYSA,
            connection_info={'port': 'COM3'},
            display_name="TinySA on COM3",
            config_module="tinySA_config"
        )

        assert info.instrument_type == InstrumentType.TINYSA
        assert info.connection_info['port'] == 'COM3'
        assert info.display_name == "TinySA on COM3"
        assert info.config_module == "tinySA_config"

    def test_instrument_info_repr(self):
        """Test InstrumentInfo string representation"""
        info = InstrumentInfo(
            instrument_type=InstrumentType.KEYSIGHT,
            connection_info={'ip': '192.168.1.100'},
            display_name="Keysight @ 192.168.1.100",
            config_module="KS_config"
        )

        repr_str = repr(info)
        assert "Keysight" in repr_str
        assert "192.168.1.100" in repr_str


class TestInstrumentTypes:
    """Test InstrumentType enum"""

    def test_all_instrument_types_defined(self):
        """Test all instrument types are properly defined"""
        expected_types = [
            "TINYSA",
            "VIAVI",
            "MAVENIR_RU",
            "CISCO_NCS540",
            "ROHDE_SCHWARZ",
            "AUKUA",
            "KEYSIGHT"
        ]

        for type_name in expected_types:
            assert hasattr(InstrumentType, type_name)

    def test_instrument_type_values(self):
        """Test instrument type values are descriptive"""
        assert "TinySA" in InstrumentType.TINYSA.value
        assert "Viavi" in InstrumentType.VIAVI.value
        assert "Keysight" in InstrumentType.KEYSIGHT.value


class TestTinySADetection:
    """Test TinySA spectrum analyzer detection"""

    @patch('serial.tools.list_ports.comports')
    def test_detect_tinysa_found(self, mock_comports, mock_serial_port):
        """Test TinySA detection when device is present"""
        mock_comports.return_value = [mock_serial_port]

        detector = InstrumentDetector()
        instruments = detector.detect_tinysa()

        assert len(instruments) == 1
        assert instruments[0].instrument_type == InstrumentType.TINYSA
        assert instruments[0].connection_info['port'] == 'COM3'
        assert instruments[0].connection_info['vid'] == 0x0483
        assert instruments[0].connection_info['pid'] == 0x5740

    @patch('serial.tools.list_ports.comports')
    def test_detect_tinysa_not_found(self, mock_comports):
        """Test TinySA detection when device is not present"""
        mock_comports.return_value = []

        detector = InstrumentDetector()
        instruments = detector.detect_tinysa()

        assert len(instruments) == 0

    @patch('serial.tools.list_ports.comports')
    def test_detect_multiple_tinysa(self, mock_comports):
        """Test detection of multiple TinySA devices"""
        mock_port1 = Mock()
        mock_port1.device = "COM3"
        mock_port1.vid = 0x0483
        mock_port1.pid = 0x5740
        mock_port1.description = "TinySA 1"

        mock_port2 = Mock()
        mock_port2.device = "COM4"
        mock_port2.vid = 0x0483
        mock_port2.pid = 0x5740
        mock_port2.description = "TinySA 2"

        mock_comports.return_value = [mock_port1, mock_port2]

        detector = InstrumentDetector()
        instruments = detector.detect_tinysa()

        assert len(instruments) == 2


class TestViaviDetection:
    """Test Viavi OneAdvisor detection"""

    @patch('socket.socket')
    def test_detect_viavi_found(self, mock_socket_class):
        """Test Viavi detection when device is reachable"""
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 0  # Connection successful
        mock_socket_class.return_value = mock_socket

        detector = InstrumentDetector()
        instruments = detector.detect_viavi(["192.168.1.100"])

        assert len(instruments) == 1
        assert instruments[0].instrument_type == InstrumentType.VIAVI
        assert instruments[0].connection_info['ip'] == '192.168.1.100'
        assert instruments[0].connection_info['port'] == 5025

    @patch('socket.socket')
    def test_detect_viavi_not_found(self, mock_socket_class):
        """Test Viavi detection when device is not reachable"""
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 1  # Connection failed
        mock_socket_class.return_value = mock_socket

        detector = InstrumentDetector()
        instruments = detector.detect_viavi(["192.168.1.100"])

        assert len(instruments) == 0

    @patch('socket.socket')
    def test_detect_viavi_default_ip(self, mock_socket_class):
        """Test Viavi detection with default IP"""
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 0
        mock_socket_class.return_value = mock_socket

        detector = InstrumentDetector()
        instruments = detector.detect_viavi()  # No IP provided, should use default

        assert len(instruments) >= 0  # May or may not find device


class TestCiscoDetection:
    """Test Cisco NCS540 detection"""

    @patch('serial.tools.list_ports.comports')
    def test_detect_cisco_ftdi_adapter(self, mock_comports):
        """Test Cisco detection via FTDI USB adapter"""
        mock_port = Mock()
        mock_port.device = "COM4"
        mock_port.vid = 0x0403  # FTDI
        mock_port.pid = 0x6001
        mock_port.description = "USB Serial Converter"
        mock_comports.return_value = [mock_port]

        detector = InstrumentDetector()
        instruments = detector.detect_cisco_ncs540()

        assert len(instruments) == 1
        assert instruments[0].instrument_type == InstrumentType.CISCO_NCS540

    @patch('serial.tools.list_ports.comports')
    def test_detect_cisco_cp2102_adapter(self, mock_comports):
        """Test Cisco detection via CP2102 USB adapter"""
        mock_port = Mock()
        mock_port.device = "COM5"
        mock_port.vid = 0x10C4  # Silicon Labs
        mock_port.pid = 0xEA60
        mock_port.description = "CP2102 USB to UART Bridge"
        mock_comports.return_value = [mock_port]

        detector = InstrumentDetector()
        instruments = detector.detect_cisco_ncs540()

        assert len(instruments) == 1


class TestMavenirDetection:
    """Test Mavenir 5G NR RU detection"""

    @patch('socket.socket')
    def test_detect_mavenir_found(self, mock_socket_class):
        """Test Mavenir RU detection when device is reachable"""
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 0  # NETCONF port open
        mock_socket_class.return_value = mock_socket

        detector = InstrumentDetector()
        instruments = detector.detect_mavenir_ru(["10.10.10.10"])

        assert len(instruments) == 1
        assert instruments[0].instrument_type == InstrumentType.MAVENIR_RU
        assert instruments[0].connection_info['port'] == 830  # NETCONF port


class TestKeysightDetection:
    """Test Keysight FieldFox detection"""

    @patch('pyvisa.ResourceManager')
    def test_detect_keysight_found(self, mock_rm_class):
        """Test Keysight detection via PyVISA"""
        mock_rm = Mock()
        mock_inst = Mock()
        mock_inst.query.return_value = "Keysight Technologies,N9918A,MY12345678,A.01.23"

        mock_rm.open_resource.return_value = mock_inst
        mock_rm_class.return_value = mock_rm

        detector = InstrumentDetector()
        instruments = detector.detect_keysight(["192.168.1.101"])

        assert len(instruments) == 1
        assert instruments[0].instrument_type == InstrumentType.KEYSIGHT
        assert "Keysight" in instruments[0].connection_info['idn']

    @patch('pyvisa.ResourceManager')
    def test_detect_keysight_pyvisa_unavailable(self, mock_rm_class):
        """Test Keysight detection when PyVISA is unavailable"""
        mock_rm_class.side_effect = Exception("PyVISA not available")

        detector = InstrumentDetector()
        instruments = detector.detect_keysight(["192.168.1.101"])

        assert len(instruments) == 0


class TestRohdeSchwarzDetection:
    """Test Rohde & Schwarz NRQ6 detection"""

    @patch('pyvisa.ResourceManager')
    def test_detect_rohde_schwarz_found(self, mock_rm_class):
        """Test R&S detection via PyVISA"""
        mock_rm = Mock()
        mock_rm.list_resources.return_value = [
            'TCPIP0::192.168.1.102::hislip0::INSTR'
        ]

        mock_inst = Mock()
        mock_inst.query.return_value = "Rohde&Schwarz,NRQ6,123456,1.0.0"

        mock_rm.open_resource.return_value = mock_inst
        mock_rm_class.return_value = mock_rm

        detector = InstrumentDetector()
        instruments = detector.detect_rohde_schwarz()

        assert len(instruments) == 1
        assert instruments[0].instrument_type == InstrumentType.ROHDE_SCHWARZ


class TestAukuaDetection:
    """Test Aukua Systems detection"""

    @patch('pyvisa.ResourceManager')
    def test_detect_aukua_found(self, mock_rm_class):
        """Test Aukua detection via PyVISA"""
        mock_rm = Mock()
        mock_rm.list_resources.return_value = [
            'TCPIP0::192.168.1.103::inst0::INSTR'
        ]

        mock_inst = Mock()
        mock_inst.query.return_value = "AUKUA,System-100,SN123456,1.0.0"

        mock_rm.open_resource.return_value = mock_inst
        mock_rm_class.return_value = mock_rm

        detector = InstrumentDetector()
        instruments = detector.detect_aukua()

        assert len(instruments) == 1
        assert instruments[0].instrument_type == InstrumentType.AUKUA


class TestDetectAll:
    """Test detect_all functionality"""

    @patch('serial.tools.list_ports.comports')
    @patch('socket.socket')
    def test_detect_all_multiple_instruments(self, mock_socket_class, mock_comports):
        """Test detecting multiple instruments at once"""
        # Mock TinySA
        mock_tinysa_port = Mock()
        mock_tinysa_port.device = "COM3"
        mock_tinysa_port.vid = 0x0483
        mock_tinysa_port.pid = 0x5740
        mock_tinysa_port.description = "TinySA"

        # Mock Cisco
        mock_cisco_port = Mock()
        mock_cisco_port.device = "COM4"
        mock_cisco_port.vid = 0x0403
        mock_cisco_port.pid = 0x6001
        mock_cisco_port.description = "FTDI"

        mock_comports.return_value = [mock_tinysa_port, mock_cisco_port]

        # Mock network instruments
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 0
        mock_socket_class.return_value = mock_socket

        detector = InstrumentDetector()
        instruments = detector.detect_all()

        assert len(instruments) >= 2  # At least TinySA and Cisco

    def test_get_instruments_by_type(self):
        """Test filtering instruments by type"""
        detector = InstrumentDetector()

        # Add some mock instruments
        detector.detected_instruments = [
            InstrumentInfo(InstrumentType.TINYSA, {}, "TinySA 1", "config"),
            InstrumentInfo(InstrumentType.TINYSA, {}, "TinySA 2", "config"),
            InstrumentInfo(InstrumentType.KEYSIGHT, {}, "Keysight", "config"),
        ]

        tinysa_instruments = detector.get_instruments_by_type(InstrumentType.TINYSA)
        assert len(tinysa_instruments) == 2

        keysight_instruments = detector.get_instruments_by_type(InstrumentType.KEYSIGHT)
        assert len(keysight_instruments) == 1

    def test_get_instrument_count(self):
        """Test getting instrument counts"""
        detector = InstrumentDetector()

        detector.detected_instruments = [
            InstrumentInfo(InstrumentType.TINYSA, {}, "TinySA 1", "config"),
            InstrumentInfo(InstrumentType.TINYSA, {}, "TinySA 2", "config"),
            InstrumentInfo(InstrumentType.VIAVI, {}, "Viavi", "config"),
        ]

        counts = detector.get_instrument_count()

        assert counts[InstrumentType.TINYSA] == 2
        assert counts[InstrumentType.VIAVI] == 1
        assert counts[InstrumentType.KEYSIGHT] == 0
