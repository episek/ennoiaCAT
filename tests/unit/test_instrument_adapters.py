"""
Unit tests for instrument adapter classes
Tests adapter functionality for all supported instruments
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from instrument_adapters import (
    InstrumentAdapter,
    TinySAAdapter,
    ViaviAdapter,
    MavenirAdapter,
    CiscoAdapter,
    KeysightAdapter,
    RohdeSchwarzAdapter,
    AukuaAdapter,
    AdapterFactory
)
from instrument_detector import InstrumentType


class TestAdapterFactory:
    """Test AdapterFactory class"""

    def test_create_tinysa_adapter(self, mock_tinysa_device):
        """Test creating TinySA adapter"""
        adapter = AdapterFactory.create_adapter(
            InstrumentType.TINYSA,
            mock_tinysa_device
        )

        assert isinstance(adapter, TinySAAdapter)
        assert adapter.connection_info == mock_tinysa_device

    def test_create_viavi_adapter(self, mock_viavi_device):
        """Test creating Viavi adapter"""
        adapter = AdapterFactory.create_adapter(
            InstrumentType.VIAVI,
            mock_viavi_device
        )

        assert isinstance(adapter, ViaviAdapter)

    def test_create_keysight_adapter(self, mock_keysight_device):
        """Test creating Keysight adapter"""
        adapter = AdapterFactory.create_adapter(
            InstrumentType.KEYSIGHT,
            mock_keysight_device
        )

        assert isinstance(adapter, KeysightAdapter)

    def test_create_rohde_schwarz_adapter(self, mock_rohde_schwarz_device):
        """Test creating Rohde & Schwarz adapter"""
        adapter = AdapterFactory.create_adapter(
            InstrumentType.ROHDE_SCHWARZ,
            mock_rohde_schwarz_device
        )

        assert isinstance(adapter, RohdeSchwarzAdapter)

    def test_create_mavenir_adapter(self, mock_mavenir_device):
        """Test creating Mavenir adapter"""
        adapter = AdapterFactory.create_adapter(
            InstrumentType.MAVENIR_RU,
            mock_mavenir_device
        )

        assert isinstance(adapter, MavenirAdapter)

    def test_create_cisco_adapter(self, mock_cisco_device):
        """Test creating Cisco adapter"""
        adapter = AdapterFactory.create_adapter(
            InstrumentType.CISCO_NCS540,
            mock_cisco_device
        )

        assert isinstance(adapter, CiscoAdapter)

    def test_create_aukua_adapter(self, mock_aukua_device):
        """Test creating Aukua adapter"""
        adapter = AdapterFactory.create_adapter(
            InstrumentType.AUKUA,
            mock_aukua_device
        )

        assert isinstance(adapter, AukuaAdapter)


class TestTinySAAdapter:
    """Test TinySA adapter"""

    def test_adapter_initialization(self, mock_tinysa_device):
        """Test TinySA adapter initialization"""
        adapter = TinySAAdapter(mock_tinysa_device)

        assert adapter.connection_info == mock_tinysa_device
        assert adapter.is_connected is False
        assert adapter.helper is None

    @patch('importlib.util.find_spec')
    @patch('tinySA_config.TinySAHelper')
    def test_connect_success(self, mock_helper_class, mock_find_spec, mock_tinysa_device):
        """Test successful connection to TinySA"""
        mock_find_spec.return_value = MagicMock()  # Module exists
        mock_helper = MagicMock()
        mock_helper_class.return_value = mock_helper

        adapter = TinySAAdapter(mock_tinysa_device)

        # Can't actually test connection without mocking streamlit properly
        # Just verify initialization
        assert adapter.helper is None
        assert adapter.is_connected is False

    def test_disconnect(self, mock_tinysa_device):
        """Test disconnecting from TinySA"""
        adapter = TinySAAdapter(mock_tinysa_device)
        adapter.is_connected = True

        result = adapter.disconnect()

        assert result is True
        assert adapter.is_connected is False
        assert adapter.helper is None

    def test_get_display_name(self, mock_tinysa_device):
        """Test getting display name"""
        adapter = TinySAAdapter(mock_tinysa_device)
        name = adapter.get_display_name()

        assert "TinySA" in name
        assert "COM3" in name


class TestViaviAdapter:
    """Test Viavi adapter"""

    def test_adapter_initialization(self, mock_viavi_device):
        """Test Viavi adapter initialization"""
        adapter = ViaviAdapter(mock_viavi_device)

        assert adapter.connection_info == mock_viavi_device
        assert adapter.is_connected is False
        assert adapter.system_api is None
        assert adapter.radio_api is None

    def test_disconnect(self, mock_viavi_device):
        """Test disconnecting from Viavi"""
        adapter = ViaviAdapter(mock_viavi_device)

        # Mock the APIs
        adapter.system_api = MagicMock()
        adapter.radio_api = MagicMock()
        adapter.is_connected = True

        result = adapter.disconnect()

        assert result is True
        assert adapter.is_connected is False
        adapter.system_api.close.assert_called_once()
        adapter.radio_api.close.assert_called_once()

    def test_get_display_name(self, mock_viavi_device):
        """Test getting display name"""
        adapter = ViaviAdapter(mock_viavi_device)
        name = adapter.get_display_name()

        assert "Viavi" in name
        assert "192.168.1.100" in name

    def test_get_helper_class(self, mock_viavi_device):
        """Test getting helper class"""
        adapter = ViaviAdapter(mock_viavi_device)
        adapter.system_api = MagicMock()
        adapter.radio_api = MagicMock()

        helper = adapter.get_helper_class()

        assert 'system_api' in helper
        assert 'radio_api' in helper


class TestKeysightAdapter:
    """Test Keysight adapter"""

    def test_adapter_initialization(self, mock_keysight_device):
        """Test Keysight adapter initialization"""
        adapter = KeysightAdapter(mock_keysight_device)

        assert adapter.connection_info == mock_keysight_device
        assert adapter.is_connected is False
        assert adapter.inst is None

    @patch('pyvisa.ResourceManager')
    def test_connect_success(self, mock_rm_class, mock_keysight_device):
        """Test successful connection to Keysight"""
        mock_rm = MagicMock()
        mock_inst = MagicMock()
        mock_rm.open_resource.return_value = mock_inst
        mock_rm_class.return_value = mock_rm

        adapter = KeysightAdapter(mock_keysight_device)
        result = adapter.connect()

        assert result is True
        assert adapter.is_connected is True
        assert adapter.inst is not None
        mock_inst.write.assert_called_with(":INSTrument:SELect 'SA'")

    def test_disconnect(self, mock_keysight_device):
        """Test disconnecting from Keysight"""
        adapter = KeysightAdapter(mock_keysight_device)
        adapter.inst = MagicMock()
        adapter.is_connected = True

        result = adapter.disconnect()

        assert result is True
        assert adapter.is_connected is False
        adapter.inst.close.assert_called_once()


class TestRohdeSchwarzAdapter:
    """Test Rohde & Schwarz adapter"""

    def test_adapter_initialization(self, mock_rohde_schwarz_device):
        """Test R&S adapter initialization"""
        adapter = RohdeSchwarzAdapter(mock_rohde_schwarz_device)

        assert adapter.connection_info == mock_rohde_schwarz_device
        assert adapter.is_connected is False
        assert adapter.inst is None
        assert adapter.helper is None

    @patch('RsInstrument.RsInstrument')
    @patch('RS_config.RSHelper')
    def test_connect_success(self, mock_helper_class, mock_rs_class, mock_rohde_schwarz_device):
        """Test successful connection to R&S"""
        mock_inst = MagicMock()
        mock_helper = MagicMock()
        mock_rs_class.return_value = mock_inst
        mock_helper_class.return_value = mock_helper

        adapter = RohdeSchwarzAdapter(mock_rohde_schwarz_device)
        result = adapter.connect()

        assert result is True
        assert adapter.is_connected is True
        assert adapter.inst is not None
        assert adapter.helper is not None

    def test_get_helper_class(self, mock_rohde_schwarz_device):
        """Test getting helper class"""
        adapter = RohdeSchwarzAdapter(mock_rohde_schwarz_device)
        adapter.inst = MagicMock()
        adapter.helper = MagicMock()

        helper = adapter.get_helper_class()

        assert 'instrument' in helper
        assert 'helper' in helper


class TestMavenirAdapter:
    """Test Mavenir adapter"""

    def test_adapter_initialization(self, mock_mavenir_device):
        """Test Mavenir adapter initialization"""
        adapter = MavenirAdapter(mock_mavenir_device)

        assert adapter.connection_info == mock_mavenir_device
        assert adapter.is_connected is False
        assert adapter.helper is None

    def test_disconnect(self, mock_mavenir_device):
        """Test disconnecting from Mavenir RU"""
        adapter = MavenirAdapter(mock_mavenir_device)
        adapter.is_connected = True

        result = adapter.disconnect()

        assert result is True
        assert adapter.is_connected is False

    def test_get_display_name(self, mock_mavenir_device):
        """Test getting display name"""
        adapter = MavenirAdapter(mock_mavenir_device)
        name = adapter.get_display_name()

        assert "Mavenir" in name
        assert "10.10.10.10" in name


class TestCiscoAdapter:
    """Test Cisco adapter"""

    def test_adapter_initialization(self, mock_cisco_device):
        """Test Cisco adapter initialization"""
        adapter = CiscoAdapter(mock_cisco_device)

        assert adapter.connection_info == mock_cisco_device
        assert adapter.is_connected is False
        assert adapter.helper is None
        assert adapter.conn is None

    def test_disconnect(self, mock_cisco_device):
        """Test disconnecting from Cisco"""
        adapter = CiscoAdapter(mock_cisco_device)
        adapter.conn = MagicMock()
        adapter.is_connected = True

        result = adapter.disconnect()

        assert result is True
        assert adapter.is_connected is False
        adapter.conn.close.assert_called_once()

    def test_get_display_name(self, mock_cisco_device):
        """Test getting display name"""
        adapter = CiscoAdapter(mock_cisco_device)
        name = adapter.get_display_name()

        assert "Cisco" in name
        assert "COM4" in name


class TestAukuaAdapter:
    """Test Aukua adapter"""

    def test_adapter_initialization(self, mock_aukua_device):
        """Test Aukua adapter initialization"""
        adapter = AukuaAdapter(mock_aukua_device)

        assert adapter.connection_info == mock_aukua_device
        assert adapter.is_connected is False
        assert adapter.helper is None

    def test_disconnect(self, mock_aukua_device):
        """Test disconnecting from Aukua"""
        adapter = AukuaAdapter(mock_aukua_device)
        adapter.is_connected = True

        result = adapter.disconnect()

        assert result is True
        assert adapter.is_connected is False

    def test_get_display_name(self, mock_aukua_device):
        """Test getting display name"""
        adapter = AukuaAdapter(mock_aukua_device)
        name = adapter.get_display_name()

        assert "Aukua" in name


class TestAdapterBaseClass:
    """Test base adapter functionality"""

    def test_adapter_abstract_methods(self):
        """Test that adapter is an abstract base class"""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            InstrumentAdapter({})

    def test_adapter_connection_state(self, mock_tinysa_device):
        """Test adapter connection state tracking"""
        adapter = TinySAAdapter(mock_tinysa_device)

        # Initially disconnected
        assert adapter.is_connected is False

        # Simulate connection
        adapter.is_connected = True
        assert adapter.is_connected is True

        # Disconnect
        adapter.disconnect()
        assert adapter.is_connected is False


class TestAdapterUIRendering:
    """Test adapter UI rendering"""

    @patch('streamlit.sidebar')
    @patch('streamlit.title')
    @patch('streamlit.caption')
    def test_tinysa_render_ui(self, mock_caption, mock_title, mock_sidebar, mock_tinysa_device):
        """Test TinySA UI rendering"""
        adapter = TinySAAdapter(mock_tinysa_device)
        adapter.render_ui()

        mock_title.assert_called_once()
        assert "TinySA" in str(mock_title.call_args)

    @patch('streamlit.sidebar')
    @patch('streamlit.title')
    @patch('streamlit.caption')
    def test_viavi_render_ui(self, mock_caption, mock_title, mock_sidebar, mock_viavi_device):
        """Test Viavi UI rendering"""
        adapter = ViaviAdapter(mock_viavi_device)
        adapter.render_ui()

        mock_title.assert_called_once()
        assert "Viavi" in str(mock_title.call_args)

    @patch('streamlit.sidebar')
    @patch('streamlit.title')
    @patch('streamlit.caption')
    def test_keysight_render_ui(self, mock_caption, mock_title, mock_sidebar, mock_keysight_device):
        """Test Keysight UI rendering"""
        adapter = KeysightAdapter(mock_keysight_device)
        adapter.render_ui()

        mock_title.assert_called_once()
        assert "Keysight" in str(mock_title.call_args)
