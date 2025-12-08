"""
Instrument Adapter Classes
Provides a unified interface for different instrument types
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import streamlit as st


class InstrumentAdapter(ABC):
    """Base class for all instrument adapters"""

    def __init__(self, connection_info: Dict[str, Any]):
        self.connection_info = connection_info
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the instrument"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the instrument"""
        pass

    @abstractmethod
    def get_helper_class(self):
        """Get the helper/config class for this instrument"""
        pass

    @abstractmethod
    def render_ui(self):
        """Render instrument-specific UI elements"""
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """Get the display name for this instrument"""
        pass


class TinySAAdapter(InstrumentAdapter):
    """Adapter for TinySA Spectrum Analyzer"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.helper = None

    def connect(self) -> bool:
        try:
            # Try importing without initializing heavy dependencies
            import sys
            import importlib.util

            # Check if tinySA_config exists
            spec = importlib.util.find_spec("tinySA_config")
            if spec is None:
                st.error("TinySA configuration module not found")
                return False

            # Import the module
            from tinySA_config import TinySAHelper
            self.helper = TinySAHelper()
            self.is_connected = True
            return True
        except ImportError as e:
            st.error(f"Failed to import TinySA module: {e}")
            st.info("Some features may require PyTorch. Install with: pip install torch")
            return False
        except Exception as e:
            st.error(f"Failed to connect to TinySA: {e}")
            st.info("Try connecting without AI features enabled")
            return False

    def disconnect(self) -> bool:
        self.helper = None
        self.is_connected = False
        return True

    def get_helper_class(self):
        return self.helper

    def render_ui(self):
        st.sidebar.image('ennoia.jpg', width=200)
        st.title("TinySA Spectrum Analyzer")
        st.caption(f"Connected to {self.connection_info.get('port', 'Unknown')}")

    def get_display_name(self) -> str:
        return f"TinySA on {self.connection_info.get('port', 'Unknown')}"


class ViaviAdapter(InstrumentAdapter):
    """Adapter for Viavi OneAdvisor"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.system_api = None
        self.radio_api = None

    def connect(self) -> bool:
        try:
            from ennoia_viavi.system_api import OneAdvisorSystemAPI
            from ennoia_viavi.radio_api import OneAdvisorRadioAPI

            ip = self.connection_info.get('ip')
            self.system_api = OneAdvisorSystemAPI(ip)
            self.system_api.open()

            radio_port = self.system_api.get_radio_scpi_port()
            self.radio_api = OneAdvisorRadioAPI(ip, scpi_port=radio_port)
            self.radio_api.open()

            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Viavi: {e}")
            return False

    def disconnect(self) -> bool:
        try:
            if self.radio_api:
                self.radio_api.close()
            if self.system_api:
                self.system_api.close()
            self.is_connected = False
            return True
        except Exception:
            return False

    def get_helper_class(self):
        return {
            'system_api': self.system_api,
            'radio_api': self.radio_api
        }

    def render_ui(self):
        st.sidebar.image("viavi.png", width=200)
        st.sidebar.image("ennoia.jpg", width=200)
        st.title("🗼 Viavi OneAdvisor Agentic AI Control & Analysis")
        st.caption(f"Connected to {self.connection_info.get('ip', 'Unknown')}")

    def get_display_name(self) -> str:
        return f"Viavi OneAdvisor @ {self.connection_info.get('ip', 'Unknown')}"


class MavenirAdapter(InstrumentAdapter):
    """Adapter for Mavenir 5G NR Radio Unit"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.helper = None

    def connect(self) -> bool:
        try:
            from mav_config import TinySAHelper
            self.helper = TinySAHelper()
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Mavenir RU: {e}")
            return False

    def disconnect(self) -> bool:
        self.helper = None
        self.is_connected = False
        return True

    def get_helper_class(self):
        return self.helper

    def render_ui(self):
        st.sidebar.image('mavenir_logo.png', width=200)
        st.title("Mavenir 5G NR Radio Unit")
        st.caption(f"Connected to {self.connection_info.get('ip', 'Unknown')}")

    def get_display_name(self) -> str:
        return f"Mavenir 5G NR RU @ {self.connection_info.get('ip', 'Unknown')}"


class CiscoAdapter(InstrumentAdapter):
    """Adapter for Cisco NCS540"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.helper = None
        self.conn = None

    def connect(self) -> bool:
        try:
            from CS_config import CSHelper
            self.helper = CSHelper()
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Cisco NCS540: {e}")
            return False

    def disconnect(self) -> bool:
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
        self.helper = None
        self.is_connected = False
        return True

    def get_helper_class(self):
        return self.helper

    def render_ui(self):
        st.sidebar.image('cisco_logo.png', width=200)
        st.sidebar.image('ennoia_white_black_hi-def.png', width=200)
        st.title("Cisco NCS540")
        st.caption(f"Connected via {self.connection_info.get('port', 'Unknown')}")

    def get_display_name(self) -> str:
        return f"Cisco NCS540 on {self.connection_info.get('port', 'Unknown')}"


class KeysightAdapter(InstrumentAdapter):
    """Adapter for Keysight FieldFox"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.inst = None

    def connect(self) -> bool:
        try:
            import pyvisa
            rm = pyvisa.ResourceManager()
            resource = self.connection_info.get('resource')
            self.inst = rm.open_resource(resource)
            self.inst.read_termination = '\n'
            self.inst.write_termination = '\n'
            self.inst.timeout = 5000
            self.inst.write(":INSTrument:SELect 'SA'")
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Keysight: {e}")
            return False

    def disconnect(self) -> bool:
        try:
            if self.inst:
                self.inst.close()
            self.is_connected = False
            return True
        except Exception:
            return False

    def get_helper_class(self):
        return self.inst

    def render_ui(self):
        st.sidebar.image('ennoia.jpg', width=200)
        st.title("Keysight FieldFox Spectrum Analyzer")
        st.caption(f"Connected to {self.connection_info.get('ip', 'Unknown')}")

    def get_display_name(self) -> str:
        return f"Keysight FieldFox @ {self.connection_info.get('ip', 'Unknown')}"


class RohdeSchwarzAdapter(InstrumentAdapter):
    """Adapter for Rohde & Schwarz NRQ6"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.inst = None
        self.helper = None

    def connect(self) -> bool:
        try:
            from RsInstrument import RsInstrument
            from RS_config import RSHelper

            resource = self.connection_info.get('resource')
            self.inst = RsInstrument(resource, id_query=True, reset=True)
            self.helper = RSHelper()
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Rohde & Schwarz: {e}")
            return False

    def disconnect(self) -> bool:
        try:
            if self.inst:
                self.inst.close()
            self.helper = None
            self.is_connected = False
            return True
        except Exception:
            return False

    def get_helper_class(self):
        return {
            'instrument': self.inst,
            'helper': self.helper
        }

    def render_ui(self):
        st.sidebar.image('RS_logo.png', width=200)
        st.title("Rohde & Schwarz NRQ6")
        st.caption(f"Connected via {self.connection_info.get('resource', 'Unknown')}")

    def get_display_name(self) -> str:
        return f"Rohde & Schwarz via {self.connection_info.get('resource', 'Unknown')}"


class AukuaAdapter(InstrumentAdapter):
    """Adapter for Aukua Systems"""

    def __init__(self, connection_info: Dict[str, Any]):
        super().__init__(connection_info)
        self.helper = None

    def connect(self) -> bool:
        try:
            from AK_config import AKHelper
            self.helper = AKHelper()
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to Aukua: {e}")
            return False

    def disconnect(self) -> bool:
        self.helper = None
        self.is_connected = False
        return True

    def get_helper_class(self):
        return self.helper

    def render_ui(self):
        st.sidebar.image('aukua rgb high.jpg', width=200)
        st.sidebar.image('ennoia_white_black_hi-def.png', width=200)
        st.title("Aukua Systems")
        st.caption(f"Connected via {self.connection_info.get('resource', 'Unknown')}")

    def get_display_name(self) -> str:
        return f"Aukua via {self.connection_info.get('resource', 'Unknown')}"


class AdapterFactory:
    """Factory for creating instrument adapters"""

    @staticmethod
    def create_adapter(instrument_type, connection_info: Dict[str, Any]) -> Optional[InstrumentAdapter]:
        """Create an adapter for the given instrument type"""
        from instrument_detector import InstrumentType

        adapter_map = {
            InstrumentType.TINYSA: TinySAAdapter,
            InstrumentType.VIAVI: ViaviAdapter,
            InstrumentType.MAVENIR_RU: MavenirAdapter,
            InstrumentType.CISCO_NCS540: CiscoAdapter,
            InstrumentType.KEYSIGHT: KeysightAdapter,
            InstrumentType.ROHDE_SCHWARZ: RohdeSchwarzAdapter,
            InstrumentType.AUKUA: AukuaAdapter,
        }

        adapter_class = adapter_map.get(instrument_type)
        if adapter_class:
            return adapter_class(connection_info)
        return None
