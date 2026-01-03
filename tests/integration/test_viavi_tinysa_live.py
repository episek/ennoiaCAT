"""
Live Integration Tests for Viavi OneAdvisor and tinySA
Tests both instruments with SLM and OpenAI/LLM modes
"""
import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ennoia_viavi import RadioAPI, SystemAPI
    VIAVI_AVAILABLE = True
except ImportError:
    VIAVI_AVAILABLE = False

try:
    import tinySA
    TINYSA_AVAILABLE = True
except ImportError:
    TINYSA_AVAILABLE = False


@pytest.mark.instruments
@pytest.mark.integration
class TestViaviOneAdvisor:
    """Test suite for Viavi OneAdvisor"""

    @pytest.fixture(scope="class")
    def viavi_connection(self):
        """Setup Viavi OneAdvisor connection"""
        if not VIAVI_AVAILABLE:
            pytest.skip("Viavi module not available")

        try:
            system_api = SystemAPI()
            radio_api = RadioAPI()
            yield {"system": system_api, "radio": radio_api}
        except Exception as e:
            pytest.skip(f"Could not connect to Viavi: {e}")

    def test_viavi_connection(self, viavi_connection):
        """Test basic connection to Viavi OneAdvisor"""
        system_api = viavi_connection["system"]
        # Test system info retrieval
        info = system_api.get_system_info()
        assert info is not None
        print(f"✓ Viavi Connected: {info}")

    def test_viavi_spectrum_analyzer_mode(self, viavi_connection):
        """Test entering Spectrum Analyzer mode"""
        radio_api = viavi_connection["radio"]
        result = radio_api.enter_spectrum_analyzer_mode()
        assert result is not None
        print("✓ Entered Spectrum Analyzer mode")

    def test_viavi_frequency_sweep_2_4ghz(self, viavi_connection):
        """Test 2.4 GHz WiFi band sweep"""
        radio_api = viavi_connection["radio"]

        # Configure for 2.4 GHz WiFi
        start_freq = 2.400e9  # 2.4 GHz
        stop_freq = 2.500e9   # 2.5 GHz
        rbw = 1e6             # 1 MHz

        result = radio_api.configure_spectrum_sweep(
            start_freq=start_freq,
            stop_freq=stop_freq,
            rbw=rbw
        )
        assert result is not None
        print(f"✓ Configured 2.4 GHz sweep: {start_freq/1e9} - {stop_freq/1e9} GHz")

    def test_viavi_frequency_sweep_5ghz(self, viavi_connection):
        """Test 5 GHz WiFi band sweep"""
        radio_api = viavi_connection["radio"]

        # Configure for 5 GHz WiFi
        start_freq = 5.150e9  # 5.15 GHz
        stop_freq = 5.850e9   # 5.85 GHz
        rbw = 1e6             # 1 MHz

        result = radio_api.configure_spectrum_sweep(
            start_freq=start_freq,
            stop_freq=stop_freq,
            rbw=rbw
        )
        assert result is not None
        print(f"✓ Configured 5 GHz sweep: {start_freq/1e9} - {stop_freq/1e9} GHz")

    def test_viavi_data_capture(self, viavi_connection):
        """Test data capture from spectrum analyzer"""
        radio_api = viavi_connection["radio"]

        data = radio_api.get_trace_data()
        assert data is not None
        assert len(data) > 0
        print(f"✓ Captured {len(data)} data points")


@pytest.mark.instruments
@pytest.mark.integration
class TestTinySA:
    """Test suite for tinySA"""

    @pytest.fixture(scope="class")
    def tinysa_connection(self):
        """Setup tinySA connection"""
        if not TINYSA_AVAILABLE:
            pytest.skip("tinySA module not available")

        try:
            # Try to find and connect to tinySA
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            tinysa_port = None

            for port in ports:
                if 'USB' in port.description or 'Serial' in port.description:
                    try:
                        tsa = tinySA.TinySA(port.device)
                        tinysa_port = port.device
                        yield tsa
                        return
                    except:
                        continue

            if tinysa_port is None:
                pytest.skip("tinySA not found on any USB port")

        except Exception as e:
            pytest.skip(f"Could not connect to tinySA: {e}")

    def test_tinysa_connection(self, tinysa_connection):
        """Test basic connection to tinySA"""
        tsa = tinysa_connection
        # Get version info
        version = tsa.version()
        assert version is not None
        print(f"✓ tinySA Connected: {version}")

    def test_tinysa_frequency_sweep_low(self, tinysa_connection):
        """Test low frequency sweep (100 MHz - 350 MHz)"""
        tsa = tinysa_connection

        start_freq = 100e6   # 100 MHz
        stop_freq = 350e6    # 350 MHz

        tsa.set_sweep(start_freq, stop_freq)
        frequencies = tsa.get_frequencies()

        assert frequencies is not None
        assert len(frequencies) > 0
        print(f"✓ Low frequency sweep: {start_freq/1e6} - {stop_freq/1e6} MHz, {len(frequencies)} points")

    def test_tinysa_frequency_sweep_high(self, tinysa_connection):
        """Test high frequency sweep (2.4 GHz - 2.5 GHz)"""
        tsa = tinysa_connection

        start_freq = 2.4e9   # 2.4 GHz
        stop_freq = 2.5e9    # 2.5 GHz

        tsa.set_sweep(start_freq, stop_freq)
        frequencies = tsa.get_frequencies()

        assert frequencies is not None
        assert len(frequencies) > 0
        print(f"✓ High frequency sweep: {start_freq/1e9} - {stop_freq/1e9} GHz, {len(frequencies)} points")

    def test_tinysa_scan_data(self, tinysa_connection):
        """Test scanning and data retrieval"""
        tsa = tinysa_connection

        # Perform scan
        tsa.scan()
        data = tsa.data()

        assert data is not None
        assert len(data) > 0
        print(f"✓ Scan completed, retrieved {len(data)} data points")


@pytest.mark.instruments
@pytest.mark.integration
@pytest.mark.slm
class TestViaviWithSLM:
    """Test Viavi OneAdvisor with SLM mode"""

    @pytest.fixture(scope="class")
    def slm_mode(self):
        """Setup SLM mode for testing"""
        os.environ['USE_SLM'] = '1'
        yield
        os.environ.pop('USE_SLM', None)

    def test_viavi_slm_natural_language_2_4ghz(self, slm_mode):
        """Test SLM with natural language: 'scan 2.4 GHz WiFi band'"""
        pytest.skip("Requires SLM model to be loaded - implement with map_api_vi")

    def test_viavi_slm_natural_language_5ghz(self, slm_mode):
        """Test SLM with natural language: 'scan 5 GHz WiFi band'"""
        pytest.skip("Requires SLM model to be loaded - implement with map_api_vi")


@pytest.mark.instruments
@pytest.mark.integration
@pytest.mark.llm
class TestViaviWithOpenAI:
    """Test Viavi OneAdvisor with OpenAI/LLM mode"""

    @pytest.fixture(scope="class")
    def openai_mode(self):
        """Setup OpenAI mode for testing"""
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("OPENAI_API_KEY not set")
        os.environ['USE_OPENAI'] = '1'
        yield
        os.environ.pop('USE_OPENAI', None)

    def test_viavi_openai_natural_language_2_4ghz(self, openai_mode):
        """Test OpenAI with natural language: 'scan 2.4 GHz WiFi band'"""
        pytest.skip("Requires OpenAI integration - implement with map_api_vi")

    def test_viavi_openai_natural_language_5ghz(self, openai_mode):
        """Test OpenAI with natural language: 'scan 5 GHz WiFi band'"""
        pytest.skip("Requires OpenAI integration - implement with map_api_vi")


@pytest.mark.instruments
@pytest.mark.integration
@pytest.mark.slm
class TestTinySAWithSLM:
    """Test tinySA with SLM mode"""

    @pytest.fixture(scope="class")
    def slm_mode(self):
        """Setup SLM mode for testing"""
        os.environ['USE_SLM'] = '1'
        yield
        os.environ.pop('USE_SLM', None)

    def test_tinysa_slm_natural_language_fm(self, slm_mode):
        """Test SLM with natural language: 'scan FM radio band'"""
        pytest.skip("Requires SLM model integration with tinySA")

    def test_tinysa_slm_natural_language_wifi(self, slm_mode):
        """Test SLM with natural language: 'scan WiFi 2.4 GHz'"""
        pytest.skip("Requires SLM model integration with tinySA")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
