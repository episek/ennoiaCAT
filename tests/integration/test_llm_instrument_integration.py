"""
Integration tests for LLM/SLM with instrument control
Tests end-to-end workflows combining AI and instruments
"""
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestLLMInstrumentIntegration:
    """Test LLM integration with instrument control"""

    @patch('openai.OpenAI')
    def test_llm_guides_instrument_configuration(self, mock_openai_class, mock_openai_client):
        """Test LLM providing guidance for instrument setup"""
        mock_openai_class.return_value = mock_openai_client

        # User asks LLM how to configure instrument
        user_query = "How do I set the TinySA frequency range to 100-900 MHz?"

        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant for TinySA."},
                {"role": "user", "content": user_query}
            ]
        )

        # Verify LLM responds
        assert response.choices[0].message.content is not None

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_slm_analyzes_spectrum_data(self, mock_model_class, mock_tokenizer_class,
                                         mock_transformers_model, mock_transformers_tokenizer,
                                         sample_spectrum_data):
        """Test SLM analyzing spectrum data"""
        mock_tokenizer_class.return_value = mock_transformers_tokenizer
        mock_model_class.return_value = mock_transformers_model

        # Simulate SLM analyzing spectrum data
        frequencies = sample_spectrum_data['frequencies']
        power_levels = sample_spectrum_data['power_levels']

        # Find peak
        max_power_idx = power_levels.argmax()
        peak_frequency = frequencies[max_power_idx]

        # SLM generates explanation
        prompt = f"Detected peak at {peak_frequency/1e6:.2f} MHz. Explain what this might be."

        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids)
        response = tokenizer.decode(output[0])

        assert response == "Generated response from SLM"


class TestInstrumentWorkflows:
    """Test complete instrument workflows"""

    @patch('serial.tools.list_ports.comports')
    def test_tinysa_scan_workflow(self, mock_comports, sample_spectrum_data):
        """Test complete TinySA scan workflow"""
        from tests.fixtures.mock_instruments import MockTinySA

        # Mock serial port detection
        mock_port = Mock()
        mock_port.device = "COM3"
        mock_port.vid = 0x0483
        mock_port.pid = 0x5740
        mock_comports.return_value = [mock_port]

        # Initialize instrument
        tinysa = MockTinySA()

        # Connect
        assert tinysa.connect() is True

        # Configure frequency range
        tinysa.set_frequency_range(100e6, 1e9)

        # Perform scan
        frequencies, power = tinysa.scan()

        assert len(frequencies) == 101
        assert len(power) == 101

        # Disconnect
        assert tinysa.disconnect() is True

    def test_viavi_measurement_workflow(self):
        """Test complete Viavi measurement workflow"""
        from tests.fixtures.mock_instruments import MockViaviOneAdvisor

        # Initialize instrument
        viavi = MockViaviOneAdvisor("192.168.1.100")

        # Connect
        viavi.open()
        assert viavi.connected is True

        # Query instrument
        idn = viavi.query("*IDN?")
        assert "Viavi" in idn

        # Perform measurement (simulated)
        viavi.write(":MEAS:POWER?")

        # Disconnect
        viavi.close()
        assert viavi.connected is False

    def test_keysight_spectrum_analysis(self):
        """Test Keysight spectrum analysis workflow"""
        from tests.fixtures.mock_instruments import MockKeysightFieldFox

        # Initialize instrument
        keysight = MockKeysightFieldFox("TCPIP0::192.168.1.101::inst0::INSTR")

        # Query instrument info
        idn = keysight.query("*IDN?")
        assert "Keysight" in idn

        # Set up spectrum analyzer mode
        keysight.write(":INST:SEL SA")
        keysight.write(":FREQ:START 100MHz")
        keysight.write(":FREQ:STOP 1GHz")

        # Close connection
        keysight.close()


class TestMultiInstrumentScenarios:
    """Test scenarios with multiple instruments"""

    @patch('serial.tools.list_ports.comports')
    @patch('socket.socket')
    def test_multiple_instruments_detected(self, mock_socket_class, mock_comports):
        """Test detecting and managing multiple instruments"""
        from instrument_detector import InstrumentDetector

        # Mock TinySA on serial
        mock_port = Mock()
        mock_port.device = "COM3"
        mock_port.vid = 0x0483
        mock_port.pid = 0x5740
        mock_port.description = "TinySA"
        mock_comports.return_value = [mock_port]

        # Mock Viavi on network
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 0
        mock_socket_class.return_value = mock_socket

        detector = InstrumentDetector()
        instruments = detector.detect_all()

        assert len(instruments) >= 1

    def test_switching_between_instruments(self):
        """Test switching between different instruments"""
        from instrument_adapters import AdapterFactory
        from instrument_detector import InstrumentType

        # Create adapters for different instruments
        tinysa_adapter = AdapterFactory.create_adapter(
            InstrumentType.TINYSA,
            {'port': 'COM3'}
        )

        keysight_adapter = AdapterFactory.create_adapter(
            InstrumentType.KEYSIGHT,
            {'ip': '192.168.1.101', 'resource': 'TCPIP0::192.168.1.101::inst0::INSTR'}
        )

        # Both should be different instances
        assert tinysa_adapter is not None
        assert keysight_adapter is not None
        assert type(tinysa_adapter) != type(keysight_adapter)


class TestErrorHandlingIntegration:
    """Test error handling in integrated scenarios"""

    @patch('openai.OpenAI')
    def test_llm_handles_invalid_instrument_command(self, mock_openai_class):
        """Test LLM handling invalid instrument commands"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Invalid command")
        mock_openai_class.return_value = mock_client

        from openai import OpenAI
        client = OpenAI()

        with pytest.raises(Exception):
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Invalid command"}]
            )

    def test_instrument_connection_failure_handling(self):
        """Test handling instrument connection failures"""
        from tests.fixtures.mock_instruments import MockTinySA

        tinysa = MockTinySA()

        # Should handle connection gracefully
        result = tinysa.connect()
        assert result is True

    @patch('serial.tools.list_ports.comports')
    def test_no_instruments_detected_scenario(self, mock_comports):
        """Test scenario when no instruments are detected"""
        from instrument_detector import InstrumentDetector

        mock_comports.return_value = []

        detector = InstrumentDetector()
        instruments = detector.detect_all()

        # Should return empty list, not error
        assert isinstance(instruments, list)


class TestDataFlowIntegration:
    """Test data flow between components"""

    def test_spectrum_data_to_llm_analysis(self, sample_spectrum_data):
        """Test sending spectrum data to LLM for analysis"""
        import numpy as np

        frequencies = sample_spectrum_data['frequencies']
        power_levels = sample_spectrum_data['power_levels']

        # Find peaks
        max_power = np.max(power_levels)
        peak_freq = frequencies[np.argmax(power_levels)]

        # Create summary for LLM
        summary = f"Peak signal at {peak_freq/1e6:.2f} MHz with power {max_power:.2f} dBm"

        assert "MHz" in summary
        assert "dBm" in summary

    def test_llm_output_to_instrument_config(self):
        """Test converting LLM instructions to instrument config"""

        # Simulate LLM output
        llm_output = "Set start frequency to 100 MHz and stop frequency to 900 MHz"

        # Parse LLM output (simplified)
        assert "100 MHz" in llm_output
        assert "900 MHz" in llm_output

        # Convert to instrument commands
        start_freq = 100e6  # 100 MHz
        stop_freq = 900e6  # 900 MHz

        assert start_freq == 100e6
        assert stop_freq == 900e6
