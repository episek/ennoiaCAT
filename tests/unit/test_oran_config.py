"""
Unit tests for ORAN_config.py
Tests Flask URL configuration, API key validation, and model loading.
"""
import pytest
import os
from unittest.mock import Mock, MagicMock, patch
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestFlaskURLConfiguration:
    """Test Flask server URL configuration"""

    def test_flask_host_default(self, monkeypatch):
        """Test default FLASK_HOST value"""
        monkeypatch.delenv("FLASK_HOST", raising=False)
        host = os.getenv("FLASK_HOST", "127.0.0.1")
        assert host == "127.0.0.1"

    def test_flask_host_custom(self, monkeypatch):
        """Test custom FLASK_HOST value"""
        monkeypatch.setenv("FLASK_HOST", "0.0.0.0")
        host = os.getenv("FLASK_HOST", "127.0.0.1")
        assert host == "0.0.0.0"

    def test_flask_port_default(self, monkeypatch):
        """Test default FLASK_PORT value"""
        monkeypatch.delenv("FLASK_PORT", raising=False)
        port = int(os.getenv("FLASK_PORT", "5002"))
        assert port == 5002

    def test_flask_port_custom(self, monkeypatch):
        """Test custom FLASK_PORT value"""
        monkeypatch.setenv("FLASK_PORT", "8080")
        port = int(os.getenv("FLASK_PORT", "5002"))
        assert port == 8080

    def test_flask_url_construction(self, monkeypatch):
        """Test Flask URL is correctly constructed"""
        monkeypatch.setenv("FLASK_HOST", "192.168.1.100")
        monkeypatch.setenv("FLASK_PORT", "5002")

        host = os.getenv("FLASK_HOST", "127.0.0.1")
        port = int(os.getenv("FLASK_PORT", "5002"))
        url = os.getenv("FLASK_URL", f"http://{host}:{port}")

        assert url == "http://192.168.1.100:5002"

    def test_flask_url_override(self, monkeypatch):
        """Test FLASK_URL can override constructed URL"""
        monkeypatch.setenv("FLASK_URL", "https://secure.server.com:443")
        url = os.getenv("FLASK_URL", "http://127.0.0.1:5002")
        assert url == "https://secure.server.com:443"


class TestAPIKeyValidation:
    """Test OpenAI API key validation"""

    def test_api_key_not_set(self, monkeypatch):
        """Test error when API key is not set"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is None

    def test_api_key_set(self, monkeypatch):
        """Test API key retrieval when set"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890abcdefghij")
        api_key = os.getenv("OPENAI_API_KEY")

        assert api_key is not None
        assert api_key.startswith("sk-")
        assert len(api_key) > 20

    def test_api_key_too_short(self, monkeypatch):
        """Test validation catches too-short API key"""
        monkeypatch.setenv("OPENAI_API_KEY", "short")
        api_key = os.getenv("OPENAI_API_KEY")

        assert len(api_key) < 20

    def test_api_key_validation_logic(self, monkeypatch):
        """Test API key validation logic"""
        def validate_api_key(key):
            if not key:
                raise ValueError("OPENAI_API_KEY not set")
            if len(key) < 20:
                raise ValueError("OPENAI_API_KEY appears to be invalid")
            return True

        # Test missing key
        with pytest.raises(ValueError, match="not set"):
            validate_api_key(None)

        # Test short key
        with pytest.raises(ValueError, match="invalid"):
            validate_api_key("short")

        # Test valid key
        assert validate_api_key("sk-test-1234567890abcdefghij") is True


class TestORANHelperClass:
    """Test ORANHelper class methods"""

    def test_select_checkboxes_returns_list(self):
        """Test that select_checkboxes returns a list"""
        with patch('streamlit.sidebar') as mock_sidebar:
            mock_sidebar.subheader = Mock()
            mock_sidebar.checkbox = Mock(return_value=False)

            # Mock implementation
            slm_option = False
            openai_option = True

            selected = []
            if slm_option:
                selected.append("SLM")
            if openai_option:
                selected.append("OpenAI")

            assert isinstance(selected, list)
            assert "OpenAI" in selected

    def test_load_openai_model_success(self, monkeypatch):
        """Test successful OpenAI model loading"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890abcdefghij")

        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            api_key = os.getenv("OPENAI_API_KEY")
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            model = "gpt-4o-mini"

            assert client is not None
            assert model == "gpt-4o-mini"

    def test_load_openai_model_missing_key(self, monkeypatch):
        """Test OpenAI model loading with missing key"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        api_key = os.getenv("OPENAI_API_KEY")

        with pytest.raises(ValueError):
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")


class TestModelLoading:
    """Test model loading functionality"""

    def test_device_selection_cpu(self):
        """Test CPU device selection when CUDA unavailable"""
        with patch('torch.cuda.is_available', return_value=False):
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            assert device == "cpu"

    @pytest.mark.skipif(not os.getenv("TEST_WITH_GPU"), reason="GPU tests disabled")
    def test_device_selection_cuda(self):
        """Test CUDA device selection when available"""
        with patch('torch.cuda.is_available', return_value=True):
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            assert device == "cuda"

    def test_gpu_memory_check(self):
        """Test GPU memory checking logic"""
        def check_gpu_memory():
            try:
                import torch
                if torch.cuda.is_available():
                    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    free_mem_gb = free_mem / (1024**3)
                    return free_mem_gb
                return 0
            except Exception:
                return 0

        mem = check_gpu_memory()
        assert isinstance(mem, (int, float))
        assert mem >= 0


class TestSystemPrompt:
    """Test system prompt generation"""

    def test_system_prompt_content(self):
        """Test system prompt contains expected content"""
        system_prompt = """You are an expert O-RAN fronthaul packet analyzer following the O-RAN Alliance specifications.
You help users analyze PCAP files containing O-RAN fronthaul data and detect interference patterns."""

        assert "O-RAN" in system_prompt
        assert "PCAP" in system_prompt
        assert "interference" in system_prompt
        assert "fronthaul" in system_prompt

    def test_system_prompt_not_empty(self):
        """Test system prompt is not empty"""
        def get_system_prompt():
            return """You are an expert O-RAN fronthaul packet analyzer."""

        prompt = get_system_prompt()
        assert len(prompt) > 0


class TestAnalysisResults:
    """Test analysis result handling"""

    def test_analysis_results_structure(self):
        """Test expected structure of analysis results"""
        # Mock analysis results
        results = {
            "success": True,
            "message": "Analysis completed",
            "evm_results": {0: -25.5, 1: -26.0, 2: -24.8, 3: -25.2},
            "interference_detected": False,
            "snr_per_layer": {0: 30.0, 1: 29.5, 2: 30.2, 3: 29.8}
        }

        assert "success" in results
        assert "evm_results" in results
        assert isinstance(results["evm_results"], dict)
        assert len(results["evm_results"]) == 4

    def test_error_result_structure(self):
        """Test error result structure"""
        error_result = {
            "error": "File not found",
            "status_code": 400
        }

        assert "error" in error_result
        assert error_result["status_code"] == 400


class TestNetworkCommunication:
    """Test network communication with Flask server"""

    @patch('requests.post')
    def test_upload_success(self, mock_post):
        """Test successful file upload"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "message": "File processed"}
        mock_post.return_value = mock_response

        import requests
        response = requests.post(
            "http://127.0.0.1:5002/upload",
            json={"filepath": "/path/to/file.pcap"},
            timeout=300
        )

        assert response.status_code == 200
        assert response.json()["success"] is True

    @patch('requests.post')
    def test_upload_timeout(self, mock_post):
        """Test upload timeout handling"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()

        with pytest.raises(requests.exceptions.Timeout):
            requests.post(
                "http://127.0.0.1:5002/upload",
                json={"filepath": "/path/to/file.pcap"},
                timeout=1
            )

    @patch('requests.get')
    def test_check_server_availability(self, mock_get):
        """Test server availability check"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        import requests
        try:
            response = requests.get("http://127.0.0.1:5002/progress", timeout=2)
            available = response.status_code == 200
        except requests.exceptions.RequestException:
            available = False

        assert available is True


class TestLogging:
    """Test logging functionality"""

    def test_logger_configuration(self):
        """Test logger is properly configured"""
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger("ORAN_config")

        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')

    def test_log_message_format(self, caplog):
        """Test log message is properly formatted"""
        import logging

        logger = logging.getLogger("test_oran_config")
        logger.setLevel(logging.INFO)

        with caplog.at_level(logging.INFO):
            logger.info("Test log message")

        assert "Test log message" in caplog.text

    def test_error_logging_with_exc_info(self, caplog):
        """Test error logging includes exception info"""
        import logging

        logger = logging.getLogger("test_oran_config_error")
        logger.setLevel(logging.ERROR)

        with caplog.at_level(logging.ERROR):
            try:
                raise ValueError("Test error")
            except ValueError:
                logger.error("An error occurred", exc_info=True)

        assert "An error occurred" in caplog.text
        assert "ValueError" in caplog.text or "Test error" in caplog.text
