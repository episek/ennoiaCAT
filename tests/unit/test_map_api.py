"""
Unit tests for map_api.py
Tests the MapAPI class functionality.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestMapAPI:
    """Tests for the MapAPI class."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline."""
        with patch('map_api.pipeline') as mock_pipe:
            mock_pipe.return_value = Mock()
            yield mock_pipe

    @pytest.fixture
    def mock_torch(self):
        """Mock torch availability."""
        with patch('map_api.torch') as mock_t:
            mock_t.cuda.is_available.return_value = False
            yield mock_t

    def test_map_api_import(self):
        """Test that MapAPI can be imported."""
        try:
            from map_api import MapAPI
            assert MapAPI is not None
        except ImportError as e:
            pytest.skip(f"Cannot import MapAPI: {e}")

    def test_get_system_prompt(self, mock_pipeline, mock_torch):
        """Test system prompt generation."""
        try:
            from map_api import MapAPI
        except ImportError:
            pytest.skip("MapAPI dependencies not available")

        # Create instance without tokenizer to skip pipeline creation
        api = MapAPI.__new__(MapAPI)

        original_dict = {"start": 100, "stop": 200}
        user_input = "Set start to 150"

        prompt = api.get_system_prompt(original_dict, user_input)

        assert "JSON" in prompt
        assert "start" in prompt
        assert "150" in prompt
        assert "dictionary" in prompt.lower()

    def test_get_few_shot_examples(self, mock_pipeline, mock_torch):
        """Test few-shot examples generation."""
        try:
            from map_api import MapAPI
        except ImportError:
            pytest.skip("MapAPI dependencies not available")

        api = MapAPI.__new__(MapAPI)

        examples = api.get_few_shot_examples()

        assert isinstance(examples, list)
        assert len(examples) > 0
        # Check structure of examples
        for example in examples:
            assert "role" in example
            assert "content" in example
            assert example["role"] in ["user", "assistant"]

    def test_system_prompt_contains_user_input(self, mock_pipeline, mock_torch):
        """Test that system prompt includes user input."""
        try:
            from map_api import MapAPI
        except ImportError:
            pytest.skip("MapAPI dependencies not available")

        api = MapAPI.__new__(MapAPI)

        test_input = "Change frequency to 500 MHz"
        prompt = api.get_system_prompt({}, test_input)

        assert test_input in prompt

    def test_system_prompt_contains_dict(self, mock_pipeline, mock_torch):
        """Test that system prompt includes original dictionary."""
        try:
            from map_api import MapAPI
        except ImportError:
            pytest.skip("MapAPI dependencies not available")

        api = MapAPI.__new__(MapAPI)

        test_dict = {"frequency": 100, "power": -10}
        prompt = api.get_system_prompt(test_dict, "test")

        assert "frequency" in prompt
        assert "100" in prompt
        assert "power" in prompt


class TestMapAPIWithMockedDependencies:
    """Tests that mock heavy dependencies."""

    def test_initialization_with_tokenizer(self):
        """Test MapAPI initialization with mocked dependencies."""
        with patch('map_api.pipeline') as mock_pipe, \
             patch('map_api.torch') as mock_torch:

            mock_torch.cuda.is_available.return_value = False
            mock_pipe.return_value = Mock()

            try:
                from map_api import MapAPI

                mock_tokenizer = Mock()
                api = MapAPI(
                    model="test-model",
                    tokenizer=mock_tokenizer,
                    max_new_tokens=100,
                    temperature=0.5
                )

                mock_pipe.assert_called_once()
                assert api.device == "cpu"
            except ImportError:
                pytest.skip("MapAPI dependencies not available")

    def test_initialization_with_cuda(self):
        """Test MapAPI initialization with CUDA available."""
        with patch('map_api.pipeline') as mock_pipe, \
             patch('map_api.torch') as mock_torch:

            mock_torch.cuda.is_available.return_value = True
            mock_pipe.return_value = Mock()

            try:
                from map_api import MapAPI

                mock_tokenizer = Mock()
                api = MapAPI(tokenizer=mock_tokenizer)

                assert api.device == "cuda"
            except ImportError:
                pytest.skip("MapAPI dependencies not available")
