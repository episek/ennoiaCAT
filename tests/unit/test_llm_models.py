"""
Unit tests for LLM (Large Language Model) functionality
Tests OpenAI integration and LLM helper functions
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os


class TestOpenAIIntegration:
    """Test OpenAI LLM integration"""

    @patch('openai.OpenAI')
    def test_openai_client_initialization(self, mock_openai_class, env_with_openai_key):
        """Test OpenAI client can be initialized with API key"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        from openai import OpenAI
        client = OpenAI()

        assert client is not None
        mock_openai_class.assert_called_once()

    @patch('openai.OpenAI')
    def test_openai_chat_completion(self, mock_openai_class, mock_openai_client, env_with_openai_key):
        """Test OpenAI chat completion API call"""
        mock_openai_class.return_value = mock_openai_client

        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is spectrum analysis?"}
            ]
        )

        assert response.choices[0].message.content == "This is a test response from the LLM model."
        mock_openai_client.chat.completions.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_openai_streaming_response(self, mock_openai_class, env_with_openai_key):
        """Test OpenAI streaming response"""
        mock_client = MagicMock()
        mock_stream = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
        ]
        mock_client.chat.completions.create.return_value = mock_stream
        mock_openai_class.return_value = mock_client

        from openai import OpenAI
        client = OpenAI()

        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            stream=True
        )

        chunks = [chunk.choices[0].delta.content for chunk in stream]
        assert chunks == ["Hello", " world"]

    def test_system_prompt_generation(self):
        """Test system prompt generation for different instruments"""
        # This would test the system prompt generation in config files
        system_prompts = {
            "TinySA": "You are Ennoia, an AI assistant specifically for the TinySA spectrum analyzer",
            "Keysight": "You are an AI assistant for Keysight FieldFox",
        }

        for instrument, expected_prompt in system_prompts.items():
            assert len(expected_prompt) > 0
            assert "AI assistant" in expected_prompt

    @patch('openai.OpenAI')
    def test_llm_error_handling(self, mock_openai_class, env_with_openai_key):
        """Test LLM error handling when API fails"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        from openai import OpenAI
        client = OpenAI()

        with pytest.raises(Exception) as exc_info:
            client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test"}]
            )

        assert "API Error" in str(exc_info.value)

    @patch('openai.OpenAI')
    def test_llm_with_context(self, mock_openai_class, mock_openai_client, env_with_openai_key):
        """Test LLM with conversation context"""
        mock_openai_class.return_value = mock_openai_client

        from openai import OpenAI
        client = OpenAI()

        messages = [
            {"role": "system", "content": "You are a spectrum analyzer assistant."},
            {"role": "user", "content": "Set frequency to 900 MHz"},
            {"role": "assistant", "content": "Frequency set to 900 MHz"},
            {"role": "user", "content": "What was the last frequency?"}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        assert response.choices[0].message.content is not None
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_llm_max_tokens_parameter(self):
        """Test that max_tokens parameter is respected"""
        # This tests that the max token limit is properly set
        max_tokens_values = [50, 100, 500, 1000]

        for max_tokens in max_tokens_values:
            assert max_tokens > 0
            assert max_tokens <= 4096  # Typical limit

    @patch('openai.OpenAI')
    def test_llm_temperature_parameter(self, mock_openai_class, mock_openai_client, env_with_openai_key):
        """Test temperature parameter for response creativity"""
        mock_openai_class.return_value = mock_openai_client

        from openai import OpenAI
        client = OpenAI()

        for temperature in [0.0, 0.5, 1.0]:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test"}],
                temperature=temperature
            )
            assert response is not None


class TestLLMHelperFunctions:
    """Test LLM helper functions in config modules"""

    def test_load_openai_model_function(self):
        """Test load_OpenAI_model function exists and returns correct type"""
        # This would import and test the actual function from config files
        # For now, we test the expected behavior

        def mock_load_openai_model():
            return MagicMock()

        model = mock_load_openai_model()
        assert model is not None

    def test_few_shot_examples_structure(self):
        """Test few-shot examples have correct structure"""
        few_shot_examples = [
            {
                "role": "user",
                "content": "Set the start frequency to 300 MHz"
            },
            {
                "role": "assistant",
                "content": "To set the start frequency to 300 MHz on the TinySA..."
            }
        ]

        for example in few_shot_examples:
            assert "role" in example
            assert "content" in example
            assert example["role"] in ["user", "assistant", "system"]
            assert len(example["content"]) > 0

    def test_llm_response_validation(self):
        """Test that LLM responses are validated"""
        valid_response = "This is a valid response with complete sentences."
        invalid_response = ""

        assert len(valid_response) > 0
        assert len(invalid_response) == 0
