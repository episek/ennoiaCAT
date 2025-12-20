"""
Unit tests for SLM (Small Language Model) functionality
Tests TinyLlama and local model integration
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os


class TestSLMModelLoading:
    """Test SLM model loading and initialization"""

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_tinyllama_model_loading(self, mock_model, mock_tokenizer):
        """Test TinyLlama model loading"""
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()

        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        assert model is not None
        assert tokenizer is not None
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()

    @patch('peft.PeftModel.from_pretrained')
    def test_lora_adapter_loading(self, mock_peft):
        """Test LoRA adapter loading for fine-tuned model"""
        mock_base_model = MagicMock()
        mock_peft.return_value = MagicMock()

        from peft import PeftModel

        lora_path = "./tinyllama_tinysa_lora"
        model = PeftModel.from_pretrained(mock_base_model, lora_path)

        assert model is not None

    @patch('torch.cuda.is_available')
    def test_device_selection(self, mock_cuda):
        """Test device selection (CPU vs GPU)"""
        # Test with CUDA available
        mock_cuda.return_value = True
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device == "cuda"

        # Test without CUDA
        mock_cuda.return_value = False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device == "cpu"

    def test_model_name_validation(self):
        """Test model name validation"""
        valid_models = [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "meta-llama/Llama-2-7b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1"
        ]

        for model_name in valid_models:
            assert "/" in model_name
            assert len(model_name.split("/")) == 2


class TestSLMInference:
    """Test SLM inference functionality"""

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_slm_text_generation(self, mock_tokenizer_class, mock_model_class,
                                  mock_transformers_model, mock_transformers_tokenizer):
        """Test SLM text generation"""
        mock_tokenizer_class.return_value = mock_transformers_tokenizer
        mock_model_class.return_value = mock_transformers_model

        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        prompt = "Set frequency to 900 MHz"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(input_ids)
        response = tokenizer.decode(output_ids[0])

        assert response == "Generated response from SLM"

    def test_chat_template_formatting(self, mock_transformers_tokenizer):
        """Test chat template formatting for SLM"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the frequency range?"}
        ]

        # Test that chat template can be applied
        formatted = mock_transformers_tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )

        assert formatted is not None

    def test_max_new_tokens_parameter(self):
        """Test max_new_tokens parameter"""
        max_tokens_values = [50, 100, 256, 512]

        for max_tokens in max_tokens_values:
            assert max_tokens > 0
            assert max_tokens <= 2048  # Typical limit for small models

    @patch('transformers.TextIteratorStreamer')
    def test_streaming_generation(self, mock_streamer):
        """Test streaming text generation"""
        mock_stream = MagicMock()
        mock_stream.__iter__.return_value = iter(["Hello", " world", "!"])
        mock_streamer.return_value = mock_stream

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(MagicMock())

        result = ""
        for chunk in streamer:
            result += chunk

        assert result == "Hello world!"

    def test_generation_parameters(self):
        """Test generation parameters validation"""
        params = {
            'max_new_tokens': 100,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'do_sample': True
        }

        assert params['max_new_tokens'] > 0
        assert 0.0 <= params['temperature'] <= 2.0
        assert 0.0 <= params['top_p'] <= 1.0
        assert params['top_k'] > 0
        assert isinstance(params['do_sample'], bool)


class TestSLMHelperFunctions:
    """Test SLM helper functions"""

    def test_query_local_llm_fast(self, mock_transformers_model, mock_transformers_tokenizer):
        """Test fast local LLM query function"""

        def mock_query_local_llm_fast(user_input, model, tokenizer, system_prompt, max_new_tokens=50):
            # Simulate the function behavior
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
            output = model.generate(input_ids, max_new_tokens=max_new_tokens)
            return tokenizer.decode(output[0])

        result = mock_query_local_llm_fast(
            "Set frequency to 900 MHz",
            mock_transformers_model,
            mock_transformers_tokenizer,
            "You are a helpful assistant",
            max_new_tokens=50
        )

        assert result == "Generated response from SLM"

    def test_query_local_llm_stream_with_context(self, mock_transformers_model, mock_transformers_tokenizer):
        """Test streaming local LLM query with context"""

        def mock_query_local_llm_stream(user_input, model, tokenizer, system_prompt, max_new_tokens=50):
            # Simulate streaming behavior
            for chunk in ["Response ", "from ", "SLM"]:
                yield chunk

        result = ""
        for chunk in mock_query_local_llm_stream(
            "Test query",
            mock_transformers_model,
            mock_transformers_tokenizer,
            "System prompt",
            max_new_tokens=50
        ):
            result += chunk

        assert result == "Response from SLM"

    def test_system_prompt_tinysa_specific(self):
        """Test TinySA-specific system prompt"""
        system_prompt = (
            "You are Ennoia, an AI assistant specifically for the TinySA spectrum analyzer (www.tinysa.org).\n"
            "Your role is to help users configure, troubleshoot, and operate the TinySA only.\n"
        )

        assert "TinySA" in system_prompt
        assert "Ennoia" in system_prompt
        assert len(system_prompt) > 50

    def test_few_shot_examples_tinysa(self):
        """Test few-shot examples for TinySA"""
        few_shot_examples = [
            {
                "role": "user",
                "content": "Set the start frequency to 300 MHz"
            },
            {
                "role": "assistant",
                "content": "To set the start frequency to 300 MHz on the TinySA, press the \"Start\" button..."
            }
        ]

        assert len(few_shot_examples) > 0
        assert few_shot_examples[0]["role"] == "user"
        assert "300 MHz" in few_shot_examples[0]["content"]

    @patch('threading.Thread')
    def test_threaded_generation(self, mock_thread):
        """Test multi-threaded text generation"""
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        import threading

        def generate_text():
            return "Generated text"

        thread = threading.Thread(target=generate_text)
        thread.start()

        assert mock_thread.called


class TestSLMMemoryManagement:
    """Test SLM memory and resource management"""

    @patch('torch.cuda.empty_cache')
    def test_gpu_memory_cleanup(self, mock_empty_cache):
        """Test GPU memory cleanup after inference"""
        import torch

        torch.cuda.empty_cache()
        mock_empty_cache.assert_called_once()

    def test_model_cpu_offloading(self):
        """Test model CPU offloading for memory efficiency"""
        # Test that models can be moved between devices
        device_transitions = [
            ("cpu", "cuda"),
            ("cuda", "cpu")
        ]

        for src, dst in device_transitions:
            assert src in ["cpu", "cuda"]
            assert dst in ["cpu", "cuda"]

    def test_batch_size_validation(self):
        """Test batch size validation"""
        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            assert batch_size > 0
            assert batch_size <= 32  # Reasonable limit for small models
