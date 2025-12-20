# EnnoiaCAT Test Suite

Comprehensive test suite for the EnnoiaCAT multi-instrument test platform.

## Overview

This test suite covers:
- **LLM (Large Language Models)**: OpenAI API integration
- **SLM (Small Language Models)**: TinyLlama and local model integration
- **Instrument Detection**: Auto-detection of test instruments
- **Instrument Adapters**: Unified interface for all instruments
- **Integration Tests**: End-to-end workflows

## Directory Structure

```
tests/
├── unit/                          # Unit tests
│   ├── test_llm_models.py         # LLM functionality tests
│   ├── test_slm_models.py         # SLM functionality tests
│   ├── test_instrument_detector.py # Instrument detection tests
│   └── test_instrument_adapters.py # Adapter tests
├── integration/                   # Integration tests
│   └── test_llm_instrument_integration.py
├── fixtures/                      # Test fixtures and mocks
│   └── mock_instruments.py        # Mock instrument implementations
├── conftest.py                    # Pytest configuration and fixtures
├── pytest.ini                     # Pytest settings
├── run_tests.py                   # Test runner script
└── README.md                      # This file
```

## Installation

Install test dependencies:

```bash
pip install -r tests/requirements.txt
```

Or install specific testing tools:

```bash
pip install pytest pytest-cov pytest-mock
```

## Running Tests

### Run All Tests

```bash
pytest tests
```

Or use the test runner:

```bash
python tests/run_tests.py
```

### Run Specific Test Suites

**Unit tests only:**
```bash
python tests/run_tests.py --suite unit
```

**Integration tests only:**
```bash
python tests/run_tests.py --suite integration
```

**LLM tests:**
```bash
python tests/run_tests.py --suite llm
```

**SLM tests:**
```bash
python tests/run_tests.py --suite slm
```

**Instrument tests:**
```bash
python tests/run_tests.py --suite instruments
```

### Run with Coverage

```bash
python tests/run_tests.py --coverage
```

**Generate HTML coverage report:**
```bash
python tests/run_tests.py --coverage --html
```

The HTML report will be available in `htmlcov/index.html`.

### Verbose Output

```bash
python tests/run_tests.py --verbose
```

## Test Categories

### Unit Tests

#### LLM Model Tests (`test_llm_models.py`)
- OpenAI client initialization
- Chat completion API calls
- Streaming responses
- System prompt generation
- Error handling
- Context management
- Parameter validation (temperature, max_tokens)

#### SLM Model Tests (`test_slm_models.py`)
- TinyLlama model loading
- LoRA adapter integration
- Device selection (CPU/GPU)
- Text generation
- Chat template formatting
- Streaming generation
- Memory management
- Generation parameters

#### Instrument Detector Tests (`test_instrument_detector.py`)
- TinySA detection (USB/Serial)
- Viavi OneAdvisor detection (Network)
- Cisco NCS540 detection (Serial)
- Mavenir RU detection (Network/NETCONF)
- Keysight FieldFox detection (VISA)
- Rohde & Schwarz NRQ6 detection (VISA)
- Aukua Systems detection (VISA)
- Multi-instrument detection
- Instrument filtering by type

#### Instrument Adapter Tests (`test_instrument_adapters.py`)
- Adapter factory pattern
- TinySA adapter (connect, disconnect, UI)
- Viavi adapter (API management)
- Keysight adapter (VISA communication)
- Rohde & Schwarz adapter (RsInstrument)
- Mavenir adapter (NETCONF)
- Cisco adapter (Serial)
- Aukua adapter
- Base adapter functionality
- UI rendering

### Integration Tests

#### LLM-Instrument Integration (`test_llm_instrument_integration.py`)
- LLM guiding instrument configuration
- SLM analyzing spectrum data
- Complete instrument workflows
- Multi-instrument scenarios
- Error handling
- Data flow between components

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `mock_serial_port`: Mock serial port
- `mock_tinysa_device`: Mock TinySA device info
- `mock_viavi_device`: Mock Viavi device info
- `mock_keysight_device`: Mock Keysight device info
- `mock_rohde_schwarz_device`: Mock R&S device info
- `mock_openai_client`: Mock OpenAI client
- `mock_transformers_model`: Mock transformers model
- `mock_transformers_tokenizer`: Mock tokenizer
- `sample_spectrum_data`: Sample spectrum analyzer data
- `env_with_openai_key`: Environment with API key

## Writing New Tests

### Basic Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestMyFeature:
    """Test my new feature"""

    def test_basic_functionality(self):
        """Test basic functionality"""
        # Arrange
        expected = "result"

        # Act
        actual = my_function()

        # Assert
        assert actual == expected

    @patch('module.dependency')
    def test_with_mock(self, mock_dep):
        """Test with mocked dependency"""
        mock_dep.return_value = "mocked"

        result = function_using_dependency()

        assert result is not None
        mock_dep.assert_called_once()
```

### Using Fixtures

```python
def test_with_fixture(mock_tinysa_device):
    """Test using a fixture"""
    adapter = TinySAAdapter(mock_tinysa_device)
    assert adapter.connection_info['port'] == 'COM3'
```

### Marking Tests

```python
@pytest.mark.llm
def test_llm_feature():
    """Test LLM feature"""
    pass

@pytest.mark.slow
def test_long_running():
    """Test that takes a long time"""
    pass

@pytest.mark.requires_hardware
def test_real_instrument():
    """Test requiring actual hardware"""
    pass
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements.txt
      - name: Run tests
        run: python tests/run_tests.py --coverage
```

## Test Coverage Goals

Target coverage percentages:
- **Overall**: >80%
- **Core modules** (detector, adapters): >90%
- **LLM/SLM integration**: >75%
- **UI components**: >60%

## Troubleshooting

### Import Errors

If you encounter import errors, ensure the project root is in your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Or on Windows:
```cmd
set PYTHONPATH=%PYTHONPATH%;%CD%
```

### Mock Issues

If mocks aren't working as expected:
1. Check the patch path is correct
2. Verify the mock is applied before the import
3. Use `mock_calls` to debug call history

### Slow Tests

To skip slow tests:
```bash
pytest -m "not slow"
```

## Best Practices

1. **Keep tests independent** - Each test should be self-contained
2. **Use descriptive names** - Test names should describe what they test
3. **Follow AAA pattern** - Arrange, Act, Assert
4. **Mock external dependencies** - Don't rely on external services
5. **Test edge cases** - Include error conditions and boundary values
6. **Keep tests fast** - Unit tests should run in milliseconds
7. **Document complex tests** - Add docstrings explaining non-obvious tests

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Update this README if adding new test categories

## Support

For issues or questions about tests:
- Check test output for detailed error messages
- Review fixture definitions in `conftest.py`
- Consult pytest documentation: https://docs.pytest.org/
