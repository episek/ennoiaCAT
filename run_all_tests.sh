#!/bin/bash
# Master test runner for EnnoiaCAT
# Runs all tests with coverage

echo "==================================================================="
echo "EnnoiaCAT - Complete Test Suite Runner"
echo "==================================================================="
echo

# Check if pytest is installed
python -c "import pytest" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: pytest is not installed"
    echo "Please run: pip install -r tests/requirements.txt"
    exit 1
fi

# Run the test suite
python tests/run_tests.py --suite all --coverage --verbose

echo
echo "==================================================================="
echo "Test run complete"
echo "==================================================================="
