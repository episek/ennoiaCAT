@echo off
REM Master test runner for EnnoiaCAT
REM Runs all tests with coverage

echo ===================================================================
echo EnnoiaCAT - Complete Test Suite Runner
echo ===================================================================
echo.

REM Check if pytest is installed
python -c "import pytest" 2>NUL
if errorlevel 1 (
    echo Error: pytest is not installed
    echo Please run: pip install -r tests/requirements.txt
    pause
    exit /b 1
)

REM Run the test suite
python tests/run_tests.py --suite all --coverage --verbose

echo.
echo ===================================================================
echo Test run complete
echo ===================================================================

pause
