"""
Test Runner Script for EnnoiaCAT
Runs all tests with different configurations
"""
import subprocess
import sys
import argparse
from datetime import datetime


def run_command(command, description):
    """Run a command and display results"""
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(
        command,
        shell=True,
        capture_output=False,
        text=True
    )

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run EnnoiaCAT tests")
    parser.add_argument(
        '--suite',
        choices=['all', 'unit', 'integration', 'llm', 'slm', 'instruments'],
        default='all',
        help='Test suite to run (default: all)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run with coverage report'
    )
    parser.add_argument(
        '--html',
        action='store_true',
        help='Generate HTML coverage report'
    )

    args = parser.parse_args()

    start_time = datetime.now()

    print("=" * 70)
    print("EnnoiaCAT Test Suite")
    print("=" * 70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Suite: {args.suite}")
    print("=" * 70)

    # Build pytest command
    pytest_cmd = "pytest"

    # Add verbosity
    if args.verbose:
        pytest_cmd += " -vv"

    # Add coverage
    if args.coverage:
        pytest_cmd += " --cov=. --cov-report=term"
        if args.html:
            pytest_cmd += " --cov-report=html"

    # Select test suite
    if args.suite == 'unit':
        pytest_cmd += " tests/unit"
    elif args.suite == 'integration':
        pytest_cmd += " tests/integration"
    elif args.suite == 'llm':
        pytest_cmd += " -m llm"
    elif args.suite == 'slm':
        pytest_cmd += " -m slm"
    elif args.suite == 'instruments':
        pytest_cmd += " -m instruments"
    else:
        pytest_cmd += " tests"

    # Run tests
    result = run_command(pytest_cmd, f"Running {args.suite} tests")

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Status: {'✓ PASSED' if result == 0 else '✗ FAILED'}")
    print(f"Duration: {duration.total_seconds():.2f} seconds")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if args.coverage and args.html:
        print("\nHTML coverage report generated in: htmlcov/index.html")

    return result


if __name__ == "__main__":
    sys.exit(main())
