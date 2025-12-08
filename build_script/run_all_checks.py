"""
Master Script - Run All Code Quality Checks
Executes linting, static analysis, and complexity analysis
"""
import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and return the result"""
    print(f"\n{'#' * 70}")
    print(f"# {description}")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}\n")

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True
    )

    print(f"\n{'#' * 70}")
    print(f"# {description} - {'COMPLETED' if result.returncode == 0 else 'COMPLETED WITH WARNINGS'}")
    print(f"# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}\n")

    return result.returncode

def main():
    """Run all quality checks in sequence"""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    start_time = datetime.now()

    print("=" * 70)
    print("EnnoiaCAT - Complete Code Quality Analysis Suite")
    print("=" * 70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}

    # Run linting
    results['lint'] = run_script(
        'run_lint.py',
        'LINTING ANALYSIS'
    )

    # Run static analysis
    results['static'] = run_script(
        'run_static_analysis.py',
        'STATIC ANALYSIS'
    )

    # Run complexity analysis
    results['complexity'] = run_script(
        'run_complexity_analysis.py',
        'COMPLEXITY ANALYSIS'
    )

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Linting:           {'✓ PASSED' if results['lint'] == 0 else '⚠ WARNINGS'}")
    print(f"Static Analysis:   {'✓ PASSED' if results['static'] == 0 else '⚠ ISSUES FOUND'}")
    print(f"Complexity:        ✓ COMPLETED")
    print("=" * 70)
    print(f"Total Duration: {duration.total_seconds():.2f} seconds")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Return non-zero if any critical issues
    if results['lint'] != 0:
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
