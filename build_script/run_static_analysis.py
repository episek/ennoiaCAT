"""
Static Analysis Script for EnnoiaCAT Project
Runs mypy for type checking and bandit for security analysis
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return the result"""
    print(f"\n{'=' * 60}")
    print(f"Running {description}...")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode

def main():
    """Run all static analysis checks"""
    # Get the project root directory (parent of build_script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    os.chdir(project_root)

    print("=" * 60)
    print("EnnoiaCAT Static Analysis")
    print("=" * 60)

    # Run mypy for type checking
    mypy_cmd = (
        'mypy . '
        '--ignore-missing-imports '
        '--exclude "/(build|dist|\.git|__pycache__|venv|env|ennoia|tinyllama_tinysa_lora)/" '
        '--no-strict-optional '
        '--show-error-codes'
    )
    mypy_result = run_command(mypy_cmd, "MyPy Type Checker")

    # Run bandit for security analysis
    bandit_cmd = (
        'bandit -r . '
        '-f text '
        '--exclude ./build,./dist,./.git,./venv,./env,./ennoia,./tinyllama_tinysa_lora '
        '-ll '  # Only show medium and high severity issues
        '--format json -o build_script/bandit_report.json'
    )
    bandit_result = run_command(bandit_cmd, "Bandit Security Scanner")

    # Also generate a readable text report
    bandit_text_cmd = (
        'bandit -r . '
        '-f txt '
        '--exclude ./build,./dist,./.git,./venv,./env,./ennoia,./tinyllama_tinysa_lora '
        '-ll'
    )
    run_command(bandit_text_cmd, "Bandit Security Scanner (Text Output)")

    # Summary
    print("\n" + "=" * 60)
    print("STATIC ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"MyPy Type Checking: {'✓ PASSED' if mypy_result == 0 else '✗ ISSUES FOUND'}")
    print(f"Bandit Security Scan: {'✓ PASSED' if bandit_result == 0 else '⚠ ISSUES FOUND'}")
    print("\nDetailed bandit report saved to: build_script/bandit_report.json")

    return 0

if __name__ == "__main__":
    sys.exit(main())
