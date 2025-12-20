"""
Linting Script for EnnoiaCAT Project
Runs flake8 and pylint to check code quality
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
    """Run all linting checks"""
    # Get the project root directory (parent of build_script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    os.chdir(project_root)

    print("=" * 60)
    print("EnnoiaCAT Linting Analysis")
    print("=" * 60)

    # Run flake8
    flake8_cmd = (
        'python -m flake8 . '
        '--max-line-length=120 '
        '--exclude=.git,__pycache__,build,dist,*.egg-info,venv,env,ennoia,tinyllama_tinysa_lora '
        '--count --statistics'
    )
    flake8_result = run_command(flake8_cmd, "Flake8")

    # Run pylint
    pylint_cmd = (
        'python -m pylint *.py '
        '--max-line-length=120 '
        '--disable=C0103,C0114,C0115,C0116,W0621,R0913,R0914,R0915 '
        '--output-format=text '
        '--reports=y'
    )
    pylint_result = run_command(pylint_cmd, "Pylint")

    # Summary
    print("\n" + "=" * 60)
    print("LINTING SUMMARY")
    print("=" * 60)
    print(f"Flake8: {'PASSED' if flake8_result == 0 else 'FAILED'}")
    print(f"Pylint: {'PASSED' if pylint_result == 0 else 'FAILED (Review warnings)'}")

    # Exit with error if any check failed critically
    if flake8_result != 0:
        print("\nCritical linting issues found!")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
