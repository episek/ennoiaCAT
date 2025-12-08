"""
Complexity Analysis Script for EnnoiaCAT Project
Runs radon to measure code complexity metrics
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
    """Run all complexity analysis checks"""
    # Get the project root directory (parent of build_script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    os.chdir(project_root)

    print("=" * 60)
    print("EnnoiaCAT Complexity Analysis")
    print("=" * 60)

    # Run radon cyclomatic complexity
    print("\n" + "=" * 60)
    print("Cyclomatic Complexity (CC)")
    print("=" * 60)
    print("A=1-5 (simple), B=6-10 (well structured), C=11-20 (slightly complex)")
    print("D=21-30 (more complex), E=31-40 (very complex), F=41+ (extremely complex)")
    cc_cmd = (
        'python -m radon cc . '
        '--exclude "build,dist,.git,__pycache__,venv,env,ennoia,tinyllama_tinysa_lora,RsInstrument,tests,build_script" '
        '-s -a --total-average'
    )
    cc_result = run_command(cc_cmd, "Radon Cyclomatic Complexity")

    # Run radon maintainability index
    print("\n" + "=" * 60)
    print("Maintainability Index (MI)")
    print("=" * 60)
    print("A=100-20 (very high), B=19-10 (medium), C=9-0 (extremely low)")
    mi_cmd = (
        'python -m radon mi . '
        '--exclude "build,dist,.git,__pycache__,venv,env,ennoia,tinyllama_tinysa_lora,RsInstrument,tests,build_script" '
        '-s'
    )
    mi_result = run_command(mi_cmd, "Radon Maintainability Index")

    # Run radon raw metrics
    print("\n" + "=" * 60)
    print("Raw Metrics (LOC, SLOC, Comments, etc.)")
    print("=" * 60)
    raw_cmd = (
        'python -m radon raw . '
        '--exclude "build,dist,.git,__pycache__,venv,env,ennoia,tinyllama_tinysa_lora,RsInstrument,tests,build_script" '
        '-s'
    )
    raw_result = run_command(raw_cmd, "Radon Raw Metrics")

    # Run radon halstead metrics
    print("\n" + "=" * 60)
    print("Halstead Metrics (Program vocabulary, difficulty, effort)")
    print("=" * 60)
    hal_cmd = (
        'python -m radon hal . '
        '--exclude "build,dist,.git,__pycache__,venv,env,ennoia,tinyllama_tinysa_lora,RsInstrument,tests,build_script"'
    )
    hal_result = run_command(hal_cmd, "Radon Halstead Metrics")

    # Generate JSON reports for further analysis
    print("\nGenerating JSON reports...")

    subprocess.run(
        'python -m radon cc . --json --exclude "build,dist,.git,__pycache__,venv,env,ennoia,tinyllama_tinysa_lora,RsInstrument,tests,build_script" > build_script/cc_report.json',
        shell=True
    )

    subprocess.run(
        'python -m radon mi . --json --exclude "build,dist,.git,__pycache__,venv,env,ennoia,tinyllama_tinysa_lora,RsInstrument,tests,build_script" > build_script/mi_report.json',
        shell=True
    )

    # Summary
    print("\n" + "=" * 60)
    print("COMPLEXITY ANALYSIS SUMMARY")
    print("=" * 60)
    print("Analysis complete! Review the metrics above.")
    print("JSON reports saved to:")
    print("  - build_script/cc_report.json (Cyclomatic Complexity)")
    print("  - build_script/mi_report.json (Maintainability Index)")
    print("\nRecommendations:")
    print("  • Functions with CC > 10 should be refactored")
    print("  • Files with MI < 20 need attention")
    print("  • Aim for average CC grade of A or B")

    return 0

if __name__ == "__main__":
    sys.exit(main())
