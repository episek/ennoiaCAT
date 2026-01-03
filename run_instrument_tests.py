"""
Comprehensive Test Runner for Viavi OneAdvisor and tinySA
Runs tests in both SLM and OpenAI modes with various configurations
"""
import subprocess
import sys
import os
from datetime import datetime
import json


class InstrumentTestRunner:
    def __init__(self):
        self.results = {
            "viavi": {
                "basic": {"status": "pending", "details": []},
                "slm": {"status": "pending", "details": []},
                "openai": {"status": "pending", "details": []}
            },
            "tinysa": {
                "basic": {"status": "pending", "details": []},
                "slm": {"status": "pending", "details": []},
                "openai": {"status": "pending", "details": []}
            }
        }
        self.start_time = datetime.now()

    def print_header(self, title):
        """Print formatted header"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80 + "\n")

    def print_section(self, title):
        """Print formatted section"""
        print("\n" + "-" * 80)
        print(f"  {title}")
        print("-" * 80)

    def run_test(self, description, command):
        """Run a test command and capture result"""
        self.print_section(description)
        print(f"Command: {command}\n")

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )

        success = result.returncode == 0
        status = "[PASS]" if success else "[FAIL]"

        print(f"\nStatus: {status}")
        if not success and result.stderr:
            print(f"Error: {result.stderr[:500]}")

        return {
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    def test_viavi_basic(self):
        """Test basic Viavi functionality"""
        self.print_header("VIAVI ONEADVISOR - BASIC TESTS")

        tests = [
            ("Connection Test", "pytest tests/integration/test_viavi_tinysa_live.py::TestViaviOneAdvisor::test_viavi_connection -v"),
            ("Spectrum Analyzer Mode", "pytest tests/integration/test_viavi_tinysa_live.py::TestViaviOneAdvisor::test_viavi_spectrum_analyzer_mode -v"),
            ("2.4 GHz Sweep", "pytest tests/integration/test_viavi_tinysa_live.py::TestViaviOneAdvisor::test_viavi_frequency_sweep_2_4ghz -v"),
            ("5 GHz Sweep", "pytest tests/integration/test_viavi_tinysa_live.py::TestViaviOneAdvisor::test_viavi_frequency_sweep_5ghz -v"),
            ("Data Capture", "pytest tests/integration/test_viavi_tinysa_live.py::TestViaviOneAdvisor::test_viavi_data_capture -v"),
        ]

        for desc, cmd in tests:
            result = self.run_test(desc, cmd)
            self.results["viavi"]["basic"]["details"].append({
                "test": desc,
                "success": result["success"]
            })

        all_passed = all(t["success"] for t in self.results["viavi"]["basic"]["details"])
        self.results["viavi"]["basic"]["status"] = "passed" if all_passed else "failed"

    def test_tinysa_basic(self):
        """Test basic tinySA functionality"""
        self.print_header("TINYSA - BASIC TESTS")

        tests = [
            ("Connection Test", "pytest tests/integration/test_viavi_tinysa_live.py::TestTinySA::test_tinysa_connection -v"),
            ("Low Frequency Sweep", "pytest tests/integration/test_viavi_tinysa_live.py::TestTinySA::test_tinysa_frequency_sweep_low -v"),
            ("High Frequency Sweep", "pytest tests/integration/test_viavi_tinysa_live.py::TestTinySA::test_tinysa_frequency_sweep_high -v"),
            ("Scan Data", "pytest tests/integration/test_viavi_tinysa_live.py::TestTinySA::test_tinysa_scan_data -v"),
        ]

        for desc, cmd in tests:
            result = self.run_test(desc, cmd)
            self.results["tinysa"]["basic"]["details"].append({
                "test": desc,
                "success": result["success"]
            })

        all_passed = all(t["success"] for t in self.results["tinysa"]["basic"]["details"])
        self.results["tinysa"]["basic"]["status"] = "passed" if all_passed else "failed"

    def test_viavi_slm(self):
        """Test Viavi with SLM mode"""
        self.print_header("VIAVI ONEADVISOR - SLM MODE TESTS")

        # Set environment for SLM mode
        os.environ['USE_SLM'] = '1'

        tests = [
            ("SLM 2.4 GHz NL", "pytest tests/integration/test_viavi_tinysa_live.py::TestViaviWithSLM::test_viavi_slm_natural_language_2_4ghz -v"),
            ("SLM 5 GHz NL", "pytest tests/integration/test_viavi_tinysa_live.py::TestViaviWithSLM::test_viavi_slm_natural_language_5ghz -v"),
        ]

        for desc, cmd in tests:
            result = self.run_test(desc, cmd)
            self.results["viavi"]["slm"]["details"].append({
                "test": desc,
                "success": result["success"]
            })

        os.environ.pop('USE_SLM', None)

        all_passed = all(t["success"] for t in self.results["viavi"]["slm"]["details"])
        self.results["viavi"]["slm"]["status"] = "passed" if all_passed else "failed"

    def test_viavi_openai(self):
        """Test Viavi with OpenAI mode"""
        self.print_header("VIAVI ONEADVISOR - OPENAI/LLM MODE TESTS")

        if not os.getenv('OPENAI_API_KEY'):
            print("[!] SKIPPED: OPENAI_API_KEY not set")
            self.results["viavi"]["openai"]["status"] = "skipped"
            return

        os.environ['USE_OPENAI'] = '1'

        tests = [
            ("OpenAI 2.4 GHz NL", "pytest tests/integration/test_viavi_tinysa_live.py::TestViaviWithOpenAI::test_viavi_openai_natural_language_2_4ghz -v"),
            ("OpenAI 5 GHz NL", "pytest tests/integration/test_viavi_tinysa_live.py::TestViaviWithOpenAI::test_viavi_openai_natural_language_5ghz -v"),
        ]

        for desc, cmd in tests:
            result = self.run_test(desc, cmd)
            self.results["viavi"]["openai"]["details"].append({
                "test": desc,
                "success": result["success"]
            })

        os.environ.pop('USE_OPENAI', None)

        all_passed = all(t["success"] for t in self.results["viavi"]["openai"]["details"])
        self.results["viavi"]["openai"]["status"] = "passed" if all_passed else "failed"

    def test_tinysa_slm(self):
        """Test tinySA with SLM mode"""
        self.print_header("TINYSA - SLM MODE TESTS")

        os.environ['USE_SLM'] = '1'

        tests = [
            ("SLM FM Radio NL", "pytest tests/integration/test_viavi_tinysa_live.py::TestTinySAWithSLM::test_tinysa_slm_natural_language_fm -v"),
            ("SLM WiFi 2.4 GHz NL", "pytest tests/integration/test_viavi_tinysa_live.py::TestTinySAWithSLM::test_tinysa_slm_natural_language_wifi -v"),
        ]

        for desc, cmd in tests:
            result = self.run_test(desc, cmd)
            self.results["tinysa"]["slm"]["details"].append({
                "test": desc,
                "success": result["success"]
            })

        os.environ.pop('USE_SLM', None)

        all_passed = all(t["success"] for t in self.results["tinysa"]["slm"]["details"])
        self.results["tinysa"]["slm"]["status"] = "passed" if all_passed else "failed"

    def print_summary(self):
        """Print test summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time

        self.print_header("TEST SUMMARY")

        print("VIAVI ONEADVISOR:")
        print(f"  Basic Tests:  {self.results['viavi']['basic']['status'].upper()}")
        for test in self.results['viavi']['basic']['details']:
            symbol = "[+]" if test['success'] else "[-]"
            print(f"    {symbol} {test['test']}")

        print(f"\n  SLM Tests:    {self.results['viavi']['slm']['status'].upper()}")
        for test in self.results['viavi']['slm']['details']:
            symbol = "[+]" if test['success'] else "[-]"
            print(f"    {symbol} {test['test']}")

        print(f"\n  OpenAI Tests: {self.results['viavi']['openai']['status'].upper()}")
        for test in self.results['viavi']['openai']['details']:
            symbol = "[+]" if test['success'] else "[-]"
            print(f"    {symbol} {test['test']}")

        print("\nTINYSA:")
        print(f"  Basic Tests:  {self.results['tinysa']['basic']['status'].upper()}")
        for test in self.results['tinysa']['basic']['details']:
            symbol = "[+]" if test['success'] else "[-]"
            print(f"    {symbol} {test['test']}")

        print(f"\n  SLM Tests:    {self.results['tinysa']['slm']['status'].upper()}")
        for test in self.results['tinysa']['slm']['details']:
            symbol = "[+]" if test['success'] else "[-]"
            print(f"    {symbol} {test['test']}")

        print("\n" + "=" * 80)
        print(f"Total Duration: {duration.total_seconds():.2f} seconds")
        print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Save results to JSON
        with open('test_results.json', 'w') as f:
            json.dump({
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "results": self.results
            }, f, indent=2)

        print("\n[+] Results saved to test_results.json")

    def run_all(self):
        """Run all tests"""
        self.print_header("INSTRUMENT TEST SUITE - VIAVI ONEADVISOR & TINYSA")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Run all test suites
            self.test_viavi_basic()
            self.test_tinysa_basic()
            self.test_viavi_slm()
            self.test_viavi_openai()
            self.test_tinysa_slm()

        except KeyboardInterrupt:
            print("\n\n[!] Tests interrupted by user")
        except Exception as e:
            print(f"\n\n[-] Error running tests: {e}")
        finally:
            self.print_summary()


if __name__ == "__main__":
    runner = InstrumentTestRunner()
    runner.run_all()
