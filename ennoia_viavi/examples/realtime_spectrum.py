#!/usr/bin/env python
"""
Real-time spectrum display example using OneAdvisorRadioAPI.

This is intentionally simple (text-based); you can plug this into
Matplotlib, Streamlit, etc., in EnnoiaCAT.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import time
import argparse

from ennoia_viavi import OneAdvisorRadioAPI, OneAdvisorSystemAPI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("host", help="OneAdvisor IP address")
    parser.add_argument("--center-ghz", type=float, default=3.5)
    parser.add_argument("--span-mhz", type=float, default=100.0)
    parser.add_argument("--ref-level", type=float, default=0.0)
    parser.add_argument("--rate-hz", type=float, default=5.0, help="update rate (Hz)")
    args = parser.parse_args()

    sys_api = OneAdvisorSystemAPI(args.host)
    sys_api.open()
    print("System IDN:", sys_api.idn())
    # Optional: ensure Radio Analysis app is running
    # sys_api.launch_application("RadioAnalysis")
    radio_port = sys_api.get_radio_scpi_port()
    sys_api.close()

    ra = OneAdvisorRadioAPI(args.host, scpi_port=radio_port)
    try:
        ra.open()
        print("Radio IDN:", ra.idn())

        ra.set_spectrum_mode("spectrumTuned")
        ra.configure_spectrum(
            center_hz=args.center_ghz * 1e9,
            span_hz=args.span_mhz * 1e6,
            rbw_auto=False,
            rbw_hz=100e3,
            ref_level_dbm=args.ref_level,
            atten_mode="Auto",
        )

        period = 1.0 / args.rate_hz
        print("Press Ctrl+C to stop.\n")
        while True:
            t0 = time.time()
            trace = ra.get_spectrum_trace()
            if not trace:
                print("No trace data.")
            else:
                peak = max(trace)
                print(f"Trace points={len(trace)}, peak={peak:.2f} dBm")
            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ra.close()


if __name__ == "__main__":
    main()
