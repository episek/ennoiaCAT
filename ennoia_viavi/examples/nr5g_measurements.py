#!/usr/bin/env python
"""
5G NR quick-measure example using OneAdvisorRadioAPI.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


import argparse
from pprint import pprint

from ennoia_viavi import OneAdvisorRadioAPI, OneAdvisorSystemAPI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("host", help="OneAdvisor IP address")
    parser.add_argument("--center-ghz", type=float, default=3.5)
    parser.add_argument("--band", type=str, default="FR1")
    parser.add_argument("--chan-standard", type=int, default=700, help="example band code")
    parser.add_argument("--chan-num", type=int, default=1)
    parser.add_argument("--ref-level", type=float, default=0.0)
    args = parser.parse_args()

    sys_api = OneAdvisorSystemAPI(args.host)
    sys_api.open()
    print("System IDN:", sys_api.idn())
    radio_port = sys_api.get_radio_scpi_port()
    sys_api.close()

    ra = OneAdvisorRadioAPI(args.host, scpi_port=radio_port)
    try:
        ra.open()
        print("Radio IDN:", ra.idn())

        summary = ra.nr5g_quick_summary(
            center_hz=args.center_ghz * 1e9,
            band=args.band,
            chan_standard=args.chan_standard,
            chan_num=args.chan_num,
            ref_level_dbm=args.ref_level,
        )

        print("\n5G NR quick summary:")
        pprint(summary)

    finally:
        ra.close()


if __name__ == "__main__":
    main()
