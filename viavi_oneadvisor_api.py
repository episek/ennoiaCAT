#!/usr/bin/env python
"""
viavi_oneadvisor_api.py

Minimal Python API wrapper for VIAVI OneAdvisor / CellAdvisor 5G
Radio Analysis module (Spectrum Analyzer + 5G NR Signal Analyzer).

Tested conceptually against the "CellAdvisor 5G and OneAdvisor 800
Radio Analysis Module Programming Manual" SCPI command set.

Connection model (per VIAVI docs):
- Discovery / gateway port: usually 5025
- Radio Analysis SCPI service: typically "CA5G-SCPI" or "ONA-800-SCPI"
  on port 5600, discoverable via :PRTM:LIST? on 5025.

This script:
- Optionally discovers the radio SCPI port via :PRTM:LIST?
- Connects to the radio SCPI socket
- Provides helpers for:
    * Spectrum Analyzer configuration (center/span, RBW/VBW, amplitude)
    * Reading trace data
    * 5G NR basic configuration and measurements (OBW/ACLR/SEM/EVM skeleton)
    * Simple Wi-Fi band scans using spectrum mode

You can always send any raw SCPI command using `write()` / `query()`.

Author: (Eran Pisek)
"""

import socket
import re
from typing import List, Optional, Tuple


class ViaviOneAdvisor:
    """
    High-level wrapper around VIAVI OneAdvisor / CellAdvisor 5G
    Radio Analysis SCPI interface.

    Typical usage:
        inst = ViaviOneAdvisor("192.168.1.100")
        inst.open()  # will discover the radio SCPI port via :PRTM:LIST?
        inst.configure_spectrum(center_hz=3.5e9, span_hz=100e6)
        trace = inst.get_spectrum_trace()
        inst.close()
    """

    def __init__(
        self,
        host: str,
        discovery_port: int = 5025,
        scpi_port: Optional[int] = None,
        timeout: float = 5.0,
    ) -> None:
        self.host = host
        self.discovery_port = discovery_port
        self.scpi_port = scpi_port  # radio analysis SCPI port (e.g., 5600)
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    # ------------------------------------------------------------------
    # Low-level connection & SCPI helpers
    # ------------------------------------------------------------------

    def open(self) -> None:
        """
        Open socket to the Radio Analysis SCPI service.
        If scpi_port is None, use :PRTM:LIST? on discovery_port to
        discover CA5G-SCPI / ONA-800-SCPI port, then connect to it.
        """
        if self.scpi_port is None:
            self.scpi_port = self._discover_scpi_port()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.scpi_port))

    def close(self) -> None:
        """Close the underlying socket."""
        if self._sock:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _discover_scpi_port(self) -> int:
        """
        Connect to discovery_port (usually 5025) and query :PRTM:LIST?
        to find the CA5G-SCPI/ONA-800-SCPI port.

        Falls back to 5600 if discovery fails.
        """
        default_port = 5600
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect((self.host, self.discovery_port))
                s.sendall(b":PRTM:LIST?\n")
                data = self._recv_until(s, b"\n")
        except Exception:
            return default_port

        text = data.decode(errors="ignore")
        # Typical response contains service names and ports; e.g.
        # "CA5G-SCPI:5600;ONA-800-SCPI:5600;..."
        # We'll look for CA5G-SCPI or ONA-800-SCPI and grab the port.
        m = re.search(r"(CA5G-SCPI|ONA-800-SCPI)\s*[:=]\s*(\d+)", text)
        if m:
            return int(m.group(2))

        return default_port

    @staticmethod
    def _recv_until(sock: socket.socket, terminator: bytes = b"\n") -> bytes:
        """
        Receive until 'terminator' is seen or connection closes.
        Assumes small ASCII-like responses (SCPI strings, CSV, etc.).
        """
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if terminator in chunk:
                break
        return b"".join(chunks)

    def write(self, cmd: str) -> None:
        """
        Send a SCPI command (no response expected).

        Example:
            inst.write("*RST")
            inst.write("SPECtrum:AMPlitude:REFerence 0")
        """
        if not self._sock:
            raise RuntimeError("Socket not open. Call open() first.")
        line = (cmd.strip() + "\n").encode()
        self._sock.sendall(line)

    def query(self, cmd: str) -> str:
        """
        Send a SCPI query and return the response as string (stripped).

        Example:
            idn = inst.query("*IDN?")
            span = inst.query("SPECtrum:FREQuency:SPAN?")
        """
        if not self._sock:
            raise RuntimeError("Socket not open. Call open() first.")
        self.write(cmd)
        resp = self._recv_until(self._sock, b"\n")
        return resp.decode(errors="ignore").strip()

    # ------------------------------------------------------------------
    # General utility
    # ------------------------------------------------------------------

    def idn(self) -> str:
        """Return instrument ID string."""
        return self.query("*IDN?")

    def opc(self) -> bool:
        """
        Check Operation Complete.
        Often you can use *OPC? to wait for long operations.
        """
        try:
            val = self.query("*OPC?")
            return val.strip() in ("1", "ON", "On")
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Spectrum Analyzer helpers (SPECtrum:...)
    # ------------------------------------------------------------------

    def configure_spectrum(
        self,
        center_hz: Optional[float] = None,
        span_hz: Optional[float] = None,
        start_hz: Optional[float] = None,
        stop_hz: Optional[float] = None,
        rbw_hz: Optional[float] = None,
        vbw_hz: Optional[float] = None,
        rbw_auto: Optional[bool] = None,
        vbw_auto: Optional[bool] = None,
        ref_level_dbm: Optional[float] = None,
        atten_db: Optional[float] = None,
        atten_mode: Optional[str] = None,  # "Auto"|"Couple"|"Manual"
    ) -> None:
        """
        Configure Spectrum Analyzer frequency span, RBW/VBW and amplitude.

        Commands used (from Radio Analysis manual):
        - SPECtrum:FREQuency:CENTer
        - SPECtrum:FREQuency:SPAN / :STARt / :STOP
        - SPECtrum:RBW:MODE / SPECtrum:RBW
        - SPECtrum:VBW:MODE / SPECtrum:VBW
        - SPECtrum:AMPlitude:REFerence
        - SPECtrum:AMPlitude:ATTenuation / :MODE
        """
        # Frequency configuration
        if center_hz is not None:
            self.write(f"SPECtrum:FREQuency:CENTer {center_hz} Hz")
        if span_hz is not None:
            self.write(f"SPECtrum:FREQuency:SPAN {span_hz} Hz")
        if start_hz is not None:
            self.write(f"SPECtrum:FREQuency:STARt {start_hz} Hz")
        if stop_hz is not None:
            self.write(f"SPECtrum:FREQuency:STOP {stop_hz} Hz")

        # RBW / VBW
        if rbw_auto is not None:
            mode = "Auto" if rbw_auto else "Manual"
            self.write(f"SPECtrum:RBW:MODE {mode}")
        if rbw_hz is not None:
            self.write(f"SPECtrum:RBW {rbw_hz} Hz")

        if vbw_auto is not None:
            mode = "Auto" if vbw_auto else "Manual"
            self.write(f"SPECtrum:VBW:MODE {mode}")
        if vbw_hz is not None:
            self.write(f"SPECtrum:VBW {vbw_hz} Hz")

        # Amplitude
        if ref_level_dbm is not None:
            self.write(f"SPECtrum:AMPlitude:REFerence {ref_level_dbm}")
        if atten_db is not None:
            self.write(f"SPECtrum:AMPlitude:ATTenuation {atten_db}")
        if atten_mode is not None:
            self.write(f"SPECtrum:AMPlitude:MODE {atten_mode}")

    def set_spectrum_mode(self, mode: str = "spectrumTuned") -> None:
        """
        Set Spectrum Analyzer measurement mode.

        Valid values (per manual) include e.g.:
        - spectrumTuned
        - channelPower
        - occupiedBW
        - spectrumEmissionMask
        - adjacentChannelPower
        - multiAdjacentChannelPower
        - spuriousEmissionMask
        - audioDemod
        - fieldStrength
        - routeMap
        - totalHamonicDistortion
        - gatedSweep
        - powerMeter
        """
        self.write(f"SPECtrum:MODE {mode}")

    def get_spectrum_trace(self) -> List[float]:
        """
        Query current spectrum trace data and return as list of floats.

        Uses:
            SPECtrum:TRACe:DATA?

        Response is typically comma-separated values.
        """
        data_str = self.query("SPECtrum:TRACe:DATA?")
        # Remove any leading headers if present
        parts = [p for p in data_str.replace("\n", "").split(",") if p.strip()]
        vals: List[float] = []
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                # Ignore non-numeric tokens (just in case)
                continue
        return vals

    # ------------------------------------------------------------------
    # Simple Wi-Fi band scans (using Spectrum Analyzer)
    # ------------------------------------------------------------------

    def scan_wifi_band(
        self,
        band: str = "2.4",
        span_hz: float = 100e6,
        rbw_hz: float = 100e3,
        ref_level_dbm: float = -10.0,
    ) -> Tuple[List[float], float, float]:
        """
        Convenience helper to scan a Wi-Fi band using spectrum mode.

        band: "2.4" | "5" | "6"
            2.4 GHz band: center ~2.437 GHz
            5 GHz band:   center ~5.5 GHz
            6 GHz band:   center ~6.5 GHz (WiFi 6E-ish)
        span_hz:
            Frequency span around that center
        rbw_hz:
            Resolution bandwidth
        ref_level_dbm:
            Reference level for display

        Returns:
            (trace_values, center_hz, span_hz)
        """
        if band == "2.4":
            center_hz = 2.437e9
        elif band == "5":
            center_hz = 5.5e9
        elif band == "6":
            center_hz = 6.5e9
        else:
            raise ValueError("band must be '2.4', '5', or '6'")

        self.set_spectrum_mode("spectrumTuned")
        self.configure_spectrum(
            center_hz=center_hz,
            span_hz=span_hz,
            rbw_auto=False,
            rbw_hz=rbw_hz,
            ref_level_dbm=ref_level_dbm,
            atten_mode="Auto",
        )
        # Allow sweep to settle, or use *OPC? if preferred
        # (we'll just rely on continuous sweep here)
        trace = self.get_spectrum_trace()
        return trace, center_hz, span_hz

    # ------------------------------------------------------------------
    # 5G NR helpers (NR5G:...)
    # ------------------------------------------------------------------

    def configure_nr5g_basic(
        self,
        center_hz: float,
        band: str = "FR1",  # "FR1" or "FR2"
        chan_standard: Optional[int] = None,
        chan_num: Optional[int] = None,
        ref_level_dbm: Optional[float] = None,
    ) -> None:
        """
        Basic 5G NR Signal Analyzer configuration (downlink).

        Commands used (from manual):
        - NR5G:FREQuency:CENTer
        - NR5G:FREQuency:BAND (FR1/FR2)
        - NR5G:CHANnel:STANdard
        - NR5G:CHANnel:NUM
        - NR5G:AMPLitude:REFerence
        """
        self.write(f"NR5G:FREQuency:CENTer {center_hz} Hz")
        self.write(f"NR5G:FREQuency:BAND {band}")

        if chan_standard is not None:
            self.write(f"NR5G:CHANnel:STANdard {chan_standard}")
        if chan_num is not None:
            self.write(f"NR5G:CHANnel:NUM {chan_num}")
        if ref_level_dbm is not None:
            self.write(f"NR5G:AMPLitude:REFerence {ref_level_dbm}")

    def set_nr5g_mode(self, mode: str = "occupiedBW") -> None:
        """
        Set 5G NR measurement mode.

        Valid options include (per manual, NR5G:MODE):
        - spectrumTuned
        - channelPower
        - occupiedBW
        - spectrumEmissionMask
        - adjacentChannelPower
        - multiAdjacentChannelPower
        - spuriousEmissionMask
        - constellation
        - beamScanner
        - CarrierAggregation
        - routeMap5GNR
        - powerVSTimeSymbol
        - powerVSTimeFrame
        """
        self.write(f"NR5G:MODE {mode}")

    # ---- example 5G NR measurements ----------------------------------

    def nr5g_measure_obw(self) -> float:
        """
        Run an Occupied Bandwidth measurement for 5G NR and return
        integrated power (as reported by NR5G:OBWidth:RESult:INTE:POWE?).

        NOTE:
        - Assumes NR5G mode is already set to 'occupiedBW' and
          that a valid 5G NR signal is present and locked.
        - For strict sequencing, you can send *OPC? after switching
          mode or changing frequency.
        """
        self.set_nr5g_mode("occupiedBW")
        # Optionally: self.query("*OPC?") to wait for completion
        resp = self.query("NR5G:OBWidth:RESult:INTE:POWE?")
        try:
            return float(resp)
        except ValueError:
            raise RuntimeError(f"Unexpected OBW result: {resp!r}")

    def nr5g_measure_aclr(
        self,
    ) -> Tuple[float, float, float]:
        """
        Example ACLR helper: returns (carrier_dBm, lower_rel_dB, upper_rel_dB).

        Uses (see NR5G:ACLR section in manual):
        - NR5G:ACLR:ABSolute1:LOWer?
        - NR5G:ACLR:ABSolute1:UPPer?
        - NR5G:ACLR:RELative1:LOWer?
        - NR5G:ACLR:RELative1:UPPer?

        Here we only return the relative dB for lower/upper, plus
        integrated carrier power via OBW as a simple reference.
        """
        self.set_nr5g_mode("adjacentChannelPower")
        # Carrier power via OBW integrated power:
        carrier_power = self.nr5g_measure_obw()

        lower_rel = float(self.query("NR5G:ACLR:RELative1:LOWer?"))
        upper_rel = float(self.query("NR5G:ACLR:RELative1:UPPer?"))

        return carrier_power, lower_rel, upper_rel

    def nr5g_measure_sem(
        self,
    ) -> Tuple[float, float]:
        """
        Example SEM (Spectrum Emission Mask) helper.

        Very simplified: just demonstrates how you might pull out
        pass/fail and a representative peak in upper side.

        Uses commands like:
        - NR5G:SEM:PEAK1:UPPer?
        - NR5G:SEM:PEAK1:UPPer:JUDGe?

        Returns:
            (peak_upper_dBm, pass_fail_flag)
        """
        self.set_nr5g_mode("spectrumEmissionMask")
        peak_upper = float(self.query("NR5G:SEM:PEAK1:UPPer?"))
        # pass/fail: typical result might be 0/1 or text
        judge = self.query("NR5G:SEM:PEAK1:UPPer:JUDGe?")
        return peak_upper, judge

    def nr5g_measure_evm_pdsch(self) -> dict:
        """
        Example EVM helper for PDSCH constellation.

        Uses:
        - NR5G:CONStellation:EVM:DATA:RMS?
        - NR5G:CONStellation:EVM:DATA:PEAK?
        - NR5G:CONStellation:EVM:PDSCH:QPSK?
        - NR5G:CONStellation:EVM:PDSCH:QAM16?
        - NR5G:CONStellation:EVM:PDSCH:QAM64?
        - NR5G:CONStellation:EVM:PDSCH:QAM256?

        Returns dict with EVM metrics (as floats where possible).
        """
        self.set_nr5g_mode("constellation")

        out = {}
        try:
            out["rms_data"] = float(
                self.query("NR5G:CONStellation:EVM:DATA:RMS?")
            )
        except ValueError:
            out["rms_data"] = None

        try:
            out["peak_data"] = float(
                self.query("NR5G:CONStellation:EVM:DATA:PEAK?")
            )
        except ValueError:
            out["peak_data"] = None

        for label, cmd in [
            ("qpsk", "NR5G:CONStellation:EVM:PDSCH:QPSK?"),
            ("qam16", "NR5G:CONStellation:EVM:PDSCH:QAM16?"),
            ("qam64", "NR5G:CONStellation:EVM:PDSCH:QAM64?"),
            ("qam256", "NR5G:CONStellation:EVM:PDSCH:QAM256?"),
        ]:
            try:
                out[label] = float(self.query(cmd))
            except ValueError:
                out[label] = None

        return out


# ----------------------------------------------------------------------
# Example CLI usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Simple VIAVI OneAdvisor Radio Analysis SCPI demo."
    )
    parser.add_argument("host", help="Instrument IP address")
    parser.add_argument(
        "--discovery-port", type=int, default=5025, help="Discovery port (default: 5025)"
    )
    parser.add_argument(
        "--scpi-port",
        type=int,
        default=None,
        help="Radio SCPI port (default: auto-discover via :PRTM:LIST?)",
    )
    args = parser.parse_args()

    inst = ViaviOneAdvisor(
        host=args.host,
        discovery_port=args.discovery_port,
        scpi_port=args.scpi_port,
    )

    try:
        print(f"Connecting to {args.host}...")
        inst.open()
        print("Connected.")
        print("IDN:", inst.idn())

        # --- Simple spectrum scan around 3.5 GHz (5G NR FR1)  ---
        print("\nConfiguring Spectrum Analyzer for 3.5 GHz, 100 MHz span...")
        inst.set_spectrum_mode("spectrumTuned")
        inst.configure_spectrum(
            center_hz=3.5e9,
            span_hz=100e6,
            rbw_auto=False,
            rbw_hz=100e3,
            ref_level_dbm=0.0,
            atten_mode="Auto",
        )
        time.sleep(0.5)
        trace = inst.get_spectrum_trace()
        print(f"Got {len(trace)} spectrum points at 3.5 GHz.")

        # --- 5G NR occupied bandwidth example -------------------
        print("\nConfiguring 5G NR Signal Analyzer...")
        inst.configure_nr5g_basic(
            center_hz=3.5e9,
            band="FR1",
            chan_standard=700,  # example; check your band code in manual
            chan_num=1,
            ref_level_dbm=0.0,
        )
        time.sleep(0.5)
        obw_power = inst.nr5g_measure_obw()
        print(f"5G NR OBW integrated power ≈ {obw_power:.2f} dBm (example).")

        # --- Wi-Fi 2.4 GHz scan example -------------------------
        print("\nScanning Wi-Fi 2.4 GHz band...")
        wifi_trace, center, span = inst.scan_wifi_band(
            band="2.4", span_hz=100e6, rbw_hz=100e3, ref_level_dbm=-10.0
        )
        print(
            f"Wi-Fi scan: {len(wifi_trace)} points, center={center/1e9:.3f} GHz, "
            f"span={span/1e6:.1f} MHz"
        )

    finally:
        inst.close()
        print("Connection closed.")
