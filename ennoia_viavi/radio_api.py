import socket
import time
from typing import List, Optional, Tuple

from .system_api import OneAdvisorSystemAPI


class OneAdvisorRadioAPI:
    """
    Radio Analysis SCPI client for VIAVI OneAdvisor / CellAdvisor 5G.

    Responsibilities:
    - Connect to Radio SCPI port (e.g. 5600) directly or via system discovery.
    - Spectrum Analyzer (SPECtrum:...)
    - 5G NR Analyzer (NR5G:...)
    - Wi-Fi band scans via spectrum mode
    """

    def __init__(
        self,
        host: str,
        scpi_port: Optional[int] = None,
        timeout: float = 5.0,
        system_api: Optional[OneAdvisorSystemAPI] = None,
        discovery_port: int = 5025,
    ) -> None:
        """
        host: OneAdvisor IP
        scpi_port: Radio Analysis SCPI port. If None, we auto-discover it.
        system_api: optional OneAdvisorSystemAPI; if not provided, we do a
                    minimal discovery by directly connecting to discovery_port.
        """
        self.host = host
        self.scpi_port = scpi_port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self.system_api = system_api
        self.discovery_port = discovery_port

    # -------------------- connection --------------------

    def _discover_scpi_port(self) -> int:
        """
        Use OneAdvisorSystemAPI / :PRTM:LIST? to infer Radio SCPI port.
        """
        if self.system_api is not None:
            return self.system_api.get_radio_scpi_port()

        # Minimal inline discovery if system_api not supplied
        from .system_api import OneAdvisorSystemAPI  # local import

        sys = OneAdvisorSystemAPI(self.host, port=self.discovery_port, timeout=self.timeout)
        try:
            sys.open()
            port = sys.get_radio_scpi_port()
        finally:
            sys.close()
        return port

    def open(self) -> None:
        """
        Open socket to Radio SCPI service.
        """
        if self._sock is not None:
            return
        if self.scpi_port is None:
            self.scpi_port = self._discover_scpi_port()

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.host, self.scpi_port))
        self._sock = s

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _ensure_open(self) -> None:
        if self._sock is None:
            self.open()

    def _recv_until(self, terminator: bytes = b"\n") -> bytes:
        if self._sock is None:
            raise RuntimeError("Socket not open")
        chunks: list[bytes] = []
        while True:
            chunk = self._sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if terminator in chunk:
                break
        return b"".join(chunks)

    # -------------------- SCPI primitives --------------------

    def write(self, cmd: str) -> None:
        """
        Send SCPI command to Radio Analysis service.
        """
        self._ensure_open()
        assert self._sock is not None
        line = (cmd.strip() + "\n").encode("ascii", errors="ignore")
        self._sock.sendall(line)

    def query(self, cmd: str) -> str:
        """
        Send SCPI query and return response (stripped).
        """
        self.write(cmd)
        resp = self._recv_until(b"\n")
        return resp.decode("ascii", errors="ignore").strip()

    # -------------------- common utilities --------------------

    def idn(self) -> str:
        return self.query("*IDN?")

    def opc(self) -> bool:
        try:
            return self.query("*OPC?").strip() in ("1", "ON", "On")
        except Exception:
            return False

    # ==================================================================
    # Spectrum Analyzer (SPECtrum:...)
    # ==================================================================

    def set_spectrum_mode(self, mode: str = "spectrumTuned") -> None:
        """
        Set Spectrum Analyzer measurement mode.

        Example modes (depending on licensed options):
        - spectrumTuned
        - channelPower
        - occupiedBW
        - spectrumEmissionMask
        - adjacentChannelPower
        - multiAdjacentChannelPower
        - spuriousEmissionMask
        - fieldStrength
        - powerMeter
        """
        self.write(f"SPECtrum:MODE {mode}")

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
        atten_mode: Optional[str] = None,  # "Auto"|"Manual" etc.
    ) -> None:
        """
        Configure Spectrum Analyzer frequency, RBW/VBW, and amplitude.
        """
        # Frequency
        if center_hz is not None:
            self.write(f"SPECtrum:FREQuency:CENTer {center_hz} Hz")
        if span_hz is not None:
            self.write(f"SPECtrum:FREQuency:SPAN {span_hz} Hz")
        if start_hz is not None:
            self.write(f"SPECtrum:FREQuency:STARt {start_hz} Hz")
        if stop_hz is not None:
            self.write(f"SPECtrum:FREQuency:STOP {stop_hz} Hz")

        # RBW
        if rbw_auto is not None:
            self.write(f"SPECtrum:RBW:MODE {'Auto' if rbw_auto else 'Manual'}")
        if rbw_hz is not None:
            self.write(f"SPECtrum:RBW {rbw_hz} Hz")

        # VBW
        if vbw_auto is not None:
            self.write(f"SPECtrum:VBW:MODE {'Auto' if vbw_auto else 'Manual'}")
        if vbw_hz is not None:
            self.write(f"SPECtrum:VBW {vbw_hz} Hz")

        # Amplitude
        if ref_level_dbm is not None:
            self.write(f"SPECtrum:AMPlitude:REFerence {ref_level_dbm}")
        if atten_db is not None:
            self.write(f"SPECtrum:AMPlitude:ATTenuation {atten_db}")
        if atten_mode is not None:
            self.write(f"SPECtrum:AMPlitude:MODE {atten_mode}")

    def get_spectrum_trace(self, drop_padding: bool = True) -> List[float]:
        """
        Return current spectrum trace (power values) as list of floats.

        On some VIAVI analyzers the SCPI trace buffer has a fixed length
        (e.g., 3001 / 3007 points) but only the first N points are really used
        for the current sweep, and the rest are returned as 0.0.

        This helper optionally strips trailing padding zeros so that the
        length of the returned list matches the "real" trace length.
        """
        data_str = self.query("SPECtrum:TRACe:DATA?")
        parts = [p for p in data_str.replace("\n", "").split(",") if p.strip()]

        vals: List[float] = []
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                continue

        if drop_padding and vals:
            # Drop trailing exact zeros (padding)
            while vals and vals[-1] == 0.0:
                vals.pop()

        return vals
    
    def get_spectrum_xaxis(self) -> Tuple[float, float, int]:
        """
        Get x-axis parameters for current spectrum:
        returns (start_hz, stop_hz, num_points)
        """
        start = float(self.query("SPECtrum:FREQuency:STARt?"))
        stop = float(self.query("SPECtrum:FREQuency:STOP?"))

        try:
            npts = int(self.query("SPECtrum:TRACe:POINts?"))
        except Exception:
            # Fallback: use the length of the data
            npts = len(self.get_spectrum_trace())

        return start, stop, npts

    # -------------------- Wi-Fi band helpers (spectrum-based) --------------------

    def scan_wifi_band(
        self,
        band: str = "2.4",
        span_hz: float = 100e6,
        rbw_hz: float = 100e3,
        ref_level_dbm: float = -10.0,
        settle_time_s: float = 0.3,
    ) -> Tuple[List[float], float, float]:
        """
        Quick Wi-Fi band scan using spectrum mode.

        band: '2.4', '5', or '6'
        span_hz: frequency span
        rbw_hz: RBW (fixed)
        ref_level_dbm: reference level
        settle_time_s: time to wait for sweep to settle
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
        time.sleep(settle_time_s)
        trace = self.get_spectrum_trace()
        return trace, center_hz, span_hz

    # ==================================================================
    # 5G NR Analyzer (NR5G:...)
    # ==================================================================

    def configure_nr5g_basic(
        self,
        center_hz: float,
        band: str = "FR1",  # "FR1" or "FR2"
        chan_standard: Optional[int] = None,
        chan_num: Optional[int] = None,
        ref_level_dbm: Optional[float] = None,
    ) -> None:
        """
        Basic 5G NR analyzer setup for downlink signals.
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

        Example modes (NR5G:MODE):
        - spectrumTuned
        - channelPower
        - occupiedBW
        - spectrumEmissionMask
        - adjacentChannelPower
        - constellation
        - beamScanner
        - powerVSTimeSymbol
        - powerVSTimeFrame
        """
        self.write(f"NR5G:MODE {mode}")

    # -------------------- NR 5G example measurements --------------------

    def nr5g_measure_obw(self) -> float:
        """
        Measure occupied-bandwidth integrated power (simple example).

        Assumes NR5G:MODE is set to occupiedBW.
        """
        self.set_nr5g_mode("occupiedBW")
        self.opc()  # optional wait
        resp = self.query("NR5G:OBWidth:RESult:INTE:POWE?")
        try:
            return float(resp)
        except ValueError:
            raise RuntimeError(f"Unexpected OBW result: {resp!r}")

    def nr5g_measure_aclr(self) -> Tuple[float, float, float]:
        """
        Example ACLR helper: returns (carrier_dBm, lower_rel_dB, upper_rel_dB).

        Carrier power is taken from OBW integrated power (as a simple reference).
        """
        self.set_nr5g_mode("adjacentChannelPower")
        self.opc()
        carrier_power = self.nr5g_measure_obw()

        lower_rel = float(self.query("NR5G:ACLR:RELative1:LOWer?"))
        upper_rel = float(self.query("NR5G:ACLR:RELative1:UPPer?"))

        return carrier_power, lower_rel, upper_rel

    def nr5g_measure_sem(self) -> Tuple[float, str]:
        """
        Example SEM (Spectrum Emission Mask) helper.

        Returns:
            (peak_upper_dBm, pass_fail_string)
        """
        self.set_nr5g_mode("spectrumEmissionMask")
        self.opc()
        peak_upper = float(self.query("NR5G:SEM:PEAK1:UPPer?"))
        judge = self.query("NR5G:SEM:PEAK1:UPPer:JUDGe?")
        return peak_upper, judge

    def nr5g_measure_evm_pdsch(self) -> dict:
        """
        Example PDSCH EVM helper.

        Returns dict with EVM percentages where available:
            {
                "rms_data": ...,
                "peak_data": ...,
                "qpsk": ...,
                "qam16": ...,
                "qam64": ...,
                "qam256": ...
            }
        """
        self.set_nr5g_mode("constellation")
        self.opc()

        out: dict = {}
        # Overall RMS / PEAK
        try:
            out["rms_data"] = float(self.query("NR5G:CONStellation:EVM:DATA:RMS?"))
        except ValueError:
            out["rms_data"] = None

        try:
            out["peak_data"] = float(self.query("NR5G:CONStellation:EVM:DATA:PEAK?"))
        except ValueError:
            out["peak_data"] = None

        # PDSCH by modulation
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

    # -------------------- high-level “one-shot” NR test --------------------

    def nr5g_quick_summary(
        self,
        center_hz: float,
        band: str = "FR1",
        chan_standard: Optional[int] = None,
        chan_num: Optional[int] = None,
        ref_level_dbm: Optional[float] = 0.0,
    ) -> dict:
        """
        One-shot 5G NR summary:
        - config frequency / band
        - OBW power
        - ACLR
        - SEM (upper)
        - EVM summary

        Returns JSON-friendly dict.
        """
        self.configure_nr5g_basic(
            center_hz=center_hz,
            band=band,
            chan_standard=chan_standard,
            chan_num=chan_num,
            ref_level_dbm=ref_level_dbm,
        )

        # give it some time to lock / settle
        time.sleep(0.5)

        summary: dict = {
            "center_hz": center_hz,
            "band": band,
            "chan_standard": chan_standard,
            "chan_num": chan_num,
            "ref_level_dbm": ref_level_dbm,
        }

        try:
            summary["obw_power_dbm"] = self.nr5g_measure_obw()
        except Exception as e:
            summary["obw_power_dbm_error"] = str(e)

        try:
            carrier, lower, upper = self.nr5g_measure_aclr()
            summary["aclr"] = {
                "carrier_dbm": carrier,
                "lower_rel_db": lower,
                "upper_rel_db": upper,
            }
        except Exception as e:
            summary["aclr_error"] = str(e)

        try:
            peak_upper, judge = self.nr5g_measure_sem()
            summary["sem"] = {
                "peak_upper_dbm": peak_upper,
                "upper_judgement": judge,
            }
        except Exception as e:
            summary["sem_error"] = str(e)

        try:
            summary["evm"] = self.nr5g_measure_evm_pdsch()
        except Exception as e:
            summary["evm_error"] = str(e)

        return summary
