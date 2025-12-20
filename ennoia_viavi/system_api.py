import socket
from typing import List, Optional


class OneAdvisorSystemAPI:
    """
    System-level SCPI client for VIAVI OneAdvisor / CellAdvisor 5G.

    Responsibilities (default port 5025):
    - *IDN? / *RST / *OPC?
    - :SYSTem:APPLication:APPLications?
    - :SYSTem:APPLication:LAUNch <app> [port]
    - :PRTM:LIST?  (service / port discovery)
    """

    def __init__(self, host: str, port: int = 5025, timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    # -------------------- low-level socket --------------------

    def open(self) -> None:
        if self._sock is not None:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
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
        Send SCPI command, no response expected.
        """
        self._ensure_open()
        assert self._sock is not None
        line = (cmd.strip() + "\n").encode("ascii", errors="ignore")
        self._sock.sendall(line)

    def query(self, cmd: str) -> str:
        """
        Send SCPI query and return response as string (strip newline).
        """
        self.write(cmd)
        resp = self._recv_until(b"\n")
        return resp.decode("ascii", errors="ignore").strip()

    # -------------------- basic utilities --------------------

    def idn(self) -> str:
        return self.query("*IDN?")

    def reset(self) -> None:
        self.write("*RST")

    def opc(self) -> bool:
        try:
            return self.query("*OPC?").strip() in ("1", "ON", "On")
        except Exception:
            return False

    # -------------------- application control --------------------

    def list_applications(self) -> List[str]:
        """
        Return a list of application names from :SYST:APPL:APPLications?
        """
        resp = self.query(":SYSTem:APPLication:APPLications?")
        # Typically comma- or semicolon-separated
        parts = [p.strip() for p in resp.replace(";", ",").split(",") if p.strip()]
        return parts

    def launch_application(self, app_name: str, port: Optional[int] = None) -> None:
        """
        Launch application by name, optionally on specific test port.

        app_name: string from list_applications()
        """
        if port is None:
            cmd = f":SYSTem:APPLication:LAUNch {app_name}"
        else:
            cmd = f":SYSTem:APPLication:LAUNch {app_name},{port}"
        self.write(cmd)
        # Optional: wait for completion
        self.opc()

    # -------------------- service / port discovery --------------------

    def list_services_raw(self) -> str:
        """
        Return raw :PRTM:LIST? string which lists services and ports.
        """
        return self.query(":PRTM:LIST?")

    def get_radio_scpi_port(self) -> int:
        """
        Parse :PRTM:LIST? to find CA5G-SCPI / ONA-800-SCPI service port.
        Fallback to 5600 if not found.
        """
        import re

        raw = self.list_services_raw()
        m = re.search(r"(CA5G-SCPI|ONA-800-SCPI)\s*[:=]\s*(\d+)", raw)
        if m:
            return int(m.group(2))
        return 5600  # typical default
