import serial
import time
import serial.tools.list_ports
import psutil
import threading
import streamlit as st

# -------------------------------
# Helper functions
# -------------------------------
def list_serial_ports():
    ports = []
    for port in serial.tools.list_ports.comports():
        desc = port.description or ""
        hwid = port.hwid or ""
        # 🔹 Filter: keep only USB serial, drop Bluetooth
        if "USB" in desc.upper() or "USB" in hwid.upper():
            ports.append(port.device)
    return ports

def list_eth_interfaces():
    """Return dict of network interfaces with their IP addresses"""
    interfaces = {}
    addrs = psutil.net_if_addrs()
    for iface, addr_list in addrs.items():
        for addr in addr_list:
            if addr.family == 2:  # AF_INET (IPv4)
                interfaces[iface] = addr.address
    return interfaces

def auto_connect_any(username="cisco", password="cisco"):
    """
    Connect to the first available COM port with a device.
    Returns (conn, port) or (None, None) if nothing found.
    """
    ports = [p.device for p in serial.tools.list_ports.comports()]
    for port in ports:
        try:
            conn = NCS540Serial(port=port, username=username, password=password)
            # If login works, return immediately
            return conn, port
        except Exception:
            continue
    return None, None

def use_selected_interfaces():
    """Helper that consumes iface_out and iface_in"""
    if "iface_out" not in st.session_state or "iface_in" not in st.session_state:
        st.error("❌ Interfaces not set yet")
        return

    iface_out = st.session_state.iface_out
    iface_in  = st.session_state.iface_in
    ip_out = st.session_state.selected_ip_out
    ip_in  = st.session_state.selected_ip_in

    #st.write(f"Output iface: {iface_out} (IP {ip_out})")
    #st.write(f"Capture iface: {iface_in} (IP {ip_in})")

    
class NCS540Serial:
    def __init__(self, port="COM8", username="cisco", password="cisco"):
        self.ser = serial.Serial(
            port=port,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )
        self.username = username
        self.password = password
        self.keep_alive_thread = None
        self.keep_alive_running = False
        self.login()
        self.disable_paging()

    def login(self):
        self.ser.reset_input_buffer()
        time.sleep(0.5)
        self.ser.write(b"\n")
        time.sleep(0.5)
        output = self.ser.read_all().decode(errors="ignore")

        if "Username:" in output:
            self.ser.write((self.username + "\r\n").encode())
            time.sleep(0.5)
        if "Password:" in output:
            self.ser.write((self.password + "\r\n").encode())
            time.sleep(0.5)
        self.ser.read_all()


    def disable_paging(self):
        """Disable --More-- pagination"""
        self.send_cmd("terminal length 0", wait=1)

    def send_cmd(self, cmd, wait=1.0):
        """Send CLI command and return clean output"""
        self.ser.reset_input_buffer()
        self.ser.write((cmd.strip() + "\r\n").encode())
        time.sleep(wait)
        output = self.ser.read_all().decode(errors="ignore")

        # Detect dropped session → re-login
        if "Username:" in output or "Password:" in output:
            self.login()
            return self.send_cmd(cmd, wait)

        # Clean output
        cleaned = output.replace("\r", "")

        return self._clean_output(output)
        # # Filter common security warning
        # cleaned_lines = []
        # for line in cleaned.splitlines():
            # if "SECURITY-PSLIB" in line:
                # continue
            # cleaned_lines.append(line)
        # return "\n".join(cleaned_lines)

    def _clean_output(self, raw):
        lines = []
        for line in raw.replace("\r", "").splitlines():
            if "SECURITY-PSLIB" in line:
                continue
            if line.strip().endswith("#") or line.strip().startswith("RP/"):
                continue
            lines.append(line)
        return "\n".join(lines)

    def start_keep_alive(self, interval=60):
        """Start background thread to keep session alive with newlines."""
        if hasattr(self, "keep_alive_thread") and self.keep_alive_thread and self.keep_alive_thread.is_alive():
            return  # already running
        self.keep_alive_running = True
        self.keep_alive_thread = threading.Thread(
            target=self._keep_alive_loop, args=(interval,), daemon=True
        )
        self.keep_alive_thread.start()

    def _keep_alive_loop(self, interval):
        while self.keep_alive_running:
            try:
                # send harmless newline to keep session alive
                self.ser.write(b"\n")
            except Exception:
                pass
            time.sleep(interval)


    def stop_keep_alive(self):
        self.keep_alive_running = False

    def close(self):
        self.ser.close()

