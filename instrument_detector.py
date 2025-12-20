"""
Instrument Detection and Registry Module
Automatically detects available test instruments and manages instrument selection
"""
import serial
from serial.tools import list_ports
import socket
import subprocess
import pyvisa
from typing import Dict, List, Optional, Tuple
from enum import Enum


class InstrumentType(Enum):
    """Supported instrument types"""
    TINYSA = "TinySA Spectrum Analyzer"
    VIAVI = "Viavi OneAdvisor"
    MAVENIR_RU = "Mavenir 5G NR Radio Unit"
    CISCO_NCS540 = "Cisco NCS540"
    ROHDE_SCHWARZ = "Rohde & Schwarz NRQ6"
    AUKUA = "Aukua Systems"
    KEYSIGHT = "Keysight FieldFox"


class InstrumentInfo:
    """Information about a detected instrument"""
    def __init__(self, instrument_type: InstrumentType, connection_info: Dict,
                 display_name: str, config_module: str):
        self.instrument_type = instrument_type
        self.connection_info = connection_info
        self.display_name = display_name
        self.config_module = config_module

    def __repr__(self):
        return f"<{self.display_name}: {self.connection_info}>"


class InstrumentDetector:
    """Detects and manages available test instruments"""

    # USB VID/PID for TinySA
    TINYSA_VID = 0x0483
    TINYSA_PID = 0x5740

    def __init__(self):
        self.detected_instruments: List[InstrumentInfo] = []

    def detect_tinysa(self) -> List[InstrumentInfo]:
        """Detect TinySA spectrum analyzer devices"""
        instruments = []
        device_list = list_ports.comports()

        for device in device_list:
            if device.vid == self.TINYSA_VID and device.pid == self.TINYSA_PID:
                info = InstrumentInfo(
                    instrument_type=InstrumentType.TINYSA,
                    connection_info={
                        'port': device.device,
                        'vid': device.vid,
                        'pid': device.pid,
                        'description': device.description
                    },
                    display_name=f"TinySA on {device.device}",
                    config_module="tinySA_config"
                )
                instruments.append(info)

        return instruments

    def detect_viavi(self, ip_range: Optional[List[str]] = None) -> List[InstrumentInfo]:
        """
        Detect Viavi OneAdvisor devices on network

        IMPORTANT: To connect to Viavi OneAdvisor, you must configure your laptop's
        network adapter IP address to 192.168.1.100 (same subnet as the Viavi device).

        Args:
            ip_range: List of IP addresses to check. If None, checks common defaults.
        """
        instruments = []

        if ip_range is None:
            ip_range = ["192.168.1.100"]  # Default Viavi IP

        for ip in ip_range:
            try:
                # Try to connect to common SCPI port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((ip, 5025))  # SCPI port
                sock.close()

                if result == 0:
                    info = InstrumentInfo(
                        instrument_type=InstrumentType.VIAVI,
                        connection_info={
                            'ip': ip,
                            'protocol': 'SCPI',
                            'port': 5025
                        },
                        display_name=f"Viavi OneAdvisor @ {ip}",
                        config_module="ennoia_viavi.system_api"
                    )
                    instruments.append(info)
            except Exception:
                continue

        return instruments

    def detect_cisco_ncs540(self) -> List[InstrumentInfo]:
        """Detect Cisco NCS540 devices via serial ports"""
        instruments = []
        device_list = list_ports.comports()

        # Look for common USB-to-serial adapters that might be NCS540
        for device in device_list:
            # Common FTDI or CP2102 chips used in Cisco console cables
            if device.vid in [0x0403, 0x10C4]:  # FTDI or Silicon Labs
                info = InstrumentInfo(
                    instrument_type=InstrumentType.CISCO_NCS540,
                    connection_info={
                        'port': device.device,
                        'vid': device.vid,
                        'pid': device.pid,
                        'description': device.description
                    },
                    display_name=f"Cisco NCS540 (potential) on {device.device}",
                    config_module="ncs540_serial"
                )
                instruments.append(info)

        return instruments

    def detect_mavenir_ru(self, ip_range: Optional[List[str]] = None) -> List[InstrumentInfo]:
        """
        Detect Mavenir 5G NR Radio Unit devices

        Args:
            ip_range: List of IP addresses to check
        """
        instruments = []

        if ip_range is None:
            ip_range = ["10.10.10.10"]  # Default RU IP from config

        for ip in ip_range:
            try:
                # Try NETCONF/REST ports
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((ip, 830))  # NETCONF port
                sock.close()

                if result == 0:
                    info = InstrumentInfo(
                        instrument_type=InstrumentType.MAVENIR_RU,
                        connection_info={
                            'ip': ip,
                            'protocol': 'NETCONF',
                            'port': 830
                        },
                        display_name=f"Mavenir 5G NR RU @ {ip}",
                        config_module="mav_config"
                    )
                    instruments.append(info)
            except Exception:
                continue

        return instruments

    def detect_keysight(self, ip_range: Optional[List[str]] = None) -> List[InstrumentInfo]:
        """
        Detect Keysight FieldFox devices

        Args:
            ip_range: List of IP addresses to check
        """
        instruments = []

        if ip_range is None:
            ip_range = ["192.168.1.100"]  # Default FieldFox IP

        try:
            rm = pyvisa.ResourceManager()
            for ip in ip_range:
                try:
                    resource_str = f"TCPIP0::{ip}::inst0::INSTR"
                    inst = rm.open_resource(resource_str, timeout=2000)
                    idn = inst.query("*IDN?")
                    inst.close()

                    if "Keysight" in idn or "Agilent" in idn:
                        info = InstrumentInfo(
                            instrument_type=InstrumentType.KEYSIGHT,
                            connection_info={
                                'ip': ip,
                                'resource': resource_str,
                                'idn': idn.strip()
                            },
                            display_name=f"Keysight FieldFox @ {ip}",
                            config_module="KS_config"
                        )
                        instruments.append(info)
                except Exception:
                    continue
        except Exception:
            pass  # PyVISA not available

        return instruments

    def detect_rohde_schwarz(self) -> List[InstrumentInfo]:
        """Detect Rohde & Schwarz NRQ6 devices via PyVISA"""
        instruments = []

        try:
            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()

            for resource in resources:
                try:
                    # Check if it matches NRQ pattern
                    if 'nrq' in resource.lower() or 'hislip' in resource.lower():
                        inst = rm.open_resource(resource, timeout=2000)
                        idn = inst.query("*IDN?")
                        inst.close()

                        if "Rohde" in idn or "R&S" in idn or "NRQ" in idn:
                            info = InstrumentInfo(
                                instrument_type=InstrumentType.ROHDE_SCHWARZ,
                                connection_info={
                                    'resource': resource,
                                    'idn': idn.strip()
                                },
                                display_name=f"Rohde & Schwarz via {resource}",
                                config_module="RS_config"
                            )
                            instruments.append(info)
                except Exception:
                    continue
        except Exception:
            pass  # PyVISA not available

        return instruments

    def detect_aukua(self) -> List[InstrumentInfo]:
        """
        Detect Aukua Systems devices

        Note: Aukua detection logic needs to be refined based on actual hardware
        """
        instruments = []

        try:
            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()

            for resource in resources:
                try:
                    inst = rm.open_resource(resource, timeout=2000)
                    idn = inst.query("*IDN?")
                    inst.close()

                    if "Aukua" in idn or "AUKUA" in idn:
                        info = InstrumentInfo(
                            instrument_type=InstrumentType.AUKUA,
                            connection_info={
                                'resource': resource,
                                'idn': idn.strip()
                            },
                            display_name=f"Aukua Systems via {resource}",
                            config_module="AK_config"
                        )
                        instruments.append(info)
                except Exception:
                    continue
        except Exception:
            pass

        return instruments

    def detect_pyvisa_instruments(self) -> List[InstrumentInfo]:
        """Detect instruments via PyVISA (for general SCPI instruments)"""
        instruments = []

        try:
            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()

            for resource in resources:
                # Try to identify the instrument type
                try:
                    inst = rm.open_resource(resource, timeout=2000)
                    idn = inst.query("*IDN?")
                    inst.close()

                    # Check if it's a known instrument (skip if already detected)
                    if "Viavi" in idn or "OneAdvisor" in idn:
                        info = InstrumentInfo(
                            instrument_type=InstrumentType.VIAVI,
                            connection_info={
                                'resource': resource,
                                'idn': idn
                            },
                            display_name=f"Viavi via VISA: {resource}",
                            config_module="ennoia_viavi.system_api"
                        )
                        instruments.append(info)
                except Exception:
                    continue

        except Exception:
            pass  # PyVISA not available or no instruments

        return instruments

    def detect_all(self, viavi_ips: Optional[List[str]] = None,
                   mavenir_ips: Optional[List[str]] = None,
                   keysight_ips: Optional[List[str]] = None) -> List[InstrumentInfo]:
        """
        Detect all available instruments

        Args:
            viavi_ips: List of Viavi IP addresses to check
            mavenir_ips: List of Mavenir RU IP addresses to check
            keysight_ips: List of Keysight IP addresses to check

        Returns:
            List of detected instruments
        """
        self.detected_instruments = []

        # Detect TinySA (USB)
        self.detected_instruments.extend(self.detect_tinysa())

        # Detect Viavi (Network)
        self.detected_instruments.extend(self.detect_viavi(viavi_ips))

        # Detect Cisco NCS540 (Serial)
        self.detected_instruments.extend(self.detect_cisco_ncs540())

        # Detect Mavenir RU (Network)
        self.detected_instruments.extend(self.detect_mavenir_ru(mavenir_ips))

        # Detect Keysight (Network/VISA)
        self.detected_instruments.extend(self.detect_keysight(keysight_ips))

        # Detect Rohde & Schwarz (VISA)
        self.detected_instruments.extend(self.detect_rohde_schwarz())

        # Detect Aukua (VISA)
        self.detected_instruments.extend(self.detect_aukua())

        # Detect via PyVISA (catch-all)
        # self.detected_instruments.extend(self.detect_pyvisa_instruments())

        return self.detected_instruments

    def get_instruments_by_type(self, instrument_type: InstrumentType) -> List[InstrumentInfo]:
        """Get all detected instruments of a specific type"""
        return [inst for inst in self.detected_instruments
                if inst.instrument_type == instrument_type]

    def get_instrument_count(self) -> Dict[InstrumentType, int]:
        """Get count of each instrument type"""
        counts = {itype: 0 for itype in InstrumentType}
        for inst in self.detected_instruments:
            counts[inst.instrument_type] += 1
        return counts


def main():
    """Test the detector"""
    detector = InstrumentDetector()
    instruments = detector.detect_all()

    print(f"Detected {len(instruments)} instrument(s):")
    for inst in instruments:
        print(f"  - {inst.display_name}")
        print(f"    Type: {inst.instrument_type.value}")
        print(f"    Config: {inst.config_module}")
        print(f"    Connection: {inst.connection_info}")
        print()

    counts = detector.get_instrument_count()
    print("\nSummary:")
    for itype, count in counts.items():
        if count > 0:
            print(f"  {itype.value}: {count}")


if __name__ == "__main__":
    main()
