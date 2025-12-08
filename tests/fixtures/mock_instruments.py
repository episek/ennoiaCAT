"""
Mock instrument implementations for testing
"""
from unittest.mock import MagicMock


class MockTinySA:
    """Mock TinySA spectrum analyzer"""

    def __init__(self, port='COM3'):
        self.port = port
        self.connected = False
        self.start_freq = 100e6
        self.stop_freq = 1e9
        self.points = 101

    def connect(self):
        self.connected = True
        return True

    def disconnect(self):
        self.connected = False
        return True

    def scan(self):
        """Return mock scan data"""
        import numpy as np
        frequencies = np.linspace(self.start_freq, self.stop_freq, self.points)
        power = -80 + 20 * np.random.rand(self.points)
        return frequencies, power

    def set_frequency_range(self, start, stop):
        self.start_freq = start
        self.stop_freq = stop


class MockViaviOneAdvisor:
    """Mock Viavi OneAdvisor"""

    def __init__(self, ip='192.168.1.100'):
        self.ip = ip
        self.connected = False

    def open(self):
        self.connected = True

    def close(self):
        self.connected = False

    def query(self, command):
        if command == "*IDN?":
            return "Viavi Solutions,OneAdvisor-800,SN123456,1.0.0"
        return "OK"

    def write(self, command):
        pass


class MockKeysightFieldFox:
    """Mock Keysight FieldFox"""

    def __init__(self, resource):
        self.resource = resource
        self.connected = False
        self.read_termination = '\n'
        self.write_termination = '\n'
        self.timeout = 5000

    def query(self, command):
        if command == "*IDN?":
            return "Keysight Technologies,N9918A,MY12345678,A.01.23"
        return "OK"

    def write(self, command):
        pass

    def close(self):
        self.connected = False


class MockRohdeSchwarz:
    """Mock Rohde & Schwarz NRQ6"""

    def __init__(self, resource):
        self.resource = resource
        self.connected = False

    def query(self, command):
        if command == "*IDN?":
            return "Rohde&Schwarz,NRQ6,123456,1.0.0"
        return "OK"

    def write(self, command):
        pass

    def close(self):
        self.connected = False
