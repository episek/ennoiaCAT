from serial.tools import list_ports

ports = list_ports.comports()

for port in ports:
    print(f"Device: {port.device}")
    print(f"  Description: {port.description}")
    print(f"  VID: {hex(port.vid) if port.vid else 'N/A'}")
    print(f"  PID: {hex(port.pid) if port.pid else 'N/A'}")
    print(f"  Serial Number: {port.serial_number}")
    print()
	