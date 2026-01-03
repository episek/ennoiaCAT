import pyvisa

rm = pyvisa.ResourceManager()
resource = 'TCPIP0::192.168.1.100::5025::SOCKET'

try:
    instr = rm.open_resource(resource)
    instr.timeout = 5000
    instr.write_termination = '\n'
    instr.read_termination = '\n'

    print("Connected to:", instr.query("*IDN?"))
    instr.close()

except pyvisa.VisaIOError as e:
    print("Could not connect to instrument:", e)
