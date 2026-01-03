# import pyvisa
# rm = pyvisa.ResourceManager()  # or adjust path
# print(rm.list_resources())

import pyvisa
from RsInstrument import RsInstrument, RsInstrException

# Define VISA libraries to test
visa_libs = {
    "NI-VISA": r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\visa64.dll",
    "R&S VISA": r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\RsVisa64.dll",
}

# Define potential VISA resource strings for a given IP
ip_address = "192.168.1.100"
resource_variants = [
    f"TCPIP::{ip_address}::INSTR",
    f"TCPIP0::{ip_address}::INSTR",
    f"TCPIP0::{ip_address}::5025::SOCKET",
    f"TCPIP::{ip_address}::5025::SOCKET",
    f"TCPIP0::{ip_address}::inst0::INSTR",
]

# Try each VISA lib + resource string combination
results = []

for visa_name, visa_path in visa_libs.items():
    try:
        rm = pyvisa.ResourceManager(visa_path)
        for resource in resource_variants:
            try:
                instr = RsInstrument(resource, id_query=True, reset=False,
                                     options=f"SelectVisa='ni', VisaLibrary='{visa_path}'")
                idn = instr.query("*IDN?")
                results.append((visa_name, resource, "✅ SUCCESS", idn.strip()))
                instr.close()
            except RsInstrumentException as e:
                results.append((visa_name, resource, "❌ FAIL", str(e).split('\n')[0]))
    except Exception as e:
        results.append((visa_name, "ALL", "❌ VISA INIT FAIL", str(e).split('\n')[0]))

# Display results
for row in results:
    print(f"{row[0]:<12} | {row[1]:<40} | {row[2]:<10} | {row[3]}")
