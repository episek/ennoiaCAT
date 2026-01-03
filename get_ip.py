import pyvisa

rm = pyvisa.ResourceManager()
resources = rm.list_resources()
for res in resources:
    try:
        inst = rm.open_resource(res)
        print(res, inst.query("*IDN?").strip())
        inst = rm.open_resource(res)
        parts = res.split(':')
        ip_address = parts[2]  # assuming IP is the 3th element
        print("IP Address:", ip_address)
    except:
        continue
		