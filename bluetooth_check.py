import asyncio
from bleak import BleakScanner

async def run():
    devices = await BleakScanner.discover()
    for d in devices:
        print(f"{d.name} - {d.address} - RSSI: {d.rssi}")

asyncio.run(run())
