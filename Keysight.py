import pyvisa
import numpy as np
import pylab as pl

class FieldFox:
    def __init__(self, resource_str):
        rm = pyvisa.ResourceManager()
        self.inst = rm.open_resource(resource_str)  # This is the VISA instrument
        self.inst.timeout = 5000  # milliseconds

    def open(self):
        try:
            self.inst = self.rm.open_resource(self.resource_name)
            self.inst.timeout = 5000
            print("Connected to:", self.inst.query("*IDN?"))
        except Exception as e:
            print("Failed to connect:", e)

    def write(self, command):
        if self.inst:
            self.inst.write(command)
        else:
            raise RuntimeError("Instrument not connected.")

    def query(self, command):
        if self.inst:
            return self.inst.query(command)
        else:
            raise RuntimeError("Instrument not connected.")

    def set_mode_sa(self):
        self.inst.write(":INSTrument:SELect 'SA'")

    def set_span(self, span_hz):
        self.inst.write(f":FREQuency:SPAN {span_hz}")

    def set_center(self, center_hz):
        self.inst.write(f":FREQuency:CENTer {center_hz}")

    def set_start(self, start_hz):
        self.inst.write(f":FREQuency:STARt {start_hz}")

    def set_stop(self, stop_hz):
        self.inst.write(f":FREQuency:STOP {stop_hz}")
        
    def set_start_stop(self, start_hz, stop_hz):
        self.inst.write(f":FREQ:STAR {start_hz}")
        self.inst.write(f":FREQ:STOP {stop_hz}")

    def set_rbw(self, rbw_hz):
        self.inst.write(f":BANDwidth:RESolution {rbw_hz}")

    def set_level(self, ref_level_dbm):
        self.inst.write(f":DISPlay:WINDow:TRACe:Y:RLEVel {ref_level_dbm}")
        
    def get_points(self):
        return int(self.inst.query("SWE:POIN?").strip())
        
    def restart_continuous(self):
        self.inst.write("INIT:CONT ON")

    def single_sweep(self):
        # self.write(":INITiate:IMMediate")
        # self.instrument.query("*OPC?")  # Wait for sweep to complete        
        self.inst.write(":INITiate:IMMediate; *WAI")
        self.inst.write(":INIT:CONT OFF")    # Pause continuous updates
        self.inst.write(":INIT:IMM")         # Manual sweep
        self.inst.query("*OPC?")             # Wait until it's done            

    def close(self):
        if self.inst is not None:
            self.inst.close()
            
    def query(self, cmd):
        return self.inst.query(cmd).strip()

    def fetch_trace(self):
        #self.inst.write(":FORMat:DATA ASCII")
        trace = self.inst.query(":TRACe:DATA?")
        #self.inst.trace_data = np.array([float(val) for val in trace.strip().split(",")])
        return trace

    def fetch_frequencies(self):
        start = float(self.inst.query(":FREQuency:STARt?"))
        stop = float(self.inst.query(":FREQuency:STOP?"))
        points = int(self.inst.query(":SWEep:POINts?"))
        self.inst.frequencies = np.linspace(start, stop, points)
        return self.inst.frequencies

    def plot_trace(self):
        if self.inst.frequencies is None:
            self.inst.inst.inst.fetch_frequencies()
        if self.inst.inst.trace_data is None:
            self.inst.fetch_trace()
        pl.grid(True)
        pl.plot(self.inst.frequencies, self.inst.trace_data)
        pl.xlabel("Frequency (Hz)")
        pl.ylabel("Amplitude (dBm)")
        pl.title("FieldFox Spectrum Trace")
        pl.show()

    def write_csv(self, filename="output.csv"):
        if self.inst.frequencies is None or self.inst.trace_data is None:
            self.fetch_frequencies()
            self.inst.fetch_trace()
        with open(filename, "w") as f:
            for f_, a in zip(self.inst.frequencies, self.inst.trace_data):
                f.write(f"{f_},{a}\n")

# if __name__ == '__main__':
    # inst = FieldFox()
    # inst.set_center(1e9)
    # inst.set_span(100e6)
    # inst.set_rbw(100e3)
    # inst.set_level(-10)
    # inst.single_sweep()
    # inst.fetch_frequencies()
    # trace = inst.fetch_trace()
    # inst.plot_trace()
    # inst.write_csv("fieldfox_output.csv")
