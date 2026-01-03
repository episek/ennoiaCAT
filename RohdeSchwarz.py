import json
from RsInstrument import RsInstrument
import pyvisa
import numpy as np
import pylab as pl
from typing import List, Tuple, ClassVar

def load_rs_config(path="rs_config.json"):
    with open(path, "r") as f:
        return json.load(f)

def connect_to_instrument(name=None, config=None):
    if config is None:
        config = load_rs_config()

    if name is None:
        name = config.get("default_instrument")

    inst_cfg = config["instruments"][name]

    options = f"SelectVisa='{inst_cfg['visa_library']}'"

    self.inst = RsInstrument(
        inst_cfg["resource"],
        id_query=inst_cfg.get("id_query", True),
        reset=inst_cfg.get("reset", False),
        options=options
    )

    return self.inst


# def apply_rf_settings(self, settings):
    # if "center_frequency_hz" in settings:
        # self.inst.write(f"SENSe:FREQuency:CENTer {settings['center_frequency_hz']}")
    # if "start_frequency_hz" in settings:
        # self.inst.write(f"SENSe:FREQuency:STARt {settings['start_frequency_hz']}")
    # if "stop_frequency_hz" in settings:
        # self.inst.write(f"SENSe:FREQuency:STOP {settings['stop_frequency_hz']}")
    # if "span_hz" in settings:
        # self.inst.write(f"SENSe:FREQuency:SPAN {settings['span_hz']}")
    # if "rbw_hz" in settings:
        # self.inst.write(f"SENSe:BANDwidth:RESolution {settings['rbw_hz']}")

    # if "sample_rate_sa_per_s" in settings:
        # self.inst.write(f"SENSe:BANDwidth:SRATe:CUV {settings['sample_rate_sa_per_s']}")
    # if "trace_length_samples" in settings:
        # self.inst.write(f"SENSE:TRACe:IQ:RLENgth {int(settings['trace_length_samples'])}")
    # if "sweep_time_s" in settings:
        # self.inst.write(f"SENSe:SWETime {settings['sweep_time_s']}")
    # if "averaging_enable" in settings:
        # on_off = "ON" if settings["averaging_enable"] else "OFF"
        # self.inst.write(f"SENSe:AVERage:STATe {on_off}")
    # if "averaging_count" in settings:
        # self.inst.write(f"SENSe:AVERage:COUNt {int(settings['averaging_count'])}")

class NRQ:

    def __init__(self, resource_str):
        rm = pyvisa.ResourceManager()
        self.inst = rm.open_resource(resource_str)  # This is the VISA instrument
        self.inst.timeout = 5000  # milliseconds

    def write(self, command):
        if self.inst:
            self.inst.write(command)
        else:
            raise RuntimeError("self.instument not connected.")

    def query(self, command):
        if self.inst:
            return self.inst.query(command)
        else:
            raise RuntimeError("self.instument not connected.")

    def set_mode_iq(self):
        self.inst.write('SENSe:FUNCtion "XTIMe:VOLT:IQ"')

    def set_span(self, span_hz):
        self.inst.write(f":FREQuency:SPAN {span_hz}")

    def set_center(self, center_hz):
        self.inst.write(f"SENSe:FREQuency:CENTer {center_hz}")

    def set_start(self, start_hz):
        self.inst.write(f"SENSe:FREQuency:STARt {start_hz}")

    def set_stop(self, stop_hz):
        self.inst.write(f"SENSe:FREQuency:STOP {stop_hz}")
        
    def set_start_stop(self, start_hz, stop_hz):
        self.inst.write(f"SENSe:FREQuency:STARt {start_hz}")
        self.inst.write(f"SENSe:FREQuency:STOP {stop_hz}")

    def set_rbw(self, rbw_hz):
        self.inst.write(f"SENSe:BANDwidth:RESolution {rbw_hz}")

    def set_level(self, ref_level_dbm):
        self.inst.write(f":DISPlay:WINDow:TRACe:Y:RLEVel {ref_level_dbm}")
        
    def set_bw_res_manual(self):
        self.inst.write('SENSE:BANDwidth:RESolution:TYPE:AUTO:STATe OFF')
        
    def set_bw_res_normal(self):
        self.inst.write('SENSE:BANDwidth:RESolution:TYPE NORMal')

    def set_bw_res(self, bw_res):
        self.inst.write('SENSE:BANDwidth:RES {bw_res}''')        
        
    def set_trace_length(self, trace_length):    
        self.inst.write(f"SENSE:TRACe:IQ:RLENgth {trace_length}") 

    def get_points(self):
        return int(self.query("SWE:POIN?").strip())
        
    def query_bin_or_ascii_float_list(self, query: str) -> List[float]:
        try:
            a = self.query_bin_or_ascii_float_list('FETCh1?')
            print("Fetched data:", a)
            return self._core.io.query_bin_or_ascii_float_list(query)
        except Exception as e:
            print("Error during fetch:", e)
            return ([])  
            
    def restart_continuous(self):
        self.inst.write("INIT:CONT ON")

    def single_sweep(self):
        # self.inst.write(":INITiate:IMMediate")
        # self.instrument.query("*OPC?")  # Wait for sweep to complete        
        self.inst.write(":INITiate:IMMediate; *WAI")
        self.inst.write(":INIT:CONT OFF")    # Pause continuous updates
        self.inst.write(":INIT:IMM")         # Manual sweep
        self.inst.query("*OPC?")             # Wait until it's done            

    def close(self):
        if self.inst is not None:
            self.inst.close()
            
    def query_center(self):
        return self.inst.query('SENSe:FREQuency:CENTer?')

    def fetch_trace(self):
        #self.inst.write(":FORMat:DATA ASCII")
        trace = self.inst.query(":TRACe:DATA?")
        #self.inst.trace_data = np.array([float(val) for val in trace.strip().split(",")])
        return trace

    def fetch_frequencies(self):
        start = float(self.query(":FREQuency:STARt?"))
        stop = float(self.query(":FREQuency:STOP?"))
        points = int(self.query(":SWEep:POINts?"))
        self.inst.frequencies = np.linspace(start, stop, points)
        return self.inst.frequencies

    def plot_trace(self):
        if self.inst.frequencies is None:
            self.inst.inst.inst.fetch_frequencies()
        if self.inst.inst.trace_data is None:
            self.inst.fetch_trace()
        pl.grid(True)
        pl.plot(self.frequencies, self.inst.trace_data)
        pl.xlabel("Frequency (Hz)")
        pl.ylabel("Amplitude (dBm)")
        pl.title("FieldFox Spectrum Trace")
        pl.show()

    def write_csv(self, filename="output.csv"):
        if self.inst.frequencies is None or self.inst.trace_data is None:
            self.inst.fetch_frequencies()
            self.inst.fetch_trace()
        with open(filename, "w") as f:
            for f_, a in zip(self.frequencies, self.inst.trace_data):
                f.write(f"{f_},{a}\n")




# # Example usage
# config = load_rs_config()
# self.inst = connect_to_self.instument("NRQ6", config)
# print(self.query("*IDN?"))
# self.inst.close()
