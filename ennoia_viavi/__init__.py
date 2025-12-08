"""
ennoia_viavi

Unified Python API for VIAVI OneAdvisor / CellAdvisor 5G:

- OneAdvisorSystemAPI: system control (port 5025)
- OneAdvisorRadioAPI: radio analysis (spectrum + 5G NR) via Radio SCPI port

Typical usage:

    from ennoia_viavi import OneAdvisorSystemAPI, OneAdvisorRadioAPI

    sys = OneAdvisorSystemAPI("192.168.1.100")
    print(sys.idn())
    apps = sys.list_applications()
    sys.launch_application("RadioAnalysis")

    ra = OneAdvisorRadioAPI("192.168.1.100", system_api=sys)
    ra.open()
    ra.configure_spectrum(center_hz=3.5e9, span_hz=100e6)
    trace = ra.get_spectrum_trace()
    ra.close()
"""

from .system_api import OneAdvisorSystemAPI
from .radio_api import OneAdvisorRadioAPI

__all__ = [
    "OneAdvisorSystemAPI",
    "OneAdvisorRadioAPI",
]
