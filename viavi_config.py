# viavi_config.py
import os
import json
import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

class ViaviHelper:
    """Helper class for Viavi OneAdvisor spectrum analyzer operations"""

    def __init__(self):
        pass

    def get_system_prompt(self):
        """Get system prompt for AI assistant"""
        system_prompt = (
            "You are Ennoia AI, an assistant for RF spectrum analysis using Viavi OneAdvisor. "
            "Help users configure, troubleshoot, and operate the Viavi OneAdvisor spectrum analyzer. "
            "Provide clear, concise answers about frequency settings, sweep configurations, "
            "and spectrum analysis tasks. Answer with complete sentences."
        )
        return system_prompt

    def get_few_shot_examples(self):
        """Get few-shot examples for AI training"""
        few_shot_examples = [
            {
                "role": "user",
                "content": "Set the start frequency to 600 MHz and stop frequency to 900 MHz"
            },
            {
                "role": "assistant",
                "content": (
                    "I'll configure the spectrum analyzer to scan from 600 MHz to 900 MHz. "
                    "This will set the center frequency to 750 MHz with a span of 300 MHz."
                )
            },
            {
                "role": "user",
                "content": "Scan for 5G signals in the C-band"
            },
            {
                "role": "assistant",
                "content": (
                    "I'll configure the analyzer to scan the 5G C-band spectrum from 3.3 GHz to 3.8 GHz. "
                    "This will help detect any 5G NR carriers in this frequency range."
                )
            },
        ]
        return few_shot_examples

    @staticmethod
    def parse_freq(val, default=None):
        """Converts values like 3500, '3.5 GHz', '3500 MHz', etc. → Hz."""
        if val is None:
            return default

        # Already numeric
        if isinstance(val, (int, float)):
            return float(val)

        # Handle list/array/tuple from LLM or JSON
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 0:
                return default
            return ViaviHelper.parse_freq(val[0], default=default)

        s = str(val).strip().lower()

        # Match "<number> <optional unit>"
        m = re.match(r'^([0-9]*\.?[0-9]+)\s*(ghz|mhz|khz|hz)?$', s)
        if m:
            num = float(m.group(1))
            unit = m.group(2) or "hz"

            if unit == "ghz":
                return num * 1e9
            if unit == "mhz":
                return num * 1e6
            if unit == "khz":
                return num * 1e3
            return num  # plain Hz

        # Fallback: try to parse as plain float (already in Hz)
        try:
            return float(s)
        except Exception:
            return default

    @staticmethod
    def extract_start_stop(text):
        """Extract natural-language frequencies from text."""
        if not text:
            return None, None

        txt = text.lower()

        def conv(num, unit):
            num = float(num)
            if unit == "g": return num * 1e9
            if unit == "m": return num * 1e6
            if unit == "k": return num * 1e3
            return num

        # Case 1: explicit "start freq" and "stop freq"
        start_match = re.search(r"start\s*(?:freq(?:uency)?)?\s*([\d\.]+)\s*(g|m|k)?hz", txt)
        stop_match  = re.search(r"stop\s*(?:freq(?:uency)?)?\s*([\d\.]+)\s*(g|m|k)?hz", txt)

        if start_match and stop_match:
            s = conv(start_match.group(1), start_match.group(2))
            e = conv(stop_match.group(1), stop_match.group(2))
            return (min(s, e), max(s, e))

        # Case 2: "600 to 900 MHz" / "600-900 MHz" / "from 600 to 900 MHz"
        range_match = re.search(
            r"(?:from\s+)?([\d\.]+)\s*(g|m|k)?hz?\s*(?:to|-|–)\s*([\d\.]+)\s*(g|m|k)?hz?",
            txt
        )
        if range_match:
            n1, u1, n2, u2 = range_match.group(1), range_match.group(2), range_match.group(3), range_match.group(4)
            s = conv(n1, u1)
            e = conv(n2, u2)
            return (min(s, e), max(s, e))

        # Case 3: "scan 600 900 MHz" (space separated numbers with unit after)
        space_match = re.search(
            r"(?:scan|sweep|measure)\s+([\d\.]+)\s+([\d\.]+)\s*(g|m|k)?hz?",
            txt
        )
        if space_match:
            n1 = float(space_match.group(1))
            n2 = float(space_match.group(2))
            unit = space_match.group(3)
            s = conv(n1, unit)
            e = conv(n2, unit)
            return (min(s, e), max(s, e))

        # Case 4: Just two numbers with units "300 MHz 900 MHz"
        two_freq_match = re.findall(r"([\d\.]+)\s*(g|m|k)?hz", txt)
        if len(two_freq_match) >= 2:
            s = conv(two_freq_match[0][0], two_freq_match[0][1])
            e = conv(two_freq_match[1][0], two_freq_match[1][1])
            return (min(s, e), max(s, e))

        return None, None

    @staticmethod
    def auto_rbw(span_hz):
        """Basic rule-of-thumb RBW."""
        if span_hz <= 10e6:
            return 10e3
        if span_hz <= 100e6:
            return 100e3
        return 1e6

    @staticmethod
    def find_peaks(y, max_peaks=5, min_dist=5):
        """Simple peak finder (no SciPy)."""
        peaks = []
        for i in range(1, len(y) - 1):
            if y[i] > y[i - 1] and y[i] > y[i + 1]:
                peaks.append((i, y[i]))
        peaks.sort(key=lambda p: p[1], reverse=True)
        selected = []
        for idx, val in peaks:
            if all(abs(idx - s[0]) >= min_dist for s in selected):
                selected.append((idx, val))
            if len(selected) >= max_peaks:
                break
        return selected

    @staticmethod
    def bandpower_linear(freqs_hz, trace_dbm, f1_hz, f2_hz):
        """Approximate band power (dBm) by integrating over [f1,f2]."""
        freqs_hz = np.array(freqs_hz)
        trace_dbm = np.array(trace_dbm)

        if f2_hz <= f1_hz:
            return None

        mask = (freqs_hz >= f1_hz) & (freqs_hz <= f2_hz)
        if not np.any(mask):
            return None

        # Convert dBm to mW
        p_mw = 10 ** (trace_dbm[mask] / 10.0)
        f_band = freqs_hz[mask]

        # Approximate integral via trapezoidal rule
        power_mw = np.trapz(p_mw, f_band) / (f2_hz - f1_hz)
        if power_mw <= 0:
            return None
        return 10 * np.log10(power_mw)

    @staticmethod
    def detect_5gnr_like_carriers(freqs_hz, trace_dbm, min_bw_mhz=5.0, rel_thresh_db=6.0):
        """
        Very simple 5G NR-like carrier detector:
        - Find strong peaks.
        - Estimate -3 dB bandwidth around each.
        - Mark carriers with BW >= min_bw_mhz in 600 MHz – 6 GHz.
        """
        freqs_hz = np.array(freqs_hz)
        trace_dbm = np.array(trace_dbm)
        peaks = ViaviHelper.find_peaks(trace_dbm, max_peaks=10, min_dist=10)
        carriers = []

        for idx, val in peaks:
            f0 = freqs_hz[idx]
            if f0 < 600e6 or f0 > 6e9:
                continue

            # -3 dB bandwidth estimation
            half_power = val - 3.0
            left = idx
            while left > 0 and trace_dbm[left] > half_power:
                left -= 1
            right = idx
            n = len(trace_dbm)
            while right < n - 1 and trace_dbm[right] > half_power:
                right += 1

            bw_hz = freqs_hz[right] - freqs_hz[left]
            bw_mhz = bw_hz / 1e6
            if bw_mhz < min_bw_mhz:
                continue

            carriers.append(
                {
                    "Center (MHz)": f0 / 1e6,
                    "Approx BW (MHz)": bw_mhz,
                    "Peak Power (dBm)": float(val),
                }
            )
        return carriers

    @staticmethod
    def wifi_band_from_freq(freq_mhz):
        """Classify WiFi band from RF frequency."""
        if 2400 <= freq_mhz <= 2500:
            return "2.4 GHz"
        if 5000 <= freq_mhz <= 5900:
            return "5 GHz"
        if 5925 <= freq_mhz <= 7125:
            return "6 GHz"
        return None

    @staticmethod
    def detect_wifi_like_carriers(freqs_hz, trace_dbm, rel_thresh_db=10.0):
        """
        Very simple WiFi-like detector:
        - Look for peaks in WiFi bands.
        - Report strongest candidates.
        """
        freqs_hz = np.array(freqs_hz)
        trace_dbm = np.array(trace_dbm)
        peaks = ViaviHelper.find_peaks(trace_dbm, max_peaks=20, min_dist=5)
        if not peaks:
            return []

        max_val = max(v for _, v in peaks)
        carriers = []
        for idx, val in peaks:
            if max_val - val > rel_thresh_db:
                continue
            f_hz = freqs_hz[idx]
            f_mhz = f_hz / 1e6
            band = ViaviHelper.wifi_band_from_freq(f_mhz)
            if band is None:
                continue
            carriers.append(
                {
                    "Center (MHz)": f_mhz,
                    "Band": band,
                    "Peak Power (dBm)": float(val),
                }
            )
        return carriers

    @staticmethod
    def classify_span_wifi_bands(freqs_hz):
        """Classify which WiFi bands are covered by the current span"""
        if not freqs_hz:
            return []
        f_min = min(freqs_hz) / 1e6
        f_max = max(freqs_hz) / 1e6
        bands = []
        # overlap checks in MHz
        if f_max >= 2400 and f_min <= 2505:
            bands.append("2.4 GHz")
        if f_max >= 5150 and f_min <= 5850:
            bands.append("5 GHz")
        if f_max >= 5925 and f_min <= 7125:
            bands.append("6 GHz")
        return bands

    def get_operator_frequencies(self):
        """Load operator frequency table from JSON"""
        file_path = 'operator_table.json'
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                return data
            except json.JSONDecodeError as e:
                st.warning(f"JSON decode error: {e}")
                return []
        else:
            st.warning(f"File not found: {file_path}")
            return []

    def analyze_signal_peaks(self, sstr, freq_mhz, operator_table, window_size=5, peak_height=-75, peak_distance=10):
        """
        Analyze signal peaks, group them by 3GPP band, and reduce the number of peaks.

        Parameters:
        - sstr: List or array of signal strengths.
        - freq_mhz: List or array of frequencies (in MHz) corresponding to `sstr`.
        - operator_table: List of dictionaries containing operator band info.
        - window_size: Number of samples before and after peak to average.
        - peak_height: Minimum signal strength (in dBm) to consider a peak.
        - peak_distance: Minimum distance (in MHz) between peaks to consider them as distinct.

        Returns:
        - List of dictionaries with band match information.
        """
        from scipy.signal import find_peaks as scipy_find_peaks

        peaks, _ = scipy_find_peaks(sstr, height=peak_height)
        grouped_peaks = []

        for peak in peaks:
            freq = freq_mhz[peak]
            closest_band = None
            min_diff = float('inf')

            # Find the closest operator band to the peak frequency
            for band in operator_table:
                try:
                    uplink = [int(x.strip()) for x in band['Uplink Frequency (MHz)'].split(' - ')]
                    downlink = [int(x.strip()) for x in band['Downlink Frequency (MHz)'].split(' - ')]
                except Exception:
                    continue  # Skip malformed entries

                if uplink[0] <= freq <= uplink[1] or downlink[0] <= freq <= downlink[1]:
                    diff = min(
                        abs(uplink[0] - freq), abs(uplink[1] - freq),
                        abs(downlink[0] - freq), abs(downlink[1] - freq)
                    )
                    if diff < min_diff:
                        min_diff = diff
                        closest_band = band

            if closest_band:
                # Group peaks by band and distance
                found_group = False
                for group in grouped_peaks:
                    # Check if the peak is within the distance of an existing group
                    if abs(group['frequency'] - freq) <= peak_distance:
                        group['peaks'].append(peak)
                        found_group = True
                        break

                if not found_group:
                    # Create a new group for this peak
                    grouped_peaks.append({
                        'band': closest_band,
                        'frequency': freq,
                        'peaks': [peak]
                    })

        # Process the grouped peaks and compute their average strength
        result = []
        for group in grouped_peaks:
            band = group['band']
            all_peaks = group['peaks']
            # Get the average signal strength for all peaks in this group
            avg_strength = round(float(np.mean([sstr[peak] for peak in all_peaks])), 2)

            # Use the first and last peak's frequencies to define the frequency range
            start_idx = max(all_peaks[0] - window_size, 0)
            end_idx = min(all_peaks[-1] + window_size, len(sstr))
            freq_range = f"{int(freq_mhz[start_idx])} - {int(freq_mhz[end_idx - 1])}"

            result.append({
                "operator": band.get('Operators', 'Unknown'),
                "strength": avg_strength,
                "technology": band.get('Technology', 'Unknown'),
                "service": "Mobile",
                "frequency_range": freq_range,
                "band_3GPP": band.get('3GPP Band', 'Unknown'),
                "source": "Internal Database"
            })

        return result
