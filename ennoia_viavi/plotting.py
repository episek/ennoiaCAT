"""
plotting.py

Matplotlib helpers for visualizing VIAVI OneAdvisor traces.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrum_trace(
    trace_dbm: List[float],
    start_mhz: float,
    stop_mhz: float,
    ax: Optional[plt.Axes] = None,
    title: str = "Spectrum Trace",
    xlabel: str = "Frequency (GHz)",
    ylabel: str = "Power (dBm)",
    grid: bool = True,
) -> plt.Axes:
    """
    Plot a single spectrum trace vs frequency.

    Parameters
    ----------
    trace_dbm : list of float
        Power values in dBm (from SPECtrum:TRACe:DATA?).
    start_mhz : float
        Start frequency of the trace.
    stop_mhz : float
        Stop frequency of the trace.
    ax : matplotlib.axes.Axes, optional
        Existing Axes to draw on; if None, a new figure/axes is created.
    title, xlabel, ylabel : str
        Labels for the plot.
    grid : bool
        Whether to show a grid.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    n = len(trace_dbm)
    if n == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    freqs_mhz = np.linspace(start_mhz, stop_mhz, n)
    freqs_ghz = freqs_mhz / 1e3

    ax.plot(freqs_ghz, trace_dbm)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if grid:
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    return ax


def plot_wifi_scan(
    trace_dbm: List[float],
    center_hz: float,
    span_hz: float,
    band_label: str = "Wi-Fi",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Convenience helper for plotting a Wi-Fi band scan.

    Parameters
    ----------
    trace_dbm : list of float
        Power values from scan_wifi_band().
    center_hz : float
        Center frequency used for the scan.
    span_hz : float
        Span used for the scan.
    band_label : str
        Label (e.g., '2.4 GHz Wi-Fi').
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    start_hz = center_hz - span_hz / 2
    stop_hz = center_hz + span_hz / 2
    title = f"{band_label} Spectrum Scan"

    return plot_spectrum_trace(
        trace_dbm=trace_dbm,
        start_hz=start_hz,
        stop_hz=stop_hz,
        ax=ax,
        title=title,
    )
