import numpy as np
from dataclasses import dataclass
from scipy.optimize import curve_fit

@dataclass
class TimescaleResult:
    tau_int: float
    A: float
    B: float
    r2: float

def _exp_decay(tau: np.ndarray, A: float, tau_int: float, B: float) -> np.ndarray:
    return A * np.exp(-tau / tau_int) + B

def compute_autocorr_timescale(
    spike_times: np.ndarray,
    t_start: float,
    t_end: float,
    bin_size: float = 0.01,
    max_lag: float = 0.5,
) -> TimescaleResult:
    """
    Compute intrinsic timescale from a single neuron's spike times using
    autocorrelation + exponential fit.

    spike_times : 1D array of spike times (seconds)
    t_start, t_end : analysis window (seconds)
    bin_size : bin size for spike counts (seconds)
    max_lag : maximum lag to consider in fit (seconds)
    """
    # Bin spikes into counts
    edges = np.arange(t_start, t_end + bin_size, bin_size)
    counts, _ = np.histogram(spike_times, bins=edges)
    counts = counts.astype(float)

    # Remove mean
    counts -= counts.mean()

    # Autocorrelation via FFT
    n = len(counts)
    f = np.fft.fft(counts, n=2*n)
    ac = np.fft.ifft(f * np.conjugate(f)).real[:n]
    ac /= ac[0]  # normalize so C(0) = 1

    # Build lag axis
    taus = np.arange(n) * bin_size
    mask = (taus > 0) & (taus <= max_lag)
    taus_fit = taus[mask]
    ac_fit = ac[mask]

    # Initial guesses
    p0 = [1.0, 0.1, 0.0]  # A, tau_int, B

    try:
        popt, _ = curve_fit(_exp_decay, taus_fit, ac_fit, p0=p0, bounds=([0, 1e-3, -1], [2, 10.0, 1]))
        A, tau_int, B = popt
        y_pred = _exp_decay(taus_fit, *popt)
        ss_res = np.sum((ac_fit - y_pred)**2)
        ss_tot = np.sum((ac_fit - np.mean(ac_fit))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    except Exception:
        A, tau_int, B, r2 = np.nan, np.nan, np.nan, np.nan

    return TimescaleResult(tau_int=tau_int, A=A, B=B, r2=r2)
