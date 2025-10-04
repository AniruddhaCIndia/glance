import pycbc
from pycbc.filter import match
from pycbc.filter import overlap
import numpy as np
from scipy.signal import stft


def waveform_match(h1, h2, flow):

    tlen = max(len(h1), len(h2))

    h1.resize(tlen)
    h2.resize(tlen)

    delta_f = 1.0 / h1.duration
    flen = tlen//2 + 1

    psd = pycbc.psd.analytical.aLIGOAdVO4T1800545(flen, delta_f, flow)
    m, i = match(h1, h2, psd=psd, low_frequency_cutoff=flow)

    return m, i

def find_valid_chunk(time, data, duration, t_s, t_e, return_indices=False):
    """
    Efficiently find the first valid continuous chunk of given duration
    with no NaN values, even for very large arrays.

    Parameters
    ----------
    time : array-like
        Time axis corresponding to the data (same length as data).
    data : array-like
        Data array (must match time length).
    duration : float
        Desired chunk duration in seconds.
    return_indices : bool, optional (default=False)
        If True, return the (start, end) indices instead of sliced arrays.

    Returns
    -------
    time_chunk, data_chunk : ndarray
        Time and data arrays for the valid chunk of length `duration`.
    OR
    (start_idx, end_idx) : tuple of int
        If return_indices=True.

    Raises
    ------
    ValueError
        If no valid chunk is found.
    """
    time = np.asarray(time)
    data = np.asarray(data)

    if len(time) != len(data):
        raise ValueError("time and data must have the same length.")

    dt = np.median(np.diff(time))  # sampling interval
    samples_needed = int(np.round(duration / dt))

    if samples_needed > len(data):
        raise ValueError("Requested duration is longer than the data length.")

    # Restrict to search window if given
    if t_s is not None:
        start_idx = np.searchsorted(time, t_s, side="left")
    else:
        start_idx = 0
    if t_e is not None:
        end_idx = np.searchsorted(time, t_e, side="right")
    else:
        end_idx = len(time)

    time = time[start_idx:end_idx]
    data = data[start_idx:end_idx]

    # mask invalid points
    invalid = np.isnan(data).astype(int)

    # cumulative sum of invalids, pad with leading zero
    csum = np.cumsum(np.insert(invalid, 0, 0))

    # window sum of invalids = number of NaNs in each window
    window_invalids = csum[samples_needed:] - csum[:-samples_needed]

    # Find first window with 0 invalids
    idx_candidates = np.where(window_invalids == 0)[0]

    if len(idx_candidates) == 0:
        raise ValueError(f"No valid {duration:.2f}-second chunk found.")

    start_idx = idx_candidates[0]
    end_idx = start_idx + samples_needed

    if return_indices:
        return start_idx, end_idx
    else:
        return time[start_idx:end_idx], data[start_idx:end_idx]


def waveform_overlap(htilde, stilde, psd, df):

    safe_psd = np.where(psd == 0, 1e10, psd)
    
    inner = 4 * df * np.sum(htilde.conj() * stilde / safe_psd)
    sigma_h = np.sqrt(4 * df * np.sum(np.abs(htilde)**2 / safe_psd))
    sigma_s = np.sqrt(4 * df * np.sum(np.abs(stilde)**2 / safe_psd))

    return inner / (sigma_h * sigma_s)

def waveform_overlap_pycbc(h1, h2, flow):

    tlen = max(len(h1), len(h2))

    h1.resize(tlen)
    h2.resize(tlen)

    delta_f = 1.0 / h1.duration
    flen = tlen//2 + 1

    psd = pycbc.psd.analytical.aLIGOAdVO4T1800545(flen, delta_f, flow)
    m = overlap(h1, h2, psd=psd, low_frequency_cutoff=flow, normalized=True)

    return m

def timeseries_time2frequency_finder(timeseries, sampling_freq, desired_time, epoch, nperseg=64):
    """
    Find the dominant frequency at a specific time in the time series.

    Parameters:
        timeseries (array): The input signal.
        sampling_freq (float): Sampling frequency of the signal.
        desired_time (float): The time at which to find the dominant frequency.
        nperseg (int): Number of samples per segment for STFT. Default is 256.

    Returns:
        float: The dominant frequency at the given time.
    """
    f, t, Zxx = stft(timeseries, fs=sampling_freq, nperseg=nperseg)
    t= t+ epoch
    desired_time_idx = np.argmin(np.abs(t - desired_time))  # Closest time index
    dominant_frequency = f[np.argmax(np.abs(Zxx[:, desired_time_idx]))]  # Find dominant frequency
    return dominant_frequency

def timeseries_frequency2time_finder(timeseries, sampling_freq, desired_freq,  epoch, nperseg=64,):
    """
    Find the time when the desired frequency is dominant in the time series.

    Parameters:
        timeseries (array): The input signal.
        sampling_freq (float): Sampling frequency of the signal.
        desired_freq (float): The frequency at which to find the dominant time.
        nperseg (int): Number of samples per segment for STFT. Default is 256.

    Returns:
        float: The time at which the desired frequency is dominant.
    """
    f, t, Zxx = stft(timeseries, fs=sampling_freq, nperseg=nperseg)
    t = epoch + t
    desired_freq_idx = np.argmin(np.abs(f - desired_freq))  # Closest frequency index
    dominant_time = t[np.argmax(np.abs(Zxx[desired_freq_idx, :]))]  # Find dominant time
    return dominant_time

def find_larger_indices(time_series: np.ndarray, t_a: float, t_b: float) -> tuple:
    """
    Find indices in a time series for values closest but larger than t_a and t_b.
    
    Parameters:
    - time_series (np.ndarray): The time series array, assumed sorted
    - t_a (float): The first target time
    - t_b (float): The second target time
    
    Returns:
    - tuple: Indices for the closest larger values of t_a and t_b
    """
    idx_a = np.searchsorted(time_series, t_a, side='right')
    idx_b = np.searchsorted(time_series, t_b, side='right')
    
    return idx_a, idx_b

