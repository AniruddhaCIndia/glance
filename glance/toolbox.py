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