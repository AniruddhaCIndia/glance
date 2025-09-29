import numpy as np
from gwpy.timeseries import TimeSeries

def waveform_extender(waveform, data_dict, det):

    """
    Align and extend a waveform to match the strain data time segment of a given detector.

    This function adjusts the start and end times of a given waveform so that it
    matches the time span of the strain data for a specified detector. It trims
    or pads the waveform with zeros as needed to ensure consistent alignment.

    Parameters
    ----------
    waveform_h : TimeSeries
        TimeSeries object containing the waveform data to be extended or trimmed.
        Must have attributes `t0`, `epoch`, `dt`, and support slicing.
    data_dict : dict
        Dictionary containing strain data and metadata for each detector,
        typically returned from `strain_define`. Each detector key should contain:
            - "start_time": float, start time of strain data.
            - "duration": int, duration of strain data in seconds.
    det : str
        Detector name (e.g., "H1", "L1") whose strain data segment will be used
        to align the waveform.

    Returns
    -------
    strain_h : TimeSeries
        A TimeSeries object containing the extended or trimmed waveform
        aligned to match the time segment of the specified detector.

    Notes
    -----
    - This function assumes a global variable `delta_t` specifying the sampling interval.
    - Padding is done with zeros to extend the waveform; trimming removes samples
      that lie outside the desired segment.
    - The waveform is aligned such that its start time matches the detector’s
      strain data start time.

    Example
    -------
    >>> extended_waveform = waveform_extender(waveform_h, data_dict, "H1")
    >>> print(extended_waveform.t0)
    1126259446.0
    >>> print(len(extended_waveform))
    131072

    """

    waveform_h = waveform[f'{det}']
    delta_t = waveform_h.dt.value

    t_start_h = waveform_h.t0.value
    t_end_h = waveform_h.epoch.value + len(waveform_h) * delta_t

    td_start_h = data_dict[f'{det}']['start_time']
    td_end_h = data_dict[f'{det}']['start_time'] + data_dict[f'{det}']['duration']

    if t_start_h < td_start_h:
        trim_start_samples = int((td_start_h - t_start_h) * 1 / delta_t)
        waveform_h1 = waveform_h[trim_start_samples:]
    elif t_start_h > td_start_h:
        pad_start_samples = int((t_start_h - td_start_h) * 1 / delta_t)
        waveform_h1 = np.concatenate((np.zeros(pad_start_samples), waveform_h))

    if t_end_h > td_end_h:
        trim_end_samples = int((t_end_h - td_end_h) * 1 / delta_t)
        waveform_h1 = waveform_h1[:-trim_end_samples]
    elif t_end_h < td_end_h:
        pad_end_samples = int((td_end_h - t_end_h) * 1 / delta_t)
        waveform_h1 = np.concatenate((waveform_h1, np.zeros(pad_end_samples)))

    strain_h = TimeSeries(waveform_h1,
                        dt=delta_t,
                        t0=td_start_h,
                        dtype=float)
    return strain_h


def residual_definer(waveform, data_dict, det, duration, f_low, max_l_time):
    """
    Compute the residual signal between a detector's strain data and the model waveform.

    The residual is calculated by aligning the waveform with the detector strain
    data based on a given reference time, applying optional high-pass filtering,
    and then subtracting the waveform from the strain data over the relevant time interval.

    Parameters
    ----------
    waveform : dict
        Dictionary containing time-domain waveforms for detectors. Keys are detector names
        (e.g., 'H1', 'L1', 'V1') and values are TimeSeries objects.
    data_dict : dict
        Dictionary containing strain data and metadata for each detector. Should have the structure:
            data_dict[det] = {
                "strain": TimeSeries,
                "start_time": float,
                "duration": int,
                "sampling_frequency": float
            }
    det : str
        Detector name (e.g., "H1", "L1", or "V1") for which the residual should be computed.
    duration : float
        Duration of the signal segment to compare (in seconds).
    f_low : float
        Low-frequency cutoff for high-pass filtering (in Hz).
    max_l_time : float
        Reference time (typically the maximum likelihood geocenter time) used for aligning
        the waveform and strain data.

    Returns
    -------
    aligned_res : TimeSeries
        The residual TimeSeries obtained by subtracting the model waveform
        from the detector's strain data in the aligned time segment.

    Notes
    -----
    - The function determines the time indices corresponding to the aligned start
      and end points of the segment based on `max_l_time` and `duration`.
    - High-pass filtering is optionally applied to remove low-frequency noise components.
    - Currently, high-pass filtering is applied only when specified by `f_low`.
    """
    idx_H1 = np.argmin( abs(max_l_time - duration/2 - np.array(data_dict[det]['strain'].times)) )
    idx_H2 = np.argmin( abs(max_l_time + duration/2 - np.array(data_dict[det]['strain'].times)) )
    aligned_res = data_dict[det]["strain"].highpass(f_low, type='iir')[int(idx_H1) : int(idx_H2) ] - waveform[det][int(idx_H1) : int(idx_H2) ].highpass(f_low, type='iir')
    aligned_res = data_dict[det]["strain"][int(idx_H1) : int(idx_H2) ].highpass(f_low) - waveform[det][int(idx_H1) : int(idx_H2) ]
    aligned_res = data_dict[det]["strain"][int(idx_H1) : int(idx_H2) ] - waveform[det][int(idx_H1) : int(idx_H2) ]
    return aligned_res

