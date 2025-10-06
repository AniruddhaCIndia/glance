import numpy as np
from gwpy.timeseries import TimeSeries
import os
import h5py
from pycbc.detector import Detector


def strain_define(strain_paths):
    """
    Load and organize strain data from multiple HDF5 files.

    This function reads strain data from HDF5 files, extracts metadata
    from filenames, and stores the data along with metadata in a dictionary.

    Parameters
    ----------
    strain_paths : list of str
        List of file paths to strain data files in HDF5 format.
        Each filename is expected to follow the pattern:
        <detector>-<something>-<start_time>-<duration>.hdf5

    Returns
    -------
    data_dict : dict
        Dictionary containing strain data and metadata for each detector.
        Keys are detector names extracted from filenames. Each value is a
        dictionary with the following structure:
            - "strain": TimeSeries object containing the strain data.
            - "start_time": float, start time of the strain data segment.
            - "duration": int, duration of the segment in seconds.
            - "sampling_frequency": int, sampling frequency (Hz).

    Notes
    -----
    - This function assumes a fixed sampling frequency of 4096 Hz.
    - The filename must contain metadata in the format:
      <detector>-<something>-<start_time>-<duration>.hdf5
      where `start_time` is a float and `duration` is an integer.
    - Requires `h5py` and `gwpy.timeseries.TimeSeries`.

    Example
    -------
    >>> strain_paths = ["H1-L1-1126259446-32.hdf5", "L1-H1-1126259446-32.hdf5"]
    >>> data_dict = strain_define(strain_paths)
    >>> print(data_dict.keys())
    dict_keys(['H1', 'L1'])
    """
    strain_filepaths = strain_paths
    data_dict = {}

    for file in strain_filepaths:
        file_name = os.path.basename(file)
        key = file_name.split("-")[0]

        parts = file_name.split("-")
        start_time = float(parts[2])
        duration = int(parts[3].split(".")[0])
        sampling_frequency = 4096

        with h5py.File("/data/achakraborty/GWTC_4_files/" + file, 'r') as f:
            strain_dataset = f['strain/Strain']
            file_content = strain_dataset[:]

        strain_data = TimeSeries(
            file_content,
            dt=1/sampling_frequency,
            t0=start_time,
            dtype=float
        )

        data_dict[key] = {
            "strain": strain_data,
            "start_time": start_time,
            "duration": duration,
            "sampling_frequency": sampling_frequency,
        }

    print(data_dict.keys())
    return data_dict


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


def antenna_pattern_matrix(det1, det2, ra, dec, tc, pol, mat_inv=True):
    detector1 = Detector(det1)
    detector2 = Detector(det2)

    fp1, fc1 = detector1.antenna_pattern(ra, dec, pol, tc)
    fp2, fc2 = detector2.antenna_pattern(ra, dec, pol, tc)

    mat_12 = np.array([
        [fp1, fc1],
        [fp2, fc2]
    ])

    mat_i_12 = np.linalg.inv(mat_12)

    return mat_i_12 if mat_inv else mat_12


def polarization_extractor(mat_i_12, data1, data2):
    hp_12 = mat_i_12[0][0] * data1 + mat_i_12[0][1] * data2
    hc_12 = mat_i_12[1][0] * data1 + mat_i_12[1][1] * data2
    return hp_12, hc_12


def extract_aligned_data(det, data, ra, dec, t_ref, points_before, points_after):
    gps = t_ref + Detector(det).time_delay_from_earth_center(ra, dec, t_ref)
    times = np.array(data.sample_times)
    idx = np.searchsorted(times, gps, 'right')
    sliced_data = np.array(data[idx - points_before : idx + points_after])
    sliced_times = times[idx - points_before : idx + points_after]
    return sliced_data, sliced_times, gps


def cross_correlation(det1, det2, data1, data2, data1n, data2n, ra, dec, tc, tcn, pol, poln, chop_duration, steps=64, polarization=True):
    duration = chop_duration
    delta_t = data1.sample_time[1] - data1.sample_time[0]
    points_before = int((duration/2 - 0.1) / delta_t)
    points_after = int((duration/2 + 0.1)/ delta_t)
    
    # Aligned data for signal
    data1c, x1c, gps1 = extract_aligned_data(det1, data1, ra, dec, tc, points_before, points_after)
    data2c, x2c, gps2 = extract_aligned_data(det2, data2, ra, dec, tc, points_before, points_after)

    # Aligned data for noise
    data1nc, x1nc, gps1n = extract_aligned_data(det1, data1n, ra, dec, tcn, points_before, points_after)
    data2nc, x2nc, gps2n = extract_aligned_data(det2, data2n, ra, dec, tcn, points_before, points_after)
    
    if polarization:
        mat_i_12 = antenna_pattern_matrix(det1, det2, ra, dec, tc, pol, mat_inv=True)
        hp_12, hc_12 = polarization_extractor(mat_i_12, data1c, data2c)

        matn_i_12 = antenna_pattern_matrix(det1, det2, ra, dec, tcn, poln, mat_inv=True)
        hpn_12, hcn_12 = polarization_extractor(matn_i_12, data1nc, data2nc)

        tpp, hpXhpn, spp = cross_correlator(hp_12, hpn_12, x1c, steps, gps1, midpoint=False)
        tcc, hcXhcn, scc = cross_correlator(hc_12, hcn_12, x1c, steps, gps1, midpoint=False)
        tpc, hpXhcn, spc = cross_correlator(hp_12, hcn_12, x1c, steps, gps1, midpoint=False)
        tcp, hcXhpn, scp = cross_correlator(hc_12, hpn_12, x1c, steps, gps1, midpoint=False)

        return tpp, hpXhpn, spp, tcc, hcXhcn, scc, tpc, hpXhcn, spc, tcp, hcXhpn, scp
    
    else:
        t1, dXd1, s1 = cross_correlator(data1c, data1nc, x1c, steps, gps1, midpoint=False)
        t2, dXd2, s2 = cross_correlator(data2c, data2nc, x1c, steps, gps1, midpoint=False)
        return t1, dXd1, s1, t2, dXd2, s2


def compute_all_cross_correlations(det1, det2, det3, data1, data2, data3, data1n, data2n, data3n, ra, dec, tc, tcn, pol, poln, steps=64, polarization=True):
    results = {}
    
    def format_result(output, polarization):
        if polarization:
            return {
                'tpp': output[0],
                'hpXhpn': output[1],
                'spp': output[2],
                'tcc': output[3],
                'hcXhcn': output[4],
                'scc': output[5],
                'tpc': output[6],
                'hpXhcn': output[7],
                'spc': output[8],
                'tcp': output[9],
                'hcXhpn': output[10],
                'scp': output[11],
            }
        else:
            return {
                't1': output[0],
                'dXd1': output[1],
                's1': output[2],
                't2': output[3],
                'dXd2': output[4],
                's2': output[5],
            }
    
    results['H1_L1'] = format_result(
        cross_correlation(det1, det2, data1, data2, data1n, data2n, ra, dec, tc, tcn, pol, poln, steps, polarization),
        polarization
    )

    results['H1_V1'] = format_result(
        cross_correlation(det1, det3, data1, data3, data1n, data3n, ra, dec, tc, tcn, pol, poln, steps, polarization),
        polarization
    )

    results['L1_V1'] = format_result(
        cross_correlation(det2, det3, data2, data3, data2n, data3n, ra, dec, tc, tcn, pol, poln, steps, polarization),
        polarization
    )

    return results


