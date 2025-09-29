from scipy.integrate import simpson
import numpy as np

def cross_correlator(array1, array2, time, steps, specific_time=None, midpoint=True):

    """
    Compute the segmented cross-correlation between two signals.

    This function divides the input time series into segments and calculates
    the cross-correlation for each segment using numerical integration
    (Simpson's rule). Optionally, it can return the cross-correlation for
    a segment corresponding to a specific time.

    Parameters
    ----------
    array1 : array-like
        First time series data array.
    array2 : array-like
        Second time series data array. Must be the same length as `array1`.
    time : array-like
        Array of time points corresponding to `array1` and `array2`.
    steps : int
        Number of segments to divide the time series into.
    specific_time : float, optional
        If provided, the function returns the cross-correlation for the segment
        that contains this specific time point.
    midpoint : bool, default=True
        Determines how segment times are calculated:
        - If True: time for each segment is taken as the midpoint of the segment.
        - If False: time for each segment is taken as the endpoint of the segment.

    Returns
    -------
    time_of_cross_corr : numpy.ndarray
        Array of segment times corresponding to each cross-correlation value.
    data_cross_corr : numpy.ndarray
        Array of cross-correlation values for each segment.
    specific_cross_corr : float, optional
        If `specific_time` is provided, the cross-correlation value of the segment
        containing that time point is returned as a third output.

    Notes
    -----
    - The length of `array1`, `array2`, and `time` must be the same.
    - `steps` should evenly divide the length of the arrays for meaningful segmentation.
    - The integration uses Simpson's rule and normalizes by segment length and sampling interval.

    Example
    -------
    >>> import numpy as np
    >>> time = np.linspace(0, 10, 1000)
    >>> array1 = np.sin(time)
    >>> array2 = np.cos(time)
    >>> time_of_cross_corr, data_cross_corr = cross_correlator(array1, array2, time, steps=10)
    >>> print(time_of_cross_corr)
    >>> print(data_cross_corr)

    """

    N = int(len(array1))
    s = int(steps)
    p = int(N / s)

    if specific_time is not None:
        idx = (np.abs(time - specific_time)).argmin()
        specific_chunk = idx // p

    data_cross_corr, time_of_cross_corr = [], []
    dt = time[1] - time[0]

    for i in range(s):
        start = i * p
        end = start + p

        result = (1 / (p * dt)) * simpson(
            y=array1[start:end] * array2[start:end],
            x=time[start:end]
        )
        data_cross_corr.append(result)

        if midpoint:
            t_mid = 0.5 * (time[start + p // 2 - 1] + time[start + p // 2])
        else:
            t_mid = time[end - 1]
        time_of_cross_corr.append(t_mid)

    data_cross_corr = np.array(data_cross_corr)
    time_of_cross_corr = np.array(time_of_cross_corr)

    if specific_time is not None:
        return time_of_cross_corr, data_cross_corr, data_cross_corr[specific_chunk]

    return time_of_cross_corr, data_cross_corr


def grouped_sums(cc_data, cc_time, t_start, t_end, steps):
    """
    Calculate sums of grouped data within a specified time range.

    This function takes time-series data (`cc_data`) with corresponding timestamps (`cc_time`),
    selects the portion of the data between `t_start` and `t_end`, divides it into equal-sized groups
    based on the given number of steps, and returns the sum of each group.

    Parameters:
    -----------
    cc_data : array-like
        Array of data values corresponding to timestamps in `cc_time`.

    cc_time : array-like
        Array of timestamps corresponding to `cc_data`. Must be equispaced for correct grouping.

    t_start : float
        Start time for the data segment to analyze. The function finds the nearest timestamp.

    t_end : float
        End time for the data segment to analyze. The function finds the nearest timestamp.

    steps : int
        Number of groups to divide the selected data segment into.

    Returns:
    --------
    grouped_sums : numpy.ndarray
        Array containing the sum of data in each group.

    Notes:
    ------
    - If the selected segment length is not an exact multiple of the group size,
      the extra data at the end is trimmed.
    - The function uses integer division to determine the group size.
    - Both `cc_time` and `cc_data` should be numpy arrays or array-like objects.
    - Slicing excludes `idx2`, i.e. `cc_data[idx1:idx2]`, meaning the exact end time may not be included.
    """
    idx1, idx2 = np.argmin(abs(cc_time-t_start)), np.argmin(abs(cc_time - t_end))
    cc_data_chop = cc_data[idx1 : idx2]
    group_size = int(abs(idx2 - idx1)/int(steps))
    trimmed_data = cc_data_chop[:len(cc_data_chop) - len(cc_data_chop) % group_size]
    grouped_sums = trimmed_data.reshape(-1, group_size).sum(axis=1)

    return grouped_sums


def cross_correlator_normalized(array1, array2, time, steps, specific_time=None, midpoint=True):
    N = int(len(array1))
    s = int(steps)
    p = int(N / s)

    if specific_time is not None:
        idx = (np.abs(time - specific_time)).argmin()
        specific_chunk = idx // p

    data_cross_corr, time_of_cross_corr = [], []

    for i in range(s):
        start = i * p
        end = start + p

        num = simpson(y=array1[start:end] * array2[start:end], x=time[start:end])
        norm_a = simpson(y=array1[start:end]**2, x=time[start:end])
        norm_b = simpson(y=array2[start:end]**2, x=time[start:end])

        if norm_a > 0 and norm_b > 0:
            result = num / np.sqrt(norm_a * norm_b)
        else:
            result = 0.0

        data_cross_corr.append(result)

        if midpoint:
            t_mid = 0.5 * (time[start + p // 2 - 1] + time[start + p // 2])
        else:
            t_mid = time[end - 1]
        time_of_cross_corr.append(t_mid)

    data_cross_corr = np.array(data_cross_corr)
    time_of_cross_corr = np.array(time_of_cross_corr)

    if specific_time is not None:
        return time_of_cross_corr, data_cross_corr, data_cross_corr[specific_chunk]

    return time_of_cross_corr, data_cross_corr


def SNR_calculator(cross, cross_noise, time, steps, specific_time=None, midpoint=True):
    s1 = int(steps)
    N1 = len(cross)
    p1 = int(N1 / s1)

    p2 = int(p1)
    N2 = len(cross_noise)
    s2 = int(N2 / p2)
    

    sig, noise, snr_time = [], [], []

    if specific_time is not None:
        idx = (np.abs(time - specific_time)).argmin()
        specific_chunk = idx // p1

    for i in range(s1):
        start1 = i * p1
        end1 = start1 + p1
        a = np.sum(cross[start1:end1])
        sig.append(a)

        if midpoint:
            t_mid = 0.5 * (time[start1 + p1 // 2 - 1] + time[start1 + p1 // 2])
        else:
            t_mid = time[start1 + p1 - 1]
        snr_time.append(t_mid)

    for i in range(s2):
        start2 = i * p2
        end2 = start2 + p2
        n = np.sum(cross_noise[start2:end2])
        noise.append(n)

    sig = np.array(sig)
    noise = np.array(noise)
    snr = abs(sig / np.std(noise, ddof=1))
    snr_time = np.array(snr_time)

    if specific_time is not None:
        return snr_time, snr, snr[specific_chunk]

    return snr_time, snr