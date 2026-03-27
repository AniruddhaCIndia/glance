from scipy.integrate import simpson
import numpy as np

def cross_correlator(
    array1, 
    array2, 
    time, 
    steps, 
    specific_time=None, 
    midpoint=True,
    pearson=False
):
    """
    Compute chunk-wise normalized cross-correlation between two time series.

    This function divides two input arrays into equal-sized segments ("chunks")
    and computes a normalized cross-correlation value for each chunk. Optionally,
    the correlation can be computed in a Pearson-like manner by mean-centering
    each chunk.

    Parameters
    ----------
    array1 : np.ndarray
        First input array (1D), representing a time series signal.
    array2 : np.ndarray
        Second input array (1D), must be the same length as `array1`.
    time : np.ndarray
        1D array of time values corresponding to the input arrays.
    steps : int
        Number of chunks to divide the data into.
    specific_time : float, optional
        If provided, returns the correlation value corresponding to the chunk
        that contains the time closest to this value.
    midpoint : bool, default=True
        Determines how the representative time for each chunk is computed:
        - True: uses the midpoint of each chunk.
        - False: uses the last time value of each chunk.
    pearson : bool, default=False
        If True, mean-centers each chunk before computing correlation,
        resulting in a Pearson-like correlation coefficient.

    Returns
    -------
    time_of_cross_corr : np.ndarray
        Array of representative time values for each chunk.
    data_cross_corr : np.ndarray
        Array of normalized cross-correlation values for each chunk.
    specific_value : float, optional
        Returned only if `specific_time` is provided. It is the correlation
        value of the chunk closest to the specified time.

    Notes
    -----
    - The input arrays are truncated so their length is divisible by `steps`.
    - Correlation is computed as:

        corr = mean(a1 * a2) / sqrt(mean(a1^2) * mean(a2^2))

      which is equivalent to cosine similarity unless `pearson=True`, in which
      case it becomes a Pearson-like correlation.
    - If a chunk has zero variance in either signal, its correlation is set to 0.

    Examples
    --------
    >>> t = np.linspace(0, 10, 1000)
    >>> x = np.sin(t)
    >>> y = np.cos(t)
    >>> tc, cc = cross_correlator(x, y, t, steps=10)

    >>> tc, cc, val = cross_correlator(x, y, t, steps=10, specific_time=5.0)
    """

    N = len(array1)
    s = int(steps)
    p = N // s

    # Trim to fit exact chunks
    N_trim = p * s
    a1 = array1[:N_trim].reshape(s, p)
    a2 = array2[:N_trim].reshape(s, p)
    t = time[:N_trim].reshape(s, p)

    if pearson:
        # Mean-center along each chunk
        a1 = a1 - np.mean(a1, axis=1, keepdims=True)
        a2 = a2 - np.mean(a2, axis=1, keepdims=True)

    # Compute correlation components
    num = np.mean(a1 * a2, axis=1)
    norm_a = np.mean(a1**2, axis=1)
    norm_b = np.mean(a2**2, axis=1)

    denom = np.sqrt(norm_a * norm_b)

    # Safe division
    data_cross_corr = np.where(
        (norm_a > 0) & (norm_b > 0),
        num / denom,
        0.0
    )

    # Time handling
    if midpoint:
        mid_idx = p // 2
        time_of_cross_corr = 0.5 * (t[:, mid_idx - 1] + t[:, mid_idx])
    else:
        time_of_cross_corr = t[:, -1]

    if specific_time is not None:
        idx = np.abs(time - specific_time).argmin()
        specific_chunk = idx // p
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
    idx1 = np.searchsorted(cc_time, t_start, side='right')
    idx2 = np.searchsorted(cc_time, t_end, side='right')

    if idx1 == idx2:
        return cc_data[idx1]
    else:
        cc_data_chop = cc_data[idx1 : idx2]
        group_size = int(abs(idx2 - idx1)/int(steps))
        trimmed_data = cc_data_chop[:len(cc_data_chop) - len(cc_data_chop) % group_size]
        grouped_sums = trimmed_data.reshape(-1, group_size).sum(axis=1)

        return grouped_sums


def grouped_std_deviation(data: np.ndarray, group_size: int) -> float:
    """
    Computes the standard deviation of sums of non-overlapping groups of elements in an array.

    Parameters:
    - data (np.ndarray): Input array of numbers
    - group_size (int): Number of elements per group

    Returns:
    - float: Standard deviation of the sums of groups
    """
    # Ensure data length is divisible by group_size
    trimmed_data = data[:len(data) - len(data) % group_size]
    
    # Reshape and sum over groups
    grouped_sums = trimmed_data.reshape(-1, group_size).sum(axis=1)
    
    # Return the standard deviation
    return np.mean(grouped_sums), np.std(grouped_sums)


def SNR_calculator(cross, cross_noise, time, steps, specific_time=None, midpoint=True):
    """
    Compute a chunk-wise, chunk-size-independent SNR of a cross-correlation signal.

    This function divides the cross-correlation signal into chunks, computes
    the mean correlation in each chunk, and compares it to the standard deviation
    of the raw noise signal (not chunked) to yield an SNR independent of chunk size.

    Parameters
    ----------
    cross : np.ndarray
        Cross-correlation signal (1D array).
    cross_noise : np.ndarray
        Noise cross-correlation (1D array), assumed zero-mean or baseline.
    time : np.ndarray
        Time points corresponding to `cross`.
    steps : int
        Number of chunks to divide the data into.
    specific_time : float, optional
        If provided, returns the SNR for the chunk containing the time closest
        to this value.
    midpoint : bool, default=True
        Determines how the representative time for each chunk is computed:
        - True: uses the midpoint of each chunk.
        - False: uses the last time point of each chunk.

    Returns
    -------
    snr_time : np.ndarray
        Array of representative times for each chunk.
    snr : np.ndarray
        Chunk-wise SNR values (mean signal / noise std), independent of chunk size.
    specific_snr : float, optional
        SNR value for the chunk containing `specific_time`. Returned only if
        `specific_time` is provided.

    Notes
    -----
    - This SNR is mathematically equivalent to a Z-score of the signal with
      respect to the noise, using the raw noise std to remain chunk-size independent.
    - The numerator uses the **mean cross-correlation in each chunk**, which smooths
      fluctuations without artificially inflating SNR.
    """
    
    N = len(cross)
    steps = int(steps)
    chunk_size = N // steps  # floor division to fit exact chunks
    N_trim = chunk_size * steps
    
    # Reshape for vectorized computation
    cross_chunks = cross[:N_trim].reshape(steps, chunk_size)
    time_chunks = time[:N_trim].reshape(steps, chunk_size)
    
    # Compute mean signal per chunk
    sig = cross_chunks.mean(axis=1)
    
    # Compute representative time per chunk
    if midpoint:
        mid_idx = chunk_size // 2
        snr_time = 0.5 * (time_chunks[:, mid_idx - 1] + time_chunks[:, mid_idx])
    else:
        snr_time = time_chunks[:, -1]
    
    # Noise std over all raw samples (chunk-size independent)
    noise_std = np.std(cross_noise, ddof=1)
    
    # Compute chunk-wise SNR
    snr = sig / noise_std
    
    if specific_time is not None:
        idx = (np.abs(time - specific_time)).argmin()
        specific_chunk = idx // chunk_size
        return snr_time, snr, snr[specific_chunk]
    
    return snr_time, snr