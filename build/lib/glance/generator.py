from scipy.integrate import simpson
import numpy as np
import h5py
from gwpy.timeseries import TimeSeries
import os
import re
import ast
from scipy.signal import stft
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import glob
from scipy.stats import gaussian_kde
import pycbc.noise
import pycbc.psd
import pycbc.waveform
import pycbc.filter
from pycbc.detector import Detector
from pycbc.psd import welch, interpolate
from pycbc.waveform.generator import FDomainDetFrameGenerator, FDomainCBCGenerator, TDomainCBCGenerator
from pycbc.waveform import get_waveform_filter_length_in_time as seglen
from pycbc.filter import matched_filter
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.filter import highpass, lowpass, highpass_fir, lowpass_fir
import bilby
from bilby.gw.detector import InterferometerList
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.source import lal_binary_black_hole
from gwpy.timeseries import TimeSeries


def data_generator_pycbc(params, seed, detectors="HLV", noise=True):

    """
    Generate gravitational wave strain data for a network of detectors with optional noise.

    Parameters
    ----------
    params : dict
        Dictionary of waveform and simulation parameters. Expected keys include
        'f_lower', 'f_final', 'delta_f', and 'delta_t', among others required by 
        `FDomainDetFrameGenerator`.
    
    seed : int or None
        Seed for random number generation to ensure reproducibility of noise.
        If None, a system-generated seed will be used.
    
    detectors : str, optional
        String specifying the detectors to use (default is "HLV" for Hanford, Livingston, Virgo).
        Currently supports only "HLV".

    noise : bool, optional
        Whether to add simulated Gaussian noise to the signal using detector-specific PSDs.
        Default is True.

    Returns
    -------
    data1 : pycbc.types.TimeSeries
        Time-domain strain data for the H1 detector (Hanford).
    
    data2 : pycbc.types.TimeSeries
        Time-domain strain data for the L1 detector (Livingston).
    
    data3 : pycbc.types.TimeSeries
        Time-domain strain data for the V1 detector (Virgo).

    Notes
    -----
    - The waveform is generated in the frequency domain using `FDomainDetFrameGenerator`.
    - Noise is generated using analytical PSD models and added to the signal if `noise=True`.
    - Seeds are randomly selected and controlled to ensure different noise realizations per detector.
    """

    generate_model = FDomainDetFrameGenerator(FDomainCBCGenerator, **params)
    htilde = generate_model.generate()
    
    ht1 = htilde['H1'].to_timeseries()
    ht2 = htilde['L1'].to_timeseries()
    ht3 = htilde['V1'].to_timeseries()
    
    f_low = params['f_lower']
    f_up = params['f_final']
    delta_f = params['delta_f']
    delta_t = params['delta_t']
    
    tsamples = int(1 / delta_t * 1 / delta_f)
    f_len = int(f_up / delta_f) + 1

    #### Choose PSDs according to need ####
    if noise:
        psd_ligo = pycbc.psd.analytical.aLIGOAdVO4T1800545(f_len, delta_f, f_low)
        psd_virgo = pycbc.psd.analytical.AdvVirgo(f_len, delta_f, f_low)
        system_rng = np.random.default_rng()
        base = system_rng.integers(1, int(1e9))

        if seed is not None:
            combined_seed = (base + seed) % int(1e9)
        else:
            combined_seed = base

        rng = np.random.default_rng(combined_seed)
        unique_seeds = rng.choice(np.arange(1, int(1e9)), size=3, replace=False)
        n1 = pycbc.noise.noise_from_psd(tsamples, delta_t, psd_ligo, seed=int(unique_seeds[0]))
        n2 = pycbc.noise.noise_from_psd(tsamples, delta_t, psd_ligo, seed=int(unique_seeds[1]))
        n3 = pycbc.noise.noise_from_psd(tsamples, delta_t, psd_virgo, seed=int(unique_seeds[2]))
        
        data1 = ht1 + np.array(n1)
        data2 = ht2 + np.array(n2)
        data3 = ht3 + np.array(n3)
        
    else:
        data1 = ht1
        data2 = ht2
        data3 = ht3

    return data1, data2, data3


def waveform_builder(samples_dict, args):

    """
    Generate time-domain waveforms for multiple detectors based on posterior samples.

    This function uses the maximum likelihood samples to produce time-domain
    gravitational waveforms for each specified detector.

    Parameters
    ----------
    samples_dict : object
        An object containing posterior sample data with a method
        `maxL_td_waveform()` that generates time-domain waveforms.
        This object should be able to accept arguments passed in `args`.
    args : dict
        Dictionary of waveform generation arguments, such as:
            - approximant: waveform approximant name
            - delta_t: sampling interval
            - f_low: minimum frequency
            - f_ref: reference frequency
            - flags (optional): special waveform flags

    Returns
    -------
    waveform : dict
        Dictionary containing waveforms for each detector with keys:
            - "H": TimeSeries for Hanford detector
            - "L": TimeSeries for Livingston detector
            - "V": TimeSeries for Virgo detector

    Example
    -------
    >>> args = {'approximant': 'IMRPhenomXPHM', 'delta_t': 1/4096, 'f_low': 20.0, 'f_ref': 50.0}
    >>> waveforms = waveform_builder(samples_dict, args)
    >>> print(waveforms.keys())
    dict_keys(['H', 'L', 'V'])
    """
    
    waveform_h = samples_dict.maxL_td_waveform(**args, project="H1")
    waveform_l = samples_dict.maxL_td_waveform(**args, project="L1")
    waveform_v = samples_dict.maxL_td_waveform(**args, project="V1")

    waveform = {}
    waveform['H'] = waveform_h
    waveform['L'] = waveform_l
    waveform['V'] = waveform_v

    return waveform


def ifo_builder(injection_parameters, waveform_generator, noise=True, seed = 10023):

    bilby.core.utils.random.seed(seed)

    interferometers = InterferometerList(['H1', 'L1', 'V1'])
    H1, L1, V1 = interferometers[0], interferometers[1], interferometers[2]

    H1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        asd_file='/home/achakraborty/O4_asds/H1_O4_strain.txt'
        )
    L1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        asd_file='/home/achakraborty/O4_asds/L1_O4_strain.txt'
        )
    V1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        asd_file='/home/achakraborty/O4_asds/V1_O4_strain.txt'
        )
    
    start_time = injection_parameters['geocent_time'] - 32
    
    if noise:
        interferometers.set_strain_data_from_power_spectral_densities(sampling_frequency=4096, 
                                                        duration=64, 
                                                        start_time=start_time)
    else:
        interferometers.set_strain_data_from_zero_noise(sampling_frequency=4096, 
                                                        duration=64, 
                                                        start_time=start_time)        

    H1.minimum_frequency = 20
    L1.minimum_frequency = 20
    V1.minimum_frequency = 20

    interferometers.inject_signal(
        waveform_generator= waveform_generator,
        parameters=injection_parameters,
    )

    return H1, L1, V1


def waveform_from_ifo(ifo_list):

    t1 = ifo_list[0].time_array
    t2 = ifo_list[1].time_array
    t3 = ifo_list[2].time_array

    h1 = ifo_list[0].time_domain_strain
    h2 = ifo_list[1].time_domain_strain
    h3 = ifo_list[2].time_domain_strain

    h1 = TimeSeries(data = h1, times = t1)
    h2 = TimeSeries(data = h2, times = t2)
    h3 = TimeSeries(data = h3, times = t3)

    return h1, h2, h3
