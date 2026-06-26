# %%
import numpy as np
import matplotlib.pyplot as plt
import bilby
import h5py
from bilby.core.prior import Uniform, PowerLaw
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters, generate_all_bbh_parameters
bilby.core.utils.log.setup_logger(log_level='WARNING')
from gwpy.timeseries import TimeSeries
import os
from pesummary.io import read

# %%
def lal_bbh_wo_lensing(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, a, b, k, phi_0, 
        b_prime, k_prime, phi_0_prime, f_0, **kwargs):
    
    gr_waveform = bilby.gw.source.lal_binary_black_hole(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2, luminosity_distance=luminosity_distance,
        a_1=a_1, tilt_1=tilt_1, phi_12=phi_12, a_2=a_2, tilt_2=tilt_2, phi_jl=phi_jl,
        theta_jn=theta_jn, phase=phase, **kwargs)

    correction = np.zeros_like(frequency_array, dtype=complex)
    correction = correction + a * (1 + b * (np.cos(2* np.pi* frequency_array/ f_0 + phi_0) * np.exp(-frequency_array * k) )) * np.exp(1j * b_prime * np.cos(2* np.pi* frequency_array/ f_0 + phi_0_prime) * np.exp(-frequency_array * k_prime))

    if gr_waveform is None:
        return None
    else:
        return dict(plus= gr_waveform['plus'] * correction, cross= gr_waveform['cross'] * correction)

# %%
sampling_frequency = 1024
reference_frequency = 10
waveform_min_frequency = 0
detector_min_frequency = 20
detector_max_frequency = 448

#%%

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--approximant', type=str, required=True)
args = parser.parse_args()

approximant = args.approximant
print(f"Using approximant: {approximant}")

# %%
strain_paths = ['/home/achakraborty/project_S231123/zenodo_data_files/H-H1_GWOSC_DiscO4a_4KHZ_R1-1384779776-4096.hdf5',
                '/home/achakraborty/project_S231123/zenodo_data_files/L-L1_GWOSC_DiscO4a_4KHZ_R1-1384779776-4096.hdf5'
]

strain_filepaths = strain_paths
data_dict = {}
for file in strain_filepaths:

    file_name = os.path.basename(file)
    key = file_name.split("-")[0]

    parts = file_name.split("-")
    start_time = float(parts[2]) 
    duration = int(parts[3].split(".")[0]) 
    sampling_frequency = 4096  

    with h5py.File(file, 'r') as f:
        strain_dataset = f['strain/Strain']
        file_content = strain_dataset[:]

    strain_data = TimeSeries(file_content,
                dt=1/sampling_frequency,
                t0=start_time,
                dtype=float)

    data_dict[key] = {
        "strain": strain_data,
        "start_time": start_time,
        "duration": duration,
        "sampling_frequency": sampling_frequency,
    }

print(data_dict.keys())
print(sampling_frequency)

posterior_path = '/home/achakraborty/project_S231123/zenodo_data_files/posterior_samples.h5'
result = read(posterior_path, package="gw")
result_all = result.samples_dict

waveform_model = approximant
if approximant == "IMRPhenomXPHM":
    waveform_model = "IMRPhenomXPHM-SpinTaylor"

result_1 = result_all[f'C00:{waveform_model}']   #### Change here
tc = result_1['geocent_time'].maxL

# %%
asd_data_h = np.loadtxt('/home/achakraborty/project_S231123/data_files/IMRPhenomXPHM_H1_psd.dat.txt', skiprows=1)
asd_data_l = np.loadtxt('/home/achakraborty/project_S231123/data_files/IMRPhenomXPHM_L1_psd.dat.txt', skiprows=1)

freq_h = asd_data_h[:,0]
freq_l = asd_data_l[:,0]
H1_psd = asd_data_h[:,1]
L1_psd = asd_data_l[:,1]

plt.loglog(freq_h, H1_psd)

# %%
H1 = bilby.gw.detector.get_empty_interferometer("H1")
L1 = bilby.gw.detector.get_empty_interferometer("L1")

res_h = data_dict['H']['strain'].resample(sampling_frequency)
res_l = data_dict['H']['strain'].resample(sampling_frequency)

H1.set_strain_data_from_gwpy_timeseries(data_dict['H']['strain'].crop(tc-4, tc+4))
L1.set_strain_data_from_gwpy_timeseries(data_dict['L']['strain'].crop(tc-4, tc+4))

# %%
H1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=freq_h, psd_array=H1_psd)
L1.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=freq_l, psd_array=L1_psd)

H1.minimum_frequency = detector_min_frequency
L1.minimum_frequency = detector_min_frequency

H1.maximum_frequency = detector_max_frequency
L1.maximum_frequency = detector_max_frequency


# %%
samples1 = result_1

chirp_mass = samples1['chirp_mass'].maxL
mass_ratio = samples1['mass_ratio'].maxL
a_1 = samples1['a_1'].maxL
a_2 = samples1['a_2'].maxL
tilt_1 = samples1['tilt_1'].maxL
tilt_2 = samples1['tilt_2'].maxL
phi_jl = samples1['phi_jl'].maxL
phi_12 = samples1['phi_12'].maxL
luminosity_distance = samples1['luminosity_distance'].maxL
theta_jn = samples1['theta_jn'].maxL
psi = samples1['psi'].maxL
phase = samples1['phase'].maxL
geocent_time = samples1['geocent_time'].maxL
ra = samples1['ra'].maxL
dec = samples1['dec'].maxL


# %%
prior = bilby.core.prior.PriorDict()

### Source parameters 

prior['chirp_mass'] = chirp_mass #Uniform(name='chirp_mass', minimum=10.0, maximum=100.0)
prior['mass_ratio'] = mass_ratio #Uniform(name='mass_ratio', minimum=0.2, maximum=1)
prior['phase'] = phase #Uniform(name="phase", minimum=0, maximum=2*np.pi)
prior['geocent_time'] = geocent_time #Uniform(name="geocent_time", minimum=time_of_event-0.25, maximum=time_of_event+0.25)
prior['a_1'] =  a_1 #Uniform(name='a_1', minimum=0, maximum=0.99)
prior['a_2'] =  a_2 #Uniform(name='a_2', minimum=0, maximum=0.99)
prior['tilt_1'] = tilt_1 #Sine(name='tilt_1')
prior['tilt_2'] = tilt_2 #Sine(name='tilt_2')
prior['phi_12'] = phi_12 #0.0
prior['phi_jl'] = phi_jl #0.0
prior['dec'] = dec #0.5
prior['ra'] = ra #0.5
prior['theta_jn'] = theta_jn #0.0
prior['psi'] = psi #0.5
prior['luminosity_distance'] = luminosity_distance #PowerLaw(alpha=2, name='luminosity_distance', minimum=100, maximum=50000, unit='Mpc', latex_label='$d_L$')

### Lensing parameters

prior['b'] = Uniform(name='b', minimum=0.0, maximum=0.99)
prior['f_0'] = Uniform(name='f_0', minimum=1, maximum=detector_max_frequency)
prior['phi_0'] = Uniform(name='phi_0', minimum=0.0, maximum=2*np.pi)
prior['b_prime'] = Uniform(name='b_prime', minimum=0.0, maximum=0.99)
prior['phi_0_prime'] = Uniform(name='phi_0_prime', minimum=0.0, maximum=2*np.pi)
prior['a'] = 1
prior['k'] = 0
prior['k_prime'] = 0

# %%
interferometers = [H1, L1]

if approximant == "SEOBNRv5PHM":
    fd_model = bilby.gw.source.gwsignal_binary_black_hole
else:
    fd_model = bilby.gw.source.lal_binary_black_hole

ST_flags = {'PhenomXPFinalSpinMod' : 2,
         'PhenomXPrecVersion' : 320,
         'PhenomXHMReleaseVersion' : 122022}

waveform_arguments = dict(
    waveform_approximant = approximant, 
    reference_frequency = reference_frequency,
    minimum_frequency = waveform_min_frequency,
    catch_waveform_errors = True)

if approximant == 'IMRPhenomXPHM':
    waveform_arguments.update(ST_flags) 

waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model = lal_bbh_wo_lensing,
    waveform_arguments = waveform_arguments,
    sampling_frequency = sampling_frequency,
    duration = 8, 
    parameter_conversion = convert_to_lal_binary_black_hole_parameters)

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers, waveform_generator, priors=prior,)

result_short = bilby.run_sampler(
        likelihood, prior, sampler = 'dynesty', outdir = 'PE_lensing_final_run', label = f"{approximant}",
        conversion_function = bilby.gw.conversion.generate_all_bbh_parameters,
        nlive = 1000, dlogz = 0.01, 
        npool = 16,
    )
