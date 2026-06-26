# %%
import numpy as np
import matplotlib.pyplot as plt
import bilby
#import pyseobnr
from bilby.core.prior import Uniform, PowerLaw, Sine, Cosine
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters, generate_all_bbh_parameters
bilby.core.utils.log.setup_logger(log_level='WARNING')
import os
os.environ['LAL_DATA_PATH'] = "/home/achakraborty/surrogate_data/"

# %%
sampling_frequency = 4096
reference_frequency = 10
waveform_min_frequency = 10
waveform_max_frequency = 448
detector_min_frequency = 20
detector_max_frequency = 512
duration = 4
approximant = 'NRSur7dq4' # Excluded from the list of approximants for PE as of now, as it is very slow. Will be added later.
print(f"Using approximant: {approximant}")

# %%
asd_data_h = np.loadtxt('/home/achakraborty/project_S231123/data_files/IMRPhenomXPHM_H1_psd.dat.txt', skiprows=1)
asd_data_l = np.loadtxt('/home/achakraborty/project_S231123/data_files/IMRPhenomXPHM_L1_psd.dat.txt', skiprows=1)

freq_h = asd_data_h[:,0]
freq_l = asd_data_l[:,0]
H1_psd = asd_data_h[:,1]
L1_psd = asd_data_l[:,1]

# %%
injection_parameters = dict(
    chirp_mass = 100,
    mass_ratio = 0.75,
    a_1=0.5,
    a_2=0.5,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,
    luminosity_distance=3000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1.4e9,
    ra=1.375,
    dec=-1.2108,
)

waveform_arguments = dict(
    waveform_approximant= approximant,
    reference_frequency= reference_frequency,
    minimum_frequency= waveform_min_frequency,
    maximum_frequency= waveform_max_frequency,
    #PhenomXHMReleaseVersion = 122022, 
    #PhenomXPFinalSpinMod = 2, 
    #PhenomXPrecVersion = 320
)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# %%
bilby.core.utils.random.seed(88170231)
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])

ifos[0].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=freq_h, psd_array=H1_psd)

ifos[1].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
    frequency_array=freq_l, psd_array=L1_psd)

ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)

ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

ifos[0].minimum_frequency = detector_min_frequency
ifos[0].maximum_frequency = detector_max_frequency

ifos[1].minimum_frequency = detector_min_frequency
ifos[1].maximum_frequency = detector_max_frequency

net_snr = np.sqrt(abs(ifos.meta_data['H1']['matched_filter_SNR'])**2 + abs(ifos.meta_data['L1']['matched_filter_SNR'])**2)

# %%

app0 = "NRSur7dq4"
app1 = "IMRPhenomXPHM"
app2 = "IMRPhenomTPHM"
app3 = "IMRPhenomXO4a"
app4 = "SEOBNRv5PHM"

apps = [app1, app2, app3] # Excluded app4 as of now

# import multiprocessing as mp
# mp.set_start_method("spawn", force=True)

for app in apps:

    waveform_arguments2 = dict(
        waveform_approximant = app,
        reference_frequency = reference_frequency,
        minimum_frequency = waveform_min_frequency,
        maximum_frequency = waveform_max_frequency,
        catch_waveform_errors = True,
    )

    if app == "SEOBNRv5PHM":
        fd_model = bilby.gw.source.gwsignal_binary_black_hole
    elif app == "IMRPhenomXPHM":
        fd_model = bilby.gw.source.lal_binary_black_hole
        waveform_arguments2.update({'PhenomXHMReleaseVersion': 122022, 
                                   'PhenomXPFinalSpinMod': 2, 
                                   'PhenomXPrecVersion': 320}
        )
    else:
        fd_model = bilby.gw.source.lal_binary_black_hole

    waveform_generator2 = bilby.gw.WaveformGenerator(
        duration= duration,
        sampling_frequency= sampling_frequency,
        frequency_domain_source_model= fd_model,
        parameter_conversion= bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments= waveform_arguments2,
    )

    priors = bilby.core.prior.PriorDict()

    priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=50, maximum=250)
    priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1)
    priors['phase'] = Uniform(name='phase', minimum= 0, maximum= 2*np.pi, boundary='periodic')
    priors['a_1'] =  Uniform(name='a_1', minimum=0, maximum=0.99)
    priors['a_2'] =  Uniform(name='a_2', minimum=0, maximum=0.99)
    priors['tilt_1'] = Sine(name='tilt_1')
    priors['tilt_2'] = Sine(name='tilt_2')
    priors['theta_jn'] = Sine(name='theta_jn')
    priors['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=1e2, maximum=1e4)
    priors['phi_12'] = injection_parameters['phi_12']
    priors['phi_jl'] = injection_parameters['phi_jl']
    priors['dec'] = injection_parameters['dec']
    priors['ra'] = injection_parameters['ra']
    priors['psi'] = injection_parameters['psi']
    priors['geocent_time'] = injection_parameters['geocent_time']

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator = waveform_generator2, priors=priors,
    )

    outdir = f"pe_for_far_spins_{injection_parameters['chirp_mass']}_{injection_parameters['mass_ratio']}_{round(net_snr, 2)}_{approximant}"
    label = f"{waveform_arguments2['waveform_approximant']}"
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        npoints=500,
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label,
        npool=8,
    )
