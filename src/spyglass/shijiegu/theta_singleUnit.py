import os
import elephant
import quantities as pq
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import filtfilt, lfilter

from elephant.conversion import BinnedSpikeTrain
from spyglass.spikesorting.v0.spikesorting_curation import WaveformParameters, WaveformSelection, Waveforms

from spyglass.spikesorting.v0.spikesorting_curation import QualityMetrics
from spyglass.shijiegu.singleUnit import electrode_unit
from spyglass.shijiegu.Analysis_SGU import SingleUnit,TrialChoice,EpochPos
import matplotlib.pyplot as plt
import neo
from elephant.conversion import BinnedSpikeTrain
from scipy.signal import find_peaks
from spyglass.shijiegu.load import load_run_sessions, load_theta
from spyglass.shijiegu.Analysis_SGU import SingleUnit, MUATheta, TetrodeNumber, TrialChoice, ThetaZscore
from spyglass.shijiegu.changeOfMind_figures.figure4d import load_theta_df
from spyglass.shijiegu.decodeHelpers import session2position_name, runSessionNames
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

def return_list_of_spike_train(nwb_copy_file_name, session_name, pyramidal_only = True):
    ## Return list of spike trains of pyramidal cells only
    
    # find epoch/session name and position interval name
    key = (EpochPos & {'nwb_file_name':nwb_copy_file_name,'epoch':int(session_name[:2])}).fetch1()
    epoch_name = key['epoch_name']
    position_interval = key['position_interval']
    is_run_session = False
    if epoch_name.split('_')[1][4:8] == 'Sess':
        is_run_session = True

    if is_run_session:
        # Remove Data before 1st trial and after last trial
        StateScript = pd.DataFrame(
            (TrialChoice & {'nwb_file_name':nwb_copy_file_name,
                            'epoch_name':session_name}).fetch1('choice_reward')
        )

        trial_1_t = StateScript.loc[1].timestamp_O
        trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O
    
    key = {"nwb_file_name": nwb_copy_file_name,
       "sorter":"mountainsort4",
       "epoch":int(session_name[:2]),
       "curation_id":1}
    
    cell_type_pd = pd.DataFrame((SingleUnit() & key).fetch1("cell_type_pd"))
    sort_groups = np.unique([ind[0] for ind in cell_type_pd.index])

    nwb_units_all = {}
    for sort_group_id in sort_groups:
        nwb_units = electrode_unit(nwb_copy_file_name,session_name,sort_group_id,curation_id = 1)
        nwb_units_all[sort_group_id] = nwb_units
        
    spike_trains = []
    cell_types = {}
    for e in nwb_units_all.keys():
        for u in nwb_units_all[e].index:
            nwb_unit = nwb_units_all[e].loc[u]
            cell_types[(e,u)] = cell_type_pd.loc[(e,u)].classification
            if pyramidal_only and cell_type_pd.loc[(e,u)].classification != 'pyramidal':
                continue
            spike_train = return_spike_train(nwb_unit,
                                             tmin = trial_1_t if is_run_session else None,
                                             tmax = trial_last_t if is_run_session else None)
            spike_trains.append(spike_train)
    return spike_trains, nwb_units_all, cell_types



def mua_pyramidal_only(nwb_copy_file_name, session_name, bin_size = 10 * pq.ms):
    spike_trains, _, _2 = return_list_of_spike_train(nwb_copy_file_name, session_name)

    binned_spiketrains = BinnedSpikeTrain(
        spike_trains,
        bin_size=bin_size)

    # spike count
    binned_spiketrains_np = binned_spiketrains.to_array() # number of cells x number of timestamps
    mua_np = np.sum(binned_spiketrains_np, axis=0) # sum across cells

    # Get the time bins
    binned_spiketrains_t = binned_spiketrains.bin_centers
    
    # Create xarray Dataset
    mua_df = pd.DataFrame(data=mua_np, index=binned_spiketrains_t, columns = ['mua'])
    mua_df.index.name='time'
    mua_xr=xr.Dataset.from_dataframe(mua_df)
    
    return mua_xr

def make_phase_from_amplitude(amplitude_xr, datafield = 'mua'):
    min_peak_distance = 0.1 # seconds, minimum distance between peaks
    min_peak_distance_samples = int(min_peak_distance / (amplitude_xr.time[1] - amplitude_xr.time[0]).item())
    
    if datafield == 'mua':
        data = amplitude_xr.mua
    elif datafield == 'rightside':
        data = amplitude_xr.rightside
    else:
        data = amplitude_xr.leftside

    peak_ind,_ = find_peaks(np.array(data), distance = min_peak_distance_samples)
    print("len(peak_ind)", len(peak_ind))
    
    phase_values = np.arange(len(peak_ind)) * 2 * np.pi #unwrap phase to be increasing
    time_points = np.array(amplitude_xr.time)[peak_ind]
    new_time_points = np.array(amplitude_xr.time)
    interpolated_phase = np.interp(new_time_points, time_points, phase_values, left=np.nan, right=np.nan)

    y = np.cos(interpolated_phase)
    
    theta_xr = xr.Dataset(
        data_vars={
            "0": (("time"), y), #legacy format for theta
            "amplitude": (("time"), np.array(data)),
            "phase": (("time"), interpolated_phase % (2*np.pi)),
        },
        coords={
            "time": np.array(amplitude_xr.time),
        },
        attrs={"description": "theta from mua or from corpus collosum"}
    )
    return theta_xr


def get_theta_from_mua(mua, window_size = 0.04): # 0.04
    ## find theta peak:
    ## To establish a common reference phase,
    #  a phase histogram (π/6 or 30° bin size) of aggregate single (principal) cell firing
    #  in CA1 was calculated across locomotor periods for each recording day;
    #  the phase of maximal CA1 firing was then defined to be 0° (Skaggs et al., 1996), with the half-cycle offset (±π) corresponding to the phase segregating individual cycles.
       
    if  window_size > 0:
        mua = smoothen_mua(mua, moving_ave_window = window_size) #40ms window
    else:
        mua = mua
    
    theta_mua_xr = make_phase_from_amplitude(mua, datafield = 'mua')
    return theta_mua_xr
    
    
    
def return_spike_train(nwb_unit, tmin, tmax):
    spike_times = nwb_unit.spike_times * pq.s
    sort_interval = nwb_unit.sort_interval.ravel()
    t_start, t_stop = sort_interval
    
    # Create a SpikeTrain object
    if tmin is not None:
        t_start = max(t_start, tmin)
    if tmax is not None:
        t_stop = min(t_stop, tmax)
    
    spike_times = spike_times[(spike_times >= t_start) & (spike_times <= t_stop)]
    spike_train_obj = neo.SpikeTrain(spike_times, t_start = t_start, t_stop = t_stop)
    
    return spike_train_obj


def smoothen_mua(mua, moving_ave_window = 0.04):
    # moving_ave_window: moving average window length in seconds, 40ms window
    
    """mua is an xarray with data mua and coordinate of time"""
    sampling_freq = 1 / (mua.time[1] - mua.time[0]).item()  # in Hz
    
    window_size = int(moving_ave_window * sampling_freq)
    if window_size % 2 == 0:
        window_size += 1  # make it odd

    smoothend_mua = moving_average_filtfilt(np.array(mua.mua), window_size)
    mua_smoothened_xr = xr.Dataset(
        data_vars={
            "mua": (("time"), smoothend_mua),
        },
        coords={
            "time": np.array(mua.time),
        },
        attrs={"description": "smoothened mua"}
    )
    return mua_smoothened_xr


def moving_average_filtfilt(data, window_size):
    """
    Applies a moving average filter using scipy.signal.filtfilt for zero-phase smoothing.

    Args:
        data (np.ndarray): The input signal to be smoothed.
        window_size (int): The size of the moving average window. 
                           Should be an odd integer for a centered window.

    Returns:
        np.ndarray: The smoothed signal.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer for a centered moving average.")

    # Create the numerator (b) and denominator (a) coefficients for the filter
    # For a simple moving average, b is an array of ones divided by window_size
    # and a is 1.
    b = np.ones(window_size) / window_size
    a = 1.0

    # Apply the filter using filtfilt for zero-phase filtering
    smoothed_data = filtfilt(b, a, data)
    return smoothed_data

def get_theta_from_cc(nwb_copy_file_name,session_name,datafield = 'rightside'):
    # load theta and also make theta phase
    theta_data,theta_timestamps = load_theta(nwb_copy_file_name,session_name)
    
    CCTetrodeInd = (TetrodeNumber() & {'nwb_file_name': nwb_copy_file_name,
                                            'epoch':session_name[:2]}).fetch1('cc_tetrode_ind')

    if len(CCTetrodeInd) == 1:
        colnames = ["rightside"]
    else:
        CCTetrodeInd = CCTetrodeInd[:2]
        colnames = ["rightside", "leftside"]
    theta_df = pd.DataFrame(data=theta_data[:,CCTetrodeInd], index=theta_timestamps, columns=colnames)
    theta_df.index.name='time'
    theta_df = theta_df.rename(columns={'0': 'rightside', '1': 'leftside'})
    theta_df=xr.Dataset.from_dataframe(theta_df)

    theta_with_phase = make_phase_from_amplitude(theta_df, datafield = datafield)

    return theta_with_phase

from scipy import interpolate
def zscore_theta(nwb_copy_file_name,session_name,data_type,use_spyglass = False):
    # load theta
    key = {"nwb_file_name": nwb_copy_file_name,
        "epoch": str(session_name[:2]),
        "data_type":data_type}
    theta_pd = load_theta_df(key, spyglass = use_spyglass)

    # load position info
    position_info = (IntervalPositionInfo() & {
                'nwb_file_name':nwb_copy_file_name,
                'interval_list_name':session2position_name(nwb_copy_file_name, session_name),
                'position_info_param_name':'default_decoding'}).fetch1_dataframe()

    # find all bins of theta with animal velocity >= 4cm/s
    f_nearest = interpolate.interp1d(np.array(position_info.index),
                                    position_info.head_speed,
                                    bounds_error=False,
                                    kind='nearest',
                                    fill_value=np.nan)
    speed_interp = f_nearest(np.array(theta_pd.time))

    print("before speed thresholding: length of",len(theta_pd), int(np.min(np.diff(theta_pd.time)) * len(theta_pd) / 60),"min of recording")
    theta_pd = theta_pd[speed_interp >= 4]
    print("after speed thresholding: length of",len(theta_pd), int(np.min(np.diff(theta_pd.time)) * len(theta_pd) / 60),"min of recording")
    
    # calculate mean amplitude and sd
    mean = np.mean(theta_pd.amplitude ** 2)
    sd = np.std(theta_pd.amplitude ** 2)
    
    return theta_pd, mean, sd


def zscore_theta_animal(animal, list_of_days, data_type, use_spyglass = False):

    for day in list_of_days:
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        for ind in range(len(session_interval)):
            session_name = session_interval[ind]
            
            theta_pd, mean, sd = zscore_theta(
                nwb_copy_file_name, session_name, data_type, use_spyglass = use_spyglass)
            
            key = {"nwb_file_name":nwb_copy_file_name,
                   "epoch":int(session_name[:2]),
                   "data_type":data_type,
                    "zscore": {"mean": mean, "sd": sd}}
            ThetaZscore().insert1(key, replace = True)
    
    return 1
    
