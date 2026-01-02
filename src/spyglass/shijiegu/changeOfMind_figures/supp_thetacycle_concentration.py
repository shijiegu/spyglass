import pandas as pd
# ignore datajoint+jupyter async warnings
import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=ResourceWarning)

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import logging
import os
import cupy as cp
from scipy import linalg

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')
from spyglass.decoding.v0.clusterless import ClusterlessClassifierParameters

from spyglass.shijiegu.Analysis_SGU import (EpochPos,TrialChoice,Decode,
    MUATheta,DecodeIngredients,DecodeResultsLinear,ChangeofMindRemoteTheta,MUA,DecodeIngredientsLikelihood,DecodeResultsLinear)
from spyglass.shijiegu.changeOfMind_figures.supp_decode import return_diff_file
from spyglass.linearization.v0.main import IntervalLinearizedPosition
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.shijiegu.decodeHelpers import session2position_name, runSessionNames
from ripple_detection.core import segment_boolean_series

from spyglass.shijiegu.changeOfMind_triggered_position import load_triggered_position_decode_day
from spyglass.shijiegu.ripple_add_replay import select_subset_helper_pd, select_subset_helper
from spyglass.shijiegu.changeOfMind_helper import nodes
from spyglass.shijiegu.ripple_add_replay import position_posterior2arm_posterior
from spyglass.shijiegu.changeOfMind_triggered import linear_map

def return_concentration_session(nwb_copy_file_name, session_name, data_type = "mua", posterior_thresholds = [0.15, 0.2, 0.3]):
    """
    data_type: "mua" or "corpus_callosum" or "sorted_pyramidal"
    """
    
    # find remote intervals
    pandas = (ChangeofMindRemoteTheta() & {"nwb_file_name":nwb_copy_file_name,
                                           "proportion":0.1,
                                            "delta_t_minus":5,
                                            "delta_t_plus":5,
                                        "epoch":str(session_name[:2])}).fetch1("pandas")
    log_df = pd.DataFrame(pandas)
    log_df =log_df[log_df.has_remote_interval]
    colname = 'remote_interval'
    
    # load theta
    key = {"nwb_file_name": nwb_copy_file_name,
        "epoch": str(session_name[:2]),
        "data_type":data_type}
    theta_pd = pd.read_csv((MUATheta() & key).fetch1("theta_xr"))
    
    animal = nwb_copy_file_name[:5]
    if animal == "eliot":
        encoding_set = '2Dheadspeed_above_4_andlowmua'
        classifier_param_name = 'default_decoding_gpu_4armMaze'
    else:
        encoding_set = '2Dheadspeed_above_4'
        classifier_param_name = 'default_decoding_gpu_4armMaze'
    
    # load decode
    decode_path = (DecodeResultsLinear & {"nwb_file_name":nwb_copy_file_name,
                                "interval_list_name":session_name,
                                "encoding_set":encoding_set,
                                "classifier_param_name":classifier_param_name}).fetch1("posterior")
    decode = xr.open_dataset(decode_path)
            
    ## load LinearPosition
    pos1d = pd.read_csv((DecodeIngredients & {"nwb_file_name":nwb_copy_file_name,
                                        "interval_list_name":session_name}).fetch1("position_1d"))
            
    pos2d = pd.read_csv((DecodeIngredients & {"nwb_file_name":nwb_copy_file_name,
                                        "interval_list_name":session_name}).fetch1("position_2d"))
    
    concentration_dict_all_trials = {threshold: [] for threshold in posterior_thresholds}
    theta_cycle_num_all = []
    for trialID in log_df.index:
        remote_intervals = log_df.loc[trialID,colname]
        for remote_interval in remote_intervals:
            # select theta in this interval
            remote_interval_extended = remote_interval.copy()
            remote_interval_extended[0] -= 0.02  # extend a bit
            remote_interval_extended[1] += 0.02 # extend a bit
            theta_subset = theta_pd[(theta_pd.time >= remote_interval_extended[0]) & (theta_pd.time <= remote_interval_extended[-1])]
            if len(theta_subset) == 0:
                continue
                
            concentration_dict, theta_cycle_num = return_concentration_event(
                remote_interval_extended, pos1d, pos2d, decode, theta_subset, thresholds_nonlocal = posterior_thresholds)
            theta_cycle_num_all.append(theta_cycle_num)
            
            
            for threshold in posterior_thresholds:
                if concentration_dict is not None:
                    concentration_dict_all_trials[threshold].extend(concentration_dict[threshold])  
            
    return concentration_dict_all_trials, theta_cycle_num_all


def return_concentration_day(animal, days):
    concentration_dict_all_trials = {0.15: [], 0.2: [], 0.3: []}
    theta_cycle_num_all_trials = []
    
    for d in days:
        nwb_copy_file_name = animal + d + '_.nwb'
        animal = nwb_copy_file_name[:5]
        session_names, _ = runSessionNames(nwb_copy_file_name)
        
        # load change of mind triggered position decode on this day!
        paramters = {"proportion":0.1,
                    "delta_t_minus":5,
                    "delta_t_plus":5,
                    "max_flag":1,
                    "segment_only":False,
                    "multiple_CoM":True, "single_CoM":True, "first_CoM":False
                    }
    
        for session_name in session_names:
            print(f"Processing {animal} {d} session: {session_name} per theta cycle")
            concentration_dict_all_trials_session, theta_cycle_num_all_session = return_concentration_session(
                nwb_copy_file_name, session_name, data_type = "mua", posterior_thresholds = [0.15, 0.2, 0.3])
            for threshold in concentration_dict_all_trials_session.keys():
                concentration_dict_all_trials[threshold].extend(concentration_dict_all_trials_session[threshold])
            theta_cycle_num_all_trials.extend(theta_cycle_num_all_session)
    
    return concentration_dict_all_trials, theta_cycle_num_all_trials

def return_concentration_event(t0t1, pos1d, pos2d, decode, theta, thresholds_nonlocal = [0.1, 0.2, 0.3]):
    
    subset_ind = (pos1d.time >= t0t1[0]) & (pos1d.time <= t0t1[1])
    pos1d_subset = pos1d.loc[subset_ind]
    pos2d_subset = pos2d.loc[subset_ind]

    # a) speed thresholding
    ind = pos2d_subset.head_speed >= 4
    pos1d_subset = pos1d_subset[ind]
    pos2d_subset = pos2d_subset[ind]

    # b) animals in outer arms only
    ind = pos1d_subset.track_segment_id >= 5
    pos1d_subset = pos1d_subset[ind]
    pos2d_subset = pos2d_subset[ind]
    
    if len(pos1d_subset) == 0:
        return None
    
    # get decode
    decode_subset = decode.isel(time = pos2d_subset.index)
    posterior_position_subset = decode_subset.causal_posterior.sum(dim='state')
    
    # map posterior over location to posterior over arm 
    posterior_by_arm = position_posterior2arm_posterior(posterior_position_subset,linear_map)
    arm_id = np.array(pos1d_subset.track_segment_id - 5).astype("int")
    
    # get nonlocal posterior by thresholding low local posterior
    
    cycle_times = theta_amplitude_to_cycle(theta)
    number_of_arms_greater_than_threshold_cycle = {threshold: [] for threshold in thresholds_nonlocal}
    for cycle_time in cycle_times:
        t0, t1 = cycle_time
        time_ind = np.argwhere((pos1d_subset.time >= t0) & (pos1d_subset.time <= t1)).ravel()
        if len(time_ind) == 0:
            continue
        
        posterior_cycle = np.array([posterior_by_arm[:, t_ind] for t_ind in time_ind])
        posterior_time = pos1d_subset.iloc[time_ind]
        number_of_arms_greater_than_threshold_cycle_ = threshold_arms(posterior_cycle,
                                                                      posterior_time,
                                                                      time_threshold = 0.02,
                                                                      posterior_thresholds = thresholds_nonlocal)
        for threshold in thresholds_nonlocal:
            number_of_arms_greater_than_threshold_cycle[threshold].append(
                number_of_arms_greater_than_threshold_cycle_[threshold]
                )
        
    return number_of_arms_greater_than_threshold_cycle, len(cycle_times)

def threshold_arm(posterior, posterior_time, time_threshold, threshold):
    # posterior_local: time x arms
    arms_above_threshold = 0
    for arm_ind in range(posterior.shape[1]):
        posterior_arm = posterior[:, arm_ind]
        above_threshold_pd = pd.Series(posterior_arm>=threshold, index = posterior_time.time)
        segments = segment_boolean_series(
            above_threshold_pd, minimum_duration=time_threshold)
        if len(segments)>0:
            arms_above_threshold += 1

    return arms_above_threshold
        
def threshold_arms(posterior, posterior_time, time_threshold, posterior_thresholds):
    # posterior_local: time x arms
    arms_above_threshold = {}
    for threshold in posterior_thresholds:
        arms_above_threshold[threshold] = threshold_arm(posterior, posterior_time, time_threshold, threshold)
    return arms_above_threshold

def theta_amplitude_to_cycle(theta_df):
    """Given theta dataframe with time, phase,
    amplitude columns, return list of start and end time per theta cycle
    """
    phase = np.array(theta_df.phase)
    time = np.array(theta_df.time)
    amplitude = np.array(theta_df.amplitude)
    
    # find phase crossings
    crossing_ind = np.argwhere(np.diff(phase) < 0).ravel()
    if len(crossing_ind) < 2:
        cycle_times = [(time[0], time[-1])]
        return cycle_times
    #np.where((phase_shifted < 0) & (phase >=0))[0]
    
    cycle_times = []
    for i in range(len(crossing_ind)-1):
        start_ind = crossing_ind[i]
        end_ind = crossing_ind[i+1]
        cycle_times.append((time[start_ind], time[end_ind]))
    
    return cycle_times
