### Two functions in this script:
# 1. Change-of-mind triggered theta power
# 2. Cycle-by-cycle analysis: those with remote content, those with long extended events, and those with neither.
import numpy as np
import pandas as pd
import xarray as xr
from spyglass.shijiegu.Analysis_SGU import (ChangeofMind, ChangeofMindRemoteTheta,
                                            ChangeofMindTheta, ChangeofMindTriggeredDecode, ThetaZscore)
from spyglass.shijiegu.ripple_add_replay import select_subset_helper, select_subset_helper_pd
from spyglass.shijiegu.theta_singleUnit import get_theta_from_cc
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.load import load_decode
from spyglass.shijiegu.decodeHelpers import session2position_name, runSessionNames
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.shijiegu.changeOfMind_figures.figure4d import load_theta_df
from scipy.signal import find_peaks

def triggered_theta_session(
    nwb_copy_file_name, session_name,
    triggered_decode_parameter_name,
    proportion = 0.1, use_spyglass = False):
    
    all_theta = []
    all_speed = []
    all_trial_info = []
    
    # 1. load triggered decode 
    epoch_num = int(session_name[:2])
            
    key_pre = {"nwb_file_name": nwb_copy_file_name, "epoch":epoch_num,
                "proportion":proportion, "parameter": triggered_decode_parameter_name}
    query = ChangeofMindTriggeredDecode & key_pre
    if len(query) == 0:
        print("No triggered decode found for ", key_pre)
        return None, None, None
            
    loaded_data = ChangeofMindTriggeredDecode().fetch1_dataframe(key_pre)
    (triggered_times_triggered, triggered_times_abs,
     triggered_trial_infos) = (
        loaded_data["time_triggered"], loaded_data["time_abs"], loaded_data["triggered_trial_info"],
        )
    
    # 2. load position info
    position_info = (IntervalPositionInfo() & {
                'nwb_file_name':nwb_copy_file_name,
                'interval_list_name':session2position_name(nwb_copy_file_name, session_name),
                'position_info_param_name':'default_decoding'}).fetch1_dataframe()
    
    # 3. load theta zscore parameters
    key = {"nwb_file_name":nwb_copy_file_name,
            "epoch":int(session_name[:2]),
            "data_type":"corpus_callosum"}
    zscore_dict = (ThetaZscore & key).fetch1("zscore")
    
    # 4. load theta LFP
    animal = nwb_copy_file_name[:5]
    key = {"nwb_file_name": nwb_copy_file_name,
        "epoch": str(session_name[:2]),
        "data_type":"corpus_callosum"}
    theta_cc = load_theta_df(key, spyglass = use_spyglass)
    theta_cc = theta_cc.set_index('time')
    
    # for each trial, find change-of-mind triggered theta
    for ind in range(len(triggered_times_abs)):
        t0t1 = (triggered_times_abs[ind][0], triggered_times_abs[ind][-1])
        print("t1-t0",t0t1[1]-t0t1[0])
        theta_subset = select_subset_helper_pd(theta_cc, t0t1)
        position_subset = select_subset_helper_pd(position_info, t0t1)
        
        t_mid = triggered_times_abs[ind][np.argmin(np.abs(triggered_times_triggered[ind]))]
        theta_subset["time"] = theta_subset.index - t_mid
        theta_subset["zscored"] = (theta_subset.amplitude ** 2 - zscore_dict["mean"]) / zscore_dict["sd"]
        position_subset.time = position_subset.index - t_mid
        all_theta.append(theta_subset)
        all_speed.append(position_subset)
        all_trial_info.append((nwb_copy_file_name, session_name, triggered_trial_infos[ind][0])) # nwb_file_name, session_name, trialID
    
    return all_theta, all_speed, all_trial_info

def trial_info_to_intervals(trial_info,
                            minimum_duration_long, parameter_name_long_theta,sd,hpd,
                            parameter_name_remote, minimum_duration_remote, min_posterior):
    
    nwb_copy_file_name = trial_info[0]
    session_name = trial_info[1]
    trialID = trial_info[2]
    
    # load extended_intervals from ChangeofMindTheta
    q_long = {"proportion": 0.1,
                 "minimum_duration":minimum_duration_long,
                  "parameter":parameter_name_long_theta,
                  "local_parameter":f"dur_{minimum_duration_long}_sd_{sd}_hpd{hpd}"
                 }
    q_long["nwb_file_name"] = nwb_copy_file_name
    
    # load remote_intervals from ChangeofMindRemoteTheta
    q_remote = q_long.copy()
    q_remote["parameter"] = parameter_name_remote
    q_remote["minimum_duration"] = minimum_duration_remote
    q_remote["remote_parameter"] = f"dur_{minimum_duration_remote}_sum_{min_posterior}" #f"parameter_name_remote
    q_long["epoch"] = int(session_name[:2])
    q_remote["epoch"] = int(session_name[:2])
            
    if len(ChangeofMindTheta() & q_long) > 0:
        long_df = ChangeofMindTheta().fetch1_dataframe(q_long)         # trials with long theta
        extended_intervals = long_df.loc[trialID, "long_theta_intervals"] # a list of tuples, each tuple is (start_time, end_time)
    else:
        extended_intervals = []

    if len(ChangeofMindRemoteTheta() & q_remote) > 0:
        remote_df = ChangeofMindRemoteTheta().fetch1_dataframe(q_remote) # trials with remote theta for now\
        remote_intervals = remote_df.loc[trialID, "remote_interval"] # a list of tuples, each tuple is (start_time, end_time)
    else:
        remote_intervals = []
    
    return extended_intervals, remote_intervals

def theta_cycle_analysis(all_theta, all_speed, all_extended_intervals, all_remote_intervals):
    all_features = []
    all_responses = []
    for ind in range(len(all_theta)):
        theta_subset = all_theta[ind]
        speed_subset = all_speed[ind]
        extended_intervals = all_extended_intervals[ind]
        remote_intervals = all_remote_intervals[ind]
        
        # find t0, the time of change-of-mind, which is the time when triggered decode is closest to 0
        t0 = theta_subset.index[np.argmin(np.abs(theta_subset.time))]
        
        # features: pre/post change-of-mind, speed, contains extended content, contains remote content
        # responses: theta power, cycle length
        theta_cycles = chop_theta_by_cycle(theta_subset)
        features, responses, feature_names, response_names = theta_cycle2features(
            theta_subset, theta_cycles, extended_intervals, remote_intervals, speed_subset, t0)
        all_features.extend(features)
        all_responses.extend(responses)
    return np.array(all_features), np.array(all_responses), feature_names, response_names

def chop_theta_by_cycle(theta_subset):
    # chop theta by cycles 
    phase = theta_subset['phase'].values
    
    times = theta_subset.index.values
    
    # find start of cycles: where phase crosses from negative to positive
    
    #crossings = (phase[:-1] < np.pi) & (phase[1:] >= np.pi)
    
    start_indices, _ = find_peaks(-phase) #np.where(crossings)[0] + 1
    
    if len(start_indices) < 2:
    
        return pd.DataFrame(columns=['start_time', 'end_time'])
    
    start_times = times[start_indices[:-1]]
    
    end_times = times[start_indices[1:]]
    
    theta_cycles = pd.DataFrame({
    
        'start_time': start_times,
    
        'end_time': end_times
    
    })
    
    return theta_cycles

def theta_cycle2features(theta_subset, theta_cycles,
                         extended_intervals, remote_intervals, #extended_intervals is a list of tuples, each tuple is (start_time, end_time)
                         speed_subset, t0): 
    # tmin is the minimum time for a cycle to be included in the analysis, unix time. For example, if tmin = 190000, then only cycles that start after 190000 will be included in the analysis.
    #t0 is the time of change-of-mind, t_back and t_forward are the time windows for pre and post theta power calculation
    feature_names = ["pre", "speed", "extended", "remote"]
    # pre is a boolean indicating whether the cycle is pre-change-of-mind or post-change-of-mind
    response_names = ["power", "length"]
    #pre-change-of-mind, speed, contains local extended content, contains remote content
    features = []
    responses = []
    for ind in range(len(theta_cycles)):
        cycle = theta_cycles.iloc[ind]
        start_time = cycle.start_time
        end_time = cycle.end_time
        
        pre = start_time < t0 
        # calculate speed during this cycle
        speed = np.mean(speed_subset[(speed_subset.index >= start_time) & (speed_subset.index < end_time)].head_speed)
        extended = any([(start_time < interval[1]) and (end_time > interval[0]) for interval in extended_intervals])
        remote = any([(start_time < interval[1]) and (end_time > interval[0]) for interval in remote_intervals])
        
        features.append([pre, speed, extended, remote])
        responses.append([np.mean(theta_subset[(theta_subset.index >= start_time) & (theta_subset.index < end_time)].zscored),
                          end_time - start_time])
    return features, responses, feature_names, response_names

def digitize_one_subset(subset, t_axis, datafield):
    lfps_binned = np.zeros_like(t_axis)
    lfps_binned_count = np.zeros_like(t_axis)
    
    bins = np.digitize(np.array(subset.time), t_axis) - 1
    data = np.abs(np.array(subset[datafield]))
    for ind in range(len(bins)):
        target_ind = bins[ind]
        lfps_binned[target_ind] += data[ind]
        lfps_binned_count[target_ind] += 1
    lfps_binned = lfps_binned/lfps_binned_count
    return bins, lfps_binned

def subsets2average(subsets, t0 = 2, t1 = 2, delta_t = 0.04, datafield = "amplitude"):
    t_axis = np.arange(-t0, t1+delta_t, delta_t)
    
    subsets_binned = []
    for subset in subsets:
        bins, lfps_binned = digitize_one_subset(subset, t_axis, datafield)
        subsets_binned.append(lfps_binned.reshape((1,-1)))
    
    subsets_binned_stacked = np.vstack(subsets_binned)
    
    mean = np.nanmean(subsets_binned_stacked, axis = 0)
    N = np.sum(~np.isnan(subsets_binned_stacked), axis = 0)
    sd = np.nanstd(subsets_binned_stacked, axis = 0)

    return t_axis, mean, sd / np.sqrt(N)#, subsets_binned

def triggered_theta_day(animal, list_of_days,
                        triggered_decode_parameter_name = "params_both_max_run_time_2_state",
                        t0 = 2, t1 = 2, delta_t = 0.04):
    all_theta_days = []
    all_speed_days = []
    all_trial_info_days = []
    for day in list_of_days:
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        for ind in range(len(session_interval)):
            session_name = session_interval[ind]
            
            all_theta, all_speed, all_trial_info = triggered_theta_session(
                nwb_copy_file_name, session_name,
                triggered_decode_parameter_name,
                proportion = 0.1)
            
            if all_theta is None:
                continue
            all_theta_days.extend(all_theta)
            all_speed_days.extend(all_speed)
            all_trial_info_days.extend(all_trial_info)
    
    # calculate average trace
    t_axis, theta_mean, theta_sd = subsets2average(all_theta_days,
                                       t0 = t0, t1 = t1, delta_t = delta_t, datafield = "zscored"
                                       )
    
    t_axis, speed_mean, speed_sd = subsets2average(all_speed_days,
                                       t0 = t0, t1 = t1, delta_t = delta_t, datafield = "head_speed"
                                       )
    
    return all_theta_days, all_speed_days, all_trial_info_days, t_axis, theta_mean, theta_sd, speed_mean, speed_sd, 




        