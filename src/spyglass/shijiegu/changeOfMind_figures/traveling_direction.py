import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pickle
from spyglass.shijiegu.changeOfMind import color_by_rat
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.decodeHelpers import runSessionNames, session2position_name
from spyglass.shijiegu.Analysis_SGU import ChangeofMind, ChangeofMindTheta, ChangeofMindRemoteTheta, Imu
from spyglass.shijiegu.changeOfMind_figures.extra_correctness import (model2numbers,
                has_remote_theta, has_outer_remote_theta,
                has_home_remote_theta, has_long_theta)
from spyglass.shijiegu.changeOfMind_triggered_position import load_triggered_position_decode_session_spyglass
from spyglass.shijiegu.changeOfMind_figures.figure3_thetaGLM import check_interval_exists
from spyglass.shijiegu.changeOfMind_figures.figure4d import select_subset_helper_pd2, segment_boolean_series, setdiff1d_stable, unique_stable, find_future_arm
from spyglass.common.common_position import TrackGraph, IntervalLinearizedPosition, IntervalPositionInfo
from spyglass.shijiegu.changeOfMind_triggered import region
from spyglass.shijiegu.ripple_add_replay import select_subset_helper, select_subset_helper_pd
from spyglass.shijiegu.changeOfMind import find_direction_dot_product, find_direction_dot_product_single_arm

parameter_name_long_theta = "params_both_max_segment_run_time_2_state"
parameter_name_remote = "params_both_max_run_time_2_state"
minimum_duration_long = 0.03
minimum_duration_remote = 0.02
min_posterior = 0.2
sd = 6
hpd = False

def return_traveling_directions_behavior(animal, list_of_days, imu_name = None, n = 3):
    backing_total_times = []
    forward_total_times = []
    for day_ind in range(len(list_of_days)):
        day = list_of_days[day_ind]
        
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        
        for session_name in session_interval:
            
            # load triggered position data
            loaded_data = load_triggered_position_decode_session_spyglass(nwb_copy_file_name, int(session_name[:2]),
                                                               "params_both_max_segment_run_time_2_state", 0.1)
            if len(loaded_data.keys()) == 0:
                continue
            
            trial_infos = loaded_data['triggered_trial_info']
            time_intervals = loaded_data["triggered_positions_baseoff"]
            
            # load linear position
            epoch_pos_name = session2position_name(nwb_copy_file_name, session_name)
            # linear_position_xr = xr.Dataset.from_dataframe((IntervalLinearizedPosition() &
            #               {'nwb_file_name': nwb_copy_file_name,
            #                'interval_list_name': epoch_pos_name,
            #                'track_graph_name': '4 arm lumped 2023',
            #                'position_info_param_name': 'default_decoding'}
            #              ).fetch1_dataframe())
            
            for ind in range(len(trial_infos)):
                trialID, arm = trial_infos[ind]
                time_interval = np.array(time_intervals[ind].index)
                
                # load IMU
                key_imu={'nwb_file_name':nwb_copy_file_name,
                        'epoch':int(session_name[:2]),
                        'trial':trialID,
                        "parameter":imu_name}
                
                query_imu = Imu() & key_imu
                if len(query_imu) == 0:
                    continue
                postion_info_gyro = Imu().fetch1_dataframe(key_imu)
                
                # for 4 equally spaced time periods from time_interval[0] to time_interval[-1]:
                ts = np.linspace(time_interval[0], time_interval[-1], n + 1)
                backing_total_times_ind = []
                forward_total_times_ind = []
                for ind_t in range(len(ts)-1):
                    t0t1 = (ts[ind_t], ts[ind_t+1])
                
                    postion_info_gyro_subset = select_subset_helper_pd(postion_info_gyro, t0t1)
                    # linear_position_subset = linear_position_xr.interp(time=np.array(postion_info_gyro_subset.index),
                    #                                     method="nearest")
                
                    # find traveling direction
                    head_direction, rightward = find_direction_dot_product_single_arm(arm, postion_info_gyro_subset)
    
                
                    # get intervals of head_direction > 0.5 and head_direction < -0.5
                    head_direction_boolean = head_direction > 0.5
                    head_direction_boolean_series = pd.Series(head_direction_boolean, index=postion_info_gyro_subset.index)
                    backing = segment_boolean_series(head_direction_boolean_series, minimum_duration=0.2)
                    if len(backing) == 0:
                        backing_total_time = 0
                    else:
                        backing_total_time = np.diff(backing, axis=1).sum()
                
                    head_direction_boolean = head_direction < -0.5
                    head_direction_boolean_series = pd.Series(head_direction_boolean, index=postion_info_gyro_subset.index)
                    forward = segment_boolean_series(head_direction_boolean_series, minimum_duration=0.2)
                    if len(forward) == 0:
                        forward_total_time = 0
                    else:
                        forward_total_time = np.diff(forward, axis=1).sum()
            
                    backing_total_times_ind.append(backing_total_time)
                    forward_total_times_ind.append(forward_total_time)
                backing_total_times.append(backing_total_times_ind)
                forward_total_times.append(forward_total_times_ind)
                
    return backing_total_times, forward_total_times

def plot_traveling_directions_prep(backing_total_times, forward_total_times, n = 3):
    # loop through each trials first
    backing_bool_all = []
    for ind in range(len(backing_total_times)):
        backing_bool = [(backing_total_times[ind][t] >= forward_total_times[ind][t]) for t in range(n)]
        backing_bool_all.append(backing_bool)
    backing_bool_all = np.array(backing_bool_all)
    # plot the mean proportion and standard error of trials that are backing vs forward for each time period
    mean_proportions = np.mean(backing_bool_all, axis=0)
    sem_proportions = np.std(backing_bool_all, axis=0) / np.sqrt(backing_bool_all.shape[0])
    return mean_proportions, sem_proportions

def plot_traveling_directions(mean_proportions, sem_proportions):
    animals = list(mean_proportions.keys())
    num_animals = len(animals)
    
    #if num_animals == 1:
    fig, ax = plt.subplots(figsize=(3,3))
    #axes = [ax]
    #else:
    #    fig, axes = plt.subplots(1, num_animals, figsize=(4*num_animals, 3))
    
    for i, animal in enumerate(animals):
        #ax = axes
        mean_prop = np.array(mean_proportions[animal])
        sem_prop = np.array(sem_proportions[animal])
        
        n = len(mean_prop)
        
        x = range(n)
        ax.plot(x, mean_prop, marker='o', linewidth=2, markersize=8, color = color_by_rat[animal])
        ax.fill_between(x, mean_prop - sem_prop, mean_prop + sem_prop, alpha=0.3, color = color_by_rat[animal])
        ax.set_xticks(range(n))
        ax.set_xticklabels([f'{i*25}-{(i+1)*25}%' for i in range(n)])
        ax.set_xlabel('Time Period In Outer Arm')
        ax.set_ylabel('Proportion')
        ax.set_title("Proportion of Trials with backing")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([0.7, 0.8, 0.9, 1])
    
    ax.set_ylim(0.7, 1)
        #ax.set_title(f"animal {animal[0].upper()}")
    
    plt.tight_layout()
    #plt.show()
    
    
    
    return fig, ax
        


def return_traveling_directions(animal, list_of_days, remote_flag = True, imu_name = None):

    directions = []
    backing_total_times = []
    forward_total_times = []
    backing_remote_event_num = []
    forward_remote_event_num = []
    
    for day_ind in range(len(list_of_days)):
        day = list_of_days[day_ind]
        
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
            
        q_long = {"proportion": 0.1,
                 "minimum_duration":minimum_duration_long,
                  "parameter":parameter_name_long_theta,
                  "local_parameter":f"dur_{minimum_duration_long}_sd_{sd}_hpd{hpd}"
                 }

        q_long["nwb_file_name"] = nwb_copy_file_name
        q_remote = q_long.copy()
        q_remote["parameter"] = parameter_name_remote
        q_remote["minimum_duration"] = minimum_duration_remote
        q_remote["remote_parameter"] = f"dur_{minimum_duration_remote}_sum_{min_posterior}" #f"parameter_name_remote
    
        for session_name in session_interval:
            q_long["epoch"] = int(session_name[:2])
            q_remote["epoch"] = int(session_name[:2])
            
            if len(ChangeofMindTheta() & q_long) > 0:
                long_df = ChangeofMindTheta().fetch1_dataframe(q_long)         # trials with long theta
            else:
                long_df = []

            if len(ChangeofMindRemoteTheta() & q_remote) > 0:
                remote_df = ChangeofMindRemoteTheta().fetch1_dataframe(q_remote) # trials with remote theta for now
            else:
                remote_df = []
            
            if remote_flag and len(remote_df) == 0:
                continue
            if not remote_flag and len(long_df) == 0:
                continue
                
            # load triggered position data
            loaded_data = load_triggered_position_decode_session_spyglass(nwb_copy_file_name, int(session_name[:2]),
                                                               "params_both_max_segment_run_time_2_state", 0.1)
            if len(loaded_data.keys()) == 0:
                continue
            
            trial_infos = loaded_data['triggered_trial_info']
            time_intervals = loaded_data["triggered_positions_baseoff"]
            
    
            # change of mind trials
            # df = ChangeofMind().fetch1_dataframe(q_remote)
            # theta_df_subset = df[df.change_of_mind]
            
            # load linear position
            # epoch_pos_name = session2position_name(nwb_copy_file_name, session_name)
            # linear_position_xr = xr.Dataset.from_dataframe((IntervalLinearizedPosition() &
            #               {'nwb_file_name': nwb_copy_file_name,
            #                'interval_list_name': epoch_pos_name,
            #                'track_graph_name': '4 arm lumped 2023',
            #                'position_info_param_name': 'default_decoding'}
            #              ).fetch1_dataframe())

    
            for ind in range(len(trial_infos)):
                trialID, arm = trial_infos[ind]
                time_interval = np.array(time_intervals[ind].index)
                t0t1 = (time_interval[0], time_interval[-1])
                
                # load IMU
                key_imu={'nwb_file_name':nwb_copy_file_name,
                        'epoch':int(session_name[:2]),
                        'trial':trialID,
                        "parameter":imu_name}
                
                query_imu = Imu() & key_imu
                if len(query_imu) == 0:
                    continue
                postion_info_gyro = Imu().fetch1_dataframe(key_imu)
                
                postion_info_gyro_subset = select_subset_helper_pd(postion_info_gyro,t0t1)
                # linear_position_subset = linear_position_xr.interp(time=np.array(postion_info_gyro_subset.index),
                #                                         method="nearest")
                
                # # find traveling direction
                head_direction, rightward = find_direction_dot_product_single_arm(arm, #find_direction_dot_product(linear_position_subset.to_dataframe(),
                                                         postion_info_gyro_subset)
                # find time intervals where head_direction > 0.5 and head_direction < -0.5
                head_direction_boolean = head_direction > 0.5
                head_direction_boolean_series = pd.Series(head_direction_boolean, index=postion_info_gyro_subset.index)
                backing = segment_boolean_series(head_direction_boolean_series, minimum_duration=0.2)
                if len(backing) == 0:
                    backing_total_time = 0
                else:
                    backing_total_time = np.diff(backing, axis=1).sum()
                
                head_direction_boolean = head_direction < -0.5
                head_direction_boolean_series = pd.Series(head_direction_boolean, index=postion_info_gyro_subset.index)
                forward = segment_boolean_series(head_direction_boolean_series, minimum_duration=0.2)
                if len(forward) == 0:
                    forward_total_time = 0
                else:
                    forward_total_time = np.diff(forward, axis=1).sum()
                    
                backing_total_times.append(backing_total_time)
                forward_total_times.append(forward_total_time)
                
                # for each remote interval, find the proportion of time spent backing vs forward
                if remote_flag:
                    remote_intervals_all = remote_df.loc[trialID, "remote_interval"]
                else:
                    remote_intervals_all = long_df.loc[trialID, "long_theta_intervals"]
                    
                remote_intervals = []
                    
                for remote_interval in remote_intervals_all:
                    if check_interval_exists([remote_interval], t0t1[0], t0t1[1]):
                        remote_intervals.append(remote_interval)
                    
                backing_remote_event_num_ = 0
                forward_remote_event_num_ = 0
                for remote_interval in remote_intervals:
                    remote_interval_subset = select_subset_helper_pd(postion_info_gyro, remote_interval)
                    # linear_position_subset_remote = linear_position_xr.interp(time=np.array(remote_interval_subset.index),
                    #                                     method="nearest")
                    if len(remote_interval_subset) == 0:
                        continue
                    head_direction_remote, rightward_remote = find_direction_dot_product_single_arm(arm, #find_direction_dot_product(linear_position_subset_remote.to_dataframe(),
                                                        remote_interval_subset)
                        
                    directions.append(np.mean(head_direction_remote))
                    if np.mean(head_direction_remote) > 0.5:
                        backing_remote_event_num_ += 1
                    elif np.mean(head_direction_remote) < -0.5:
                        forward_remote_event_num_ += 1
                
                backing_remote_event_num.append(backing_remote_event_num_)
                forward_remote_event_num.append(forward_remote_event_num_)
    
    forward_rate = np.array(forward_remote_event_num) / np.array(forward_total_times)
    backing_rate = np.array(backing_remote_event_num) / np.array(backing_total_times)
    forward_rate_notnan = forward_rate[~np.isnan(forward_rate)]
    backing_rate_notnan = backing_rate[~np.isnan(backing_rate)]
                        
    return directions, forward_rate_notnan, backing_rate_notnan
                

                