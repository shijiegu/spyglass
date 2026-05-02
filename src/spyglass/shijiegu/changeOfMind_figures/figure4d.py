import numpy as np
import pandas as pd
import xarray as xr
from ripple_detection.core import segment_boolean_series
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.load import load_decode
from spyglass.shijiegu.decodeHelpers import session2position_name, runSessionNames
from spyglass.shijiegu.ripple_add_replay import select_subset_helper
from spyglass.shijiegu.Analysis_SGU import get_linearization_map
from spyglass.common.common_position import TrackGraph, IntervalLinearizedPosition, IntervalPositionInfo

from spyglass.shijiegu.decodeHelpers import session2position_name
from spyglass.shijiegu.changeOfMind_remote_interval import load_remote_animal, load_remote_animal_spyglass, loc1d_to_2d_vector, find_angle
from spyglass.shijiegu.Analysis_SGU import ChangeofMind, ChangeofMindRemoteTheta, MUATheta, ChangeofMindTheta
from spyglass.shijiegu.changeOfMind_triggered import (seq2, rev2, rev3, seq1, rev1,
                                                      form_null_model,
                                                      find_large_position_minus_decode_trials_lightweight, find_large_position_minus_decode_trials)
from spyglass.shijiegu.changeOfMind_triggered_position import load_triggered_position_decode_day
from spyglass.shijiegu.changeOfMind_helper import unique_stable, setdiff1d_stable
from spyglass.shijiegu.gyroscope import load_tracking_result, load_tracking_data_position
from spyglass.shijiegu.helpers import interpolate_to_new_time
from spyglass.shijiegu.Analysis_SGU import MUAThetaNWB, AnalysisNwbfile, DecodeResultsLinear, DecodeIngredients, Imu
from spyglass.shijiegu.ripple_add_replay import position_posterior2arm_posterior
from spyglass.shijiegu.changeOfMind_triggered import linear_map
from spyglass.shijiegu.changeOfMind_remote_location import region

import pynwb

same_side_map = {1:[1,2],2:[1,2],3:[3,4],4:[3,4]} 
switch_side_map = {1:[3,4],2:[3,4],3:[1,2],4:[1,2]} 

output_folder = '/stelmo/shijie/gyro/'

def find_angle_animal(animal, list_of_days, list_of_days_to_process,
                      theta_params, encoding_set,
                      classifier_param_name, proportion, use_1d = 1, debug = False,
                      use_gyro = True, imu_param_name = None):
    loaded_data = load_remote_animal(
        animal, list_of_days,
        theta_params,
        classifier_param_name,
        spyglass = True,
        proportion = proportion, use_1d = use_1d)
    
    (info_animal,
        time_intervals_animal, arm_identities_animal
            ) = (loaded_data['info_animal'],
                loaded_data['time_intervals_animal'], loaded_data['arm_identities_animal'])
    
    angles = []
    nwb_copy_file_name_old = None
    session_name_old = None
    
    # because each trial has multiple intervals, we duplicate info_animal so that each interval has its own row.
    info_animal_concatenated = []
    arm_identities_animal_concatenated = []
    for i in range(len(info_animal)):
        for j in range(len(time_intervals_animal[i])):
            info_animal_concatenated.append([info_animal[i][0], info_animal[i][1], info_animal[i][2][0]])
            arm_identities_animal_concatenated.append(arm_identities_animal[i][j])
    time_intervals_animal_concatenated = np.concatenate(time_intervals_animal)
    
    info_animal = info_animal_concatenated
    time_intervals_animal = time_intervals_animal_concatenated
    arm_identities_animal = arm_identities_animal_concatenated
    # for debug
    if debug:
        ind_list = [debug]
    else:
        ind_list = np.arange(len(info_animal))
        
    parsed_info = []
    parsed_arm = []
    parsed_animal_location = []
    parsed_max_posterior_2d = []
    parsed_head_orientation = []
    for ind in ind_list:
        # for each interval,
        # 1. load decode,
        # 2. find max decode location during that interval
        # 3. translate location to 2D location
        # 4. compute the cosine v1 = head direction v2 = animal -> 2D location
    
        nwb_copy_file_name, session_name, trialID = info_animal[ind]
        day = nwb_copy_file_name[5:13]
        if day not in list_of_days_to_process:
            continue
        t0, t1 = time_intervals_animal[ind]
        arm_identity = arm_identities_animal[ind]
        
        # 1. Get decode and animal head direction
        if session_name != session_name_old or nwb_copy_file_name_old != nwb_copy_file_name:
            
            decode = load_decode(nwb_copy_file_name,session_name,classifier_param_name,encoding_set)
            position_axis = np.array(decode.coords['position'])
            
            # 1.1. get decode and animal head direction
            position_name = session2position_name(nwb_copy_file_name, session_name)
            if not use_gyro:
                position_info = (IntervalPositionInfo() & {
                                'nwb_file_name':nwb_copy_file_name,
                                'interval_list_name':position_name,
                                'position_info_param_name':'default_decoding'}).fetch1_dataframe()
                # position_info1d = (IntervalLinearizedPosition() & {
                #                 'nwb_file_name':nwb_copy_file_name,
                #                 'interval_list_name':position_name,
                #                 'position_info_param_name':'default_decoding'}).fetch1_dataframe() #for debug use only
            else: # use gyro integrated data
                key = {"nwb_file_name": nwb_copy_file_name,
                    "epoch": str(session_name[:2]),
                    "trial": trialID,
                    "parameter": imu_param_name
                    }
                q = (Imu() & key)
                if len(q) == 0:
                    print(key)
                    continue
                else:
                    position_info = pd.DataFrame(q.fetch1("pos_info"))
                    position_info['time'] = position_info.index
                    #position_info.index.name = 'time'
                # nwb_file_name = nwb_copy_file_name.replace('_.nwb','.nwb')
                # position_info = load_tracking_result(output_folder, nwb_file_name, session_name, int(trialID[0]))
                # if position_info is None:
                #     continue
                position_info = interpolate_to_new_time(
                    position_info,
                    decode.time.values)
            session_name_old = session_name
            nwb_copy_file_name_old = nwb_copy_file_name
        
        pos2d_subset = select_subset_helper_pd(position_info,(t0,t1))
        head_orientation = np.array(pos2d_subset.head_orientation)
        animal_location = np.hstack((np.array(pos2d_subset.head_position_x).reshape(-1,1),np.array(pos2d_subset.head_position_y).reshape(-1,1)))
        
        #pos1d_subset = select_subset_helper_pd(position_info1d,(t0,t1))#for debug use only

        decode_subset = select_subset_helper(decode,(t0,t1),target_len = len(pos2d_subset),
                                                epsilon = 0.001)
        
        #assert 1 == 0
        # 2. find max decode location during that interval
        posterior_position_subset = decode_subset.causal_posterior.sum(dim='state') #causal decoder
        if len(decode_subset.time) != len(pos2d_subset):
            continue
        max_posterior_1d = np.array(position_axis[posterior_position_subset.argmax(dim = 'position')])

        # 3. translate location to 2D location
        max_posterior_2d = loc1d_to_2d_vector(max_posterior_1d, None) #exclude posterior1d in arm_identity
        # 4. compute the cosine v1 = head direction v2 = animal -> 2D location
        angles_, v1, v2 = find_angle(max_posterior_2d, head_orientation, animal_location)
        angle = np.nanmean(angles_)
        angles.append(angle)
        
        parsed_info.append(info_animal[ind])
        parsed_arm.append(arm_identity)
        parsed_animal_location.append(animal_location)
        parsed_max_posterior_2d.append(max_posterior_2d)
        parsed_head_orientation.append(head_orientation)
        
    
    #if debug:
    return angles, v1, v2, parsed_max_posterior_2d, parsed_head_orientation, parsed_animal_location, parsed_info, parsed_arm
    #return angles

def find_choice_animal(animal, list_of_days, parameter_name, proportion = 0.1, minimum_duration = 0.02, min_posterior = 0.2, debug = False):
    # for each remote interval, 
    # return - if it is animal's final choice
    #        - if it is animal's past choice
    #        - if it is animal's past reward choice
    #        - the correctness of the sampled choice
    
    loaded_data = load_remote_animal_spyglass(
        animal, list_of_days,
        parameter_name,
        minimum_duration = minimum_duration,min_posterior = min_posterior,
        proportion = proportion)
    
    nwb_copy_file_name_old = None
    session_name_old = None
    
    (info_animal,change_of_mind_num_animal,
        time_intervals_animal, arm_identities_animal
            ) = (loaded_data['info_animal'], loaded_data['change_of_mind_num_animal'],
                 loaded_data['time_intervals_animal'], loaded_data['arm_identities_animal'])
            
    # for debug
    if debug:
        ind_list = [debug]
    else:
        ind_list = np.arange(len(info_animal))
        
    # for each session
    # find trials implicated in remote replay
    # for each trial find all arms implicated in remote replay
    #    
    tally_dict_arm = {}  # keys are nwb_copy_file_name, session_name, trialID, value are arms   
    tally_dict_counter = {} #aux variable!
    for ind in ind_list:
        # each is a trial
        # in the case of multi-change of mind in one trial,
        # each change of mind is split into a sperate interval
    
        nwb_copy_file_name, session_name, trialID = info_animal[ind]
        trialID = trialID[0]

        arm_identities = arm_identities_animal[ind]
        time_intervals = time_intervals_animal[ind]
        com_num = change_of_mind_num_animal[ind]
        #arm_identities, time_intervals = consolidate(arm_identities, time_intervals)
        
        # # 1. Get decode and animal head direction
        # if session_name != session_name_old or nwb_copy_file_name_old != nwb_copy_file_name:
        #     # new session
            
        #     session_name_old = session_name
        #     nwb_copy_file_name_old = nwb_copy_file_name
            
        key = (nwb_copy_file_name, session_name, trialID)
        
        if key not in tally_dict_arm.keys():
            tally_dict_arm[key] = []
            tally_dict_counter[key] = 0
        for arm_ind in range(len(arm_identities)):
            arm = arm_identities[arm_ind]
            time_interval = time_intervals[arm_ind]
            tally_dict_arm[key].append((com_num[arm_ind], time_interval, arm))
        tally_dict_counter[key] += 1
            
    #tally_dict_arm = {key:np.unique(tally_dict_arm[key]) for key in tally_dict_arm.keys()}
    return tally_dict_arm

def consolidate(arm_identities, time_intervals):
    
    arm_set, indices =  remove_contiguous_duplicates_and_get_last_indices(arm_identities)
    consolidated_intervals = [time_intervals[ind] for ind in indices]
    
    #arm_set = np.unique(arm_identities)
    
    # consolidated_intervals = []
    # arm_set = unique_stable(arm_identities)
    # for a in indices:
    #     if arm_old == a:
    #         continue
    #     else:
    #         arm_old = a
    #     ind = np.argwhere(np.array(arm_identities) == a).ravel()[-1]
    #     consolidated_intervals.append(time_intervals[ind])
    return arm_set, consolidated_intervals

def trial_to_features(tally_dict, correct_sequence, minimum_duration =0.02, proportion = 0.1, debug = False):
    # current arm is nan if not parsing multiple change of mind. 
    """for each trial implicated in remote replay. translate to GLM feature:
    implicated in remote replay?    is choice t-1.   is previous rewarded choice.   is future.    is the correct future.
    arm 1           0                   0                       1                       1                   0
    arm 2           1                   1                       0                       0                   0
    arm 3           0                   0                       1                       1                   0
    arm 4           0                   0                       1                       1                   0
    Args:
        tally_dict (dict): it looks like this:
        {   ('lewis20240105_.nwb','02_Rev2Session1',70): [(0, array([1.70448169e+09, 1.70448169e+09]), 2)],
            ('lewis20240105_.nwb','02_Rev2Session1',98): [(0, array([1.70448251e+09, 1.70448251e+09]), 2),
                                                       (0, array([1.70448251e+09, 1.70448251e+09]), 3),
                                                       (1, array([1.70448251e+09, 1.70448251e+09]), 2),
                                                       (1, array([1.70448251e+09, 1.70448251e+09]), 3)],
        }
    """
    if correct_sequence.lower() == "seq2":
        seq = seq2
    elif correct_sequence.lower() == "rev2":
        seq = rev2
    elif correct_sequence.lower() == "seq1":
        seq = seq1
    elif correct_sequence.lower() == "rev1":
        seq = rev1
    session_name_old = None
    nwb_copy_file_name_old = None
    
    features_all = []
    response_all = []
    trial_info_all = []
    home_visit_all = []
    home_replay_all = []
    home_trial_info_all = []
    for key in tally_dict:
    
        nwb_copy_file_name, session_name, trialID = key

        if session_name != session_name_old or nwb_copy_file_name_old != nwb_copy_file_name:
            # new session
            
            session_name_old = session_name
            nwb_copy_file_name_old = nwb_copy_file_name
            position_name = session2position_name(nwb_copy_file_name, session_name)
            
            # 1.1. 
            q = {"nwb_file_name": nwb_copy_file_name,
                "epoch":int(session_name[:2]),
                "minimum_duration": minimum_duration,
                "proportion":str(proportion)}
                
            log_df = ChangeofMind().fetch1_dataframe(q)
            
            position1d = (IntervalLinearizedPosition() & {
                            'nwb_file_name':nwb_copy_file_name,
                            'interval_list_name':position_name,
                            'track_graph_name': '4 arm lumped 2023',
                            'position_info_param_name':'default_decoding'}).fetch1_dataframe() #for debug use only
        
        intervals_to_consider = np.unique([_[0] for _ in tally_dict[key]])
        
            
        # for each change of mind incidence
        for interval_id in intervals_to_consider:
            arms = np.unique([_[2] for _ in tally_dict[key] if _[0] == interval_id])
            interval = np.concatenate([_[1] for _ in tally_dict[key] if _[0] == interval_id])
            t0 = np.max(interval)
            outer_arms = arms[arms > 0]
            
            # all future intervals on this trial
            next_intervals_ = [_[0] for _ in tally_dict[key] if _[0] > interval_id]
            if len(next_intervals_) > 0:
                t2 = np.min(next_intervals_)
            else:
                t2 = log_df.loc[trialID].timestamp_O
            
            if len(outer_arms) == 0:
                print("No outer arm replay:", nwb_copy_file_name, session_name, trialID)
                #continue
            
            # get this trial's info
            if np.isnan(log_df.loc[trialID].past_reward) or np.isnan(log_df.loc[trialID].past):
                continue
                       
            reward = int(log_df.loc[trialID].rewardNum == 2)
            
            recent = int(log_df.loc[trialID].past)
            
            recent_reward = int(log_df.loc[trialID].past_reward)
            
            future_correct = int(seq[recent_reward])
            
            # current arm
            current_arm = unique_stable(select_subset_helper_pd2(position1d,(t0,t0+0.1)).track_segment_id)
            
            # past
            if log_df.loc[trialID].CoMNum_by_arm == 1 and log_df.loc[trialID].CoMNum_by_time == 1:
                immediate_past = recent
                print("\n")
                print("trial ID:", trialID)
                print("current_arm", current_arm)
                print("recent:", recent)
                
            else:
                current_arm = current_arm[current_arm > 5]
                t_home = log_df.loc[trialID].timestamp_H
                immediate_past = find_past_arm(t_home, t0, position1d, current_arm - 5, recent)
                print("\n")
                print("trial ID:", nwb_copy_file_name, session_name, trialID)
                print("current_arm", current_arm - 5)
                print("recent:", recent)
                print("immediate_past:", immediate_past)
                if immediate_past == -1 or np.isnan(immediate_past): # this means statescript and camera data disagree, corrupt data
                    immediate_past = recent

                
            if log_df.loc[trialID].CoMNum_by_arm == 1 and log_df.loc[trialID].CoMNum_by_time == 1:
                future = int(log_df.loc[trialID].OuterWellIndex)
            else:
                # print("parsing multiple change of mind")
                t1 = log_df.loc[trialID].timestamp_O
                #continue
                current_arm = current_arm[current_arm > 5]
                future = find_future_arm(t0, t1, position1d, current_arm - 5)
                if future == -1: # this means statescript and camera data disagree, corrupt data
                    continue
                
                print("trial ID:", nwb_copy_file_name, session_name, trialID)
                print("current_arm for future", current_arm - 5)
                print("future:", future)
            
            # find after each interval if there is a home visit
            if log_df.loc[trialID].CoMNum_by_arm == 1 and log_df.loc[trialID].CoMNum_by_time == 1:
                t1 = log_df.loc[trialID].timestamp_O
            else:
                t1 = t2
            position1d_subset = select_subset_helper_pd2(position1d,(t0,t1))
            track_segment_id_bool = np.array(position1d_subset.track_segment_id) == 0
            track_segment_id_bool = pd.Series(track_segment_id_bool, index = position1d_subset.index)
            home_intervals = segment_boolean_series(track_segment_id_bool, minimum_duration=0.2)
            outer_arms_after = [np.min(
                    select_subset_helper_pd2(position1d,(home_interval[0],home_interval[1]
                                                                        )).linear_position
                    ) for home_interval in home_intervals]
            home_visit = np.any(np.array(outer_arms_after) < 30) # home visit if linearized position < 30
     
                
            ##### parse if arm being switch-side or same-side
            last_trial_arm = int(log_df.loc[trialID].past)
            same_side_arm = same_side_map[last_trial_arm]
            switch_side_arms = switch_side_map[last_trial_arm]        
                
            feature_dict = {"recent":[immediate_past],
                            "recent_reward":[recent_reward],
                            "future":[future],
                            "future_correct":[future_correct],
                            "same_side_arm":same_side_arm,
                            "switch_side_arms":switch_side_arms,
                            "home_visit":home_visit}
            
            features, response, home_visit, home_replay = arm_to_features(arms, feature_dict)
            #assert np.sum(np.array(response) > 0) > 0
            
            # convert current arm to 1 indexing
            current_arm = int(current_arm[0] - 5)
            if len(outer_arms) > 0:
                features_all.append(features)
                response_all.append(response)
                trial_info_all.append((nwb_copy_file_name,session_name, trialID,current_arm, reward))
                home_trial_info_all.append((nwb_copy_file_name,session_name, trialID,home_visit, reward))
                home_visit_all.append(home_visit)
                home_replay_all.append(home_replay)
            
    if debug:
        return features_all, response_all, trial_info_all
    return features_all, response_all, trial_info_all, home_visit_all, home_replay_all, home_trial_info_all

def find_past_arm(t0, t1, position1d, current_arm, recent):
    # in between remote and last choice of this trial,
    # find the last the outer arm went
    
    position1d_subset = select_subset_helper_pd2(position1d,(t0,t1))
    track_segment_id_bool = np.array(position1d_subset.track_segment_id) > 5
    track_segment_id_bool = pd.Series(track_segment_id_bool, index = position1d_subset.index)
    outer_intervals = segment_boolean_series(track_segment_id_bool, minimum_duration=0.2)
                
    outer_arms_before = [collapse_duplicate(
        np.array(select_subset_helper_pd2(position1d,(outer_interval[0],outer_interval[1]
                                                                            )).track_segment_id)
            ) for outer_interval in outer_intervals]
    if len(outer_arms_before) == 0:
        return recent
    
    outer_arms_before = np.concatenate(outer_arms_before)
    if outer_arms_before[-1] == (current_arm + 5):
        outer_arms_before = outer_arms_before[:-1]
        
    if len(outer_arms_before) == 0:
        print("no recent on this trial")
        return recent
    
    do_not_consider_arms = None
    past = -1
    while past == -1:
        
        #current_arm = np.unique(select_subset_helper_pd2(position1d,(t0,t0+0.1)).track_segment_id)
        if outer_arms_before[-1] == do_not_consider_arms:
            outer_arms_before = outer_arms_before[:-1]
        
        # exclude current arm
        if len(outer_arms_before) == 0:
            print("failed to find the immediate past")
            return recent
        
        # print("outer_arms_before",outer_arms_before)
        # print("do_not_consider_arms",do_not_consider_arms)
                
        # convert to 1 indexing
        past = outer_arms_before[-1] - 5
        
        # make sure the rat has entered the future arm for at least 5 cm
        # select position1d data where the track_segment_id is future + 5
        past_position1d = select_subset_helper_pd2(position1d, (t0,t1))
        past_position1d = past_position1d[past_position1d.track_segment_id == past + 5]
        
        # subtract off the base location of the future arm
        past_position1d.linear_position = past_position1d.linear_position - region[past + 5][0]
        
        # check if the rat has entered the future arm for at least 5 cm
        if np.max(past_position1d.linear_position) < 5:
            print("rat did not enter the future arm for at least 5 cm, excluding this event")
            do_not_consider_arms = past + 5
            past = -1
            
    
    return past
          
def find_future_arm(t0, t1, position1d, current_arm):
    print("parsing multiple change of mind")
    
    future = -1
    
    # in between remote and final choice of this trial,
    # find the first the outer arm went
    position1d_subset = select_subset_helper_pd2(position1d,(t0,t1))
    track_segment_id_bool = np.array(position1d_subset.track_segment_id) > 5
    track_segment_id_bool = pd.Series(track_segment_id_bool, index = position1d_subset.index)
    outer_intervals = segment_boolean_series(track_segment_id_bool, minimum_duration=0.2)
                    
    outer_arms_after = [collapse_duplicate(
        np.array(select_subset_helper_pd2(position1d,(outer_interval[0],outer_interval[1]
                                                                            )).track_segment_id)
            ) for outer_interval in outer_intervals]
    outer_arms_after = np.concatenate(outer_arms_after)
    if outer_arms_after[0] == (current_arm + 5):
        outer_arms_after = outer_arms_after[1:]
    
    do_not_consider_arms = None
    while future == -1:
                
        
        #current_arm = np.unique(select_subset_helper_pd2(position1d,(t0,t0+0.1)).track_segment_id)
        if outer_arms_after[0] == do_not_consider_arms:
            outer_arms_after = outer_arms_after[1:]
        
        # exclude current arm
        if len(outer_arms_after) == 0:
            return future
        
        # print("outer_arms_after",outer_arms_after)
        # print("do_not_consider_arms",do_not_consider_arms)
                
        # convert to 1 indexing
        future = outer_arms_after[0] - 5
        
        # make sure the rat has entered the future arm for at least 5 cm
        # select position1d data where the track_segment_id is future + 5
        future_position1d = select_subset_helper_pd2(position1d, (t0,t1))
        future_position1d = future_position1d[future_position1d.track_segment_id == future + 5]
        
        # subtract off the base location of the future arm
        future_position1d.linear_position = future_position1d.linear_position - region[future + 5][0]
        
        # check if the rat has entered the future arm for at least 5 cm
        if np.max(future_position1d.linear_position) < 5:
            print("rat did not enter the future arm for at least 5 cm, excluding this event")
            do_not_consider_arms = future + 5
            future = -1
            
    
    return future

def collapse_duplicate(arr):
    """
    Remove neighoring duplicate elements.
    given arr = np.array([1,1,2,2,3,4,1,1]), return np.array([1,2,3,4,1])
    """
    if len(arr) == 0:
        return arr
    result = [arr[0]]
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            result.append(arr[i])
    arr_dup = np.array(result)
    return arr_dup        
    
def arm_to_features(arms, feature_dict, feature_names = ["recent","recent_reward", "future", "future_correct", "same_side_arm", "switch_side_arms"]):
    """helper of trial_to_features"""
    features = [] # arms
    arms_outer = arms[arms > 0]
    for i in range(4):
        arm = i + 1
        feature = [int(np.isin(arm,feature_dict[f])) for f in feature_names]
        features.append(feature)
    features = np.array(features)
    
    response = [np.isin(i+1,arms_outer) for i in range(4)]
    home_replay = int(np.isin(0, arms))
    home_visit = feature_dict["home_visit"]
    
    return features, response, home_visit, home_replay
    
    
def time_to_phase(theta_df, time_array):
    phase_array = np.interp(time_array,
                            theta_df['time'],
                            theta_df['phase'])
    return phase_array


def remote_phase(nwb_copy_file_name, session_name, decode_name, minimum_duration,
                 proportion = 0.1,
                 min_posterior = None, sd = None, hpd = False,
                 long_flag = False, data_type = "mua"):
    # data_type: "mua" or "corpus_callosum" or "sorted_pyramidal"
    
    if long_flag:
        local_parameter = f"dur_{minimum_duration}_sd_{sd}_hpd{hpd}"
        label = "local_parameter"
    else:
        local_parameter = f"dur_{minimum_duration}_sum_{min_posterior}"
        label = "remote_parameter"
    
    phase_start_all = []
    phase_end_all = []

    # load long theta table
    key = {
                    "nwb_file_name":nwb_copy_file_name,
                    "epoch":str(session_name[:2]),
                    "proportion": proportion,
                    "parameter":decode_name,
                    label:local_parameter}
    
    if long_flag: #local long theta table
        query = ChangeofMindTheta() & key
        if len(query) == 0:
            return phase_start_all, phase_end_all
        pandas = query.fetch1("pandas")
        log_df = pd.DataFrame(pandas)
        log_df =log_df[log_df.long_theta]
        colname = 'long_theta_intervals'
    else: # remote theta table
        query = ChangeofMindRemoteTheta() & key
        if len(query) == 0:
            return phase_start_all, phase_end_all
        pandas = query.fetch1("pandas")
        log_df = pd.DataFrame(pandas)
        log_df =log_df[log_df.has_remote_interval]
        colname = 'remote_interval'

    # load theta
    key = {"nwb_file_name": nwb_copy_file_name,
        "epoch": str(session_name[:2]),
        "data_type":data_type}
    theta_pd = load_theta_df(key, spyglass = True)

    for trialID in log_df.index:
        remote_intervals = log_df.loc[trialID,colname]
        for remote_interval in remote_intervals:
            phase0 = time_to_phase(theta_pd, remote_interval[0])
            phase_start_all.append(phase0)

            phase1 = time_to_phase(theta_pd, remote_interval[-1])
            phase_end_all.append(phase1)
            
    return phase_start_all, phase_end_all

def select_subset_helper_pd2(xr_ob,plottimes):
    # assumes index is time, in seconds
    (t0_peak,t1_peak) = (plottimes[0]-0.001, plottimes[1]+0.001)
    subset_ind = (xr_ob.index >= t0_peak) & (xr_ob.index <= t1_peak)
    subset = xr_ob.loc[subset_ind]
        
    return subset


def select_subset_helper_pd(xr_ob,plottimes):
    # assumes index is time, in seconds
    (t0_peak,t1_peak) = (plottimes[0]-0.001, plottimes[1]+0.001)
    subset_ind = (xr_ob.time >= t0_peak) & (xr_ob.time <= t1_peak)
    subset = xr_ob.loc[subset_ind]
        
    return subset

def remote_phase_by_bin(nwb_copy_file_name, session_name, decode_name, minimum_duration,
                 proportion = 0.1,
                 min_posterior = None, sd = None, hpd = False,
                 long_flag = False, data_type = "mua", use_spyglass = False):
    local_parameter = f"dur_{minimum_duration}_sum_{min_posterior}"
    label = "remote_parameter"
    
    phase = []
    p_local = []
    
    classifier_param_name = "default_decoding_gpu_4armMaze"
        
    if "all_maze" in decode_name:
        encoding_set = "all_maze"
    elif "run" in decode_name:
        encoding_set = '2Dheadspeed_above_4'
    else:
        encoding_set = '2Dheadspeed_above_4'

    # load long theta table
    key = {
                    "nwb_file_name":nwb_copy_file_name,
                    "epoch":str(session_name[:2]),
                    "proportion": proportion,
                    "parameter":decode_name,
                    label:local_parameter}
    
    query = ChangeofMindRemoteTheta() & key
    if len(query) == 0:
        return phase, p_local
    
    pandas = query.fetch1("pandas")
    log_df = pd.DataFrame(pandas)
    log_df =log_df[log_df.has_remote_interval]
    colname = 'remote_interval'

    # load theta
    key = {"nwb_file_name": nwb_copy_file_name,
        "epoch": str(session_name[:2]),
        "data_type":data_type}
    theta_pd = load_theta_df(key, spyglass = use_spyglass)
    
    # load decode
    decode_path = (DecodeResultsLinear & {"nwb_file_name":nwb_copy_file_name,
                                          "interval_list_name":session_name,
                                          "encoding_set":encoding_set,
                                          "classifier_param_name":classifier_param_name}).fetch1("posterior")
    decode = xr.open_dataset(decode_path)
    
    # load 1D data
    ## load LinearPosition
    pos1d = pd.read_csv((DecodeIngredients & {"nwb_file_name":nwb_copy_file_name,
                                        "interval_list_name":session_name}).fetch1("position_1d"))

    for trialID in log_df.index:
        remote_intervals = log_df.loc[trialID,colname]
        for remote_interval in remote_intervals:
            (t0, t1) = (remote_interval[0] - 0.06, remote_interval[-1] + 0.06)
            pos1d_subset_center = select_subset_helper_pd(pos1d, [remote_interval[0], remote_interval[-1]])
            arm_id = int(np.array(pos1d_subset_center.track_segment_id - 5)[0])
            # restrict to time animal is in outer arm
            arm_track_segment = int(np.array(pos1d_subset_center.track_segment_id)[0])
            print("arm_track_segment", arm_track_segment)
            
            # position
            pos1d_subset = select_subset_helper_pd(pos1d, [t0, t1])
            pos1d_subset = pos1d_subset[pos1d_subset.track_segment_id == arm_track_segment]
            (t0, t1) = (np.array(pos1d_subset.time)[0], np.array(pos1d_subset.time)[-1])
            
            # get decode
            decode_subset = select_subset_helper(decode, [t0, t1])
            posterior_position_subset = decode_subset.causal_posterior.sum(dim='state')
            
            if len(pos1d_subset) - len(posterior_position_subset.time) > 3:
                continue
            
            t_axis = np.arange(t0,t1,0.004)
            phase0 = time_to_phase(theta_pd, t_axis)
            
            # map posterior over location to posterior over arm 
            posterior_by_arm = position_posterior2arm_posterior(posterior_position_subset,linear_map)
            
            
            # get local posterior
            posterior_local = np.array(posterior_by_arm[arm_id, :])
            
            posterior_local_interp = np.interp(t_axis, #query
                            np.array(posterior_position_subset.time),
                            posterior_local)
            p_local.extend(posterior_local_interp.tolist())
            phase += phase0.tolist()

    return phase, p_local

def distance_phase_by_bin(nwb_copy_file_name, session_name, decode_name, minimum_duration,
                 proportion = 0.1,
                 sd = None, hpd = False,
                 use_spyglass = True,
                 data_type = "mua"):
    local_parameter = f"dur_{minimum_duration}_sd_{sd}_hpd{hpd}"
    label = "local_parameter"
    
    phase = []
    p_local = []
    
    classifier_param_name = "default_decoding_gpu_4armMaze"
        
    if "all_maze" in decode_name:
        encoding_set = "all_maze"
    elif "run" in decode_name:
        encoding_set = '2Dheadspeed_above_4'
    else:
        encoding_set = '2Dheadspeed_above_4'

    # load long theta table
    key = {
                    "nwb_file_name":nwb_copy_file_name,
                    "epoch":str(session_name[:2]),
                    "proportion": proportion,
                    "parameter":decode_name,
                    label:local_parameter}
    
    query = ChangeofMindTheta() & key
    if len(query) == 0:
        return phase, p_local
    
    pandas = query.fetch1("pandas")
    log_df = pd.DataFrame(pandas)
    log_df =log_df[log_df.long_theta]
    colname = 'long_theta_intervals'

    # load theta
    key = {"nwb_file_name": nwb_copy_file_name,
        "epoch": str(session_name[:2]),
        "data_type":data_type}
    theta_pd = load_theta_df(key, spyglass = use_spyglass)
    
    # load decode
    decode_path = (DecodeResultsLinear & {"nwb_file_name":nwb_copy_file_name,
                                          "interval_list_name":session_name,
                                          "encoding_set":encoding_set,
                                          "classifier_param_name":classifier_param_name}).fetch1("posterior")
    decode = xr.open_dataset(decode_path)
    position_axis = np.array(decode.coords['position'])
    
    # load 1D data
    ## load LinearPosition
    pos1d = pd.read_csv((DecodeIngredients & {"nwb_file_name":nwb_copy_file_name,
                                        "interval_list_name":session_name}).fetch1("position_1d"))

    for trialID in log_df.index:
        remote_intervals = log_df.loc[trialID,colname]
        for remote_interval in remote_intervals:
            (t0, t1) = (remote_interval[0] - 0.06, remote_interval[-1] + 0.06)
            pos1d_subset_center = select_subset_helper_pd(pos1d, [remote_interval[0], remote_interval[-1]])
            arm_id = int(np.array(pos1d_subset_center.track_segment_id - 5)[0])
            # restrict to time animal is in outer arm
            arm_track_segment = int(np.array(pos1d_subset_center.track_segment_id)[0])
            
            # position
            pos1d_subset = select_subset_helper_pd(pos1d, [t0, t1])
            pos1d_subset = pos1d_subset[pos1d_subset.track_segment_id == arm_track_segment]
            (t0, t1) = (np.array(pos1d_subset.time)[0], np.array(pos1d_subset.time)[-1])
            
            # get decode
            decode_subset = select_subset_helper(decode, [t0, t1])
            posterior_position_subset = decode_subset.causal_posterior.sum(dim='state')
            
            if len(pos1d_subset) - len(posterior_position_subset.time) > 3:
                continue
            
            t_axis = np.arange(t0,t1,0.004)
            
            
            tracking = np.array(pos1d_subset.linear_position)
            
            posterior_position_subset = decode_subset.causal_posterior.sum(dim='state')
            max_posterior_position = np.array(position_axis[posterior_position_subset.argmax(dim = 'position')])
            
            # TODO: RESTRICT DECODE TO THE ARM THE ANIMAL IS IN.
            local_flag = np.logical_and(
                max_posterior_position >= region[arm_track_segment][0],
                max_posterior_position <= region[arm_track_segment][1])
            is_local = pd.Series(local_flag, index = np.array(posterior_position_subset.time))
            is_local_segments = np.array(segment_boolean_series(
                is_local, minimum_duration=0.1))
      
            time_tracking = np.array(pos1d_subset.time)
            time_decode = np.array(posterior_position_subset.time)
            for segment in is_local_segments:
                t_axis = np.arange(segment[0],segment[-1]+0.004,0.004)
            
                # distance
                tracking_ind = np.logical_and(time_tracking>=segment[0],
                                              time_tracking<=segment[-1])
                tracking_interp = np.interp(t_axis, #query
                            time_tracking[tracking_ind],
                            tracking[tracking_ind])
                
                decode_ind = np.logical_and(time_decode>=segment[0],
                                              time_decode<=segment[-1])
                decode_interp = np.interp(t_axis, #query
                                time_decode[decode_ind],
                                max_posterior_position[decode_ind])
                distance_interp = np.abs(tracking_interp - decode_interp)
                
                phase0 = time_to_phase(theta_pd, t_axis)
                
                p_local += distance_interp.tolist()
                phase += phase0.tolist()

    return phase, p_local
    
    
    

def load_theta_df(key, spyglass = False):
    if not spyglass:
        theta_pd = pd.read_csv((MUATheta() & key).fetch1("theta_xr"))
        return theta_pd
    
    analysis_nwb_file_name = (MUAThetaNWB() & key).fetch1("analysis_file_name")
    analysis_file_abs_path = (AnalysisNwbfile() & {
        "analysis_file_name":analysis_nwb_file_name}).fetch1('analysis_file_abs_path')
    with pynwb.NWBHDF5IO(analysis_file_abs_path, 'r',load_namespaces=True) as io:
        nwb_file = io.read()
        df = nwb_file.scratch["theta"].to_dataframe()
    return df
        
def remove_contiguous_duplicates_and_get_last_indices(data_list):
    """
    Removes contiguous duplicates in a list and returns the indices corresponding to
    the last element in each contiguous region of duplicates in the original list.

    Args:
        data_list (list): The input list with potential contiguous duplicates.

    Returns:
        tuple: A tuple containing two lists:
               - The list with contiguous duplicates removed.
               - The indices of the last element in each original contiguous region.
    """
    if not data_list:
        return [], []

    # Lists to store the unique values and their corresponding last indices
    unique_list = []
    last_indices = []

    # Variables to track the current unique element and the start of its contiguous region
    current_element = data_list[0]
    start_index = 0

    # Iterate through the list starting from the second element
    for i in range(1, len(data_list)):
        if data_list[i] != current_element:
            # A new contiguous region has started
            unique_list.append(current_element)
            # The last element of the previous region was at the index before 'i'
            last_indices.append(i - 1)

            # Update the current element and start index for the new region
            current_element = data_list[i]
            start_index = i

    # The loop finishes before processing the last contiguous region
    # Add the last unique element and its last index (the end of the list)
    unique_list.append(current_element)
    last_indices.append(len(data_list) - 1)

    return unique_list, last_indices

# GLM for location vs replay amount
import numpy.matlib as ml
from scipy.stats import ranksums
from scipy.stats import fisher_exact
import starbars
import statsmodels.api as sm

def GLM_by_animal(hist_pct):
    GLM_x = []
    GLM_y = []
    arms = [0,1,2,3,4]
    for arm in arms:
        # add arm catogory
        arm_catogory = np.array([_ == arm for _ in arms]).reshape((1,-1))
        
        pct_arm = hist_pct[arm]
        for bin in range(len(pct_arm)):
            bin_catogory = np.array([_ == bin for _ in range(len(pct_arm))]).reshape((1,-1))
            x = np.hstack((bin_catogory, arm_catogory))
            GLM_x.append(x)
            GLM_y.append(pct_arm[bin])

    GLM_x = np.vstack(GLM_x).astype("float32")
    GLM_y = np.hstack(GLM_y)
    
    feature_names = ['bin1', 'bin2', 'bin3', 'bin4', 'bin5']
    x_dict = {feature: GLM_x[:,feature_ind] for feature_ind, feature in enumerate(feature_names)}

    x = pd.DataFrame(x_dict)
    x_ = sm.add_constant(x)
    ols_model = sm.OLS(np.log(GLM_y + 0.0001), x_)
    result = ols_model.fit()#arize
    

    return ols_model, result

def GLM_mixed(hist_pct):
    return None


    