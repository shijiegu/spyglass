import numpy as np
import pandas as pd
from ripple_detection.core import segment_boolean_series
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.load import load_decode
from spyglass.shijiegu.decodeHelpers import session2position_name, runSessionNames
from spyglass.shijiegu.ripple_add_replay import select_subset_helper, select_subset_helper_pd
from spyglass.shijiegu.Analysis_SGU import get_linearization_map
from spyglass.common.common_position import TrackGraph, IntervalLinearizedPosition, IntervalPositionInfo

from spyglass.shijiegu.decodeHelpers import session2position_name
from spyglass.shijiegu.changeOfMind_remote_interval import load_remote_animal, loc1d_to_2d_vector, find_angle
from spyglass.shijiegu.Analysis_SGU import ChangeofMind, ChangeofMindRemoteTheta, MUATheta, ChangeofMindTheta
from spyglass.shijiegu.changeOfMind_triggered import (seq2, rev2, rev3, seq1, rev1,
                                                      form_null_model,
                                                      find_large_position_minus_decode_trials_lightweight, find_large_position_minus_decode_trials)
from spyglass.shijiegu.changeOfMind_triggered_position import load_triggered_position_decode_day
from spyglass.shijiegu.changeOfMind_helper import unique_stable, setdiff1d_stable
from spyglass.shijiegu.gyroscope import load_tracking_result, load_tracking_data_position
from spyglass.shijiegu.helpers import interpolate_to_new_time

same_side_map = {1:[2],2:[1],3:[4],4:[3]} 
switch_side_map = {1:[3,4],2:[3,4],3:[1,2],4:[1,2]} 

output_folder = '/stelmo/shijie/gyro/'

def find_remote_theta_dynamics_animal(animal,list_of_days,classifier_param_name,encoding_set,
                             proportion = 0.05, use_1d = 1,
                             delta_t_minus = 5,delta_t_plus = 0,
                             multiple_CoM = True, single_CoM = True, first_CoM = False,
                             max_flag = False):
    remote_event_intervals = [] # each tuple is (eventID, interval) # not trials but events of change-of-mind, the 0 of the time interval is the stopping time of that change of mind event
    long_theta_intervals = [] # like remote_event_intervals, for long theta events
    event_info = []
    
    loaded_data = load_remote_animal(
        animal, list_of_days,
        encoding_set,
        classifier_param_name,
        proportion = proportion, use_1d = use_1d)
    
    (_, remote_info,
            remote_time_intervals, remote_arm_identities
                ) = (loaded_data['all_info_animal'], loaded_data['info_animal'],
                    loaded_data['time_intervals_animal'], loaded_data['arm_identities_animal'])
    
    eventID = 0
    for day in list_of_days:
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        
        paramters = {"proportion": proportion,
                     "delta_t_minus":delta_t_minus, "delta_t_plus":delta_t_plus,
                     "max_flag":max_flag, "segment_only": False,
                     "segment_only": False,
                     "multiple_CoM":multiple_CoM, "single_CoM":single_CoM, "first_CoM":first_CoM,
                     }
        
        loaded_data_for_remote = load_triggered_position_decode_day(animal, day, encoding_set, classifier_param_name,
                                                         control = False,
                                                         **paramters)

        triggered_positions_abs, triggered_positions_zeroed, triggered_trial_infos = (
                    loaded_data_for_remote["triggered_positions_baseoff"],
                    loaded_data_for_remote["triggered_positions"],
                    loaded_data_for_remote["triggered_trial_info"],
            )
        
        paramters = {"proportion": proportion,
                     "delta_t_minus":delta_t_minus, "delta_t_plus":delta_t_plus,
                     "max_flag":max_flag, "segment_only": False,
                     "segment_only": True,
                     "multiple_CoM":multiple_CoM, "single_CoM":single_CoM, "first_CoM":first_CoM,
                     }
        
        loaded_data_segment = load_triggered_position_decode_day(animal, day, encoding_set, classifier_param_name,
                                                         control = False,
                                                         **paramters)
        
        loaded_data_control = load_triggered_position_decode_day(animal, day, encoding_set, classifier_param_name,
                                                         control = True,
                                                         **paramters)

        (triggered_positions_abs, triggered_positions_zeroed,
         triggered_trial_infos, triggered_decode) = (
                    loaded_data_for_remote["triggered_positions_baseoff"],
                    loaded_data_for_remote["triggered_positions"],
                    loaded_data_for_remote["triggered_trial_info"],
                    loaded_data_for_remote["triggered_decodes_baseoff"]
            )
         
        (triggered_positions_abs2, triggered_positions_zeroed2,
         triggered_trial_infos2, triggered_decode2) = (
                    loaded_data_segment["triggered_positions_baseoff"],
                    loaded_data_segment["triggered_positions"],
                    loaded_data_segment["triggered_trial_info"],
                    loaded_data_segment["triggered_decodes_baseoff"]
            )
        
        # make null model for deciding long theta intervals
        gaussian_process, _ , _2 = form_null_model(loaded_data_control["triggered_positions_baseoff"], loaded_data_control["triggered_decodes_baseoff"])

        for session_ind in range(len(session_interval)):
            
            session_name = session_interval[session_ind]
            position_name = position_interval[session_ind]
            epoch_num = int(session_name[:2])

            
            # load ChangeofMind info
            key={'nwb_file_name':nwb_copy_file_name,'epoch':epoch_num,'proportion': proportion}
            print(ChangeofMind & key)
            log = ChangeofMind().fetch1_dataframe(key)
            
            trigger_indices_session = [ind for ind in range(len(triggered_trial_infos)) if triggered_trial_infos[ind][1] == session_name]
            longtheta_indices_session = [ind for ind in range(len(triggered_trial_infos2)) if triggered_trial_infos2[ind][1] == session_name]
            remote_indices_session = [ind for ind in range(len(remote_info)) if remote_info[ind][0] == nwb_copy_file_name and remote_info[ind][1] == session_name]
            
            trials_session = np.unique(np.array([remote_info[ind][2][0] for ind in remote_indices_session]))
            for trialID in trials_session:
                event_indices_triggers = [ind for ind in trigger_indices_session if triggered_trial_infos[ind][1] == session_name and int(triggered_trial_infos[ind][2]) == trialID] #find all find the corresponding entries on that session and of that that trial, if any, there might be 2-3 on that trial
                event_indices_longtheta = [ind for ind in longtheta_indices_session if triggered_trial_infos2[ind][1] == session_name and int(triggered_trial_infos2[ind][2]) == trialID]
                event_indices_remote = [ind for ind in remote_indices_session if remote_info[ind][1] == session_name and int(remote_info[ind][2][0]) == trialID]
                ##
                #if len(event_indices_triggers) != len(event_indices_remote):
                #    # for now, we just do those trials where we know the correspondence
                #    continue
                
                for event_num in range(len(event_indices_remote)):
                    
                    event_index_remote = event_indices_remote[event_num]
                    
                    # find the correspondent event in the trigger
                    interval0 = remote_time_intervals[event_index_remote][0]
                    interval_last = remote_time_intervals[event_index_remote][-1]
                    
                    found_correspondence = False
                    for event_index_triggers in event_indices_triggers:
                        triggered_position = triggered_positions_abs[event_index_triggers]
                        triggered_position_zeroed = triggered_positions_zeroed[event_index_triggers]
                        t0 = np.array(triggered_position.iloc[np.argwhere(triggered_position_zeroed.index == 0).ravel()].index)
                        if t0 <= interval0[0] and triggered_position.index[-1] >= interval_last[-1]:
                            found_correspondence = True
                            break
                    
                    if not found_correspondence:
                        continue
                        
                    # find the all correspondent events in the trigger, segment
                    positions = []
                    decodes = []
                    for event_index_longtheta in event_indices_longtheta:
                        triggered_position = triggered_positions_abs2[event_index_longtheta]
                        if triggered_position.index[0] <= interval0[0] and triggered_position.index[-1] >= interval_last[-1]:
                            positions.append(triggered_position)
                            decodes.append(triggered_decode2[event_index_longtheta])

                    # find if this event has long theta,
                    # if not continue
                    flag, interval_long_theta = find_large_position_minus_decode_trials_lightweight(
                        gaussian_process, positions, decodes, minimum_duration = 0.04)
                    
                    if not flag:
                        continue
                    
                    event_info.append((eventID, nwb_copy_file_name, session_name, trialID))

                    intervals = remote_time_intervals[event_index_remote]
                    for interval in intervals:
                        remote_event_intervals.append((eventID, interval - t0))
                        
                    for interval in interval_long_theta:
                        long_theta_intervals.append((eventID, interval - t0))
                    eventID += 1
        
    return remote_event_intervals, long_theta_intervals, event_info

def find_angle_animal(animal, list_of_days, list_of_days_to_process, encoding_set,
                      classifier_param_name, proportion, use_1d = 1, debug = False, use_gyro = True):
    loaded_data = load_remote_animal(
        animal, list_of_days,
        encoding_set,
        classifier_param_name,
        proportion = proportion, use_1d = use_1d)
    
    (all_info_animal, info_animal,
        time_intervals_animal, arm_identities_animal
            ) = (loaded_data['all_info_animal'], loaded_data['info_animal'],
                loaded_data['time_intervals_animal'], loaded_data['arm_identities_animal'])
    
    angles = []
    nwb_copy_file_name_old = None
    session_name_old = None
    
    # for debug
    if debug:
        ind_list = [debug]
    else:
        ind_list = np.arange(len(info_animal))
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
        t0, t1 = time_intervals_animal[ind][0]
        arm_identity = arm_identities_animal[ind][0]
        
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
                nwb_file_name = nwb_copy_file_name.replace('_.nwb','.nwb')
                position_info = load_tracking_result(output_folder, nwb_file_name, session_name, int(trialID[0]))
                if position_info is None:
                    continue
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
    
    #if debug:
    return angles, v1, v2, max_posterior_2d, head_orientation, animal_location
    #return angles

def find_choice_animal(animal, list_of_days, encoding_set, classifier_param_name, proportion, use_1d, debug = False):
    # for each remote interval, 
    # return - if it is animal's final choice
    #        - if it is animal's past choice
    #        - if it is animal's past reward choice
    #        - the correctness of the sampled choice
    loaded_data = load_remote_animal(
        animal, list_of_days,
        encoding_set,
        classifier_param_name,
        proportion = proportion, use_1d = use_1d)
    
    (all_info_animal, info_animal,
        time_intervals_animal, arm_identities_animal
            ) = (loaded_data['all_info_animal'], loaded_data['info_animal'],
                loaded_data['time_intervals_animal'], loaded_data['arm_identities_animal'])
    
    angles = []
    nwb_copy_file_name_old = None
    session_name_old = None
    trialID_old = None
    
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
        # each is a change of mind event
        # in the case of multi-change of mind in one trial,
        # each change of mind is split into one such interval
    
        nwb_copy_file_name, session_name, trialID = info_animal[ind]
        trialID = trialID[0]

        arm_identities = arm_identities_animal[ind]
        time_intervals = time_intervals_animal[ind]
        arm_identities, time_intervals = consolidate(arm_identities, time_intervals)
        
        # 1. Get decode and animal head direction
        if session_name != session_name_old or nwb_copy_file_name_old != nwb_copy_file_name:
            # new session
            
            session_name_old = session_name
            nwb_copy_file_name_old = nwb_copy_file_name
            
        key = (nwb_copy_file_name, session_name, trialID)
        
        if key not in tally_dict_arm.keys():
            tally_dict_arm[key] = []
            tally_dict_counter[key] = 0
        for arm_ind in range(len(arm_identities)):
            arm = arm_identities[arm_ind]
            time_interval = time_intervals[arm_ind]
            tally_dict_arm[key].append((tally_dict_counter[key], time_interval, arm))
        tally_dict_counter[key] += 1
            
    #tally_dict_arm = {key:np.unique(tally_dict_arm[key]) for key in tally_dict_arm.keys()}
    return tally_dict_arm

def consolidate(arm_identities, time_intervals):
    
    arm_set = np.unique(arm_identities)
    consolidated_intervals = []
    for a in arm_set:
        ind = np.argwhere(np.array(arm_identities) == a).ravel()[-1]
        consolidated_intervals.append(time_intervals[ind])
    return arm_set, consolidated_intervals

def trial_to_features(tally_dict, correct_sequence, proportion = 0.1, debug = False):
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
                "proportion":str(proportion)}
                
            log_df = ChangeofMind().fetch1_dataframe(q)
            
            position1d = (IntervalLinearizedPosition() & {
                            'nwb_file_name':nwb_copy_file_name,
                            'interval_list_name':position_name,
                            'position_info_param_name':'default_decoding'}).fetch1_dataframe() #for debug use only
        
        intervals_to_consider = np.unique([_[0] for _ in tally_dict[key]])
        # for each change of mind incidence
        for interval_id in intervals_to_consider:
            arms = np.unique([_[2] for _ in tally_dict[key] if _[0] == interval_id])
            interval = np.concatenate([_[1] for _ in tally_dict[key] if _[0] == interval_id])
            t0 = np.max(interval)
            arms = arms[arms > 0]
            if len(arms) == 0:
                continue
            
            # get this trial's info
            if np.isnan(log_df.loc[trialID].past_reward) or np.isnan(log_df.loc[trialID].past):
                continue
            
            
            recent = int(log_df.loc[trialID].past)
            
            recent_reward = int(log_df.loc[trialID].past_reward)
            
            future_correct = int(seq[recent_reward])
            current_arm = np.nan
            
            if log_df.loc[trialID].CoMNum_by_arm == 1:
                immediate_past = recent
            else:
                # in between remote and final choice of this trial,
                # find the first the outer arm went
                t_home = log_df.loc[trialID].timestamp_H
                
                position1d_subset = select_subset_helper_pd(position1d,(t_home,t0))
                track_segment_id_bool = np.array(position1d_subset.track_segment_id) > 5
                track_segment_id_bool = pd.Series(track_segment_id_bool, index = position1d_subset.index)
                outer_intervals = segment_boolean_series(track_segment_id_bool, minimum_duration=0.2)
                
                outer_arms_before = [unique_stable(
                    select_subset_helper_pd(position1d,(outer_interval[0],outer_interval[1]
                                                                        )).track_segment_id
                    ) for outer_interval in outer_intervals]
                # exclude current arm
                current_arm = unique_stable(select_subset_helper_pd(position1d,(t0,t0+0.1)).track_segment_id)
                outer_arms_before = setdiff1d_stable(np.array(outer_arms_before), current_arm)
                
                # convert to 1 indexing
                if len(outer_arms_before) == 0:
                    immediate_past = recent
                else:
                    immediate_past = outer_arms_before[-1] - 5 
                
            if log_df.loc[trialID].CoMNum_by_arm == 1:
                future = int(log_df.loc[trialID].OuterWellIndex)
            else:
                print("parsing multiple change of mind")
                #continue
                t1 = log_df.loc[trialID].timestamp_O
                
                # in between remote and final choice of this trial,
                # find the first the outer arm went
                position1d_subset = select_subset_helper_pd(position1d,(t0,t1))
                track_segment_id_bool = np.array(position1d_subset.track_segment_id) > 5
                track_segment_id_bool = pd.Series(track_segment_id_bool, index = position1d_subset.index)
                outer_intervals = segment_boolean_series(track_segment_id_bool, minimum_duration=0.2)
                
                outer_arms_after = [unique_stable(
                    select_subset_helper_pd(position1d,(outer_interval[0],outer_interval[1]
                                                                        )).track_segment_id
                    ) for outer_interval in outer_intervals]
                # exclude current arm
                current_arm = np.unique(select_subset_helper_pd(position1d,(t0,t0+0.1)).track_segment_id)
                outer_arms_after = setdiff1d_stable(np.array(outer_arms_after), current_arm)
                
                # convert to 1 indexing
                future = outer_arms_after[0] - 5
                
            ##### parse if arm being switch-side or same-side
            last_trial_arm = int(log_df.loc[trialID].past)
            same_side_arm = same_side_map[last_trial_arm]
            switch_side_arms = switch_side_map[last_trial_arm]        
                
            feature_dict = {"recent":[immediate_past],
                            "recent_reward":[recent_reward],
                            "future":[future],
                            "future_correct":[future_correct],
                            "same_side_arm":same_side_arm,
                            "switch_side_arms":switch_side_arms}
            features, response = arm_to_features(arms, feature_dict)
            features_all.append(features)
            response_all.append(response)
            trial_info_all.append((nwb_copy_file_name,session_name, trialID,current_arm))
    
    if debug:
        return features_all, response_all, trial_info_all
    return features_all, response_all
        
        
    
def arm_to_features(arms, feature_dict):
    """helper of trial_to_features"""
    features = [] # arms
    for i in range(4):
        arm = i + 1
        feature = [int(np.isin(arm,feature_dict[f])) for f in feature_dict]
        features.append(feature)
    features = np.array(features)
    
    response = [np.isin(i+1,arms) for i in range(4)]
    
    return features, response
    
    
def time_to_phase(theta_df, time_array):
    phase_array = np.interp(time_array,
                            theta_df['time'],
                            theta_df['phase'])
    return phase_array


def remote_phase(nwb_copy_file_name, session_name, long_flag = False, data_type = "mua"):
    # data_type: "mua" or "corpus_callosum" or "sorted_pyramidal"
    
    phase_start_all = []
    phase_end_all = []

    # load long theta table
    if long_flag: #local long theta table
        pandas = (ChangeofMindTheta() & {"nwb_file_name":nwb_copy_file_name,
                                         "delta_t_minus":5,
                                        "delta_t_plus":5,
                                        "epoch":str(session_name[:2]),
                                        "max_flag":1}).fetch1("pandas")
        log_df = pd.DataFrame(pandas)
        log_df =log_df[log_df.long_theta]
        colname = 'long_theta_intervals'
    else: # remote theta table
        pandas = (ChangeofMindRemoteTheta() & {"nwb_file_name":nwb_copy_file_name,
                                        "epoch":str(session_name[:2])}).fetch1("pandas")
        log_df = pd.DataFrame(pandas)
        log_df =log_df[log_df.has_remote_interval]
        colname = 'remote_interval'

    # load theta
    key = {"nwb_file_name": nwb_copy_file_name,
        "epoch": str(session_name[:2]),
        "data_type":data_type}
    theta_pd = pd.read_csv((MUATheta() & key).fetch1("theta_xr"))

    for trialID in log_df.index:
        remote_intervals = log_df.loc[trialID,colname]
        for remote_interval in remote_intervals:
            phase0 = time_to_phase(theta_pd, remote_interval[0])
            phase_start_all.append(phase0)

            phase1 = time_to_phase(theta_pd, remote_interval[-1])
            phase_end_all.append(phase1)
            
    return phase_start_all, phase_end_all

    