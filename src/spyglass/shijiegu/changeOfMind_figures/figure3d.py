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
from spyglass.shijiegu.changeOfMind_remote_interval import load_remote_animal, load_remote_animal_spyglass, loc1d_to_2d_vector, find_angle
from spyglass.shijiegu.Analysis_SGU import ChangeofMind, ChangeofMindRemoteTheta, MUATheta, ChangeofMindTheta
from spyglass.shijiegu.changeOfMind_triggered import (seq2, rev2, rev3, seq1, rev1,
                                                      form_null_model,
                                                      find_large_position_minus_decode_trials_lightweight, find_large_position_minus_decode_trials)
from spyglass.shijiegu.changeOfMind_triggered_position import load_triggered_position_decode_day
from spyglass.shijiegu.changeOfMind_helper import unique_stable, setdiff1d_stable
from spyglass.shijiegu.gyroscope import load_tracking_result, load_tracking_data_position
from spyglass.shijiegu.helpers import interpolate_to_new_time

same_side_map = {1:[1,2],2:[1,2],3:[3,4],4:[3,4]} 
switch_side_map = {1:[3,4],2:[3,4],3:[1,2],4:[1,2]} 

output_folder = '/stelmo/shijie/gyro/'

def find_theta_dynamics_animals(animal, list_of_days,
                 decode_name_long, parsing_name_long,
                 decode_name_remote, parsing_name_remote, proportion = 0.1):
    
    remote_intervals = []
    long_intervals = []
    event_info = []
    
    eventID = 0
    
    for day_ind in range(len(list_of_days)):
        day = list_of_days[day_ind]
        
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
            
        q_long = {"proportion": proportion,
                  "parameter":decode_name_long,
                 "local_parameter":parsing_name_long,
                 }

        q_long["nwb_file_name"] = nwb_copy_file_name
        q_remote = q_long.copy()
        q_remote["parameter"] = decode_name_remote
        q_remote["remote_parameter"] = parsing_name_remote

    
        for session_name in session_interval:
            q_long["epoch"] = int(session_name[:2])
            q_remote["epoch"] = int(session_name[:2])
            
            if len(ChangeofMindTheta() & q_long) == 0:
                continue
                
            long_df = ChangeofMindTheta().fetch1_dataframe(q_long)         # trials with long theta
            if len(ChangeofMindRemoteTheta() & q_remote) == 0:
                continue
            remote_df = ChangeofMindRemoteTheta().fetch1_dataframe(q_remote) # trials with remote theta for now
            
            # find trials with both remote and local long theta events
            trial_long = np.array(long_df[long_df.long_theta].index)
            trial_remote = np.array(remote_df[remote_df.has_remote_interval].index)
            trialIDs = np.intersect1d(trial_long, trial_remote)
            for trialID in trialIDs:
                intvl_long = long_df.loc[trialID].long_theta_intervals
                com_ids = long_df.loc[trialID].change_of_mind_num
                t0 = long_df.loc[trialID].initial_time
                for intvl_ind in range(len(intvl_long)):
                    intvl = intvl_long[intvl_ind]
                    com_id = com_ids[intvl_ind]
                    if com_id == 0:
                        long_intervals.append((eventID, intvl - t0))
                    
                intvl_remote = remote_df.loc[trialID].remote_interval
                com_ids = remote_df.loc[trialID].change_of_mind_num
                t0 = remote_df.loc[trialID].initial_time
                for intvl_ind in range(len(intvl_remote)):
                    intvl = intvl_remote[intvl_ind]
                    com_id = com_ids[intvl_ind]
                    if com_id == 0:
                        remote_intervals.append((eventID, intvl - t0))
                    
                event_info.append((eventID, nwb_copy_file_name, session_name, trialID))
                eventID += 1
            
    return remote_intervals, long_intervals, event_info        
        

def find_remote_theta_dynamics_animal_old(animal,list_of_days,classifier_param_name,encoding_set,
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