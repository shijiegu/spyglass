import spyglass as nd
import pandas as pd
import numpy as np
import xarray as xr
from scipy import stats
from scipy import linalg
from scipy import ndimage
import matplotlib.pyplot as plt
from spyglass.common import (Session, IntervalList,LabMember, LabTeam, Raw, Session, Nwbfile,
                            Electrode,LFPBand,interval_list_intersect)
from spyglass.common import TaskEpoch
from spyglass.spikesorting.v0 import (SortGroup, 
                                    SpikeSortingRecording,SpikeSortingRecordingSelection)
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.common.common_position import IntervalPositionInfo, RawPosition, IntervalLinearizedPosition, TrackGraph

from ripple_detection.core import segment_boolean_series

from spyglass.shijiegu.Analysis_SGU import ChangeofMind,RippleTimesWithDecode
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.ripple_add_replay import plot_decode_spiking,select_subset_helper
from spyglass.shijiegu.changeOfMind_triggered import return_change_of_mind_times_from_log
from spyglass.shijiegu.load import load_decode

def restrict_home(log_df,linear_position_info,position_info,trial,max_range):
    """
    remove home segment in position_info
    This function is used in triggered_ripple_session().
    """
    # first restrict to this trial
    # for each trial
    start = log_df.loc[trial,'timestamp_H']
    end = log_df.loc[trial,'timestamp_O']+2
    
    # restrict to this trial's position info
    trialInd = (linear_position_info.index >= start) & (linear_position_info.index <= end)
    trialPosInfo = linear_position_info.loc[trialInd,:]
    trialPosInfo2D = position_info.loc[trialInd,:]
    
    # restrict to low speed
    lowSpeedInd = np.array(trialPosInfo2D.head_speed) <= 4
    trialPosInfo = trialPosInfo.loc[lowSpeedInd,:]
    if len(trialPosInfo) == 0:
        return ()
    
    # remove home
    homeInd = np.array(trialPosInfo.track_segment_id) != 0
    trialPosInfo = trialPosInfo.loc[homeInd,:]
    if len(trialPosInfo) == 0:
        return ()
    
    # max range
    trialInd = (trialPosInfo.index >= max_range[0]) & (trialPosInfo.index <= max_range[-1])
    trialPosInfo = trialPosInfo.loc[trialInd,:]
    if len(trialPosInfo) == 0:
        return ()
    actual_range = (trialPosInfo.index[0], trialPosInfo.index[-1])
    
    return actual_range
    

def restrict_time(log_df,linear_position_info,position_info,trial,t0,
                  post = False, both = False, nearby = True, remove_home = True):
    """
    if post = True:
        return time after change of mind.
    if post = False:
        return time before change of mind.
    if both = True:
        return time both before and after change of mind.
    restrict max range to a smaller range, 
    where the first home segment and the last outer arm segment are removed.
    This is to remove any well/reward related ripples.
    
    This function is used in triggered_ripple_session().
    """
    # first restrict to this trial
    # for each trial
    # 1. restrict to this trial
    # 2. restrict to 
    if trial >= len(log_df)-1:
        return ()
    
    if post:
        max_range = (t0, t0+5)
    elif both:
        max_range = (t0-5, t0+5)
    else:
        max_range = (t0-5, t0)

    tmin = log_df.loc[trial,'timestamp_H']
    tmax = log_df.loc[trial + 1,'timestamp_H']
    final_well = int(log_df.loc[trial,'OuterWellIndex'])

    tmin = max(tmin, max_range[0])
    tmax = min(tmax, max_range[-1])
    
    if np.isnan(tmin) or np.isnan(tmax):
        return ()
    
    # restrict to this trial's position info
    trialInd = (linear_position_info.index >= tmin) & (linear_position_info.index <= tmax)
    trialPosInfo = linear_position_info.loc[trialInd,:]
    trialPosInfo2D = position_info.loc[trialInd,:]
    
    # restrict to low speed
    lowSpeedInd = np.array(trialPosInfo2D.head_speed) <= 4
    trialPosInfo = trialPosInfo.loc[lowSpeedInd,:]
    if len(trialPosInfo) == 0:
        return ()
    
    # Here post or pre diverge:
    home_boolean = pd.Series(np.array(trialPosInfo.track_segment_id) == 0, 
                        index = trialPosInfo.index)
    home_segments = np.array(segment_boolean_series(
                        home_boolean, minimum_duration=0)).reshape((-1,2))
    if remove_home:
        if len(home_segments)>0:
            # remove first home segment to exclude any home well replays
            trialPosInfo = trialPosInfo.loc[trialPosInfo.linear_position >= 50]
        if len(trialPosInfo) == 0:
            return ()
    else:
        # just restirct to home area
        if len(home_segments)>0:
            return (home_segments[0][0], home_segments[0][1])
        else:
            return ()
        
    if not nearby:  
        # exclude the final segment in time
        same_arm_last_segment = pd.Series(np.array(trialPosInfo.track_segment_id) == (final_well + 5), 
                                            index = trialPosInfo.index)
        same_arm_last_segment_segments = np.array(segment_boolean_series(
                    same_arm_last_segment, minimum_duration=0)).reshape((-1,2))
        
        if len(same_arm_last_segment_segments) > 0:
            trialPosInfo = trialPosInfo.loc[trialPosInfo.index < same_arm_last_segment_segments[-1][0],:]
    # if in that trial that the rat did not get to another arm within the time range
    if len(trialPosInfo) == 0:
        return ()
    
    actual_range = (trialPosInfo.index[0], trialPosInfo.index[-1])
    
    return actual_range

def triggered_ripple_animal(animal, list_of_days, 
                            encoding_set = None, classifier_param_name = None, decode_threshold_method = None,
                            nearby = False,
                            post = False, both = False, home_ripple = False,
                            trials = None):
    #nearby: if 0, use change of mind trials
    #        if 1, use non-rewarded nearby not change of mind trials
    #        if 2, use rewarded nearby not change of mind trials
    ripple_ind, ranges, session_names, durations, trials_subset, trialIDs  =  {}, {}, {}, {}, None, {}
    
    for day in list_of_days:
        ripple_ind_day, ranges_day, durations_day, session_names_day, trialIDs_day = [], [], [], [], []
        
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        for ind in range(len(session_interval)):
            session_name = session_interval[ind]
            position_name = position_interval[ind]
            if trials is not None:
                if nwb_copy_file_name in trials.keys():
                    trials_subset = trials[nwb_copy_file_name][session_name]
        
            ranges_ses, ripple_ind_ses, ripple_durations_ses, trials_ses  = triggered_ripple_session(
                nwb_copy_file_name,session_name,position_name,
                encoding_set = encoding_set,
                classifier_param_name = classifier_param_name,
                decode_threshold_method = decode_threshold_method,
                post = post,
                both = both,
                home_ripple = home_ripple,
                nearby = nearby,
                trials_subset = trials_subset)
            
            if len(ripple_ind_ses) > 0:
                ripple_ind_day.append(ripple_ind_ses)
                session_names_day.append([(nwb_copy_file_name, session_name) for r in ripple_ind_ses])
                ranges_day.append(ranges_ses)
                durations_day.append(ripple_durations_ses)
                trialIDs_day.append(trials_ses)

        
        if len(ripple_ind_day) > 0:
            ripple_ind[day] = np.concatenate(ripple_ind_day)
            session_names[day] = np.concatenate(session_names_day)
            ranges[day] = np.concatenate(ranges_day)
            durations[day] = np.concatenate(durations_day)
            trialIDs[day] = np.concatenate(trialIDs_day)
        else:
            ripple_ind[day] = []
            durations[day] = []
            ranges[day] = []
            session_names[day] = []
            trialIDs[day] = []
            
    return ripple_ind, session_names, ranges, durations, trialIDs

def triggered_ripple_session(nwb_copy_file_name,session_name,position_name,proportion = 0.1,
                             nearby = False,
                             encoding_set = None, classifier_param_name = None, decode_threshold_method = None,
                             post = False, both = False, home_ripple = False, trials_subset = None):
    """
    if post = True, find ripple times post of change of mind after the outer well poke
    if post = False, find ripple times before change of mind.
    if both = True, find ripple times both before and after change of mind.
    trials_subset: ony consider those trials
    """
    if nearby > 0:
        print("Finding ripple times in nearby trials.")
        post = True

    # 1. load session's linear position info
    print('currently investigating:')
    print(session_name)
    print(position_name)
    animal = nwb_copy_file_name[:5]

    linear_position_info=(IntervalLinearizedPosition() & {
            'nwb_file_name':nwb_copy_file_name,
            'interval_list_name':position_name,
            'position_info_param_name':'default_decoding'}).fetch1_dataframe()

    position_info = (IntervalPositionInfo() & {
            'nwb_file_name':nwb_copy_file_name,
            'interval_list_name':position_name,
            'position_info_param_name':'default_decoding'}).fetch1_dataframe()
        
    # 2. load stateScript
    q = {"nwb_file_name": nwb_copy_file_name,
        "epoch":int(session_name[:2]),
        "proportion":str(proportion)}
    if len(ChangeofMind() & q) == 0:
        print(f"No change of mind on session {nwb_copy_file_name}, epoch {session_name}.")
        return [], [], [], []
    
    log_df = ChangeofMind().fetch1_dataframe(q)
    
    # 3. load ripples
    key = {"nwb_file_name": nwb_copy_file_name, "interval_list_name":session_name,
           "encoding_set":encoding_set,"classifier_param_name":classifier_param_name,"decode_threshold_method": decode_threshold_method}
    #ripple_times_query = (RippleTimes() & key).fetch1("ripple_times")
    if len(RippleTimesWithDecode() & key) == 0:
        return [], [], [], []
        
    ripple_times_query = (RippleTimesWithDecode() & key).fetch1("ripple_times")

    if type(ripple_times_query) is dict:
        ripple_times = pd.DataFrame(ripple_times_query)
    else:
        #ripple_times = pd.read_csv(ripple_times_query)
        ripple_times = pd.read_pickle(ripple_times_query)
        
    # 4. find return time
    if nearby:
        if nearby == 1:
            # non-rewarded nearby trials
            rowID, turnaround_times = return_change_of_mind_times_from_log(log_df, linear_position_info, nearby,
                                                                           multiple_CoM = False, single_CoM = False,
                                                                           first_CoM = True, last_CoM = False,
                                                                           subset_trials = "nonrewarded",
                                                                           home_ripple = home_ripple)
            if len(rowID) > 0:
                for r in rowID:
                    assert log_df.loc[r,'rewardNum'] == 1
        elif nearby == 2:
            # rewarded nearby trials
            rowID, turnaround_times = return_change_of_mind_times_from_log(log_df, linear_position_info, nearby,
                                                                           multiple_CoM = False, single_CoM = False,
                                                                           first_CoM = True, last_CoM = False,
                                                                           subset_trials = "rewarded",
                                                                           home_ripple = home_ripple)
            if len(rowID) > 0:
                for r in rowID:
                    assert log_df.loc[r,'rewardNum'] == 2
    else:
        rowID, turnaround_times = return_change_of_mind_times_from_log(log_df, linear_position_info, nearby,
                                                                       multiple_CoM = False, single_CoM = False,
                                                                       first_CoM = True, last_CoM = False,
                                                                       subset_trials = None)
    if home_ripple:
        turnaround_times = [[log_df.loc[trial,'timestamp_H']] for trial in rowID]
    
    print(session_name, len(rowID))
    # 5. for each trial, restrict time
    actual_ranges = []
    for ind in range(len(rowID)):
        trial = rowID[ind]
        # do not consider the trials in which the rat made the final arm choice that is the same as the change of mind
        if log_df.loc[trial,'OuterWellIndex'] == log_df.loc[trial,'initial_choice']:
            continue
        if trials_subset is not None and not np.isin(trial, trials_subset):
            continue
        if len(turnaround_times[ind]) == 0:
            continue
        t0 = turnaround_times[ind][0]
        
        print("pre restrict time", trial, t0)
        actual_range = restrict_time(log_df,linear_position_info, position_info,
                                     trial, t0, post = post, both = both, nearby = nearby, remove_home = not home_ripple)
        if len(actual_range) > 0:
            actual_ranges.append(actual_range)
        
    # loop through the ripple_times table
    ripple_indeces, ripple_durations = find_ripple_in_range(actual_ranges, ripple_times)

    return actual_ranges, ripple_indeces, ripple_durations, rowID

def find_ripple_in_range(actual_ranges,ripple_times):
    ripple_indeces = []   #this list tallies ripple near turn around time
    ripple_durations = [] #this list tallies the sum of ripple duration for each ripple in ripple_indeces
    for r_ind in ripple_times.index:
        start_time = ripple_times.loc[r_ind].start_time	
        end_time = ripple_times.loc[r_ind].end_time
        for actual_range in actual_ranges:
            overlap_start = max(start_time, actual_range[0])
            overlap_end = min(end_time, actual_range[-1])
            if overlap_start < overlap_end:
                ripple_indeces.append(r_ind)
                ripple_durations.append(overlap_end - overlap_start)

    return ripple_indeces, ripple_durations

def triggered_ripple_counterfactual_animal(animal, dates_to_plot, encoding_set, classifier_param_name, decode_thresh):
    triggered_positions = {}
    triggered_positions_abs = {}
    triggered_decodes = {}
    triggered_decodes_baseoff = {}
    triggered_decodes_abs = {}
    triggered_trial_info = {}

    
    # find decode and position
    (triggered_positions[animal], triggered_positions_abs[animal],
     triggered_decodes[animal], triggered_decodes_baseoff[animal], triggered_decodes_abs[animal],
     triggered_trial_info[animal]) = find_triggered_animal(animal,dates_to_plot,
                                                                       delta_t_minus = 0,delta_t_plus = 1,
                                                                       max_flag = 0, segment_only = True)
    
    # find_large_position_minus_decode_trials
    CUTOFF = 25
    replay_trials, inds = find_large_position_minus_decode_trials(animal, triggered_trial_info, 
                                                triggered_positions_abs, triggered_decodes_baseoff,cutoff = CUTOFF)
    
    trials_date_session_dict = trials_date_session_to_dict(replay_trials)
    
    # input this to triggered_ripple_animal
    (ranges, ripple_ind, session_names, ranges_nearby, ripple_ind_nearby, session_names_nearby) = triggered_ripple_animal(
        animal, dates_to_plot, encoding_set, classifier_param_name, decode_thresh, post = True, trials = trials_date_session_dict)
    return (ranges, ripple_ind, session_names, ranges_nearby, ripple_ind_nearby, session_names_nearby)
