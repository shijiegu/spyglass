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

from spyglass.shijiegu.Analysis_SGU import TrialChoice,RippleTimes,RippleTimesWithDecode
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.ripple_add_replay import plot_decode_spiking,select_subset_helper
from spyglass.shijiegu.changeOfMind import (find_turnaround_time, findProportion,
            find_trials, load_epoch_data_wrapper, find_direction, find_trials_animal)
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
    

def restrict_time(log_df,linear_position_info,position_info,trial,max_range, post = False):
    """
    if post = True:
        return time after change of mind.
    if post = False:
        return time before change of mind.
    restrict max range to a smaller range, 
    where the first home segment and the last outer arm segment are removed.
    This is to remove any well/reward related ripples.
    
    This function is used in triggered_ripple_session().
    """
    # first restrict to this trial
    # for each trial
    if trial >= len(log_df)-1:
        return ()
    if post:
        start = log_df.loc[trial,'timestamp_O']
        end = log_df.loc[trial + 1,'timestamp_H'] + 3
    else:
        start = log_df.loc[trial,'timestamp_H']
        end = log_df.loc[trial,'timestamp_O']
    if np.isnan(start) or np.isnan(end):
        return ()
    
    # restrict to this trial's position info
    trialInd = (linear_position_info.index >= start) & (linear_position_info.index <= end)
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
    if len(home_segments)>0:
        if post:
            # remove last outer arm segment to exclude the next outer well replays
            trialPosInfo = trialPosInfo.loc[trialPosInfo.index < home_segments[-1][0],:]
        else:
            # remove first home segment to exclude any home well replays
            trialPosInfo = trialPosInfo.loc[trialPosInfo.index > home_segments[0][-1],:]
    if len(trialPosInfo) == 0:
        return ()
    if post:
        return (trialPosInfo.index[0], trialPosInfo.index[-1])
        
        
    # exclude the final segment in time
    last_arm = np.array(trialPosInfo.track_segment_id)[-1]
    same_arm_last_segment = pd.Series(np.array(trialPosInfo.track_segment_id) == last_arm, 
                                          index = trialPosInfo.index)
    same_arm_last_segment_segments = np.array(segment_boolean_series(
                same_arm_last_segment, minimum_duration=0)).reshape((-1,2))
        
    trialPosInfo = trialPosInfo.loc[trialPosInfo.index < same_arm_last_segment_segments[-1][0],:]
    # if in that trial that the rat did not get to another arm within the time range
    if len(trialPosInfo) == 0:
        return ()
    
    # max range
    trialInd = (trialPosInfo.index >= max_range[0]) & (trialPosInfo.index <= max_range[-1])
    trialPosInfo = trialPosInfo.loc[trialInd,:]
    if len(trialPosInfo) == 0:
        return ()
    
    actual_range = (trialPosInfo.index[0], trialPosInfo.index[-1])
    
    return actual_range

def triggered_ripple_animal(animal, list_of_days, 
                            encoding_set = None, classifier_param_name = None, decode_threshold_method = None,
                            post = False, trials = None):
    ripple_ind = {}
    ripple_ind_nearby = {}
    ranges = {}
    ranges_nearby = {}
    session_names = {}
    session_names_nearby = {}
    
    for day in list_of_days:
        ripple_ind_day = []
        ripple_ind_nearby_day = []
        ranges_day = []
        ranges_nearby_day =[]
        session_names_day = []
        session_names_nearby_day = []
        
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        for ind in range(len(session_interval)):
            session_name = session_interval[ind]
            position_name = position_interval[ind]
            if nwb_copy_file_name in trials.keys():
                trials_subset = trials[nwb_copy_file_name][session_name]
        
            ranges_ses, ripple_ind_ses, ranges_nearby_ses, ripple_ind_nearby_ses = triggered_ripple_session(
                nwb_copy_file_name,session_name,position_name,
                encoding_set = encoding_set, classifier_param_name = classifier_param_name, decode_threshold_method = decode_threshold_method,
                post = post,
                trials_subset = trials_subset)
            
            if len(ripple_ind_ses) > 0:
                ripple_ind_day.append(ripple_ind_ses)
                session_names_day.append([(nwb_copy_file_name, session_name) for r in ripple_ind_ses])
            if len(ripple_ind_nearby_ses) > 0:
                ripple_ind_nearby_day.append(ripple_ind_nearby_ses)
                session_names_nearby_day.append(
                    [(nwb_copy_file_name, session_name) for r in ripple_ind_nearby_ses])
                
            
            if len(ranges_ses) > 0:
                ranges_day.append(ranges_ses)
            if len(ranges_nearby_ses) > 0:
                ranges_nearby_day.append(ranges_nearby_ses)
        
        if len(ripple_ind_day) > 0:
            ripple_ind[day] = np.concatenate(ripple_ind_day)
            session_names[day] = np.concatenate(session_names_day)
        else:
            ripple_ind[day] = []
        if len(ripple_ind_nearby_day) > 0:
            ripple_ind_nearby[day] = np.concatenate(ripple_ind_nearby_day)
            session_names_nearby[day] = np.concatenate(session_names_nearby_day)
        else:
            ripple_ind_nearby[day] = []
        if len(ranges_day) > 0:
            ranges[day] = np.concatenate(ranges_day)
        else:
            ranges[day] = []
        if len(ranges_nearby_day) > 0:
            ranges_nearby[day] = np.concatenate(ranges_nearby_day)
        else:
            ranges_nearby[day] = []
            
    return ranges, ripple_ind, session_names, ranges_nearby, ripple_ind_nearby, session_names_nearby

def triggered_ripple_session(nwb_copy_file_name,session_name,position_name,
                             encoding_set = None, classifier_param_name = None, decode_threshold_method = None,
                             post = False, trials_subset = None):
    """
    if post = True, find ripple times post of change of mind after the outer well poke
    if post = False, find ripple times before change of mind.
    trials_subset: ony consider those trials
    """

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
    key={'nwb_file_name':nwb_copy_file_name,'epoch':int(session_name[:2])}
    log=(TrialChoice & key).fetch1('choice_reward')
    log_df=pd.DataFrame(log)
    
    # 3. load ripples
    key = {"nwb_file_name": nwb_copy_file_name, "interval_list_name":session_name,
           "encoding_set":encoding_set,"classifier_param_name":classifier_param_name,"decode_threshold_method": decode_threshold_method}
    #ripple_times_query = (RippleTimes() & key).fetch1("ripple_times")
    ripple_times_query = (RippleTimesWithDecode() & key).fetch1("ripple_times")

    if type(ripple_times_query) is dict:
        ripple_times = pd.DataFrame(ripple_times_query)
    else:
        #ripple_times = pd.read_csv(ripple_times_query)
        ripple_times = pd.read_pickle(ripple_times_query)
        
    # 4. find return time
    rowID, _, proportions, turnaround_times = find_trials(log_df,
                                                            linear_position_info,
                                                            position_info,
                                                            proportion_threshold = 0.1)
    print(session_name, len(rowID))
    # 5. for each trial, restrict time
    actual_ranges = []
    for ind in range(len(rowID)):
        trial = rowID[ind]
        if not np.isin(trial, trials_subset):
            continue
        if len(turnaround_times[ind]) == 0:
            continue
        t0 = turnaround_times[ind][0]
        max_range = (t0-5,t0+5)
        actual_range = restrict_time(log_df,linear_position_info,position_info,
                                     trial,max_range,post = post)
        if len(actual_range) > 0:
            actual_ranges.append(actual_range)
        
    # loop through the ripple_times table
    ripple_ind = find_ripple_in_range(actual_ranges,ripple_times)
    
    # 6. find nearby non-rewarded trial
    if post:
        actual_ranges_nearby = []
        for ind in range(len(rowID)):
            trial = rowID[ind]
            if len(turnaround_times[ind]) == 0:
                continue
            candidate_trials = [trial-2, trial+2, trial-1, trial+1, trial-3, trial+3]
            for t_nearby in candidate_trials:
                if t_nearby <= 0 or t_nearby>=len(log_df)-2:
                    continue
                condition = ~np.isin(t_nearby,rowID)
                if condition:
                    actual_range_nearby = restrict_time(
                        log_df, linear_position_info, position_info, t_nearby, None, post = True)
                    if len(actual_range_nearby) > 0:
                        actual_ranges_nearby.append(actual_range_nearby)
                        break
                
        
        # loop through the ripple_times table
        ripple_ind_nearby = find_ripple_in_range(actual_ranges_nearby,ripple_times)

        return actual_ranges, ripple_ind, actual_ranges_nearby, ripple_ind_nearby
    
    actual_ranges_nearby = []
    for ind in range(len(rowID)):
        trial = rowID[ind]
        if len(turnaround_times[ind]) == 0:
            continue
        candidate_trials = [trial-2, trial+2, trial-1, trial+1, trial-3, trial+3]
        for t_nearby in candidate_trials:
            if t_nearby <= 0 or t_nearby>=len(log_df)-2:
                continue
            condition1= log_df.loc[t_nearby].rewardNum == 1
            condition2 = ~np.isin(t_nearby,rowID)
            if condition1 and condition2:
                t0 = log_df.loc[t_nearby].timestamp_O
                max_range = (t0 - 5, t0 + 5)
                # remove home arm
                actual_range_nearby = restrict_home(log_df,linear_position_info,position_info,t_nearby,max_range)
                if len(actual_range_nearby) > 0:
                    actual_ranges_nearby.append(actual_range_nearby)
                    break
                
    # loop through the ripple_times table
    ripple_ind_nearby = find_ripple_in_range(actual_ranges_nearby,ripple_times)

    return actual_ranges, ripple_ind, actual_ranges_nearby, ripple_ind_nearby

def find_ripple_in_range(actual_ranges,ripple_times):
    ripple_ind = [] #this list tallies ripple near turn around time
    for r_ind in ripple_times.index:
        start_time = ripple_times.loc[r_ind].start_time	
        end_time = ripple_times.loc[r_ind].end_time
        for actual_range in actual_ranges:
            start_condition = start_time >= actual_range[0] and start_time <= actual_range[-1]
            end_condition = end_time >= actual_range[0] and end_time <= actual_range[-1]
            if start_condition & end_condition:
                ripple_ind.append(r_ind)
    return ripple_ind

