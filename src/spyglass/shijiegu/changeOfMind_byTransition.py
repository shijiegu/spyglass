import pandas as pd
import numpy as np
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.Analysis_SGU import ChangeofMind
from spyglass.common.common_position import IntervalPositionInfo, RawPosition, IntervalLinearizedPosition
import random

def find_COM_transitions(animal, dates_to_plot, proportion_threshold = 0.1, nearby = False):
    """return the number of COM trials for each day in dates_to_plot"""
    #trials_days = find_trials_animal(animal,dates_to_plot,proportion_threshold = proportion_threshold)

    P_all = []
    P_wouldhave_all = []
    
    for day in dates_to_plot:
        
        P_day = np.zeros((4,4))
        P_wouldhave_day = np.zeros((4,4))
        
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        
        for session_ind in range(len(session_interval)):
            session, pos_name = session_interval[session_ind], position_interval[session_ind]
            
            # load Change of Mind info
            q = {"nwb_file_name":nwb_copy_file_name,
                 "epoch":int(session[:2]),
                 "proportion": proportion_threshold}
            q_result = ChangeofMind() & q
            if len(q_result) == 0:
                continue
            #info = pd.read_pickle(q_result.fetch1("change_of_mind_info"))
            info = ChangeofMind().fetch1_dataframe(q)
            
            # load position info
            P, P_wouldhave = info2matrix(info, nearby = nearby)
            P_day += P
            P_wouldhave_day += P_wouldhave
        
        P_all.append(P_day)
        P_wouldhave_all.append(P_wouldhave_day)

    return P_all, P_wouldhave_all

def info2matrix(info, nearby = False):
    """Considering only the first change of mind"""
    P = np.zeros((4,4))
    P_wouldhave = np.zeros((4,4))
    
    trials = info[info['change_of_mind']].index
    if nearby:
        trials_nearby = [ return_a_nearby_random_trial(t,
                                                       trials,
                                                       min_trial = 1, max_trial = len(info) - 1) for t in trials ]
        trials = trials_nearby
        
    for ind in trials:
        if ind == 1:
            continue # do not parse the 1st trial
        """previous trial"""
        i = int(info.loc[ind - 1,'OuterWellIndex'])
            
        """initial choice change of mind"""
        #CoM_t = info.loc[ind,'CoM_t']
        #if len(CoM_t[0]) == 0:
        #    continue
        #t = CoM_t[0][0]
        #j_wouldhave = time2arm(t, linear_position_info)
        j_wouldhave = info.loc[ind,'initial_choice']
        if j_wouldhave is None or np.isnan(j_wouldhave):
            # we do not have camera data for this time.
            if not nearby:
                print(f"Warning: trial {ind} does not have position data for initial choice")
                continue
            else:
                j_wouldhave = i  # nearby trial, no change of mind
        
        """final choice change of mind"""
        j = int(info.loc[ind,'OuterWellIndex'])
        
        P[int(i) - 1, int(j) - 1] += 1
        P_wouldhave[int(i) - 1, int(j_wouldhave) - 1] += 1
        
    return P, P_wouldhave

def return_a_nearby_random_trial(t0, change_of_mind_trials, min_trial = 1, max_trial = 79):
    candidate_trials = [t0-1, t0+1, t0-2, t0+2, t0-3, t0+3]
    t0_rand = np.nan
    for t in random.sample(candidate_trials, len(candidate_trials)):
            
        condition1 = ~np.isin(t, change_of_mind_trials)
        condition2 = t >= min_trial and t <= max_trial
            
        if condition1 and condition2:
            t0_rand = t
            break
    return t0_rand

def time2arm(t, linear_position_info):
    """Given t, find animal outer arm location"""

    (t0_peak,t1_peak) = (t-0.1, t+0.1)
    subset_ind = (linear_position_info.index >= t0_peak) & (linear_position_info.index <= t1_peak)
    subset_linear = linear_position_info.loc[subset_ind]
    if len(subset_linear) == 0:
        return None
    arm = np.max(np.unique(subset_linear.track_segment_id)) - 5

    return arm