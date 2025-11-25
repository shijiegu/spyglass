import numpy as np
import pandas as pd
import pickle
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import seaborn as sns
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.Analysis_SGU import ChangeofMindTheta
from spyglass.shijiegu.changeOfMind_remote_interval import find_remote_theta_animal
from spyglass.shijiegu.changeOfMind_triggered import seq2, rev2, rev3, seq1, rev1



def remote_theta_count_per_day(loaded_data):
    """tally trials with remote theta in other arms """
    
    (all_info_animal, info_animal,
        time_intervals_animal, arm_identities_animal
        ) = (
         loaded_data['all_info_animal'], loaded_data['info_animal'],
         loaded_data['time_intervals_animal'], loaded_data['arm_identities_animal'])
    
    # find days
    days = np.unique([a[0] for a in all_info_animal])
    
    total_num_trials = []
    remote_num_trials = []
    dates = []
    dates2session_trial = {}
    for d in days:
        
        day = d[5:13]
        
        #a = len([a[1] for a in all_info_animal if a[0] == d])
        #b = len([a[1] for a in info_animal if a[0] == d])
        
        
        # sessions considered 
        sessions = np.unique([a[1] for a in info_animal if a[0] == d])
        
        # trials considered on this day
        

        # trials with remote theta content
        trials = []
        dates2session_trial[day] = []
        for session in sessions:
            trials_ = np.unique(
                [int(np.unique(a[2])) for a in info_animal if a[0] == d and a[1] == session])
            trials.append(trials_)
            for t in trials_:
                dates2session_trial[day].append((session, t))
        if len(trials) > 0:
            trials = np.concatenate(trials)

        
        dates.append(day)
        remote_num_trials.append(len(trials))
        
        
        
        
    return dates, remote_num_trials, dates2session_trial


def long_theta_count_per_day(animal, list_of_days, proportion, delta_t_minus, delta_t_plus, max_flag):
    """tally trials with long theta ahead of the animal within the same arm """
    total_num_trials = []
    long_theta_trials = []
    dates2session_trial = {}
    
    for day in list_of_days:
        dates2session_trial[day] = []
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        
        q = {"proportion": proportion,
             "delta_t_minus":delta_t_minus, "delta_t_plus":delta_t_plus,
             "max_flag":max_flag}

        q["nwb_file_name"] = nwb_copy_file_name
        
        total_num_trials_day = 0
        long_theta_trials_day = 0
        for session_name in session_interval:
            q["epoch"] = session_name[:2]
            theta_df = ChangeofMindTheta().fetch1_dataframe(q)
            
            long_theta_trials_day += len(theta_df[theta_df.long_theta])
            total_num_trials_day += len(theta_df[theta_df.change_of_mind])
            
            theta_df_tmp = theta_df[theta_df.long_theta]
            for trialID in theta_df_tmp.index:
                dates2session_trial[day].append((session_name,trialID))
                
        long_theta_trials.append(long_theta_trials_day)
        total_num_trials.append(total_num_trials_day)
    
    return total_num_trials, long_theta_trials, dates2session_trial


def intersection(session_trial_long, session_trial_remote):
    """find intersection of trials"""
    session_trial_both = {}
    session_trial_union = {}
    ratio_long = {}
    ratio_remote = {}
    
    for day in session_trial_long.keys():
        
        d = day
        session_trial_both[day] = []
        session_trial_union[day] = []
        
        long = session_trial_long[d]
        remote = session_trial_remote[d]
        both = intersect_2sets(long, remote)
        either = union_2sets(long, remote, both)
        
        session_trial_both[d] = both
        session_trial_union[d] = either
        if len(long) > 0:
            ratio_long[d] = len(both) / len(long)
        else:
            ratio_long[d] = 0
        if len(remote) > 0:
            ratio_remote[d] = len(both) / len(remote)
        else:
            ratio_remote[d] = 0
        
    return session_trial_both, ratio_long, ratio_remote, session_trial_union

def intersect_2sets(A,B):
    pool = []
    for a in A:
        for b in B:
            if a[0] == b[0] and a[1] == b[1]:
                pool.append(a)
    return pool

def union_2sets(A,B,intersection):
    unique_A = []
    for a in A:
        in_intersect = False
        for i in intersection:
            if a[0] == i[0] and a[1] == i[1]:
                in_intersect = True
                break
        
        if not in_intersect:
            unique_A.append(a)
    
    unique_B = []
    for a in B:
        in_intersect = False
        for i in intersection:
            if a[0] == i[0] and a[1] == i[1]:
                in_intersect = True
                break
        
        if not in_intersect:
            unique_B.append(a)
            
    return unique_A + unique_B + intersection

def return_com_theta_length_remote_feature(animal, remote_info, long_theta_info, proportion = 0.1, correct_sequence = None):
    """
    remote_info looks like this:
    [['lewis20240105_.nwb', '02_Rev2Session1', [42, 42, 42]],
    ['lewis20240105_.nwb', '02_Rev2Session1', [50, 50]],
    ['lewis20240105_.nwb', '02_Rev2Session1', [66]],
    ['lewis20240105_.nwb', '02_Rev2Session1', [66]],
    ['lewis20240105_.nwb', '02_Rev2Session1', [70]],
    ['lewis20240105_.nwb', '02_Rev2Session1', [98, 98, 98, 98, 98]],
    ['lewis20240105_.nwb', '02_Rev2Session1', [98, 98, 98, 98, 98]],
    ['lewis20240105_.nwb', '04_Rev2Session2', [9]],
    ['lewis20240105_.nwb', '06_Rev2Session3', [22, 22]]
    
    long_theta_info looks like this:
    {'20240105': [('02_Rev2Session1', 20),
                    ('02_Rev2Session1', 25),
                    ('02_Rev2Session1', 50),
                    ('02_Rev2Session1', 62),
                    ('02_Rev2Session1', 70),
                    ('02_Rev2Session1', 71),
                    ('02_Rev2Session1', 95),
                    ('02_Rev2Session1', 98),
                    ('04_Rev2Session2', 9)}
    """
    if correct_sequence.lower() == "seq2":
        seq = seq2
    elif correct_sequence.lower() == "rev2":
        seq = rev2
    elif correct_sequence.lower() == "seq1":
        seq = seq1
    elif correct_sequence.lower() == "rev1":
        seq = rev1
    
    
    delta_correct_all = []
    info = []
    long_theta = []
    remote_theta = []
    
    for d in long_theta_info:
        nwb_copy_file_name = animal + d + '_.nwb'
        
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        
        for session_name in session_interval:
            # 1.1. 
            q = {"nwb_file_name": nwb_copy_file_name,
                "epoch":int(session_name[:2]),
                "proportion":str(proportion),
                "delta_t_minus":5,
                "delta_t_plus":5}
                    
            log_df = ChangeofMindTheta().fetch1_dataframe(q)
            
            # find trials with change of mind
            CoMtrials = log_df[log_df.change_of_mind].index
            
            # find trials with long theta
            long_theta_trials = [_[1] for _ in long_theta_info[d] if _[0] == session_name]
            
            # find trials with remote content
            remote_trials = [np.unique(_[2]) for _ in remote_info if (_[0] == nwb_copy_file_name and _[1] == session_name)]
            
            for trial in CoMtrials:
                # get correctness had the animal picked initial_choice
                past_reward = log_df.loc[trial].past_reward
                if np.isnan(past_reward):
                    continue
                else:
                    past_reward = int(past_reward)
                initial_choice = int(log_df.loc[trial].initial_choice)
                initial_correct = seq[past_reward] == initial_choice
                
                delta_correct = int((log_df.loc[trial].rewardNum == 2)) - int(initial_correct)
                is_long_theta = np.isin(trial, long_theta_trials)
                is_remote_theta = np.isin(trial, remote_trials)
                
                delta_correct_all.append(delta_correct)
                long_theta.append(is_long_theta)
                remote_theta.append(is_remote_theta)
                info.append((nwb_copy_file_name, session_name, trial))
    
    return delta_correct_all, long_theta, remote_theta, info
            
            
    
    
        
        