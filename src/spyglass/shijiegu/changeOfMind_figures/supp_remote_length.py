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

def return_concentration_session(nwb_copy_file_name, session_name):
    """return the length of remote intervals in seconds for this session"""
    
    # find remote intervals
    pandas = (ChangeofMindRemoteTheta() & {"nwb_file_name":nwb_copy_file_name,
                                           "proportion":0.1,
                                            "delta_t_minus":5,
                                            "delta_t_plus":5,
                                            "epoch":str(session_name[:2])}).fetch1("pandas")
    log_df = pd.DataFrame(pandas)
    log_df =log_df[log_df.has_remote_interval]
    
    remote_lengths = []
    delta_ts_same = [] # delta t between remote representations of the same arm
    delta_ts_diff = [] # delta t between remote representations of different arms
    delta_ts = [] # delta t between any two remote representations
    for trialID in log_df.index:
        remote_intervals = log_df.loc[trialID,'remote_interval']
        remote_arms = log_df.loc[trialID,'remote_content']
        
        for remote_interval in remote_intervals:
            t0t1 = remote_interval[0], remote_interval[1]
            remote_lengths.append(t0t1[1]-t0t1[0])
        
        delta_t_same, delta_t_diff, delta_t = compute_delta_t_remote_intervals(remote_intervals, remote_arms)
        delta_ts_same.extend(delta_t_same)
        delta_ts_diff.extend(delta_t_diff)
        delta_ts.extend(delta_t)
    
    return remote_lengths, delta_ts_same, delta_ts_diff, delta_ts

def compute_delta_t_remote_intervals(remote_intervals, remote_arms):
    """compute delta t between remote intervals"""
    delta_ts_same = []
    delta_ts_diff = []
    delta_ts = []
    for i in range(len(remote_intervals)):
        j = i+1
        if j >= len(remote_intervals):
            break
        t0t1_i = remote_intervals[i][0], remote_intervals[i][1]
        t0t1_j = remote_intervals[j][0], remote_intervals[j][1]
        center_i = (t0t1_i[0] + t0t1_i[1]) / 2
        center_j = (t0t1_j[0] + t0t1_j[1]) / 2
        delta_t = abs(center_i - center_j)
        delta_ts.append(delta_t)
        if remote_arms[i] == remote_arms[j]:
            delta_ts_same.append(delta_t)
        else:
            delta_ts_diff.append(delta_t)
    return delta_ts_same, delta_ts_diff, delta_ts

def return_concentration_day(animal, days):
    remote_length_all_trials = []
    delta_ts_same_all_trials = []
    delta_ts_diff_all_trials = []
    delta_ts_all_trials = []
    
    for d in days:
        nwb_copy_file_name = animal + d + '_.nwb'
        animal = nwb_copy_file_name[:5]
        session_names, _ = runSessionNames(nwb_copy_file_name)
        
        for session_name in session_names:
            print(f"Processing {animal} {d} session: {session_name} for remote length")
            (remote_length_session,
             delta_ts_same, delta_ts_diff, delta_ts) = return_concentration_session(
                nwb_copy_file_name, session_name)
             
            remote_length_all_trials.extend(remote_length_session)
            delta_ts_same_all_trials.extend(delta_ts_same)
            delta_ts_diff_all_trials.extend(delta_ts_diff)
            delta_ts_all_trials.extend(delta_ts)
            
    return remote_length_all_trials, delta_ts_same_all_trials, delta_ts_diff_all_trials, delta_ts_all_trials
    