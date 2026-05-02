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
    DecodeIngredients,DecodeResultsLinear,MUA,DecodeIngredientsLikelihood,DecodeResultsLinear)
from spyglass.shijiegu.changeOfMind_figures.supp_decode import return_diff_file
from spyglass.linearization.v0.main import IntervalLinearizedPosition
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.shijiegu.decodeHelpers import session2position_name, runSessionNames

from spyglass.shijiegu.changeOfMind_triggered_position import (
    load_triggered_position_decode_day, load_triggered_position_decode_session_spyglass)
from spyglass.shijiegu.ripple_add_replay import select_subset_helper_pd, select_subset_helper
from spyglass.shijiegu.changeOfMind_helper import nodes
from spyglass.shijiegu.ripple_add_replay import position_posterior2arm_posterior
from spyglass.shijiegu.changeOfMind_triggered import linear_map

def return_concentration_animal(animal, days):
    nonlocal_concentration_animal = {}
    for d in days:
        print(f"Processing {animal} day: {d}")
        nonlocal_concentration_day = return_concentration_day(animal, d)

        for threshold in nonlocal_concentration_day.keys():
            if threshold not in nonlocal_concentration_animal:
                nonlocal_concentration_animal[threshold] = nonlocal_concentration_day[threshold]
            else:
                nonlocal_concentration_animal[threshold] = np.concatenate((nonlocal_concentration_animal[threshold],
                                                                            nonlocal_concentration_day[threshold]))
    return nonlocal_concentration_animal

def return_concentration_day(animal, d):
    nwb_copy_file_name = animal + d + '_.nwb'
    animal = nwb_copy_file_name[:5]
    session_names, _ = runSessionNames(nwb_copy_file_name)
    
    encoding_set = '2Dheadspeed_above_4'
    classifier_param_name = 'default_decoding_gpu_4armMaze'
        
    # load change of mind triggered position decode on this day!
    # paramters = {"proportion":0.1,
    #                 "delta_t_minus":5,
    #                 "delta_t_plus":5,
    #                 "max_flag":1,
    #                 "segment_only":False,
    #                 "multiple_CoM":True, "single_CoM":True, "first_CoM":False
    #                 }
    parameter_name = "params_both_max_run_time_2_state"

    # # load dataset
    # loaded_data = load_triggered_position_decode_day(animal, d, encoding_set, classifier_param_name,
    #                                                 control = False,
    #                                                 **paramters)
    # triggered_trial_info = loaded_data["triggered_trial_info"]

    local_concentrations = []
    nonlocal_concentrations = []
    for session_name in session_names:
        print(f"Processing {animal} {d} session: {session_name}")
        
        epoch_num = int(session_name[:2])
        # load triggered position info for this session
        loaded_data = load_triggered_position_decode_session_spyglass(
            nwb_copy_file_name, epoch_num, parameter_name, proportion = 0.1,
                                                    )
        if len(loaded_data) == 0:
            continue
        # load decode
        decode_path = (DecodeResultsLinear & {"nwb_file_name":nwb_copy_file_name,
                                "interval_list_name":session_name,
                                "encoding_set":encoding_set,
                                "classifier_param_name":classifier_param_name}).fetch1("posterior")
        decode = xr.open_dataset(decode_path)
            
        ## load LinearPosition
        pos1d = pd.read_csv((DecodeIngredients & {"nwb_file_name":nwb_copy_file_name,
                                        "interval_list_name":session_name}).fetch1("position_1d"))
            
        pos2d = pd.read_csv((DecodeIngredients & {"nwb_file_name":nwb_copy_file_name,
                                        "interval_list_name":session_name}).fetch1("position_2d"))

        triggered_positions = loaded_data["triggered_positions_baseoff"]
        
        for triggered_position in triggered_positions:
            local_concentration_session, nonlocal_concentration_session = return_concentration_session(
                triggered_position, pos1d, pos2d, decode)
            if local_concentration_session is None or nonlocal_concentration_session is None:
                continue
            local_concentrations.append(local_concentration_session)
            nonlocal_concentrations.append(nonlocal_concentration_session)
       
    nonlocal_concentration_day = {}     
    #local_concentration_day = np.hstack(local_concentrations).ravel()
    for nonlocal_concentration in nonlocal_concentrations:
        for threshold in nonlocal_concentration.keys():
            if threshold not in nonlocal_concentration_day:
                nonlocal_concentration_day[threshold] = nonlocal_concentration[threshold]
            else:
                nonlocal_concentration_day[threshold] = np.concatenate((nonlocal_concentration_day[threshold],
                                                                        nonlocal_concentration[threshold]))
        
    return nonlocal_concentration_day
        
def return_concentration_session(triggered_position, pos1d, pos2d, decode):
    t0t1 = [triggered_position.index[0], triggered_position.index[-1]]
    
    subset_ind = (pos1d.time >= t0t1[0]) & (pos1d.time <= t0t1[1])
    pos1d_subset = pos1d.loc[subset_ind]
    pos2d_subset = pos2d.loc[subset_ind]

    # a) speed thresholding
    ind = pos2d_subset.head_speed >= 4
    pos1d_subset = pos1d_subset[ind]
    pos2d_subset = pos2d_subset[ind]

    # b) animals in outer arms only
    ind = pos1d_subset.track_segment_id >= 5
    pos1d_subset = pos1d_subset[ind]
    pos2d_subset = pos2d_subset[ind]
    
    if len(pos1d_subset) == 0:
        return None, None
    
    # c) when animal in outer arm > proportion 0.1
    track_segment_node_start = np.array([nodes[i][0] for i in np.array(pos1d_subset.track_segment_id)])
    track_segment_node_end = np.array([nodes[i][1] for i in np.array(pos1d_subset.track_segment_id)])

    projected_xy = np.hstack((np.array(pos1d_subset.projected_x_position).reshape((-1,1)),
                                            np.array(pos1d_subset.projected_y_position).reshape((-1,1))))
        
    full_length = linalg.norm(track_segment_node_start - track_segment_node_end, axis = 1)
    partial_length = linalg.norm(track_segment_node_start - projected_xy, axis = 1)
    proportion = partial_length / full_length

    ind = proportion >= 0.1
    pos1d_subset = pos1d_subset[ind]
    pos2d_subset = pos2d_subset[ind]
    
    if len(pos1d_subset) == 0:
        return None, None
    
    # get decode
    decode_subset = decode.isel(time = pos2d_subset.index)
    posterior_position_subset = decode_subset.causal_posterior.sum(dim='state')
    
    # map posterior over location to posterior over arm 
    posterior_by_arm = position_posterior2arm_posterior(posterior_position_subset,linear_map)
    arm_id = np.array(pos1d_subset.track_segment_id - 5).astype("int")
    
    # get local posterior
    posterior_local = np.array([posterior_by_arm[arm_id[t_ind], t_ind] for t_ind in range(len(arm_id))])
    
    # get nonlocal posterior by thresholding low local posterior
    threshold = 0.5
    time_ind = np.argwhere(posterior_local <= threshold).ravel()
    
    thresholds_nonlocal = [0.1, 0.2, 0.3]

    posteriors_nonlocal = {}
    for threshold_nonlocal in thresholds_nonlocal:
        posteriors_nonlocal[threshold_nonlocal] = np.array(
            [np.sum(posterior_by_arm[:,t_ind][np.setdiff1d([0,1,2,3,4],arm_id[t_ind])] >= threshold_nonlocal) for t_ind in time_ind]
        )

    return posterior_local, posteriors_nonlocal
        
