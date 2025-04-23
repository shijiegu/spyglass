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

from spyglass.shijiegu.Analysis_SGU import TrialChoice,EpochPos,MUA,get_linearization_map,TrialChoiceChangeofMind
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.ripple_add_replay import plot_decode_spiking,select_subset_helper,select_subset_helper_pd
from spyglass.shijiegu.changeOfMind import (find_turnaround_time, findProportion,
            find_trials, normalize, load_epoch_data_wrapper, find_direction, find_trials_animal)
from spyglass.shijiegu.changeOfMindRipple import triggered_ripple_animal
from spyglass.shijiegu.load import load_decode
from spyglass.shijiegu.pairwiseDecode import behavior_transitions_count

def session_long_theta_trials(replay_trials_tuple,nwb_copy_file_name,session_name,log_df,type = 1):
    """given all replay_trials_tuples from this animal,
    find in this session, transition tuple
        type1: last trial - would have been
        type2: would have been - this trial in reality
    """
    trials = []
    for tup in replay_trials_tuple:
        if tup[0] == nwb_copy_file_name and tup[1] == session_name:
            would_have_been = tup[3]
            if type == 1:
                trials.append([tup[2],(int(log_df.loc[tup[2]-1].OuterWellIndex), would_have_been)])
            else:
                trials.append([tup[2],(would_have_been, int(log_df.loc[tup[2]].OuterWellIndex))])
    return trials

def find_transition_interval(log_df,transition):
    # find time between transitions
    trials = []

    for trial in log_df.index[log_df.OuterWellIndex == transition[0]]:
        j = log_df.loc[trial+1].OuterWellIndex
        if j == transition[1]:
            trials.append(trial)
    return trials

def find_theta_transition(trials_long_theta,transition):
    trials_theta_transition = []
    for tup in trials_long_theta:
        if tup[1][0] == transition[0] and tup[1][1] == transition[1]:
            trials_theta_transition.append(tup[0])
    return np.unique(trials_theta_transition)
    
#def would_have_been(t, log):
#    log.loc[t - 1].OuterWellIndex, log.loc[t - 1].OuterWellIndex

def find_diff_transition(log_df,transition,trials_long_theta,trials_short_theta):
    trials_all_transition = find_transition_interval(log_df,transition)
    trials_theta_transition = find_theta_transition(trials_long_theta,transition)
    trials_short_theta_transition = find_theta_transition(trials_short_theta,transition)

    diff_theta = [] # pile 1
    diff_nontheta = [] # pile 2
    diff_theta_trialID = []
    diff_nontheta_trialID = []
    
    for t in range(1,len(trials_all_transition)):
        
        i = trials_all_transition[t - 1]
        j = trials_all_transition[t]
    
        if np.any(np.logical_and(trials_theta_transition >= i, trials_theta_transition <= j)):
            # there is theta replay in between
            
            #print("i,j",i,j)
            #print(trials_theta_transition[np.logical_and(trials_theta_transition >= i, trials_theta_transition <= j)])
            i_all = trials_theta_transition[np.logical_and(trials_theta_transition >= i, trials_theta_transition <= j)]
            for i_ in i_all:
                #diff_theta.append(j - i_)
                diff_theta.append(j - i_)
                diff_theta_trialID.append(i_)

        if np.any(np.logical_and(trials_short_theta_transition >= i,
                                   trials_short_theta_transition <= j)):
            i_all = trials_short_theta_transition[np.logical_and(trials_short_theta_transition >= i,
                                                        trials_short_theta_transition <= j)]
            for i_ in i_all:
                diff_nontheta.append(j - i_)
                diff_nontheta_trialID.append(i_) 
            
    return diff_theta, diff_nontheta, diff_theta_trialID, diff_nontheta_trialID
    
def intersect_rows(arr1, arr2):
    """
    Finds the intersection of rows between two NumPy arrays.

    Args:
        arr1 (np.ndarray): The first array.
        arr2 (np.ndarray): The second array.

    Returns:
        np.ndarray: A new array containing the intersection of rows.
    """
    return np.array([row for row in arr1 if any(np.array_equal(row, other_row) for other_row in arr2)])
    
def find_delta_t(nwb_copy_file_name, session_name,
                 proportion,
                 replay_trials, replay_trials_non, paired, type = 1, glm = False):
    """_summary_

    Args:
        nwb_copy_file_name (_type_): _description_
        session_name (_type_): _description_
        replay_trials (dict):
            with long theta
            replay_trials_non["lewis"]['20240105'] = [
                ('lewis20240105_.nwb', '02_Rev2Session1', 101, 3),
                ('lewis20240105_.nwb', '04_Rev2Session2', 5, 1),]
        replay_trials_non (dict):
            same as replay_trials, without long theta
        type (int):
            if int == 1: return would have been transition 1
            if int == 2: return would have been transition 2
        glm (bool):
            if glm == 1: add trial number
            if glm == 0: 
            

    Returns:
        _type_: _description_
    """
    
    animal = nwb_copy_file_name[:5]
    d = nwb_copy_file_name[5:13]
    print(nwb_copy_file_name,session_name)

    
    # find all transitions in which long theta sequence happens
    
    log_df = pd.read_pickle( (TrialChoiceChangeofMind() & {"nwb_file_name": nwb_copy_file_name,
                                                            "epoch":int(session_name[:2]),
                                                            "proportion":str(proportion)}).fetch1("change_of_mind_info") )
    
    trials_long_theta = session_long_theta_trials(replay_trials[animal][d],
                                       nwb_copy_file_name,
                                       session_name,log_df, type = type)
    trials_short_theta = session_long_theta_trials(replay_trials_non[animal][d],
                                       nwb_copy_file_name,
                                       session_name,log_df, type = type)
    print("trials_long_theta",trials_long_theta)
    print("trials_short_theta",trials_short_theta)
    
    # find transitions that showed up in both "trials_long_theta" and "trials_short_theta"
    transitions_long = []
    for tup in trials_long_theta:
        transitions_long.append(tup[1])
    transitions_long = np.unique(transitions_long,axis=0)    
    
    transitions_sh = []
    for tup in trials_short_theta:
        transitions_sh.append(tup[1])
    transitions_sh = np.unique(transitions_sh,axis=0)   
    
    transitions = intersect_rows(transitions_long, transitions_sh)   
    print("transitions",transitions)
    
    diff_theta_transition = {}
    diff_nontheta_transition = {}
    #if not paired:
    #    transitions = []
    #    for tup in trials_long_theta:
    #        transitions.append(tup[1])
    #    for tup in trials_short_theta:
    #        transitions.append(tup[1])
    #    transitions = np.unique(transitions,axis=0)   
        
    for transition in transitions:
        diff_theta_all = []
        diff_nontheta_all = []
    
        diff_theta, diff_nontheta, diff_theta_trial, diff_nontheta_trial = find_diff_transition(
            log_df,transition,trials_long_theta,trials_short_theta)
        
        if glm:
            diff_theta_all, diff_nontheta_all = parse_trials_theta_glm(diff_theta, diff_nontheta,
                                                               diff_theta_trial, diff_nontheta_trial)
        else:
            diff_theta_all, diff_nontheta_all = parse_trials_theta(diff_theta, diff_nontheta,
                                                               diff_theta_trial, diff_nontheta_trial,paired)
        
        diff_theta_transition[tuple(transition)] = diff_theta_all
        diff_nontheta_transition[tuple(transition)] = diff_nontheta_all
            
    return diff_theta_transition, diff_nontheta_transition

def parse_trials_theta_glm(diff_theta, diff_nontheta, diff_theta_trial, diff_nontheta_trial):
    # will not need paired parsing
    diff_theta_all = []
    diff_nontheta_all = []
    
    for ind in range(len(diff_theta_trial)):
        theta_trialID = diff_theta_trial[ind] # trialID
        diff_theta_all.append((diff_theta[ind],theta_trialID))
        
    for ind in range(len(diff_nontheta_trial)):
        theta_trialID = diff_nontheta_trial[ind] # trialID
        diff_nontheta_all.append((diff_nontheta[ind],theta_trialID))
    return diff_theta_all, diff_nontheta_all
    

def parse_trials_theta(diff_theta, diff_nontheta, diff_theta_trial, diff_nontheta_trial,paired):
    # parser workhorse of find_delta_t by finding pairs of theta, nontheta data
    # process the results "from find_diff_transition"
    # find the index of diff_theta
    diff_theta_all = []
    diff_nontheta_all = []
    
    for ind in range(len(diff_theta_trial)):
    
        theta_trialID = diff_theta_trial[ind] # trialID
        if not paired:
            diff_theta_all.append(diff_theta[ind])
            continue
              
        # if paired:   
        # find the most recent nontheta ind
        last_ind = np.argwhere(np.array(diff_nontheta_trial) < theta_trialID).ravel()
                    
        if len(last_ind) > 0:
            diff_theta_all.append(diff_theta[ind])
            diff_nontheta_all.append(diff_nontheta[last_ind[-1]])
                        
        # find the first nontheta ind
        first_ind = np.argwhere(np.array(diff_nontheta_trial) > theta_trialID).ravel()       
        if len(first_ind) > 0:
            diff_theta_all.append(diff_theta[ind])
            diff_nontheta_all.append(diff_nontheta[first_ind[0]])
        
    if not paired:
        if len(diff_nontheta) > 0:
        # find the index of diff_theta
            for ind in range(len(diff_nontheta_trial)):
                diff_nontheta_all.append(diff_nontheta[ind])
        
    return diff_theta_all, diff_nontheta_all
    