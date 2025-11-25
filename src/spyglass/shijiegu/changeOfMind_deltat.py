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

from spyglass.shijiegu.Analysis_SGU import TrialChoice,EpochPos,MUA,get_linearization_map,ChangeofMind
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.ripple_add_replay import plot_decode_spiking,select_subset_helper,select_subset_helper_pd
from spyglass.shijiegu.load import load_decode
from spyglass.shijiegu.pairwiseDecode import behavior_transitions_count
from spyglass.shijiegu.Analysis_SGU import ChangeofMindTheta

def return_trial_deltat(t0, t1):
    t0_index = np.digitize(t1,t0) #bins[i-1] <= x < bins[i]
    t1 = t1[t0_index>=1]
    t0_index = t0_index[t0_index>=1]
    
    t0_index_unique, indices = np.unique(t0_index, return_index=True)

    t0 = t0[t0_index_unique-1]
    t1 = t1[indices]
    
    min_len = np.min([len(t1),len(t0)])
    t1 = t1[:min_len]
    t0 = t0[:min_len]
    

    return t0, t1, t1 - t0

def test():
    """test return_trial_deltat()"""
    # tests 1: t1[0]<t0[0]
    t0 = np.array([7,10,12])
    t1 = np.array([2,3,8])
    t0_, t1_, delta_t = return_trial_deltat(t0, t1)
    assert (t0_ == np.array(7)).all()
    assert (t1_ == np.array(8)).all()
    assert (delta_t == np.array(1)).all()

    # tests 2: t1[0]>t0[0], with extra at the end
    t0 = np.array([7,10,12])
    t1 = np.array([8,13,15])
    t0_, t1_, delta_t = return_trial_deltat(t0, t1)
    assert (t0_ == np.array([7,12])).all()
    assert (t1_ == np.array([8,13])).all()
    assert (delta_t == np.array([1,1])).all()

    # tests 3: blend of tests 1,2
    t0 = np.array([7,10,12])
    t1 = np.array([6,11,15,16])
    t0_, t1_, delta_t = return_trial_deltat(t0, t1)
    assert (t0_ == np.array([10,12])).all()
    assert (t1_ == np.array([11,15])).all()
    assert (delta_t == np.array([1,3])).all()

    # tests 4: extra in between t0s
    t0 = np.array([7,10,15,19])
    t1 = np.array([6,11,12,17])
    t0_, t1_, delta_t = return_trial_deltat(t0, t1)
    assert (t0_ == np.array([10,15])).all()
    assert (t1_ == np.array([11,17])).all()
    assert (delta_t == np.array([1,2])).all()

    # tests 5: extra in between t0s
    t0 = np.array([7,10,15])
    t1 = np.array([6,11,12,17,20])
    t0_, t1_, delta_t = return_trial_deltat(t0, t1)
    assert (t0_ == np.array([10,15])).all()
    assert (t1_ == np.array([11,17])).all()
    assert (delta_t == np.array([1,2])).all()

def count_by_day(animal, days):
    """
    return for all days, 4 matrices as listed below:
    com_final_count / com_initial_count / theta_count / behavior_count
    """

    com_final_count = {d:np.zeros((4,4)) for d in days}     # change of mind behavior/final choice arm x (arm y) arm z, where we count arm x -> arm z
    com_initial_count = {d:np.zeros((4,4)) for d in days}   # change of mind behavior/initial choice arm x (arm y) arm z, where we count arm x -> arm y
    com_theta_count = {d:np.zeros((4,4)) for d in days}         # change of mind behavior/initial choice arm x (arm y) arm z, where we count arm x -> arm y with theta
    behavior_count = {d:np.zeros((4,4)) for d in days}      # all behavior
    
    com_final_N = {d:0 for d in days}     # change of mind behavior/final choice arm x (arm y) arm z, where we count arm x -> arm z
    com_initial_N = {d:0 for d in days}   # change of mind behavior/initial choice arm x (arm y) arm z, where we count arm x -> arm y
    com_theta_N = {d:0 for d in days}         # change of mind behavior/initial choice arm x (arm y) arm z, where we count arm x -> arm y with theta
    behavior_N = {d:0 for d in days}      # all behavior
            
    transitions = [(1,2),(1,3),(1,4),
                   (2,1),(2,3),(2,4),
                   (3,1),(3,2),(3,4),
                   (4,1),(4,2),(4,3),
                   ]
    for d in days:
        nwb_file_name = animal.lower() + d + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        total_trials = 0
        for session_name in session_interval:
            q = {"nwb_file_name": nwb_copy_file_name,
                    "epoch":session_name[:2],
                    "proportion": 0.1,
                    "delta_t_minus":5, "delta_t_plus":5,
                    "max_flag":1}
                
            theta_df = ChangeofMindTheta().fetch1_dataframe(q)
            
            
            for transition in transitions:
                t0, t1 = find_exact_transition(theta_df, transition)
                a1, a2 = transition
                a1 -= 1 #zero index
                a2 -= 1 #zero index
                behavior_count[d][a1][a2] += len(t0)
                
                t0, t1 = find_exact_transition_com(theta_df, transition)
                com_initial_count[d][a1][a2] += len(t0)
                
                t0, t1 = find_exact_transition_com_final(theta_df, transition)
                com_final_count[d][a1][a2] += len(t0)
                
                t0, t1 = find_exact_transition_com_theta(theta_df, transition)
                com_theta_count[d][a1][a2] += len(t0)
    
    # normalization
    for d in days:
        com_initial_N[d] = np.sum(com_initial_count[d])
        com_final_N[d] = np.sum(com_final_count[d])
        com_theta_N[d] = np.sum(com_theta_count[d])
        behavior_N[d] = np.sum(behavior_count[d])
        
    for d in days:
        com_initial_count[d] = com_initial_count[d] / np.sum(com_initial_count[d], axis = 1).reshape((-1,1))
        com_final_count[d] = com_final_count[d] / np.sum(com_final_count[d], axis = 1).reshape((-1,1))
        com_theta_count[d] = com_theta_count[d] / np.sum(com_theta_count[d], axis = 1).reshape((-1,1))
        behavior_count[d] = behavior_count[d] / np.sum(behavior_count[d])
        
                
    return com_initial_count, com_final_count, com_theta_count, behavior_count, com_initial_N, com_final_N, com_theta_N, behavior_N
    
def delta_by_day(animal, days, initial_transitions, final_transitions):
    """return"""
    # return for each transition in transition_final, transition_initial
    # for each day
    #   the number of change of mind trials
    # for each session
    #   the number of transition of behavior on that day

    com_count = {}
    theta_count = {}
    behavior_count = {}
    transitions = initial_transitions + final_transitions #list concatenation
    for transition in transitions:
        com_count[transition] = {d:0 for d in days}
        theta_count[transition] = {d:0 for d in days}
        behavior_count[transition] = {d:0 for d in days}
            
        
    for d in days:
        nwb_file_name = animal.lower() + d + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        total_trials = 0
        for session_name in session_interval:
            q = {"nwb_file_name": nwb_copy_file_name,
                    "epoch":session_name[:2],
                    "proportion": 0.1,
                    "delta_t_minus":5, "delta_t_plus":5,
                    "max_flag":1}
                
            theta_df = ChangeofMindTheta().fetch1_dataframe(q)
            total_trials += len(theta_df) - 2
            for transition in final_transitions:
                t0, t1 = find_exact_transition(theta_df, transition)
                behavior_count[transition][d] += len(t0)
                com_count[transition][d] += np.sum(theta_df.loc[t1].change_of_mind)
                theta_count[transition][d] += np.sum(theta_df.loc[t1].long_theta)
            
            for transition in initial_transitions:
                t0, t1 = find_exact_transition(theta_df, transition)
                behavior_count[transition][d] += len(t0)
                
                # a bit different
                t0, t1 = find_exact_transition_com(theta_df, transition)
                com_count[transition][d] += len(t0)
                theta_count[transition][d] += np.sum(theta_df.loc[t1].long_theta)
        
        for transition in transitions:
            behavior_count[transition][d] = behavior_count[transition][d]/total_trials
            com_count[transition][d] = com_count[transition][d]/total_trials
            theta_count[transition][d] = theta_count[transition][d]/total_trials
                
    return behavior_count, com_count, theta_count

def long_theta_delta_t(animal, dates2session_trial, transition_final, transition_initial):
    session_name_old = None

    pre_post_control, pre_post_com, pre_post_theta = {}, {}, {}
    pre_post_control_others, pre_post_com_others, pre_post_theta_others = {}, {}, {}

    for transition in transition_final:
        pre_post_control[transition] = []
        pre_post_com[transition] = []
        pre_post_theta[transition] = []
        for transition_other in transition_initial[transition]:
            pre_post_control_others[transition_other] = []
            pre_post_com_others[transition_other] = []
            pre_post_theta_others[transition_other] = []

    for d in dates2session_trial.keys():
        nwb_copy_file_name = animal + d + '_.nwb'
        info_day = dates2session_trial[d]
        if len(info_day) == 0:
            continue

        for info in info_day:
            session_name, trialID = info
                
            # 1. Get decode and animal head direction
            if session_name != session_name_old:
                # new session
                    
                session_name_old = session_name
                
                q = {"nwb_file_name": nwb_copy_file_name,
                    "epoch":session_name[:2],
                    "proportion": 0.1,
                    "delta_t_minus":5, "delta_t_plus":5,
                    "max_flag":1}
                
                theta_df = ChangeofMindTheta().fetch1_dataframe(q)
                #trials = find_transition_interval(theta_df, transition)

            for transition in transition_final:
                t0, t1 = find_exact_transition(theta_df, transition)
                
                ### Pre-post comparison
                for ind in range(1,len(t0)-1):
                    t_current = t0[ind]
                    t_previous = t0[ind - 1]
                    t_next = t0[ind + 1]
                    
                    delta_pre = t_current - t_previous
                    delta_post = t_next - t_current

                    com = theta_df.loc[t_current + 1].change_of_mind
                    theta = np.array(theta_df.loc[t_current + 1].long_theta)
                    if com:
                        pre_post_com[transition].append((delta_pre,delta_post))
                        if theta:
                            pre_post_theta[transition].append((delta_pre,delta_post))
                    else:
                        pre_post_control[transition].append((delta_pre,delta_post))


                # find other transitions
                for transition_other in transition_initial[transition]:
                    t0_o, t1_o = find_exact_transition_including_com(theta_df, transition_other)
        
                    for ind in range(1,len(t0_o)-1):
                        t_current = t0_o[ind]
                        t_previous = t0_o[ind - 1]
                        t_next = t0_o[ind + 1]
                        
                        delta_pre = t_current - t_previous
                        delta_post = t_next - t_current
        
                        com = theta_df.loc[t_current + 1].change_of_mind
                        theta = np.array(theta_df.loc[t_current + 1].long_theta)
                        if com:
                            pre_post_com_others[transition_other].append((delta_pre,delta_post))
                            if theta:
                                pre_post_theta_others[transition_other].append((delta_pre,delta_post))
                        else:
                            pre_post_control_others[transition_other].append((delta_pre,delta_post))


                
                """ # old code
                change_of_mind = trials_t1.loc[t1].change_of_mind
                long_theta = trials_t1.loc[t1].theta_dev

                ### GLM
                y = delta_t[1:]#-delta_t[:-1]
                x1 = long_theta[:-1]
                x2 = change_of_mind[:-1]
                y_all.append(y)
                x1_all.append(x1)
                x2_all.append(x2)


                ### Pre-post comparison
                for ind in range(1,len(delta_t)-1):
                    delta_pre = delta_t[ind-1]
                    com = trials_t1.loc[t1[ind]].change_of_mind
                    theta = np.abs(np.array(trials_t1.loc[t1[ind]].theta_dev)) > 0
                    
                    delta_post = delta_t[ind+1]
                    #if delta_t[ind] != 1:
                    #    continue

                    if com:
                        pre_post_com.append((delta_pre,delta_post))
                        if theta:
                            pre_post_theta.append((delta_pre,delta_post))
                    else:
                        pre_post_control.append((delta_pre,delta_post))
                """
    return pre_post_control, pre_post_com, pre_post_theta, pre_post_control_others, pre_post_com_others, pre_post_theta_others 
            
def find_exact_transition(log_df, transition):
    trials_t0 = log_df[log_df.OuterWellIndex == transition[0]]
    trials_t1 = log_df[log_df.OuterWellIndex == transition[1]]
    t0, t1, delta_t = return_trial_deltat(trials_t0.index, trials_t1.index)
    t0 = t0[delta_t == 1]
    t1 = t1[delta_t == 1]

    return t0, t1

def find_exact_transition_com(log_df, transition):
    # find number of transition[0] -> transition[1], where transition[1] is the initial change of mind
    trials_t0 = log_df[log_df.OuterWellIndex == transition[0]]
    trials_t1 = log_df[log_df.initial_choice == transition[1]]
    t0, t1, delta_t = return_trial_deltat(trials_t0.index, trials_t1.index)
    t0 = t0[delta_t == 1]
    t1 = t1[delta_t == 1]
    return t0, t1

def find_exact_transition_com_final(log_df, transition):
    # find number of transition[0] -> transition[1], where transition[1] is the final choice on change of mind trials
    trials_t0 = log_df[log_df.OuterWellIndex == transition[0]]
    subset = log_df[log_df.OuterWellIndex == transition[1]]
    trials_t1 = subset[subset.change_of_mind]
    t0, t1, delta_t = return_trial_deltat(trials_t0.index, trials_t1.index)
    t0 = t0[delta_t == 1]
    t1 = t1[delta_t == 1]
    return t0, t1

def find_exact_transition_com_theta(log_df, transition):
    # find number of transition[0] -> transition[1], where transition[1] is the final choice on change of mind trials
    trials_t0 = log_df[log_df.OuterWellIndex == transition[0]]
    subset = log_df[log_df.initial_choice == transition[1]]
    trials_t1 = subset[subset.long_theta]
    t0, t1, delta_t = return_trial_deltat(trials_t0.index, trials_t1.index)
    t0 = t0[delta_t == 1]
    t1 = t1[delta_t == 1]
    return t0, t1
    
    
    
def find_exact_transition_including_com(log_df, transition):
    # include initial Change of Mind transitions
    trials_t0 = log_df[log_df.OuterWellIndex == transition[0]]
    trials_t1 = np.array(log_df[log_df.OuterWellIndex == transition[1]].index)
    trials_t1_com = np.array(log_df[log_df.initial_choice == transition[1]].index)
    if len(trials_t1_com) > 0:
        trials_t1_index = np.sort(np.concatenate((trials_t1,trials_t1_com)))
    else:
        trials_t1_index = trials_t1
    
    
    t0, t1, delta_t = return_trial_deltat(trials_t0.index, trials_t1_index)
    t0 = t0[delta_t == 1]
    t1 = t1[delta_t == 1]

    return t0, t1

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
    