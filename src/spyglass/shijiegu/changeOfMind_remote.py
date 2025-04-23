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

from spyglass.shijiegu.Analysis_SGU import TrialChoice,EpochPos,MUA,get_linearization_map
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.ripple_add_replay import (plot_decode_spiking,
                                                 select_subset_helper,select_subset_helper_pd,
                                                 find_start_end,position_posterior2arm_posterior)
from spyglass.shijiegu.changeOfMind import (find_turnaround_time, findProportion,
            find_trials, find_trials_session, load_epoch_data_wrapper, find_direction, find_trials_animal)
from spyglass.shijiegu.load import load_decode
from spyglass.shijiegu.changeOfMind_triggered import region, linear_map, find_triggered_session
import statsmodels.api as sm

def find_remote_theta_animal(animal,list_of_days,
                             proportion = 0.05,
                             delta_t_minus = 1,delta_t_plus = 3,
                             max_flag = False,
                             nearby = False, # use nearby trial's outbound or inbound
                             home = False # find remote replay when the animal is walking out of home arm
                             ):
    (day_session_animal, trials_animal, time_intervals_animal, arm_identities_animal, triggered_trial_info_animal) = (
        [],[],[],[],[])
    
    for day in list_of_days:
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        for ind in range(len(session_interval)):
            session_name = session_interval[ind]
            position_name = position_interval[ind]
            (trials_session,
             time_intervals_session,
             arm_identity_session, triggered_trial_info) = find_remote_theta_session(
                 nwb_copy_file_name,session_name,position_name,proportion,
                 delta_t_minus,delta_t_plus,
                 max_flag,nearby,home)
            
            day_session_animal.append([nwb_copy_file_name,session_name])
            trials_animal.append(trials_session)
            time_intervals_animal.append(time_intervals_session)
            arm_identities_animal.append(arm_identity_session)
            triggered_trial_info_animal.append(triggered_trial_info)
            
    return day_session_animal,trials_animal,time_intervals_animal,arm_identities_animal,triggered_trial_info_animal
            
def find_remote_theta_session(nwb_copy_file_name,session_name,position_name,proportion,
                              delta_t_minus,delta_t_plus,
                              max_flag,nearby,home):
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
    
    key={'nwb_file_name':nwb_copy_file_name,'epoch':int(session_name[:2])}
    log=(TrialChoice & key).fetch1('choice_reward')
    log_df=pd.DataFrame(log)
    
    decode_options = {}
    if animal.lower() == "eliot":
        decode_options["encoding_set"] = '2Dheadspeed_above_4_andlowmua'
        decode_options["classifier_param_name"] = 'default_decoding_gpu_4armMaze'
        decode_options["decode_threshold_method"] = 'MUA_0SD'
        decode_options["causal"] = False
    else:
        decode_options["encoding_set"] = '2Dheadspeed_above_4'
        decode_options["classifier_param_name"] = 'default_decoding_gpu_4armMaze'
        decode_options["decode_threshold_method"] = 'MUA_M05SD'
        decode_options["causal"] = False
    
    decode = load_decode(nwb_copy_file_name,session_name,decode_options["classifier_param_name"],
                         decode_options["encoding_set"])
    
    (triggered_positions, triggered_positions_abs,
     triggered_decodes, triggered_decodes_baseoff,
     triggered_decodes_abs, triggered_trial_info) = find_triggered_session(
         nwb_copy_file_name,session_name,position_name,
         delta_t_minus,delta_t_plus,max_flag,
         proportion = proportion,first_CoM= True, segment_only = False, nearby = nearby)
     
    trials_session = []
    time_intervals_session = []
    arm_identity_session =[]
    
    for trial_ind in range(len(triggered_positions)):
        triggered_position = triggered_positions[trial_ind]
        triggered_position_abs = triggered_positions_abs[trial_ind]
        
        (trials, time_intervals, arm_identity) = turnaround_triggered_remote_decode(
            triggered_position,triggered_position_abs,decode,log_df,
            linear_position_info,position_info,max_flag,home)
    
        if len(trials) == 0:
            continue
        
        if home and len(trials_session) > 0:
            if trials_session[-1][-1] == trials[-1]:
                # because for each trial there can be multiple change of minds,
                # but we only look at the the home remote,
                # so this "continue" essentially just make sure we have at most interval on each trial
                continue
        trials_session.append(trials)
        time_intervals_session.append(time_intervals)
        arm_identity_session.append(arm_identity)
        
    # in return, we include, triggered_trial_info, which is all the snippets remote representation finder inspected.
    return trials_session,time_intervals_session,arm_identity_session,triggered_trial_info
        
def turnaround_triggered_remote_decode_home(triggered_position,triggered_position_abs,
                                decode,log_df,linear_position_info,max_flag = 1):
    """
    This function is similar to turnaround_triggered_remote_decode(),
    but only for parsing remote content at home
    # 1. find time points out side of arm position
    # 2. for each time interval, find arm
    #   decode should pass certain criteria:
    #   (a) be continuous in decoder state
    #   (b) posterior >= threshold%
    # 3. return for each trial a list of time range and arm identity for the decode

    INPUT: decode should be the absolute
    """
    # segment only: consider posterior in that arm only
    
    position_axis = np.array(decode.coords['position'])
        
    # infer the turn around time
    t = np.array(triggered_position_abs.index)[np.argmin(np.abs(triggered_position.index-0))]

    # find the arm the animal is at: select subset of decode
    (t0,t1) = (triggered_position_abs.index[0]-0.0001,triggered_position_abs.index[-1]+0.0001)
    # add a little for floating point
    decode_subset = select_subset_helper(decode,(t0,t1))  
    
    posterior_position_subset = decode_subset.causal_posterior.sum(dim='state')
    
    # find the arm the animal is at
    (t0_peak,t1_peak) = (t-0.001, t+0.001)
    subset_ind = (linear_position_info.index >= t0_peak) & (linear_position_info.index <= t1_peak)
    subset_linear = linear_position_info.loc[subset_ind]
    subset_arm = np.unique(subset_linear.track_segment_id)
    
    # get the arm bound
    (arm_base, arm_top) = region[int(subset_arm)]
    
    # get max posterior
    if max_flag:
        max_posterior_position = np.array(position_axis[posterior_position_subset.argmax(dim = 'position')])

    # get mean posterior
    else:
        posterior_position_subset_array = np.array(posterior_position_subset).T
        posterior_position_subset_array = posterior_position_subset_array/np.sum(posterior_position_subset_array, axis = 0)
        max_posterior_position = np.matmul(position_axis,posterior_position_subset_array)

    # find remote time
    is_remote = np.logical_or(max_posterior_position > arm_top, max_posterior_position < arm_base)
    is_remote_pd = pd.Series(is_remote, index = posterior_position_subset.time)
    is_remote_segments = np.array(segment_boolean_series(
            is_remote_pd, minimum_duration=0.020))
    if len(is_remote_segments) == 0:
        return [],[],[]

    
    time_intervals = []
    arm_identity = []
    trials = []
    for i in range(is_remote_segments.shape[0]):
        
        (t0,t1) = is_remote_segments[i]
        
        # restrict to continuous state:
        #    length of the continuous state should be greater than 20ms
        decode_subset = select_subset_helper(decode,(t0,t1))
        state_subset = np.array(decode_subset.causal_posterior.sum(dim='position'))
        time=np.array(decode_subset.causal_posterior.time)
        
        snippets_conti = find_start_end(state_subset[:,0]>0.5) #continuous
        snippets=[time[s] for s in snippets_conti if np.diff(time[s])[0]>0.020]
        
        #assert 1 == 0
        for s in range(len(snippets)):
            # overall sum of decode posterior in the max posterior arm should be greater than 0.2
            posterior_by_arm = position_posterior2arm_posterior(
                select_subset_helper(posterior_position_subset,snippets[s]),
                linear_map)
            
            # the max posterior arm
            (t0_peak,t1_peak) = snippets[s]
            subset_ind = (posterior_position_subset.time >= t0_peak) & (posterior_position_subset.time <= t1_peak)
            subset_arm_snippet = linear2arm(max_posterior_position[subset_ind])
            # if a decode goes to home arm, it will not classified
            subset_arm_snippet = subset_arm_snippet[~np.isnan(subset_arm_snippet)]
            # exclude
            subset_arm_snippet = np.setdiff1d(subset_arm_snippet,subset_arm)
            
            if len(subset_arm_snippet) == 0 or len(subset_arm_snippet) > 1:
                # the latter scenario is unclear situation, ignore
                continue

            max_arm_ind = int(np.unique(subset_arm_snippet) - 5)
    
            if np.mean(posterior_by_arm[max_arm_ind,:]) < 0.2:
                continue
            
            time_intervals.append(snippets[s])
            arm_identity.append(max_arm_ind)
            trials.append(add_trial(t0_peak,log_df))
    return trials, time_intervals, arm_identity

def turnaround_triggered_remote_decode(triggered_position,triggered_position_abs,
                                decode,log_df,linear_position_info,position_info,max_flag = 1,home = False,
                                minimum_duration = 0.05): # in seconds
    """
    if home = 1: find remote representation at home arm during running instead of at outer well.
    
    # 1. find time points out side of arm position
    # 2. for each time interval, find arm
    #   decode should pass certain criteria:
    #   (a) be continuous in decoder state
    #   (b) posterior >= threshold%
    # 3. return for each trial a list of time range and arm identity for the decode

    INPUT: decode should be the absolute
    """
    # segment only: consider posterior in that arm only
    
    position_axis = np.array(decode.coords['position'])
        
    # infer the turn around time
    t = np.array(triggered_position_abs.index)[np.argmin(np.abs(triggered_position.index-0))]
    
    # find the arm the animal is at: select subset of decode
    (t0,t1) = (triggered_position_abs.index[0]-0.0001,triggered_position_abs.index[-1]+0.0001)
    # add a little for floating point
        
    # find the arm the animal is at
    (t0_peak,t1_peak) = (t-0.001, t+0.001)
    subset_ind = (linear_position_info.index >= t0_peak) & (linear_position_info.index <= t1_peak)
    subset_linear = linear_position_info.loc[subset_ind]
    subset_arm = np.unique(subset_linear.track_segment_id)

    if home:
        # find the trial
        trialID = add_trial(t,log_df)
        t_home = log_df.loc[trialID,"timestamp_H"]

        if np.isnan(t_home):
            return [],[],[]
        
        (t0,t1) = (t_home,t)
        # rough set
        linear_pos_subset = select_subset_helper_pd(linear_position_info,(t0,t1))
        
        # stricter
        t1 = linear_pos_subset[np.array(linear_pos_subset.track_segment_id) >= 6].index[0]
        linear_pos_subset = select_subset_helper_pd(linear_position_info,(t0,t1))
        
        
    decode_subset = select_subset_helper(decode,(t0,t1))      
    posterior_position_subset = decode_subset.acausal_posterior.sum(dim='state')
    
    # get max posterior
    if max_flag:
        max_posterior_position = np.array(position_axis[posterior_position_subset.argmax(dim = 'position')])

    # get mean posterior
    else:
        posterior_position_subset_array = np.array(posterior_position_subset).T
        posterior_position_subset_array = posterior_position_subset_array/np.sum(posterior_position_subset_array, axis = 0)
        max_posterior_position = np.matmul(position_axis,posterior_position_subset_array)

    # find remote time
    
    
    if home:
        
        if abs(len(linear_pos_subset) - len(max_posterior_position)) > 3:
            print("skipped due to decode and camera time frame do not fully match.")
            return [],[],[]
        
        #animal is physically at the non arm segment
        #is_physically_at = np.array(linear_pos_subset.track_segment_id) < 6 # this simple criteria might include home well replay events
        is_physically_at = np.logical_and(np.array(linear_pos_subset.linear_position) >= 10,
                                          np.array(linear_pos_subset.linear_position) <= linear_map[1][1])
                
        is_remote = np.zeros_like(max_posterior_position)
        #just to initialize, it calculates whether CA1 decoded representation is remote
        

        min_len = np.min([len(is_physically_at),len(is_remote)])
        is_physically_at = is_physically_at[:min_len]
        is_remote = is_remote[:min_len]
            
        for k in region.keys():
            if k == int(subset_arm):
                continue
            (arm_base, arm_top) = region[k]
            is_remote = is_remote + np.logical_and(max_posterior_position <= arm_top, max_posterior_position >= arm_base)
        is_remote = np.logical_and(is_physically_at,is_remote)
    else:
        is_remote = np.zeros_like(max_posterior_position) #just to initialize
        # find representation in other arms
        for k in region.keys():
            if k == int(subset_arm):
                continue
            (arm_base, arm_top) = region[k]
            is_remote = is_remote + np.logical_and(max_posterior_position <= arm_top, max_posterior_position >= arm_base)
        #is_remote = np.logical_or(max_posterior_position > arm_top, max_posterior_position < arm_base)
    
    
    # restrict to moving time
    pos_subset = select_subset_helper_pd(position_info,(t0,t1))  
    is_moving = np.array(pos_subset.head_speed) > 4
    min_len = np.min([len(is_moving),len(is_remote)])
    # choose min because one variable is a subset of decode and the other is a subset of position.
    # there could be 1 or 2 time point difference.
    is_moving = is_moving[:min_len]
    is_remote = is_remote[:min_len]
    
    is_remote = np.logical_and(is_remote, is_moving)
    
    is_remote_pd = pd.Series(is_remote, index = posterior_position_subset.time)
    is_remote_segments = np.array(segment_boolean_series(
            is_remote_pd, minimum_duration=minimum_duration))
        
    if len(is_remote_segments) == 0:
        return [],[],[]
    
    for i in range(is_remote_segments.shape[0]):
        
        (t0,t1) = is_remote_segments[i]

    time_intervals = []
    arm_identity = []
    trials = []
    for i in range(is_remote_segments.shape[0]):
        
        (t0,t1) = is_remote_segments[i]
        
        # restrict to continuous state:
        #    length of the continuous state should be greater than 15ms
        decode_subset = select_subset_helper(decode,(t0,t1))
        state_subset = np.array(decode_subset.causal_posterior.sum(dim='position'))
        time=np.array(decode_subset.causal_posterior.time)
        
        snippets_conti = find_start_end(state_subset[:,0]>0.5) #continuous
        snippets=[time[s] for s in snippets_conti if np.diff(time[s])[0]>0.02]
        
        for s in range(len(snippets)):

            # overall sum of decode posterior in the max posterior arm should be greater than 0.2
            posterior_by_arm = position_posterior2arm_posterior(
                select_subset_helper(posterior_position_subset,snippets[s]),
                linear_map)
            
            # the max posterior arm
            (t0_peak,t1_peak) = snippets[s]
            subset_ind = (posterior_position_subset.time >= t0_peak) & (posterior_position_subset.time <= t1_peak)
            subset_arm_snippet = linear2arm(max_posterior_position[subset_ind])
            
            # if a decode goes to home arm, it will not classified
            subset_arm_snippet = np.unique(subset_arm_snippet[~np.isnan(subset_arm_snippet)])
            
            # exclude
            #subset_arm_snippet = np.setdiff1d(subset_arm_snippet,subset_arm)
            
            if len(subset_arm_snippet) == 0 or len(subset_arm_snippet) > 1:
                # the latter scenario is unclear situation, ignore
                continue

            max_arm_ind = int(np.unique(subset_arm_snippet) - 5)
    
            if np.mean(posterior_by_arm[max_arm_ind,:]) < 0.2:
                continue
            
            time_intervals.append(snippets[s])
            arm_identity.append(max_arm_ind)
            trials.append(add_trial(t0_peak,log_df))
    return trials, time_intervals, arm_identity
            
def linear2arm(position):
    arm = np.zeros_like(position) + np.nan
    for p_ind in range(len(position)):
        p = position[p_ind]
        for k in region.keys():
            if p>=region[k][0] and p<region[k][1]:
                arm[p_ind] = k
                continue
    return arm

def add_trial(t0,log_df):
    trial_ind=np.array(log_df.index)
    trial_number = trial_ind[np.argwhere((np.array(log_df.timestamp_O[:-1])-t0) > 0).ravel()[0]]
    return trial_number
    
                    
def do_GLM(animal, day_sessions, trials, arm_identities, time_intervals):
    """work with output from find_remote_theta_animal()
    The 3 GLMS are 
        - predictors are imminent choice, past_reward, past; response is in theta.
        - predictors are in theta; response is in choice.
        - predictors are in theta; response is in future visits on this trial after.
    """
    
    # make GLM entry
    day_session_animal = day_sessions[animal]
    trials_animal = trials[animal]
    arm_identities_animal = arm_identities[animal]
    time_intervals_animal = time_intervals[animal]

    GLM_entries1 = [] # the last column is response; predictors are imminent choice, past_reward, past; response is in theta.
    GLM_entries2 = [] # the last column is response; predictors are in theta; response is in choice.
    GLM_entries3 = []

    for day_session_ind in range(len(day_session_animal)):

        # just get data
        (nwb_copy_file_name,session_name) = day_session_animal[day_session_ind]
        position_name = (EpochPos() & {"nwb_file_name": nwb_copy_file_name, 
               "epoch_name":session_name}).fetch1("position_interval")
        
        trials_session = trials_animal[day_session_ind]
        arms_session = arm_identities_animal[day_session_ind]
        time_session = time_intervals_animal[day_session_ind]
        
        linear_position_info=(IntervalLinearizedPosition() & {
            'nwb_file_name':nwb_copy_file_name,
            'interval_list_name':position_name,
            'position_info_param_name':'default_decoding'}).fetch1_dataframe()
        
        rowID, _, proportions, turnaround_times = find_trials_session(
            nwb_copy_file_name,session_name,position_name,return_all = True)
        
        if len(trials_session) == 0:
            continue
        
        # uniqu-ify entries
        trial_arm = np.unique(np.hstack((
                np.concatenate(trials_session).reshape((-1,1)),
                np.concatenate(arms_session).reshape((-1,1)))), axis = 0)
        

        # load stateScript: for final choice
        key={'nwb_file_name':nwb_copy_file_name,'epoch':int(session_name[:2])}
        log=(TrialChoice & key).fetch1('choice_reward')
        log_df=pd.DataFrame(log)

        # for each trial, put together x and y for GLM
        # model 1:
        trials_involved = np.unique(trial_arm[:,0])
        for t in trials_involved:
            a = trial_arm[trial_arm[:,0] == t,1]
            print('a',a)
            
            imminent = int(log_df.loc[t,"OuterWellIndex"])
            past_reward = int(log_df.loc[t,"past_reward"])
            past = int(log_df.loc[t,"past"])

            for a_ in [1,2,3,4]:
                GLM_entries1.append((int(imminent == a_), int(past_reward == a_), int(past == a_), int(np.isin(a_,a))))

        # model 2:
        trials_involved = np.unique(trial_arm[:,0])
        for t in trials_involved:
            a = trial_arm[trial_arm[:,0] == t,1]
            if len(a) == 0:
                continue
            for a_ in [1,2,3,4]:
                GLM_entries2.append((int(np.isin(a_,a)), int(imminent == a_)))
                
        # model 3:
        # first unique-fy per arm segment
        time_session_flatten = np.concatenate(time_session)
        seg_session = np.array([time2seg(interval[0],interval[1],linear_position_info)
                       for interval in time_session_flatten])
        trial_arm_seg, ind = np.unique(np.hstack((
                np.concatenate(trials_session).reshape((-1,1)),
                np.concatenate(arms_session).reshape((-1,1)),
                seg_session.reshape((-1,1))
                )), axis = 0, return_index=True)
        # trial_arm_seg is n x 3 matrix where each row is a unique combination 
        #   of trial x replayed arm x animal physical arm seg
        content_t0s = time_session_flatten[ind]
        # content_t0s are the replay time intervals, of size n x 2
        
        # for each change of mind turn around with remote content
        rowID = np.array(rowID)
        for t in trials_involved:
            ind = np.argwhere(rowID == t).ravel()[0]
            turnaround_times_t = turnaround_times[ind]
            turnaround_times_t.append(log_df.loc[t,"timestamp_O"])
            for turn_ind in range(len(turnaround_times_t)-1):
                # find remote content between turnarounds
                # t0 t1 mark the turn around behavior time
                (t0, t1) = (turnaround_times_t[turn_ind],turnaround_times_t[turn_ind+1])
                
                # replayed arm
                contents = []
                for content_ind in range(trial_arm_seg.shape[0]):
                    # check this replay occured in time interval
                    if content_t0s[content_ind][0] >= t0 and content_t0s[content_ind][1] < t1:
                        # if so, add arm to the content list
                        contents.append(trial_arm_seg[content_ind,1])
                contents = np.array(contents)
                if len(contents) == 0:
                    continue
                
                # add to GLM entries
                future = find_future_arms(t, t0, log_df,
                                          linear_position_info)
                if len(future) == 0:
                    continue
                for a_ in [1,2,3,4]:
                    GLM_entries3.append((int(np.isin(a_,contents)), int(np.isin(a_,future))))
        #assert 1 == 0
            
            

    GLM_entries1 = np.vstack(GLM_entries1)
    GLM_entries2 = np.vstack(GLM_entries2)
    GLM_entries3 = np.vstack(GLM_entries3)
    
    # do GLM
    model1 = do_GLM_subprocess(GLM_entries1)
    model2 = do_GLM_subprocess(GLM_entries2)
    model3 = do_GLM_subprocess(GLM_entries3, constant = False)
    
    return model1, model2, model3

def do_GLM_subprocess(GLM_entries1, constant = True):
    model1 = {}
    x = GLM_entries1[:,:-1]
    if constant:
        x_ = sm.add_constant(x)
    else:
        x_ = x
    y = GLM_entries1[:,-1]
    model1['x'] = x_
    model1['y'] = y

    #glm_poisson1 = sm.GLM(y,x_,family=sm.families.Poisson())
    glm_poisson1 = sm.GLM(y, x_, family = sm.families.Binomial())
    res1 = glm_poisson1.fit()
    model1['fit'] = res1
    model1['CI'] = res1.conf_int(alpha=0.05)
    model1['model'] = glm_poisson1
    
    return model1

def time2seg(t0,t1,linear_position_info):
    """translate time to the track segment the animal is on"""
    # restrict to this trial's position info
    trialInd = (linear_position_info.index >= t0) &(linear_position_info.index <= t1)
    trialPosInfo = linear_position_info.loc[trialInd,:]
        
    trialSeg = unique(np.array(trialPosInfo.track_segment_id))
    
    return trialSeg[0]

def find_future_arms(trialID, t0, log_df,
                     linear_position_info,
                     proportion_threshold = 0.05):
    """
    Find outer arms the rat visited after this current outer arm visit.

    log_df is behavior parsing
    linear_position_info is frame-by-frame position
    position_info is frame-by-frame 2d position
    """
    
    # for each trial
    start = t0
    end = log_df.loc[trialID,'timestamp_O']

    # restrict to this trial's position info
    
    # find the segment the animal is on
    seg_initial = find_track_seg(start, start + 0.2, linear_position_info)
    seg_last = find_track_seg(end - 0.2, end + 0.2, linear_position_info)

    # in the case the rat went back to the initial segement
    seg_all = find_track_seg(start, end, linear_position_info)

    trialSeg = np.setdiff1d(seg_all,seg_initial) # exclude the segment the animal is on now

    if seg_initial == seg_last:
        trialSeg = list(trialSeg)
        trialSeg.append(seg_last[-1])
        trialSeg = np.array(trialSeg)
    
    return trialSeg

def find_track_seg(t0,t1,linear_position_info):
    trialInd = (linear_position_info.index >= t0) & (linear_position_info.index <= t1)
    trialPosInfo = linear_position_info.loc[trialInd,:]
    trialSeg = unique(np.array(trialPosInfo.track_segment_id)) - 5
    trialSeg = trialSeg[trialSeg > 0]
    return trialSeg

def unique(arr):
    # stable unique rather than sorted unique
    _, idx = np.unique(arr, return_index=True)
    unique_stable = arr[np.sort(idx)]
    return unique_stable
    