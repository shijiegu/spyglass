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

from spyglass.shijiegu.Analysis_SGU import TrialChoice,EpochPos,MUA,get_linearization_map,DecodeIngredients, DecodeResults2D, ChangeofMind, ChangeofMindTheta, DecodeResultsLinear

from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.ripple_add_replay import (plot_decode_spiking,
                                                 select_subset_helper,select_subset_helper_pd,
                                                 find_start_end,position_posterior2arm_posterior)
from spyglass.shijiegu.changeOfMind import find_trials_session
from spyglass.shijiegu.load import load_decode
from spyglass.shijiegu.changeOfMind_triggered import linear_map, find_triggered_session
import statsmodels.api as sm
from spyglass.shijiegu.changeOfMind_triggered import select_subset_helper_position
from spyglass.shijiegu.changeOfMind_triggered_position import load_triggered_position_decode_day
from spyglass.shijiegu.changeOfMind_triggered import wellregion, region


from spyglass.shijiegu.changeOfMind import node_positions
node_indices = [[1,8],[8,6],[6,4],[4,2],[2,1]]

def get_handedness(p_rat, p0, p1):
    """
    https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html
    # Given a line segment between P0 (x0,y0) and P1 (x1,y1),
    #    another point P (x,y) has the following relationship to the line segment.
    # Compute (y - y0)(x1 - x0) - (x - x0)(y1 - y0)
    """
    (x, y) = p_rat[:,0], p_rat[:,1]
    (x0, y0) = p0
    (x1, y1) = p1
    handedness = (y - y0) * (x1 - x0) - (x - x0) * (y1 - y0)
    return handedness < 0

def is_rat_interior(p_rat):
    handedness = 1
    for e in node_indices:
        n0, n1 = e
        p0 = node_positions[n0]
        p1 = node_positions[n1]
        handedness_ = get_handedness(p_rat, p0, p1)
        handedness = handedness * handedness_
    return handedness > 0

def find_posterior_sum_segment(triggered_position, triggered_trial_info, log,
                               position_2d, position_1d, posterior1d,
                               decode_positions,animal_positions,speed_threshold,
                               use_center = False, use_home = True, normalized = True):
    
    # find t0, t1 to consider
    trialID = triggered_trial_info[2]
    (t0, t1) = (triggered_position.index[0],triggered_position.index[-1])
    timestamp_H = log.loc[trialID,'timestamp_H']
    if not np.isnan(timestamp_H):
        t0 = timestamp_H #np.min([t0, timestamp_H])
        
    position2d_subset = position_2d[np.logical_and(position_2d.time>=t0, position_2d.time<=t1)]
    position1d_subset = position_1d[np.logical_and(position_1d.time>=t0, position_1d.time<=t1)]
    
    # Restrict to moments when the animal is moving
    if speed_threshold is not None:
        subset_ind = position2d_subset.head_speed >= speed_threshold
        position2d_subset = position2d_subset[subset_ind]
        position1d_subset = position1d_subset[subset_ind]
    
    if use_center:
        return find_arm_posterior_sum_position_2D(
            position2d_subset, posterior1d, is_rat_interior, decode_positions[:,0], decode_positions[:,1],normalized)
    
    # Restrict to moments when the animal is at home arm
    if use_home:
        subset_ind = position1d_subset.track_segment_id == 0
        position2d_subset = position2d_subset[subset_ind]
        position1d_subset = position1d_subset[subset_ind]
    
    if use_home:
        assert decode_positions.shape[0] == 1
    assert decode_positions.shape[1] == 2
    
    posterior_all = []
    num_time_bins = []
    for decode_position_ind in range(decode_positions.shape[0]):
        posterior_bin = []
        time_bin_bin = []   
        decode_p0, decode_p1 = decode_positions[decode_position_ind]
        for position_ind in range(animal_positions.shape[0]):
            p0, p1 = animal_positions[position_ind]
            posterior_bin_, num_time_bins_ = find_posterior_sum_position(position1d_subset, posterior1d, p0, p1, decode_p0, decode_p1, normalized)
            posterior_bin.append(posterior_bin_)
            time_bin_bin.append(num_time_bins_)
            
        posterior_all.append(posterior_bin)
        if decode_position_ind == 0:
            num_time_bins.append(time_bin_bin)
    
    return posterior_all, num_time_bins
    
def find_posterior_sum_position(position1d_subset, posterior1d, p0, p1, decode_p0, decode_p1, normalized = True):
    """
    find posterior sum when the animal is between position 0 and position 1,
     sum across all the decode between decode_p0 and decode_p1
    """
    
    position_ind = np.logical_and(position1d_subset.linear_position >= p0,
                                  position1d_subset.linear_position < p1)
    if np.sum(position_ind) == 0:
        return np.nan, 0
    bin_time = position1d_subset[position_ind].time #time in this spatial bin
    
    if normalized:
        posterior1d_subset = posterior1d.sel(time = np.array(bin_time)).sum("time") / len(bin_time)
    else:
        posterior1d_subset = posterior1d.sel(time = np.array(bin_time)).sum("time")

    posterior1d_subset = select_subset_helper_position(posterior1d_subset, [decode_p0, decode_p1]) #find home arm local representation
    posterior1d_bin = float(posterior1d_subset.sum("position").acausal_posterior)
    
    return posterior1d_bin, len(bin_time)

def find_arm_posterior_sum_position_2D(position_2d, posterior1d, is_rat_interior, decode_p0, decode_p1, normalized = True):
    """
    find posterior sum when the animal is in a 2D region (determined by is_rat_interior), which is a list of edges, clockwise
     sum across all the decode between decode_p0 and decode_p1
    """
    p_rat = np.hstack((np.array(position_2d.head_position_x).reshape((-1,1)),
                       np.array(position_2d.head_position_y).reshape((-1,1))))
    position_ind = is_rat_interior(p_rat)
    bin_time = position_2d[position_ind].time #time in this spatial bin
    
    if normalized:
        posterior1d_subset = posterior1d.sel(time = np.array(bin_time)).sum("time") / len(bin_time)
    else:
        posterior1d_subset = posterior1d.sel(time = np.array(bin_time)).sum("time")
        
    posterior1d_subset_sum = []
    for region_ind in range(len(decode_p0)):
        decode_p0_, decode_p1_ = decode_p0[region_ind], decode_p1[region_ind]
        posterior1d_subset_region = select_subset_helper_position(
            posterior1d_subset, [decode_p0_, decode_p1_]) #find home arm local representation
        posterior1d_subset_sum.append(float(posterior1d_subset_region.sum("position").acausal_posterior))
    
    return posterior1d_subset_sum, len(bin_time)


def find_remote_theta_animal_new(animal,list_of_days,classifier_param_name,encoding_set,
                                 control = False,
                                 use_1d = True, use_center = False, use_outer = False, use_home = True, use_all_outers = False,
                                 proportion = 0.05,
                                 delta_t_minus = 5,delta_t_plus = 0, segment_only = False,
                                 speed_threshold = None,
                                 normalized = True,
                                 max_flag = False):
    """
    use_1d: if True, use 1D decoding. if False, use 1D decoding collapsed from 2D decoding
    use_center: if True, consider moments when the rat in the center platform, and find decodes that in are outer arms
    use_outer: if True, consider moments when the rat is in outer arms, and find decodes that are in the same outer arms
    use_all_outers: if True, consider moments when the rat is in one outer arm, and find decodes that are in all outer arms.
    """
    # define home bins
    linear_map,node_location=get_linearization_map()
    if use_center: 
        decode_positions = linear_map[6:10]
        animal_positions = None #will use isinterior function instead in the find_posterior_sum_segment()
    elif use_home:
        home_end_1D = linear_map[0][1] #home end location in cm
        decode_positions = np.array([0,home_end_1D]).reshape((1,-1))
        animal_positions_ = np.linspace(10,home_end_1D,10) #10 bins
        animal_positions = np.array(
            [[animal_positions_[ind],animal_positions_[ind+1]] for ind in range(len(animal_positions_)-1)]
            ).reshape((-1,2))
        

    # alternatively
    # home_positions = results1d.position[results1d.position <= home_end_1D]
    
    posterior_home_all = []
    triggered_trial_info_all = []
    bin_nums_all = []

    for day in list_of_days:
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        
        paramters = {"proportion": proportion,
                         "delta_t_minus":delta_t_minus, "delta_t_plus":delta_t_plus,
                         "max_flag":max_flag, "segment_only": segment_only,
                         "control": control, "multiple_CoM": 0, "single_CoM": 1, "first_CoM": 0
                         }
        loaded_data = load_triggered_position_decode_day(animal, day, encoding_set, classifier_param_name,
                                            **paramters)

        triggered_positions, triggered_trial_infos = (
                    loaded_data["triggered_positions_baseoff"],
                    loaded_data["triggered_trial_info"],
            )
            
        for ind in range(len(session_interval)):
            
            session_name = session_interval[ind]
            position_name = position_interval[ind]
            epoch_num = int(session_name[:2])
            
            event_indices_session = [ind for ind in range(len(triggered_trial_infos)) if triggered_trial_infos[ind][1]==session_name]
            if len(event_indices_session) == 0:
                continue
            
            # log
            key={'nwb_file_name':nwb_copy_file_name,'epoch':epoch_num,'proportion': proportion}
            print(ChangeofMind & key)
            log = ChangeofMind().fetch1_dataframe(key)
            session_name = (TrialChoice & key).fetch1('epoch_name')
            
            entry = DecodeIngredients & {'nwb_file_name':nwb_copy_file_name,
                             'interval_list_name':session_name}
            # position_1d,position_2d,
            position_1d = pd.read_csv(entry.fetch1('position_1d')) #still need 1D position
            position_2d = pd.read_csv(entry.fetch1('position_2d')) # need 2D position

            # load decode
            results1d = load_decode(nwb_copy_file_name,
                                    session_name,
                                    classifier_param_name = classifier_param_name,
                                    encoding_set = encoding_set,
                                    use_1d = use_1d)
            posterior1d = results1d.sum("state")
            
            event_indices_session = [ind for ind in range(len(triggered_trial_infos)) if triggered_trial_infos[ind][1]==session_name]
            #event_indices_session = [ind for ind in range(len(triggered_trial_infos)) if triggered_trial_infos[ind][1]==session_name and triggered_trial_infos[ind][0]==day]
            
            for event_index in event_indices_session:
                print(f"working on event_index{event_index}")
                # get arm
                if use_outer and not use_all_outers:
                    arm = triggered_trial_infos[event_index][-1]
                    arm_start_end = region[arm + 5]
                    decode_positions = arm_start_end.reshape((1,-1))
                    
                    animal_positions_ = np.linspace(arm_start_end[0],arm_start_end[-1],10) #15 bins
                    animal_positions = np.array(
                        [[animal_positions_[ind],animal_positions_[ind+1]] for ind in range(len(animal_positions_)-1)]
                        ).reshape((-1,2))
        
                
                if use_all_outers:
                    arm = triggered_trial_infos[event_index][-1]
                    arm_start_end = region[arm + 5]
                    animal_positions = arm_start_end.reshape((1,-1))
                    
                    all_arms = [1,2,3,4]#[a for a in [1,2,3,4] if a != arm]
                    decode_positions = []
                    for a in all_arms:
                        arm_start_end = wellregion[a + 5]
                        decode_positions = decode_positions + list(arm_start_end)
                    decode_positions = np.array(decode_positions).reshape((-1,2))
                
                posterior_home, num_time_bins = find_posterior_sum_segment(triggered_positions[event_index], triggered_trial_infos[event_index],
                                                        log,
                                                        position_2d, position_1d,
                                                        posterior1d, decode_positions, animal_positions, speed_threshold, use_center, use_home, normalized)

                posterior_home_all.append(posterior_home)
                triggered_trial_info_all.append(triggered_trial_infos[event_index])
                bin_nums_all.append(num_time_bins)

    if use_home:
        return posterior_home_all, animal_positions, bin_nums_all
    elif use_outer and not use_all_outers:
        return posterior_home_all, animal_positions - animal_positions[0], bin_nums_all
    return posterior_home_all, triggered_trial_info_all, bin_nums_all
            
                    
  
def classify_remote_chosen_theta_animal(animal,list_of_days,posterior_by_arm, trial_info, proportion = 0.1, normalized = True):
    
    # for each trial, find the final chosen arm
    chosen_arms = []
    initial_arms = []
    event_indices = []

    for day in list_of_days:
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
                
        for ind in range(len(session_interval)):
            session_name = session_interval[ind]
            position_name = position_interval[ind]
            epoch_num = int(session_name[:2])
                
            # log
            key={'nwb_file_name':nwb_copy_file_name,'epoch':epoch_num,'proportion': proportion}
            print(ChangeofMind & key)
            log = ChangeofMind().fetch1_dataframe(key)
            session_name = (TrialChoice & key).fetch1('epoch_name')

            event_indices_session = [ind for ind in range(len(trial_info)) if trial_info[ind][1]==session_name and trial_info[ind][0][5:13]==day]
            for event_index in event_indices_session:
                print(f"working on event_index{event_index}")
                
                trialID = trial_info[event_index][2]
                initial_choice = int(log.loc[trialID,'initial_choice'])
                # cross check
                assert initial_choice== int(trial_info[event_index][-1])
                
                # do not consider trials in which the final choice is the same as the initial choice
                final_choice = int(log.loc[trialID,'OuterWellIndex'])
                if final_choice == initial_choice:
                    continue
                
                # do not consider trials with more than 1 change of mind
                if log.loc[trialID,'CoMNum_by_arm'] > 1:
                    continue
                
                event_indices.append(event_index)
                initial_arms.append(initial_choice)
                chosen_arms.append(final_choice)
    
    # for each remote event, classify whether it is to the chosen arm
    posterior_chosen = []
    posterior_unchosen = []
    
    for trial_ind in range(len(chosen_arms)):
        # zero out initial arm
        initial_arm = initial_arms[trial_ind] - 1
        posterior_others = posterior_by_arm[event_indices[trial_ind],:]
        
        # re-normalize the rest
        if normalized:
            posterior_others[initial_arm] = 0
            posterior_others = posterior_others/np.sum(posterior_others)
        
        posterior_chosen.append(posterior_others[chosen_arms[trial_ind]-1])
        
        ind_not_chosen = np.setdiff1d([1,2,3,4],[chosen_arms[trial_ind], initial_arms[trial_ind]])-1
        
        posterior_unchosen.append(posterior_others[ind_not_chosen])
                
    return posterior_chosen, posterior_unchosen, event_indices
            
    
                    
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

def do_GLM_subprocess(GLM_entries1, constant = True, simple_linear = False):
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
    if simple_linear:
        glm_poisson1 = sm.GLM(y, x_, family = sm.families.Gaussian())
    else:
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
    