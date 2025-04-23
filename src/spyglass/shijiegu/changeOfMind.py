import spyglass as nd
import pandas as pd
import numpy as np
import xarray as xr
from scipy import stats
from scipy import linalg
from scipy import ndimage
import os
import matplotlib.pyplot as plt
from spyglass.common import (Session, IntervalList,LabMember, LabTeam, Raw, Session, Nwbfile,
                            Electrode,LFPBand,interval_list_intersect)
from spyglass.common import TaskEpoch
from spyglass.spikesorting.v0 import (SortGroup, Curation,
                                    SpikeSortingRecording,SpikeSortingRecordingSelection)
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.common.common_position import IntervalPositionInfo, RawPosition, IntervalLinearizedPosition, TrackGraph

from spyglass.shijiegu.Analysis_SGU import TrialChoice,EpochPos,MUA,get_linearization_map, TrialChoiceChangeofMind
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.ripple_add_replay import plot_decode_spiking, plot_decode_sortedSpikes
from spyglass.shijiegu.load import load_epoch_data
from ripple_detection.core import segment_boolean_series
from spyglass.shijiegu.singleUnit import get_nwb_units
from spyglass.shijiegu.singleUnit_sortedDecode import place_field_direction, color_cells_by_place_direction


# in the linearized track, segment 0 correspond to home, 1 to platform etc.
labels={}
labels[0]='home'
labels[1]='platform'
labels[6]='arm 1'
labels[7]='arm 2'
labels[8]='arm 3'
labels[9]='arm 4'


graph = TrackGraph() & {'track_graph_name': '4 arm lumped 2023'}
node_positions = graph.fetch1("node_positions")
#linear_map,node_location=get_linearization_map()
nodes={}
nodes[6] = (node_positions[2],node_positions[3])
nodes[7] = (node_positions[4],node_positions[5])
nodes[8] = (node_positions[6],node_positions[7])
nodes[9] = (node_positions[8],node_positions[9])

vectors = {}
for key in nodes.keys():
    vector = nodes[key][1] - nodes[key][0]
    vectors[key] = vector/np.linalg.norm(vector)
    
def find_statescrripts(animal,list_of_days):
    logs_days = {}
    for day in list_of_days:
        logs = []
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        
        # load stateScript
        for session_name in session_interval:
            key={'nwb_file_name':nwb_copy_file_name,'epoch_name':session_name}
            log=(TrialChoice & key).fetch1('choice_reward')
            log_df=pd.DataFrame(log)
            logs.append(log_df)
        logs_days[day] = logs
    return logs_days
    
def find_trials_animal(animal,list_of_days,plot = False,proportion_threshold = 0.2):
    trials_days = {}
    for day in list_of_days:
        trials = []
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        for ind in range(len(session_interval)):
            session_name = session_interval[ind]
            position_name = position_interval[ind]
            if plot:
                trials.append(find_trials_session_plot(nwb_copy_file_name,session_name,position_name,proportion_threshold=proportion_threshold))
            else:
                trials.append(find_trials_session(nwb_copy_file_name,
                                                  session_name,position_name,
                                                  proportion_threshold=proportion_threshold,return_all = True))
        trials_days[day] = trials
        
    return trials_days

def insertTrialChoiceChangeOfMind(trials_days, proportion_threshold):
    for date in trials_days.keys():
        for session_ind in range(len(trials_days[date])):
            trials_info = trials_days[date][session_ind]
            (nwb_file_name_copy, session_name)= trials_days[date][session_ind][4]
    
            key={'nwb_file_name':nwb_file_name_copy,
                 'epoch_name':session_name}
            
            log=(TrialChoice & key).fetch1('choice_reward')
            epoch_num = (TrialChoice & key).fetch1('epoch')
            log_df=pd.DataFrame(log)
            log_df2 = log_df.copy()
            
            # initialization
            log_df2.insert(5,'change_of_mind',[False for i in range(len(log_df))])
            #hold boolean to indicate whether a trial is a change of mind
            log_df2.insert(6,'CoMMaxProportion',[[] for i in range(len(log_df))])
            #hold max proportion of traversed arms, in the case of multiple change of mind, this is the max of all the arms
            log_df2.insert(7,'CoM_t',[[] for i in range(len(log_df))])
            # change of mind time, there can be multiple times
            log_df2.insert(8,'CoM_arm',[[] for i in range(len(log_df))])
            # change of mind arm, there can be multiple arms
    
            # fill the table
            trials = trials_info[0]
            
            if len(trials) == 0:
                print("No change of mind on session " + session_name)
                
            else:
                max_proportion = np.nanmax(trials_info[2], axis = 1)
                turn_around_t = trials_info[3] # there are can be multiple turn arounds per trial!
                for trialID_ind in range(len(trials)):
                    trialID = trials[trialID_ind]
                    log_df2.loc[trialID,'change_of_mind'] = True
                    log_df2.loc[trialID,'CoMMaxProportion'] = max_proportion[trialID_ind]
                    log_df2.loc[trialID,'CoM_t'].append(turn_around_t[trialID_ind])
                    turn_around_arms = list(unique_stable(trials_info[1][trialID_ind] - 5).astype("int"))
                    log_df2.loc[trialID,'CoM_arm'].append(turn_around_arms)
                
            animal = nwb_file_name_copy[:5]
            savePath = os.path.join(f'/cumulus/shijie/recording_pilot/{animal}/decoding',
                            nwb_file_name_copy+'_'+session_name + str(proportion_threshold) + '_changeofMindlog.pkl')
            log_df2.to_pickle(savePath)
    
            # insert
            key = {"nwb_file_name":nwb_file_name_copy,
                   "epoch":epoch_num,
                   "proportion":str(proportion_threshold),
                   "change_of_mind_info":savePath}
            TrialChoiceChangeofMind().insert1(key, replace = True)



def find_trials_session_plot(nwb_copy_file_name,session_name,position_name,
                             proportion_threshold = 0.1, sorted = False, curation_id = 1, decode_options = {}):
    # in addition to finding trials with change of mind, this function also plots decode data
    # curation_id is only for sorted data
    # 
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

    camera_frequency = 1/stats.mode(np.diff(linear_position_info.index))[0]
    
    # 2. load stateScript
    key={'nwb_file_name':nwb_copy_file_name,'epoch':int(session_name[:2])}
    log=(TrialChoice & key).fetch1('choice_reward')
    log_df=pd.DataFrame(log)
    
    rowID, trials, proportions, turnaround_times = find_trials(log_df,
                                                               linear_position_info, position_info, proportion_threshold = proportion_threshold)
    
    # 3. load data
    if len(decode_options.keys()) == 0:
        if animal.lower() == "eliot":
            decode_options["encoding_set"] = '2Dheadspeed_above_4_andlowmua'
            decode_options["classifier_param_name"] = 'default_decoding_gpu_4armMaze'
            decode_options["decode_threshold_method"] = 'MUA_0SD'
            decode_options["causal"] = False
            decode_options["likelihood"] = False
        else:
            decode_options["encoding_set"] = '2Dheadspeed_above_4'
            decode_options["classifier_param_name"] = 'default_decoding_gpu_4armMaze'
            decode_options["decode_threshold_method"] = 'MUA_M05SD'
            decode_options["causal"] = False
            decode_options["likelihood"] = False
    
    (_,decode,head_speed,head_orientation,
            linear_position_df,lfp_df,theta_df,ripple_df,neural_df,mua_xr,mua_threshold,spikeColInd) = load_epoch_data_wrapper(
                nwb_copy_file_name, session_name, position_name, decode_options)
            
    output_folder = f'/cumulus/shijie/recording_pilot/{animal}/changeOfMind'
    
    # 3.5 for sorted spikes only:
    if sorted:
        
        #sort_group_ids = list(nwb_units_all.keys())
    
        (cells, smoothed_placefield, placefield_peak,
            spike_count_by_arm_direction, time_spent_by_arm_direction, betaPdfs, means) = place_field_direction(nwb_copy_file_name,
                                                                                   session_name,position_name,
                                                                                   curation_id = curation_id)
    
        print(f"This session has {len(cells)} neurons.")
    
        # make colorlist for cells 
        cell_color = color_cells_by_place_direction(cells, placefield_peak, spike_count_by_arm_direction)
    
    # 4. do plotting
    for t in rowID:
        plottimes = findPlottingStartEnd(t,log_df,linear_position_info)
        turnaround_t = findTurnAround_t(t,rowID, turnaround_times)
        arm_direction_t, arm_direction, _ = findDirectionPlot(t,log_df,linear_position_info,position_info)
        head_direction_sign = pd.Series(arm_direction, index = arm_direction_t)
        
        if sorted:
            filename = animal+'_'+nwb_copy_file_name+'_'+session_name+'_trial'+str(t)+'sortedSpiked'
            plot_decode_sortedSpikes(nwb_copy_file_name,session_name,
                         plottimes,[],linear_position_df,decode,lfp_df,theta_df,
                         neural_df,placefield_peak,head_speed,head_orientation,
                         cell_color = cell_color,
                         ripple_consensus_trace = None,
                         title='',savefolder = output_folder,savename = filename,
                         likelihood = decode_options["likelihood"],causal = decode_options["causal"],
                         replay_type_time = None, replay_type = None, curation_id = curation_id,
                         plot_changeofmind = True, turnaround = turnaround_t, head_direction_sign = head_direction_sign)

            
        else:
            filename = animal+'_'+nwb_copy_file_name+'_'+session_name+'_trial'+str(t)
            plot_decode_spiking(plottimes,[],linear_position_df,decode,lfp_df,theta_df,
                                neural_df,mua_xr,head_speed,head_orientation,
                                ripple_consensus_trace=None,
                                title = '', savefolder = output_folder, savename = filename,
                                simple = True, tetrode2ind = spikeColInd, likelihood = False,mua_thresh=mua_threshold,causal = decode_options["causal"],
                                plot_spiking=True, 
                                plot_changeofmind = True, turnaround = turnaround_t, head_direction_sign = head_direction_sign)
            
    if sorted:
        return rowID, smoothed_placefield, placefield_peak, spike_count_by_arm_direction, time_spent_by_arm_direction, betaPdfs, means
    return rowID
    
    
###### THE FOLLOWING 3 FUNCTIONS ARE USED FOR PLOTTING
def findDirectionPlot(t,log_df,linear_position_info,position_info):
    start = log_df.loc[t,'timestamp_H']
    end = log_df.loc[t,'timestamp_O']
    trialInd = (linear_position_info.index >= start) & (linear_position_info.index <= end)
    trialInfo = linear_position_info.loc[trialInd]
    trialInfo2D = position_info.loc[trialInd]
    
    arm_direction_t, arm_direction, all_arms_direction = find_direction(trialInfo, trialInfo2D)
    return arm_direction_t, arm_direction, all_arms_direction

def findPlottingStartEnd(t,log_df,linear_position_info):
    start = log_df.loc[t,'timestamp_H']
    end = log_df.loc[t,'timestamp_O']
    trialInd = (linear_position_info.index >= start) & (linear_position_info.index <= end)
    trialLinearInfo = linear_position_info.loc[trialInd]
    trialPosInfo = trialLinearInfo.loc[:,'track_segment_id']
    start_time_ind = np.argwhere(trialPosInfo >= 6).ravel()[0] #outer arm
    trialPosInfo = trialLinearInfo.iloc[start_time_ind:]
    return [trialPosInfo.index[0]-1,trialPosInfo.index[-1]+2]

def findTurnAround_t(t,rowID, turnaround_times):
    """this is for plotting only, just add 40ms to the left and to the right for camera sampling error."""
    turnaround_t = turnaround_times[np.argwhere(np.array(rowID) == t).ravel()[0]]
    turnaround_array = np.zeros((len(turnaround_t),2))
    turnaround_array[:,0] = np.array(turnaround_t) - 0.04
    turnaround_array[:,1] = np.array(turnaround_t) + 0.04
    
    return turnaround_array

######
def find_trials_session(nwb_copy_file_name,session_name,position_name,return_all = False,proportion_threshold = 0.1):
    # 1. load session's linear position info
    print('currently investigating:')
    print(session_name)
    print(position_name)

    linear_position_info=(IntervalLinearizedPosition() & {
        'nwb_file_name':nwb_copy_file_name,
        'interval_list_name':position_name,
        'position_info_param_name':'default_decoding'}).fetch1_dataframe()

    position_info = (IntervalPositionInfo() & {
        'nwb_file_name':nwb_copy_file_name,
        'interval_list_name':position_name,
        'position_info_param_name':'default_decoding'}).fetch1_dataframe()

    camera_frequency = 1/stats.mode(np.diff(linear_position_info.index))[0]
    
    # 2. load stateScript
    key={'nwb_file_name':nwb_copy_file_name,'epoch':int(session_name[:2])}
    log=(TrialChoice & key).fetch1('choice_reward')
    log_df=pd.DataFrame(log)
    
    rowID, trials, proportions, turnaround_times = find_trials(log_df,
                                                               linear_position_info, position_info, proportion_threshold = proportion_threshold)
    if return_all:
        return rowID, trials, proportions, turnaround_times, (nwb_copy_file_name, session_name)
    return rowID
    

def find_trials(log_df, linear_position_info, position_info, proportion_threshold = 0.2, nearby = False):
    """
    Find trials with more than 2 arm segments visits, 
        it also records for each trial with turning around behavior, 
            the time at which turning around happens, and the max proportion reached into the arm.
    
    The functions calls findProportion, which calls find_turnaround_time.
    log_df is behavior parsing
    linear_position_info is frame-by-frame position
    position_info is frame-by-frame 2d position
    
    output:
    trials is the arm segment number. 6 is arm 1, 7 is arm 2, etc
    rowID is trial number.
    """
    camera_frequency = 1/np.mean(np.diff(linear_position_info.index))
    maxLength = int(60*camera_frequency) #use at most 60 seconds prior to nose poke at the outer well. 
    trials = np.zeros((len(log_df.index),maxLength)) + np.nan
    proportions = np.zeros((len(log_df.index),maxLength)) + np.nan
    #directions = np.zeros((len(log_df.index),maxLength)) + np.nan
    turnaround_times = []
    rowInd = 0
    rowID = []
    maxLength_inpractice = 0
    for t in log_df.index: 
        
        # for each trial
        start = log_df.loc[t,'timestamp_H']
        end = log_df.loc[t,'timestamp_O']

        # restrict to this trial's position info
        trialInd = (linear_position_info.index >= start) &(linear_position_info.index <= end)
        trialPosInfo = linear_position_info.loc[trialInd,:]
        trialPosInfo = trialPosInfo.tail(maxLength) #use at most xx seconds prior to nose poke at the outer well. 
        
        trialPosInfo2D = position_info.loc[trialInd,:]
        trialPosInfo2D = trialPosInfo2D.tail(maxLength) #use at most xx seconds prior to nose poke at the outer well. 
        
        trialSeg = np.array(trialPosInfo.track_segment_id)
        
        # only save those that have more than 2 outer arms (home and center segment is there for sure)
        if (len(np.unique(trialSeg)) >= 4):
            
            # change into proportion
            proportion, track_segment_id, max_proportion, turnaround_time = findProportion(trialPosInfo)

            if np.max(max_proportion) >= proportion_threshold:
                trials[rowInd,:len(track_segment_id)] = track_segment_id
                proportions[rowInd,:len(track_segment_id)] = proportion
                
                #directions[rowInd,:len(track_segment_id)] = find_direction(trialPosInfo, trialPosInfo2D)
                
                rowInd = rowInd + 1
                rowID.append(t)
                turnaround_times.append(turnaround_time)
                maxLength_inpractice = np.max([maxLength_inpractice,len(track_segment_id)])
                
    trials = trials[:rowInd,:maxLength_inpractice]
    proportions = proportions[:rowInd,:maxLength_inpractice]
    #directions = directions[:rowInd,:maxLength_inpractice]
    
    # trials are track_segment_id for each trial
    # rowIDs are the trials
    if nearby:
        
        # figure out nearby trials
        rowID_ = []
        turnaround_times_ = []
        for r in rowID:
            for r_ in [r - 1, r + 1, r + 2, r - 2, r + 3, r - 3]:
                condition1 = np.isin(r_,np.array(log_df.index[:-1]))
                condition2 = ~np.isin(r_,np.array(rowID))
                if condition1 and condition2:
                    break
            rowID_.append(r_)
        
        # figure out turnaround_times
        for r in rowID_:
            turnaround_times_.append([log_df.loc[r].timestamp_O])
        return rowID_, np.nan, np.nan, turnaround_times_
        

    return rowID, trials, proportions, turnaround_times

def unique_stable(arr):
    unique_values, indices = np.unique(arr, return_index=True)
    unique_values_stable = unique_values[np.argsort(indices)]

    return unique_values_stable

def find_direction(trialInfo, trialInfo2D):
    #trialInfo is 1D position info
    outerArmInd = trialInfo.track_segment_id >= 6
    trialInfo = trialInfo.loc[outerArmInd,:]
    trialInfo2D = trialInfo2D.loc[outerArmInd,:]
    
    head_orientation = np.array(trialInfo2D.head_orientation)
    head_orientation_cos = np.cos(head_orientation)
    head_orientation_sin = np.sin(head_orientation)

    all_arms_direction = []
    for key in vectors.keys():
        all_arms_direction.append(head_orientation_cos*vectors[key][0] + head_orientation_sin*vectors[key][1])
    all_arms_direction = np.array(all_arms_direction).T
    sub = (np.arange(len(trialInfo)),np.array(trialInfo.track_segment_id)-6)
    ind = np.ravel_multi_index(sub,np.shape(all_arms_direction))
    
    arm_direction = all_arms_direction.flat[ind]
    arm_direction[arm_direction > 0] = 1
    arm_direction[arm_direction < 0] = -1
    return trialInfo.index, arm_direction, all_arms_direction


def find_turnaround_time(proportion,trialPosInfoOuter):
    """detects turn around time"""
    proportion_diff = np.concatenate(([0],np.diff(proportion))) < 0
    proportion_diff = ndimage.binary_closing(proportion_diff,iterations = 75) 
    # irregularity under 0.3s are smoothened, 75 * 0.002 (2ms) * 2
    
    turnsaround = pd.Series(proportion_diff, index = trialPosInfoOuter.index)
    #turnsaround = pd.Series(proportion_diff < 0.0001, index = trialPosInfoOuter.index)
    turnsaround_segments = np.array(segment_boolean_series(
            turnsaround, minimum_duration=0.1)).reshape((-1,2))

    return turnsaround_segments[:,0]

def findProportion(trialPosInfo):
    outerArmInd = trialPosInfo.track_segment_id >= 4
    trialPosInfoOuter = trialPosInfo.loc[outerArmInd,:]
    
    # exclude the final segment in time
    last_arm = np.array(trialPosInfoOuter.track_segment_id)[-1]
    same_arm_last_segment = pd.Series(np.array(trialPosInfoOuter.track_segment_id) == last_arm, 
                                      index = trialPosInfoOuter.index)
    same_arm_last_segment_segments = np.array(segment_boolean_series(
            same_arm_last_segment, minimum_duration=0)).reshape((-1,2))
    
    trialPosInfoOuter = trialPosInfoOuter.loc[trialPosInfoOuter.index <= same_arm_last_segment_segments[-1][0],:]
    
    segments_involved = unique_stable(trialPosInfoOuter.track_segment_id)
    
    track_segment_node_start = np.array([nodes[i][0] for i in np.array(trialPosInfoOuter.track_segment_id)])
    track_segment_node_end = np.array([nodes[i][1] for i in np.array(trialPosInfoOuter.track_segment_id)])

    trialPosInfoOuter.projected_xy = np.hstack((np.array(trialPosInfoOuter.projected_x_position).reshape((-1,1)),
                                            np.array(trialPosInfoOuter.projected_y_position).reshape((-1,1))))
    
    full_length = linalg.norm(track_segment_node_start - track_segment_node_end, axis = 1)
    partial_length = linalg.norm(track_segment_node_start - trialPosInfoOuter.projected_xy, axis = 1)
    proportion = partial_length / full_length
    track_segment_id = trialPosInfoOuter.track_segment_id

    max_proportion = []
    turnaround_times = []
    for seg in segments_involved:
        seg_index = np.argwhere(trialPosInfoOuter.track_segment_id == seg).ravel()
        max_proportion.append( np.nanmax(proportion[seg_index]) )
        ts = find_turnaround_time(proportion[seg_index],
                                 trialPosInfoOuter[trialPosInfoOuter.track_segment_id == seg])
        # there can be multiple turnaround times in each arm
        for t in ts:
            turnaround_times.append(t)
        
    return proportion, np.array(track_segment_id).astype("int"), max_proportion, turnaround_times

def load_epoch_data_wrapper(nwb_copy_file_name, session_name, position_name, decode_options):
    # This function loads decode, LFP etc from various tables.

    epoch_num = (EpochPos & {'nwb_file_name':nwb_copy_file_name,'position_interval':position_name}).fetch1("epoch")
    (_,log_df,decode,head_speed,head_orientation,linear_position_df,
            lfp_df,theta_df,ripple_df,neural_df,_) = load_epoch_data(nwb_copy_file_name,epoch_num,
                                                      decode_options["classifier_param_name"],
                                                      decode_options["encoding_set"])
    """ find tetrodes with signal """
    groups_with_cell=(SpikeSortingRecordingSelection & {
            'nwb_file_name' : nwb_copy_file_name}).fetch('sort_group_id')
    groups_with_cell=np.setdiff1d(groups_with_cell,[100,101])
    channel_IDs = list(neural_df.keys())
        
    spikeColInd = {}
    for g in groups_with_cell:
        spikeColInd_ = np.argwhere(np.isin(channel_IDs,(Electrode() &  {'nwb_file_name' : nwb_copy_file_name,
                                                         'electrode_group_name':str(g)}).fetch('electrode_id'))).ravel()
        spikeColInd[g] = spikeColInd_


    """load MUA"""
    decode_threshold_method = decode_options["decode_threshold_method"]
    mua_path=(MUA & {'nwb_file_name': nwb_copy_file_name,
                     'interval_list_name':session_name}).fetch1('mua_trace')
    mua_xr = xr.open_dataset(mua_path)
    if decode_threshold_method == 'MUA_0SD':
        mua_threshold=(MUA & {'nwb_file_name': nwb_copy_file_name,
                    'interval_list_name':session_name}).fetch1('mean')
    elif decode_threshold_method == 'MUA_05SD':
        mua_threshold = (MUA & {'nwb_file_name': nwb_copy_file_name,
                    'interval_list_name':session_name}).fetch1('mean') + 0.5 * (MUA & {'nwb_file_name': nwb_copy_file_name,
                    'interval_list_name':session_name}).fetch1('sd')
    elif decode_threshold_method == 'MUA_M05SD':
        mua_threshold = (MUA & {'nwb_file_name': nwb_copy_file_name,
                    'interval_list_name':session_name}).fetch1('mean') - 0.5 * (MUA & {'nwb_file_name': nwb_copy_file_name,
                    'interval_list_name':session_name}).fetch1('sd')
    else:
        mua_threshold = 0

    return (log_df,decode,head_speed,head_orientation,
            linear_position_df,lfp_df,theta_df,ripple_df,neural_df,mua_xr,mua_threshold,spikeColInd)
    
def normalize(T_):
    T=T_.copy()
    for ti in range(4):
        if np.sum(T[ti])!=0:
            T[ti]=T[ti]/np.sum(T[ti])
    return T
