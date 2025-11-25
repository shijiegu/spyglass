import spyglass as nd
import pandas as pd
import numpy as np
import xarray as xr
from scipy import stats

import os
import matplotlib.pyplot as plt
from spyglass.common import (Session, IntervalList,LabMember, LabTeam, Raw, Session, Nwbfile,
                            Electrode,LFPBand,interval_list_intersect)
from spyglass.common import TaskEpoch
from spyglass.spikesorting.v0 import (SortGroup, Curation,
                                    SpikeSortingRecording,SpikeSortingRecordingSelection)
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.common.common_position import IntervalPositionInfo, RawPosition, IntervalLinearizedPosition, TrackGraph
from spyglass.common.common_nwbfile import AnalysisNwbfile

from spyglass.shijiegu.Analysis_SGU import TrialChoice,EpochPos,MUA,get_linearization_map, ChangeofMind
from spyglass.shijiegu.decodeHelpers import runSessionNames, session2position_name
from spyglass.shijiegu.ripple_add_replay import plot_decode_spiking, plot_decode_sortedSpikes
from spyglass.shijiegu.load import load_epoch_data
from ripple_detection.core import segment_boolean_series
from spyglass.shijiegu.singleUnit import get_nwb_units
from spyglass.shijiegu.singleUnit_sortedDecode import place_field_direction, color_cells_by_place_direction

from spyglass.shijiegu.changeOfMind_helper import findProportion
from spyglass.shijiegu.changeOfMind_byTransition import time2arm
from spyglass.shijiegu.changeOfMind_triggered import return_change_of_mind_times_from_log

color_by_rat = {"eliot": "C0","molly":"C2","lewis":"C4","julio":"C1","klein":"deepskyblue"}

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
nodes[5] = (node_positions[1],node_positions[0])
nodes[6] = (node_positions[2],node_positions[3])
nodes[7] = (node_positions[4],node_positions[5])
nodes[8] = (node_positions[6],node_positions[7])
nodes[9] = (node_positions[8],node_positions[9])

vectors = {}
for key in nodes.keys():
    vector = nodes[key][1] - nodes[key][0]
    vectors[key] = vector/np.linalg.norm(vector)
vectors[1] = np.array([0,0])

rotation = np.array([[0,-1],[1,0]])
rotated_vectors = {key: np.matmul(rotation, vectors[key].reshape((2,1))) for key in vectors.keys()}
    
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
    
def find_trials_animal(animal, list_of_days,
                       plot = False,
                       sorted_spikes = False,
                       plot_ripple = False,
                       plot_spike = True,
                       proportion_threshold = 0.2):
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
                trials.append(find_trials_session_plot(nwb_copy_file_name,session_name,position_name,
                                                       sorted_spikes = sorted_spikes,
                                                       plot_ripple = plot_ripple,
                                                       plot_spike = plot_spike,
                                                       proportion_threshold=proportion_threshold))
            else:
                trials.append(find_trials_session(nwb_copy_file_name,
                                                  session_name,position_name,
                                                  proportion_threshold=proportion_threshold,return_all = True))
        trials_days[day] = trials
        
    return trials_days

def find_arm_proportion(arm, proportion_threshold, arm_id, proportion):
    arm_proportion = proportion[arm_id == arm]
    if len(arm_proportion) == 0:
        return 0
    
    max_prop = np.max(proportion[arm_id == arm]) 
    if max_prop > proportion_threshold:
        return max_prop
    return 0

def insertTrialChoiceChangeOfMind(trials_days, proportion_threshold):
    for date in trials_days.keys():
        
        for session_ind in range(len(trials_days[date])):
            trials_info = trials_days[date][session_ind]
            (nwb_file_name_copy, session_name)= trials_days[date][session_ind][4]
            position_name = session2position_name(nwb_file_name_copy, session_name)
            linear_position_info = (IntervalLinearizedPosition & {"nwb_file_name":nwb_file_name_copy, 
                                                "position_info_param_name": "default",
                                                "interval_list_name":position_name}).fetch1_dataframe()
    
            key={'nwb_file_name':nwb_file_name_copy,
                 'epoch_name':session_name}
            
            log=(TrialChoice & key).fetch1('choice_reward')
            epoch_num = (TrialChoice & key).fetch1('epoch')
            log_df=pd.DataFrame(log)
            log_df2 = log_df.copy()
            
            # initialization
            log_df2.insert(5,'change_of_mind',[False for i in range(len(log_df))])
            #hold boolean to indicate whether a trial is a change of mind
            
            log_df2.insert(6,'CoMMaxProportion',[np.nan for i in range(len(log_df))])
            #hold max proportion of traversed arms, in the case of multiple change of mind, this is the max of all the arms
                      
            log_df2.insert(7,'initial_choice',[np.nan for i in range(len(log_df))])
            # the first arm the animal has changed its mind
            
            log_df2.insert(8,'initial_time',[np.nan for i in range(len(log_df))])
            # the first time the animal has stopped to change its mind
            
            log_df2.insert(9,'proportion_arm1',[np.nan for i in range(len(log_df))])
            # the first time the animal has stopped to change its mind
            
            log_df2.insert(10,'proportion_arm2',[np.nan for i in range(len(log_df))])
            # the first time the animal has stopped to change its mind
            
            log_df2.insert(11,'proportion_arm3',[np.nan for i in range(len(log_df))])
            # the first time the animal has stopped to change its mind
            
            log_df2.insert(12,'proportion_arm4',[np.nan for i in range(len(log_df))])
            # the first time the animal has stopped to change its mind
            
            log_df2.insert(13,'CoMNum_by_time',[0 for i in range(len(log_df))])
            
            log_df2.insert(14,'CoMNum_by_arm',[0 for i in range(len(log_df))])
            
    
            # fill the table
            trials = trials_info[0]
            
            if len(trials) == 0:
                print("No change of mind on session " + session_name)
                
            else:
                arm_id = trials_info[1] - 5
                proportion = trials_info[2]
                max_proportion = np.nanmax(trials_info[2], axis = 1)
                turn_around_t = trials_info[3] # there are can be multiple turn arounds per trial!
                for trialID_ind in range(len(trials)):
                    if len(turn_around_t[trialID_ind]) == 0:
                        continue
                    trialID = trials[trialID_ind]
                    
                    log_df2.loc[trialID,'change_of_mind'] = True
                    log_df2.loc[trialID,'CoMMaxProportion'] = max_proportion[trialID_ind]
                    log_df2.loc[trialID,'initial_choice'] = time2arm(turn_around_t[trialID_ind][0],
                                                                     linear_position_info)
                    log_df2.loc[trialID,'initial_time'] = turn_around_t[trialID_ind][0]
                    log_df2.loc[trialID,'CoMNum_by_time'] = len(turn_around_t[trialID_ind])
                    
                    arm_id_trial = arm_id[trialID_ind]
                    
                    log_df2.loc[trialID,'proportion_arm1'] = find_arm_proportion(1, proportion_threshold, arm_id_trial, proportion[trialID_ind])
                    log_df2.loc[trialID,'proportion_arm2'] = find_arm_proportion(2, proportion_threshold, arm_id_trial, proportion[trialID_ind])
                    log_df2.loc[trialID,'proportion_arm3'] = find_arm_proportion(3, proportion_threshold, arm_id_trial, proportion[trialID_ind])
                    log_df2.loc[trialID,'proportion_arm4'] = find_arm_proportion(4, proportion_threshold, arm_id_trial, proportion[trialID_ind])
                    
                    log_df2.loc[trialID,'CoMNum_by_arm'] = np.sum([log_df2.loc[trialID,'proportion_arm1'] > 0,
                                                                   log_df2.loc[trialID,'proportion_arm2'] > 0,
                                                                   log_df2.loc[trialID,'proportion_arm3'] > 0,
                                                                   log_df2.loc[trialID,'proportion_arm4'] > 0,
                                                                   ])
                    
                    # Due to the nwb format, lists of different lengths are not able to be saved.
                    # Omitting the following field.
                    #turn_around_arms = list(unique_stable(trials_info[1][trialID_ind] - 5).astype("int"))
                    #log_df2.loc[trialID,'CoM_arm'].append(turn_around_arms)
                
            #animal = nwb_file_name_copy[:5]
            #savePath = os.path.join(f'/cumulus/shijie/recording_pilot/{animal}/decoding',
            #                nwb_file_name_copy+'_'+session_name + str(proportion_threshold) + '_changeofMindlog.pkl')
            #log_df2.to_pickle(savePath)
    
            # insert
            key = {"nwb_file_name":nwb_file_name_copy,
                   "epoch":epoch_num,
                   "proportion":str(proportion_threshold)}
            # Insert into analysis nwb file
            nwb_analysis_file = AnalysisNwbfile()
            key["analysis_file_name"] = AnalysisNwbfile().create(key["nwb_file_name"])
            key["pandas_id"] = nwb_analysis_file.add_nwb_object(
                analysis_file_name=key["analysis_file_name"],
                nwb_object=log_df2,
            )
            nwb_analysis_file.add(
                nwb_file_name=key["nwb_file_name"],
                analysis_file_name=key["analysis_file_name"],
            )
            
            ChangeofMind().insert1(key, replace = True)
            
            AnalysisNwbfile().log(key, table=ChangeofMind().full_table_name)
    return 1



def find_trials_session_plot(nwb_copy_file_name,session_name,position_name,
                             proportion_threshold = 0.1, sorted_spikes = False,
                             plot_ripple = False,
                             plot_spike = True,
                             curation_id = 1, decode_options = {}):
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
    key={'nwb_file_name':nwb_copy_file_name,'epoch':int(session_name[:2]),
         "proportion":str(proportion_threshold)}
    log_df = ChangeofMind().fetch1_dataframe(key)
    
    rowID, turnaround_times = return_change_of_mind_times_from_log(log_df, linear_position_info, nearby = False,
                                                                   multiple_CoM = True,
                                                                   single_CoM = True,
                                                                   first_CoM = False, 
                                                                   last_CoM = False)
    
    # 3. load data
    if len(decode_options.keys()) == 0:
        if animal.lower() == "eliot":
            decode_options["encoding_set"] = '2Dheadspeed_above_4_andlowmua'
            decode_options["classifier_param_name"] = 'default_decoding_gpu_4armMaze'
            decode_options["decode_threshold_method"] = 'MUA_0SD'
            decode_options["causal"] = True
            decode_options["likelihood"] = False
        else:
            decode_options["encoding_set"] = '2Dheadspeed_above_4'
            decode_options["classifier_param_name"] = 'default_decoding_gpu_4armMaze'
            decode_options["decode_threshold_method"] = 'MUA_M05SD'
            decode_options["causal"] = True
            decode_options["likelihood"] = False
    
    (_,decode,head_speed,head_orientation,
            linear_position_df,lfp_df,theta_df,
            ripple_df,neural_df,mua_xr,mua_mean,mua_sd,spikeColInd) = load_epoch_data_wrapper(
                nwb_copy_file_name, session_name, position_name, decode_options,
                load_ripple_flag = plot_ripple, load_spike_flag = plot_spike)
            
    output_folder = f'/cumulus/shijie/recording_pilot/{animal}/changeOfMind'
    
    # 3.5 for sorted spikes only:
    if sorted_spikes:
        
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
        
        if sorted_spikes:
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
                                simple = True, tetrode2ind = spikeColInd, likelihood = False,mua_thresh=mua_mean,causal = decode_options["causal"],
                                plot_spiking = plot_spike, 
                                plot_changeofmind = True, turnaround = turnaround_t, head_direction_sign = head_direction_sign)
            
    if sorted_spikes:
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
    maxLength = int(120*camera_frequency) #use at most 120 seconds prior to nose poke at the final outer well the rat picked. 
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
            proportion, track_segment_id, max_proportion, turnaround_time = findProportion(trialPosInfo, camera_frequency)
            
            if len(turnaround_time) == 0:
                continue

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

def find_direction_dot_product(trialInfo, trialInfo2D):
    #trialInfo is 1D position info
    #trialInfo2D is 2D position info
    outerArmInd = np.array(trialInfo.track_segment_id)
    
    head_orientation = np.array(trialInfo2D.head_orientation)
    head_orientation_cos = np.cos(head_orientation)
    head_orientation_sin = np.sin(head_orientation)

    arm_direction = np.array([vectors[ind] for ind in outerArmInd])
    arm_direction[outerArmInd == 0,0] = np.nan
    arm_direction[outerArmInd == 1,0] = np.nan
    arm_direction[outerArmInd == 0,1] = np.nan
    arm_direction[outerArmInd == 1,1] = np.nan
    rotated_arm_direction = np.matmul(rotation,arm_direction.T).T

    arm_direction = head_orientation_cos*arm_direction[:,0] + head_orientation_sin*arm_direction[:,1]
    rightward = head_orientation_cos*rotated_arm_direction[:,0] + head_orientation_sin*rotated_arm_direction[:,1]
    return arm_direction, rightward
    
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


def load_epoch_data_wrapper(nwb_copy_file_name, session_name, position_name, decode_options,
                            load_ripple_flag = True,
                            load_spike_flag = True,
                            use_1d_decode = True):
    # This function loads decode, LFP etc from various tables.

    epoch_num = (EpochPos & {'nwb_file_name':nwb_copy_file_name,'position_interval':position_name}).fetch1("epoch")
    (_,log_df,decode,head_speed,head_orientation,linear_position_df,
            lfp_df,theta_df,ripple_df,neural_df,_) = load_epoch_data(nwb_copy_file_name,epoch_num,
                                                      decode_options["classifier_param_name"],
                                                      decode_options["encoding_set"],
                                                      load_ripple_flag = load_ripple_flag,
                                                      load_spike_flag = load_spike_flag,
                                                      use_1d_decode = use_1d_decode)
    if load_spike_flag:
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
    else:
        spikeColInd = {}


    """load MUA"""
    #decode_threshold_method = decode_options["decode_threshold_method"]
    q = MUA & {'nwb_file_name': nwb_copy_file_name,
               'interval_list_name':session_name}
    mua_path= q.fetch1('mua_trace')
    mua_xr = xr.open_dataset(mua_path)
    mua_mean = q.fetch1("mean")
    mua_sd = q.fetch1("sd")
    """
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
    """

    return (log_df,decode,head_speed,head_orientation,
            linear_position_df,lfp_df,theta_df,ripple_df,neural_df,mua_xr,mua_mean,mua_sd,spikeColInd)
    
def normalize(T_):
    T=T_.copy()
    for ti in range(4):
        if np.sum(T[ti])!=0:
            T[ti]=T[ti]/np.sum(T[ti])
    return T
