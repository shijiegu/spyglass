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
nodes[6] = (node_positions[2],node_positions[3]) #arm1
nodes[7] = (node_positions[4],node_positions[5]) #arm2
nodes[8] = (node_positions[6],node_positions[7]) #arm3
nodes[9] = (node_positions[8],node_positions[9]) #arm4

linear_map,welllocations = get_linearization_map(track_graph_name='4 arm lumped 2023')
region={}
region[6] = linear_map[3]
region[7] = linear_map[5]
region[8] = linear_map[7]
region[9] = linear_map[9]

vectors = {}
for key in nodes.keys():
    vector = nodes[key][1] - nodes[key][0]
    vectors[key] = vector/np.linalg.norm(vector)
    
seq2 = {}
seq2[3] = 4
seq2[4] = 2
seq2[2] = 1
seq2[1] = 3

rev2={}
rev2[3] = 1
rev2[1] = 2
rev2[2] = 4
rev2[4] = 3

rev3={}
rev3[3] = 2
rev3[1] = 4
rev3[4] = 3
rev3[2] = 1

seq1={}
seq1[2] = 4
seq1[4] = 1
seq1[1] = 3
seq1[3] = 2

rev1={}
rev1[2] = 3
rev1[3] = 1
rev1[1] = 4
rev1[4] = 2

def turnaround_triggered_position(t,linear_position_info,
                                  delta_t_minus = 1,delta_t_plus = 1):
    # all time are in seconds
    # also return the arm the animal is in while change of mind
    
    # find the arm the animal is at
    (t0_peak,t1_peak) = (t-0.001, t+0.001)
    subset_ind = (linear_position_info.index >= t0_peak) & (linear_position_info.index <= t1_peak)
    subset_linear = linear_position_info.loc[subset_ind]
    subset_arm = np.unique(subset_linear.track_segment_id)
    arm_base = region[int(subset_arm)][0]

    # select subset
    (t0,t1) = (t - delta_t_minus, t + delta_t_plus)
    subset_ind = ((linear_position_info.index >= t0) & (linear_position_info.index <= t1))
    subset_linear = linear_position_info.loc[subset_ind]
    
    subset_ind2 = np.array(subset_linear.track_segment_id) == subset_arm
    subset_linear = subset_linear.loc[subset_ind2]

    peak_index = np.argwhere(subset_linear.index >= t).ravel()[0]
    y0 = subset_linear.iloc[peak_index].linear_position

    triggered_position = pd.Series(np.array(subset_linear.linear_position) - y0,
                                   index = subset_linear.index - t)
    triggered_position_abs = pd.Series(np.array(subset_linear.linear_position) - arm_base, 
                                       #with the base of the arm removed
                                   index = subset_linear.index)
    
    return triggered_position, triggered_position_abs, subset_arm[0] - 5

def turnaround_triggered_decode(triggered_position,triggered_position_abs,
                                decode,linear_position_info,max_flag = 1,segment_only = False):
    # segment only: consider posterior in that arm only
    
    position_axis = np.array(decode.coords['position'])
        
    # infer the turn around time
    
    t = np.array(triggered_position_abs.index)[np.argmin(np.abs(triggered_position.index-0))]

    # select subset of decode
    (t0,t1) = (triggered_position_abs.index[0]-0.0005,triggered_position_abs.index[-1]+0.0005)
    # add a little for floating point
    decode_subset = select_subset_helper(decode,(t0,t1),target_len = len(triggered_position_abs),
                                         epsilon = 0.001)  
    if len(decode_subset.time) != len(triggered_position_abs):
        return [],[],[]
    
    posterior_position_subset = decode_subset.causal_posterior.sum(dim='state')
    
    # find the arm the animal is at
    (t0_peak,t1_peak) = (t-0.001, t+0.001)
    subset_ind = (linear_position_info.index >= t0_peak) & (linear_position_info.index <= t1_peak)
    subset_linear = linear_position_info.loc[subset_ind]
    subset_arm = np.unique(subset_linear.track_segment_id)
    arm_base = region[int(subset_arm)][0]
    
    if segment_only:
        # find the arm the animal is at
        (t0_peak,t1_peak) = (t-0.001, t+0.001)
        subset_ind = (linear_position_info.index >= t0_peak) & (linear_position_info.index <= t1_peak)
        subset_linear = linear_position_info.loc[subset_ind]
        subset_arm = int(np.unique(subset_linear.track_segment_id))
        posterior_position_subset = select_subset_helper_position(posterior_position_subset,region[subset_arm])

    # get max posterior
    if max_flag:
        max_posterior_position = np.array(position_axis[posterior_position_subset.argmax(dim = 'position')])

    # get mean posterior
    else:
        posterior_position_subset_array = np.array(posterior_position_subset).T
        if segment_only: # restirct to times with more than 50% posterior in the arm.
            invalid_ind = np.sum(posterior_position_subset_array, axis = 0) <= 0.5
        posterior_position_subset_array = posterior_position_subset_array/np.sum(posterior_position_subset_array, axis = 0)
        if segment_only:
            position_axis_ = np.array(posterior_position_subset.coords['position'])
            max_posterior_position = np.matmul(position_axis_,posterior_position_subset_array)
            max_posterior_position[invalid_ind] = np.nan
        else:
            max_posterior_position = np.matmul(position_axis,posterior_position_subset_array)

    # find the decode position at turning around
    try:
        t0_ind = np.argwhere(np.array(decode_subset.time >= (t-0.0001))).ravel()[0] # add a little float point for stability
        y0 = max_posterior_position[t0_ind]
    except:
        y0 = max_posterior_position[-1]
    #print(y0)

    triggered_decode = pd.Series(max_posterior_position - y0,
                                 index = np.array(posterior_position_subset.time) - t)
    triggered_decode_base_subtracted = pd.Series(max_posterior_position - arm_base,
                                 index = np.array(posterior_position_subset.time))
    triggered_decode_abs = pd.Series(max_posterior_position,
                                 index = np.array(posterior_position_subset.time))
    
    return triggered_decode, triggered_decode_base_subtracted, triggered_decode_abs

def turnaround_triggered_mua(triggered_position_abs,triggered_position,decode,mean,sd):
    # any xarray will do.
        
    # infer the turn around time
    
    t = np.array(triggered_position_abs.index)[np.argmin(np.abs(triggered_position.index-0))]
    #print(t)

    # select subset of decode
    (t0,t1) = (triggered_position_abs.index[0]-0.0005,triggered_position_abs.index[-1]+0.0005)
    # add a little for floating point
    decode_subset = select_subset_helper(decode,(t0,t1))

    # z score
    y = np.array(decode_subset.to_dataarray()).ravel()
    y = (y-mean)/sd
    decode_subset = pd.Series(y,
                            index = np.array(decode_subset.time) -t)
    return decode_subset

def turnaround_triggered_speed(triggered_position_abs,triggered_position,df):
    # any pandas array will do.
        
    # infer the turn around time
    t = np.array(triggered_position_abs.index)[np.argmin(np.abs(triggered_position.index-0))]

    # select subset of decode
    (t0,t1) = (triggered_position_abs.index[0]-0.0005,triggered_position_abs.index[-1]+0.0005)
    # add a little for floating point
    decode_subset = select_subset_helper_pd(df,(t0,t1))  

    decode_subset.index = decode_subset.index - t
    return decode_subset
    

def find_triggered_animal(animal,list_of_days,delta_t_minus = 1,delta_t_plus = 1,
                          max_flag = False, segment_only = False,
                          nearby = False, # use nearby trial's outbound or inbound
                          proportion = 0.1, 
                          multiple_CoM = False, single_CoM = False, first_CoM = False, last_CoM = False):
    triggered_positions = []
    triggered_positions_abs = [] #absolute time
    triggered_decodes = []
    triggered_decodes_baseoff = []
    triggered_decodes_abs = [] #absolute time
    triggered_day_session_trial = []
    
    animal = animal[:5]
    
    for day in list_of_days:
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        for ind in range(len(session_interval)):
            session_name = session_interval[ind]
            position_name = position_interval[ind]
            
            (positions, positions_abs,
             decodes, decodes_baseoff, decodes_abs,
             day_session_trial) = find_triggered_session(nwb_copy_file_name,
                                                            session_name, position_name,
                                                            delta_t_minus, delta_t_plus,
                                                            max_flag, segment_only, nearby = nearby,
                                                            proportion = proportion,
                                                            multiple_CoM = multiple_CoM,
                                                            single_CoM = single_CoM,
                                                            first_CoM = first_CoM,
                                                            last_CoM = last_CoM)
            for position in positions:
                triggered_positions.append(position)
            for position in positions_abs:
                triggered_positions_abs.append(position)
            for decode in decodes:
                triggered_decodes.append(decode)
            for decode in decodes_abs:
                triggered_decodes_abs.append(decode)
            for decode in decodes_baseoff:
                triggered_decodes_baseoff.append(decode)
            for d in day_session_trial:
                triggered_day_session_trial.append(d)
                
        
    return (triggered_positions, triggered_positions_abs,
            triggered_decodes, triggered_decodes_baseoff, triggered_decodes_abs,
            triggered_day_session_trial)
    
def find_triggered_mua_animal(animal,list_of_days,proportion_threshold = 0.1,delta_t_minus = 1,delta_t_plus = 1,
                          nearby = False # use nearby trial's outbound or inbound
                          ):
    triggered_mua = []
    triggered_positioninfo = [] #absolute timee time
    triggered_day_session_trial = []
    
    for day in list_of_days:
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        for ind in range(len(session_interval)):
            session_name = session_interval[ind]
            position_name = position_interval[ind]
            
            (muas,
             speeds,
             day_session_trials) = find_triggered_mua_session(
                 nwb_copy_file_name,session_name,position_name,
                 delta_t_minus,delta_t_plus,
                 proportion_threshold = proportion_threshold,nearby = nearby)
            for mua in muas:
                triggered_mua.append(mua)
            for speed in speeds:
                triggered_positioninfo.append(speed)
            for day_session_trial in day_session_trials:
                triggered_day_session_trial.append(day_session_trial)              
        
    return (triggered_mua, triggered_positioninfo,
            triggered_day_session_trial)
    
def find_triggered_mua_session(nwb_copy_file_name,session_name,position_name,delta_t_minus,delta_t_plus,
                               proportion_threshold,nearby):
    # It is very similar to decode trigger, 
    #   where we restrict triggered traces to the time animal is in the arm segment
    
    # 1. load session's linear position info
    print('currently investigating:')
    print(session_name)
    print(position_name)
    animal = nwb_copy_file_name[:5]
    
    
    linear_position_info=(IntervalLinearizedPosition() & {
        'nwb_file_name':nwb_copy_file_name,
        'interval_list_name':position_name,
        'position_info_param_name':'default_decoding'}).fetch1_dataframe()

    log_df = pd.read_pickle( (TrialChoiceChangeofMind() & {
        "nwb_file_name": nwb_copy_file_name,
        "epoch":int(session_name[:2]),
        "proportion":str(proportion_threshold)}).fetch1("change_of_mind_info") )
    
    rowID, turnaround_times = return_change_of_mind_times_from_log(log_df, nearby,
                                                                   multiple_CoM = False, single_CoM = False,
                                                                   first_CoM = True, last_CoM = False)
    # where we will look at all trials with Change of Mind occurances (multiple_CoM = False, single_CoM = False),
    # instead of looking at those trials with only one Change of Mind occurance.
    # we will look at only the first Change of mind time.
    
    # 3. load MUA
    key = {"nwb_file_name":nwb_copy_file_name,"interval_list_name":session_name}
    mua_path = (MUA() & key).fetch1("mua_trace")
    (mean, sd) = ((MUA() & key).fetch1("mean"), (MUA() & key).fetch1("sd"))
    mua_xr = xr.open_dataset(mua_path)
    
    # 4. triggered position traces
    triggered_positions = []
    triggered_positions_abs = [] #absolute time
    day_session_trial = []

    for trial_ind in range(len(rowID)):
        trial = rowID[trial_ind]
        ts = turnaround_times[trial_ind]
        
        # problemetic session/trials
        # remember to do this in line 513
        if (nwb_copy_file_name == "lewis20240113_.nwb" and 
            session_name == "02_Rev2Session1") and trial == 49:
            continue
        
        for t in ts:
            triggered_position, triggered_position_abs, arm = turnaround_triggered_position(t,linear_position_info,
                                                                                       delta_t_minus,delta_t_plus)
            triggered_positions.append(triggered_position)
            triggered_positions_abs.append(triggered_position_abs)
            day_session_trial.append((nwb_copy_file_name,session_name,trial,arm))
    
    # 5. triggered decode traces
    triggered_muas = []
    triggered_speeds = []
    for ind in range(len(triggered_positions)):
        triggered_position = triggered_positions[ind]
        triggered_position_abs = triggered_positions_abs[ind]
        triggered_mua = turnaround_triggered_mua(triggered_position_abs,
                                                 triggered_position,mua_xr,mean,sd)
        triggered_speed = turnaround_triggered_speed(triggered_position_abs,
                                                 triggered_position,position_info)
        
        triggered_muas.append(triggered_mua)
        triggered_speeds.append(triggered_speed)
    return triggered_muas, triggered_speeds, day_session_trial  #triggered_positions, triggered_positions_abs
    
def find_multiple_CoM_trials(log_df):
    rowID = []
    for t in log_df.index[:-2]:
        if not log_df.loc[t,'change_of_mind']:
            continue
            
        CoM_time = np.array(log_df.loc[t,"CoM_t"][0])
            
        CoM_arm = np.array(log_df.loc[t,"CoM_arm"][0])
        CoM_arm = CoM_arm[CoM_arm > 0]
            
        if len(CoM_time) > 1 and len(CoM_arm) > 2:
            rowID.append(t)
    return rowID

def find_single_CoM_trials(log_df):
    rowID = []
    for t in log_df.index[:-2]:
        if not log_df.loc[t,'change_of_mind']:
            continue
            
        CoM_time = np.array(log_df.loc[t,"CoM_t"][0])
            
        CoM_arm = np.array(log_df.loc[t,"CoM_arm"][0])
        CoM_arm = CoM_arm[CoM_arm > 0]
            
        if len(CoM_time) == 1 and len(CoM_arm) <= 2:
            rowID.append(t)
    return rowID

def return_change_of_mind_times_from_log(log_df, nearby = False, multiple_CoM = False, single_CoM = False, first_CoM = False, last_CoM = False):
    if multiple_CoM:
        rowID = find_multiple_CoM_trials(log_df)
        turnaround_times = np.array(log_df.loc[rowID,'CoM_t'])
    elif single_CoM:
        rowID = find_single_CoM_trials(log_df)
        turnaround_times = np.array(log_df.loc[rowID,'CoM_t'])
    else:
        rowID = np.array(log_df[log_df.change_of_mind].index)
        turnaround_times = np.array(log_df[log_df.change_of_mind].CoM_t)
        turnaround_times_ind = np.array([ind for ind in np.arange(len(turnaround_times)) if len(turnaround_times[ind][0]) > 0])
        if len(turnaround_times_ind) > 0:
            rowID = rowID[turnaround_times_ind]
            turnaround_times = turnaround_times[turnaround_times_ind]
        else:
            rowID = []
            turnaround_times = []
    turnaround_times = [tt[0] for tt in turnaround_times]
    print("\n turnaround_times 1",turnaround_times)
    if first_CoM:
        turnaround_times = [[tt[0]] for tt in turnaround_times]
    print("\n turnaround_times 2",turnaround_times)
    if multiple_CoM and last_CoM:
        turnaround_times = [[tt[-1]] for tt in turnaround_times]
        
        
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
            turnaround_times_.append([log_df.loc[r].timestamp_O - 0.2])
    
        rowID = rowID_
        turnaround_times = turnaround_times_
    return rowID, turnaround_times
    
    

def find_triggered_session(
        nwb_copy_file_name,session_name,position_name,
        delta_t_minus,delta_t_plus,
        max_flag,segment_only,nearby,
        proportion = 0.1,
        multiple_CoM = False, single_CoM = False, first_CoM = False, last_CoM = False):
    # if multiple_CoM is True, only plot trials with multiple change of mind.
    # This function returns "triggered decode", the function name is named poorly.
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

    #position_info = (IntervalPositionInfo() & {
    #    'nwb_file_name':nwb_copy_file_name,
    #    'interval_list_name':position_name,
    #    'position_info_param_name':'default_decoding'}).fetch1_dataframe()
    
    # 2. load stateScript
    #key={'nwb_file_name':nwb_copy_file_name,'epoch':int(session_name[:2])}
    #log=(TrialChoice & key).fetch1('choice_reward')
    #log_df=pd.DataFrame(log)
    
    # """if have not used TrialChoiceChangeofMind"""
    #rowID, trials, proportions, turnaround_times = find_trials(log_df,
    #                                                           linear_position_info, position_info, nearby = nearby)
    log_df = pd.read_pickle( (TrialChoiceChangeofMind() & {
        "nwb_file_name": nwb_copy_file_name,
        "epoch":int(session_name[:2]),
        "proportion":str(proportion)}).fetch1("change_of_mind_info") )
    
    rowID, turnaround_times = return_change_of_mind_times_from_log(log_df, nearby,
                                                                   multiple_CoM, single_CoM,
                                                                   first_CoM, last_CoM)
    
    # 3. load data
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
    
    # 4. triggered position traces
    triggered_positions = []
    triggered_positions_abs = [] #absolute time
    day_session_trial = []

    for trial_ind in range(len(rowID)):
        trial = rowID[trial_ind]
        ts = turnaround_times[trial_ind]
        
        # problemetic session/trials
        if (nwb_copy_file_name == "lewis20240113_.nwb" and 
            session_name == "02_Rev2Session1") and trial == 49:
            continue
        
        for t in ts:
            triggered_position, triggered_position_abs, arm = turnaround_triggered_position(t,linear_position_info,
                                                                                       delta_t_minus,delta_t_plus)
            triggered_positions.append(triggered_position)
            triggered_positions_abs.append(triggered_position_abs)
            day_session_trial.append((nwb_copy_file_name,session_name,trial,arm))
        
    # 5. triggered decode traces
    triggered_decodes = []
    triggered_decodes_baseoff = []
    triggered_decodes_abs = [] #absolute time
    for ind in range(len(triggered_positions)):
        triggered_position = triggered_positions[ind]
        triggered_position_abs = triggered_positions_abs[ind]
        triggered_decode, triggered_decode_baseoff, triggered_decode_abs = turnaround_triggered_decode(triggered_position,
                                                                            triggered_position_abs,
                                                                            decode,
                                                                            linear_position_info,
                                                                            max_flag = max_flag,
                                                                            segment_only = segment_only)
        triggered_decodes.append(triggered_decode)
        triggered_decodes_baseoff.append(triggered_decode_baseoff)
        triggered_decodes_abs.append(triggered_decode_abs)
        
    return triggered_positions, triggered_positions_abs, triggered_decodes, triggered_decodes_baseoff, triggered_decodes_abs, day_session_trial

def select_subset_helper_position(xr_ob,region):
    xr_ob=xr_ob.sel(
        position=xr_ob.position[
            np.logical_and(xr_ob.position>=region[0],xr_ob.position<=region[1])])
    return xr_ob


def find_triggered_log_animal(animal, list_of_days, proportion = 0.1, back_trial = 2, after_trial = 1, trialinfo = None):
    
    logs_tuple = {}
    logs_tuple_rand = {}
    
    for day in list_of_days:
        logs_tuple_day = []
        logs_tuple_rand_day = []
        
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        for ind in range(len(session_interval)):
            session_name = session_interval[ind]
            position_name = position_interval[ind]
            if trialinfo is not None:
                trialinfo_ = trialinfo[nwb_copy_file_name][session_name]
            else:
                trialinfo_ = None
            log_tuple, log_tuple_rand = find_triggered_log_session(
                nwb_copy_file_name,session_name,position_name,
                back_trial,after_trial, proportion,
                trialinfo_) # finds change of mind trials in that session and get trial triggered state scripts.
            logs_tuple_day.append(log_tuple)
            logs_tuple_rand_day.append(log_tuple_rand)
        logs_tuple[day] =  logs_tuple_day
        logs_tuple_rand[day] = logs_tuple_rand_day
    return logs_tuple, logs_tuple_rand
            
    

def find_triggered_log_session(nwb_copy_file_name,session_name,position_name,
                               back_trial, after_trial, proportion= 0.1, trialinfo = None):
    """_summary_

    Args:
        nwb_copy_file_name (_type_): _description_
        session_name (_type_): _description_
        position_name (_type_): _description_
        back_trial (int): number of trials BEFORE the change of mind trial to include in log_df
        after_trial (int): number of trials AFTER the change of mind trial to include in log_df
    """
    # in addition to finding trials with change of mind, this function also plots decode data
    # 
    # 1. load session's linear position info
    print('currently investigating:')
    print(session_name)
    print(position_name)

        
    # 2. load change of mind parsed result
    log_df = pd.read_pickle( (TrialChoiceChangeofMind() & {
        "nwb_file_name": nwb_copy_file_name,
        "epoch":int(session_name[:2]),
        "proportion":str(proportion)}).fetch1("change_of_mind_info") )
    rowID = np.array(log_df[log_df.change_of_mind].index)
        
    if len(rowID) == 0:
        return [],[]
    
    first_arm = []
    for trialID in rowID:
        first_arm.append(log_df.loc[trialID,"CoM_arm"][0][0])
    
    return_tuple = []
    for t_ind in range(len(rowID)):
        t = rowID[t_ind]
        if trialinfo is not None:
            if not np.isin(t,trialinfo):
                continue
        
        (t1,t2) = (t - back_trial, t + after_trial)
        if t1 < 1 or t2 > len(log_df) - 1:
            continue
        return_tuple.append((t, int(first_arm[t_ind]), log_df.loc[t1:t2]))
    
    # randomly choose trials before or after
    return_tuple_rand = []
    for t_ind in range(len(return_tuple)):
        t0 = return_tuple[t_ind][0]
        candidate_trials = [t0-2, t0+2, t0-1, t0+1, t0-3, t0+3]
        t0_rand = np.nan
        for t in candidate_trials:
            
            condition1 = ~np.isin(t,rowID)
            (t1,t2) = (t - back_trial, t + after_trial)
            condition2 = t1 >= 1 and t2 <= len(log_df) - 1
            
            if condition1 and condition2:
                t0_rand = t
                break
        return_tuple_rand.append((t0_rand, np.nan, log_df.loc[(t0_rand-back_trial
                                                                   ):(t0_rand+after_trial)]))
            
    return return_tuple, return_tuple_rand

def remove_nan(arr):
    arr = np.array(arr)
    return arr[~np.isnan(arr)]

def parse_to_transitions(log_tuple_t):
    """
    this function takes a tuple of (change of mind trial id, the arm the rat first picked, behavior log snippet)
        and returns the transition on trial before change of mind trial.
    It is a helper function used in find_triggered_transition().
    """
    (t,j_wouldhave,log_df) = log_tuple_t
    (i,j) = (int(log_df.loc[t-1,'OuterWellIndex']),int(log_df.loc[t,'OuterWellIndex']))
    return (i,j), (i,j_wouldhave)

def parse_to_correct(log_tuple_t, seq_map, rand = False):
    """
    It is a helper function used in find_triggered_transition().
    doc:
    **** It ignores illegal trials. ***
    seq_map: takes the past reward arm and returns the next correct arm
    rand: under rand = True, log_tuple_t should be a tuple of randomly selected behavior log snippet.
    """
    (t,j_wouldhave,log_df) = log_tuple_t
    rewardNum = log_df.loc[t,'rewardNum']
    if rewardNum>=1:
        correct = rewardNum == 2
    else: #ignore illegal trials
        return np.nan, np.nan

    # calculate would have correct, use past reward
    if rand:
        correct_wouldhave = np.nan
    else:
        past_reward = log_df.loc[t,'past_reward']
        if np.isnan(past_reward):
            return np.nan, np.nan
        past_reward = int(past_reward)
        correct_wouldhave = int(seq_map[past_reward]) == int(j_wouldhave)

    return correct, correct_wouldhave

def parse_to_recent(log_tuple_t):
    """
    It is a helper function used in find_triggered_transition().
    It parses whether the change of mind 
            outer arm choice is in the recent 2 trial's outer arm choices.
    It assumes the log_df snippet in log_tuple_t has at least 2 trials back.

    """
    (t,j_wouldhave,log_df) = log_tuple_t
    i_minus2 = log_df.loc[t-2,'OuterWellIndex']
    i_minus1 = log_df.loc[t-1,'OuterWellIndex']
    recent_set = np.array([i_minus1,i_minus2])
    
    is_in_recent = np.isin(log_df.loc[t,'OuterWellIndex'],recent_set)
    is_in_recent_wouldhave = np.isin(j_wouldhave,recent_set)
    return is_in_recent, is_in_recent_wouldhave

def find_triggered_trial_completion_animal(trialinfo = None,long_theta_flag = True):
    """
    This function is the workhorse function of time to trial completion analysis.
    """
    
    time_all = {}
    turn_num_all = {}
    for nwb_copy_file_name in trialinfo.keys():
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        
        time = []
        turn_num = []
        for ind in range(len(session_interval)):
            session_name = session_interval[ind]
            position_name = position_interval[ind]
            
            trialIDs = trialinfo[nwb_copy_file_name][session_name]
            if len(trialIDs) == 0:
                continue
            
            linear_position_info=(IntervalLinearizedPosition() & {
                'nwb_file_name':nwb_copy_file_name,
                'interval_list_name':position_name,
                'position_info_param_name':'default_decoding'}).fetch1_dataframe()

            position_info = (IntervalPositionInfo() & {
                'nwb_file_name':nwb_copy_file_name,
                'interval_list_name':position_name,
                'position_info_param_name':'default_decoding'}).fetch1_dataframe()
            
            # 2. load stateScript
            key={'nwb_file_name':nwb_copy_file_name,'epoch_name':session_name}
            log=(TrialChoice & key).fetch1('choice_reward')
            log_df=pd.DataFrame(log)
            
            rowID, trials, proportions, turnaround_times = find_trials(log_df,
                                                                    linear_position_info, position_info, nearby = False)
            
            mask = np.isin(rowID,trialIDs)
            if long_theta_flag:
                indices = np.where(mask)[0]
            else:
                indices = np.where(~mask)[0]
                

            for trial_ind in indices:
                trialID = rowID[trial_ind]
                turnaround_times_trial = turnaround_times[trial_ind]
                if len(turnaround_times_trial) == 0:
                #    assert 1 == 0
                    continue
                #t0 = turnaround_times_trial[-1]
                
                t0 = log_df.loc[trialID,"timestamp_H"]
                t1 = log_df.loc[trialID,"timestamp_O"]
                if np.isnan(t0):
                    continue
                time.append(t1 - t0)
                turn_num.append(len(turnaround_times_trial))
        
        if len(time) >= 5:
            time_all[nwb_copy_file_name] = np.mean(time)
            turn_num_all[nwb_copy_file_name] = np.mean(turn_num)
        else:
            time_all[nwb_copy_file_name] = np.nan
            turn_num_all[nwb_copy_file_name] = np.nan
    
    return time_all,turn_num_all
            
            

def find_triggered_transition_animal(animal, dates_to_plot, seq_maps, proportion = 0.1, trialinfo = None):
    """
    This function is the workhorse function of triggered transition and correctness.
    The parsed result aims to answer the questions
    (1) At which transitions do these change of mind occur?
    (2) Are these change of mind trials more correct than not?
    Doc:
    seq_maps are a dictionary of the rewarding sequences of the task of each day
    """
    
    # first find change of mind triggered behavior log
    logs_tuple, logs_tuple_rand = find_triggered_log_animal(animal, dates_to_plot, proportion = proportion,  trialinfo = trialinfo)
    
    transition_wouldhave = {}
    transition = {}
    correct = {}
    correct_rand = {}
    correct_wouldhave = {}
    recent = {}
    recent_wouldhave = {}
    retro_wouldhave = {}
    
    for day in logs_tuple.keys():
        transition_wouldhave_day = np.zeros((4,4))
        transition_day = np.zeros((4,4))
        correct_day = []
        correct_wouldhave_day = []
        retro_wouldhave_day = np.zeros((4,4)) #from the would have arm to j
        correct_rand_day = []
        recent_day = []
        recent_wouldhave_day = []
        seq_map = seq_maps[day]
        
        for session_ind in range(len(logs_tuple[day])):
            log_tuple = logs_tuple[day][session_ind]
            log_tuple_rand = logs_tuple_rand[day][session_ind]
            
            for trial_ind in range(len(log_tuple)):
                log_tuple_t = log_tuple[trial_ind] #for this trial
                log_tuple_rand_t = log_tuple_rand[trial_ind] #for this trial
                
                (i,j), (i,j_wouldhave) = parse_to_transitions(log_tuple_t)

                correct_t, correct_wouldhave_t = parse_to_correct(log_tuple_t, seq_map)
                correct_rand_t, _ = parse_to_correct(log_tuple_rand_t, seq_map, True)
                recent_t, recent_wouldhave_t = parse_to_recent(log_tuple_t)
                
                retro_wouldhave_day[j_wouldhave - 1, j - 1] += 1
                
                transition_wouldhave_day[i - 1,j_wouldhave - 1] += 1
                transition_day[i - 1,j - 1] += 1
                
                correct_day.append(correct_t)
                correct_wouldhave_day.append(correct_wouldhave_t)
                correct_rand_day.append(correct_rand_t)
                
                recent_day.append(recent_t)
                recent_wouldhave_day.append(recent_wouldhave_t)
                
        transition_wouldhave[day] = transition_wouldhave_day
        transition[day] = transition_day
        retro_wouldhave[day]= retro_wouldhave_day

        correct[day] = remove_nan(correct_day)
        correct_wouldhave[day] = remove_nan(correct_wouldhave_day)
        correct_rand[day] = remove_nan(correct_rand_day)
        
        recent[day] = remove_nan(recent_day)
        recent_wouldhave[day] = remove_nan(recent_wouldhave_day)
    
    return retro_wouldhave, transition_wouldhave, transition, correct, correct_rand, correct_wouldhave, recent, recent_wouldhave
       
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.gaussian_process.kernels import PairwiseKernel

def form_null_model(triggered_positions, triggered_decodes):
    position_ = []
    decodes_ = []
    for rendition_ind in range(len(triggered_positions)):
        p = triggered_positions[rendition_ind]
        d = triggered_decodes[rendition_ind]
        if len(p) == len(d):
            position_.append(p)
            decodes_.append(d)
    X_train = np.concatenate([np.array(p) for p in position_])
    Y_train = np.concatenate([np.array(d) for d in decodes_])

    # remove NaN
    notnan_ind = ~np.isnan(Y_train)
    X_train = X_train[notnan_ind]
    Y_train = Y_train[notnan_ind]

    # downsample for tractability
    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(Y_train.size), size=1000, replace=False)
    X_train, Y_train = X_train[training_indices], Y_train[training_indices]
    
    # reshape
    X_train = X_train.reshape(-1, 1)
    Y_train = Y_train.reshape(-1, 1)

    # fit GP
    kernel = 10 * RBF(length_scale=20)
    #kernel = PairwiseKernel(metric='linear',gamma = 20)
    noise_std = 100 # units cm
    
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9
    )
    gaussian_process.fit(X_train, Y_train)

    return gaussian_process, X_train, Y_train

def classify_trial(pos_query, decode_real, gaussian_process):
    # given a gaussian process null model between position and decode
    #       and real position and decode data 
    # First, we query all decode null model. 
    # flag if there are cumulative more than 40ms of (null - reality) > 0
    decode_mean_null, decode_std_null = gaussian_process.predict(np.array(pos_query).reshape((-1,1)), return_std=True)
    
    decode_null_u = pd.Series(decode_mean_null + 3 * decode_std_null, index = decode_real.index)
    decode_null_l = pd.Series(decode_mean_null - 3 * decode_std_null, index = decode_real.index)
    diff = np.logical_or((decode_real - decode_null_u) >= 0, (decode_real - decode_null_l) <= -0)
    diff = pd.Series(diff, index = decode_real.index)
    intervals = segment_boolean_series(diff, minimum_duration=0.01)
    #print("front deviation only 10")
    
    #delta_t = np.mean(np.diff(decode_real.index))
    #if np.sum(diff) * delta_t >= 0.04:
    
    
    dev_max = np.nanmax(np.abs(np.array(decode_real - pos_query)))
    if len(intervals) > 0:
        """
        # calculate largest metric
        deviation = []
        for intvl in intervals:
            decode_real_intvl = decode_real[
                np.logical_and(decode_real.index >= intvl[0],decode_real.index <= intvl[1])]
            pos_query_intvl = pos_query[
                np.logical_and(pos_query.index >= intvl[0],pos_query.index <= intvl[1])]
            dev = np.nanmax(np.abs(np.array(decode_real_intvl - pos_query_intvl)))
            
            deviation.append(dev)
        dev_max = np.max(np.array(deviation))
        """
        
        return True, dev_max
    return False, dev_max

def find_large_position_minus_decode_trials(animal, triggered_trial_info, 
                                            triggered_positions_baseoff, triggered_decodes_baseoff,
                                            triggered_positions_baseoff_nearby, triggered_decodes_baseoff_nearby):
    """return trial info and indices of the trigered parsing for these huge discrepancy dates"""

    # form a null model between deocde and position
    (positions_nearby, decodes_nearby) = (triggered_positions_baseoff_nearby[animal],
                                          triggered_decodes_baseoff_nearby[animal])
    gaussian_process, _ , _2 = form_null_model(positions_nearby, decodes_nearby)
    
    # compare real data to null
    (trial_info, positions_abs, decodes_abs) = (
        triggered_trial_info[animal],
        triggered_positions_baseoff[animal],
        triggered_decodes_baseoff[animal])
    trials = []   # with long theta
    inds = []
    dev_maxs = []
    trials_non = [] #without long theta
    inds_non = []
    dev_maxs_non = []
    
    for rendition_ind in range(len(positions_abs)):
        position_abs = positions_abs[rendition_ind]
        decode_abs = decodes_abs[rendition_ind]
        if len(position_abs) != len(decode_abs):
            continue
        flag, dev_max = classify_trial(position_abs, decode_abs, gaussian_process)
        if flag:
            inds.append(rendition_ind)
            trials.append(trial_info[rendition_ind])
            dev_maxs.append(dev_max)
        else:
            inds_non.append(rendition_ind)
            trials_non.append(trial_info[rendition_ind])
            dev_maxs_non.append(dev_max)
            

    return trials, inds, dev_maxs, trials_non, inds_non, dev_maxs_non

import matplotlib.cm as cm
cmap = cm.seismic
from matplotlib.colors import Normalize
   
def plot_physical_vs_mental_position(animal,positions_abs,
                                     decodes_abs):
    """the function signature is misleading, 
    the <position_abs> should expect base subtracted triggered_position.
    the <decodes_abs> should expect base subtracted triggered_decodes."""
    # sort by trace length
    length = [len(tt) for tt in decodes_abs]
    ind_by_length = np.argsort(length)
    norm = Normalize(vmin=0, vmax=len(ind_by_length) - 1)
    
    plt.figure(figsize=(5,5))
    ax1 = plt.gca()
    
    row_ind = 0
    for rendition_ind in ind_by_length:
        position_abs = positions_abs[rendition_ind]
        decode_abs = decodes_abs[rendition_ind]
        if len(position_abs) == len(decode_abs):
            ax1.plot(np.array(decode_abs), np.array(position_abs), linewidth = 1, alpha = 0.5, color = cmap(norm(row_ind)))
            row_ind = row_ind + 1
        else:
            print(rendition_ind)
    ax1.plot([0,ax1.get_ylim()[1]],[0,ax1.get_ylim()[1]],linestyle=":",color = "k",alpha = 0.5)
    
    ax1.axvline(86-18,linestyle=":",color = "k",alpha = 0.5)
    ax1.set_xlim([-1, 87])
    #ax1.set_ylim([-1, 87])
    ax1.set_aspect('equal')
    ax1.set_title(animal + "\n animal position (y) vs decoded position (x)")
    
    #ax2.set_title("decoded position relative to stopping")
    
    ax1.set_ylabel("linearized location (cm)")
    ax1.set_xlabel("decoded linearized loc (cm)")
    #ax2.set_ylabel("linearized location (cm)")
    #plt.gca().set_aspect('equal')
    return ax1
    
def normalize_to_1(matrix):
    return matrix / np.nansum(matrix)
def matrix_correlation(M1,M2):
    return np.sum(np.multiply(M1,M2))/(np.linalg.norm(M1,ord='fro')*np.linalg.norm(M2,ord='fro'))

def find_transitions(replay_trials):
    """this function works with find_large_position_minus_decode_trials()"""
    # initialize
    (T, T_wouldhave, T_wouldhave2) = ({}, {}, {})
    
    dates_to_plot = np.unique([replay_tuple[0] for replay_tuple in replay_trials])
    for d in dates_to_plot:
        (T[d], T_wouldhave[d], T_wouldhave2[d]) = (np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)))
    
    # fill the matrices
    for info in replay_trials:
        d = info[0]
    
        # parse
        key={'nwb_file_name':info[0],'epoch_name':info[1]}
        log=(TrialChoice & key).fetch1('choice_reward')
        log_df=pd.DataFrame(log)
        (i,j), (i,j_wouldhave) = parse_to_transitions((info[2], info[3], log_df))
    
        # record/log
        T[d][i-1,j-1] += 1
        T_wouldhave[d][i-1,j_wouldhave-1] += 1
        T_wouldhave2[d][j_wouldhave-1,j-1] += 1
    return T, T_wouldhave, T_wouldhave2

def find_transitions_sum(T, T_wouldhave, T_wouldhave2):
    # sum across days
    dates_to_plot = list(T.keys())

    (T_sum, T_wouldhave_sum, T_wouldhave2_sum, T_AB_sum) = (np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)))
    for d_ind in range(len(dates_to_plot)):
        d = dates_to_plot[d_ind]
        
        T_sum += T[d]
        T_wouldhave_sum += T_wouldhave[d]
        T_wouldhave2_sum += T_wouldhave2[d]
       
        if np.sum(T_wouldhave[d]) > 5:
            A = normalize(T_wouldhave[d])
            B = normalize(T_wouldhave2[d])
            A[np.isnan(A)] = 0
            B[np.isnan(B)] = 0
            AB = np.matmul(A, B) #T_wouldhave[d]
        
            T_AB_sum += AB
    return T_sum, T_wouldhave_sum, T_wouldhave2_sum, T_AB_sum

def find_behavior_sum(animal,replay_trials,dates_to_plot = None):
    
    # behavior change
    delta_behavior = np.zeros((4,4))
    if dates_to_plot is None:
        dates_to_plot = np.unique([replay_tuple[0][5:13] for replay_tuple in replay_trials])
    C_behavior_all, C_behavior_reward_all = behavior_transitions_count(animal,dates_to_plot)
    delta_behavior = normalize(C_behavior_all[dates_to_plot[-1]]) - normalize(C_behavior_all[dates_to_plot[0]])
    return delta_behavior

def trials_date_session_to_dict(replay_trials):
    # initialize
    trials_date_session = {}
    days = np.unique([r[0] for r in replay_trials])
    for d in days:
        trials_date_session[d] = {}
        session_interval, position_interval = runSessionNames(d)
        for s in session_interval:
            trials_date_session[d][s]= []
           
    # add content
    for r in replay_trials:
        (date,session,trialID,_)  = r
        trials_date_session[date][session].append(trialID)
    
    for d in days:
        for s in trials_date_session[d].keys():
            trials_date_session[d][s] = np.unique(trials_date_session[d][s])

    return trials_date_session


def triggered_ripple_counterfactual_animal(animal, dates_to_plot, encoding_set, classifier_param_name, decode_thresh):
    triggered_positions = {}
    triggered_positions_abs = {}
    triggered_decodes = {}
    triggered_decodes_baseoff = {}
    triggered_decodes_abs = {}
    triggered_trial_info = {}

    
    # find decode and position
    (triggered_positions[animal], triggered_positions_abs[animal],
     triggered_decodes[animal], triggered_decodes_baseoff[animal], triggered_decodes_abs[animal],
     triggered_trial_info[animal]) = find_triggered_animal(animal,dates_to_plot,
                                                                       delta_t_minus = 0,delta_t_plus = 1,
                                                                       max_flag = 0, segment_only = True)
    
    # find_large_position_minus_decode_trials
    CUTOFF = 25
    replay_trials, inds = find_large_position_minus_decode_trials(animal, triggered_trial_info, 
                                                triggered_positions_abs, triggered_decodes_baseoff,cutoff = CUTOFF)
    
    trials_date_session_dict = trials_date_session_to_dict(replay_trials)
    
    # input this to triggered_ripple_animal
    (ranges, ripple_ind, session_names, ranges_nearby, ripple_ind_nearby, session_names_nearby) = triggered_ripple_animal(
        animal, dates_to_plot, encoding_set, classifier_param_name, decode_thresh, post = True, trials = trials_date_session_dict)
    return (ranges, ripple_ind, session_names, ranges_nearby, ripple_ind_nearby, session_names_nearby)

import statsmodels.api as sm
def fitLM(x,y):
    # linear fit
    xc = sm.add_constant(x)
    results = sm.OLS(y, xc).fit()
    pvalue = results.f_pvalue
    
    return results, pvalue
    
def remove_diag(T_):
    T=T_.copy()
    np.fill_diagonal(T, np.nan)
    return T

def correlate_transition_deltabehavior(animal,T_wouldhave_sum, T_wouldhave2_sum, delta_behavior,
                                       x1 = None, x2 = None, y = None):
    fig, axes = plt.subplots(1,2,figsize = (10,3),sharex = True)

    if x1 is not None:
        x = x1
    else:
        x = remove_diag(normalize(T_wouldhave_sum)).ravel()
        y = remove_diag(delta_behavior).ravel()
    
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        x1 = x
    results, pvalue = fitLM(x,y)
    
    axes[0].scatter(x, y)
    x_plot=np.linspace(np.min(x),np.max(x),10).reshape(-1,1)
    y_plot=results.predict(sm.add_constant(x_plot))
    pvalue_beta = results.pvalues[1]
    
    axes[0].plot(x_plot,y_plot,color='blue',linewidth=3,label = str(pvalue))
    axes[0].text(np.min(x),np.min(y),f'p value = {np.round(pvalue_beta,4)}')
    
    axes[0].set_title(animal)
    axes[0].set_ylabel("behavior change")   
    axes[0].set_xlabel("would have transition 1")    

    if x2 is not None:
        x = x2
    else:
        x = remove_diag(normalize(T_wouldhave2_sum)).ravel()
        y = remove_diag(delta_behavior).ravel()
    
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        x2 = x
    results, pvalue = fitLM(x,y)
    
    axes[1].scatter(x, y)
    x_plot=np.linspace(np.min(x),np.max(x),10).reshape(-1,1)
    y_plot=results.predict(sm.add_constant(x_plot))
    pvalue_beta = results.pvalues[1]
    
    axes[1].plot(x_plot,y_plot,color='blue',linewidth=3,label = str(pvalue))
    axes[1].text(np.min(x),np.min(y),f'p value = {np.round(pvalue_beta,4)}')
    
    axes[1].set_title(animal)
    axes[1].set_ylabel("behavior change")   
    axes[1].set_xlabel("would have transition 2") 
    return x1, x2, y 