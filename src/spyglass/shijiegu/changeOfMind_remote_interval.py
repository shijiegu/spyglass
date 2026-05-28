import numpy as np
import pandas as pd
import pickle
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from ripple_detection.core import segment_boolean_series

from spyglass.shijiegu.decodeQuality import return_low_hpd_time
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.changeOfMind_triggered import region
from spyglass.shijiegu.ripple_add_replay import (find_start_end,
                                                 position_posterior2arm_posterior,
                                                 select_subset_helper,select_subset_helper_pd)

from spyglass.shijiegu.load import load_decode
from spyglass.shijiegu.changeOfMind_triggered import linear_map
from spyglass.shijiegu.changeOfMind_triggered_position import load_triggered_position_decode_day
from spyglass.shijiegu.Analysis_SGU import TrialChoice,DecodeIngredients,ChangeofMind,get_linearization_map,ChangeofMindTriggeredDecode,ChangeofMindRemoteTheta
from spyglass.shijiegu.changeOfMind_remote import is_rat_interior
from spyglass.shijiegu.changeOfMind import nodes, vectors
from scipy import stats

def parse_remote_master(animal,list_of_days,params,minimum_duration = 0.02, min_sum_posterior = 0.2, fill_spyglass = False):
    """This function calls find_remote_theta_animal() and saves the data"""
    
    all_info_animal, info_animal, time_intervals_animal, arm_identities_animal = find_remote_theta_animal(
        animal,list_of_days,fill_spyglass = fill_spyglass,
        minimum_duration = minimum_duration,
        min_sum_posterior = min_sum_posterior,
        **params)
    
    success = 1
    #all_info_animal_control, info_animal_control, time_intervals_animal_control, arm_identities_animal_control = find_remote_theta_animal(
    #    animal,list_of_days,classifier_param_name,encoding_set,control=True,fill_spyglass = False,**params)
    
    #success = save_remote_animal(animal, list_of_days, encoding_set, classifier_param_name,params,
    #                                all_info_animal, info_animal, time_intervals_animal, arm_identities_animal,
    #                                all_info_animal_control, info_animal_control, time_intervals_animal_control, arm_identities_animal_control,
    #                                )
    return success


output_folder = '/stelmo/shijie/change_of_mind_analysis/figure4/'
def return_save_name_remote_parser(animal, encoding_set, classifier_param_name, d1, d2, proportion = 0.1, use_1d = 1):
    save_name = f'{animal.lower()}_{encoding_set}_{classifier_param_name}_{d1}_{d2}_p{proportion}_use1d{use_1d}'
    return save_name

def save_remote_animal(animal, list_of_days, encoding_set, classifier_param_name, params,
                                all_info_animal, info_animal, time_intervals_animal, arm_identities_animal,
                                all_info_animal_control, info_animal_control, time_intervals_animal_control, arm_identities_animal_control
                                ):
    
    d1= list_of_days[0]
    d2= list_of_days[-1]
    proportion = params["proportion"]
    use_1d = int(params["use_1d"])
    save_name = return_save_name_remote_parser(animal, encoding_set, classifier_param_name, d1, d2, proportion, use_1d)
    file_path = output_folder + save_name + '.pkl'
    
    data = {}
    (data["all_info_animal"],data["all_info_animal_control"],
     data["info_animal"], data["info_animal_control"],
     data["time_intervals_animal"], data["time_intervals_animal_control"],
     data["arm_identities_animal"], data["arm_identities_animal_control"]) = (
                all_info_animal,all_info_animal_control,
                info_animal, info_animal_control,
                time_intervals_animal,time_intervals_animal_control,
                arm_identities_animal, arm_identities_animal_control)
        
    # Open the file in binary write mode and dump the data
    with open(file_path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data successfully pickled and saved to {file_path}")
    return 1

def load_remote_animal(animal, list_of_days, encoding_set, classifier_param_name,
                       proportion = 0.1, use_1d = 1, minimum_duration = 0.02, min_posterior = 0.2, spyglass = False):
    if spyglass:
        print("Loading from spyglass database instead of pickle file.")
        loaded_data = load_remote_animal_spyglass(animal, list_of_days,
                                                  encoding_set,
                                                  minimum_duration = minimum_duration,
                                                  min_posterior = min_posterior,
                                                  proportion = proportion, use_1d = use_1d)
        return loaded_data
    d1, d2 = list_of_days[0], list_of_days[-1]
    save_name = return_save_name_remote_parser(animal, encoding_set, classifier_param_name, d1, d2, proportion, use_1d)
    file_path = output_folder + save_name + '.pkl'
    
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
        print(f"Successfully loaded data from '{file_path}':")
    return loaded_data

def load_remote_animal_spyglass(animal, list_of_days, parameter_name, minimum_duration = 0.02,min_posterior=0.2,
                                proportion = 0.1, use_1d = 1):
    """load from spyglass database instead of pickle file."""
    day_session_animal = []
    time_intervals_animal = []
    arm_identities_animal = []
    change_of_mind_num_animal = []
    
    for day in list_of_days:
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        remote_parameter = f"dur_{minimum_duration}_sum_{min_posterior}"
        for ind in range(len(session_interval)):
            
            session_name = session_interval[ind]
            position_name = position_interval[ind]
            epoch_num = int(session_name[:2])
    
            key_pre = {"nwb_file_name": nwb_copy_file_name, "epoch":epoch_num,
                       "minimum_duration":minimum_duration,"remote_parameter":remote_parameter,
                       "proportion":proportion, "parameter": parameter_name}
            query = ChangeofMindRemoteTheta & key_pre
            if len(query) == 0:
                print("No triggered decode found for ", key_pre)
                continue
            
            df = ChangeofMindRemoteTheta().fetch1_dataframe(key_pre)
            
            trials = df[df.has_remote_interval].index
            for trial in trials:
                remote_interval = df.loc[trial,'remote_interval']
                remote_content = df.loc[trial,'remote_content']
                change_of_mind_num = df.loc[trial,'change_of_mind_num']
                
                
                # if use post stopping content only
                # initial_stopping = df.loc[trial,'initial_time']
                # post_ind = [ind for ind in range(len(remote_interval)) if remote_interval[ind][0] >= initial_stopping]
                # remote_interval = [remote_interval[ind] for ind in post_ind]
                # remote_content = [remote_content[ind] for ind in post_ind]
                # if len(remote_content) == 0:
                #     continue
                
                change_of_mind_num_animal.append(change_of_mind_num)
                day_session_animal.append([nwb_copy_file_name, session_name, [trial for i in range(len(remote_interval))]])
                time_intervals_animal.append(remote_interval)
                arm_identities_animal.append(remote_content)

    return {
        "info_animal": day_session_animal,
        "change_of_mind_num_animal": change_of_mind_num_animal,
        "time_intervals_animal": time_intervals_animal,
        "arm_identities_animal": arm_identities_animal
    }
            
                

def find_remote_theta_animal(animal,list_of_days,
                             parameter_name = None,
                             use_1d = True,
                             use_center = False, use_outer = False, use_home = True,
                             proportion = 0.05,
                             speed_threshold = 4,
                             minimum_duration = 0.02,
                             min_sum_posterior = 0.2,
                             fill_spyglass = False):
    """
    default parameters should be:
        multiple_CoM = True, single_CoM = True, first_CoM = False,
        max_flag = True,
        delta_t_minus = 5,delta_t_plus = 5,
        "segment_only": False,
        
    Similar to find_remote_theta_animal_new(), but instead of a lumpsum of posterior in arms, it classfies intervals.
    use_1d: if True, use 1D decoding. if False, use 1D decoding collapsed from 2D decoding
    use_center: if True, consider moments when the rat in the center platform, and find decodes that in are outer arms
    use_home: if True, consider moments when the rat is in home arm, and find decodes that are in outer arms
    use_outer: if True, consider moments when the rat is in outer arms, and find decodes that are in all other outer arms including the home arm
    """
    if "2_state" in parameter_name:
        classifier_param_name = "default_decoding_gpu_4armMaze"
    elif "3_state" in parameter_name:
        classifier_param_name = "default_decoding_gpu_4armMaze_3states"
    else:
        classifier_param_name = "default_decoding_gpu_4armMaze"
        
    if "all_maze" in parameter_name:
        encoding_set = "all_maze"
    elif "run" in parameter_name:
        encoding_set = '2Dheadspeed_above_4'
    else:
        encoding_set = '2Dheadspeed_above_4'
        
    (day_session_animal, time_intervals_animal, arm_identities_animal) = (
        [],[],[])
    all_day_session_animal = [] # all the trials considered
    
    for day in list_of_days:
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        
        
        for ind in range(len(session_interval)):
            
            session_name = session_interval[ind]
            position_name = position_interval[ind]
            epoch_num = int(session_name[:2])
            
            # load triggered position and decode
            key_pre = {"nwb_file_name": nwb_copy_file_name, "epoch":epoch_num,
                       "proportion":proportion, "parameter": parameter_name}
            query = ChangeofMindTriggeredDecode & key_pre
            if len(query) == 0:
                print("No triggered decode found for ", key_pre)
                continue
            parameters = (ChangeofMindTriggeredDecode & key_pre).fetch1("parameter_value")
            
            loaded_data = ChangeofMindTriggeredDecode().fetch1_dataframe(key_pre)
            (triggered_positions, triggered_positions_abs,
             triggered_times_triggered, triggered_times_abs,
             triggered_trial_infos) = (
                    loaded_data["triggered_positions_baseoff"], loaded_data["triggered_positions"],
                    loaded_data["time_triggered"], loaded_data["time_abs"], 
                    loaded_data["triggered_trial_info"],
            )
            # make triggered_positions a dataframe, with index of triggered_times_abs
            for tp_ind in range(len(triggered_positions)):
                triggered_positions[tp_ind] = pd.DataFrame({
                    'linear_position': triggered_positions[tp_ind],
                }, index = triggered_times_abs[tp_ind])
                triggered_positions_abs[tp_ind] = pd.DataFrame({
                    'linear_position': triggered_positions_abs[tp_ind],
                }, index = triggered_times_triggered[tp_ind])
            
            entry = DecodeIngredients & {'nwb_file_name':nwb_copy_file_name,
                             'interval_list_name':session_name}
            # position_1d,position_2d,
            position_1d = pd.read_csv(entry.fetch1('position_1d')) #still need 1D position
            position_2d = pd.read_csv(entry.fetch1('position_2d')) # need 2D position
            
            # load ChangeofMind info
            key={'nwb_file_name':nwb_copy_file_name,'epoch':epoch_num,'proportion': proportion}
            print(ChangeofMind & key)
            log = ChangeofMind().fetch1_dataframe(key)
            log2 = log.copy()

            # initialization, for spyglass insertion
            log2.insert(6,'has_remote_interval',[False for i in range(len(log2))])
            log2.insert(7,'remote_interval',[[] for i in range(len(log2))])
            log2.insert(8,'remote_content',[[] for i in range(len(log2))])
            log2.insert(9,'change_of_mind_num',[[] for i in range(len(log2))])

            # load decode
            results1d = load_decode(nwb_copy_file_name,
                                    session_name,
                                    classifier_param_name = classifier_param_name,
                                    encoding_set = encoding_set,
                                    use_1d = use_1d)
            #posterior1d = results1d.sum("state")
            
            event_indices_session = np.arange(len(triggered_positions))#[ind for ind in range(len(triggered_trial_infos)) if triggered_trial_infos[ind][1] == session_name]
            
            for event_index in event_indices_session:
                triggered_position = triggered_positions[event_index]
                triggered_position_abs = triggered_positions_abs[event_index]
                triggered_trial_info = triggered_trial_infos[event_index]

                (trial,
                time_interval,
                replayed_arm_identity) = find_remote_theta_interval(
                    triggered_position, triggered_position_abs, triggered_trial_info,
                    results1d, log, position_1d, position_2d,
                    parameters["max_flag"],use_home,use_outer,use_center,
                    minimum_duration = minimum_duration,
                    min_sum_posterior = min_sum_posterior)
                
                
                if len(trial) > 0:
                    print("Found remote theta in trial ", trial)
                    day_session_animal.append([nwb_copy_file_name,session_name,trial])
                    time_intervals_animal.append(time_interval)
                    arm_identities_animal.append(replayed_arm_identity)
                    
                    trialID = trial[0]
                    log2.loc[trialID,'has_remote_interval'] = True
                    log2.at[trialID,'remote_interval'] += time_interval
                    log2.at[trialID,'remote_content'] += replayed_arm_identity
                    
                    if len(log2.at[trialID,'change_of_mind_num']) == 0:
                        log2.at[trialID,'change_of_mind_num'] += [0 for _ in range(len(replayed_arm_identity))]
                    else:
                        change_of_mind_num = int(np.max(log2.loc[trialID,'change_of_mind_num'])) + 1
                        log2.at[trialID,'change_of_mind_num'] += [change_of_mind_num for _ in range(len(replayed_arm_identity))]
                all_day_session_animal.append([nwb_copy_file_name,session_name,trial])
                
            # save back to spyglass
            if fill_spyglass:
                q = {}
                q["parameter"] = parameter_name
                q["pandas"] = log2.to_dict()
                q["nwb_file_name"] = nwb_copy_file_name
                q["epoch"] = epoch_num
                q["proportion"] = proportion
                q["remote_parameter"] = f"dur_{minimum_duration}_sum_{min_sum_posterior}"
                ChangeofMindRemoteTheta().insert1(q, replace = True)
            
    return all_day_session_animal, day_session_animal,time_intervals_animal,arm_identities_animal

def find_remote_interval(decode_subset, position2d, threshold = 20, minimum_duration = 0.02):
    
    position_axis = np.array(decode_subset.coords['position'])
    posterior_position_subset = decode_subset.causal_posterior.sum(dim='state')
    max_posterior_position1d = np.array(position_axis[posterior_position_subset.argmax(dim = 'position')])
    max_posterior_position2d = loc1d_to_2d_vector(max_posterior_position1d)
    
    # find remote time
    is_remote = np.sqrt(np.sum((max_posterior_position2d - position2d) ** 2, axis = 1)) > threshold
    
    is_remote_pd = pd.Series(is_remote, index = decode_subset.time)
    is_remote_segments = np.array(segment_boolean_series(
            is_remote_pd, minimum_duration=minimum_duration))
    
    if len(is_remote_segments) == 0:
        return [],[]

    time_intervals = []
    arm_identity = []
    
    for i in range(is_remote_segments.shape[0]):
        
        (t0,t1) = is_remote_segments[i]
        
        # restrict to continuous state:
        #    length of the continuous state should be greater than 20ms
        decode_subset_ = select_subset_helper(decode_subset,(t0,t1))
        state_subset = np.array(decode_subset_.causal_posterior.sum(dim='position'))
        time=np.array(decode_subset_.causal_posterior.time)
        
        snippets_conti = find_start_end(state_subset[:,0] > 0.5) #continuous
        snippets = [time[s] for s in snippets_conti if np.diff(time[s])[0]>minimum_duration]
        
        for s in range(len(snippets)):

            (t0_peak,t1_peak) = snippets[s]
            
            # overall sum of decode posterior in the max posterior arm should be greater than 0.2
            posterior_by_arm = position_posterior2arm_posterior(
                select_subset_helper(posterior_position_subset,snippets[s]),
                linear_map)
            
            # classify the max/mean posterior arm, exclude the arm the animal is physically at
            subset_ind = (decode_subset.time >= t0_peak) & (decode_subset.time <= t1_peak)
            subset_arm_snippet = linear2arm_including_home(max_posterior_position1d[subset_ind])
            if len(subset_arm_snippet) == 0:
                continue
            subset_arm_snippet = subset_arm_snippet[~np.isnan(subset_arm_snippet)]
            if len(subset_arm_snippet) == 0:
                continue
            
            modes = np.unique(subset_arm_snippet)
            final_arms = []
            for mode in modes:

                max_arm_ind = int(mode - 5)
    
                if np.mean(posterior_by_arm[max_arm_ind,:]) < 0.2:
                    continue
                final_arms.append(max_arm_ind)
            
            time_intervals.append(snippets[s])
            arm_identity.append(final_arms)

    return time_intervals, arm_identity

def find_remote_theta_interval(triggered_position,triggered_position_abs,triggered_trial_info,
                                decode,log_df,position_1d,position_2d,
                                max_flag = 1,use_home = False,use_outer = True, use_center = False,
                                minimum_duration = 0.02, min_sum_posterior = 0.2): # in seconds
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
    position_axis = np.array(decode.coords['position'])
        
    # find the arm the animal is at
    subset_arm = triggered_trial_info[-1] + 5
    
    # find the trial
    # find t0, t1 to consider
    trialID = triggered_trial_info[-2]
    (t0, t1) = (triggered_position.index[0],triggered_position.index[-1])
        
    if use_home:
        timestamp_H = log_df.loc[trialID,'timestamp_H']
        if not np.isnan(timestamp_H):
            t0 = timestamp_H
        
    # set by time
    position2d_subset = position_2d[np.logical_and(position_2d.time>=t0, position_2d.time<=t1)]
    position1d_subset = position_1d[np.logical_and(position_1d.time>=t0, position_1d.time<=t1)]
    decode_subset = select_subset_helper(decode,(t0,t1))
    if abs(len(position1d_subset) - len(decode_subset.time)) > 3:
        print("skipped due to decode and camera time frame do not fully match.")
        return [],[],[]
    
    # set by location    
    if use_home:
        subset_arm = 5
        # set by location
        #animal is physically at the home segment and not in the well area
        #stricter: remove well area
        subset_ind = np.logical_and(np.array(position1d_subset.linear_position) >= 10,
                                    np.array(position1d_subset.linear_position) <= linear_map[1][1])
    elif use_outer:  
        
        # set by location
        subset_ind = position1d_subset.track_segment_id == subset_arm
    else: # use center
        subset_arm = 5
        p_rat = np.hstack((np.array(position2d_subset.head_position_x).reshape((-1,1)),
                           np.array(position2d_subset.head_position_y).reshape((-1,1))))
        subset_ind = is_rat_interior(p_rat)
    
    position2d_subset = position2d_subset[subset_ind]
    position1d_subset = position1d_subset[subset_ind]
    decode_subset = decode_subset.isel(time = np.argwhere(subset_ind).ravel())
        
    # all previous operations restrict time to consider  
    
        
    posterior_position_subset = decode_subset.causal_posterior.sum(dim='state')
    # chew down decode to either mean or max position
    # get max posterior
    if max_flag:
        max_posterior_position = np.array(position_axis[posterior_position_subset.argmax(dim = 'position')])
    # get mean posterior
    else:
        posterior_position_subset_array = np.array(posterior_position_subset).T
        posterior_position_subset_array = posterior_position_subset_array/np.sum(posterior_position_subset_array, axis = 0)
        max_posterior_position = np.matmul(position_axis,posterior_position_subset_array)

    # find remote time
    is_remote = np.zeros_like(max_posterior_position) #just to initialize
    if use_home: # find remote arm representations when the animal is in the home arm
        
        for k in region.keys():
            (arm_base, arm_top) = region[k]
            is_remote = is_remote + np.logical_and(max_posterior_position <= arm_top, max_posterior_position >= arm_base)

    elif use_outer: # find remote representations when the animal is in outer arms
        # find representation in other arms
        for k in region.keys():
            if k == int(subset_arm):
                continue
            (arm_base, arm_top) = region[k]
            is_remote = is_remote + np.logical_and(max_posterior_position <= arm_top, max_posterior_position >= arm_base)
        # find remote representation at home
        is_remote = is_remote + np.logical_and(max_posterior_position >= 0, max_posterior_position <= linear_map[0][1])
    else: # use center
        for k in region.keys():
            (arm_base, arm_top) = region[k]
            is_remote = is_remote + np.logical_and(max_posterior_position <= arm_top, max_posterior_position >= arm_base)
            
    
    # restrict to moving time
    is_moving = np.array(position2d_subset.head_speed) > 4
    min_len = np.min([len(is_moving),len(is_remote)])
    # choose min because one variable is a subset of decode and the other is a subset of position.
    # there could be 1 or 2 time point difference.
    is_moving = is_moving[:min_len]
    is_remote = is_remote[:min_len]
    
    is_remote = np.logical_and(is_remote, is_moving)
    
    if min_sum_posterior == 0:
        # if no posterior threshold, there will be no continuity and state requirement,
        trials = [trialID]
        arm_identity = [0,1,2,3,4] # all arms including home arm
        time_intervals = []
        # for each region, count the number of time bins that the max posterior position falls into that region,
        for k in [5,6,7,8,9]:
            (arm_base, arm_top) = region[k]
            time_in_arm = np.logical_and(max_posterior_position <= arm_top, max_posterior_position >= arm_base)
            time_in_arm = time_in_arm[:min_len]
            time_in_arm = np.logical_and(time_in_arm, is_remote)
            delta_t = np.sum(time_in_arm) * np.median(np.diff(posterior_position_subset.time))
            t0 = float(posterior_position_subset.time[0])
            time_intervals.append((t0, t0 + delta_t))
        return trials, time_intervals, arm_identity
    
    is_remote_pd = pd.Series(is_remote, index = posterior_position_subset.time)
    is_remote_segments = np.array(segment_boolean_series(
            is_remote_pd, minimum_duration=minimum_duration))
    
    
    if len(is_remote_segments) == 0:
        return [],[],[]

    time_intervals = []
    arm_identity = []
    trials = []
    
    

            
    for i in range(is_remote_segments.shape[0]):
        
        (t0,t1) = is_remote_segments[i]
        
        # restrict to continuous state:
        #    length of the continuous state should be greater than 20ms
        decode_subset_ = select_subset_helper(decode_subset,(t0,t1))
        state_subset = np.array(decode_subset_.causal_posterior.sum(dim='position'))
        time=np.array(decode_subset_.causal_posterior.time)
        
        snippets_conti = find_start_end(state_subset[:,0] > 0.5) #continuous
        snippets = [time[s] for s in snippets_conti if np.diff(time[s])[0]>minimum_duration]
        
        for s in range(len(snippets)):

            (t0_peak,t1_peak) = snippets[s]
            
            # overall sum of decode posterior in the max posterior arm should be greater than 0.2
            posterior_by_arm = position_posterior2arm_posterior(
                select_subset_helper(posterior_position_subset,snippets[s]),
                linear_map)
            
            # classify the max/mean posterior arm, exclude the arm the animal is physically at
            subset_ind = (posterior_position_subset.time >= t0_peak) & (posterior_position_subset.time <= t1_peak)
            subset_arm_snippet = linear2arm_including_home(max_posterior_position[subset_ind])
            if len(subset_arm_snippet) == 0:
                continue
            subset_arm_snippet = subset_arm_snippet[~np.isnan(subset_arm_snippet)]
            if len(subset_arm_snippet) == 0:
                continue
            
            mode, count = stats.mode(subset_arm_snippet)
            
            if mode == subset_arm or count <= (len(subset_arm_snippet) * 0.8):
                #ambiguous situation, we will not consider those
                continue

            max_arm_ind = int(mode - 5)
    
            if np.mean(posterior_by_arm[max_arm_ind,:]) < min_sum_posterior:
                continue
            
            time_intervals.append(snippets[s])
            arm_identity.append(max_arm_ind)
            trials.append(trialID)
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

def linear2arm_including_home(position):
    arm = np.zeros_like(position) + np.nan
    for p_ind in range(len(position)):
        p = position[p_ind]
        for k in region.keys():
            if p>=region[k][0] and p<region[k][1]:
                arm[p_ind] = k
                continue
        if p >= 0 and p <= linear_map[0][1]:
            arm[p_ind] = 5
    return arm

def add_trial(t0,log_df):
    trial_ind=np.array(log_df.index)
    trial_number = trial_ind[np.argwhere((np.array(log_df.timestamp_O[:-1])-t0) > 0).ravel()[0]]
    return trial_number

def dotproduct(head_direction, arm):
    # arms are 0,1,2,3,4.
    
    # get arm direction
    arm_vector = vectors[arm + 5]
    
    return np.dot(head_direction, arm_vector)

#### return angle between remote content and the rat head direction
### code in figure4d calls the following functions

def find_angle(max_posterior_2d,head_orientation,animal_location):
    # in radian
    
    # make unit vector
    head_orientation_unit = np.hstack((np.cos(head_orientation).reshape((-1,1)),np.sin(head_orientation).reshape((-1,1)))) #unit vector
    displacement_unit = max_posterior_2d - animal_location
    displacement_unit = displacement_unit / np.linalg.norm(displacement_unit, axis = 1).reshape((-1,1))
    # shape of both displacement_unit and head_orientation_unit are (number of time bin, 2)
    
    # finally: return radian between head orientation and remote content
    dot_product = [np.dot(head_orientation_unit[i],displacement_unit[i].T) for i in range(displacement_unit.shape[0])]
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    return angle, head_orientation_unit, displacement_unit

linear_map,welllocations = get_linearization_map()
linear_map_arms = linear_map[[0,3,5,7,9]]

def loc1d_to_2d_vector(loc1d_vector, arm = None):
    loc2d_vector = np.array([loc1d_to_2d(loc1d, arm) for loc1d in loc1d_vector])
    return loc2d_vector
    
def loc1d_to_2d(loc1d, arm_avoid = None):
    # linear_map is like this:
    # array([[ 57.63896252,   0.        ], arm 0 base - outer
    #    [165.66041604, 252.64353221],     arm 1 base - outer
    #    [331.10211383, 418.29227956],     arm 2 base - outer
    #    [496.49241045, 583.18763812],     arm 3 base - outer
    #    [657.72023142, 743.03215907]]).   arm 4 base - outer

    # the arm the 1d location belongs, 1-indexed
    row_ind = np.argwhere(np.logical_and(linear_map_arms[:,0] <= loc1d, linear_map_arms[:,1] > loc1d)).ravel()
    if len(row_ind) == 0:
        return np.array([np.nan,np.nan])
    arm_id = int(row_ind)
    if arm_id == arm_avoid:
        return np.array([np.nan,np.nan])

    # the base - outer 1d
    arm = linear_map_arms[arm_id].ravel()
    if arm_id == 0:
        arm = arm[::-1]

    # convert to proportion
    proportion = (loc1d - arm[0])/(arm[1]-arm[0])

    # get 2D node location
    node = nodes[int(arm_id + 5)]
    loc2d = (node[1] - node[0]) * proportion + node[0]

    return loc2d

def loc1d_to_baseoff_vector(loc1d_vector, arm_avoid = None):
    loc2d_vector = np.array([loc1d_to_baseoff(loc1d) for loc1d in loc1d_vector])
    return loc2d_vector

def loc1d_to_baseoff(loc1d, arm_avoid = None):
    # linear_map is like this:
    # array([[ 0,       57.63896252],      arm 0 base - outer
    #    [165.66041604, 252.64353221],     arm 1 base - outer
    #    [331.10211383, 418.29227956],     arm 2 base - outer
    #    [496.49241045, 583.18763812],     arm 3 base - outer
    #    [657.72023142, 743.03215907]]).   arm 4 base - outer

    # the arm the 1d location belongs, 1-indexed
    row_ind = np.argwhere(np.logical_and(linear_map_arms[:,0] <= loc1d, linear_map_arms[:,1] > loc1d)).ravel()
    if len(row_ind) == 0:
        return np.array([np.nan,np.nan])
    arm_id = int(row_ind)
    if arm_id == arm_avoid:
        return np.array([np.nan,np.nan])

    # the base - outer 1d
    arm = linear_map_arms[arm_id].ravel()
    d = loc1d - arm[0]
    if arm_id == 0:
        d = arm[1] - loc1d

    return arm_id, d