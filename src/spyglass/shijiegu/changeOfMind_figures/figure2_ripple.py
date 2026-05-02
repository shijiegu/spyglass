import numpy as np
import pandas as pd
import pickle
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from spyglass.shijiegu.Analysis_SGU import RippleTimesWithDecode, RippleTimesByParameters, TrialChoice, ChangeofMindRemoteSWR, ChangeofMind

from spyglass.shijiegu.changeOfMind_remote import find_posterior_sum_segment, find_remote_theta_animal_new
from spyglass.shijiegu.changeOfMind_triggered_position import load_triggered_position_decode_day, return_triggered_2d_position
from spyglass.shijiegu.changeOfMindRipple import triggered_ripple_animal
from spyglass.shijiegu.ripple_add_replay import select_subset_helper_pd, select_subset_helper
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.load import load_decode
from spyglass.shijiegu.changeOfMind_triggered import region
from spyglass.shijiegu.decodeHelpers import session2position_name
from spyglass.shijiegu.changeOfMind_remote_interval import loc1d_to_2d_vector

from spyglass.common.common_position import IntervalLinearizedPosition, IntervalPositionInfo
from spyglass.shijiegu.changeOfMind_remote_interval import find_remote_interval

def test_parsing_day(session_names_day, ripple_ind_day, reward, parameter_name,
                     encoding_set = None, decode_threshold_method = None):
    # for eaach ripple, identify trial number in which it occurred
    # make sure that trial is rewarded or not rewarded
    session_name = ""

    ind = 0
    for session in session_names_day:
        if session[1] != session_name:
            nwb_copy_file_name, session_name = session
            key={'nwb_file_name':nwb_copy_file_name,'epoch':int(session_name[:2])}
            log=(TrialChoice & key).fetch1('choice_reward')
            log_df=pd.DataFrame(log)

            # key = {"nwb_file_name": nwb_copy_file_name, "interval_list_name":session_name,
            #        "encoding_set": encoding_set,
            #        "decode_threshold_method":decode_threshold_method}

            # ripple_times_query = (RippleTimesWithDecode() & key).fetch1("ripple_times")
            key = {"nwb_file_name": nwb_copy_file_name, "epoch":int(session_name[:2]),
                   "parameter_name": parameter_name}
            ripple_times = pd.DataFrame((RippleTimesByParameters() & key).fetch1("ripple_times"))

            #ripple_times = pd.read_pickle(ripple_times_query)

        trialID = ripple_times.loc[ripple_ind_day[ind]].trial_number
        assert log_df.loc[trialID].rewardNum == reward
        ind += 1

def test_parsing(session_names, ripple_ind, reward, parameter_name, encoding_set, decode_threshold_method):
    for d in ripple_ind.keys():
        ripple_ind_day = ripple_ind[d]
        session_names_day = session_names[d]

        test_parsing_day(session_names_day, ripple_ind_day, reward, parameter_name, encoding_set, decode_threshold_method)

def return_ratio_day(ripple_ind_day, range_day, duration_day):
    if len(duration_day) == 0:
        return 0
    ratio = np.sum(duration_day) / np.sum(np.diff(range_day, axis = 1))
    #ratio = len(ripple_ind_day) / np.sum(np.diff(range_day, axis = 1))
    return ratio
    
def return_ripple_metrics(data, d, both_flag):

    if both_flag:
        ratio = return_ratio_day(data["ripple_ind_both"][d], data["ranges_both"][d], data["durations_both"][d])
    else:
        ratio = return_ratio_day(data["ripple_ind"][d], data["ranges"][d], data["durations"][d])
        
    ratio_nearby = return_ratio_day(data["ripple_ind_nearby"][d], data["ranges_nearby"][d], data["durations_nearby"][d])
    ratio_nearby_rewarded = return_ratio_day(data["ripple_ind_nearby_rewarded"][d], data["ranges_nearby_rewarded"][d], data["durations_nearby_rewarded"][d])
    
    # 'trialID', 'trialID_both', 'trialID_nearby', 'trialID_nearby_rewarded'
    if both_flag:
        trial_num = len(data["trialID_both"][d])
    else:
        trial_num = len(data["trialID"][d])
    trial_num_nearby = len(data["trialID_nearby"][d])
    trial_num_nearby_rewarded = len(data["trialID_nearby_rewarded"][d])
    
    if both_flag:
        ripple_num = len(data["ripple_ind_both"][d])
    else:
        ripple_num = len(data["ripple_ind"][d])
    ripple_num_nearby = len(data["ripple_ind_nearby"][d])
    ripple_num_nearby_rewarded = len(data["ripple_ind_nearby_rewarded"][d])
    
    ripple_rate = ripple_num / trial_num if trial_num > 0 else 0
    ripple_rate_nearby = ripple_num_nearby / trial_num_nearby if trial_num_nearby > 0 else 0
    ripple_rate_nearby_rewarded = ripple_num_nearby_rewarded / trial_num_nearby_rewarded if trial_num_nearby_rewarded > 0 else 0

    return ratio, ratio_nearby, ratio_nearby_rewarded, ripple_rate, ripple_rate_nearby, ripple_rate_nearby_rewarded

def figure2_ripple_data(animal, dates_to_plot, parameter_name,
                        # the following at legacy arguments
                        encoding_set = None, classifier_param_name = None, decode_thresh = None, home_ripple = False):

    (ripple_ind, session_names, ranges, durations, trialIDs) = triggered_ripple_animal(
         animal, dates_to_plot, parameter_name, encoding_set, classifier_param_name, decode_thresh,
         nearby = False, post = False, home_ripple = home_ripple)

    (ripple_ind_both, session_names_both, ranges_both, durations_both, trialIDs_both) = triggered_ripple_animal(
         animal, dates_to_plot, parameter_name, encoding_set, classifier_param_name, decode_thresh, nearby = False,
         post = False, both = True, home_ripple = home_ripple)
    
    (ripple_ind_nearby, session_names_nearby, ranges_nearby, durations_nearby, trialIDs_nearby) = triggered_ripple_animal(
         animal, dates_to_plot, parameter_name, encoding_set, classifier_param_name, decode_thresh,
         nearby = 1, post = True, home_ripple = home_ripple)
    
    (ripple_ind_nearby_rewarded, session_names_nearby_rewarded, ranges_nearby_rewarded, durations_nearby_rewarded, trialIDs_nearby_rewarded) = triggered_ripple_animal(
         animal, dates_to_plot, parameter_name, encoding_set, classifier_param_name, decode_thresh,
         nearby = 2, post = True, home_ripple = home_ripple)
    
    ### save data
    data = {}
    data["ripple_ind"] = ripple_ind
    data["ripple_ind_both"] = ripple_ind_both
    data["ripple_ind_nearby"] = ripple_ind_nearby
    data["ripple_ind_nearby_rewarded"] = ripple_ind_nearby_rewarded

    
    data["session_names"] = session_names
    data["session_names_both"] = session_names_both
    data["session_names_nearby"] = session_names_nearby
    data["session_names_nearby_rewarded"] = session_names_nearby_rewarded
    
    data["ranges"] = ranges
    data["ranges_both"] = ranges_both
    data["ranges_nearby"] = ranges_nearby
    data["ranges_nearby_rewarded"] = ranges_nearby_rewarded
    
    data["durations"] = durations
    data["durations_both"] = durations_both
    data["durations_nearby"] = durations_nearby
    data["durations_nearby_rewarded"] = durations_nearby_rewarded
    
    data["trialID"] = trialIDs
    data["trialID_both"] = trialIDs_both
    data["trialID_nearby"] = trialIDs_nearby
    data["trialID_nearby_rewarded"] = trialIDs_nearby_rewarded

    output_folder = '/stelmo/shijie/change_of_mind_analysis/figure2'
    if home_ripple:
        file_path = f"{output_folder}/{animal[:5]}_home_ripple.pickle"
    else:
        file_path = f"{output_folder}/{animal[:5]}_ripple.pickle"
    with open(file_path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Data successfully pickled and saved to {file_path}")

    # if not home_ripple:
    #     test_parsing(session_names_nearby_rewarded, ripple_ind_nearby_rewarded, 2, parameter_name, encoding_set, decode_thresh)
    #     test_parsing(session_names_nearby, ripple_ind_nearby, 1, parameter_name, encoding_set, decode_thresh)
        
    return 1

def return_decode_pos_snippets(animal, day, ripple_inds, session_names, parameter_name,
                               classifier_param_name = 'default_decoding_gpu_4armMaze',
                               encoding_set = '2Dheadspeed_above_4'):
    
    positions_1d = []
    positions_2d = []
    decodes_1d = []
    decodes_2d = []
    decodes = []
    infos = []
    position_infos = []
    
    # for all sessions in the day,
    nwb_file_name = animal + day + '.nwb'
    nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
    session_name_old = None
    
    for ripple_ind, session_name_ in zip(ripple_inds, session_names):
        
        session_name = session_name_[1]
        
        if session_name != session_name_old:
            
            position_name = session2position_name(nwb_copy_file_name, session_name)
            session_name_old = session_name
            
            # load ripple times
            key = {"nwb_file_name": nwb_copy_file_name, "epoch":int(session_name[:2]),
                "parameter_name": parameter_name}
        
            ripple_times_query = (RippleTimesByParameters() & key).fetch1("ripple_times")

            ripple_times = pd.DataFrame(ripple_times_query)
            
            
        
            # load 1d position data
            position1d = (IntervalLinearizedPosition() & {
                'nwb_file_name':nwb_copy_file_name,
                'interval_list_name':position_name,
                'track_graph_name': '4 arm lumped 2023',
                'position_info_param_name':'default_decoding'}).fetch1_dataframe() #for debug use only
            
            position2d = (IntervalPositionInfo() & {
                'nwb_file_name':nwb_copy_file_name,
                'interval_list_name':position_name,
                'position_info_param_name':'default_decoding'}).fetch1_dataframe() #for debug use only
    
            # load decode
            decode = load_decode(nwb_copy_file_name,session_name,
                    classifier_param_name = classifier_param_name,
                    encoding_set = encoding_set)
            position_axis = np.array(decode.coords['position'])
        
        snippet = ripple_times.loc[ripple_ind][['start_time','end_time']].to_numpy().reshape(-1,2)
        
        (t0,t1) = snippet.ravel()
            
        # select position subset
        position_subset = position1d.loc[(position1d.index >= t0) & (position1d.index <= t1)]

                    
        # select decode subset
        decode_subset = select_subset_helper(decode,
                                                (position_subset.index[0]-0.001,position_subset.index[-1]+0.001),target_len = len(position_subset),
                                            epsilon = 0.001)  
        if len(position_subset) != len(decode_subset.time):
                continue
                    
        # get max decode position at each time point
        posterior_position_subset = decode_subset.causal_posterior.sum(dim='state')
        max_posterior_1d = np.array(position_axis[posterior_position_subset.argmax(dim = 'position')])
        max_posterior_2d = loc1d_to_2d_vector(max_posterior_1d, None) #exclude posterior1d in arm_identity

        positions_1d.append(np.array(position_subset.linear_position))
        
        position_info_2d = (position2d.loc[(position2d.index >= position_subset.index[0]) & (position2d.index <= position_subset.index[-1])])
        position_infos.append(position_info_2d)
        
        animal_location_2d = np.hstack((np.array(position_info_2d.head_position_x).reshape(-1,1),np.array(position_info_2d.head_position_y).reshape(-1,1)))
        positions_2d.append(animal_location_2d)
        
        decodes.append(decode_subset)
        decodes_1d.append(max_posterior_1d)
        decodes_2d.append(max_posterior_2d)
        infos.append((nwb_copy_file_name, session_name, (position_subset.index[0],position_subset.index[-1])))

    return positions_1d, positions_2d, decodes_1d, decodes_2d, infos, decodes, position_infos
    
def remote_ripple_identification(animal, dates_to_plot, data, parameter_name = "2state_10_cm", fill_spyglass = False):
    """data should have ripple_ind_both and session_names_both as fields"""
    
    for day in dates_to_plot:
        positions1d_swr, positions2d_swr, decodes1d_swr, decodes2d_swr, infos_swr, decodes_swr, positions_infos = return_decode_pos_snippets(animal,day,
                                                                                    data['ripple_ind_both'][day],
                                                                                    data['session_names_both'][day],
                                                                                    'default_conservative')
        nwb_copy_file_name_old = None
        session_name_old = None
        for ind in range(len(positions1d_swr)):
            position1d_swr = positions1d_swr[ind]
            position2d_swr = positions2d_swr[ind]
            decode1d_swr = decodes1d_swr[ind]
            decode2d_swr = decodes2d_swr[ind]
            decode_swr = decodes_swr[ind]
            info_swr = infos_swr[ind]
            position_info_swr = positions_infos[ind]
            
            nwb_copy_file_name, session_name, remote_interval = info_swr
            if nwb_copy_file_name != nwb_copy_file_name_old or session_name != session_name_old:
                # insert into table 
                if fill_spyglass and nwb_copy_file_name_old is not None and session_name_old is not None:  
                    q = {}
                    q["parameter_name"] = parameter_name
                    q["pandas"] = log2.to_dict()
                    q["nwb_file_name"] = nwb_copy_file_name_old
                    q["epoch"] = epoch_num
                    q["proportion"] = 0.1
                    ChangeofMindRemoteSWR().insert1(q, replace = True)
                
                nwb_copy_file_name_old = nwb_copy_file_name
                session_name_old = session_name

                # load ChangeofMind info
                epoch_num = int(session_name[:2])
                key={'nwb_file_name':nwb_copy_file_name,'epoch':epoch_num,'proportion': 0.1}
                print(ChangeofMind & key)
                log = ChangeofMind().fetch1_dataframe(key)
                log2 = log.copy()

                # initialization, for spyglass insertion
                log2.insert(6,'has_remote_interval',[False for i in range(len(log2))])
                log2.insert(7,'remote_interval',[[] for i in range(len(log2))])
                log2.insert(8,'remote_content',[[] for i in range(len(log2))])
                log2.insert(9,'change_of_mind_num',[[] for i in range(len(log2))])
            
            # find which trial this ripple belongs to
            log2_com = log2[log2.change_of_mind]
            trialID = np.array(log2_com.index[np.argwhere(log2_com.timestamp_H <= remote_interval[0])[-1]])[0]
            
            log2.at[trialID, 'change_of_mind_num'] = log2.loc[trialID].CoMNum_by_arm
            
            remote_interval, remote_content = find_remote_interval(decode_swr, position2d_swr, int(parameter_name.split("_")[1]))
            log2.at[trialID, 'remote_content'] += remote_content
            log2.at[trialID, 'remote_interval'] += remote_interval
            if len(remote_interval) > 0:
                log2.at[trialID, 'has_remote_interval'] = True
        
    return 1

# success = {}
# dates_to_plot_animals = {}
# encoding_sets = {}
# classifier_param_names = {}
# decode_threshs = {}

# animal = 'Eliot'
# dates_to_plot_animals[animal] = ['20221017','20221018','20221019','20221020','20221021','20221022','20221023','20221024','20221025','20221026']
# encoding_sets[animal] = '2Dheadspeed_above_4_andlowmua'
# classifier_param_names[animal] = 'default_decoding_gpu_4armMaze'
# decode_threshs[animal] = 'MUA_0SD'

# animal = 'Molly'
# dates_to_plot_animals[animal]  = ['20220416','20220417','20220418','20220419','20220420']
# encoding_sets[animal] = '2Dheadspeed_above_4'	
# classifier_param_names[animal] = 'default_decoding_gpu_4armMaze'
# decode_threshs[animal] = 'MUA_M05SD'

# animal = 'Klein'
# dates_to_plot_animals[animal] = ['20231101','20231102','20231103','20231104','20231105',
#                  '20231106','20231107','20231108','20231109','20231111']
# encoding_sets[animal] = '2Dheadspeed_above_4'	
# classifier_param_names[animal] = 'default_decoding_gpu_4armMaze'
# decode_threshs[animal] = 'MUA_0SD'

# animal = 'Lewis'
# dates_to_plot_animals[animal] = ['20240105','20240106','20240107','20240108','20240109',
#                  '20240110','20240113','20240114']
# encoding_sets[animal] = '2Dheadspeed_above_4'	
# classifier_param_names[animal] = 'default_decoding_gpu_4armMaze'
# decode_threshs[animal] = 'MUA_M05SD'

# animal = 'Julio'
# dates_to_plot_animals[animal] = ['20230801','20230802','20230803','20230804',
#                                  '20230805','20230806','20230807','20230808',
#                                  '20230809','20230810','20230811']
    
# encoding_sets[animal] = '2Dheadspeed_above_4'	
# classifier_param_names[animal] = 'default_decoding_gpu_4armMaze'
# decode_threshs[animal] = 'MUA_0SD'


# for animal in ["Molly","Eliot","Klein","Lewis","Julio"]:#"#decode_threshs.keys():
#     print("Processing animal:", animal)
#     dates_to_plot = dates_to_plot_animals[animal]
#     encoding_set = encoding_sets[animal]
#     classifier_param_name = classifier_param_names[animal]
#     decode_thresh = decode_threshs[animal]

#     success[animal] = figure2_ripple_data(
#         animal, dates_to_plot, encoding_set, classifier_param_name, decode_thresh, home_ripple = False)
#     #success[animal] = figure2_ripple_data(
#     #    animal, dates_to_plot, encoding_set, classifier_param_name, decode_thresh, home_ripple = True)
