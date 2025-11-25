import numpy as np
import pandas as pd
import pickle
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from spyglass.shijiegu.Analysis_SGU import RippleTimesWithDecode, TrialChoice

from spyglass.shijiegu.changeOfMind_remote import find_posterior_sum_segment, find_remote_theta_animal_new
from spyglass.shijiegu.changeOfMind_triggered_position import load_triggered_position_decode_day, return_triggered_2d_position
from spyglass.shijiegu.changeOfMindRipple import triggered_ripple_animal

def test_parsing_day(session_names_day, ripple_ind_day, reward, encoding_set, decode_threshold_method):
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

            key = {"nwb_file_name": nwb_copy_file_name, "interval_list_name":session_name,
                   "encoding_set": encoding_set,
                   "decode_threshold_method":decode_threshold_method}

            ripple_times_query = (RippleTimesWithDecode() & key).fetch1("ripple_times")
            ripple_times = pd.read_pickle(ripple_times_query)

        trialID = ripple_times.loc[ripple_ind_day[ind]].trial_number
        assert log_df.loc[trialID].rewardNum == reward
        ind += 1

def test_parsing(session_names, ripple_ind, reward, encoding_set, decode_threshold_method):
    for d in ripple_ind.keys():
        ripple_ind_day = ripple_ind[d]
        session_names_day = session_names[d]

        test_parsing_day(session_names_day, ripple_ind_day, reward, encoding_set, decode_threshold_method)

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

def figure2_ripple_data(animal, dates_to_plot, encoding_set, classifier_param_name, decode_thresh, home_ripple = False):

    (ripple_ind, session_names, ranges, durations, trialIDs) = triggered_ripple_animal(
         animal, dates_to_plot, encoding_set, classifier_param_name, decode_thresh,
         nearby = False, post = False, home_ripple = home_ripple)

    (ripple_ind_both, session_names_both, ranges_both, durations_both, trialIDs_both) = triggered_ripple_animal(
         animal, dates_to_plot, encoding_set, classifier_param_name, decode_thresh, nearby = False,
         post = False, both = True, home_ripple = home_ripple)
    
    (ripple_ind_nearby, session_names_nearby, ranges_nearby, durations_nearby, trialIDs_nearby) = triggered_ripple_animal(
         animal, dates_to_plot, encoding_set, classifier_param_name, decode_thresh,
         nearby = 1, post = True, home_ripple = home_ripple)
    
    (ripple_ind_nearby_rewarded, session_names_nearby_rewarded, ranges_nearby_rewarded, durations_nearby_rewarded, trialIDs_nearby_rewarded) = triggered_ripple_animal(
         animal, dates_to_plot, encoding_set, classifier_param_name, decode_thresh,
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

    if not home_ripple:
        test_parsing(session_names_nearby_rewarded, ripple_ind_nearby_rewarded, 2, encoding_set, decode_thresh)
        test_parsing(session_names_nearby, ripple_ind_nearby, 1, encoding_set, decode_thresh)
        
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
