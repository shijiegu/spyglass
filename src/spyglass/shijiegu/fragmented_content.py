import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spyglass.shijiegu.ripple_add_replay import position_posterior2arm_posterior
from spyglass.shijiegu.Analysis_SGU import get_linearization_map, TrialChoice, RippleTimesWithDecode, DecodeResultsLinear
from spyglass.shijiegu.load import load_run_sessions, load_decode

linear_map,node_location=get_linearization_map()

def decode2content(interval,decode,mode = 'causal'):
    t0, t1 = interval
    mask_time = ((decode.time >= t0) & (decode.time < t1))
    if mode == 'causal':
        position_posterior = decode.isel(time=mask_time).causal_posterior.sum('state')
    elif mode == 'likelihood':
        position_posterior = decode.isel(time=mask_time).likelihood.sum('state')
    else:
        position_posterior = decode.isel(time=mask_time).acausal_posterior.sum('state')
    
                
    replay_content = position_posterior2arm_posterior(position_posterior,linear_map)
    replay_content = np.mean(replay_content,axis = 1)
    return replay_content

def ripple_times_to_content(ripple_times,decode,fragmented = True,
                            animal_location = "home",
                            mode = 'causal'):
    contents = [] # a list of tuples (trialID, time interval, content(home, arm1, arm2, arm3, arm4))
    for i in range(len(ripple_times)):
        if ripple_times.loc[i].animal_location[:4] != animal_location:
            continue
        ripple_time = ripple_times.loc[i]
        if fragmented:
            intvls = ripple_time.frag_intvl
        else:
            intvls = ripple_time.cont_intvl
            #replay_contents = ripple_time.cont_intvl_replay
        
        for intvl_ind in range(len(intvls)):
            intvl = intvls[intvl_ind]
            replay_content = decode2content(intvl,decode,mode = mode)
            replay_content[1:] = replay_content[1:]/np.sum(replay_content[1:]) # normalize by total content in arm 1-4  
            contents.append((ripple_time.trial_number, intvl, replay_content))
    return contents

def ripple_times_to_content_animal(animal, dates_to_plot,categories,
                                   animal_location = "home",
                                   classifier_param_name = "default_decoding_gpu_4armMaze",
                                   encoding_set = "2Dheadspeed_above_4_andlowmua",
                                   decode_threshold_method = None,
                                   fragmented = True, mode = 'causal',use_1d_decode = True):
    contents_all = {}
    durations_all = {}
    for d in dates_to_plot:
        nwb_copy_file_name = animal.lower() + d + '_.nwb'
        run_session_ids, run_session_names, pos_session_names = load_run_sessions(nwb_copy_file_name)
        
        parsed_contents_day = np.zeros(len(categories))
        durations_day = 0
        
        for ind in range(len(run_session_names)):
            session_name = run_session_names[ind]
            position_name = pos_session_names[ind]
            nwb_copy_file_name = animal.lower() + d + '_.nwb'
            epochID = int(session_name[:2])
            
            # log information
            key = {"nwb_file_name": nwb_copy_file_name,
                   "epoch": epochID}
            log_df = pd.DataFrame((TrialChoice() & key).fetch1("choice_reward"))
            
            # load ripple times
            key = {"nwb_file_name": nwb_copy_file_name,
                    "interval_list_name": session_name,
                    "decode_threshold_method": decode_threshold_method}
            ripple_times = pd.read_pickle((RippleTimesWithDecode() & key).fetch1("ripple_times"))
            
            # load decode
            decode = load_decode(nwb_copy_file_name,session_name,
                                 classifier_param_name, encoding_set, use_1d = use_1d_decode)
            
            contents = ripple_times_to_content(ripple_times, decode,
                                              fragmented = fragmented,
                                              animal_location = animal_location,
                                              mode = mode)
            #contents[1:] = contents[1:]/np.sum(contents[1:]) # normalize by total content in arm 1-4
            parsed_contents, durations = parse_content_categories(contents, log_df, categories)
            durations_day = durations_day + durations
            parsed_contents_day = parsed_contents_day + parsed_contents
        # sum across sessions
        contents_all[d] = parsed_contents_day/durations_day #np.sum(parsed_contents_day)
        durations_all[d] = durations_day
        
    return contents_all, durations_all

def parse_content_categories(contents, log_df, categories):
    # a list of tuples (trialID, time interval, content(home, arm1, arm2, arm3, arm4))
    parsed_contents = np.zeros(len(categories))
    durations = 0
    for trialID, intvl, content in contents:
        #arms = [int(log_df.loc[trialID,cat]) for cat in categories]
        duration = intvl[1] - intvl[0]
        
        for ind, cat in enumerate(categories):
            if cat == "home":
                arm = 0
            else:
                arm = log_df.loc[trialID,cat]
            if np.isnan(arm):
                continue
            else:
                arm = int(arm)
            parsed_contents[ind] = parsed_contents[ind] + content[arm] * duration
        
        durations += duration
    
    return parsed_contents, durations
            
        