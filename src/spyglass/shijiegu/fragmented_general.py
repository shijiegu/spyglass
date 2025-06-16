from spyglass.common.common_position import IntervalPositionInfo
from spyglass.shijiegu.load import load_run_sessions, load_zscored_ripple_consensus
from spyglass.shijiegu.Analysis_SGU import RippleTimesWithDecode, TrialChoice
from spyglass.shijiegu.ripple_detection import removeDataBeforeTrial1
import pandas as pd
import numpy as np

MIN_RIPPLE_DUR = 0.02 #20ms

def cont_vs_frag_power_day(animal,dates_to_plot,
                                encoding_set,classifier_param_name,decode_threshold_method,statescript_threshold):
    (power_frag_all, power_cont_all) = ({}, {})
    for d in dates_to_plot:
        nwb_copy_file_name = animal.lower() + d + '_.nwb'
        run_session_ids, run_session_names, pos_session_names = load_run_sessions(nwb_copy_file_name)

        (power_cont_day, power_frag_day) = ([],[])
        
        for ind in range(len(run_session_names)):
            session_name = run_session_names[ind]
            position_name = pos_session_names[ind]
            consensus, consensus_t = load_zscored_ripple_consensus(nwb_copy_file_name, session_name, position_name)
            
            StateScript = pd.DataFrame(
                (TrialChoice & {'nwb_file_name':nwb_copy_file_name,'epoch_name':session_name}).fetch1('choice_reward'))
            
            if len(StateScript) < statescript_threshold:
                continue

            key = {'nwb_file_name': nwb_copy_file_name,
                   'interval_list_name': session_name,
                   'classifier_param_name': classifier_param_name,
                   'encoding_set': encoding_set,
                   'decode_threshold_method':decode_threshold_method}
            try:
                ripple_times = pd.DataFrame((RippleTimesWithDecode & key
                             ).fetch1('ripple_times'))
            except:
                ripple_times = pd.read_pickle((RippleTimesWithDecode & key
                             ).fetch1('ripple_times'))
                
            median_cont = find_SWR_zscore(ripple_times,consensus,consensus_t,stat = "median", cont = True)
            median_frag = find_SWR_zscore(ripple_times,consensus,consensus_t,stat = "median", cont = False)
            power_cont_day.append(median_cont)
            power_frag_day.append(median_frag)
            
        power_frag_all[d] = power_frag_day
        power_cont_all[d] = power_cont_day
    return power_cont_all, power_frag_all 
    


def cont_vs_frag_occurrence_day(animal,dates_to_plot,
                                encoding_set,classifier_param_name,decode_threshold_method,statescript_threshold):
    (num_all,num_cont_all,
     num_frag_all,pct_cont_all,
     pct_frag_all,time_cont_all,time_frag_all) = ({},{},{},{},{},{},{})

    for d in dates_to_plot:
        nwb_copy_file_name = animal.lower() + d + '_.nwb'
        run_session_ids, run_session_names, pos_session_names = load_run_sessions(nwb_copy_file_name)

        (num_cont_day, num_frag_day,
         pct_cont_day, pct_frag_day,
         time_cont_day, time_frag_day) = ([],[],[],[],[],[])
        num_all_day = 0

        for ind in range(len(run_session_names)):
            session_name = run_session_names[ind]
            position_name = pos_session_names[ind]
            StateScript = pd.DataFrame(
                (TrialChoice & {'nwb_file_name':nwb_copy_file_name,'epoch_name':session_name}).fetch1('choice_reward'))
            
            if len(StateScript) < statescript_threshold:
                continue
            if nwb_copy_file_name == "klein20231111_.nwb" and session_name == "10_Rev2Session5":
                continue
            
            position_df = (IntervalPositionInfo &
                {'nwb_file_name': nwb_copy_file_name,
                'interval_list_name': position_name,
                'position_info_param_name': 'default'}
                    ).fetch1_dataframe()

            trial_1_t = StateScript.loc[1].timestamp_O
            trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O
            position_df = removeDataBeforeTrial1(position_df,trial_1_t,trial_last_t)
            second_per_frame = np.mean(np.diff(position_df.index)) * 2
            position_df_ = position_df[position_df.head_speed <= 4]
            diff = np.diff(position_df_.index)
            session_duration = np.sum(diff[diff <= second_per_frame])

            print("immobile time is ",str(session_duration),'seconds.')

            #session_duration = trial_last_t - trial_1_t

            key = {'nwb_file_name': nwb_copy_file_name,
                   'interval_list_name': session_name,
                   'classifier_param_name': classifier_param_name,
                   'encoding_set': encoding_set,
                   'decode_threshold_method':decode_threshold_method}
            try:
                ripple_times = pd.DataFrame((RippleTimesWithDecode & key
                             ).fetch1('ripple_times'))
            except:
                ripple_times = pd.read_pickle((RippleTimesWithDecode & key
                             ).fetch1('ripple_times'))
            (num_cont,num_frag,
             pct_cont,pct_frag,
             time_cont,time_frag) = cont_vs_frag_occurrence(ripple_times)
            num_cont_day.append(num_cont/session_duration)
            num_frag_day.append(num_frag/session_duration)
            pct_cont_day.append(pct_cont)
            pct_frag_day.append(pct_frag)
            time_frag_day.append(time_frag/session_duration)
            time_cont_day.append(time_cont/session_duration)
            num_all_day = num_all_day + len(ripple_times)

        num_all[d] = num_all_day
        num_cont_all[d] = num_cont_day
        num_frag_all[d] = num_frag_day
        pct_cont_all[d] = pct_cont_day
        pct_frag_all[d] = pct_frag_day
        time_cont_all[d] = time_cont_day
        time_frag_all[d] = time_frag_day
    return num_all, num_cont_all, num_frag_all, pct_cont_all, pct_frag_all, time_cont_all, time_frag_all





def cont_vs_frag_occurrence(ripple_times):
    num_cont, time_cont = find_SWR_number_involved(ripple_times,cont = True)
    num_frag, time_frag = find_SWR_number_involved(ripple_times,cont = False)
    pct_cont = num_cont/(num_cont + num_frag) #find_SWR_pct_involved(ripple_times,cont = True)
    pct_frag = num_frag/(num_cont + num_frag) #find_SWR_pct_involved(ripple_times,cont = False)

    return (num_cont,num_frag,pct_cont,pct_frag,time_cont,time_frag)

def find_SWR_zscore(ripple_times,consensus,consensus_t,stat = 'max',cont = True):
    SWR_times = find_SWR_time(ripple_times,cont)
    SWR_median = []
    for intvl in SWR_times:           
        ind = np.logical_and(consensus_t >= intvl[0],consensus_t <= intvl[1])
        consensus_subset = consensus[ind]
        if stat == 'max':
            SWR_median.append(np.nanmax(consensus_subset))
        elif stat == 'median':
            SWR_median.append(np.nanmedian(consensus_subset))
    return SWR_median

def find_SWR_time_sum(ripple_times,cont = True):
    sum = 0

    for i in ripple_times.index:
        if cont:
            intvls = ripple_times.loc[i].cont_intvl
        else:
            intvls = ripple_times.loc[i].frag_intvl
        for intvl in intvls:
            sum += intvl[1] - intvl[0]
    return sum

segment2location_dict = {}
segment2location_dict[0] = "home"
segment2location_dict[1] = "well1"
segment2location_dict[2] = "well2"
segment2location_dict[3] = "well3"
segment2location_dict[4] = "well4"

def segment2location(segment):
    segment = np.max(segment)
    return segment2location_dict[segment]

def find_SWR_time(ripple_times, cont = True, return_index = False):
    # return a list of ripple time intervals that are of continuous ripples or fragmented ripples
    # if return_index:
    #   return also the ripple index of each interval.
    final_intervals = []
    index = []
    for i in ripple_times.index:
        if cont:
            intvls = ripple_times.loc[i].cont_intvl
        else:
            intvls = ripple_times.loc[i].frag_intvl
        for interval_ind in range(len(intvls)):
            intvl = intvls[interval_ind]
            if cont: # if continuous, only replays that are not in the current location is counted
                replay_content = ripple_times.loc[i].cont_intvl_replay[interval_ind]
                if len(replay_content) == 0:
                    continue
                if segment2location(replay_content) == ripple_times.loc[i].animal_location:
                    continue
            if (intvl[1] - intvl[0]) < MIN_RIPPLE_DUR:
                continue
            
            final_intervals.append(intvl)
            index.append(i)

    if return_index:
        return final_intervals, index
    return final_intervals

def find_SWR_number_involved(ripple_times,cont = True):

    sum = 0 #this is for the sum of SWR number
    time_sum = 0
    for i in ripple_times.index:
        involved = False
        if cont:
            intvls = ripple_times.loc[i].cont_intvl
        else:
            intvls = ripple_times.loc[i].frag_intvl

        for interval_ind in range(len(intvls)):
            intvl = intvls[interval_ind]
            if cont: # if continuous, only replays that are not in the current location is counted
                replay_content = ripple_times.loc[i].cont_intvl_replay[interval_ind]
                if len(replay_content) == 0:
                    continue
                if segment2location(replay_content) == ripple_times.loc[i].animal_location:
                    continue
            if (intvl[1] - intvl[0]) > 0.02:
                involved = True
                time_sum = time_sum + intvl[1] - intvl[0]
        if involved:
            sum += 1

    return sum, time_sum

def find_SWR_pct_involved(ripple_times,cont = True):
    num, _ = find_SWR_number_involved(ripple_times,cont = cont)
    return num/len(ripple_times)