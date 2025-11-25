import numpy as np
import random
from spyglass.shijiegu.changeOfMind_triggered import find_triggered_log_animal, parse_to_correct, remove_nan, parse_to_last_correct

from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename  

def find_the_last_trial_animal(animal, dates_to_plot, seq_maps, proportion = 0.1, trialinfo = None):
    """
    This function is the workhorse function of supplementary figure 3.
    The parsed result aims to answer the questions
    (1) How does change of mind's last trial's reward compare to the nearby trial?
    Doc:
    seq_maps are a dictionary of the rewarding sequences of the task of each day
    """
    
    # first find change of mind triggered behavior log
    logs_tuple, logs_tuple_rand = find_triggered_log_animal(animal, dates_to_plot, proportion = proportion,  trialinfo = trialinfo)
    
    correct = {}
    correct_last = {} # the correctness of the last choice before CoM
    correct_rand = {}
    correct_wouldhave = {}
    
    for day in logs_tuple.keys():
        correct_day = []
        correct_last_day = []
        correct_wouldhave_day = []
        correct_rand_day = []
        
        seq_map = seq_maps[day]
        
        for session_ind in range(len(logs_tuple[day])):
            log_tuple = logs_tuple[day][session_ind]
            log_tuple_rand = logs_tuple_rand[day][session_ind]
            
            for trial_ind in range(len(log_tuple)):
                log_tuple_t = log_tuple[trial_ind] #for this trial
                log_tuple_rand_t = log_tuple_rand[trial_ind] #for this trial
                

                correct_t, correct_wouldhave_t = parse_to_correct(log_tuple_t, seq_map)
                correct_last_t = parse_to_last_correct(log_tuple_t, seq_map)
                correct_rand_t, _ = parse_to_correct(log_tuple_rand_t, seq_map, True)
                
                correct_last_day.append(correct_last_t)
                correct_day.append(correct_t)
                correct_wouldhave_day.append(correct_wouldhave_t)
                correct_rand_day.append(correct_rand_t)

        correct[day] = remove_nan(correct_day)
        correct_wouldhave[day] = remove_nan(correct_wouldhave_day)
        correct_rand[day] = remove_nan(correct_rand_day)
        correct_last[day] = remove_nan(correct_last_day)
    
    return correct, correct_rand, correct_wouldhave, correct_last
       