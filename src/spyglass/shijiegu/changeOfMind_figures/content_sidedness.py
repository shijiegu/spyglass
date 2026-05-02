import numpy as np
feature_names = ["previous", "previously_rewarded", "future", "future_correct","same side","switch side"]
from spyglass.shijiegu.Analysis_SGU import ChangeofMind

same_side_map = {1:[1,2],2:[1,2],3:[3,4],4:[3,4]} 
switch_side_map = {1:[3,4],2:[3,4],3:[1,2],4:[1,2]} 

seq2={1:3, 3:4, 4:2, 2:1}#[1,3,4,2]
rev2={1:2, 2:4, 4:3, 3:1}
seqs={"seq2": seq2, "rev2": rev2}

def sideswitching_trial(previous_choice,
                        previously_rewarded,
                        initial_choice,
                        final_choice,
                        final_choice_correct):
    if not previously_rewarded:
        return 0
    if np.isin(initial_choice, switch_side_map[previous_choice]
               ) and np.isin(final_choice_correct, same_side_map[previous_choice]):
        if final_choice == final_choice_correct:
            return 1
        else:
            return 2
    return 0

def get_sidedness(features_all, response_all, tally_dict, seq_name):
    seq = seqs[seq_name]

    Y_correct = []
    Y_incorrect = []
    for trial in range(len(features_all)):
        nwb_file_name, sessionName, trialID, current_arm, reward = tally_dict[trial]
        
        # restricting to 1-change-of-mind trials
        if not np.all(np.isnan(current_arm)):
            continue
        
        # load trial info
        key = {"nwb_file_name": nwb_file_name,
               "epoch": int(sessionName[:2]),
               "proportion": 0.1}
        query = ChangeofMind() & key
        if len(query) == 0:
            continue
        df = ChangeofMind().fetch1_dataframe(key)
        
        # see if this trial is a side-switch corrected trial
        previous_choice = int(df.loc[trialID, "past"])
        previously_rewarded = int(df.loc[trialID, "past_reward"])
        initial_choice = int(df.loc[trialID, "initial_choice"])
        final_choice = int(df.loc[trialID, "OuterWellIndex"])
        final_choice_correct = int(seq[previously_rewarded])
        switch_type = sideswitching_trial(previous_choice,
                        previously_rewarded,
                        initial_choice,
                        final_choice,
                        final_choice_correct)     
        
        
        x = features_all[trial] > 0
        y = np.array(response_all[trial])

        y_sameside = np.sum(y[x[:,4]]) > 0 #(np.sum(y[x[:,5]]) + 1) / (np.sum(y[x[:,4]]) + 1) #  # contain rep of previous arm

        if switch_type == 1:
            Y_correct.append(y_sameside)
        elif switch_type == 2:
            Y_incorrect.append(y_sameside)

    Y_correct = np.array(Y_correct).astype("int")
    Y_incorrect = np.array(Y_incorrect).astype("int")
    return Y_correct, Y_incorrect