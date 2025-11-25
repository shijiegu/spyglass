import numpy as np
import pandas as pd
from scipy.stats import ranksums
import starbars
import seaborn as sns
import matplotlib.pyplot as plt
from spyglass.shijiegu.Analysis_SGU import ChangeofMindTheta
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

def get_GLM_elements_1(df, arm1, arm2):
    """find transitions from arm1 to arm2
    # find (arm1,arm2) transitions in df where we include trials
    #   type 1: amimal's final choice is arm 2, coming from arm 1 without a change of mind
    #   type 2: animal's initial choice is arm 2, with a change of mind, coming from arm 1
    # we output:
    # in between ith and jth arm1, arm2 transitions,
    # log a tuple (j-i, theta_dev ith, theta_dev jth, change_of_mind ith, reward ith)
    # the jth transition is not a change of mind.
    # If the jth transition is a change of mind, ith transition is discarded from analysis.
    # find change of mind trials in which arm 1 - arm 2 are aborted
    """
    # type 1
    initial_choice_trials = df[df.OuterWellIndex == arm2].index
    initial_choice_trials = initial_choice_trials[initial_choice_trials > 1]
    type1_trials = [t for t in initial_choice_trials if df.loc[t-1,'OuterWellIndex'] == arm1]
    type1_trials = np.array([t for t in type1_trials if df.loc[t,'change_of_mind'] == 0])
    
    # type 2
    initial_choice_trials = df[df.initial_choice == arm2].index
    initial_choice_trials = initial_choice_trials[initial_choice_trials > 1]
    previous_choice_trials = np.array([t for t in initial_choice_trials if df.loc[t-1,'OuterWellIndex'] == arm1])
    CoM_trials = np.intersect1d(initial_choice_trials, previous_choice_trials) 
    
    # merge type 1 + type 2
    both_trials = np.concatenate([type1_trials,CoM_trials])

    # find change of mind trials in which arm 1 - arm 2 are chosen
    potential_trials = df[df.OuterWellIndex == arm2].index
    potential_trials = potential_trials[potential_trials > 1]
    all_trials = np.array([t for t in potential_trials if df.loc[t-1,'OuterWellIndex'] == arm1])
    print("potential_trials", all_trials)
    
    # calculate delta t - the interval between transitions
    tuples = []
    for t in range(len(both_trials)):
        i = both_trials[t]
        if np.sum(all_trials > i) == 0: #no future visit
            continue
        j = all_trials[all_trials > i][0]
        if df.loc[j,'change_of_mind']:
            continue
        # make sure there is no significant data drop between ith and jth trial
        # TODO
    
        # add tuples
        tuples.append([i,
                      j-i,
                      df.loc[i,'theta_dev'],
                      df.loc[i,'change_of_mind'],
                      df.loc[i,'rewardNum']])
    return tuples

def get_GLM_elements_2(df, arm1, arm2):
    """find transitions from arm 1 to arm 2
    # find (arm1,arm2) transitions in df where we include trials
    #   type 1: animal's initial choice is arm 2, with a change of mind, coming from arm 1 with long theta
    #   type 2: animal's initial choice is arm 2, with a change of mind, coming from arm 1 without long theta
    # we output:
    # in between ith and jth arm1, arm2 transitions,
    # log a tuple (j-i, theta_dev ith, theta_dev jth, change_of_mind ith, reward ith)
    # the jth transition is not a change of mind.
    # If the jth transition is a change of mind, ith transition is discarded from analysis.
    # find change of mind trials in which arm 1 - arm 2 are aborted
    """
    # find change of mind trials in which arm 1 - arm 2 are aborted
    initial_choice_trials = df[df.initial_choice == arm2].index
    initial_choice_trials = initial_choice_trials[initial_choice_trials > 1]
    previous_choice_trials = np.array([t for t in initial_choice_trials if df.loc[t-1,'OuterWellIndex'] == arm1])
    CoM_trials = np.intersect1d(initial_choice_trials, previous_choice_trials) 
    print("CoM_trials", CoM_trials)

    # find change of mind trials in which arm 1 - arm 2 are chosen
    potential_trials = df[df.OuterWellIndex == arm2].index
    potential_trials = potential_trials[potential_trials > 1]
    all_trials = np.array([t for t in potential_trials if df.loc[t-1,'OuterWellIndex'] == arm1])
    print("potential_trials", all_trials)
    
    
    tuples = []
    for t in range(len(CoM_trials)):
        i = CoM_trials[t]
        if np.sum(all_trials > i) == 0: #no future visit
            continue
        j = all_trials[all_trials > i][0]
        if df.loc[j,'change_of_mind']:
            continue
        # make sure there is no significant data drop between ith and jth trial
        # TODO
    
        # add tuples
        tuples.append([i,
                      j-i,
                      df.loc[i,'theta_dev'],
                      df.loc[i,'change_of_mind'],
                      df.loc[i,'rewardNum']])
    return tuples

def get_GLM_elements_3(df, arm1, arm2):
    """find transitions from arm 1 to arm 2
    # find (arm1,arm2) transitions in df where we include trials
    #   type 1: amimal's final choice is arm 2, coming from arm 1 without a change of mind
    #   type 2: amimal's final choice is arm 2, coming from arm 1 with a change of mind
    # we output:
    # in between ith and jth arm1, arm2 transitions,
    # log a tuple (j-i, theta_dev ith, theta_dev jth, change_of_mind ith, reward ith)
    # the jth transition is not a change of mind.
    # If the jth transition is a change of mind, ith transition is discarded from analysis.
    # find change of mind trials in which arm 1 - arm 2 are aborted
    """
    
    potential_trials = df[df.OuterWellIndex == arm2].index
    potential_trials = potential_trials[potential_trials > 1]
    trials = [t for t in potential_trials if df.loc[t-1,'OuterWellIndex'] == arm1]
    
    tuples = []
    for t in range(len(trials) - 1):
        i = trials[t]
        j = trials[t + 1]
        # make sure there is no significant data drop between ith and jth trial
        # TODO
    
        # add tuples
        tuples.append([i,
                      j-i,
                      df.loc[i,'theta_dev'],
                      df.loc[i,'change_of_mind'],
                      df.loc[i,'rewardNum']])
    return tuples



def get_GLM_elements_day(animal, d, transitions,
                         proportion = 0.1, delta_t_minus = 0, delta_t_plus = 2, max_flag = 0,
                         type = "1"):  
    if type == "1":
        get_GLM_elements = get_GLM_elements_1
    elif type == "2":
        get_GLM_elements = get_GLM_elements_2
    elif type == "3":
        get_GLM_elements = get_GLM_elements_3
    tuples_transitions = {}
    for transition in transitions:
        tuples_transitions[transition] = []
    
    nwb_file_name = animal.lower() + d + '.nwb'
    nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
    session_interval, position_interval = runSessionNames(nwb_copy_file_name)
            
    for session_ind in range(len(session_interval)):
        session, pos_name = session_interval[session_ind], position_interval[session_ind]

        key = {"nwb_file_name": nwb_copy_file_name,
               "proportion":proportion, "epoch":session[:2],
               "delta_t_minus": delta_t_minus, "delta_t_plus": delta_t_plus, 
               "max_flag": max_flag}
        
        df = ChangeofMindTheta().fetch1_dataframe(key)

        for transition in tuples_transitions:
            (arm1, arm2) = transition
            GLM_tuple = get_GLM_elements(df, arm1, arm2)
            for t in GLM_tuple:
                t.append(nwb_copy_file_name)
                t.append(session)
                tuples_transitions[transition].append(t)
    return tuples_transitions

def tuples2pd(tuples_transitions, days_to_group):
    # concatenate across days for each transition
    # also produces data in the form of pandas dataframe
    
    tuples_transitions_concatenated = {}

    dates_to_plot = list(tuples_transitions.keys())
    transitions = list(tuples_transitions[dates_to_plot[0]].keys())
    
    for transition in transitions:
        tmp = []
        for d in days_to_group:
            tmp += tuples_transitions[d][transition]
        # Define column names
        columns = ['trials','trials_until_future',
                   'theta_deviation', 'is_change_of_mind',
                   'current_reward','nwb_file','session']
        
        # Create the DataFrame
        df_glm = pd.DataFrame(tmp, columns = columns)
    
        tuples_transitions_concatenated[transition] = df_glm
    return tuples_transitions_concatenated

def scatter_plot(transitions_2_animal, transitions_3_animal):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))
    ax = axes[0]
    for transition in transitions_2_animal:
        df_glm = transitions_2_animal[transition]
        data = df_glm[df_glm.is_change_of_mind]
        ax.scatter(np.abs(data.theta_deviation), data.trials_until_future, alpha = 0.8)
    ax.set_xlabel("absoulute value of theta_deviation")
    ax.set_ylabel("trials until future")
    ax.set_title("initial choice")

    ax = axes[1]
    for transition in transitions_3_animal:
        df_glm = transitions_3_animal[transition]
        data = df_glm[df_glm.is_change_of_mind]
        ax.scatter(np.abs(data.theta_deviation), data.trials_until_future, alpha = 0.8)
    ax.set_xlabel("absoulute value of theta_deviation")
    ax.set_ylabel("trials until future")
    ax.set_title("final choice")
    return fig, axes

def violin_plot(transitions_x_animal, transitions, seperate_by, axe, animal, title):
    # Figure left
    violin_dict = {}
    xticks = []

    if transitions is None:
        transitions = list(transitions_x_animal.keys())
        
    for ind in range(len(transitions)):
        transition = transitions[ind]
    
        df_glm = transitions_x_animal[transition]
        if seperate_by == "is_change_of_mind":
            data = df_glm[df_glm.is_change_of_mind]
            label1 = "ch.of.m."
        elif seperate_by == "theta_deviation":
            df_glm = df_glm[df_glm.is_change_of_mind]
            data = df_glm[np.abs(df_glm.theta_deviation) > 0]
            label1 = "long theta"
        
        if ind == 0:
            violin_dict[label1] = [data.trials_until_future]
        else:
            violin_dict[label1].append(data.trials_until_future)
            
        df_glm = transitions_x_animal[transition]
        if seperate_by == "is_change_of_mind":
            data = df_glm[df_glm.is_change_of_mind == 0]
            label2 = "not ch.of.m."
        elif seperate_by == "theta_deviation":
            df_glm = df_glm[df_glm.is_change_of_mind]
            data = df_glm[df_glm.theta_deviation == 0]
            label2 = "short theta"
        
        if ind == 0:
            violin_dict[label2] = [data.trials_until_future]
        else:
            violin_dict[label2].append(data.trials_until_future)

    violin_dict[label1] = np.concatenate(violin_dict[label1])
    violin_dict[label2] = np.concatenate(violin_dict[label2])

    xticks = [f"{label} \n N = {len(violin_dict[label])}" for label in violin_dict]
                            
    ax = sns.violinplot(violin_dict, width=0.5, ax = axe)
    _, p_value = ranksums(violin_dict[label1], violin_dict[label2])
    print("p_value",p_value)
    if p_value < 0.01:
        # adding statistical annotation
        annotations = [(label1, label2, np.round(p_value, 4))]
        starbars.draw_annotation(annotations)
    
    ax.set_xticklabels(xticks, rotation = 40)
    ax.set_ylabel("number of trials until future")
    ax.set_title(f"{animal} \n {title} \n # of trial until future")

    