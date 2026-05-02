import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import starbars
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.decodeHelpers import session2position_name, runSessionNames
import statsmodels.api as sm
from spyglass.shijiegu.changeOfMind_triggered_position import load_triggered_position_decode_session_spyglass

from spyglass.shijiegu.Analysis_SGU import ChangeofMind, ChangeofMindRemoteTheta, MUATheta, ChangeofMindTheta

same_side = {1:[1,2],
             2:[1,2],
             3:[3,4],
             4:[3,4]}

def return_deviation(animal, list_of_days, seq,
                     minimum_duration_long, parameter_name_long, sd, hpd,
                     minimum_duration_remote, parameter_name_remote, min_posterior):

    # For long theta
    # for each change of mind trial:
    # 
    # find its max proportion across all arms
    # find number of change of mind events by arms
    # find recent reward history
    # tally if there is a long theta event or there is remote event

    features = {"max_proportion":[], "num_com":[],
                "recent_4":[], "recent_1":[],
                "wouldbe_reward":[],
                "wouldbe_same_side":[],"time_spent":[]}
    
    responses = {"long_theta": [], "remote_theta": []}
    for day_ind in range(len(list_of_days)):
        day = list_of_days[day_ind]
        
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
            
        q_long = {"proportion": 0.1,
                 "minimum_duration":minimum_duration_long,
                  "parameter":parameter_name_long,
                  "local_parameter":f"dur_{minimum_duration_long}_sd_{sd}_hpd{hpd}"
                 }

        q_long["nwb_file_name"] = nwb_copy_file_name
        q_remote = q_long.copy()
        q_remote["parameter"] = parameter_name_remote
        q_remote["minimum_duration"] = minimum_duration_remote
        q_remote["remote_parameter"] = f"dur_{minimum_duration_remote}_sum_{min_posterior}" #f"parameter_name_remote
    
        for session_name in session_interval:
            q_long["epoch"] = int(session_name[:2])
            q_remote["epoch"] = int(session_name[:2])
            
            if len(ChangeofMindTheta() & q_long) > 0:
                long_df = ChangeofMindTheta().fetch1_dataframe(q_long)         # trials with long theta
            else:
                long_df = []

            if len(ChangeofMindRemoteTheta() & q_remote) > 0:
                remote_df = ChangeofMindRemoteTheta().fetch1_dataframe(q_remote) # trials with remote theta for now
            else:
                remote_df = []
                
            # load triggered position data
            loaded_data = load_triggered_position_decode_session_spyglass(nwb_copy_file_name, int(session_name[:2]),
                                                               "params_both_max_segment_run_time_2_state", 0.1)
            if len(loaded_data.keys()) == 0:
                continue
            
            trial_infos = loaded_data['triggered_trial_info']
            positions_in_arm = loaded_data["triggered_positions_baseoff"]
            
            # change of mind trials
            df = ChangeofMind().fetch1_dataframe(q_remote)
            theta_df_subset = df[df.change_of_mind]

            trialID_last = -1
            for ind in range(len(trial_infos)):
                trialID = trial_infos[ind][0]
                if trialID != trialID_last:
                    trialID_count = 1
                    trialID_last = trialID
                else:
                    trialID_count += 1
                
                # time spent
                time = positions_in_arm[ind].index
                t0 = time[0]
                t1 = time[-1]
                time_spent = t1 - t0
                
                
                # max_proportion
                #max_proportion = np.max([df.loc[trialID, "proportion_arm1"],df.loc[trialID, "proportion_arm2"],
                #        df.loc[trialID, "proportion_arm3"],df.loc[trialID, "proportion_arm4"]])
                max_proportion = np.max(positions_in_arm[ind].linear_position)
                
                
                # last choice
                last_choice = int(df.loc[trialID].past)
                if np.isnan(df.loc[trialID].past_reward):
                    continue
                last_reward = int(df.loc[trialID].past_reward)
                initial_choice = int(df.loc[trialID].initial_choice)
                
                if trialID<4:
                    continue
                
                if df.loc[trialID].CoMNum_by_arm > 1:
                    continue
                
                features["time_spent"].append(time_spent)
                features["max_proportion"].append(max_proportion)
                
                # find change of mind number on this trial
                features["num_com"].append(trialID_count) #df.loc[trialID].CoMNum_by_arm)

                # wouldbe_reward
                features["wouldbe_reward"].append(int(initial_choice == seq[last_reward]))
                
                
                
                # recent reward
                recent_4= np.mean(df.loc[trialID-4:trialID-1].rewardNum)
                features["recent_4"].append(recent_4)
                
                recent_1= np.mean(df.loc[trialID-1].rewardNum)
                features["recent_1"].append(recent_1)
                
                # wouldbe_same_side
                wouldbe_same_side = np.isin(initial_choice, same_side[last_choice])
                features["wouldbe_same_side"].append(int(wouldbe_same_side))
                
                if len(long_df) > 0:
                    long_boolean = check_interval_exists(
                        long_df.loc[trialID].long_theta_intervals, t0, t1)
                else:
                    long_boolean = 0
                
                if len(remote_df) > 0:
                    remote_boolean = check_interval_exists(
                        remote_df.loc[trialID].remote_interval, t0, t1)
                else:
                    remote_boolean = 0
                responses["long_theta"].append(long_boolean)
                responses["remote_theta"].append(remote_boolean)
    return features, responses

def check_interval_exists(intervals, t0, t1):
    for interval in intervals:
        if interval[0] >= t0 and interval[1] <= t1:
            return 1
    return 0

## if do not wish to plot rat baselines
def model2numbers(ols_result):
    coef_names = ols_result.params.keys()
    coef_est = np.array(ols_result.params)
    pvalues = ols_result.pvalues
    CI = ols_result.conf_int(alpha=0.05)
    #yerr = np.vstack((np.array(CI[0]).reshape((1,-1)), np.array(CI[1]).reshape((1,-1)))) # 2 x coefficients

    coef_names_subset_ind = ["Rat" not in name and "cons" not in name for name in coef_names]
    coef_names = np.array(coef_names)[coef_names_subset_ind]
    coef_est = coef_est[coef_names_subset_ind]
    pvalues = pvalues[coef_names_subset_ind]
    CI = CI.loc[coef_names]

    return ols_result.params, coef_names, coef_est, pvalues, CI#, yerr

def plot_params_one_axe(ax, ols_result, title):
    coef_pd, coef_names, coef_est, pvalues, CI = model2numbers(ols_result)
    coef_est = np.exp(coef_est)
    coef_pd = np.exp(coef_pd)
    CI = np.exp(CI)

    ax.bar(coef_names, coef_est, error_kw=dict(color='k'))
    for coef_name in coef_names:
        ax.plot([coef_name,coef_name],[CI.loc[coef_name][0], CI.loc[coef_name][1]], color = 'k')

    annotations = []
    for coef_ind in range(len(pvalues)):
        pvalue = pvalues[coef_ind]
        if pvalue < 0.05:
            annotations.append((coef_ind, coef_ind, np.round(pvalue, 4)))
    
    starbars.draw_annotation(annotations, ax = ax)

    ax.set_xticks(np.arange(len(coef_names)))
    ax.set_xticklabels(coef_names, rotation=45)
    ax.set_title(title)
    ax.spines[['right', 'top']].set_visible(False)
    
    ax.axhline(1,linewidth = 0.5,linestyle='dashed', color = 'k')
    ax.set_ylabel("Fold change")
    ax.set_yscale('log') 

    
def make_mixedGLM_xy(animals, features_animals, responses_animals, feature_names):
    GLM_xy = []
    for animal in animals:
        features = features_animals[animal]
        responses = responses_animals[animal]
        x = np.hstack(
            [np.array(features[feature]).reshape((-1,1)) for feature in feature_names]
            )
        y_long = responses["long_theta"]
        y_remote = responses["remote_theta"]

        animal_category = [a == animal for a in animals]
        N = x.shape[0]
        for ind in range(N):
            feature_i = x[ind].tolist()
            GLM_xy.append(animal_category + feature_i + [y_long[ind], y_remote[ind]])

    GLM_xy = np.array(GLM_xy)
    return GLM_xy

def do_GLM(animals, GLM_xy, feature_names):
    ## Model 1: no theta or with theta
    feature_dict = {f"Rat {animals[animal_ind][0].upper()}":GLM_xy[:,animal_ind] for animal_ind in range(len(animals))}
    ind = 0
    for ind in range(len(feature_names)):
        name = feature_names[ind]
        feature_dict[name] = GLM_xy[:, (len(animals) + ind)]

    X = pd.DataFrame(feature_dict)
    y1 = GLM_xy[:,-2] #long
    y2 = GLM_xy[:,-1] #remote

    """a) Mixed Linear Effect"""
    ols_model = sm.Logit(y1,X)
    ols_result_long = ols_model.fit()
    
    ols_model = sm.Logit(y2,X)
    ols_result_remote = ols_model.fit()

    print("Mixed Logistic Effect \n",ols_result_long.summary())
    
    print("Mixed Logistic Effect \n",ols_result_remote.summary())
    
    return ols_result_long, ols_result_remote