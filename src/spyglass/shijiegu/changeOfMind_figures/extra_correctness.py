import os
import numpy as np
import pickle
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.Analysis_SGU import ChangeofMind, ChangeofMindTheta, ChangeofMindRemoteTheta


def has_remote_theta(df,trialID):
    if len(df) == 0:
        return False
    remote_interval = df.loc[trialID,'remote_interval']
    initial_stopping = df.loc[trialID,'initial_time']
                
    post_ind = [ind for ind in range(len(remote_interval)) if remote_interval[ind][0] >= initial_stopping]
    
    if len(post_ind) > 0:
        return True
    else:
        return False
    
def has_outer_remote_theta(df,trialID):
    if len(df) == 0:
        return False
    remote_interval = df.loc[trialID,'remote_interval']
    remote_content = df.loc[trialID,'remote_content']
    initial_stopping = df.loc[trialID,'initial_time']
                
    post_ind = [ind for ind in range(len(remote_interval)) if remote_interval[ind][0] >= initial_stopping]
    if len(post_ind) == 0:
        return False
        
    remote_content = [remote_content[ind] for ind in post_ind]

    if len(np.setdiff1d(remote_content, [0])) > 0:
        return True
    else:
        return False
    
def has_home_remote_theta(df,trialID):
    if len(df) == 0:
        return False
    remote_interval = df.loc[trialID,'remote_interval']
    remote_content = df.loc[trialID,'remote_content']
    initial_stopping = df.loc[trialID,'initial_time']
                
    post_ind = [ind for ind in range(len(remote_interval)) if remote_interval[ind][0] >= initial_stopping]
    if len(post_ind) == 0:
        return False
        
    remote_content = [remote_content[ind] for ind in post_ind]

    if np.isin(0,remote_content):
        return True
    else:
        return False
    
def has_long_theta(df,trialID):
    if len(df) == 0:
        return False
    long_interval = df.loc[trialID,'long_theta_intervals']
    initial_stopping = df.loc[trialID,'initial_time']
                
    post_ind = [ind for ind in range(len(long_interval)) if long_interval[ind][0] >= initial_stopping]
    
    if len(post_ind) > 0:
        return True
    else:
        return False
    
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

def get_savename(animal, output_folder, parameter_name_remote, minimum_duration_long, minimum_duration_remote, min_posterior,sd):
    output_path = os.path.join(output_folder,f"{animal}_{parameter_name_remote}_{minimum_duration_long}_{minimum_duration_remote}_{min_posterior}_sd{sd}_figure4Fa") #os.join(output_folder,plot_data_filename)
    return output_path

def figure4_correctness(
    output_folder,
    list_of_days_animals, learning_flag,
    parameter_name_long_theta = "params_both_max_segment_run_time_2_state",
    parameter_name_remote = "params_both_max_run_time_2_state",
    minimum_duration_long = 0.03,
    minimum_duration_remote = 0.02,
    min_posterior = 0.2,
    sd = 6,
    hpd = False):
    
    success = {}
    
    for animal in ["molly","klein","eliot","julio","lewis"]:
        # trials with either long theta or remote theta
        
        # loop through all trials
        list_of_days = list_of_days_animals[animal]
        learning_flag_days = learning_flag[animal]
        
        reward = []
        theta = []
        long_theta = []
        remote_theta = []
        remote_theta_outer = []
        remote_theta_home = []
        com_num = []
        learning = []
        max_distance = []
    
        for day_ind in range(len(list_of_days)):
            day = list_of_days[day_ind]
            learning_flag_day = learning_flag_days[day_ind]
            
            nwb_file_name = animal.lower() + day + '.nwb'
            nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
            print(nwb_copy_file_name)
            session_interval, position_interval = runSessionNames(nwb_copy_file_name)
                
            q_long = {"proportion": 0.1,
                    "minimum_duration":minimum_duration_long,
                    "parameter":parameter_name_long_theta,
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
        
                # change of mind trials
                df = ChangeofMind().fetch1_dataframe(q_remote)
                theta_df_subset = df[df.change_of_mind]
        
                for trialID in theta_df_subset.index:
                    # check if a trial has multiple change-of-mind
                    # if df.loc[trialID].CoMNum_by_arm > 1 or df.loc[trialID].CoMNum_by_time > 1:
                    #     continue
                        
                    # find change of mind number on this trial
                    com_num.append(df.loc[trialID].CoMNum_by_arm)

                    # find other information
                    reward.append(df.loc[trialID].rewardNum == 2)
                    
                    
                    if has_remote_theta(remote_df,trialID):
                        print("remote_content",remote_df.loc[trialID].remote_content)
                        remote_flag = True
                        remote_theta.append(1)
                    else:
                        remote_flag = False
                        remote_theta.append(0)

                    if has_outer_remote_theta(remote_df,trialID):
                        remote_theta_outer.append(1)
                    else:
                        remote_theta_outer.append(0)

                    if has_long_theta(long_df,trialID):
                        long_theta.append(1)
                    else:
                        long_theta.append(0)

                    if remote_flag or long_theta[-1] == 1:
                        theta.append(1)
                    else:
                        theta.append(0)

                    if has_home_remote_theta(remote_df,trialID):
                        remote_theta_home.append(1)
                    else:
                        remote_theta_home.append(0)

                    learning.append(learning_flag_day)
                    max_distance.append(df.loc[trialID].CoMMaxProportion)


        data = {}
        data["reward"] = reward
        data["remote_theta"] = remote_theta #np.array(remote_theta_home) + np.array(remote_theta_outer)
        data["long_theta"] = long_theta
        data["theta"] = theta
        data["com_num"] = com_num
        data["learning_flag"] = learning
        data["max_distance"] = max_distance
        data["remote_theta_home"] = remote_theta_home
        data["remote_theta_outer"] = remote_theta_outer
    
        output_path = get_savename(animal, output_folder, parameter_name_remote, q_long["local_parameter"], minimum_duration_remote, min_posterior, sd)
        with open(output_path, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Data successfully pickled and saved.")
        
        success[animal] = 1
    return success
