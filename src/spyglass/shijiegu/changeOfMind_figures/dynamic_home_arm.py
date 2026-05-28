import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pickle
from spyglass.shijiegu.changeOfMind import color_by_rat
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.decodeHelpers import runSessionNames, session2position_name
from spyglass.shijiegu.Analysis_SGU import ChangeofMind, ChangeofMindTheta, ChangeofMindRemoteTheta
from spyglass.shijiegu.changeOfMind_figures.extra_correctness import (model2numbers,
                has_remote_theta, has_outer_remote_theta,
                has_home_remote_theta, has_long_theta)
from spyglass.shijiegu.changeOfMind_triggered_position import load_triggered_position_decode_session_spyglass
from spyglass.shijiegu.changeOfMind_figures.figure3_thetaGLM import check_interval_exists
from spyglass.shijiegu.changeOfMind_figures.figure4d import select_subset_helper_pd2, segment_boolean_series, setdiff1d_stable, unique_stable, find_future_arm
from spyglass.common.common_position import TrackGraph, IntervalLinearizedPosition, IntervalPositionInfo
from spyglass.shijiegu.changeOfMind_triggered import region
from spyglass.shijiegu.changeOfMind_figures.figure4d import load_theta_df
from spyglass.shijiegu.changeOfMind_figures.supp_thetacycle_concentration import theta_amplitude_to_cycle

parameter_name_long_theta = "params_both_max_segment_run_time_2_state"
parameter_name_remote = "params_both_max_run_time_2_state"
minimum_duration_long = 0.03
minimum_duration_remote = 0.02
min_posterior = 0.2
sd = 6
hpd = False

def return_remote_intervals(animal, list_of_days, remote_flag):
    # remote_flag or local_flag

    
    remote_interval_times = [] # tuples of (start_time, end_time)-the very first remote interval start time
    remote_interval_identities = []
    remote_max_distance = []
    remote_trial_infos = []
    remote_time_spent = []
    contain_long_theta = []
    
    for day_ind in range(len(list_of_days)):
        day = list_of_days[day_ind]
        
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
            
            if remote_flag and len(remote_df) == 0:
                continue
            if not remote_flag and len(long_df) == 0:
                continue
                
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
                
                # max proportion
                max_proportion = np.max(positions_in_arm[ind].linear_position)
                
                # contain long theta
                if len(long_df) > 0:
                    contain_long_theta_ = check_interval_exists(
                        long_df.loc[trialID].long_theta_intervals, t0, t1)
                else:
                    contain_long_theta_ = False

                if remote_flag:
                    continue_flag = not check_interval_exists(
                        remote_df.loc[trialID].remote_interval, t0, t1)
                else:
                    continue_flag = not check_interval_exists(
                        long_df.loc[trialID].long_theta_intervals, t0, t1)
                
                if continue_flag:
                    continue

                # remote theta interval
                if remote_flag:
                    remote_intervals, index = return_interval(remote_df.loc[trialID].remote_interval, t0, t1)
                    remote_com_id = trialID_count
                else:
                    remote_intervals, index = return_interval(long_df.loc[trialID].long_theta_interval, t0, t1)
                    remote_com_id = trialID_count
                    
                t0 = remote_intervals[0][0]
                remote_intervals = [remote_interval-t0 for remote_interval in remote_intervals]
                remote_interval_times.append(remote_intervals)
                    
                # remote content
                if remote_flag:
                    remote_content = np.array(remote_df.loc[trialID].remote_content)
                    remote_interval_identities.append(remote_content[index])
                    
                # remote max distance
                remote_max_distance.append(max_proportion)
                
                # time spent
                remote_time_spent.append(time_spent)
                
                # contain long theta
                contain_long_theta.append(contain_long_theta_)
                    
                # trial info
                remote_trial_infos.append((nwb_file_name, session_name, trialID))      
                    
    if not remote_flag:
        remote_interval_identities = None
    return remote_interval_times, remote_interval_identities, remote_max_distance, remote_time_spent, contain_long_theta,remote_trial_infos

seq2 = {1:2, 2:4, 3:1, 4:3}
rev2 = {2:1, 4:2, 1:3, 3:4}# if current arm is 1, future arm is 2; if current arm is 2, future arm is 1; if current arm is 3, future arm is 4; if current arm is 4, future arm is 3
def return_num_of_arms(animal, list_of_days, sd = sd):
    # remote_flag or local_flag

    if animal == "molly":
        seq = seq2
    elif animal == "eliot":
        seq = seq2
    elif animal == "klein":
        seq = rev2
    elif animal == "julio":
        seq = seq2
    elif animal == "lewis":
        seq = rev2
    num_of_arms = []
    contain_home = []
    contain_long = []
    remote_time_spent = []
    remote_trial_infos = []
    remote_arm_content = []
    rewards = []
    
    for day_ind in range(len(list_of_days)):
        day = list_of_days[day_ind]
        
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
            
            
            position1d = (IntervalLinearizedPosition() & {
                            'nwb_file_name':nwb_copy_file_name,
                            'interval_list_name':session2position_name(nwb_copy_file_name, session_name),
                            'track_graph_name': '4 arm lumped 2023',
                            'position_info_param_name':'default_decoding'}).fetch1_dataframe() #for debug use only
    
            trialID_last = -1
            for ind in range(len(trial_infos)):
                trialID = trial_infos[ind][0]
                current_arm = trial_infos[ind][-1]
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
            
                
                # find future arm the animal goes to
                last_reward = theta_df_subset.loc[trialID].past_reward
                if np.isnan(last_reward):
                    continue
                else:
                    last_reward = int(last_reward)
                if df.loc[trialID].CoMNum_by_arm == 1 and df.loc[trialID].CoMNum_by_time == 1:
                    reward = theta_df_subset.loc[trialID].rewardNum == 2
                else:
                    future = find_future_arm(t1, df.loc[trialID].timestamp_O, position1d, current_arm)
                    if future == -1: # this means statescript and camera data disagree, corrupt data
                        continue
                    reward = future == seq[last_reward]
                    
                # reward
                #reward = theta_df_subset.loc[trialID].rewardNum == 2
                rewards.append(reward)

                
                # contain long theta
                if len(long_df) > 0:
                    contain_long_theta = check_interval_exists(
                        long_df.loc[trialID].long_theta_intervals, t0, t1)
                else:
                    contain_long_theta = False

                if len(remote_df) > 0:
                    contain_remote_theta = check_interval_exists(
                        remote_df.loc[trialID].remote_interval, t0, t1)
                else:
                    contain_remote_theta = False

                # remote theta interval
                if contain_remote_theta:
                    remote_intervals, index = return_interval(remote_df.loc[trialID].remote_interval, t0, t1)
                    remote_content = np.array(remote_df.loc[trialID].remote_content)
                    remote_interval_identities = remote_content[index]
                    print("remote_interval_identities", remote_interval_identities)
                    remote_arm_content.append(remote_interval_identities)
                    if contain_long_theta:
                        num_of_arms.append(len(np.unique(remote_interval_identities)) + 1)
                    else:
                        num_of_arms.append(len(np.unique(remote_interval_identities)))
                    contain_home.append(np.isin(0, remote_interval_identities))
                else:
                    if contain_long_theta:
                        num_of_arms.append(1)
                    else:
                        num_of_arms.append(0)
                    remote_arm_content.append([]) # find out animal location during long theta interval
                    contain_home.append(False)
     
                # time spent
                remote_time_spent.append(time_spent)
                
                # contain long theta
                contain_long.append(contain_long_theta)
                    
                # trial info
                remote_trial_infos.append((nwb_file_name, session_name, trialID))      
                    
    return num_of_arms, remote_time_spent, contain_long, contain_home, remote_arm_content, remote_trial_infos, rewards
                
def return_intvl(intervals, theta_pd):
    N = 0
    for intvl in intervals: 
        theta_subset = theta_pd[(theta_pd.time >= intvl[0]) & (theta_pd.time <= intvl[-1])]
        cycle_times = theta_amplitude_to_cycle(theta_subset)
        N += len(cycle_times)
    return N

def return_remote_time(animal, list_of_days, sd = sd, theta_type = "mua"):
    # remote_flag or local_flag

    if animal == "molly":
        seq = seq2
    elif animal == "eliot":
        seq = seq2
    elif animal == "klein":
        seq = rev2
    elif animal == "julio":
        seq = seq2
    elif animal == "lewis":
        seq = rev2

    remote_time_spent = []
    remote_trial_infos = []
    rewards = []
    Ns_remote = []
    Ns_long = []
    long_time_spent = []
    
    for day_ind in range(len(list_of_days)):
        day = list_of_days[day_ind]
        
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
            key = {"nwb_file_name": nwb_copy_file_name,
                    "epoch": str(session_name[:2]),
                    "data_type": theta_type}
            theta_pd = load_theta_df(key, spyglass = True)
            
            q_long["epoch"] = int(session_name[:2])
            q_remote["epoch"] = int(session_name[:2])

            if len(ChangeofMindRemoteTheta() & q_remote) > 0:
                remote_df = ChangeofMindRemoteTheta().fetch1_dataframe(q_remote) # trials with remote theta for now
            else:
                remote_df = []
                
            if len(ChangeofMindTheta() & q_long) > 0:
                long_df = ChangeofMindTheta().fetch1_dataframe(q_long)         # trials with long theta
            
                
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
            
            
            position1d = (IntervalLinearizedPosition() & {
                            'nwb_file_name':nwb_copy_file_name,
                            'interval_list_name':session2position_name(nwb_copy_file_name, session_name),
                            'track_graph_name': '4 arm lumped 2023',
                            'position_info_param_name':'default_decoding'}).fetch1_dataframe() #for debug use only
    
            trialID_last = -1
            for ind in range(len(trial_infos)):
                trialID = trial_infos[ind][0]
                current_arm = trial_infos[ind][-1]
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
            
                
                # find future arm the animal goes to
                last_reward = theta_df_subset.loc[trialID].past_reward
                if np.isnan(last_reward):
                    continue
                else:
                    last_reward = int(last_reward)
                if df.loc[trialID].CoMNum_by_arm == 1 and df.loc[trialID].CoMNum_by_time == 1:
                    reward = theta_df_subset.loc[trialID].rewardNum == 2
                else:
                    future = find_future_arm(t1, df.loc[trialID].timestamp_O, position1d, current_arm)
                    if future == -1: # this means statescript and camera data disagree, corrupt data
                        continue
                    reward = future == seq[last_reward]
                    
                if len(remote_df) > 0:
                    contain_remote_theta = check_interval_exists(
                            remote_df.loc[trialID].remote_interval, t0, t1)
                else:
                    contain_remote_theta = False
                    
                if len(long_df) > 0:
                    contain_long_theta = check_interval_exists(
                        long_df.loc[trialID].long_theta_intervals, t0, t1)
                else:
                    contain_long_theta = False

                # remote theta interval
                if contain_remote_theta:
                    remote_intervals, index = return_interval(remote_df.loc[trialID].remote_interval, t0, t1)

                    # get total duration of remote_intervals
                    time_spent = np.sum([interval[1]-interval[0] for interval in remote_intervals])
                    
                    # get number of theta cycles
                    N_remote = return_intvl(remote_intervals, theta_pd)
                    Ns_remote.append(N_remote)
     
                    # time spent
                    remote_time_spent.append(time_spent)
                    
                    # trial info
                    remote_trial_infos.append((nwb_file_name, session_name, trialID))# reward
                    
                    #reward = theta_df_subset.loc[trialID].rewardNum == 2
                    rewards.append(reward)

                else:
                    rewards.append(reward)
                    Ns_remote.append(0)
                    remote_time_spent.append(0)
                    remote_trial_infos.append((nwb_file_name, session_name, trialID))
                
                if contain_long_theta:
                    long_intervals, _ = return_interval(long_df.loc[trialID].long_theta_intervals, t0, t1)
                    
                    # get total duration of long_intervals
                    time_spent_long = np.sum([interval[1]-interval[0] for interval in long_intervals])
                    
                    # get number of theta cycles
                    N_long = return_intvl(long_intervals, theta_pd)
                    Ns_long.append(N_long)
                    long_time_spent.append(time_spent_long)
                else:
                    Ns_long.append(0)
                    long_time_spent.append(0)
                      
                    
    return Ns_remote, remote_time_spent, remote_trial_infos, rewards, Ns_long, long_time_spent
    
    
    
def return_interval(intervals, t0, t1):
    results = []
    indeces = []
    for i, interval in enumerate(intervals):
        if interval[0] >= t0 and interval[1] <= t1:
            results.append(interval)
            indeces.append(i)
    return results, indeces

def violin_plot(remote_intervals, remote_interval_identities, remote_max_distance):
    # for each animal's data, plot the distribution of max distance for remote theta intervals that contain home vs those that only contain outer arm
    fig, axes = plt.subplots(1,1, figsize=(3.8, 2.5))
    delta = 0.3 #spacing between 2 violin plots
    
    for animal_ind, animal in enumerate(remote_max_distance.keys()):
        contain_home, _ = return_meta_data(remote_intervals[animal], remote_interval_identities[animal])
        data1 = np.array(remote_max_distance[animal])[contain_home]
        data2 = np.array(remote_max_distance[animal])[~contain_home]
        B1 = len(data1)
        B2 = len(data2)
    
        
        xs = np.array([0]) + (animal_ind)
        df = pd.DataFrame()
        df['wo home rep.'] = np.repeat(xs, B2)
        df['values'] = (data2.reshape((-1,1))).T.flatten()
        axes.scatter(df['wo home rep.'] + np.random.uniform(-0.05, 0.05, size=len(df)), df['values'], color='black', alpha=0.5, s=5)
        bp = axes.boxplot(df['values'], positions=xs, widths=delta, patch_artist=True,
                          medianprops=dict(color=color_by_rat[animal]),
                          whiskerprops=dict(linewidth=0),
                          capprops=dict(color=color_by_rat[animal]),
                          showfliers=False, showcaps=False)
        for patch in bp['boxes']:
            patch.set_facecolor('white')
            patch.set_edgecolor(color_by_rat[animal])
            patch.set_alpha(0.5)
        
        xs = np.array([delta]) + (animal_ind)
        df = pd.DataFrame()
        df['w home rep.'] = np.repeat(xs, B1)
        df['values'] = (data1.reshape((-1,1))).T.flatten()
        axes.scatter(df['w home rep.'] + np.random.uniform(-0.05, 0.05, size=len(df)), df['values'], color='black', alpha=0.5, s=5)
        bp = axes.boxplot(df['values'], positions=xs, widths=delta, patch_artist=True,
                          medianprops=dict(color=color_by_rat[animal]),
                          whiskerprops=dict(linewidth=0),
                          capprops=dict(color=color_by_rat[animal]),
                          showfliers=False, showcaps=False)
        for patch in bp['boxes']:
            patch.set_facecolor(color_by_rat[animal])
            patch.set_alpha(0.5)
        
    # remove top and right spines
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_xticks(np.arange(len(remote_max_distance.keys())) + delta/2)
    axes.set_xticklabels([f"Rat {key[0].upper()}" for key in remote_max_distance.keys()])
    axes.set_ylabel("time spent in outer arm (s)")
    axes.set_title("without | with home rep.")
    plt.tight_layout()
    plt.show()
    
    file_path = f"/home/shijiegu/Documents/spyglass/notebooks/Change of Mind Analysis/final_figures/figure3/home_rep_time.pdf"
    fig.savefig(file_path, format="pdf", bbox_inches="tight")
    # save figure as pdf, tight layout
    #fig.savefig("change_of_mind_remote_theta_max_distance_violin.pdf", format='pdf', bbox_inches='tight')



    # data = {}
    # data["reward"] = reward
    # data["remote_theta"] = remote_theta #np.array(remote_theta_home) + np.array(remote_theta_outer)
    # data["long_theta"] = long_theta
    # data["theta"] = theta
    # data["com_num"] = com_num
    # data["learning_flag"] = learning
    # data["max_distance"] = max_distance
    # data["remote_theta_home"] = remote_theta_home
    # data["remote_theta_outer"] = remote_theta_outer
    
    # output_path = get_savename(animal, output_folder, parameter_name_remote, q_long["local_parameter"], minimum_duration_remote, min_posterior, sd)
    # with open(output_path, 'wb') as file:
    #     pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    #     print(f"Data successfully pickled and saved.")
        
    # success[animal] = 1
    
def return_meta_data(remote_intervals, remote_interval_identities):
    contain_home = [] #boolean array
    delta_t = []
    for ind in range(len(remote_interval_identities)):
        intervals = remote_intervals[ind]
        identities = remote_interval_identities[ind]
        # find if identities include zero
        contain_home.append(np.isin(0, identities))

        # first outer arm - first home
        intervals_home = [intervals[ind][0] for ind in range(len(intervals)) if identities[ind] == 0]
        intervals_arm = [intervals[ind][0] for ind in range(len(intervals)) if identities[ind] != 0]
        if len(intervals_home) == 0 or len(intervals_arm)==0:
            continue
        delta_t.append(intervals_arm[0] - intervals_home[0])

    contain_home = np.array(contain_home)
    
    return contain_home, delta_t



def GLM_max_proportion(remote_max_distance, remote_intervals, remote_interval_identities):
    # GLM to predict max proportion based on whether remote theta contains home, or only outer arm
    
    animals = list(remote_max_distance.keys())
    GLM_xy = []
    for animal in animals:
        contain_home, _ = return_meta_data(remote_intervals[animal], remote_interval_identities[animal])
        max_distance = np.array(remote_max_distance[animal])
        
        animal_category = [a == animal for a in animals]
        for ind in range(len(contain_home)):
            GLM_xy.append(animal_category + [
                contain_home[ind], max_distance[ind]
            ])
        
    GLM_xy = np.array(GLM_xy)
    
    ## Model 1: no theta or with theta
    feature_dict = {f"Rat {animals[animal_ind][0].upper()}":GLM_xy[:,animal_ind] for animal_ind in range(len(animals))}
    feature_dict["contain home"] = GLM_xy[:,len(animals)]

    X = pd.DataFrame(feature_dict)
    X = sm.add_constant(X)
    y = GLM_xy[:,-1]

    """a) Mixed Linear Effect"""
    ols_model = sm.OLS(y,X)
    ols_result1 = ols_model.fit()
    
    return ols_model, ols_result1

def GLM_correctness1(contain_home_animals, num_animals, long_theta_animals, reward_animals):
    # GLM to predict max proportion based on whether remote theta contains home, or only outer arm
    
    animals = ["molly", "eliot", "klein", "julio", "lewis"]#list(contain_home_animals.keys())
    GLM_xy = []
    for animal in animals:
        contain_home = contain_home_animals[animal]
        num = np.array(num_animals[animal])
        rewards = reward_animals[animal]
        
        animal_category = [a == animal for a in animals]
        for ind in range(len(contain_home)):
            GLM_xy.append(animal_category + [
                int(num[ind] > 0), rewards[ind]
            ])
        
    GLM_xy = np.array(GLM_xy)
    
    ## Model 1: no theta or with theta
    feature_dict = {f"Rat {animals[animal_ind][0].upper()}":GLM_xy[:,animal_ind] for animal_ind in range(len(animals))}
    feature_dict["rep. of alternatives"] = GLM_xy[:,len(animals)]

    X = pd.DataFrame(feature_dict)
    X = sm.add_constant(X)
    y = GLM_xy[:,-1]

    """a) Mixed Linear Effect"""
    np.random.seed(2026)
    ols_model = sm.Logit(y,X)
    ols_result1 = ols_model.fit()#ols_model.fit_regularized(method='l1', alpha=0.01, L1_wt=0)#.fit(method='bfgs', maxiter=1000)
    
    return ols_model, ols_result1

def GLM_correctness_local(contain_home_animals, num_animals, long_theta_animals, reward_animals):
    # GLM to predict max proportion based on whether remote theta contains home, or only outer arm
    
    animals = ["molly", "eliot", "klein", "julio", "lewis"]#list(contain_home_animals.keys())
    GLM_xy = []
    for animal in animals:
        contain_home = contain_home_animals[animal]
        num = np.array(num_animals[animal])
        rewards = reward_animals[animal]
        long_theta = long_theta_animals[animal]
        
        animal_category = [a == animal for a in animals]
        for ind in range(len(contain_home)):
            GLM_xy.append(animal_category + [
                int(long_theta[ind] > 0), rewards[ind]
            ])
        
    GLM_xy = np.array(GLM_xy)
    
    ## Model 1: no theta or with theta
    feature_dict = {f"Rat {animals[animal_ind][0].upper()}":GLM_xy[:,animal_ind] for animal_ind in range(len(animals))}
    feature_dict["local extended"] = GLM_xy[:,len(animals)]

    X = pd.DataFrame(feature_dict)
    X = sm.add_constant(X)
    y = GLM_xy[:,-1]

    """a) Mixed Linear Effect"""
    np.random.seed(2026)
    ols_model = sm.Logit(y,X)
    ols_result1 = ols_model.fit()#ols_model.fit_regularized(method='l1', alpha=0.01, L1_wt=0)#.fit(method='bfgs', maxiter=1000)
    
    return ols_model, ols_result1

def GLM_correctness_remote(contain_home_animals, num_animals, long_theta_animals, reward_animals):
    # GLM to predict max proportion based on whether remote theta contains home, or only outer arm
    
    animals = ["molly", "eliot", "klein", "julio", "lewis"]#list(contain_home_animals.keys())
    GLM_xy = []
    for animal in animals:
        contain_home = contain_home_animals[animal]
        num = np.array(num_animals[animal])
        rewards = reward_animals[animal]
        long_theta = np.array(long_theta_animals[animal])
        num[long_theta] -= 1 # if long theta exists, add one more arm to the representation of alternatives
        
        
        animal_category = [a == animal for a in animals]
        for ind in range(len(contain_home)):
            GLM_xy.append(animal_category + [
                int(num[ind] > 0), rewards[ind]
            ])
        
    GLM_xy = np.array(GLM_xy)
    
    ## Model 1: no theta or with theta
    feature_dict = {f"Rat {animals[animal_ind][0].upper()}":GLM_xy[:,animal_ind] for animal_ind in range(len(animals))}
    feature_dict["remote"] = GLM_xy[:,len(animals)]

    X = pd.DataFrame(feature_dict)
    X = sm.add_constant(X)
    y = GLM_xy[:,-1]

    """a) Mixed Linear Effect"""
    np.random.seed(2026)
    ols_model = sm.Logit(y,X)
    ols_result1 = ols_model.fit()#ols_model.fit_regularized(method='l1', alpha=0.01, L1_wt=0)#.fit(method='bfgs', maxiter=1000)
    
    return ols_model, ols_result1

def GLM_correctness_time(time_spent_animals, label, reward_animals):
    # GLM to predict max proportion based on whether remote theta contains home, or only outer arm
    
    animals = ["molly", "eliot", "klein", "julio", "lewis"]#list(contain_home_animals.keys())
    GLM_xy = []
    for animal in animals:
        time_spent_animals_ = time_spent_animals[animal]
        rewards = reward_animals[animal]
        
        animal_category = [a == animal for a in animals]
        for ind in range(len(time_spent_animals_)):
            GLM_xy.append(animal_category + [
                time_spent_animals_[ind], rewards[ind]
            ])
        
    GLM_xy = np.array(GLM_xy)
    
    ## Model 1: no theta or with theta
    feature_dict = {f"Rat {animals[animal_ind][0].upper()}":GLM_xy[:,animal_ind] for animal_ind in range(len(animals))}
    feature_dict[label] = GLM_xy[:,len(animals)]

    X = pd.DataFrame(feature_dict)
    X = sm.add_constant(X)
    y = GLM_xy[:,-1]

    """a) Mixed Linear Effect"""
    np.random.seed(2026)
    ols_model = sm.Logit(y,X)
    ols_result1 = ols_model.fit()#ols_model.fit_regularized(method='l1', alpha=0.01, L1_wt=0)#.fit(method='bfgs', maxiter=1000)
    
    return ols_model, ols_result1


def GLM_arm_num_reward(contain_home_animals, num_animals, long_theta_animals, reward_animals):
    # GLM to predict max proportion based on whether remote theta contains home, or only outer arm
    
    animals = ["molly", "eliot", "klein", "julio", "lewis"]#list(contain_home_animals.keys())
    GLM_xy = []
    for animal in animals:
        contain_home = contain_home_animals[animal]
        num = np.array(num_animals[animal])
        rewards = reward_animals[animal]
        long_theta = long_theta_animals[animal]
        num[long_theta] -= 1
        
        animal_category = [a == animal for a in animals]
        for ind in range(len(contain_home)):
            if num[ind] > 0:
                GLM_xy.append(animal_category + [
                    num[ind], rewards[ind]
                ])
        
    GLM_xy = np.array(GLM_xy)
    
    ## Model 1: no theta or with theta
    feature_dict = {f"Rat {animals[animal_ind][0].upper()}":GLM_xy[:,animal_ind] for animal_ind in range(len(animals))}
    feature_dict["num of arms"] = GLM_xy[:,len(animals)]

    X = pd.DataFrame(feature_dict)
    X = sm.add_constant(X)
    y = GLM_xy[:,-1]

    """a) Mixed Linear Effect"""
    np.random.seed(2026)
    ols_model = sm.Logit(y,X)
    ols_result1 = ols_model.fit()#ols_model.fit_regularized(method='l1', alpha=0.01, L1_wt=0)#.fit(method='bfgs', maxiter=1000)
    
    return ols_model, ols_result1

def GLM_correctness3(contain_home_animals, num_animals, long_theta_animals, reward_animals):
    # GLM to predict max proportion based on whether remote theta contains home, or only outer arm
    
    animals = ["molly", "eliot", "klein", "julio", "lewis"]#list(contain_home_animals.keys())
    GLM_xy = []
    for animal in animals:
        contain_home = contain_home_animals[animal]
        num = np.array(num_animals[animal])
        long_theta = long_theta_animals[animal]
        rewards = reward_animals[animal]
        
        num[contain_home] -= 1 # home arm is not counted in the number of arms if it is represented in remote theta
        num[long_theta] -= 1 # if long theta exists, add one more arm to the number of arms
        
        animal_category = [a == animal for a in animals]
        for ind in range(len(contain_home)):
            GLM_xy.append(animal_category + [
                int(contain_home[ind]) * int((num[ind] == 0)) * int(long_theta[ind] == 0), #home alone
                int(contain_home[ind]) * int((num[ind] > 0)),
                int(contain_home[ind]) * int((long_theta[ind] > 0)),
                rewards[ind]
            ])
        
    GLM_xy = np.array(GLM_xy)
    
    ## Model 1: no theta or with theta
    feature_dict = {f"Rat {animals[animal_ind][0].upper()}":GLM_xy[:,animal_ind] for animal_ind in range(len(animals))}

    feature_dict["home alone"] = GLM_xy[:,len(animals)]
    feature_dict["home + other arms"] = GLM_xy[:,len(animals)+1]
    feature_dict["home + current arm"] = GLM_xy[:,len(animals)+2]

    X = pd.DataFrame(feature_dict)
    X = sm.add_constant(X)
    y = GLM_xy[:,-1]

    """a) Mixed Linear Effect"""
    np.random.seed(2026)
    ols_model = sm.Logit(y,X)
    ols_result1 = ols_model.fit()#.fit(method='bfgs', maxiter=1000)
    #ols_result1 = ols_model.fit_regularized(method='l1', alpha=0.01, L1_wt=0)#.fit(method='bfgs', maxiter=1000)
    
    return ols_model, ols_result1

def GLM_correctness2(contain_home_animals, num_animals, long_theta_animals, reward_animals):
    # GLM to predict max proportion based on whether remote theta contains home, or only outer arm
    
    animals = ["molly", "eliot", "klein", "julio", "lewis"]#list(contain_home_animals.keys())
    GLM_xy = []
    for animal in animals:
        contain_home = contain_home_animals[animal]
        num = np.array(num_animals[animal])
        long_theta = long_theta_animals[animal]
        rewards = reward_animals[animal]
        
        num[contain_home] -= 1 # home arm is not counted in the number of arms if it is represented in remote theta
        num[long_theta] -= 1 # if long theta exists, add one more arm to the number of arms
        
        animal_category = [a == animal for a in animals]
        for ind in range(len(contain_home)):
            GLM_xy.append(animal_category + [
                int(contain_home[ind]),
                int((long_theta[ind] > 0)),
                int((num[ind] > 0)),
                rewards[ind]
            ])
        
    GLM_xy = np.array(GLM_xy)
    
    feature_dict = {f"Rat {animals[animal_ind][0].upper()}":GLM_xy[:,animal_ind] for animal_ind in range(len(animals))}
    
    feature_dict["home"] = GLM_xy[:,len(animals)]
    feature_dict["local extended"] = GLM_xy[:,len(animals)+1]
    feature_dict["other outer arms"] = GLM_xy[:,len(animals)+2]
    

    X = pd.DataFrame(feature_dict)
    X = sm.add_constant(X)
    y = GLM_xy[:,-1]

    """a) Mixed Linear Effect"""
    np.random.seed(2026)
    ols_model = sm.Logit(y,X)
    ols_result1 = ols_model.fit()#.fit(method='bfgs', maxiter=1000)
    #ols_result1 = ols_model.fit_regularized(method='l1', alpha=0.01, L1_wt=0)#.fit(method='bfgs', maxiter=1000)
    
    return ols_model, ols_result1

def GLM_correctness_by_animal(contain_home_animals, num_animals, long_theta_animals, reward_animals):
    # GLM to predict max proportion based on whether remote theta contains home, or only outer arm
    
    animals = ["molly", "eliot", "klein", "julio", "lewis"]#list(contain_home_animals.keys())
    models = {}
    results = {}
    for animal in animals:
        GLM_xy = []
        contain_home = contain_home_animals[animal]
        num = np.array(num_animals[animal])
        long_theta = long_theta_animals[animal]
        rewards = reward_animals[animal]
        
        num[contain_home] -= 1 # home arm is not counted in the number of arms if it is represented in remote theta
        #num[long_theta] -= 1 # if long theta exists, add one more arm to the number of arms
        
        for ind in range(len(contain_home)):
            GLM_xy.append([
                contain_home[ind], long_theta[ind], num[ind] == 0, num[ind] == 1, num[ind] == 2, num[ind] >= 3, rewards[ind]
            ])
        
        GLM_xy = np.array(GLM_xy)
        
        ## Model 1: no theta or with theta
        feature_dict = {}
        feature_dict["contain home"] = GLM_xy[:,0]
        feature_dict["local extended "] = GLM_xy[:,1]
        feature_dict["0 outer arms"] = GLM_xy[:,2]
        feature_dict["1 outer arms"] = GLM_xy[:,3]
        feature_dict["2 outer arms"] = GLM_xy[:,4]
        feature_dict[">= 3 outer arms"] = GLM_xy[:,5]


        X = pd.DataFrame(feature_dict)
        X = sm.add_constant(X)
        y = GLM_xy[:,-1]

        """a) Mixed Linear Effect"""
        np.random.seed(2026)
        ols_model = sm.Logit(y,X)
        ols_result1 = ols_model.fit_regularized(method='l1', alpha=0.01, L1_wt=0)#.fit(method='bfgs', maxiter=1000)
    
        models[animal] = ols_model
        results[animal] = ols_result1
    return models, results

def correct_future_2_arm_rep(parsed_data, query = ["future_correct","past_reward"]):
    """_summary_

    Args:
        parsed_data (dict): field with 'past_reward', 'past', 'current', 'rep.', 'future', 'future_correct'
        returns:
            the probability of future_correct showing up in rep. when len(rep.) == 1 or when len(rep.) == 2
    """
    p_len1_animal = {}
    p_len2_animal = {}
    for animal in parsed_data.keys():
        p_len1 = []
        p_len2 = []
        data_animal = parsed_data[animal]
        for ind in range(len(data_animal)):
            data = data_animal[ind]

            rep = data['rep.']

            query_data = [data[q] for q in query]

            if len(rep) == 1 and len(query) == 1:
                p_len1.append(np.isin(query_data,rep))
            elif len(rep) == 2 and len(query) == 2:
                p_len2.append(np.all(np.isin(query_data,rep)))

        p_len1_animal[animal] = p_len1
        p_len2_animal[animal] = p_len2
    
    return p_len1_animal, p_len2_animal