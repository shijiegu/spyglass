from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.Analysis_SGU import ChangeofMindTheta, ChangeofMindRemoteTheta
import numpy as np

success = {}

parameter_name_long_theta = "params_both_max_segment_run_time_2_state"
parameter_name_remote = "params_both_max_run_time_2_state"#"params_both_max_run_time_2_state" #"params_both_max_all_maze_2_state"  #"params_both_max_run_time_2_state"

def count_trials(animal, list_of_days,
                 decode_name_long, parsing_name_long,
                 decode_name_remote, parsing_name_remote, proportion = 0.1):
    # minimum_duration: minimum duration in seconds

    reward = []
    theta_either, theta_both, theta_long, theta_remote, com_num = [], [], [], [], []
    
    for day_ind in range(len(list_of_days)):
        day = list_of_days[day_ind]
        
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
            
        q_long = {"proportion": proportion,
                  "parameter":decode_name_long,
                 "local_parameter":parsing_name_long,
                 }

        q_long["nwb_file_name"] = nwb_copy_file_name
        q_remote = q_long.copy()
        q_remote["parameter"] = decode_name_remote
        q_remote["remote_parameter"] = parsing_name_remote
        
        theta_either_day, theta_both_day, theta_long_day, theta_remote_day, com_num_day = 0, 0, 0, 0, 0
    
        for session_name in session_interval:
            q_long["epoch"] = int(session_name[:2])
            q_remote["epoch"] = int(session_name[:2])
            
            if len(ChangeofMindTheta() & q_long) == 0:
                continue
                
            long_df = ChangeofMindTheta().fetch1_dataframe(q_long)         # trials with long theta
            if len(ChangeofMindRemoteTheta() & q_remote) == 0:
                continue
            remote_df = ChangeofMindRemoteTheta().fetch1_dataframe(q_remote) # trials with remote theta for now
            
            com_num_day += len(long_df[long_df.change_of_mind])
            
            trialIDs_long = list(long_df[long_df.long_theta].index)
            trialsIDs_remote = list(remote_df[remote_df.has_remote_interval].index)
            trialIDs_both = np.intersect1d(trialIDs_long, trialsIDs_remote)
            trialIDs_either = np.unique(trialIDs_long + trialsIDs_remote)
            
            theta_long_day += len(long_df[long_df.long_theta])
            theta_remote_day += len(remote_df[remote_df.has_remote_interval])
            theta_both_day += len(trialIDs_both)
            theta_either_day += len(trialIDs_either)
        
        theta_long.append(theta_long_day)
        theta_remote.append(theta_remote_day)
        theta_both.append(theta_both_day)
        theta_either.append(theta_either_day)
        com_num.append(com_num_day)
        
        theta_long_ratio = np.array(theta_long)/np.array(com_num)
        theta_remote_ratio = np.array(theta_remote)/np.array(com_num)
        theta_both_ratio = np.array(theta_both)/np.array(com_num)
        theta_either_ratio = np.array(theta_either)/np.array(com_num)
        
        theta_long_ratio = np.nan_to_num(theta_long_ratio)
        theta_remote_ratio = np.nan_to_num(theta_remote_ratio)
        theta_both_ratio = np.nan_to_num(theta_both_ratio)
        theta_either_ratio = np.nan_to_num(theta_either_ratio)
        
    return theta_long_ratio, theta_remote_ratio, theta_either_ratio, theta_both_ratio, com_num
        

            
    # the 3 things in return are
           # proportion of all change of mind trials with long theta 
           # proportion of all change of mind trials with remote theta
           # proportion of remote theta trials with long theta
    
    # save as a pickle file
    
    return long_theta_ratio, remote_ratio, intersection_ratio_remote, either_ratio, intersection_ratio, session_trial_union