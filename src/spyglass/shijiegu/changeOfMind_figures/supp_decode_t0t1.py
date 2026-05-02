import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import wilcoxon



from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.Analysis_SGU import ChangeofMindTheta, ChangeofMindRemoteTheta

decode_name_long = "params_both_max_segment_run_time_2_state"
decode_name_remote = "params_both_max_run_time_2_state"
minimum_duration_long = 0.03
minimum_duration_remote = 0.02
min_posterior = 0.2
sd = 6
proportion = 0.1
hpd = False

local_parameter = f"dur_{minimum_duration_long}_sd_{sd}_hpd{hpd}"
remote_parameter = f"dur_{minimum_duration_remote}_sum_{min_posterior}"


def get_t0t1_animal(animal, list_of_days):
    # trials with either long theta or remote theta
    
    # loop through all trials
    
    long_t0 = []
    remote_t0 = []
    
    for day_ind in range(len(list_of_days)):
        day = list_of_days[day_ind]
        
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
            
        q_long = {"proportion": proportion,
                  "parameter":decode_name_long,
                 "local_parameter":local_parameter,
                 }

        q_long["nwb_file_name"] = nwb_copy_file_name
        q_remote = q_long.copy()
        q_remote["parameter"] = decode_name_remote
        q_remote["remote_parameter"] = remote_parameter
    
        for session_name in session_interval:
            q_long["epoch"] = int(session_name[:2])
            q_remote["epoch"] = int(session_name[:2])
            
            if len(ChangeofMindTheta() & q_long) == 0:
                continue
                
            if len(ChangeofMindRemoteTheta() & q_remote) == 0:
                continue
            long_df = ChangeofMindTheta().fetch1_dataframe(q_long)         # trials with long theta
            remote_df = ChangeofMindRemoteTheta().fetch1_dataframe(q_remote) # trials with remote theta for now
    
            # long trials
            long_df = long_df[long_df.long_theta]
            for trialID in long_df.index:
                long_intervals = long_df.loc[trialID].long_theta_intervals
                com_num1 = long_df.loc[trialID].CoMNum_by_time
                com_num2 = long_df.loc[trialID].CoMNum_by_arm
                if com_num1 > 1 or com_num2 > 1:
                    continue
                t0 = long_df.loc[trialID].initial_time
                for intvl_ind in range(len(long_intervals)):
                    intvl = long_intervals[intvl_ind]
                    long_t0.append(intvl[0] - t0)

            # remote event
            remote_df = remote_df[remote_df.has_remote_interval]
            for trialID in remote_df.index:
                remote_intervals = remote_df.loc[trialID].remote_interval
                com_num1 = remote_df.loc[trialID].CoMNum_by_time
                com_num2 = remote_df.loc[trialID].CoMNum_by_arm
                if com_num1 > 1 or com_num2 > 1:
                    continue
                t0 = remote_df.loc[trialID].initial_time
                for intvl_ind in range(len(remote_intervals)):
                    intvl = remote_intervals[intvl_ind]
                    remote_t0.append(intvl[0] - t0)
        
    return long_t0, remote_t0

def plot_histogram_t0t1(animal, long_t0, remote_t0, output_folder = None):

    # Perform the Wilcoxon Signed-Rank Test
    # Null hypothesis (H0): the median of the distribution is <= 0
    # Alternative hypothesis (H1): the median of the distribution is > 0
    result_long = wilcoxon(long_t0, alternative='greater')
    result_remote = wilcoxon(remote_t0, alternative='greater')
    print(f"P-value for local extended content: {result_long.pvalue}")
    print(f"P-value for remote content: {result_remote.pvalue}")

    plt.figure(figsize=(3,2))
    plt.hist(long_t0, bins = np.arange(-2, 5, 0.5),
             weights = np.ones(len(long_t0)) / len(long_t0),
             alpha = 0.5, label = "local extended")
    plt.hist(remote_t0, bins = np.arange(-2, 5, 0.5),
             weights = np.ones(len(remote_t0)) / len(remote_t0),
             alpha = 0.5, label = "remote")
    plt.xlabel("time from initial stopping (s)")
    plt.ylabel("number of trials")
    plt.gca().axvline(0, color = "k", linestyle = "--")
    # remove upper and right spines    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # set x limit
    plt.gca().set_xlim(-2, 4)
    plt.gca().set_xticks(np.arange(-2, 5, 1))
    
    # place legend outside of the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f"Rat {animal[0].upper()} \n event time relative to initial stopping")
    # save figure
    if output_folder is not None:
        plt.savefig(f"{output_folder}/t0t1_histogram_{animal}.pdf", bbox_inches='tight')
        
    return result_long.pvalue, result_remote.pvalue
