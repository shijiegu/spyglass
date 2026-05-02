import pandas as pd
import numpy as np
import xarray as xr
from scipy import stats
from scipy import linalg
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from spyglass.shijiegu.changeOfMind import color_by_rat as color_by_animal
from spyglass.shijiegu.Analysis_SGU import get_linearization_map, Imu
from position_tools import (
    get_angle,
    get_distance,
    get_speed,
    get_velocity,
    interpolate_nan,
)

output_folder = '/stelmo/shijie/change_of_mind_analysis/figure2/'
triggered_decode_parameter_name = "params_both_max_run_time_2_state"

def return_time_under_threshold(animal, t0, t1, speed_threshold, imu_name = "big_acc_bias"):
    data_filename = f"triggered_theta_{animal}_{triggered_decode_parameter_name}.pkl"
    output_path = os.path.join(output_folder,data_filename) #os.join(output_folder,plot_data_filename)

    with open(output_path, 'rb') as file:
        # Deserialize the data and assign it to a variable
        data = pickle.load(file)
        
    # for each trial, restrict to the time window between t0 and t1, and calculate the proportion of time points where speed is below the threshold
    duration, all_speed = [], []
    
    # restrict to the time window between t0 and t1
    for ind in range(len(data["all_theta_days"])):
        trial_times = np.array(data['all_theta_days'][ind].time) # shape (num_trials, num_time_points)
        unix_times = np.array(data['all_theta_days'][ind].index)
        nwb_copy_file_name, session_name, trialID = data['all_trial_info'][ind]
        
        unix_time_t0 = unix_times[np.argwhere(trial_times >= t0).ravel()[0]]
        unix_time_t1 = unix_times[np.argwhere(trial_times <= t1).ravel()[-1]]
        # find the change-of-mind time in unix time
        # it is the unix time corresponding to the trial time closest to 0
        com_t = unix_times[np.argmin(np.abs(trial_times))]
        
        key_imu={'nwb_file_name':nwb_copy_file_name,
                  'epoch':int(session_name[:2]),
                  'trial':trialID,
                  "parameter":imu_name}
        q = Imu() & key_imu
        if len(q) == 0:
            continue
        postion_info_gyro = Imu().fetch1_dataframe(key_imu)
        
        # restrict to the time window between t0 and t1

        postion_info_gyro_subset = postion_info_gyro[np.logical_and(postion_info_gyro.index>=unix_time_t0,
                                                   postion_info_gyro.index<=unix_time_t1)]
        
        head_speed_subset = get_speed(
                  np.array(postion_info_gyro_subset)[:,[0,1]],
                  postion_info_gyro_subset.index,
                  sigma=0.001,
                  sampling_frequency=1/np.median(np.diff(np.array(postion_info_gyro_subset.index))),
            )
        
        dt = np.median(np.diff(postion_info_gyro_subset.index))
        
        restricted_speeds = head_speed_subset[~np.isnan(head_speed_subset)]
        restricted_speeds = restricted_speeds[restricted_speeds < speed_threshold]
        
        # make a pandas dataframe for the speed, with unix time - change-of-mind time as the index, and speed as the column
        speed_df = pd.DataFrame(
            {'speed': head_speed_subset}, index=postion_info_gyro_subset.index - com_t)
        all_speed.append(speed_df)
        
        if len(restricted_speeds) == 0:
            duration.append(0)
            continue

        duration_below_threshold = len(restricted_speeds) * dt #restricted_speeds_time[-1] - restricted_speeds_time[0]

        duration.append(duration_below_threshold)
    
    return duration, all_speed

def return_average_head_angular(animal, t0, t1, imu_name = "big_acc_bias"):
    # returns 
    # 1. the average head angular speed for each trial, restricted to the time window the animal is in the outer arms
    # 2. the integrated head angular speed for each trial, restricted to the time window the animal is in the outer arms
    
    data_filename = f"triggered_theta_{animal}_{triggered_decode_parameter_name}.pkl"
    output_path = os.path.join(output_folder,data_filename) #os.join(output_folder,plot_data_filename)

    with open(output_path, 'rb') as file:
        # Deserialize the data and assign it to a variable
        data = pickle.load(file)
        
    # for each trial, restrict to the time window between t0 and t1, and calculate the proportion of time points where speed is below the threshold
    averages = []
    integrated = []
    angles = []
    
    # restrict to the time window between t0 and t1
    for ind in range(len(data["all_theta_days"])):
        
        trial_times = np.array(data['all_theta_days'][ind].time) # shape (num_trials, num_time_points)
        unix_times = np.array(data['all_theta_days'][ind].index)
        nwb_copy_file_name, session_name, trialID = data['all_trial_info'][ind]
        
        unix_time_t0 = unix_times[np.argwhere(trial_times >= t0).ravel()[0]]
        unix_time_t1 = unix_times[np.argwhere(trial_times <= t1).ravel()[-1]]
        
        key_imu={'nwb_file_name':nwb_copy_file_name,
                  'epoch':int(session_name[:2]),
                  'trial':trialID,
                  "parameter":imu_name}
        q = Imu() & key_imu
        if len(q) == 0:
            continue
        postion_info_gyro = Imu().fetch1_dataframe(key_imu)
        
        postion_info_gyro = postion_info_gyro[np.logical_and(postion_info_gyro.index>=unix_time_t0,
                                                   postion_info_gyro.index<=unix_time_t1)]
        
        # restrict to the time window between t0 and t1
        #trial_speeds = np.diff(np.array(postion_info_gyro.head_orientation)) *
        trial_speeds = get_velocity(
            np.array(postion_info_gyro.head_orientation),
            postion_info_gyro.index,
            sigma=0.001,
            sampling_frequency=1/np.median(np.diff(np.array(postion_info_gyro.index))))
  
        trial_speeds = trial_speeds[~np.isnan(trial_speeds)]
        
        if len(trial_speeds) == 0:
            averages.append(0)
            continue
        
        # average angular speed
        average_angular_speed = np.mean(trial_speeds)
        averages.append(average_angular_speed)
        
        # idphi
        dt = np.median(np.diff(postion_info_gyro.index))
        #idphi = np.sum(np.abs(np.diff(np.array(postion_info_gyro.head_orientation)) ))
        idphi = np.sum(np.abs(trial_speeds)) * dt
        integrated.append(idphi)
        
        angles.append(postion_info_gyro)
    
    return averages, integrated, angles

def plot_speed_over_time(animal, all_speed, t0, t1, output_folder):
    # plot the speed, restricted to the time window between t0 and t1, for each trial, with the x-axis as time relative to the change-of-mind time, and the y-axis as speed
    plt.figure(figsize=(3,3))
    speed = all_speed[animal]

    time_bins = np.arange(t0, t1, 0.06) # for finding average speed across trials for each time point
    mean_trace = np.zeros_like(time_bins)
    count_trace = np.zeros_like(time_bins)
    for trial in speed:
        # restrict to the time window between t0 and t1
        data = trial[(trial.index >= t0) & (trial.index <= t1)]
        plt.plot(data.index, data['speed'], color=color_by_animal[animal], alpha=0.2)
        # add the speed to the mean trace
        bin_indices = np.digitize(data.index, time_bins) - 1
        for i in range(len(time_bins)):
            mean_trace[i] += np.nansum(data['speed'][bin_indices == i])
            count_trace[i] += np.sum(bin_indices == i)
            
    mean_trace = mean_trace / count_trace
    plt.plot(time_bins, mean_trace, color="k", alpha=0.5, linewidth=3)
    
    plt.xlabel('Time from change-of-mind (s)')
    plt.ylabel('Head speed (cm/s)')
    plt.title(f'Rat {animal[0].upper()}\n speed around change-of-mind')
    # remove top and right spines
    plt.gca().set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # save the figure
    plt.tight_layout()
    print(os.path.join(output_folder, f"speed_zoom_in_{animal}.pdf"))
    plt.savefig(os.path.join(output_folder, f"speed_zoom_in_{animal}.pdf"), dpi=300)
    #plt.show()  
               

def plot_duration_under_threshold(duration_animal, speed_threshold, output_folder):
    # plot the distribution of duration under threshold for each animal
    # use seaborn violin plot to show the distribution of duration under threshold for each animal
    plt.figure(figsize=(3,3))
    animal_ind = 0
    for animal in ["molly","julio","klein"]:
        name = animal[0].upper()
        # first scatter plot the duration under threshold for each trial
        # add a small random jitter to the x-axis to avoid overlapping points
        n = len(duration_animal[animal])
        plt.scatter(np.ones(n)*animal_ind + np.random.uniform(-0.3, 0.3, size=n),
                    duration_animal[animal] + np.random.uniform(-0.01, 0.01, size=n),
                    color = color_by_animal[animal], 
                    alpha=0.2, s = 10)
        # then plot the violin plot to show the distribution of duration under threshold for each animal
        # sns.violinplot(x=[name]*len(duration_animal[animal]),
        #                y=duration_animal[animal],
        #                color = color_by_animal[animal],
        #                alpha = 0.5, inner=None)
        # overlay seaborn boxplot for 0.05-0.95 quantiles
        sns.boxplot(x=[name]*len(duration_animal[animal]),
                    y=duration_animal[animal],
                    color = color_by_animal[animal],
                    boxprops=dict(alpha=0.6),
                    width=0.2,
                    showfliers=False, # Hide individual outliers
                    whis=[5, 95],      # Set whiskers to 5th and 95th percentiles
                    ax=plt.gca())
        animal_ind += 1
    plt.xlabel('Subject')
    plt.ylabel('seconds')
    plt.title(f'Duration under {int(speed_threshold)} cm/s')
    # remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "duration_under_speed_threshold.pdf"), dpi=300)
    plt.show()
    
def plot_average_angular_speed(average_animal, output_folder):
    # plot the distribution of duration under threshold for each animal
    # use seaborn violin plot to show the distribution of duration under threshold for each animal
    plt.figure(figsize=(3,3))
    animal_ind = 0
    for animal in ["molly", "julio","klein"]:
        name = animal[0].upper()
        # first scatter plot the duration under threshold for each trial
        # add a small random jitter to the x-axis to avoid overlapping points
        n = len(average_animal[animal])
        plt.scatter(np.ones(n)*animal_ind + np.random.uniform(-0.3, 0.3, size=n),
                    average_animal[animal] + np.random.uniform(-0.01, 0.01, size=n),
                    color = color_by_animal[animal], 
                    alpha=0.2, s = 10)
        # then plot the violin plot to show the distribution of duration under threshold for each animal
        # sns.violinplot(x=[name]*len(average_animal[animal]),
        #                y=average_animal[animal],
        #                color = color_by_animal[animal],
        #                alpha = 0.5, inner=None)
        # overlay seaborn boxplot for 0.05-0.95 quantiles
        sns.boxplot(x=[name]*len(average_animal[animal]),
                    y=average_animal[animal],
                    color = color_by_animal[animal],
                    boxprops=dict(alpha=0.6),
                    width=0.2,
                    showfliers=False, # Hide individual outliers
                    whis=[5, 95],      # Set whiskers to 5th and 95th percentiles
                    ax=plt.gca())
        animal_ind += 1
    plt.gca().set_ylim(-2, 2)
    plt.xlabel('Subject')
    plt.ylabel('(rad/s)')
    plt.title(f'Average angular speed')
    # remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "average_angular_speed.pdf"), dpi=300)
    plt.show()
    
def plot_integrated_phi(integrated_animal, output_folder):
    # plot the distribution of duration under threshold for each animal
    # use seaborn violin plot to show the distribution of duration under threshold for each animal
    plt.figure(figsize=(3,3))
    animal_ind = 0
    for animal in ["molly", "julio","klein"]:
        name = animal[0].upper()
        # first scatter plot the duration under threshold for each trial
        # add a small random jitter to the x-axis to avoid overlapping points
        n = len(integrated_animal[animal])
        data = integrated_animal[animal]
        plt.scatter(np.ones(n)*animal_ind + np.random.uniform(-0.3, 0.3, size=n),
                    data + np.random.uniform(-0.01, 0.01, size=n),
                    color = color_by_animal[animal], 
                    alpha=0.2, s = 10)
        # then plot the violin plot to show the distribution of duration under threshold for each animal
        # sns.violinplot(x=[name]*len(integrated_animal[animal]),
        #                y=integrated_animal[animal],
        #                color = color_by_animal[animal],
        #                alpha = 0.5, inner=None)
        # overlay seaborn boxplot for 0.05-0.95 quantiles
        sns.boxplot(x=[name]*len(integrated_animal[animal]),
                    y=data,
                    color = color_by_animal[animal],
                    boxprops=dict(alpha=0.6),
                    width=0.2, showfliers=False, # Hide individual outliers
                    whis=[5, 95],      # Set whiskers to 5th and 95th percentiles
                    ax=plt.gca())
        animal_ind += 1
    
    #plt.gca().set_ylim(0, 5)
    #plt.gca().set_yticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    #plt.gca().set_yticklabels(['0', 'π', '2π', '3π', '4π'])
    plt.xlabel('Subject')
    plt.ylabel('rad')
    plt.title(f'integrated log absolute \n head angular speed')
    # remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "average_angular_integration.pdf"), dpi=300)
    plt.show()