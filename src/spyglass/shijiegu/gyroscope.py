### functions for gyroscope data processing
### Due to nwb converson issues, gyroscope data needs to be read directly from the raw files

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pynwb
import pandas as pd
from spyglass.shijiegu.load import load_run_sessions
from spyglass.shijiegu.decodeHelpers import runSessionNames

import glob
from rec_to_binaries.read_binaries import readTrodesExtractedDataFile

from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.common import (Session, IntervalList,LabMember, LabTeam, Raw, Nwbfile,
                            Electrode,StateScriptFile)
from spyglass.common.common_behav import RawPosition
from spyglass.common.common_behav import VideoFile
from spyglass.shijiegu.ripple_add_replay import select_subset_helper_pd

from trodestrack.data.load_arthur_session import load_arthur_session
from trodestrack.models.ekf import EKFConfig, extended_kalman_filter
from spyglass.shijiegu.Analysis_SGU import ChangeofMindRemoteTheta
# interpolate


def translate_time(trodes_sample_time,sample_count,time_seconds):
    '''
    INPUT:
    trodes_sample_time, (n,), trodes time in sample count to be translated to system time in seconds
    sample_count: numpy array, (N,), trodes time in sample count for the whole recording
    time_seconds: numpy array, (N,), system time in seconds for the whole recording
    see also MATLAB counterpart translate_time
    
    RETURN: translated_sys_time, (n,), system time in seconds for inquired trodes sample time
    
    '''
    
    translated_time = translate_time_old(trodes_sample_time,sample_count,time_seconds)
    if np.diff(translated_time[~np.isnan(translated_time)]).min()<0:
        print("Warning: translated time has decreasing segments, switching to new version")
        translated_time = translate_time_new(trodes_sample_time,sample_count,time_seconds)
    return translated_time

def translate_time_old(trodes_sample_time,sample_count,time_seconds):
    '''
    INPUT:
    trodes_sample_time, (n,), trodes time in sample count to be translated to system time in seconds
    sample_count: numpy array, (N,), trodes time in sample count for the whole recording
    time_seconds: numpy array, (N,), system time in seconds for the whole recording
    see also MATLAB counterpart translate_time
    
    RETURN: translated_sys_time, (n,), system time in seconds for inquired trodes sample time
    
    '''
    notnan_ind=np.argwhere(~np.isnan(trodes_sample_time)).ravel()
    xy,ind1,ind2=np.intersect1d(trodes_sample_time[notnan_ind],sample_count,return_indices=True)
    #assert np.sum(~nan_ind)==len(ind2)
    translated_sys_time=np.zeros_like(trodes_sample_time)+np.nan
    translated_sys_time[notnan_ind[ind1]]=time_seconds[ind2]
    
    return translated_sys_time*10**-9

def find_subsequence_indices(subsequence, main_list):
    """
    Finds the starting indices in main_list that match the elements 
    of the subsequence in order, respecting duplicates.

    Args:
        subsequence (list): The list (subsequence) to find.
        main_list (list): The list to search within.

    Returns:
        list: A list of indices from main_list corresponding to 
              the matched elements of the subsequence. 
              Returns an empty list if the subsequence is not fully found.
    """
    
    # Check if the subsequence is longer than the main list
    if len(subsequence) > len(main_list):
        return []

    # Stores the indices in main_list that match the subsequence elements
    matched_indices = []
    
    # Start the search from the beginning of the main_list
    # The current_search_index tracks where in main_list we should 
    # look for the next element of the subsequence.
    current_search_index = 0

    # Iterate through each element in the subsequence (arg1)
    for sub_element in subsequence:
        found = False
        
        # Search for the current sub_element in the main_list (arg2) 
        # starting from the position after the last successful match.
        for i in range(current_search_index, len(main_list)):
            
            if sub_element == main_list[i]:
                # Match found!
                matched_indices.append(i)
                
                # Update the starting search index for the next element
                # We start the next search right after the current successful match (i + 1)
                current_search_index = i + 1
                found = True
                break # Move to the next element in the subsequence
        
        # If we went through the rest of the main_list and couldn't find the 
        # current subsequence element, the full subsequence cannot be matched.
        if not found:
            return [] # Return an empty list indicating failure
            
    return matched_indices

def translate_time_new(trodes_sample_time,sample_count,time_seconds):
    '''
    INPUT:
    trodes_sample_time, (n,), trodes time in sample count to be translated to system time in seconds
    sample_count: numpy array, (N,), trodes time in sample count for the whole recording
    time_seconds: numpy array, (N,), system time in seconds for the whole recording
    see also MATLAB counterpart translate_time
    
    RETURN: translated_sys_time, (n,), system time in seconds for inquired trodes sample time
    
    '''
    notnan_ind=np.argwhere(~np.isnan(trodes_sample_time)).ravel()
    print("new version")
    
    ind2 = find_subsequence_indices(trodes_sample_time[notnan_ind], sample_count)
    translated_sys_time=np.zeros_like(trodes_sample_time)+np.nan
    translated_sys_time[notnan_ind]=time_seconds[ind2]
    
    """
    ind2 = np.argwhere(np.isin(sample_count, trodes_sample_time[notnan_ind])).ravel()
    ind1 = np.arange(len(notnan_ind))
    ind2_final = []
    match_ind_full = 0
    #print(ind2)
    for ind in ind1:
        #print("\nmatch_ind_full",match_ind_full)
        source = trodes_sample_time[notnan_ind[ind]]
        target = sample_count[ind2][match_ind_full:]
        #print( "source",source)
        #print( "target",target)
        match_ind = np.argwhere(source==target)
        if len(match_ind)>0:
            ind2_final.append(ind2[match_ind[0][0] + match_ind_full])
            match_ind_full = match_ind[0][0] + match_ind_full + 1
        #print("ind2_final",ind2_final)
            
    
    #xy,ind1,ind2=np.intersect1d(trodes_sample_time[notnan_ind],sample_count,return_indices=True)
    #assert np.sum(~nan_ind)==len(ind2)
    translated_sys_time=np.zeros_like(trodes_sample_time)+np.nan
    translated_sys_time[notnan_ind[ind1]]=time_seconds[ind2_final]
    """
    
    
    return translated_sys_time*10**-9

def read_data_in_folder(preprocessing_folder):
    """data will have the following fields
    AccelX, AccelY, AccelZ,
    GyroX, GyroY, GyroZ,
    timestamps, trodes sample count
    """
    files = glob.glob(preprocessing_folder)

    data = {}
    for filename in files:
        suffix = filename.split("_")[-1]
        suffix = suffix.split(".")[-2]
        print(suffix)
        if "analogio" in suffix:
            continue
        else:
            data[suffix] = readTrodesExtractedDataFile(filename)
            
    return data

def gyroscope_data_session(nwb_file_name, session_name, parent_folder, sample_count, time_seconds):
    """read gyroscope data for a given session
    parent_folder: something like /stelmo/shijie/recording_pilot/julio/preprocessing/
    """
    d = nwb_file_name[5:13]
    animal = nwb_file_name[0:5].lower()
    preprocessing_folder = f"{parent_folder}/{d}/{d}_{animal}_{session_name}.analog/*"
    data = read_data_in_folder(preprocessing_folder)
    
    # translate trodes sample count to system time in seconds
    data["timestamps"]['unix'] = translate_time(
        data["timestamps"]['data'].astype("int64"), sample_count, time_seconds)
    
    return data

def gyroscope_data_day(nwb_file_name, parent_folder):
    """read gyroscope data for a given day (all sessions)
    parent_folder: something like /stelmo/shijie/recording_pilot/julio/preprocessing/
    """
    d = nwb_file_name[5:13]
    animal = nwb_file_name[0:5].lower()
    
    # for the day, load trodes sample count and system time (in unix seconds)
    nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
    nwb_file_abs_path = (Nwbfile & {'nwb_file_name':nwb_copy_file_name}).fetch1('nwb_file_abs_path')
    io = pynwb.NWBHDF5IO(nwb_file_abs_path,'r')
    nwbf = io.read()
    sample_count=np.array(nwbf.processing['sample_count'].data_interfaces['sample_count'].data)
    time_seconds=np.array(nwbf.processing['sample_count'].data_interfaces['sample_count'].timestamps)
    
    # find all sessions on this day
    run_session_ids, run_session_names, pos_session_names = load_run_sessions(
        nwb_copy_file_name)
    
    data_all_sessions = {}
    for session_name in run_session_names:
        print("loading gyroscope data for session:", session_name)
        data_all_sessions[session_name] = gyroscope_data_session(nwb_file_name, session_name, parent_folder, sample_count, time_seconds)
    
    return data_all_sessions

gyro_datalabels = ['AccelX','AccelY','AccelZ','GyroX','GyroY','GyroZ']
def extract_gyro_data_in_interval(gyro_data, interval):
    """extract gyroscope data in a given interval
    interval: (start_time, end_time) in unix seconds
    """
    start_time, end_time = interval
    mask = (gyro_data['timestamps']['unix'] >= start_time) & (gyro_data['timestamps']['unix'] <= end_time)
    
    extracted_data = {}
    for label in gyro_datalabels:
        extracted_data[label] = gyro_data[label]['data'][mask].astype("float32")
    
    extracted_data['timestamps'] = gyro_data['timestamps']['unix'][mask]
    
    return extracted_data

def extract_position_data_in_interval(nwb_copy_file_name, interval, pos_name):
    t0,t1 = interval
    key = {"nwb_file_name": nwb_copy_file_name,"interval_list_name":pos_name}
    raw_position = RawPosition.PosObject & key
    spatial_series = raw_position.fetch_nwb()[0]["raw_position"]
    spatial_df = raw_position.fetch1_dataframe()
    spatial_subset_df = select_subset_helper_pd(spatial_df, (t0,t1))
    spatial_subset_df = spatial_subset_df.dropna()
    spatial_subset_df = spatial_subset_df.rename(
        columns={'xloc': 'xloc2','yloc': 'yloc2',
                'xloc2': 'xloc','yloc2': 'yloc'})
    
    spatial_subset_df_original = spatial_subset_df.copy(deep=True) 
    
    if "video_frame_ind" not in spatial_subset_df.columns:
        spatial_subset_df.insert(
            loc=0, column='video_frame_ind', value=np.arange(len(spatial_subset_df))
            )
    else:
        spatial_subset_df.video_frame_ind = spatial_subset_df.video_frame_ind-np.array(
            spatial_subset_df.video_frame_ind)[0]

    table = pa.Table.from_pandas(spatial_subset_df)
    table_original = pa.Table.from_pandas(spatial_subset_df_original)
    
    meters_per_pixel = spatial_series.conversion
    
    return table, table_original, meters_per_pixel #table has added video_frame_ind starting from 0

def prepare_data_for_tracking(extracted_imu_data, extracted_pos_data, imu_path, pos_path, meters_per_pixel):
    
    # write IMU data to parquet file
    df = pd.DataFrame({"Headstage_GyroX": extracted_imu_data["GyroX"], #X is roll
                   "Headstage_GyroY": extracted_imu_data["GyroY"], #Y is pitch (milk lick associated head bobbing)
                   "Headstage_GyroZ": extracted_imu_data["GyroZ"], #Z is yaw (rotation along xy)
                   "Headstage_AccelX": extracted_imu_data["AccelX"],
                   "Headstage_AccelY": extracted_imu_data["AccelY"],
                   "Headstage_AccelZ": extracted_imu_data["AccelZ"]},
                   index = extracted_imu_data['timestamps'])
    
    table = pa.Table.from_pandas(df)
    pq.write_table(table, imu_path)
    
    # write position data to parquet file
    pq.write_table(extracted_pos_data, pos_path)
    
    packed_data = load_arthur_session(pos_path, imu_path,
        verbose = True,
        imu_mode = "2d",  # Use full 6-axis IMU
        meters_per_pixel = meters_per_pixel)
    
    return packed_data


def run_trodes_tracking(packed_data):
    """run trodes tracking algorithm

    default values:
    process_noise_pos = 0.02
    process_noise_vel = 2.0
    process_noise_heading = 0.02
    process_noise_gyro_bias = 2e-6
    process_noise_accel_bias = 2e-4

    measurement_noise_pos = 0.005**2
    measurement_noise_heading = 0.05**2
    """
    
    process_noise_heading = 0.02 #0.1
    process_noise_pos =  0.02 #0.005 #0.02
    process_noise_gyro_bias = 2e-6 #0.9
    measurement_noise_pos = 0.005**2 #0.4**2
    measurement_noise_heading = 0.05**2 #0.9**2

    # Configure EKF for 2D camera + 3D IMU
    ekf_config = EKFConfig(
        state_mode="2d_cam_3d_imu",  # 10D state
        process_noise_pos = process_noise_pos,
        process_noise_heading = process_noise_heading,
        process_noise_vel=2.0,
        process_noise_gyro_bias=process_noise_gyro_bias,#2e-6,
        process_noise_accel_bias=2e-4,
        measurement_noise_pos = measurement_noise_pos, #0.005**2,
        measurement_noise_heading = measurement_noise_heading,
        damping_coeff=0.1,
        led_distance=packed_data.led_distance,
    )
    
    # Run filter
    result = extended_kalman_filter(
        ekf_config=ekf_config,
        t_imu=packed_data.t_imu,
        U_imu=packed_data.U_imu,  # [N × 6] for 3D mode
        t_cam=packed_data.t_cam,
        Z_cam_led1=packed_data.Z_cam_led2, # front?
        Z_cam_led2=packed_data.Z_cam_led1, # back?
        mask_cam=packed_data.mask_cam,
    )
    
    # parse results
    X_est = np.array(result.filtered_means)
    P_est = np.array(result.filtered_covariances)
    inferred_position_x, inferred_position_y = X_est[:,0], X_est[:,1]
    inferred_heading = X_est[:,5]
    
    return inferred_position_x, inferred_position_y, inferred_heading, result


def batch_process_animal(animal, list_of_days, output_folder, parent_folder = None):
    # for every day, loop through sessions.
    # and every session, loop through all change of mind trials with remote content
    # output_folder = '/stelmo/shijie/gyro/'
    if parent_folder is None:
        parent_folder = f"/stelmo/shijie/recording_pilot/{animal}/preprocessing/"

    for d in list_of_days:
        result_day = {}
        
        nwb_file_name = f'{animal}{d}.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)

        # Read in gyroscope_data_day
        data_day = gyroscope_data_day(nwb_file_name, parent_folder)

        # session information
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        for session_ind in range(len(session_interval)):
            session_name, pos_name = session_interval[session_ind], position_interval[session_ind]
            print(f"Working on session: {session_name}")

            # trials of interest
            q = {"nwb_file_name": nwb_copy_file_name,
                "epoch":int(session_name[:2]),
                "proportion":str(0.1),
                "delta_t_minus":5,
                "delta_t_plus":5}
            
            log_df = ChangeofMindRemoteTheta().fetch1_dataframe(q)
            
            change_of_mind_trials = log_df[log_df.has_remote_interval].index
            print(f"Change of mind trials with remote content are: {change_of_mind_trials}")

            # loop through trials
            for trial in change_of_mind_trials:
                if trial + 1 >= (len(log_df.index)-1):
                    print(f"Skipping trial {trial} since trial {trial+1} not found")
                    continue
                
                trial1, trial2 = (trial, trial+1)
                (t0, t1) = (log_df.loc[trial1].timestamp_H, log_df.loc[trial2].timestamp_O)
                
                extracted_imu_data = extract_gyro_data_in_interval(data_day[session_name], (t0, t1))
                if len(extracted_imu_data['timestamps']) == 0:
                    print(f"Skipping trial {trial} since no imu data found")
                    print(nwb_copy_file_name, session_name, trial)
                    continue
                extracted_pos_data, table_original, meters_per_pixel = extract_position_data_in_interval(
                    nwb_copy_file_name, (t0, t1), pos_name)
        
                imu_path = f'{output_folder}imu_{nwb_file_name}_{session_name}_trial{trial1}.parquet'  # the path to write imu data to
                pos_path = f'{output_folder}position_{nwb_file_name}_{session_name}_trial{trial1}.parquet' # the path to write pos data to
        
                packed_data = prepare_data_for_tracking(
                    extracted_imu_data, extracted_pos_data, imu_path, pos_path, meters_per_pixel)
                inferred_position_x, inferred_position_y, inferred_heading, result = run_trodes_tracking(packed_data)

                result_trial = {}
                result_trial["packed_data"] = packed_data
                result_trial["result"] = result

                # Define the file path
                file_path = os.path.join(output_folder,f'tracking_{nwb_file_name}_{session_name}_trial{trial1}_result.pkl')
                # Open the file in write binary mode ('wb') and save the data
                with open(file_path, 'wb') as file:
                    pickle.dump(result_trial, file)
                print(f"Data successfully saved to {file_path}")

                result_day[(nwb_file_name, session_name, trial1)] = result_trial

        file_day_path = os.path.join(output_folder,f'tracking_{nwb_file_name}_result.pkl')
        with open(file_day_path, 'wb') as file:
            pickle.dump(result_day, file)
        print(f"Data successfully saved to {file_day_path}")
    return 1

def load_tracking_result(output_folder, nwb_file_name, session_name, trial):
    """load tracking result from file
    """
    file_path = os.path.join(output_folder,f'tracking_{nwb_file_name}_{session_name}_trial{trial}_result.pkl')
    try:
        with open(file_path, 'rb') as file:
            result = pickle.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
       
    result = result["result"] 
    X_est = np.array(result.filtered_means)
    P_est = np.array(result.filtered_covariances)
    inferred_position_x, inferred_position_y = X_est[:,0], X_est[:,1]
    inferred_heading = X_est[:,5] + np.pi
    
    # load timestamps
    pos_df = load_tracking_data_position(output_folder, nwb_file_name, session_name, trial)
    
    
    # make pandas dataframe, of index being timestamps, and columns being x, y, heading
    pos_info = pd.DataFrame({
        "head_position_x": inferred_position_x * 100, # in cm
        "head_position_y": inferred_position_y * 100, # in cm
        "head_orientation": inferred_heading
    }, index = pos_df.index)
    
    return pos_info

def load_tracking_data_position(output_folder, nwb_file_name, session_name, trial):
    #imu_path = f'{output_folder}imu_{nwb_file_name}_{session_name}_trial{trial}.parquet'  # the path to write imu data to
    pos_path = f'{output_folder}position_{nwb_file_name}_{session_name}_trial{trial}.parquet'
    #with open(imu_path, 'rb') as file:
    #    imu_table = pq.read_table(file)
    with open(pos_path, 'rb') as file:
        pos_table = pq.read_table(file)
    return pos_table.to_pandas()
                


        
        
        
