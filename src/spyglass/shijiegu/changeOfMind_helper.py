import pandas as pd
import numpy as np
import xarray as xr
from scipy import linalg
from scipy import ndimage
from ripple_detection.core import segment_boolean_series
from spyglass.common.common_position import TrackGraph


graph = TrackGraph() & {'track_graph_name': '4 arm lumped 2023'}
node_positions = graph.fetch1("node_positions")
#linear_map,node_location=get_linearization_map()
nodes={}
nodes[6] = (node_positions[2],node_positions[3])
nodes[7] = (node_positions[4],node_positions[5])
nodes[8] = (node_positions[6],node_positions[7])
nodes[9] = (node_positions[8],node_positions[9])

def unique_stable(arr):
    unique_values, indices = np.unique(arr, return_index=True)
    unique_values_stable = unique_values[np.argsort(indices)]

    return unique_values_stable

def setdiff1d_stable(arr1, arr2):
    mask = ~np.isin(arr1, arr2)
    return unique_stable(arr1[mask])

def find_turnaround_time(proportion,trialPosInfoOuter):
    """detects turn around time"""
    proportion_diff = np.concatenate(([0],np.diff(proportion))) < 0
    proportion_diff = ndimage.binary_closing(proportion_diff,iterations = 75) 
    # irregularity under 0.3s are smoothened, 75 * 0.002 (2ms) * 2
    
    turnsaround = pd.Series(proportion_diff, index = trialPosInfoOuter.index)
    #turnsaround = pd.Series(proportion_diff < 0.0001, index = trialPosInfoOuter.index)
    turnsaround_segments = np.array(segment_boolean_series(
            turnsaround, minimum_duration=0.1)).reshape((-1,2))

    return turnsaround_segments[:,0]

def findProportion(trialPosInfo, camera_frequency):
    
    outerArmInd = trialPosInfo.track_segment_id >= 4
    trialPosInfoOuter = trialPosInfo.loc[outerArmInd,:]
    
    # exclude the final segment in time
    last_arm = np.array(trialPosInfoOuter.track_segment_id)[-1]
    same_arm_last_segment = pd.Series(np.array(trialPosInfoOuter.track_segment_id) == last_arm, 
                                      index = trialPosInfoOuter.index)
    same_arm_last_segment_segments = np.array(segment_boolean_series(
            same_arm_last_segment, minimum_duration=0)).reshape((-1,2))
    
    trialPosInfoOuter = trialPosInfoOuter.loc[trialPosInfoOuter.index <= same_arm_last_segment_segments[-1][0],:]
    
    segments_involved = unique_stable(trialPosInfoOuter.track_segment_id)
    
    track_segment_node_start = np.array([nodes[i][0] for i in np.array(trialPosInfoOuter.track_segment_id)])
    track_segment_node_end = np.array([nodes[i][1] for i in np.array(trialPosInfoOuter.track_segment_id)])

    trialPosInfoOuter.projected_xy = np.hstack((np.array(trialPosInfoOuter.projected_x_position).reshape((-1,1)),
                                            np.array(trialPosInfoOuter.projected_y_position).reshape((-1,1))))
    
    full_length = linalg.norm(track_segment_node_start - track_segment_node_end, axis = 1)
    partial_length = linalg.norm(track_segment_node_start - trialPosInfoOuter.projected_xy, axis = 1)
    proportion = partial_length / full_length
    track_segment_id = trialPosInfoOuter.track_segment_id

    max_proportion = []
    turnaround_times = []
    for seg in segments_involved:
        seg_index = np.argwhere(trialPosInfoOuter.track_segment_id == seg).ravel()
        max_proportion.append( np.nanmax(proportion[seg_index]) )
        ts = find_turnaround_time(proportion[seg_index],
                                 trialPosInfoOuter[trialPosInfoOuter.track_segment_id == seg])
        # there can be multiple turnaround times in each arm
        for t in ts:
            # make sure there are at least 0.5 second of data preceding the turnaround time
            (t0_peak,t1_peak) = (t-1, t)
            subset_ind = (trialPosInfo.index >= t0_peak) & (trialPosInfo.index <= t1_peak)
            subset_linear = trialPosInfo.loc[subset_ind]
            print("len(subset_linear) ",len(subset_linear) )
            if len(subset_linear) < 0.8 * camera_frequency: # sampling rate is 500Hz, factoring some missing packets
                continue
            turnaround_times.append(t)
        
    return proportion, np.array(track_segment_id).astype("int"), max_proportion, turnaround_times

