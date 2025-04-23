import numpy as np
import pandas as pd
from scipy import linalg

from spyglass.shijiegu.changeOfMind_triggered import turnaround_triggered_position
from spyglass.shijiegu.singleUnit_sortedDecode import place_field_direction
from spyglass.shijiegu.Analysis_SGU import ThetaIntervals, TrialChoiceChangeOfMind
from spyglass.shijiegu.singleUnit import find_spikes
from spyglass.spikesorting.v0 import Curation
from spyglass.shijiegu.fragmented import get_nwb_units
from spyglass.common.common_position import IntervalLinearizedPosition
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.load import load_run_sessions


# Back theta
backThetaNum = 8 #0.125s x 8 go back about 1 second
delta_t_minus = 1.5
assert delta_t_minus > backThetaNum * 0.125

def triggered_theta_mua_animal(animal, list_of_days, proportion_threshold = 0.2,curation_id = 1):
    (triggered_late_theta,triggered_early_theta) = ([], [])
    (triggered_late_theta_nearby,triggered_early_theta_nearby) = ([], [])
    
    for day in list_of_days:
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        run_session_ids, run_session_names, pos_session_names = load_run_sessions(nwb_copy_file_name)
        for ind in range(len(run_session_names)):
            session_name = run_session_names[ind]
            position_name = pos_session_names[ind]
            epoch = run_session_ids[ind]
            late_theta, early_theta, late_theta_nearby, early_theta_nearby = triggered_theta_mua_session(nwb_copy_file_name,epoch,session_name,position_name,
                                proportion_threshold = proportion_threshold, curation_id = curation_id)
            for lt in late_theta:
                triggered_late_theta.append(lt)
            for et in early_theta:
                triggered_early_theta.append(et)
            for lt in late_theta_nearby:
                triggered_late_theta_nearby.append(lt)
            for et in early_theta_nearby:
                triggered_early_theta_nearby.append(et)
    
    return triggered_late_theta, triggered_early_theta, triggered_late_theta_nearby, triggered_early_theta_nearby   
            

def triggered_theta_mua_session(nwb_copy_file_name,epoch,session_name,position_name,
                                proportion_threshold = 0.2,curation_id = 1):
    # epoch is the epoch ID
    
    # get cell spiking data 
    (cells, smoothed_placefield, placefield_peak,
            spike_count_by_arm_direction, betaPdfs, means) = place_field_direction(nwb_copy_file_name,
                                                                                   session_name,position_name,
                                                                                   curation_id = curation_id)
                                                                                  
    sort_group_ids = np.unique((Curation() & {'nwb_file_name': nwb_copy_file_name,
                                          "curation_id":curation_id}).fetch("sort_group_id"))
    nwb_units_all = get_nwb_units(
        nwb_copy_file_name,session_name,sort_group_ids,curation_id = 1)

    #theta_parser_master(nwb_copy_file_name,
    #                position_name, session_name,
    #                nwb_units_all,cells)

    linear_position_info=(IntervalLinearizedPosition() & {
        'nwb_file_name':nwb_copy_file_name,
        'interval_list_name':position_name,
        'position_info_param_name':'default_decoding'}).fetch1_dataframe()

    ## load theta
    theta_times_path = (ThetaIntervals() & {'nwb_file_name': nwb_copy_file_name,
                    'interval_list_name': session_name}).fetch1("theta_times")
    theta_times = pd.read_csv(theta_times_path)

    ## calcuate cell mean firing rate etc baseline information
    baseline = baseline_firing_by_theta_phase(theta_times, nwb_units_all, cells)

    ## load change of mind time
    CoM = pd.read_pickle((TrialChoiceChangeOfMind() & {"nwb_file_name":nwb_copy_file_name,
                                  "epoch":epoch}).fetch1("change_of_mind_info"))
    CoM_subset1 = CoM[CoM.change_of_mind > 0]
    CoM_subset2 = CoM_subset1[CoM_subset1.CoMMaxProportion >= proportion_threshold]

    late_theta = []
    early_theta = []
    late_theta_nearby = []
    early_theta_nearby = []
    
    for trialID in CoM_subset2.index:
        t0 = CoM_subset2.loc[trialID].CoM_t[0][0]
  
        firing_matrix_early_theta, firing_matrix_late_theta = triggered_firing_by_theta_phase(
            t0, linear_position_info, theta_times, nwb_units_all, cells, baseline = baseline, trialID = trialID)
        late_theta.append(firing_matrix_late_theta)
        early_theta.append(firing_matrix_early_theta)
        
        t0 = find_nearby_trial_time(trialID, linear_position_info, np.array(CoM_subset1.index), CoM) # use a broader set of not eligible trials
        firing_matrix_early_theta, firing_matrix_late_theta = triggered_firing_by_theta_phase(
            t0, linear_position_info, theta_times, nwb_units_all, cells, baseline = baseline)
        assert len(firing_matrix_early_theta) == len(firing_matrix_late_theta)
        late_theta_nearby.append(firing_matrix_late_theta)
        early_theta_nearby.append(firing_matrix_early_theta)

    return late_theta, early_theta, late_theta_nearby, early_theta_nearby

from spyglass.common.common_position import TrackGraph
graph = TrackGraph() & {'track_graph_name': '4 arm lumped 2023'}
node_positions = graph.fetch1("node_positions")
#linear_map,node_location=get_linearization_map()
nodes={}
nodes[6] = (node_positions[2],node_positions[3])
nodes[7] = (node_positions[4],node_positions[5])
nodes[8] = (node_positions[6],node_positions[7])
nodes[9] = (node_positions[8],node_positions[9])

def find_nearby_trial_time(trialID, linear_position_info, rowID, log_df):
    # rowID is the list of trials that should not be returned
    # return nearby trial nonrewarded
    for r_ in [trialID - 1, trialID + 1, trialID + 2, trialID - 2, trialID + 3, trialID - 3, trialID + 4, trialID - 4]:
        condition1 = np.isin(r_,np.array(log_df.index[:-1]))
        condition2 = ~np.isin(r_,np.array(rowID))
        if condition1 and condition2:
            # find the time the rat is half way through the outer arm
            
            # restrict to this trial's position info
            end = log_df.loc[r_,'timestamp_O']
            start = log_df.loc[r_,'timestamp_H']
            if np.isnan(start):
                start = end - 1
            
            trialInd = (linear_position_info.index >= start) &(linear_position_info.index <= end)
            trialPosInfo = linear_position_info.loc[trialInd,:]
            
            # find time the animal first reached half of the arm
            last_arm = np.array(trialPosInfo.track_segment_id)[-1]
            trialPosInfoOuter = trialPosInfo.loc[np.array(trialPosInfo.track_segment_id) == last_arm]
            trialPosInfoOuter.projected_xy = np.hstack((np.array(trialPosInfoOuter.projected_x_position).reshape((-1,1)),
                                            np.array(trialPosInfoOuter.projected_y_position).reshape((-1,1))))
            
            track_segment_node_start = nodes[last_arm][0]
            track_segment_node_end = nodes[last_arm][1]
            
            full_length = linalg.norm(track_segment_node_start - track_segment_node_end)
            partial_length = linalg.norm(track_segment_node_start - trialPosInfoOuter.projected_xy, axis = 1)
            proportion = partial_length / full_length
            
            t_half_ind = np.argwhere(proportion>=0.5).ravel()
            if len(t_half_ind) > 0:
                t_half = trialPosInfoOuter.index[t_half_ind[0]]
                return t_half
    return np.nan

def baseline_firing_by_theta_phase(theta_times, nwb_units_all, cells):
    firing_matrix_early_theta_baseline = firing_rate_in_theta(
        theta_times, nwb_units_all, cells, 0, int(theta_times.index[-1]/8), int(theta_times.index[-1]/4), late_phase = False, resolution = 50)
    mean_early = np.nanmean(firing_matrix_early_theta_baseline, axis = 0)
    sd_early = np.nanstd(firing_matrix_early_theta_baseline, axis = 0)
    
    firing_matrix_late_theta_baseline = firing_rate_in_theta(
        theta_times, nwb_units_all, cells, 0, int(theta_times.index[-1]/8), int(theta_times.index[-1]/4), late_phase = True, resolution = 50)

    mean_late = np.nanmean(firing_matrix_late_theta_baseline, axis = 0)
    sd_late = np.nanstd(firing_matrix_late_theta_baseline, axis = 0)
    
    baseline = {}
    baseline["mean_early"] = mean_early
    baseline["mean_late"] = mean_late
    baseline["sd_early"] = sd_early
    baseline["sd_late"] = sd_late
    
    return baseline


def triggered_firing_by_theta_phase(t0, linear_position_info, theta_times, nwb_units_all, cells, baseline = None, trialID = None):
    # find cell firing around t0, binned into theta cycle
    # This function uses firing_rate_in_theta as a helper function
    # get nwb_units_all by:
    #   nwb_units_all = get_nwb_units(
    #       nwb_copy_file_name,session_name,sort_group_ids,curation_id = 1)
    
    # find the time animal is in the outer arm
    triggered_position, triggered_position_abs, arm = turnaround_triggered_position(t0,linear_position_info,
                                                                                    delta_t_minus = delta_t_minus, delta_t_plus = delta_t_minus)
    
    # from triggered_position_abs time to figure out the theta_cycles to include
    theta_ind_minus = np.argwhere(np.array(theta_times.start_time) >= triggered_position_abs.index[0]).ravel()[0]
    theta_ind_0 = np.argwhere(np.array(theta_times.start_time) >= t0).ravel()[0]
    theta_ind_plus = np.argwhere(np.array(theta_times.start_time) <= triggered_position_abs.index[-1]).ravel()[-1]
    
    if trialID is not None:
        # this is just sanity check. If you know t0 belongs to trialID, then info read from theta_times should match.
        assert theta_times.loc[theta_ind_minus].trial_number == trialID
        assert theta_times.loc[theta_ind_0].trial_number == trialID
        assert theta_times.loc[theta_ind_plus].trial_number == trialID
        
        
    firing_matrix_early_theta = firing_rate_in_theta(
        theta_times, nwb_units_all, cells, theta_ind_minus, theta_ind_0, theta_ind_plus, late_phase = False, resolution = 50, baseline = baseline)

    firing_matrix_late_theta = firing_rate_in_theta(
        theta_times, nwb_units_all, cells, theta_ind_minus, theta_ind_0, theta_ind_plus, late_phase = True, resolution = 50, baseline = baseline)
    
    return firing_matrix_early_theta, firing_matrix_late_theta
    
def firing_rate_in_theta(theta_times, nwb_units_all, cells,
                         theta_ind_minus, theta_ind_0, theta_ind_plus,
                         late_phase = False, resolution = 50, baseline = None):
    # resolution is number of data points per theta cycle.
    # return firing matrix measured in theta time, default is 50
    firing_matrices = []
    
    for theta_ind in range(theta_ind_minus, theta_ind_plus + 1):
        intvl = eval(theta_times.loc[theta_ind].theta_interval)
        if late_phase:
            t0 = (intvl[0] + intvl[1]) / 2
            t1 = intvl[1]
        else:
            t0 = intvl[0]
            t1 = (intvl[0] + intvl[1]) / 2
            
        axis = np.linspace(t0, t1, resolution)
        firing_matrix = find_spikes(nwb_units_all,cells,axis).T # shape of the final firing_matrix is cell x time bins
        firing_matrices.append(firing_matrix)

    firing_matrix = np.concatenate(firing_matrices, axis = 1)
    theta_num = theta_ind_plus - theta_ind_minus + 1

    if late_phase:
        time = np.concatenate([np.linspace(0.5, 1, resolution - 1) + theta_ind for theta_ind in range(theta_ind_minus, theta_ind_plus + 1)])
    else:
        time = np.concatenate([np.linspace(0, 0.5, resolution - 1) + theta_ind for theta_ind in range(theta_ind_minus, theta_ind_plus + 1)])
        
    time = time - theta_ind_0 
    
    # standardize
    if baseline is not None:
        if late_phase:
            firing_matrix = (firing_matrix - baseline["mean_late"].reshape((-1,1))) / baseline["sd_late"].reshape((-1,1))
        else:
            firing_matrix = (firing_matrix - baseline["mean_early"].reshape((-1,1))) / baseline["sd_early"].reshape((-1,1))
            
    firing_matrix_in_theta = pd.DataFrame(firing_matrix.T, index = time)
        
    return firing_matrix_in_theta