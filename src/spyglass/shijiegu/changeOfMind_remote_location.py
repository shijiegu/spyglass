import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripple_detection.core import segment_boolean_series
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.load import load_decode
from spyglass.shijiegu.decodeHelpers import session2position_name, runSessionNames
from spyglass.shijiegu.ripple_add_replay import select_subset_helper, select_subset_helper_pd
from spyglass.shijiegu.Analysis_SGU import get_linearization_map, ChangeofMindTriggeredDecode
from spyglass.common.common_position import TrackGraph, IntervalLinearizedPosition, IntervalPositionInfo

from spyglass.shijiegu.decodeHelpers import session2position_name
from spyglass.shijiegu.changeOfMind_remote_interval import load_remote_animal, load_remote_animal_spyglass, loc1d_to_2d_vector, find_angle
from spyglass.shijiegu.Analysis_SGU import ChangeofMind, ChangeofMindRemoteTheta, MUATheta, ChangeofMindTheta
from spyglass.shijiegu.changeOfMind_triggered import (seq2, rev2, rev3, seq1, rev1,
                                                      form_null_model,
                                                      find_large_position_minus_decode_trials_lightweight, find_large_position_minus_decode_trials)
from spyglass.shijiegu.changeOfMind_triggered_position import load_triggered_position_decode_day
from spyglass.shijiegu.changeOfMind_helper import unique_stable, setdiff1d_stable
from spyglass.shijiegu.gyroscope import load_tracking_result, load_tracking_data_position
from spyglass.shijiegu.helpers import interpolate_to_new_time
from spyglass.shijiegu.changeOfMind import nodes, vectors
#from spyglass.shijiegu.changeOfMind_triggered import region

linear_map,welllocations = get_linearization_map(track_graph_name='4 arm lumped 2023')
region={}
region[5] = linear_map[0]
region[6] = linear_map[3]
region[7] = linear_map[5]
region[8] = linear_map[7]
region[9] = linear_map[9]
center_region = np.array([
                          linear_map[1],
                         linear_map[2],
                         linear_map[4],linear_map[6],linear_map[8]])

def find_location_animal(animal, list_of_days, list_of_days_to_process, parameter_name, proportion, use_1d = 1, debug = False):
    # loaded_data = load_remote_animal(
    #     animal, list_of_days,
    #     encoding_set,
    #     classifier_param_name,
    #     proportion = proportion, use_1d = use_1d, spyglass = True)
    loaded_data = load_remote_animal_spyglass(animal, list_of_days_to_process, parameter_name, proportion = proportion)
    
    queries = (ChangeofMindTriggeredDecode() & {"parameter": parameter_name}).fetch(as_dict = True)
    queries_decode_options = queries[0]['parameter_value']['decode_options']
    classifier_param_name = queries_decode_options["classifier_param_name"]
    encoding_set = queries_decode_options["encoding_set"]    
    
    (info_animal,time_intervals, arm_identities
            ) = (loaded_data['info_animal'],
                 loaded_data['time_intervals_animal'], loaded_data['arm_identities_animal'])
    
    replay_locations = [] #list of arrays of shape (n_replay, 2), 2D location of each replay
    animal_locations = [] #list of arrays of shape (n_replay, 2), 2D location of animal at each replay
    replay_arm_identities = []
    replay_arm_proportions = []
    nwb_copy_file_name_old = None
    session_name_old = None
    
    # for debug
    if debug:
        ind_list = [debug]
    else:
        ind_list = np.arange(len(info_animal))
    for ind in ind_list:
        # for each interval,
        # 1. load decode,
        # 2. find max decode location during that interval
        # 3. translate location to 2D location
        # 4. compute the cosine v1 = head direction v2 = animal -> 2D location
    
        nwb_copy_file_name, session_name, trialID = info_animal[ind]
        day = nwb_copy_file_name[5:13]
        if day not in list_of_days_to_process:
            continue
        
        # 1. Get decode and animal head direction
        if session_name != session_name_old or nwb_copy_file_name_old != nwb_copy_file_name:
            session_name_old = session_name
            nwb_copy_file_name_old = nwb_copy_file_name
            decode = load_decode(nwb_copy_file_name,session_name,classifier_param_name,encoding_set)
            position_axis = np.array(decode.coords['position'])
            
            # 1.1. get decode and animal head direction
            position_name = session2position_name(nwb_copy_file_name, session_name)
            position_info = (IntervalPositionInfo() & {
                            'nwb_file_name':nwb_copy_file_name,
                            'interval_list_name':position_name,
                            'position_info_param_name':'default_decoding'}).fetch1_dataframe()
            # position_info1d = (IntervalLinearizedPosition() & {
            #                 'nwb_file_name':nwb_copy_file_name,
            #                 'interval_list_name':position_name,
            #                 'position_info_param_name':'default_decoding'}).fetch1_dataframe() #for debug use only
        
        for sub_ind in range(len(time_intervals[ind])):
            t0, t1 = time_intervals[ind][sub_ind]
            arm_identity = arm_identities[ind][sub_ind]
            
            pos2d_subset = select_subset_helper_pd(position_info,(t0,t1))
            animal_location = np.hstack((np.array(pos2d_subset.head_position_x).reshape(-1,1),np.array(pos2d_subset.head_position_y).reshape(-1,1)))
            
            decode_subset = select_subset_helper(decode,(t0,t1),target_len = len(pos2d_subset),
                                                    epsilon = 0.001)
            
            # 2. find max decode location during that interval
            posterior_position_subset = decode_subset.causal_posterior.sum(dim='state') #causal decoder
            if len(decode_subset.time) != len(pos2d_subset):
                continue
            max_posterior_1d = np.array(position_axis[posterior_position_subset.argmax(dim = 'position')])

            # 3. translate location to 2D location
            max_posterior_2d = loc1d_to_2d_vector(max_posterior_1d, None) #exclude posterior1d in arm_identity
            # 4. compute the cosine v1 = head direction v2 = animal -> 2D location
            replay_locations.append(max_posterior_2d)
            animal_locations.append(animal_location)
            replay_arm_identities.append(arm_identity)
            
            # 5. reduce 1D location to proportion of arm length
            start1d, end1d = region[arm_identity + 5]
            arm_length = end1d - start1d
            proportion = (max_posterior_1d - start1d) / arm_length
            if np.any(proportion) > 1:
                assert 1 == 0, "proportion greater than 1"
            replay_arm_proportions.append(proportion)
            
    
    #if debug:
    return replay_locations, animal_locations, replay_arm_identities, replay_arm_proportions

def draw_maze(ax, zorder):
    for arm in nodes:
        #for arm_end in nodes[arm]:
        #    ax.scatter(arm_end[0],arm_end[1], color = 'k')
        ax.plot([nodes[arm][0][0], nodes[arm][1][0]],
                     [nodes[arm][0][1], nodes[arm][1][1]], color = 'grey', linewidth = 10, alpha = 0.5, zorder = zorder)

from sklearn.neighbors import KernelDensity

def get_kde(data, bandwidth = 10):
    # data is number_of_samples x 2

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)

    xmin = np.min([np.min([nodes[arm][0][0],nodes[arm][1][0]]) for arm in nodes])
    ymin = np.min([np.min([nodes[arm][0][1],nodes[arm][1][1]]) for arm in nodes])
    xmax = np.max([np.max([nodes[arm][0][0],nodes[arm][1][0]]) for arm in nodes])
    ymax = np.max([np.max([nodes[arm][0][1],nodes[arm][1][1]]) for arm in nodes])
    
    BINWIDTH = 3
    x_grid = np.arange(xmin,xmax,BINWIDTH)
    y_grid = np.arange(ymin,ymax,BINWIDTH)
    
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Reshape grid points to (n_points, 2) for evaluation
    grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    
    # Evaluate the log density model on the grid
    log_density = kde.score_samples(grid_points)
    
    # Convert log density to density and reshape for plotting
    density = np.exp(log_density).reshape(X_grid.shape)

    return density
    
def plot_density(animal, replay_locations, animal_locations, use_mean = False, use_all = False):

    if use_mean:
        replay_locations_cat = np.vstack([np.mean(replay, axis = 0) for replay in replay_locations])
        animal_locations_cat = np.vstack([np.mean(location, axis = 0) for location in animal_locations])
    elif use_all:
        replay_locations_cat = np.concatenate(replay_locations)
        animal_locations_cat = np.concatenate(animal_locations)
    
    ### Density plot
    xmin = np.min([np.min([nodes[arm][0][0],nodes[arm][1][0]]) for arm in nodes])
    ymin = np.min([np.min([nodes[arm][0][1],nodes[arm][1][1]]) for arm in nodes])
    xmax = np.max([np.max([nodes[arm][0][0],nodes[arm][1][0]]) for arm in nodes])
    ymax = np.max([np.max([nodes[arm][0][1],nodes[arm][1][1]]) for arm in nodes])

    fig, axes = plt.subplots(1,2,figsize = (12,4))
    
    # density_replay = get_kde(replay_locations_cat, bandwidth = 8)
    # vmin = np.quantile(density_replay, 0.5)
    # vmax = np.quantile(density_replay, 0.99)
    #img = axes[0].imshow(density_replay, extent=[xmin-10, xmax+10, ymax+10, ymin-10],aspect='auto',zorder = 0, vmin = vmin, vmax = vmax)
    #cbar = fig.colorbar(img, ax=axes[0])
    
    # density_occupancy = get_kde(animal_locations_cat, bandwidth = 8)
    # vmax = np.quantile(density_occupancy, 0.98)
    #img = axes[1].imshow(density_occupancy, extent=[xmin-10, xmax+10, ymax+10, ymin-10],aspect='auto',zorder = 0)
    #cbar = fig.colorbar(img, ax=axes[1])
    
    draw_maze(axes[0], zorder = 1)
    draw_maze(axes[1], zorder = 1)
    
    #axes[0].imshow(replay_occupancy.T, extent=[xe[0], xe[-1], ye[-1], ye[0]],aspect='auto')
    #axes[1].imshow(animal_occupancy.T, extent=[xe[0], xe[-1], ye[-1], ye[0]],aspect='auto')
    
    if use_all:

        # jitter
        size1 = len(replay_locations_cat)
        size2 = len(animal_locations_cat)
        
        axes[0].scatter(replay_locations_cat[:,0] + np.random.rand(size1) * 10, replay_locations_cat[:,1]  + np.random.rand(size1) * 10, color = 'k', alpha = 0.1, zorder = 2)
        axes[1].scatter(animal_locations_cat[:,0] + np.random.rand(size2) * 10, animal_locations_cat[:,1] + np.random.rand(size2) * 10, color = 'k', alpha = 0.1, zorder = 2)
    else:
        for content_ind in range(len(replay_locations)):
            replay_location = np.mean(replay_locations[content_ind],axis = 0)
            animal_location = np.mean(animal_locations[content_ind],axis = 0)
            x,y = replay_location
            axes[0].scatter(x + np.random.rand() * 10, y + np.random.rand() * 10, color = 'k', alpha = 0.6, zorder = 2) #'C'+str(content_ind % 5)
            x,y = animal_location
            axes[1].scatter(x + np.random.rand() * 10, y + np.random.rand() * 10, color = 'k', alpha = 0.6, zorder = 2) #'C'+str(content_ind % 5)
    
    
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    
    axes[0].set_title(f"{animal} \n replay location \n {len(replay_locations)} cycles")
    axes[1].set_title(f"{animal} \n animal location \n {len(replay_locations)} cycles")
    
    axes[0].tick_params(axis='both', which='both', length=0) #
    axes[1].tick_params(axis='both', which='both', length=0) #
