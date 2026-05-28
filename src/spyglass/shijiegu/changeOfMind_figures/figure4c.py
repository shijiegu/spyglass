import random

from requests import post
from spyglass.shijiegu.changeOfMind_remote_interval import loc1d_to_baseoff_vector
from spyglass.shijiegu.changeOfMind_triggered_position import load_day_position_info
from spyglass.shijiegu.changeOfMind_triggered_position import load_triggered_position_decode_session_spyglass
from spyglass.shijiegu.Analysis_SGU import ChangeofMindTriggeredDecode, DecodeResultsLinear
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.shijiegu.changeOfMind_triggered_position import return_mobility_movements, return_concentrated_hpd_moments
from spyglass.shijiegu.Analysis_SGU import EpochPos
from spyglass.shijiegu.decodeHelpers import runSessionNames, session2position_name
from spyglass.shijiegu.changeOfMind import color_by_rat

import xarray as xr
import numpy as np
import pandas as pd
import statsmodels.api as sm

from spyglass.shijiegu.helpers import intersection_of_lists, select_list_elements

def get_position_decode_data_day(animal, day, decoder_type, hpd_flag = False, do_control = False):
    
    classifier_param_name = "default_decoding_gpu_4armMaze"
    
    if "run" in decoder_type:
        encoding_set = "2Dheadspeed_above_4"
        decoder_string = "run_time"
    else:
        encoding_set = "all_maze"
        decoder_string = "all_maze"
    
    
    params_pre = f"params_pre_{decoder_string}_2_state"
    params_post = f"params_post_{decoder_string}_2_state"
    params_pre_control =  f"params_pre_control_{decoder_string}_2_state_5seconds"
    params_post_control =  f"params_post_control_{decoder_string}_2_state"

    # position info
    position_infos = load_day_position_info(animal, day)

    # decode
    decodes = load_day_decode(animal, day, 
                              classifier_param_name = classifier_param_name,
                              encoding_set = encoding_set)

    # pre
    pos_pre, decode_pre, time_pre, session_info_pre, speed_pre = return_cleaned_data(animal, day,
                                                        decodes,
                                                        position_infos,
                                                        params_pre, apply_hpd = hpd_flag)
    
    # post
    pos_post, decode_post, time_post, session_info_post, speed_post = return_cleaned_data(animal, day,
                                                           decodes,
                                                           position_infos,
                                                           params_post, apply_hpd = hpd_flag)
    
    _, ind_pre, ind_post = intersection_of_lists(session_info_pre, session_info_post)
    pos_pre, decode_pre, time_pre, session_info_pre, speed_pre = select_list_subset([pos_pre, decode_pre,
                                                                          time_pre, session_info_pre, speed_pre],
                                                                         ind_pre)

    pos_post, decode_post, time_post, session_info_post, speed_post = select_list_subset([pos_post, decode_post,
                                                                          time_post, session_info_post, speed_post],
                                                                         ind_post)
    
    if not do_control:
        data = {"pos_pre": pos_pre, "decode_pre": decode_pre, "time_pre":time_pre, "session_info_pre":session_info_pre,
            "pos_post": pos_post, "decode_post": decode_post, "time_post":time_post, "session_info_post":session_info_post, 
            "speed_pre": speed_pre, "speed_post": speed_post,
            }
        return data
        
    
    # pre control
    pos_pre_control, decode_pre_control, time_pre_control, session_info_pre_control, speed_pre_control = return_cleaned_data(animal, day,
                                                                            decodes,
                                                                            position_infos,
                                                                            params_pre_control, apply_hpd = hpd_flag)
    
    # post control
    pos_post_control, decode_post_control, time_post_control, session_info_post_control, speed_post_control = return_cleaned_data(animal, day,
                                                                            decodes,
                                                                            position_infos,
                                                                            params_post_control, apply_hpd = hpd_flag)
    
    
    # _, ind_pre, ind_post = intersection_of_lists(session_info_pre_control, session_info_post_control)

    # (pos_pre_control, decode_pre_control,
    #  time_pre_control, session_info_pre_control, speed_pre_control) = select_list_subset([pos_pre_control, decode_pre_control,
    #                                                                    time_pre_control, session_info_pre_control, speed_pre_control],
    #                                                                    ind_pre)
    # (pos_post_control, decode_post_control,
    #  time_post_control, session_info_post_control, speed_post_control) = select_list_subset([pos_post_control, decode_post_control,
    #                                                                     time_post_control, session_info_post_control, speed_post_control],
    #                                                                    ind_post)
    
    data = {"pos_pre": pos_pre, "decode_pre": decode_pre, "time_pre":time_pre, "session_info_pre":session_info_pre,
            "pos_post": pos_post, "decode_post": decode_post, "time_post":time_post, "session_info_post":session_info_post, 
            "speed_pre": speed_pre, "speed_post": speed_post,
            "pos_pre_control": pos_pre_control, "decode_pre_control": decode_pre_control, "time_pre_control":time_pre_control,
            "pos_post_control": pos_post_control, "decode_post_control": decode_post_control, "time_post_control": time_post_control,
            "speed_pre_control": speed_pre_control, "speed_post_control": speed_post_control,
            }
    
    return data

def return_cleaned_data(animal, day, decodes, position_infos, params_pre, apply_hpd = True):
    position_pre, decode_pre, time_pre, session_info = return_position_decode(animal, day, params_pre)

    # restrict to high moments with concentrated high posterior density
    if apply_hpd:
        position_pre, decode_pre, time_pre, session_info = return_concentrated_hpd_moments(
            position_pre, decode_pre, time_pre, decodes, session_info)
        
    # restrict data to movement time
    position_data_pre, decode_data_pre, time_data_pre, session_info, speed_pre = return_mobility_movements(
        position_pre, decode_pre, time_pre, position_infos, session_info, return_speed = True)
    
    return position_data_pre, decode_data_pre, time_data_pre, session_info, speed_pre
    
    
def return_position_decode(animal, day, parameter_name):
    nwb_file_name = animal.lower() + day + '.nwb'
    nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
    
    # key for everything
    key = {"nwb_file_name": nwb_copy_file_name, "proportion": 0.1}
    
    # key_pre
    key_pre = key.copy()
    key_pre["parameter"] = parameter_name
    epochs = (ChangeofMindTriggeredDecode & key_pre).fetch('epoch')
    
    position_data_pre = []
    decode_data_pre = []
    time_axis = []
    session_info = []
    for epoch in epochs:
        key_pre["epoch"] = epoch
        df = ChangeofMindTriggeredDecode().fetch1_dataframe(key_pre)
    
        position_data = df['triggered_positions_baseoff']
        decode_data = df['triggered_decodes_baseoff']
        time_abs = df['time_abs']
        
        for data in position_data:
            position_data_pre.append(data)
    
        for data in decode_data:
            decode_data_pre.append(data)

        for data in time_abs:
            time_axis.append(data)
            
        for ind in range(len(position_data)):
            session_info.append([int(epoch), np.array(df["triggered_trial_info"])[ind][0]])
            
    return position_data_pre, decode_data_pre, time_axis, session_info

def load_day_position_info(animal, day):
    # returns each session's position info
    positions = {}
    
    nwb_file_name = animal.lower() + day + '.nwb'
    nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
    
    # key for everything
    keys = (IntervalPositionInfo() & {'nwb_file_name':nwb_copy_file_name,
                          'position_info_param_name':'default_decoding'}).fetch(as_dict = True)
    for key in keys:
        position_info = (IntervalPositionInfo() & key).fetch1_dataframe()
        epoch = (EpochPos() & {'nwb_file_name': nwb_copy_file_name,
                               'position_interval': key['interval_list_name']}).fetch1("epoch")
        positions[int(epoch)] = position_info
        
    return positions

def load_day_decode(animal, day,
                    classifier_param_name = "default_decoding_gpu_4armMaze", encoding_set = "2Dheadspeed_above_4"):
    
    decodes = {}
    nwb_file_name = animal.lower() + day + '.nwb'
    nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
    
    # key for everything
    keys = (DecodeResultsLinear & {"nwb_file_name":nwb_copy_file_name,
                                   "classifier_param_name":classifier_param_name,
                                   "encoding_set": encoding_set}).fetch(as_dict = True)
    for key in keys:
        key.pop("posterior")
        decode_path = (DecodeResultsLinear() & key).fetch1("posterior")
        decode = xr.open_dataset(decode_path)
        decodes[int(key["interval_list_name"][:2])] = decode

    return decodes

def return_distance_day(position_data_pre, decode_data_pre, fix_range = None, return_speed = False):
    if return_speed:
        diff_all = []
        speed_all = []
        N_all = []
        
        for rendition_ind in range(len(position_data_pre)):
            fix_range_ = random.choice(fix_range) if fix_range is not None else None
            diff, speed, N = return_distance_max(position_data_pre[rendition_ind],
                                   decode_data_pre[rendition_ind],
                                   fix_range_, 
                                   return_speed = True)
            diff_all.append(diff)
            speed_all.append(speed)
            N_all.append(N)

        return np.array(diff_all), np.array(speed_all), np.array(N_all)
    
    diff_all = []
    for rendition_ind in range(len(position_data_pre)):
        diff = return_distance_max(position_data_pre[rendition_ind],
                                   decode_data_pre[rendition_ind], random.choices(fix_range)[0] if fix_range is not None else None)
        diff_all.append(diff)
        
        
    return np.array(diff_all)

def return_distance_max(position_data, decode_data, fix_range = None, return_speed = False):
    if fix_range is not None:
        ind = np.logical_and(position_data >= fix_range[0], position_data <= fix_range[1])
        if np.sum(ind) == 0:
            if return_speed:
                return np.nan, np.nan, np.nan
            return np.nan
        position_data = position_data[ind]
        decode_data = decode_data[ind]
    diff_max = np.nanmax(np.abs(position_data - decode_data))
    if return_speed:
        return diff_max, np.nanmax(position_data)-np.nanmin(position_data), len(position_data)
    return diff_max 

def make_GLM_xy(animals, data1_animals, data2_animals):
    GLM_xy = []
    for animal in animals:
        data1 = data1_animals[animal]
        data2 = data2_animals[animal]

        animal_category = [a == animal for a in animals]
        for ind in range(len(data1)):
            if np.isnan(data1[ind]):
                continue
            GLM_xy.append(animal_category + [
                1,
                0,
                data1[ind]])
        
        for ind in range(len(data2)):
            if np.isnan(data2[ind]):
                continue
            GLM_xy.append(animal_category + [
                0,
                1,
                data2[ind]])

    GLM_xy = np.array(GLM_xy)
    return GLM_xy

def make_GLM_xy_version2(animals, data1_animals, data2_animals, condition1_animals, condition2_animals):
    GLM_xy = []
    for animal in animals:
        data1 = data1_animals[animal]
        data2 = data2_animals[animal]
        condition1 = condition1_animals[animal]
        condition2 = condition2_animals[animal]
        # if any of data1, data2, condition1, condition2 is nan at the same ind, then drop that index
        valid_ind1 = ~(np.isnan(data1) | np.isnan(condition1))
        valid_ind2 = ~(np.isnan(data2) | np.isnan(condition2))
        data1 = data1[valid_ind1]
        data2 = data2[valid_ind2]
        condition1 = condition1[valid_ind1]
        condition2 = condition2[valid_ind2]
        animal_category = [a == animal for a in animals]
        for ind in range(len(data1)):
            if np.isnan(data1[ind]):
                continue
            GLM_xy.append(animal_category + [
                data1[ind],
                condition1[ind],
                0])
        
        for ind in range(len(data2)):
            if np.isnan(data2[ind]):
                continue
            GLM_xy.append(animal_category + [
                data2[ind],
                condition2[ind],
                1])

    GLM_xy = np.array(GLM_xy)
    return GLM_xy

def do_GLM_version2(animals, GLM_xy, condition1, condition2):
    ## Model 1: no theta or with theta
    feature_dict = {f"Rat {animals[animal_ind][0].upper()}":GLM_xy[:,animal_ind] for animal_ind in range(len(animals))}
    feature_dict[condition1] = GLM_xy[:, (len(animals))]
    if condition2 is not None:
        feature_dict[condition2] = GLM_xy[:, (len(animals) + 1)]

    X = pd.DataFrame(feature_dict)
    if len(animals) == 1:
        pass
    else:
        X = sm.add_constant(X)
    y = GLM_xy[:,-1]

    """a) Mixed Linear Effect"""
    ols_model = sm.Logit(y,X)
    ols_result1 = ols_model.fit()

    print("Mixed Logistic Effect \n",ols_result1.summary())
    
    return ols_result1, y, X

def do_GLM(animals, GLM_xy, condition1, condition2):
    ## Model 1: no theta or with theta
    feature_dict = {f"Rat {animals[animal_ind][0].upper()}":GLM_xy[:,animal_ind] for animal_ind in range(len(animals))}
    feature_dict[condition1] = GLM_xy[:, (len(animals))]
    feature_dict[condition2] = GLM_xy[:, (len(animals) + 1)]

    X = pd.DataFrame(feature_dict)
    #X = sm.add_constant(X)
    y = GLM_xy[:,-1]

    """a) Mixed Linear Effect"""
    ols_model = sm.OLS(y,X)
    ols_result1 = ols_model.fit()

    print("Mixed Linear Effect \n",ols_result1.summary())
    
    return ols_result1, y, X
    
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

def select_list_subset(lists, ind):
    lists_out = []
    for l in lists:
        lists_out.append(select_list_elements(l, ind))
    return lists_out


from spyglass.shijiegu.changeOfMind_triggered import region

def maze_distance(loaded_data, position_info, speed_threshold = 4, fix_range = None):
    d_pre = []
    d_post = []
    
    for ind in range(len(loaded_data["triggered_positions_baseoff"])):

        info = loaded_data["triggered_trial_info"][ind]
        arm = info[1]
        arm_base = region[arm + 5][0]
        arm_end = region[arm + 5][1]
        pos = np.array(loaded_data['triggered_positions_baseoff'][ind])
        dec = np.array(loaded_data['triggered_decodes_baseoff'][ind]) + arm_base
        time_unix = np.array(loaded_data['triggered_positions_baseoff'][ind].index)
        pos_centered = loaded_data['triggered_positions'][ind]
        # use pos_centered to find the moment of stopping (0s)
        
        # select subset of data when rats are in the arm 0<=x<=80
        arm_ind = np.logical_and((pos - arm_base) >= 0, (pos - arm_base) <= 80)
        arm_ind = arm_ind.ravel()
        if np.sum(arm_ind) == 0:
            continue
        pos_centered = pos_centered[arm_ind]
        pos = pos[arm_ind]
        dec = dec[arm_ind]
        time_unix = time_unix[arm_ind]
        
        
        t0 = np.argmin(np.abs(pos_centered.index - 0))
        
        
        # splice the position and decode to before and after stopping
        pos_before = pos[:t0]
        pos_after = pos[t0:]
        dec_before = dec[:t0]
        dec_after = dec[t0:]
        time_unix_before = time_unix[:t0]
        time_unix_after = time_unix[t0:]
        if len(pos_before) == 0 or len(pos_after) == 0:
            continue
        
        # restrict to the time the animal head speed >= 4cm/s
        speed = position_info["head_speed"].values
        time = position_info.index.values
        speed_interp = np.interp(time_unix_before, time, speed)
        speed_ind_before = speed_interp >= speed_threshold
        speed_interp = np.interp(time_unix_after, time, speed)
        speed_ind_after = speed_interp >= speed_threshold
        if np.sum(speed_ind_before) == 0 or np.sum(speed_ind_after) == 0:
            continue
        pos_before = pos_before[speed_ind_before]
        dec_before = dec_before[speed_ind_before]
        pos_after = pos_after[speed_ind_after]
        dec_after = dec_after[speed_ind_after]

        if fix_range is not None:
            # randomly select a range from fix_range for before and after
            fix_range_ = random.choice(fix_range)
        else:
            fix_range_ = None
        maze_distance_max_before = maze_distance_max(pos_before, dec_before, arm_base, arm_end, fix_range = fix_range_)
        maze_distance_max_after = maze_distance_max(pos_after, dec_after, arm_base, arm_end, fix_range = fix_range_)
        d_pre.append(maze_distance_max_before)
        d_post.append(maze_distance_max_after)

    return np.array(d_pre), np.array(d_post)

def maze_distance_max(pos, dec, arm_base, arm_end, fix_range = None):
    
    # # restrict the time animal is in the arm
    time_ind = np.logical_and(pos >= arm_base, pos <= arm_end)
    pos = np.array(pos)[time_ind]
    dec = np.array(dec)[time_ind]
    
    if fix_range is not None:
        time_ind = np.logical_and(
            (pos - arm_base) >= fix_range[0], (pos - arm_base) <= fix_range[1]
            )
        pos = pos[time_ind]
        dec = dec[time_ind]
    
    if len(pos) == 0 or len(dec) == 0:
        return np.nan

    # get distance
    loc1d_pos = loc1d_to_baseoff_vector(pos)
    loc1d_dec = loc1d_to_baseoff_vector(dec)

    d1 = np.abs(loc1d_pos[:,1] - loc1d_dec[:,1])
    d2 = loc1d_pos[:,1] + loc1d_dec[:,1]
    same_ind = loc1d_pos[:,0] == loc1d_dec[:,0]
    d2[same_ind] = d1[same_ind]
    
    return np.nanmax(d2)


def maze_distance_animal(animal, list_of_days, param = "params_both_max_run_time_2_state",
                         speed_threshold = 6, fix_range = None):
    d_pre_all = []
    d_post_all = []

    for day_ind in range(len(list_of_days)):
        day = list_of_days[day_ind]
        
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
    
        for session_name in session_interval:
            loaded_data = load_triggered_position_decode_session_spyglass(nwb_copy_file_name, int(session_name[:2]),
                                                               param, 0.1)
            if len(loaded_data) == 0:
                continue
            # load position info for this session
            pos_name = session2position_name(nwb_copy_file_name, session_name)
            position_info = (IntervalPositionInfo() & {'nwb_file_name': nwb_copy_file_name,
                              'position_info_param_name':'default_decoding',
                              'interval_list_name': pos_name}).fetch1_dataframe()
            
            d_pre, d_post = maze_distance(loaded_data, position_info,
                                          speed_threshold = speed_threshold,
                                          fix_range = fix_range)
            d_pre_all.extend(d_pre)
            d_post_all.extend(d_post)
    return np.array(d_pre_all), np.array(d_post_all)

import matplotlib.pyplot as plt
import pickle
from scipy.stats import ranksums, ttest_rel
import starbars
    
def plot_2D_distance_distribution(animals, param = None, savename = "prepost_2D_distance_distribution"):
    annotations = []
    dx = 0.35
    output_folder = '/stelmo/shijie/change_of_mind_analysis/figure4'
    fig, axes = plt.subplots(1, 1, figsize=(4, 2)) 
    
    for animal_ind, animal in enumerate(animals):
        if "control" in param:
            file_path = f"{output_folder}/prepost_2D_{animal}_control.pickle"
        else:
            file_path = f"{output_folder}/prepost_2D_{animal}.pickle"
        
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            d_pre = data["d_pre"]
            d_post = data["d_post"]
            omit_nan_ind = np.logical_or(np.isnan(d_pre), np.isnan(d_post))
            d_pre = d_pre[~omit_nan_ind]
            d_post = d_post[~omit_nan_ind]
            print(f"Data successfully loaded data from {file_path}")
        
        # plot violin plot for d_pre and d_post at x locations of animal_ind and animal_ind+dx
        
        for ind in range(len(d_pre)):
            axes.scatter([animal_ind], [d_pre[ind]], color = 'k', s = 0.1, zorder = 0)
            axes.scatter([animal_ind + dx], [d_post[ind]], color = 'k', s = 0.1, zorder = 0)
            axes.plot([animal_ind, animal_ind + dx], [d_pre[ind], d_post[ind]], color = 'k',
                      linewidth = 0.1, alpha = 0.5, zorder = 0)
            
        parts_pre = axes.violinplot([d_pre], positions=[animal_ind], widths=0.3,
                                    showmeans=True, showextrema=False)
        parts_post = axes.violinplot([d_post], positions=[animal_ind + dx], widths=0.3,
                                     showmeans=True, showextrema=False)
        
        # adjust face color for pre and post and remove violin body edge lines
        for pc in parts_pre['bodies']:
            pc.set_facecolor(color_by_rat[animal])
            pc.set_edgecolor('none')
            pc.set_alpha(0.7)
            pc.set_zorder(1)
        for pc in parts_post['bodies']:
            pc.set_facecolor(color_by_rat[animal])
            pc.set_edgecolor('none')
            pc.set_alpha(0.7)
            pc.set_hatch("///")
            pc.set_zorder(1)
        
        # add tick marks on the mean of the distributions
        parts_pre['cmeans'].set_linewidth(2)
        parts_pre['cmeans'].set_edgecolor('black')
        parts_post['cmeans'].set_linewidth(2)
        parts_post['cmeans'].set_edgecolor('black')
    
        loc_x, loc_y = animal_ind, animal_ind + dx
        _, p_value = ttest_rel(d_pre, d_post)
        print(f"{animal}: p value: ",p_value)
        if p_value < 0.05:
            annotations.append((loc_x, loc_y, np.round(p_value, 4)))
    
    axes.set_ylim(0, 180)
    starbars.draw_annotation(annotations, ax = axes)
    axes.set_xticks(np.arange(animal_ind+1) + dx/2)
    axes.set_xticklabels([f"Rat {animal[0].upper()}" for animal in animals])
    if "control" in param:
        axes.set_title("Pre / post run-through (COM t-1) control", pad=10)
    else:
        axes.set_title("Pre / post change of mind", pad=10)
    axes.set_ylabel("Max 2D distance (cm)")
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)  
    
    file_path = f"/home/shijiegu/Documents/spyglass/notebooks/Change of Mind Analysis/final_figures/figure3/{savename}.pdf"
    fig.savefig(file_path, format="pdf", bbox_inches="tight")  
    
    
def plot_2D_distance_distribution_post(animals, savename = "postpost_2D_distance_distribution"):
    annotations = []
    dx = 0.35
    output_folder = '/stelmo/shijie/change_of_mind_analysis/figure4'
    fig, axes = plt.subplots(1, 1, figsize=(4, 2)) 
    d_post_com_animals = {}
    d_post_control_animals = {}
    
    for animal_ind, animal in enumerate(animals):
        file_path_control = f"{output_folder}/post_2D_{animal}_control.pickle"
        file_path = f"{output_folder}/prepost_2D_{animal}.pickle"
        
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            d_pre = data["d_pre"]
            d_post = data["d_post"]
            omit_nan_ind = np.logical_or(np.isnan(d_pre), np.isnan(d_post))
            d_pre = d_pre[~omit_nan_ind]
            d_post_com = d_post[~omit_nan_ind]
            print(f"Data successfully loaded data from {file_path}")
        
        with open(file_path_control, 'rb') as file:
            data = pickle.load(file)
            d_post = data["d_post"]
            omit_nan_ind = np.isnan(d_post)
            d_post_control = d_post[~omit_nan_ind]
            print(f"Data successfully loaded data from {file_path_control}")
        
        # plot violin plot for d_pre and d_post at x locations of animal_ind and animal_ind+dx
            
        parts_pre = axes.violinplot([d_post_com], positions=[animal_ind], widths=0.3,
                                    showmeans=True, showextrema=False)
        parts_post = axes.violinplot([d_post_control], positions=[animal_ind + dx], widths=0.3,
                                     showmeans=True, showextrema=False)
        
        d_post_com_animals[animal] = d_post_com
        d_post_control_animals[animal] = d_post_control
        
        # adjust face color for pre and post and remove violin body edge lines
        for pc in parts_pre['bodies']:
            pc.set_facecolor(color_by_rat[animal])
            pc.set_edgecolor('none')
            pc.set_alpha(0.7)
            pc.set_zorder(1)
        for pc in parts_post['bodies']:
            pc.set_facecolor(color_by_rat[animal])
            pc.set_edgecolor('none')
            pc.set_alpha(0.7)
            pc.set_hatch("///")
            pc.set_zorder(1)
        
        # add tick marks on the mean of the distributions
        parts_pre['cmeans'].set_linewidth(2)
        parts_pre['cmeans'].set_edgecolor('black')
        parts_post['cmeans'].set_linewidth(2)
        parts_post['cmeans'].set_edgecolor('black')
    
        loc_x, loc_y = animal_ind, animal_ind + dx
        _, p_value = ranksums(d_post_com, d_post_control, alternative = "greater")
        print(f"{animal}: p value: ",p_value)
        if p_value < 0.05:
            annotations.append((loc_x, loc_y, np.round(p_value, 4)))
    
    axes.set_ylim(0, 180) 
    
    starbars.draw_annotation(annotations, ax = axes)
    axes.set_xticks(np.arange(animal_ind+1) + dx/2)
    axes.set_xticklabels([f"Rat {animal[0].upper()}" for animal in animals])

    axes.set_title("Post change of mind / run through post", pad=10)
    axes.set_ylabel("Max 2D distance (cm)")
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)  
    
    file_path = f"/home/shijiegu/Documents/spyglass/notebooks/Change of Mind Analysis/final_figures/figure3/{savename}.pdf"
    fig.savefig(file_path, format="pdf", bbox_inches="tight")
    
    return d_post_com_animals, d_post_control_animals
    
##### 1D plots functions #####
def intersect_pre_post(info_pre, info_post):
    pre_ind = []
    post_ind = []
    for ind_i in range(len(info_pre)):
        found = False
        info_i = info_pre[ind_i]
        for ind_j in range(len(info_post)):
            info_j = info_post[ind_j]
            if info_i == info_j:
                pre_ind.append(ind_i)
                post_ind.append(ind_j)
                break
            else:
                found = True
        if found:
            continue
    return pre_ind, post_ind

def check_nan(position_pre, position_post):
    nan_ind_pre = np.isnan(position_pre)
    nan_ind_post = np.isnan(position_post)
    ind = ~np.logical_or(nan_ind_pre,nan_ind_post)
    return ind

def make_distance_plot(animal, days, hpd_flag = False, output_folder = None, decoder_type = None):
    file_path = f"{output_folder}/prepost_{decoder_type}_{animal}_{hpd_flag}.pickle"
    with open(file_path, 'rb') as file:
        loaded_data_all_days = pickle.load(file)
        print(f"Data successfully loaded from {file_path}")
    
    position_pre, position_post = [], []
    decode_pre, decode_post =[], []
    position_pre_control, position_post_control =[],[]
    decode_pre_control, decode_post_control = [], []
    
    for day in days:
        
        load_data = loaded_data_all_days[day]

        pre_ind, post_ind = intersect_pre_post(load_data["session_info_pre"], load_data["session_info_post"])
        position_pre.extend([load_data["pos_pre"][ind] for ind in pre_ind])
        position_post.extend([load_data["pos_post"][ind] for ind in post_ind]) #load_data["pos_post"][post_ind])
        decode_pre.extend([load_data["decode_pre"][ind] for ind in pre_ind])
        decode_post.extend([load_data["decode_post"][ind] for ind in post_ind])
        position_pre_control.extend(load_data["pos_pre_control"])
        position_post_control.extend(load_data["pos_post_control"])
        decode_pre_control.extend(load_data["decode_pre_control"])
        decode_post_control.extend(load_data["decode_post_control"])

        assert len(position_pre) == len(position_post)
    
    diff_pre, max_pos_pre, N_pre = return_distance_day(position_pre, decode_pre, fix_range = None, return_speed = True)
    diff_post = return_distance_day(position_post, decode_post)
    
    pos_range = [[np.nanmin(pos), np.nanmax(pos)] for pos in position_pre]
    diff_pre_control, _, _2 = return_distance_day(
        position_pre_control, decode_pre_control, fix_range = pos_range, return_speed = True)
    
    _, max_pos_pre_control, N_pre_control = return_distance_day(
        position_pre_control, decode_pre_control, fix_range = None, return_speed = True)
    

    pos_range = [[np.nanmin(pos), np.nanmax(pos)] for pos in position_post]
    diff_post_control = return_distance_day(
        position_post_control, decode_post_control, fix_range = pos_range)
    
    # calculate speed for pre and pre_control
    speed_pre = max_pos_pre / (N_pre * 0.002) #each bin is 2ms
    speed_pre_control = max_pos_pre_control / (N_pre_control * 0.002)
    small_n_ind = N_pre_control < 250 # data with at least 0.5 second of data (250 bins) to calculate average speed
    speed_pre_control[small_n_ind] = np.nan
    
    small_n_ind = N_pre < 250
    speed_pre[small_n_ind] = np.nan
    
    diff_pre_old = diff_pre.copy()
    ind = check_nan(diff_pre_old, diff_post)
    diff_pre = diff_pre[ind]
    diff_post = diff_post[ind]
    speed_pre = speed_pre[ind]
    
    return diff_pre, diff_post, diff_pre_control, diff_post_control, speed_pre, speed_pre_control