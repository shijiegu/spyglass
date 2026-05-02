import pickle
import numpy as np
import pandas as pd
import xarray as xr
from spyglass.shijiegu.changeOfMind_triggered import find_large_position_minus_decode_trials, find_triggered_session
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.Analysis_SGU import ChangeofMind, ChangeofMindTheta, ChangeofMindTriggeredDecode, EpochPos, MUA, DecodeResultsLinear
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.shijiegu.ripple_add_replay import select_subset_helper_pd, select_subset_helper
from spyglass.shijiegu.Analysis_nwb_helper import write_data_to_analyis_nwb
from spyglass.shijiegu.changeOfMind_triggered import form_null_model, form_null_model_full

def return_save_name(animal, encoding_set, classifier_param_name, d,
                     delta_t_minus = 0, delta_t_plus = 1,
                     max_flag = 0, control = 0, proportion = 0.1, segment_only = False,
                     multiple_CoM = True, single_CoM = True, first_CoM = True):
    if segment_only:
        save_name = f'{animal.lower()}_triggered_position_{encoding_set}_{classifier_param_name}_{d}_p{proportion}_tm{delta_t_minus}_tp{delta_t_plus}_maxflag{int(max_flag)}_multiple_CoM{int(multiple_CoM)}_single_CoM{int(single_CoM)}_first_CoM{int(first_CoM)}_control{int(control)}'
    else:
        save_name = f'{animal.lower()}_triggered_position_{encoding_set}_{classifier_param_name}_{d}_p{proportion}_tm{delta_t_minus}_tp{delta_t_plus}_maxflag{int(max_flag)}_multiple_CoM{int(multiple_CoM)}_single_CoM{int(single_CoM)}_first_CoM{int(first_CoM)}_control{int(control)}_allposterior'
    return save_name
output_folder = '/stelmo/shijie/change_of_mind_analysis/'

def save_triggered_position_decode_day(animal, day, encoding_set, classifier_param_name, parameter_name,
                                       proportion = 0.1,
                                       delta_t_minus = 0, delta_t_plus = 2,
                                       max_flag = 0, control = False, segment_only = True,
                                       multiple_CoM = True, single_CoM = True, first_CoM = True):
    
    animal = animal[:5]
    table_name = ChangeofMindTriggeredDecode().full_table_name
    decode_options = {
        "encoding_set": encoding_set,
        "classifier_param_name": classifier_param_name
    }
    
    nwb_file_name = animal.lower() + day + '.nwb'
    nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
    session_interval, position_interval = runSessionNames(nwb_copy_file_name)
    for ind in range(len(session_interval)):
        session_name = session_interval[ind]
        position_name = position_interval[ind]
            
        data = {}
        p = {"nearby":control,
         "delta_t_minus":delta_t_minus, "delta_t_plus":delta_t_plus,
         "max_flag":max_flag, "proportion": proportion, "segment_only": segment_only,
         "multiple_CoM": multiple_CoM, "single_CoM": single_CoM, "first_CoM": first_CoM}
        
        (data["triggered_positions"], data["triggered_positions_baseoff"],
        data["triggered_decodes"], data["triggered_decodes_baseoff"], data["triggered_decodes_abs"],
        data["triggered_trial_info"]) = find_triggered_session(nwb_copy_file_name,
                                                        session_name, position_name, decode_options = decode_options, last_CoM = False,
                                                        **p)
        

        if len(data["triggered_decodes"]) == 0 or np.all([len(element)==0 for element in data["triggered_decodes"]]):
            print(f"No change of mind trials found for session {session_name} in file {nwb_copy_file_name}. Skipping insertion.")
            continue
        
        time_triggered = [np.array(pos.index) for pos in data["triggered_positions"]]
        time_abs = [np.array(pos.index) for pos in data["triggered_positions_baseoff"]]
        data["time_triggered"] = time_triggered
        data["time_abs"] = time_abs
        
        key = write_data_to_analyis_nwb(data, "triggered", nwb_copy_file_name, table_name)
        # key already has field such as nwb_file_name, analysis_file_name
        key['proportion'] = proportion
        key['epoch'] = int(session_name[:2])
        p['decode_options'] = decode_options
        key['parameter_value'] = p
        key['parameter'] = parameter_name
        
        ChangeofMindTriggeredDecode.insert1(key, replace = True)
                                        
        
    # Open the file in binary write mode and dump the data
    # with open(file_path, 'wb') as file:
    #    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    # print(f"Data successfully pickled and saved to {file_path}")
    

def find_triggered_animal_save(animal, dates_to_plot, encoding_set = None, classifier_param_name = None,
                               parameter_name = None,
                               proportion = 0.1,
                               delta_t_minus = 0, delta_t_plus = 2,
                               max_flag = 0, control = False, segment_only = True,
                               multiple_CoM = True, single_CoM = True, first_CoM = True):
    # Specify the file path
    for d in dates_to_plot:
        # data
        p = {"control":control,
             "delta_t_minus":delta_t_minus, "delta_t_plus":delta_t_plus,
             "max_flag":max_flag, "proportion":proportion, "segment_only": segment_only,
             "multiple_CoM": multiple_CoM, "single_CoM": single_CoM, "first_CoM":first_CoM}
        save_triggered_position_decode_day(animal, d, encoding_set, classifier_param_name, parameter_name = parameter_name, **p)
        
    return 1

def load_triggered_position_decode_session_spyglass(nwb_copy_file_name, epoch_num, parameter_name, proportion = 0.1,baseoff = True,
                                                    ):
    # load triggered position and decode
    key_pre = {"nwb_file_name": nwb_copy_file_name, "epoch":epoch_num,
                    "proportion":proportion, "parameter": parameter_name}
    query = ChangeofMindTriggeredDecode & key_pre
    if len(query) == 0:
        print("No triggered decode found for ", key_pre)
        return {}
    
    parameters = query.fetch1("parameter_value")
            
    loaded_data = ChangeofMindTriggeredDecode().fetch1_dataframe(key_pre)
    (triggered_positions, triggered_positions_abs,
     triggered_decodes,
        triggered_times_triggered, triggered_times_abs,
        triggered_trial_infos) = (
            loaded_data["triggered_positions_baseoff"], loaded_data["triggered_positions"],
            loaded_data["triggered_decodes_baseoff"],
            loaded_data["time_triggered"], loaded_data["time_abs"], 
            loaded_data["triggered_trial_info"],
        )
    if not baseoff:
        print("loading not based off decode")
        triggered_decodes = loaded_data["triggered_decodes"]
    
    triggered_positions_output, triggered_decodes_output, triggered_positions_abs_output= [], [], []
    triggered_trial_infos_output = []
    # make triggered_positions a dataframe, with index of triggered_times_abs
    for tp_ind in range(len(triggered_positions)):
        if len(triggered_decodes[tp_ind]) == 0:
            continue
            
        triggered_positions_ = pd.DataFrame({
            'linear_position': triggered_positions[tp_ind],
            }, index = triggered_times_abs[tp_ind])
        
        triggered_decodes_= pd.DataFrame(triggered_decodes[tp_ind],
            index = triggered_times_abs[tp_ind])
        
        triggered_positions_abs_ = pd.DataFrame({
                'linear_position': triggered_positions_abs[tp_ind],
            }, index = triggered_times_triggered[tp_ind])

        triggered_positions_output.append(triggered_positions_)
        triggered_decodes_output.append(triggered_decodes_)
        triggered_positions_abs_output.append(triggered_positions_abs_)
        triggered_trial_infos_output.append(triggered_trial_infos[tp_ind])
        
    
    data = {"triggered_positions_baseoff":triggered_positions_output,
            "triggered_decodes_baseoff":triggered_decodes_output,
            "triggered_positions":triggered_positions_abs_output,
            "triggered_trial_info":triggered_trial_infos_output,
            }
    return data


def load_triggered_position_decode_day(animal, d, encoding_set, classifier_param_name,
                            proportion = 0.1, control = False, delta_t_minus = 0, delta_t_plus = 2,
                            max_flag = 0, segment_only = True,
                            multiple_CoM = True, single_CoM = True, first_CoM = True):
    
    p = {"control":control,
         "delta_t_minus":delta_t_minus, "delta_t_plus":delta_t_plus,
         "max_flag":max_flag, "proportion": proportion,
         "segment_only":segment_only,
         "multiple_CoM":multiple_CoM,
         "single_CoM":single_CoM,
         "first_CoM":first_CoM}
    save_name = return_save_name(animal, encoding_set, classifier_param_name, d, **p)
    file_path = output_folder + save_name + '.pkl'
    
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
        print(f"Successfully loaded data from '{file_path}':")
    return loaded_data

def insert_ChangeofMindTheta(animal, d, parameter_name, parameter_name_control,
                             minimum_duration = 0.02,
                             proportion = 0.1, max_flag = 0, sd = 4, use_hpd = True):
    

    nwb_file_name = animal.lower() + d + '.nwb'
    nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
    print(nwb_copy_file_name)
    session_interval, position_interval = runSessionNames(nwb_copy_file_name)
    if "all_maze" not in parameter_name:
        encoding_set = "2Dheadspeed_above_4"
    else:
        encoding_set = "all_maze"
        
        
    positions_control = []
    decodes_control = []
    time_abs_control = []
    loaded_datas = []
    session_names = []
    for ind in range(len(session_interval)):  
            
        session_name = session_interval[ind]
        position_name = position_interval[ind]
        epoch_num = int(session_name[:2])
        
        loaded_data = load_triggered_position_decode_session_spyglass(nwb_copy_file_name, epoch_num,
                                                               parameter_name, proportion)
        
        loaded_data_control = load_triggered_position_decode_session_spyglass(nwb_copy_file_name, epoch_num,
                                                               parameter_name_control, proportion)
        
        if len(loaded_data)==0 or len(loaded_data["triggered_decodes_baseoff"]) == 0 or len(loaded_data_control) == 0:
            print(f"Session {session_name} has no change of mind trials.")
            continue
        
        # merge all sessions' control data
        for loaded in loaded_data_control["triggered_positions_baseoff"]:
            positions_control.append(loaded)
        for loaded in loaded_data_control["triggered_decodes_baseoff"]:
            decodes_control.append(loaded)
        for loaded in loaded_data_control["triggered_positions_baseoff"]:
            time_abs_control.append(np.array(loaded.index))
        
        # merge all sessions' data
        loaded_datas.append(loaded_data)
        session_names.append(session_name)
        
        # # load control dataset
        # paramters = {"proportion":proportion,
        #              "delta_t_minus":delta_t_minus,
        #              "delta_t_plus":delta_t_minus,
        #              "max_flag":max_flag,
        #              "segment_only":segment_only,
        #              "multiple_CoM":multiple_CoM, "single_CoM":single_CoM, "first_CoM":first_CoM
        #              }
        # loaded_data_control = load_triggered_position_decode_day(animal, d, encoding_set, classifier_param_name,
        #                                                      control = True,
        #                                                      **paramters)

        # # load dataset
        # loaded_data = load_triggered_position_decode_day(animal, d, encoding_set, classifier_param_name,
        #                                                      control = False,
        #                                                      **paramters)
        # classify long/short theta and calculate deviation
    
    
    if len(decodes_control) == 0:
        # no change of mind trials
        return 1
    
    # form a null model between deocde and position
    # restrict to movement time
    position_infos = load_day_position_info(animal, d)
    decodes = load_day_decode(animal, d,
                              classifier_param_name = "default_decoding_gpu_4armMaze",
                              encoding_set = encoding_set)
    
    if use_hpd:
        (positions_control, decodes_control, time_abs_control, _) = return_concentrated_hpd_moments(
            positions_control, decodes_control, time_abs_control, decodes)
    
    (positions_control_mobility, decodes_control_mobility, _, _2) = return_mobility_movements(
        positions_control, decodes_control, time_abs_control, position_infos)
    
    
    #gaussian_process, _ , _2, quantile_995 = form_null_model(positions_control_mobility, decodes_control_mobility)
    gaussian_process, _ , _2, gaussian_process_CI = form_null_model_full(positions_control_mobility, decodes_control_mobility)
    
    
        
    for loaded_data, session_name in zip(loaded_datas, session_names):
        
        triggered_positions = loaded_data["triggered_positions_baseoff"]
        triggered_decodes = loaded_data["triggered_decodes_baseoff"]
        triggered_trial_info = loaded_data["triggered_trial_info"]
        time_abs = [np.array(loaded.index) for loaded in triggered_positions]
        if use_hpd:
            (triggered_positions, triggered_decodes, time_abs, _) = return_concentrated_hpd_moments(
                triggered_positions, triggered_decodes, time_abs,
                decodes)
        
        (triggered_positions, triggered_decodes, _, _2) = return_mobility_movements(
            triggered_positions, triggered_decodes, time_abs, position_infos)
        
        (trials_long_theta, inds, dev_long_theta, intervals_long_theta,
        trials_short_theta, inds_shorts, dev_short_theta, intervals_short_theta) = find_large_position_minus_decode_trials(
                triggered_trial_info, 
                triggered_positions, triggered_decodes,
                gaussian_process,gaussian_process_CI,
                return_interval = True,
                minimum_duration = minimum_duration, debug = False,
                sd = sd)
    
        # insert into spyglass 
        
  
        # load Change of Mind info
        query = {"nwb_file_name":nwb_copy_file_name,
            "epoch":int(session_name[:2]),
            "proportion": proportion}
        q_result = ChangeofMind() & query
        if len(q_result) == 0:
            continue
        info = ChangeofMind().fetch1_dataframe(query)
        info2 = info.copy()

        # initialization
        info2.insert(6,'long_theta',[False for i in range(len(info2))])
        info2.insert(7,'theta_dev',[[] for i in range(len(info2))])
        info2.insert(8,'short_theta_dev',[[] for i in range(len(info2))])
        info2.insert(9,'long_theta_intervals',[[] for i in range(len(info2))])
        info2.insert(10,'short_theta_intervals',[[] for i in range(len(info2))])
        info2.insert(11,'change_of_mind_num',[[] for i in range(len(info2))])

        for ind in range(len(trials_short_theta)):
            info = trials_short_theta[ind]
            t = info[-2]
            info2.at[t,'short_theta_dev'] += dev_short_theta[ind]
            info2.at[t,'short_theta_intervals'] += intervals_short_theta[ind]
        
        for ind in range(len(trials_long_theta)):
            info = trials_long_theta[ind]
            t = info[-2]
                ##### this trial has been parsed during short theta parsing.
                # due to the multiple change of minds, some other events could have theta.
                #if info2.loc[t,'theta_dev'] > 0:
                #    assert 1 == 0
                #####
            info2.loc[t,'long_theta'] = True
            info2.at[t,'theta_dev'] += dev_long_theta[ind]
            info2.at[t,'long_theta_intervals'] += intervals_long_theta[ind]
                
            # Mar 24, 2026, by Shijie. This is actually not correct, because if there are multiple change of minds, the second one could also have long theta. But for now I will just use this to roughly count the number of change of minds.
            # Do not use change_of_mind_num for rigorous analysis
            if len(info2.at[t,'change_of_mind_num']) == 0: 
                info2.at[t,'change_of_mind_num'] += [0 for _ in range(len(intervals_long_theta[ind]))]
            else:
                change_of_mind_num = np.max(info2.loc[t,'change_of_mind_num']) + 1
                info2.at[t,'change_of_mind_num'] += [change_of_mind_num for _ in range(len(intervals_long_theta[ind]))]
                
        
        # Insert into analysis nwb file
        # nwb_analysis_file = AnalysisNwbfile()
        # q["analysis_file_name"] = AnalysisNwbfile().create(q["nwb_file_name"])
        # pandas_id = nwb_analysis_file.add_nwb_object(
        #     analysis_file_name=q["analysis_file_name"],
        #     nwb_object=info2,
        # )
        # nwb_analysis_file.add(
        #     nwb_file_name=q["nwb_file_name"],
        #     analysis_file_name=q["analysis_file_name"],
        # )
        query["parameter"] = parameter_name
        query["local_parameter"] = f"dur_{minimum_duration}_sd_{sd}_hpd{use_hpd}"
        query["pandas"] = info2.to_dict()
                
        ChangeofMindTheta().insert1(query, replace = True)
        #AnalysisNwbfile().log(q, table=ChangeofMind().full_table_name)
        
    return 1
        
def load_day_position_info(animal, day):
    # returns each session's position info
    positions = []
    
    nwb_file_name = animal.lower() + day + '.nwb'
    nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
    
    # key for everything
    keys = (IntervalPositionInfo() & {'nwb_file_name':nwb_copy_file_name,
                          'position_info_param_name':'default_decoding'}).fetch(as_dict = True)
    for key in keys:
        k = {"position_info_param_name": key["position_info_param_name"],
             "nwb_file_name": key["nwb_file_name"],
             "interval_list_name": key["interval_list_name"]}
        position_info = (IntervalPositionInfo() & k).fetch1_dataframe()
        positions.append(position_info)
    return positions

from spyglass.shijiegu.helpers import interpolate_to_new_time
from spyglass.shijiegu.decodeQuality import hpd, return_low_hpd_time
def return_mobility_movements(position_pre, decode_pre, time_pre, position_infos, session_info = None, return_speed = False):
    positions = []
    decodes = []
    times = []
    infos = []
    position_infos_out = []
    for ind in range(len(position_pre)):
        position_pre_ = position_pre[ind]
        decode_pre_ = decode_pre[ind]
        time_pre_ = time_pre[ind]

        if len(decode_pre_) == 0:
            continue
        
        # find the position info that correspond to this interval
        if session_info is None:
            for position_info in position_infos:
                if time_pre_[0] >= position_info.index[0] and time_pre_[-1] <= position_info.index[-1]:
                    break
        else:
            position_info = position_infos[session_info[ind][0]]
    
        position_info_subset = interpolate_to_new_time(position_info, time_pre_)
        speed_pre_ = np.array(position_info_subset.head_speed)
        boolean = speed_pre_ < 4
        
        
        # find the position info that correspond to this interval
        position_pre_[boolean] = np.nan
        decode_pre_[boolean] = np.nan
        time_pre_[boolean] = np.nan
        
        positions.append(position_pre_)
        decodes.append(decode_pre_)
        times.append(time_pre_)
        position_infos_out.append(speed_pre_)
        if session_info is not None:
            infos.append(session_info[ind])
    if return_speed:
        return positions, decodes, times, infos, position_infos_out
    else:
        return positions, decodes, times, infos

def return_concentrated_hpd_moments(position_pre, decode_pre, time_pre, all_decodes, session_info = None):
    positions = []
    decodes = []
    times = []
    infos = []
    for ind in range(len(position_pre)):
        position_pre_ = position_pre[ind]
        decode_pre_ = decode_pre[ind]
        if len(decode_pre_) == 0:
            continue
        time_pre_ = time_pre[ind]
        
        # find the position info that correspond to this interval
        if session_info is not None:
            decode_ =  all_decodes[session_info[ind][0]]
        else:
            for decode_ in all_decodes:
                if time_pre_[0] >= decode_.time[0] and time_pre_[-1] <= decode_.time[-1]:
                    break
        
        decode_interp = decode_.interp(time=time_pre_, method="nearest")
        decode_subset = return_low_hpd_time(decode_interp)

        if len(decode_subset.time) == 0:
            continue
        
        # find the position info that correspond to this interval
        boolean = ~np.isin(time_pre_, np.array(decode_subset.time))
        position_pre_[boolean] = np.nan
        decode_pre_[boolean] = np.nan
        time_pre_[boolean] = np.nan
        
        positions.append(position_pre_)
        decodes.append(decode_pre_)
        times.append(time_pre_)
        if session_info is not None:
            infos.append(session_info[ind])
        
    return positions, decodes, times, infos
    

def return_triggered_2d_position(triggered_positions, trial_infos):
    
    positions_2D = []
    session_name = None
    for ind in range(len(triggered_positions)):
    
        position_abs = triggered_positions[ind]
        trial_info = trial_infos[ind]
        
        # find time
        times = position_abs.index
    
        nwb_copy_file_name, session_name_, trial, arm = trial_info
        if session_name_ != session_name:
            session_name = session_name_
            
            # load session's 2D position
            position_name = (EpochPos() & {"nwb_file_name": nwb_copy_file_name, 
                   "epoch_name":session_name}).fetch1("position_interval")
            
            position_info = (IntervalPositionInfo() & {
                'nwb_file_name':nwb_copy_file_name,
                'interval_list_name':position_name,
                'position_info_param_name':'default_decoding'}).fetch1_dataframe()
    
        position_info_subset = select_subset_helper_pd(position_info,(times[0],times[-1]))
        positions_2D.append(position_info_subset)

    return positions_2D

def return_triggered_mua(triggered_positions, triggered_abs_positions, trial_infos):
    
    mua_list = []
    mean, sd = [], []
    t0s = []
    session_name = None
    for ind in range(len(triggered_positions)):
    
        position = triggered_positions[ind]
        position_abs = triggered_abs_positions[ind]
        trial_info = trial_infos[ind]
        
        # find time
        times = position_abs.index
    
        nwb_copy_file_name, session_name_, trial, arm = trial_info
        if session_name_ != session_name:
            session_name = session_name_
            
            # load MUA
            key = {"nwb_file_name":nwb_copy_file_name, "interval_list_name":session_name}
            mua_path = (MUA() & key).fetch1("mua_trace")
            (m, s) = ((MUA() & key).fetch1("mean"), (MUA() & key).fetch1("sd"))
            mua_xr = xr.open_dataset(mua_path)
            
        # figure out t0, turn around time
        t0 = position_abs.index[np.argmin(np.abs(position.index - 0))]
            
        mua_subset = select_subset_helper(mua_xr,(times[0],times[-1]))
        mua_list.append(mua_subset)
        mean.append(m)
        sd.append(s)
        t0s.append(t0)

    return mua_list, mean, sd, t0s


def load_day_decode(animal, day, classifier_param_name = "default_decoding_gpu_4armMaze", encoding_set = "2Dheadspeed_above_4"):
    decodes = []
    
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
        decodes.append(decode)

    return decodes
    
