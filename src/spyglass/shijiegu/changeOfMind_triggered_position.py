import pickle
import numpy as np
import xarray as xr
from spyglass.shijiegu.changeOfMind_triggered import find_large_position_minus_decode_trials, find_triggered_animal
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.Analysis_SGU import ChangeofMind, ChangeofMindTheta, EpochPos, MUA
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.shijiegu.ripple_add_replay import select_subset_helper_pd, select_subset_helper

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

def save_triggered_position_decode_day(animal, d, encoding_set, classifier_param_name,
                                       proportion = 0.1,
                                       delta_t_minus = 0, delta_t_plus = 2,
                                       max_flag = 0, control = False, segment_only = True,
                                       multiple_CoM = True, single_CoM = True, first_CoM = True):
    p = {"control":control,
         "delta_t_minus":delta_t_minus, "delta_t_plus":delta_t_plus,
         "max_flag":max_flag, "proportion": proportion, "segment_only": segment_only,
         "multiple_CoM": multiple_CoM, "single_CoM": single_CoM, "first_CoM": first_CoM}
        
    save_name = return_save_name(animal, encoding_set, classifier_param_name, d, **p)
    file_path = output_folder + save_name + '.pkl'
    data = {}
        
    (data["triggered_positions"], data["triggered_positions_baseoff"],
     data["triggered_decodes"], data["triggered_decodes_baseoff"], data["triggered_decodes_abs"],
     data["triggered_trial_info"]) = find_triggered_animal(animal,[d], **p)
        
    # Open the file in binary write mode and dump the data
    with open(file_path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data successfully pickled and saved to {file_path}")

def find_triggered_animal_save(animal, dates_to_plot, encoding_set, classifier_param_name,
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
        save_triggered_position_decode_day(animal, d, encoding_set, classifier_param_name, **p)
        
    return 1

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

def insert_ChangeofMindTheta(animal, d, encoding_set, classifier_param_name, segment_only = True,
                             multiple_CoM = True, single_CoM = True, first_CoM = True,
                             proportion = 0.1, delta_t_minus = 0, delta_t_plus = 2, max_flag = 0):
    # load control dataset
    paramters = {"proportion":proportion,
                 "delta_t_minus":delta_t_minus,
                 "delta_t_plus":delta_t_minus,
                 "max_flag":max_flag,
                 "segment_only":segment_only,
                 "multiple_CoM":multiple_CoM, "single_CoM":single_CoM, "first_CoM":first_CoM
                 }
    loaded_data_control = load_triggered_position_decode_day(animal, d, encoding_set, classifier_param_name,
                                                         control = True,
                                                         **paramters)

    # load dataset
    loaded_data = load_triggered_position_decode_day(animal, d, encoding_set, classifier_param_name,
                                                         control = False,
                                                         **paramters)
    # classify long/short theta and calculate deviation
    if len(loaded_data["triggered_trial_info"]) == 0:
        # no change of mind trials
        return None
    (trials_long_theta, inds, dev_long_theta, intervals_long_theta,
     trials_short_theta, inds_shorts, dev_short_theta) = find_large_position_minus_decode_trials(
            loaded_data["triggered_trial_info"], 
            loaded_data["triggered_positions_baseoff"], loaded_data["triggered_decodes_baseoff"],
            loaded_data_control["triggered_positions_baseoff"], loaded_data_control["triggered_decodes_baseoff"],
            return_interval = True)
    
    # insert into spyglass 
    nwb_file_name = animal.lower() + d + '.nwb'
    nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
    session_interval, position_interval = runSessionNames(nwb_copy_file_name)
            
    for session_ind in range(len(session_interval)):
        session, pos_name = session_interval[session_ind], position_interval[session_ind]
                
        # load Change of Mind info
        q = {"nwb_file_name":nwb_copy_file_name,
            "epoch":int(session[:2]),
            "proportion": proportion}
        q_result = ChangeofMind() & q
        if len(q_result) == 0:
            continue
        info = ChangeofMind().fetch1_dataframe(q)
        info2 = info.copy()

        # initialization
        info2.insert(6,'long_theta',[False for i in range(len(info2))])
        info2.insert(7,'theta_dev',[[] for i in range(len(info2))])
        info2.insert(8,'long_theta_intervals',[[] for i in range(len(info2))])
        info2.insert(9,'change_of_mind_num',[[] for i in range(len(info2))])

        for ind in range(len(trials_short_theta)):
            info = trials_short_theta[ind]
            if info[0] == nwb_copy_file_name and info[1] == session:
                print("trial",info[2])
                t = info[2]
                info2.loc[t,'theta_dev'] = dev_short_theta[ind]
        
        for ind in range(len(trials_long_theta)):
            info = trials_long_theta[ind]
            if info[0] == nwb_copy_file_name and info[1] == session:
                print("trial",info[2])
                t = info[2]
                ##### this trial has been parsed during short theta parsing.
                # due to the multiple change of minds, some other events could have theta.
                #if info2.loc[t,'theta_dev'] > 0:
                #    assert 1 == 0
                #####
                info2.loc[t,'long_theta'] = True
                info2.at[t,'theta_dev'] += dev_long_theta[ind]
                info2.at[t,'long_theta_intervals'] += intervals_long_theta[ind]
                
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
        q["analysis_file_name"] = ""
        q["delta_t_minus"] = delta_t_minus
        q["delta_t_plus"] = delta_t_plus
        q["max_flag"] = max_flag
        q["pandas"] = info2.to_dict()
                
        ChangeofMindTheta().insert1(q, replace = True)
        #AnalysisNwbfile().log(q, table=ChangeofMind().full_table_name)
        
    return 1
        
        
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
    
