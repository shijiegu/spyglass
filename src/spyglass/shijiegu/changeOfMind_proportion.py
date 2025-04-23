import spyglass as nd
import pandas as pd
import numpy as np
import xarray as xr
from scipy import stats
from scipy import linalg
from scipy import ndimage
import matplotlib.pyplot as plt
from spyglass.common import (Session, IntervalList,LabMember, LabTeam, Raw, Session, Nwbfile,
                            Electrode,LFPBand,interval_list_intersect)
from spyglass.common import TaskEpoch
from spyglass.spikesorting.v0 import (SortGroup, 
                                    SpikeSortingRecording,SpikeSortingRecordingSelection)
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.common.common_position import IntervalPositionInfo, RawPosition, IntervalLinearizedPosition, TrackGraph

from ripple_detection.core import segment_boolean_series

from spyglass.shijiegu.Analysis_SGU import TrialChoice,EpochPos,MUA,get_linearization_map,TrialChoiceChangeofMind
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.ripple_add_replay import plot_decode_spiking,select_subset_helper,select_subset_helper_pd
from spyglass.shijiegu.changeOfMind import (find_turnaround_time, findProportion,
            find_trials, normalize, load_epoch_data_wrapper, find_direction, find_trials_animal)
from spyglass.shijiegu.changeOfMindRipple import triggered_ripple_animal
from spyglass.shijiegu.load import load_decode
from spyglass.shijiegu.pairwiseDecode import behavior_transitions_count

def session_long_theta_trials(replay_trials_tuple,nwb_copy_file_name,session_name,log_df):
    CoMMaxProportion = []
    for tup in replay_trials_tuple:
        if tup[0] == nwb_copy_file_name and tup[1] == session_name:
            max_prop = log_df.loc[tup[2]].CoMMaxProportion
            #if len(max_prop) > 0:
            CoMMaxProportion.append(max_prop)
    return CoMMaxProportion

    
    
def find_proportion_by_theta(nwb_copy_file_name, session_name, proportion, replay_trials, replay_trials_non):
    """_summary_

    Args:
        nwb_copy_file_name (_type_): _description_
        session_name (_type_): _description_
        replay_trials (dict):
            with long theta
            replay_trials_non["lewis"]['20240105'] = [
                ('lewis20240105_.nwb', '02_Rev2Session1', 101, 3),
                ('lewis20240105_.nwb', '04_Rev2Session2', 5, 1),]
        replay_trials_non (dict):
            same as replay_trials, without long theta

    Returns:
        _type_: _description_
    """
    
    animal = nwb_copy_file_name[:5]
    d = nwb_copy_file_name[5:13]
    print(nwb_copy_file_name,session_name)

    
    # find all transitions in which long theta sequence happens
    
    log_df = pd.read_pickle( (TrialChoiceChangeofMind() & {"nwb_file_name": nwb_copy_file_name,
                                                            "epoch":int(session_name[:2]),
                                                            "proportion":str(proportion)}).fetch1("change_of_mind_info") )
    
    proportion_long_theta = session_long_theta_trials(replay_trials[animal][d],
                                       nwb_copy_file_name,
                                       session_name,log_df)
    proportion_short_theta = session_long_theta_trials(replay_trials_non[animal][d],
                                       nwb_copy_file_name,
                                       session_name,log_df)

    return proportion_long_theta, proportion_short_theta