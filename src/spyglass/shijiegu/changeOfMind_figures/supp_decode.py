import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import logging
import os
import cupy as cp

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')
from spyglass.decoding.v0.clusterless import ClusterlessClassifierParameters

from spyglass.shijiegu.Analysis_SGU import (EpochPos,TrialChoice,Decode,
    DecodeIngredients,DecodeResultsLinear,MUA,DecodeIngredientsLikelihood,DecodeResultsLinear)
from spyglass.linearization.v0.main import IntervalLinearizedPosition
from spyglass.common.common_position import IntervalPositionInfo
from spyglass.shijiegu.decodeHelpers import session2position_name

window_size_dict = {"default_decoding_gpu_4armMaze_W20msO10ms": 0.02,
               "default_decoding_gpu_4armMaze_W40msO20ms": 0.04,
               "default_decoding_gpu_4armMaze_W80msO10ms": 0.08,
               "default_decoding_gpu_4armMaze_W160msO10ms": 0.16,
              }

overlap_size_dict = {"default_decoding_gpu_4armMaze_W20msO10ms": 0.01,
               "default_decoding_gpu_4armMaze_W40msO20ms": 0.02,
               "default_decoding_gpu_4armMaze_W80msO10ms": 0.01,
               "default_decoding_gpu_4armMaze_W160msO10ms": 0.01,
              }


def return_diff_file(nwb_copy_file_name, session_name, classifier_param_name):
    window_size = window_size_dict[classifier_param_name]#int(classifier_param_name.split("_")[-1][1:3]) * 0.001 
    overlap_size = overlap_size_dict[classifier_param_name] #int(classifier_param_name.split("_")[-1][6:8]) * 0.001
    #pos_name = session2position_name(nwb_copy_file_name, session_name)

    # load decode
    decode_path = (DecodeResultsLinear & {"nwb_file_name":nwb_copy_file_name,
                           "interval_list_name":session_name,
                           "classifier_param_name":classifier_param_name}).fetch1("posterior")
    decode = xr.open_dataset(decode_path)
    
    ## load LinearPosition
    pos1d = pd.read_csv((DecodeIngredientsLikelihood & {"nwb_file_name":nwb_copy_file_name,
                                   "interval_list_name":session_name,
                                  "window_size":window_size,
                                  "overlap_size":overlap_size}).fetch1("position_1d"))
    
    pos2d = pd.read_csv((DecodeIngredientsLikelihood & {"nwb_file_name":nwb_copy_file_name,
                                   "interval_list_name":session_name,
                                  "window_size":window_size,
                                  "overlap_size":overlap_size}).fetch1("position_2d"))

    diff_by_arm = {}
    for arm in [1,2,3,4]:
        diff = return_diff(arm, decode, pos1d, pos2d)
        diff_by_arm[arm] = diff
    return diff_by_arm

# find all moving time, when rat in arm1
def return_diff(arm, decode, pos1d, pos2d):

    segment = arm + 5
    
    # arm selection
    ind = np.array(pos1d.track_segment_id).astype("int") == segment
    #ind = np.array(pos1d.track_segment_id) == segment
    pos1d_subset = pos1d[ind]
    pos2d_subset = pos2d[ind]

    
    # speed thresholding
    ind = pos2d_subset.head_speed >= 4
    pos1d_subset = pos1d_subset[ind]
    pos2d_subset = pos2d_subset[ind]
    decode_subset = decode.isel(time = pos2d_subset.index)

    # get max likelihood position for all time
    position_axis = np.array(decode.coords['position'])
    
    posterior_position_subset = decode_subset.likelihood.sum(dim='state')
    max_posterior_position = np.array(position_axis[posterior_position_subset.argmax(dim = 'position')])
    
    tracking = np.array(pos1d_subset.linear_position)
    
    # compare to animal location
    diff = np.abs(max_posterior_position - tracking)

    return diff