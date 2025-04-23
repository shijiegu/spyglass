import os
import cupy as cp
import numpy as np
import datajoint as dj
import spyglass as nd
import pandas as pd
import matplotlib.pyplot as plt
import json
import multiprocessing

# ignore datajoint+jupyter async warnings
import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=ResourceWarning)

from spyglass.common import (Session, IntervalList,LabMember, LabTeam, Raw, Session, Nwbfile,
                            Electrode,LFPBand,interval_list_intersect)
from spyglass.common import TaskEpoch
import spyglass.spikesorting.v0 as ss

from spyglass.spikesorting.v0 import (SortGroup, 
                                    SortInterval,
                                    SpikeSortingPreprocessingParameters,
                                    SpikeSortingRecording, 
                                    SpikeSorterParameters,
                                    SpikeSortingRecordingSelection,
                                    ArtifactDetectionParameters, ArtifactDetectionSelection,
                                    ArtifactRemovedIntervalList, ArtifactDetection,
                                      SpikeSortingSelection, SpikeSorting,
                                   CuratedSpikeSortingSelection,CuratedSpikeSorting,Curation,QualityMetrics)
from spyglass.spikesorting.v0.curation_figurl import CurationFigurl,CurationFigurlSelection
from spyglass.spikesorting.v0.spikesorting_curation import MetricParameters,MetricSelection,QualityMetrics
from spyglass.spikesorting.v0.spikesorting_curation import WaveformParameters,WaveformSelection,Waveforms
from spyglass.common.common_position import IntervalPositionInfo, IntervalPositionInfoSelection,IntervalLinearizedPosition

from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from pprint import pprint
from spyglass.shijiegu.ripple_detection import removeDataBeforeTrial1
from spyglass.shijiegu.helpers import interpolate_to_new_time
from spyglass.shijiegu.placefield import place_field,placefield_to_peak1dloc
from spyglass.shijiegu.Analysis_SGU import DecodeResultsLinear
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
warnings.filterwarnings('ignore')

from spyglass.shijiegu.curation_manual import load_waveforms
from spyglass.shijiegu.placefield import place_field, placefield_to_peak1dloc, cell_by_arm

from spyglass.shijiegu.helpers import interval_union
from spyglass.shijiegu.Analysis_SGU import TrialChoice,RippleTimes,EpochPos,ExtendedRippleTimes,RippleTimesWithDecode
from spyglass.shijiegu.load import load_run_sessions
from spyglass.shijiegu.singleUnit import (do_mountainSort,electrode_unit,
            RippleTime2FiringRate,findWaveForms,RippleTime2Index,get_nwb_units)

from spyglass.shijiegu.load import load_theta_maze
from spyglass.shijiegu.Analysis_SGU import get_linearization_map,segment_to_linear_range
from spyglass.shijiegu.theta import return_firing
from spyglass.shijiegu.Analysis_SGU import get_linearization_map
from scipy.stats import beta

from spyglass.shijiegu.ripple_add_replay import plot_decode_sortedSpikes

# This script has all the helper functions needed to plot sorted spike decode.
    
def color_cells_by_place_direction(cells, placefield_peak, spike_count_by_arm_direction):
    # cells with peak firing location in home/center are in black:
    # otherwise, cells with <0.2 inbound/outbound ratio in blue
    #            cells with >0.8 inbound/outbound ratio in orange
    
    cell_list_by_arm, _ = cell_by_arm([placefield_peak[k] for k in placefield_peak], cells)
    cell_color = {}
    for cell in cells:
        # first find its peak arm
        cell_color[cell] = 'k'
        arm = find_peak_arm(cell,cell_list_by_arm)
        if np.isnan(arm):
            pass
        else:
            # find the direction in the peak arm
            spike_counts_arm = spike_count_by_arm_direction[cell][:,arm]
                # matrix of size 2 by 4 arms, row 1 is outbound, row 2 is inbound
            if (spike_counts_arm[0] / spike_counts_arm[1]) < 0.2:
                cell_color[cell] = 'C0'
            elif (spike_counts_arm[0] / spike_counts_arm[1]) > 0.8:
                cell_color[cell] = 'C1'
    return cell_color
    
  
def place_field_direction(nwb_copy_file_name,session_name,position_name,curation_id = 2):
    sort_group_ids = np.unique((Curation() & {'nwb_file_name': nwb_copy_file_name,"curation_id":curation_id}).fetch("sort_group_id"))
    nwb_units_all = get_nwb_units(
            nwb_copy_file_name,session_name,sort_group_ids,curation_id = curation_id)
    
    # sometimes a tetrode has no cells passing the manual curation
    # so we need to update here.
    sort_group_ids = list(nwb_units_all.keys())
    
    cells = []
    smoothed_placefield = {}
    placefield_peak = {}
    spike_count = {}
    betaPdfs, means = ({},{})
    spike_count_by_arm_direction = {}
    time_spent_by_arm_direction = {}
    
    for a in [0,1,2,3,4]:
        (betaPdfs[a], means[a]) = ([], [])

    for sort_group_id in sort_group_ids:

        units = list(nwb_units_all[sort_group_id].index)
        
        for u in units:
            num_spikes = len(nwb_units_all[sort_group_id].loc[u].spike_times)
            if num_spikes > 10000 or num_spikes < 10:
                # remove putative noise unit
                continue

            # 1. Place Field
            (smoothed_placefield_, peak_firing_rate,
            xbins, ybins, spike_count[(sort_group_id,u)], total_spike_count) = place_field(nwb_copy_file_name,
                                                                            session_name, position_name,
                                                                            sort_group_id, u,
                                                                            BINWIDTH = 2, sigma = 2,
                                                                            curation_id = 1,
                                                                            nwb_units = nwb_units_all[sort_group_id].loc[[u]])
                    
            # track time spike count
            if spike_count[(sort_group_id,u)] > 10000 or spike_count[(sort_group_id,u)] < 10:
                # remove putative noise unit
                continue
            cells.append((sort_group_id,u))
            smoothed_placefield[(sort_group_id,u)] = smoothed_placefield_

            # 2. Place Field Peak

            placefield_peak[(sort_group_id,u)] = placefield_to_peak1dloc('4 arm lumped 2023',
                                                                        smoothed_placefield[(sort_group_id,u)], ybins, xbins)

            # 3. Beta distribution
            # get linearized location and 2D position
            pos1d = (IntervalLinearizedPosition() & {'nwb_file_name': nwb_copy_file_name,
                                                            'track_graph_name':'4 arm lumped 2023',
                                                            'interval_list_name':position_name,
                                                            'position_info_param_name':'default'}).fetch1_dataframe()

            pos2d = (IntervalPositionInfo() & {'nwb_file_name': nwb_copy_file_name,
                                                            'interval_list_name':position_name,
                                                            'position_info_param_name':'default'}).fetch1_dataframe()
            firing, firing_count, time_spent = return_firing(pos1d,pos2d,nwb_units_all,(sort_group_id,u))
            spike_count_by_arm_direction[(sort_group_id,u)] = firing_count
            time_spent_by_arm_direction[(sort_group_id,u)] = time_spent
            for a in [0,1,2,3,4]:
                betaPdf, mean = return_beta(a,firing_count)
                betaPdfs[a].append(betaPdf)
                means[a].append(mean)
    return cells, smoothed_placefield, placefield_peak, spike_count_by_arm_direction, time_spent_by_arm_direction, betaPdfs, means
        
        
def isin(A,B):
    # A is a cell (electrode, unit)
    # B is list of cells
    A = np.array([A])
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [A.dtype]}
    
    C = np.intersect1d(A.view(dtype), B.view(dtype))
    return len(C) > 0

def find_peak_arm(cell,cell_list_by_arm):
    segs = ['home','arm1','arm2','arm3','arm4']
    for segs_ind in range(5):
        if isin(cell,cell_list_by_arm[segs[segs_ind]]):
            return segs_ind

    return np.nan    
    
def return_beta(arm,firing_count):
    # beta, arm is 1 indexed
    beta_a = firing_count[0][arm]+1
    beta_b = firing_count[1][arm]+1
    betaPdf = beta(beta_a, beta_b)
    mean = beta.stats(beta_a, beta_b, moments='m')

    return betaPdf, mean