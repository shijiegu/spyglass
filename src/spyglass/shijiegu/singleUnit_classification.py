import elephant
import quantities as pq
import numpy as np
import neo
from elephant.conversion import BinnedSpikeTrain
from spyglass.spikesorting.v0.spikesorting_curation import WaveformParameters, WaveformSelection, Waveforms
import pandas as pd

def get_spike_info(nwb_units_all, nwb_copy_file_name, session_name):
    firing_rate = {}
    ac_mean = {}
    width = {}
    
    for e in nwb_units_all.keys():
        print(f"working on electrode {e}")
        key = {'nwb_file_name':nwb_copy_file_name,
               'sort_interval_name':session_name,
               "sorter":"mountainsort4",
               'sort_group_id':e,
               "curation_id":1}
        
        waveform_extractor = Waveforms().load_waveforms(key)
        
        for u in nwb_units_all[e].index:
            print(f"working on electrode {e} unit {u}")
            nwb_unit = nwb_units_all[e].loc[u]
            
            firing_rate[(e,u)] = return_firing_rate(nwb_unit)
            ac_mean[(e,u)], _ = return_AC_mean(nwb_unit)

            waveforms = waveform_extractor.get_waveforms(u)
            width[(e,u)] = return_spike_width(waveforms)
    return firing_rate, ac_mean, width


def return_firing_rate(nwb_unit):
    obs_intervals = nwb_unit.obs_intervals
    T = np.sum(np.diff(obs_intervals,axis = 1))
    num_spikes = nwb_unit.num_spikes
    return num_spikes / T


def return_AC_mean(nwb_unit, bin_size = 0.001, max_range = 0.04):
    # bin_size: in seconds
    # max_range: in seconds
    
    
    spike_times = nwb_unit.spike_times * pq.s
    sort_interval = nwb_unit.sort_interval.ravel()
    t_start, t_stop = sort_interval
    
    # Create a SpikeTrain object
    spike_train_obj = neo.SpikeTrain(spike_times, t_start = t_start, t_stop = t_stop)
    
    binned_spiketrain_i = BinnedSpikeTrain(
           spike_train_obj,
           bin_size=bin_size * 1000 * pq.ms)
    
    num_window = int(max_range/bin_size)
    
    autocorr_hist, bins = elephant.spike_train_correlation.cross_correlation_histogram(
        binned_spiketrain_i, binned_spiketrain_i,
        window=[-num_window, num_window],
        border_correction=False,
        binary=False, kernel=None
    )

    mid_index = int((len(bins)-1)/2)
    autocorr_hist_positive = np.array(autocorr_hist)[mid_index:]
    bins_ms = bins[mid_index:] * bin_size * 1000
    P = autocorr_hist_positive/np.sum(autocorr_hist_positive) 
    ac_mean = np.sum(np.array([P[ind] * bins_ms[ind] for ind in range(len(P))]))

    return ac_mean, autocorr_hist

def return_spike_width(waveforms):
    
    # find mean waveform
    mean_waveforms = waveforms.mean(axis = 0)

    # find max peak channel
    max_channel = np.argmax(np.max(np.abs(mean_waveforms), axis = 0))
    
    mean_waveform = mean_waveforms[:,max_channel]

    # find max min peak difference
    # width = (np.argmin(mean_waveform) - np.argmax(mean_waveform)) * 1/30000
    width = len(np.argwhere(mean_waveform <= 0)) * 1/30000 * 1000 #return result in ms

    return width

THRESHOLD_FR = 5 #Hz
THRESHOLD_WIDTH = 0.3 #ms
THRESHOLD_AC = 11 #ms

def classify(firing_rate, ac_mean, width):
    classification = {}
    for cell in firing_rate.keys():
        fr = firing_rate[cell]
        ac = ac_mean[cell]
        w = width[cell]
        
        if fr < THRESHOLD_FR and ac < THRESHOLD_AC and w > THRESHOLD_WIDTH:
            classification[cell] = 'pyramidal'
        elif fr > THRESHOLD_FR and ac > THRESHOLD_AC and w < THRESHOLD_WIDTH:
            classification[cell] = 'interneuron'
        else:
            classification[cell] = 'unclassified'
    return classification

def result_to_dict(firing_rate, ac_mean, width, classification):
    result = {}
    for cell in firing_rate.keys():
        result[cell] = {'firing_rate': firing_rate[cell],
                        'ac_mean': ac_mean[cell],
                        'spike_width': width[cell],
                        'classification': classification[cell]}
    return result

def result_to_dataframe(result):
    df = pd.DataFrame.from_dict(result, orient = 'index')
    df.index.names = ['sort_group_id','unit_id']
    return df