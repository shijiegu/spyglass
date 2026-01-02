import os
import elephant
import quantities as pq
import numpy as np
import neo
from elephant.conversion import BinnedSpikeTrain
from spyglass.spikesorting.v0.spikesorting_curation import WaveformParameters, WaveformSelection, Waveforms
import pandas as pd
from spyglass.spikesorting.v0.spikesorting_curation import QualityMetrics
from spyglass.shijiegu.singleUnit import electrode_unit
import matplotlib.pyplot as plt
from spyglass.shijiegu.Analysis_SGU import SingleUnit

outputpath = "/home/shijiegu/Documents/spyglass/notebooks/Change of Mind Analysis/figure_single_cell/classification"

def fill_single_unit_classification(nwb_copy_file_name, session_name, insert = True):
    key = {"nwb_file_name": nwb_copy_file_name,
       "sorter":"mountainsort4",
       "sort_interval_name":session_name,
       "curation_id":0}

    # get all sort group ids and nwb units
    sort_group_ids = np.unique((QualityMetrics & key).fetch("sort_group_id"))
    sort_group_ids_with_good_cell = []
    nwb_units_all = {}
    for sort_group_id in sort_group_ids:
        nwb_units = electrode_unit(nwb_copy_file_name,session_name,sort_group_id,curation_id = 1)
        if nwb_units is None or len(nwb_units)==0:
            continue
        sort_group_ids_with_good_cell.append(sort_group_id)
        nwb_units_all[sort_group_id] = nwb_units
    print("sort groups are: ",sort_group_ids_with_good_cell)
    
    # get spike features
    firing_rate, ac_mean, width = get_spike_info(nwb_units_all, nwb_copy_file_name, session_name)
    
    # Classify
    # Pyramidal cells
    #  criterion from Skaggs 1996: 
    #  1. have a spike width (peak to valley) of at least 300 us; and 4)
    #  2. have an overall mean rate below 5 Hz during the recording session. 
    #

    # Interneuron a unit was required to
    #  1. have a spike width less than 300 us;
    #  2. fire with a mean rate above 5 Hz during the recording session.
    classification = classify(firing_rate, ac_mean, width)
    
    # make figure
    make_classification_figure(firing_rate, ac_mean, width, classification,
                               nwb_copy_file_name, session_name, outputpath)
    
    make_classification_figure2D(firing_rate, ac_mean, width, classification,
                               nwb_copy_file_name, session_name, outputpath)
    
    if not insert:
        return True
    # save to db
    result = result_to_dict(firing_rate, ac_mean, width, classification)
    df = result_to_dataframe(result)
    result_dict = df.to_dict()
    
    key = {"nwb_file_name": nwb_copy_file_name,
      "epoch":int(session_name[:2]),
      "sorter":"mountainsort4",
      "curation_id":1,
      "cell_type_pd": result_dict}
    
    SingleUnit().insert1(key, replace = True)
    
    return True
    
    
    
    

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
               "curation_id":0}
        
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

def make_classification_figure(firing_rate, ac_mean, width, classification,
                               nwb_copy_file_name, session_name, outputpath):
    # Create a figure and a 3D axes
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(projection='3d')

    # Plot the 3D scatter points
    type_p, type_i, type_k = 0, 0, 0
    for (e,u) in firing_rate.keys():
        fr = firing_rate[(e,u)]
        ac = ac_mean[(e,u)]
        w = np.abs(width[(e,u)])
        c = classification[(e,u)]
        if c == "pyramidal":
            color = 'C0'
            if type_p == 0:
                label = "pyramidal"
            else:
                label = None
            type_p += 1
        elif c == "interneuron":
            color = 'C1'
            if type_i == 0:
                label = "interneuron"
            else:
                label = None
            type_i += 1
        else:
            color = 'k'
            if type_k == 0:
                label = "not classified"
            else:
                label = None
            type_k += 1
        ax.scatter(w * 1000, ac, fr, s=30, color = color, label = label, alpha = 0.5)
    
    # Set labels
    ax.set_zlabel("firing rate (Hz)")
    ax.set_ylabel("autocorrelation mean (ms)")
    ax.set_xlabel('spike width (us)')

    # Display the plot
    ax.view_init(elev=20, azim=20) 
    plt.title(
        f'{nwb_copy_file_name} {session_name} \n pyramidal:{int(type_p)} - interneurons:{int(type_i)} - not classified: {int(type_k)}\n classifying cell types')

    #plt.tight_layout()
    plt.subplots_adjust(left=0.3) 
    #plt.subplots_adjust(bottom=0.1)
    plt.legend()
    #plt.show()

    plt.savefig(os.path.join(outputpath,
                            f"{nwb_copy_file_name}_{session_name}_3D.pdf"), transparent=False, bbox_inches='tight')
    
def make_classification_figure2D(firing_rate, ac_mean, width, classification,
                               nwb_copy_file_name, session_name, outputpath):
    # Create a figure and a 3D axes
    fig, axes = plt.subplots(1,2,figsize = (10,4))

    # Plot the 3D scatter points
    type_p, type_i, type_k = 0, 0, 0
    for (e,u) in firing_rate.keys():
        fr = firing_rate[(e,u)]
        w = np.abs(width[(e,u)])
        ac = ac_mean[(e,u)]
        c = classification[(e,u)]
        if c == "pyramidal":
            color = 'C0'
            if type_p == 0:
                label = "pyramidal"
            else:
                label = None
            type_p += 1
        elif c == "interneuron":
            color = 'C1'
            if type_i == 0:
                label = "interneuron"
            else:
                label = None
            type_i += 1
        else:
            color = 'k'
            if type_k == 0:
                label = "not classified"
            else:
                label = None
            type_k += 1
        
        axes[0].scatter(w*1000, fr, s=30, color = color, label = label, alpha = 0.5)
        axes[1].scatter(fr, ac, s=30, color = color, label = label, alpha = 0.5)

    # Set labels
    axes[0].set_xlabel('spike width (us)')
    axes[0].set_ylabel("firing rate mean (Hz)")

    axes[1].set_xlabel("firing rate mean (Hz)")
    axes[1].set_ylabel('mean ac (ms)')

    # Display the plot
    axes[0].set_title('classifying \n interneurons and pyramidal cells')


    plt.savefig(os.path.join(outputpath,
                            f"{nwb_copy_file_name}_{session_name}_2D.pdf"), transparent=False, bbox_inches='tight')