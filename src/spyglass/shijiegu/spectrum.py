from spyglass.shijiegu.Analysis_SGU import LFPBandArtifact, ChangeofMind, ChannelNumber

from spyglass.common import AnalysisNwbfile, LFP
from spyglass.shijiegu.ripple_detection import load_LFP

from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.decodeHelpers import runSessionNames
import matplotlib.pyplot as plt
import os

import pynwb

import pandas as pd
import numpy as np

def load_LFP_one_channel_per_electrode(nwb_copy_file_name, session_name):
    lfp_data,lfp_timestamps = load_LFP(nwb_copy_file_name, None)
    key = LFP & {'nwb_file_name': nwb_copy_file_name}
    lfp_file_name = key.fetch1('analysis_file_name')
    analysisNWBFilePath = AnalysisNwbfile.get_abs_path(lfp_file_name)
    
    with pynwb.NWBHDF5IO(analysisNWBFilePath, 'r',load_namespaces=True) as io:
        nwb = io.read()
        electrodes=nwb.scratch['filtered data'].electrodes.to_dataframe()
    
    queries = (ChannelNumber() & {'nwb_file_name': nwb_copy_file_name}).fetch(as_dict = True)
    if len(queries) == 0:
        return None, None
    ripple_channel_ind = queries[0]["ripple_channel_ind"]
    subset_ind = np.isin(np.array(electrodes.index), ripple_channel_ind)
    CA1TetrodeInd = np.arange(len(electrodes))[subset_ind]
    
    return lfp_data[:,CA1TetrodeInd], lfp_timestamps

def spectrogram_moving_window(t, LFP, fNQ = 300, T = None, window_size = 0.5, step_size = 0.1):
    """
    Compute spectrogram using a moving window approach.
    
    Parameters:
    t: time axis
    LFP: LFP data (time x channels)
    fNQ: Nyquist frequency
    T: duration of the data (if None, it will be computed from t)
    window_size: size of the moving window in seconds
    step_size: step size for moving the window in seconds
    
    Returns:
    faxis: frequency axis
    Sxx_moving: spectrogram (frequency x time)
    """
    
    dt = t[1] - t[0]                # Define the sampling interval,
    if T is None:
        T = t[-1] - t[0]                # ... the duration of the data,
    N = len(LFP)                    # ... and the no. of data points
    
    window_samples = int(window_size / dt)  # Number of samples in each window
    step_samples = int(step_size / dt)      # Number of samples to step
    
    Sxx_moving = []
    #time_axis = np.arange(0, T - window_size + step_size, step_size) + 1/2 * step_size  # Time axis for the spectrogram
    
    time_axis = []
    for start in range(0, N - window_samples + 1, step_samples):
        end = start + window_samples
        faxis, Sxx = spectrogram(t[start:end], LFP[start:end], fNQ, T=window_size)
        Sxx_moving.append(Sxx)
        time_axis.append(np.mean(t[start:end]))  # Center time of the window
    
    Sxx_moving = np.array(Sxx_moving).T  # Transpose to get frequency x time
    
    return faxis, time_axis, Sxx_moving

def spectrogram(t, LFP, fNQ = 300, T = None):
    
    dt = t[1] - t[0]                # Define the sampling interval,
    if T is None:
        T = t[-1] - t[0]                # ... the duration of the data,
    N = len(LFP)                    # ... and the no. of data points
    
    x = np.hanning(N) * LFP         # Multiply data by a Hanning taper
    xf = np.fft.rfft(x - x.mean())  # Compute Fourier transform
    Sxx = 2*dt**2/T * (xf*np.conj(xf)) # Compute the spectrum
    Sxx = np.real(Sxx)                 # Ignore complex components
    
    
    df = 1 / T                      # Define frequency resolution,
    #fNQ = 1 / dt / 2                # ... and Nyquist frequency. 
    
    faxis = np.arange(0, fNQ + df, df) # Construct freq. axis
    
    Sxx = Sxx[:len(faxis)]

    return faxis, Sxx
    # plot(faxis, 10 * log10(Sxx))    # Plot spectrum vs freq.
    # xlim([0, 200])                  # Set freq. range, 
    # ylim([-80, 0])                  # ... and decibel range
    # xlabel('Frequency [Hz]')        # Label the axes
    # ylabel('Power [mV$^2$/Hz]');
    
def get_subset_lfp(lfp_timestamps, lfp_data, t0, t1):
    delta_t = 0.125 * 2
    
    time_ind = np.logical_and(lfp_timestamps >= t0, lfp_timestamps <= (t1 + delta_t))
    lfp_data_subset = lfp_data[time_ind, :]
    lfp_time_subset = lfp_timestamps[time_ind]
    
    return lfp_data_subset, lfp_time_subset

    # # bin dataset
    # bins = np.arange(lfp_time_subset[0], lfp_time_subset[-1]+delta_t, delta_t)
    # bin_index = np.digitize(lfp_time_subset, bins, right=False)
    # bin_names = np.unique(bin_index)
    
    # L = np.sum(bin_index == 1)

    # lfp_data_binned = [lfp_data_subset[bin_index == b, :] for b in bin_names if np.sum(bin_index == b) == L]
    # lfp_time_binned = [lfp_time_subset[bin_index == b] for b in bin_names if np.sum(bin_index == b) == L]
    

    # return lfp_data_binned, lfp_time_binned, bins

def get_spectrogram(lfp_data_binned, lfp_time_binned, window_size, step_size):
    
    Sxx_binned = []
    
    T =  lfp_time_binned[-1] - lfp_time_binned[0]
    #for t_ind in range(len(lfp_time_binned)):
    t_axis = lfp_time_binned #lfp_time_binned[t_ind]
    y = lfp_data_binned#[t_ind]

    Sxx = []
    for tetrode in range(y.shape[1]):
        faxis, taxis, Sxx_ = spectrogram_moving_window(
                t_axis, y[:,tetrode],
                T = T, window_size = window_size, step_size = step_size)
            #faxis, Sxx_ = spectrogram(t_axis, y[:,tetrode], T = T)
        Sxx.append(Sxx_)
    Sxx = np.mean(np.array(Sxx), 0).tolist()    
    #if len(Sxx_binned)==0 or len(Sxx) == len(Sxx_binned[-1]):
    #Sxx_binned.append(Sxx)
    Sxx = np.array(Sxx)
    
    #Sxx_binned = Sxx_binned.T 

    return Sxx, faxis, taxis #final result is frequency x over bins

def spectrum_by_animal(animal, list_of_days, mode = "change",
                       t_minus = 2, t_plus = 2, window_size = 0.5,  step_size = 0.1):
    """
    mode: "change" or "nearby"
    """
    
    all_data = {}
    for day in list_of_days:
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        print(nwb_copy_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        
        # load this day's LFP
        lfp_data, lfp_timestamps = load_LFP_one_channel_per_electrode(nwb_copy_file_name, None)
        if lfp_data is None:
            continue
        
        for ind in range(len(session_interval)):
            
            session_name = session_interval[ind]
            position_name = position_interval[ind]
            epoch_num = int(session_name[:2])
    
            key_pre = {"nwb_file_name": nwb_copy_file_name, "epoch":epoch_num,
                       "proportion":0.1}
            query = ChangeofMind & key_pre
            if len(query) == 0:
                print("No triggered decode found for ", key_pre)
                continue
            
            log = ChangeofMind().fetch1_dataframe(key_pre)
            
            # load this session's lfp
            
            trials = log[log.change_of_mind].index
            for trialID in trials:
                if mode == "change":
                    center_time = log.loc[trialID].initial_time 
                else:
                    center_time = log.loc[trialID].timestamp_O
                (t0,t1) = (center_time - t_minus, center_time + t_plus)
                    
                
                lfp_data_binned, lfp_time_binned = get_subset_lfp(
                    lfp_timestamps, lfp_data, t0, t1)
                
                #assert 1 == 0
                
                data = {"t0":t0, "t1":t1, "center_time":center_time,
                        "lfp_data_binned": lfp_data_binned,
                        "lfp_time_binned": lfp_time_binned}
                
                Sxx_binned, faxis, taxis = get_spectrogram(lfp_data_binned,
                                                    lfp_time_binned,
                                                    window_size = window_size, step_size = step_size)

                data["Sxx_binned"] = Sxx_binned
                data["faxis"] = faxis
                data["taxis"] = taxis
                all_data[(nwb_copy_file_name, session_name, trialID)] = data
                
    return all_data

def average_result(data):
    Sxx = []
    for key in data:
        Sxx.append(data[key]["Sxx_binned"])
        
    Sxx = np.stack(Sxx, axis = 2)
    return np.mean(Sxx, axis = -1)

def plot_spectrogram(taxis, faxis, Sxx, center_time, output_folder = None, name = None):
    fig, ax = plt.subplots(1,1,figsize = (3,5))
    
    df = faxis[1] - faxis[0]
    ax.imshow(np.log(Sxx),
               extent=[taxis[0] - center_time, taxis[-1] - center_time, faxis[0]-df/2, faxis[-1]+df/2],
                   vmin = 0, vmax = 10, origin='lower',aspect = 0.05)
        
    ax.set_yticks([8,30,50,100,150,250])
    ax.set_ylim([4-df/2,250+df/2])
    ax.set_xlim([-1.8,1.8])
    ax.set_xticks([-1.5, 0, 1.5])
    ax.set_xlabel("time (sec) \n since change of mind")
    
    ax.set_ylabel("Fequency (Hz)")
    ax.set_title("Power Spec")
    if name is not None:
        ax.set_title(name)
    plt.tight_layout()
    # add colorbar
    plt.colorbar(ax.images[0], ax=ax, label='Log Power')
    if output_folder is not None:
        plt.savefig(os.path.join(output_folder, f"spectrogram_{name}.pdf"), dpi=300)
                
    