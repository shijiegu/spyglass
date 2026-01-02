import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

def interpolate_to_new_time(df, new_time, upsampling_interpolation_method='linear'):
    old_time = df.index
    new_index = pd.Index(np.unique(np.concatenate(
        (old_time, new_time))), name='time')
    tmp = df.reindex(index=
                     new_index
                     ).interpolate(
                         method=upsampling_interpolation_method).reindex(index=new_time)
    tmp.index.name = df.index.name
    return tmp

def place_field(pos1d, pos2d, spike_time, BINWIDTH = 2, sigma = 2):
    # default is 2 cm bins, smoothed by 2 bins = 4 cm

    # check input
    assert len(pos1d) == len(pos2d)
    pos2d = pos2d.set_index('time')
    print("spike num:", len(spike_time))

    delta_t = np.mean(np.diff(pos2d.index)) #for later, translating the unit of time number bins to time in seconds

    # select track time only, remove Data before 1st trial and after last trial
    trial_1_t = np.array(pos1d.time)[0]
    trial_last_t = np.array(pos1d.time)[-1]
    spike_time = spike_time[np.logical_and(spike_time>=trial_1_t,
                                           spike_time<=trial_last_t)]
    total_spike_count = len(spike_time)

    # mobility only
    mobility_index = np.argwhere(pos2d.head_speed > 4).ravel() # >4cm/s
    pos2d_mobility = pos2d.iloc[mobility_index]

    pos2d_spike_time_all = interpolate_to_new_time(pos2d,spike_time,
                                                    upsampling_interpolation_method = 'nearest')
    mobility_index = np.argwhere(pos2d_spike_time_all.head_speed > 4).ravel() # >4cm/s
    pos2d_spike_time = pos2d_spike_time_all.iloc[mobility_index]

    #Define bins for all position
    xmin = np.min(pos2d_mobility.head_position_x)-10
    xmax = np.max(pos2d_mobility.head_position_x)+10
    ymin = np.min(pos2d_mobility.head_position_y)-10
    ymax = np.max(pos2d_mobility.head_position_y)+10
    xbins = np.arange(xmin,xmax,BINWIDTH)
    ybins = np.arange(ymin,ymax,BINWIDTH)
     

    # place field, aka occupancy normalized firing rate, aka P(spike | location)
    occupancy, xe, ye = np.histogram2d(pos2d_mobility.head_position_y,
                                       pos2d_mobility.head_position_x, bins = [ybins,xbins])
    smoothed_occupancy = gaussian_filter(occupancy, sigma = sigma) #1 bins = 1cm
    occupancy = occupancy * delta_t # number of entry * delta_t second / entry
    smoothed_occupancy = smoothed_occupancy * delta_t
    old_nan_index = smoothed_occupancy < 0.01
    occupancy[occupancy < 0.001] = np.nan # for numerical stability

    spike, xe, ye = np.histogram2d(pos2d_spike_time.head_position_y,
                                   pos2d_spike_time.head_position_x, bins = [ybins,xbins])

    pf = spike/occupancy

    pf = np.nan_to_num(pf, nan = 0, posinf = 0)
    smoothed_placefield = gaussian_filter(pf, sigma = sigma) #1 bin = 1cm
    smoothed_placefield[old_nan_index] = np.nan
    peak_firing_rate = np.nanmax(smoothed_placefield)

    spike_count = len(pos2d_spike_time)

    return smoothed_placefield, peak_firing_rate, xbins, ybins, spike_count, total_spike_count