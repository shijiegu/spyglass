import numpy as np
import xarray as xr
import pandas as pd
import os
from scipy import signal

from spyglass.shijiegu.Analysis_SGU import DecodeIngredients, DecodeIngredientsLikelihood

""" def convolve(arr, window, mode = "valid"):
    # wrapper for handling nan
    not_nan_ind = np.bitwise_not(np.isnan(arr))
    
    # use complex number to deal with nan
    # https://stackoverflow.com/questions/38318362/2d-convolution-in-python-with-missing-data
    arr_complex = np.zeros_like(arr, dtype=np.complex64)
    arr_complex[np.isnan(arr)] = np.array((1j))
    arr_complex[not_nan_ind] =  arr[not_nan_ind]
    convolution = signal.convolve(arr_complex, window, mode = mode, method = "direct")
    convolution[np.imag(convolution)>=(len(window))] = np.nan
    print("here")
    convolution = convolution.astype(np.float64)
    return convolution """

def convolve(arr, window, mode = "valid"):
    arr[np.isnan(arr)] = 0
    convolution = signal.convolve(arr, window, mode = mode, method = "direct")
    return convolution

def decimate(arr, n, decimate_flag = True):
    # every n data points average into one new data point
    window = (1.0 / n) * np.ones(n,)
    if decimate_flag:
        res = convolve(arr, window, mode='valid')[::n]
    else:
        res = convolve(arr, window, mode='valid')
    return res


def decimate_sum(arr, n, decimate_flag = True):
    window = np.ones(n,)
    if decimate_flag:
        res = convolve(arr, window, mode='valid')[::n]
    else:
        res = convolve(arr, window, mode='valid')
    return res

def decimate_ave_pos(pos1d, N, decimate_flag = True):
    # average in the bins to decimate
    # if decimate = False, only convolution is applied

    # decimate data
    data = {}
    for col in pos1d.columns.tolist():
        pos1d_lumped = decimate(np.array(pos1d[col]), N, decimate_flag = decimate_flag) #every 5 bins of 2ms bins -> 10 ms bins
        data[col] = pos1d_lumped
    data_df = pd.DataFrame.from_dict(data)

    return data_df

def decimate_sum_marks(marks, N, decimate_flag = True):
    # sum in the bins to decimate
    # if decimate = False, only convolution is applied
    
    # fill nan to 0
    #marks = marks.fillna(0)
    
    # convert to numpy
    marks_np = np.array(marks.to_dataarray()).squeeze()
    
    # decimate neural data
    time_len, channel_num, electrode_num = marks_np.shape
    if decimate_flag:
        final_time_len = np.floor(time_len/N)
    else:
        final_time_len = time_len - N + 1
    output_arr = np.zeros((int(final_time_len), int(channel_num), int(electrode_num)))+np.nan
    for c_ind in range(channel_num):
        for e_ind in range(electrode_num):
            output_arr[:,c_ind,e_ind] = decimate_sum(marks_np[:,c_ind,e_ind], N, decimate_flag = decimate_flag)

    # decimate time
    marks_time_lumped = decimate(np.array(marks.time), N, decimate_flag = decimate_flag)

    # make array
    marks_out = xr.DataArray(
        data=output_arr.astype("float32"),
        dims=["time", "marks", "electrodes"],
        coords=dict(
            time=(["time"], marks_time_lumped.astype("float64")),
            electrodes=(["electrodes"], np.array(marks.coords["electrodes"])),
            marks=(["marks"], np.array(marks.coords["marks"])))
        ).to_dataset(name = "unitmarks")

    return marks_out

def make_decimated_names(marks_path, window_size, overlap_size):
    folder, filename = os.path.split(marks_path)
    filename_split = filename.split(".")
    filename_split[-1] = "." + filename_split[-1]
    filename_split[-2] += "_decimated"+str(int(window_size))+"ms"+str(int(overlap_size))+"ms"
    filenameNew = ''.join(filename_split)
    return os.path.join(folder, filenameNew)

def decodePrepLikelihoodMasterSession(nwb_copy_file_name,
                                      session_name,window_size = 0.02,overlap_size = 0.01):
    # in the unit of seconds
    marks_path = ( DecodeIngredients & {'nwb_file_name':nwb_copy_file_name,
                                    'interval_list_name':session_name} ).fetch1("marks")
    marks = xr.open_dataset(marks_path)

    pos1d_path = ( DecodeIngredients & {'nwb_file_name':nwb_copy_file_name,
                        'interval_list_name':session_name} ).fetch1("position_1d")
    pos1d = pd.read_csv(pos1d_path)

    pos2d_path = ( DecodeIngredients & {'nwb_file_name':nwb_copy_file_name,
                        'interval_list_name':session_name} ).fetch1("position_2d")
    pos2d = pd.read_csv(pos2d_path)
    
    # low pass (convolution + decimating)
    N = int(overlap_size * 1000 / 2) # 5x2 = 10ms
    pos1d_decimate = decimate_ave_pos(pos1d, N)
    pos2d_decimate = decimate_ave_pos(pos2d, N)
    marks_decimate = decimate_sum_marks(marks, N)

    # overlap (convolution)
    M = int( window_size / overlap_size ) # 2 bins x 10ms/bin = 20 ms
    pos1d_overlap = decimate_ave_pos(pos1d_decimate, M, decimate_flag = False)
    pos2d_overlap = decimate_ave_pos(pos2d_decimate, M, decimate_flag = False)
    marks_overlap = decimate_sum_marks(marks_decimate, M, decimate_flag = False)

    marks_path_de = make_decimated_names(marks_path, window_size * 1000, overlap_size * 1000)
    pos1d_path_de = make_decimated_names(pos1d_path, window_size * 1000, overlap_size * 1000)
    pos2d_path_de = make_decimated_names(pos2d_path, window_size * 1000, overlap_size * 1000)
    
    """save result"""
    marks_overlap.to_netcdf(marks_path_de)
    pos1d_overlap.to_csv(pos1d_path_de)
    pos2d_overlap.to_csv(pos2d_path_de)


    key={'nwb_file_name':nwb_copy_file_name,
        'window_size':window_size,
        'overlap_size':overlap_size,
        'interval_list_name':session_name,
        'marks':marks_path_de,
        'position_1d':pos1d_path_de,
        'position_2d':pos2d_path_de}
    DecodeIngredientsLikelihood().insert1(key,replace=True)