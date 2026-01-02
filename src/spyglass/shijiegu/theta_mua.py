import numpy as np
import xarray as xr
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import filtfilt, lfilter
from spyglass.shijiegu.Analysis_SGU import MUATheta, MUA
from spyglass.shijiegu.ripple_add_replay import select_subset_helper
from spyglass.shijiegu.theta_singleUnit import smoothen_mua, get_theta_from_mua


def load_theta_from_calculations(nwb_copy_file_name, session_name, data_type):
    """
    load theta from sorted pyramidal or corpus callosum or mua
    data_type: "sorted_pyramidal", "corpus_callosum", or "mua"
    """
    
    key = {"nwb_file_name": nwb_copy_file_name,
                    "epoch": str(session_name[:2]),
                    "data_type":"corpus_callosum"}
    df = pd.read_csv((MUATheta() & key).fetch1("theta_xr"))
    
    try:
        amplitude = df.amplitude
    except:
        amplitude = np.cos(df.phase)
    
    theta_xr = xr.Dataset(
        data_vars={
            "0": (("time"), amplitude),
            "1": (("time"), amplitude),
            "amplitude": (("time"), amplitude),
            "phase": (("time"), df.phase),
        },
        coords={
            "time": np.array(df.time),
        },
        attrs={"description": "theta"}
    )
    return theta_xr

def calculate_session_theta_from_mua(nwb_copy_file_name, session_name):
    """load MUA"""
    #decode_threshold_method = decode_options["decode_threshold_method"]
    q = MUA & {'nwb_file_name': nwb_copy_file_name,
               'interval_list_name':session_name}
    mua_path= q.fetch1('mua_trace')
    mua_xr = xr.open_dataset(mua_path)
    #mua_mean, mua_sd = q.fetch1("mean"), q.fetch1("sd")
    
    mua_smoothened = smoothen_mua(mua_xr) # 40 ms window smoothened
    theta = get_theta_from_mua(mua_smoothened)
    
    return theta


    

