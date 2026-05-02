import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.load import load_decode

def hpd(pdf_vals, prob_mass = 0.5):
    # Find the HPD interval (e.g., 50%)

    # Sort by density (descending)
    sorted_indices = np.argsort(pdf_vals)[::-1]
    # Accumulate density mass
    cumulative_mass = 0
    hpd_indices = []
    for idx in sorted_indices:
        cumulative_mass += pdf_vals[idx]
        hpd_indices.append(idx)
        if cumulative_mass >= prob_mass:
            break

    return hpd_indices

def return_low_hpd_time(decode, return_boolean = False, debug = False, prob_mass = 0.5):
    
    subset_posterior = decode.causal_posterior.sum(dim='state')

    input_array = np.array(subset_posterior) #in array of shape (number of time points x number of position)
    x_axis = np.array(subset_posterior.position)
    t_axis = np.arange(input_array.shape[0])
    eligible_indices = []
    boolean = np.zeros(len(t_axis))
    hpd_all = []
    all_indices = []
    for t in t_axis:
        x_indices = hpd(input_array[t], prob_mass = prob_mass)
        all_indices.append(x_indices)
        hpd_all.append(x_axis[np.max(x_indices)] - x_axis[np.min(x_indices)])
        if x_axis[np.max(x_indices)] - x_axis[np.min(x_indices)] <= 50:
            eligible_indices.append(t)
            boolean[t] = 1
    if debug:
        return all_indices
    if return_boolean:
        return boolean
    
    subset_posterior_subset = subset_posterior.isel(time=eligible_indices)

    return subset_posterior_subset