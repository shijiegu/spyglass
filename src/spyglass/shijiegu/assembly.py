import numpy as np
import pandas as pd

from spyglass.shijiegu.Analysis_SGU import RippleTimesWithDecode
from spyglass.shijiegu.fragmented_general import find_SWR_time
from spyglass.shijiegu.singleUnit import session_unit, find_spikes
from scipy.stats import zscore
from sklearn.decomposition import PCA, FastICA
from spyglass.shijiegu.Analysis_SGU import TrialChoice
from sklearn.cluster import KMeans

# a function to return bins by time or by ripple
def get_bins_ripple(nwb_copy_file_name, session_name,
                    bin_width = 0.002, fragmented = True):
    """return time bins during ripple
    bin_width:
        in seconds. so 0.002 is 2 ms!
        if return the whole ripple interval as an event, set bin_width to 0.
    return a list of time bins, each element has a list of time bins.
    """
    try:
        ripple_times = pd.DataFrame((RippleTimesWithDecode & 
                                     {'nwb_file_name': nwb_copy_file_name, 'interval_list_name': session_name}).fetch1('ripple_times'))
    except:
        ripple_times = pd.read_pickle((RippleTimesWithDecode & 
                                       {'nwb_file_name': nwb_copy_file_name,
                                        'interval_list_name': session_name,
                                       	'decode_threshold_method': 'MUA_0SD'}).fetch1('ripple_times'))
    else:
        ripple_times = pd.read_pickle((RippleTimesWithDecode & 
                                       {'nwb_file_name': nwb_copy_file_name,
                                        'interval_list_name': session_name,
                                       	'decode_threshold_method': 'MUA_M05SD'}).fetch1('ripple_times'))
        
    
    intervals, index = find_SWR_time(ripple_times, cont = 1-fragmented, return_index = True)
    
    if bin_width == 0:
        return intervals, index
    
    list_of_bins = []
    ripple_index = []
    for intvl_ind in range(len(intervals)):
        intvl = intervals[intvl_ind]
        list_of_bins.append(np.arange(intvl[0],intvl[1] + bin_width,bin_width))
        ripple_index.append(index[intvl_ind])
    return list_of_bins, ripple_index

def get_baseline(nwb_copy_file_name, session_name, curation_id = 1, delta_t = 0.005):
    # as the name suggests, find neuronal firing baseline.
    # spikes are binned in 5ms windows
    # from the beginning of the session to the end of the session
    # windows in which no spikes fire across all neurons are discarded
    # returns a dictionary
    #  key - neuron (e,u)
    #  entry - mean and sd of each neuron firing rate in a 10 ms window throughout the whole session.
    
    (nwb_units_all, 
     _, cell_list) = session_unit(
         nwb_copy_file_name, session_name, curation_id = curation_id,
         return_cell_list = True, exclude_interneuron = False)
     
    # get sort interval
    (e, u) = cell_list[0]
    sort_interval = np.ravel(nwb_units_all[e].loc[u].sort_interval)
    intvl = get_sort_start_end(nwb_copy_file_name, session_name, sort_interval, delta_t)
     
    binned_spike_count, _ = find_spikes(
            nwb_units_all, cell_list, intvl, count = False)
    
    # get rid of time bins in which no spikes fire across all neurons
    some_firing_time = np.sum(binned_spike_count, axis = 1) > 0
    binned_spike_count = binned_spike_count[some_firing_time,:]
    
    # calculate baseline
    baseline = {}
    for c_ind in range(len(cell_list)):
        (e, u) = cell_list[c_ind]
        baseline[(e,u)] = (np.mean(binned_spike_count[:,c_ind]), np.std(binned_spike_count[:,c_ind]))
    
    return baseline

# a function to get neural data from the bin sizes
def get_binned_spikes(nwb_copy_file_name, session_name, list_of_bins,
                      curation_id = 1, z_score = True):
    """
    returns a list of neural data matrix (num_units) x (num_bins)
    if zscore: 
        subtracting its baseline rate, computed as a session average
        average. After baseline subtraction, we divided firing rate by its standard deviation, which was regularized by adding a small number
        (0.6 Hz) --- as in Chettih 2024.
    
    return binned_spike_counts: (num_units) x (num_bins)
    """
    DELTA_T = np.mean(np.diff(list_of_bins[0]))
    (nwb_units_all, 
     _, cell_list) = session_unit(
         nwb_copy_file_name, session_name, curation_id = curation_id,
         return_cell_list = True, exclude_interneuron = True)
    
    binned_spike_counts = []
    for intvl in list_of_bins:
        binned_spike_count, time_bin = find_spikes(
            nwb_units_all, cell_list, intvl, count = False)
        binned_spike_counts.append(binned_spike_count)

    binned_spike_counts_zscore = []
    if z_score:
        baseline = get_baseline(nwb_copy_file_name, session_name,
                                curation_id = 1, delta_t = DELTA_T)
        for binned_spike_count in binned_spike_counts:
            for cell_ind in range(len(cell_list)):
                (e,u) = cell_list[cell_ind]
                mean, sd = baseline[(e,u)]
                binned_spike_count[:,cell_ind] = (binned_spike_count[:,cell_ind] - mean) / sd
            binned_spike_counts_zscore.append(binned_spike_count)

        return binned_spike_counts_zscore, nwb_units_all, cell_list
    
    return binned_spike_counts, nwb_units_all, cell_list

def get_sort_start_end(nwb_copy_file_name, session_name, sort_intervals, delta_t = 0.005):
    # intersect with statescript so that only on track time is used.
    StateScript = pd.DataFrame(
        (TrialChoice & {'nwb_file_name':nwb_copy_file_name,
                        'epoch_name':session_name}).fetch1('choice_reward'))

    trial_1_t = StateScript.loc[1].timestamp_O
    trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O
    
    if len(sort_intervals.shape) == 1:
        (t0,t1) = sort_intervals
    else:
        num_of_intervals = sort_intervals.shape[0]
        (t0,t1) = (sort_intervals[0,0], sort_intervals[num_of_intervals-1,1])
    
    t0 = np.max([t0, trial_1_t])
    t1 = np.min([t1, trial_last_t])
    
    return np.arange(t0, t1 + delta_t, delta_t)
    
def get_sort_duration(sort_intervals):
    delta_t = 0
    if len(sort_intervals.shape) == 1:
        return sort_intervals[1] - sort_intervals[0]
    
    for ind in range(sort_intervals.shape[0]):
        delta_t += sort_intervals[ind][1] - sort_intervals[ind][0]
    return delta_t

def compute_assembly_vectors(binned_spike_counts):
    """Compute assembly vectors from spike data using PCA and ICA.
    Args:
        binned_spike_counts: (num_units) x (num_bins)
    Returns:
        ic_vectors (array): Assembly vectors found by ICA, shape (n_assemblies, n_units)
        pc_vectors (array): Assembly vectors found by PCA, shape (n_assemblies, n_units)
        eigenvalues (array): PCA eigenvalues
        lambda_min (float): Theoretical minimum eigenvalue from Marcenko-Pastur
        lambda_max (float): Theoretical maximum eigenvalue from Marcenko-Pastur
    """
    #Z = zscore(binned_spike_counts, axis=1) # (num_units) x (num_bins)
    Z = binned_spike_counts
    Z[np.isnan(Z)] = 0
    
        # compute eigenvalue bounds of Marcenko-Pastur
    n_rows, n_cols = Z.shape
    q = n_cols/n_rows
    lambda_min = (1 - np.sqrt(1/q))**2
    lambda_max = (1 + np.sqrt(1/q))**2

    # compute PCA and number of significant assemblies
    pca = PCA()
    Z_proj_pca = pca.fit_transform(Z.T) # (num_bins) x (num_pcs)
    pc_vectors = pca.components_
    eigenvalues = pca.explained_variance_

    signif_pc_indices = np.where(eigenvalues > lambda_max)[0]
    Z_signif_proj_pca = Z_proj_pca[:, signif_pc_indices] # retain only significant PCs

    # compute ICA in PC-reduced space
    ica = FastICA(n_components=len(signif_pc_indices), random_state=0)
    ica.fit(Z_signif_proj_pca)

    V_T = pca.components_[signif_pc_indices, :]
    ic_vectors = np.dot(ica.components_, V_T) # project ICs back to original unit space

    # sort IC vectors by decreasing variance
    ic_assembly_activities = np.square(ic_vectors @ Z)
    row_variances = np.var(ic_assembly_activities, axis=1)
    order_ind = np.argsort(row_variances)[::-1]
    ic_vectors = ic_vectors[order_ind] # sort in descending order
    ic_assembly_activities = ic_assembly_activities[order_ind,:]
    
    # TO DO: normalize weights by norm

    # flip signs if needed to make largest magnitude weight positive
    for k in range(len(ic_vectors)):
        ic_vectors[k] = ic_vectors[k] * np.sign(ic_vectors[k][np.argmax(np.abs(ic_vectors[k]))])
        pc_vectors[k] = pc_vectors[k] * np.sign(pc_vectors[k][np.argmax(np.abs(pc_vectors[k]))])

    return ic_vectors, pc_vectors, eigenvalues, lambda_min, lambda_max, ic_assembly_activities
    
    
def order_by_weight(ic_vectors):
    """
    ic_vectors: # of assembly x # of neurons 

    Returns:
    neurons_ind_all_assembly: ordering index
    neurons_labels_all_assembly: assembly that each neuron belongs to
    """
    # produce neuronal ordering
    assembly_membership = np.argmax(ic_vectors,axis = 0)
    
    # within each assembly, sort neurons by descreasing weight
    neurons_ind_all_assembly = []
    neurons_labels_all_assembly = []
    for ic_vector_ind in range(len(ic_vectors)):
        neurons_ind = np.argwhere(assembly_membership == ic_vector_ind).ravel()
        neurons_ind_all_assembly.append( neurons_ind[np.argsort(ic_vectors[ic_vector_ind,neurons_ind])] )
        neurons_labels_all_assembly.append(np.ones(len(neurons_ind)) * ic_vector_ind)
    neurons_ind_all_assembly = np.concatenate(neurons_ind_all_assembly)    
    neurons_labels_all_assembly = np.concatenate(neurons_labels_all_assembly)

    return neurons_ind_all_assembly, neurons_labels_all_assembly

def order_by_weight_SeqNMF(W):
    # for each neuron, find its assembly, then within each assembly, sort of latency.   
    
    # #max_factor, L_sort, max_sort, hybrid = ClusterByFactor(W[:,0:4,:], 1)
    #indSort = hybrid[:,2].astype('int') - 1;
    
    W_collapsed = np.nanmax(W, axis = 2)
    identity = np.argmax(W_collapsed, axis = 1)
    N,K,L = np.shape(W)

    ind_assembly_all = []
    neurons_labels_all= []

    for assembly_id in range(K):
        neuron_ind_assemly = np.argwhere(identity == assembly_id).ravel() #neuron ind that belongs to this assembly
        if len(neuron_ind_assemly) == 0:
            continue
        W_assembly = W[neuron_ind_assemly,assembly_id,:].reshape((len(neuron_ind_assemly),-1))
        time_ind_assembly = np.argmax(W_assembly,axis= 1)
        ind_assembly = np.argsort(time_ind_assembly)
        ind_assembly_all.append(neuron_ind_assemly[ind_assembly])
        neurons_labels_all.append(np.ones(len(ind_assembly)) * assembly_id)

    neurons_ind_seqnmf = np.concatenate(ind_assembly_all)
    neurons_labels_all = np.concatenate(neurons_labels_all)
    
    return neurons_ind_seqnmf, neurons_labels_all

def getSeqNMF_weight(W):
    N, num_components, _ = np.shape(W)
    
    neurons_ind_seq, neurons_labels_seq = order_by_weight_SeqNMF(W)
    weight = []
    for neuron in range(len(neurons_ind_seq)):
        assembly_ind = neurons_labels_seq[np.argwhere(neurons_ind_seq == neuron).ravel()]
        weight.append(np.max(W[neuron,int(assembly_ind),:]))
        
    weight = np.array(weight)
    # normalize within each sequence/assembly
    weight_normalized = np.zeros((num_components, N))
    for assembly in range(num_components):
        ind = neurons_ind_seq[neurons_labels_seq == assembly]
        weight_normalized[assembly, ind] = weight[ind]/np.sum(weight[ind])
    return weight_normalized


def ClusterByFactorSeqNMF(W, nclust):
    N, K, L = W.shape
    max_factor = [None] * K
    clust = np.zeros((N,K))
    L_sort = np.zeros((K,N))
    max_sort = np.zeros((K,N))
    for ii in range(K):
        data = W[:, ii, :].reshape(N, L)
        max_factor[ii] = (data == data.max(axis=1, keepdims=True)).astype(int)
        max_factor[ii][max_factor[ii].sum(axis=1) > 1] = 0
        clust[:, ii] = data.max(axis=1)
        tmp, = np.where(max_factor[ii].any(axis=1))
        L_sort[ii, :len(tmp)] = tmp
        max_sort[ii, :] = np.flip(np.argsort(data.max(axis=1)))

    idx = KMeans(n_clusters=nclust).fit_predict(clust)
    pos = np.zeros((N, 4))
    pos[:, 2] = np.arange(1, N + 1)
    for ii in range(N):
        nni = W[ii, :, :].reshape(K, L)
        try:
            pos[ii, 1], pos[ii, 0] = np.unravel_index(nni.argmax(), nni.shape)
        except:
            pos[ii, 0:2] = [L, K + 1]

    hybrid = pos
    hybrid = hybrid[np.argsort(hybrid[:, 1]), :]
    temp = []
    for ii in range(1, K + 2):
        hy = hybrid[hybrid[:, 1] == ii, :]
        temp.append(hy[np.argsort(hy[:, 0]), :])
    hybrid = np.vstack(temp)
    hybrid[:, 3] = idx[hybrid[:, 2].astype(int) - 1]
    hybrid = hybrid[np.argsort(hybrid[:, 3]), :]

    return max_factor, L_sort, max_sort, hybrid
