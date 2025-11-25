import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from scipy.special import comb
from dynamax.hidden_markov_model import CategoricalHMM

from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.Analysis_SGU import ChangeofMindTheta, ChangeofMindRemoteTheta

cycle4_1 = [2,4,1,3]
cycle4_2 = [3,4,2,1]
cycle4_3 = [1,2,3,4]
cycle4_4 = [3,1,4,2]
cycle4_5 = [1,2,4,3]
cycle4_6 = [4,3,2,1]

cycle3_1 = [1,2,3]
cycle3_2 = [1,3,2]
cycle3_3 = [1,2,4]
cycle3_4 = [1,4,2]
cycle3_5 = [2,3,4]
cycle3_6 = [2,4,3]
cycle3_7 = [1,3,4]
cycle3_8 = [1,4,3]

cycle_random = [1]

cycles = [cycle4_1, cycle4_2, cycle4_3, cycle4_4, cycle4_5, cycle4_6,
          cycle3_1, cycle3_2, cycle3_3, cycle3_4, cycle3_5, cycle3_6, cycle3_7, cycle3_8, cycle_random]

cycle_lengths = [len(cycle) for cycle in cycles]

# cycle_type: for each state in state_space_concat, which cycle it belongs to
# [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3....]
cycle_type = []
ind = 0
for cycle in cycles:
    cycle_type += [ind for _ in range(len(cycle))]
    ind += 1

# cycle_type_by_cycle: for each cycle, which type it is
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9....]
cycle_type_by_cycle = [_ for _ in range(len(cycles))]
    
# state_space: concatenated state space of all cycles
# [2,4,1,3,3,4,2,1,1,2,3,4,3,1,4,2....]
state_space = np.concatenate(cycles)

# transition matrix
def get_next_arm(start_cycle, current_state):
    L = len(start_cycle)
    next_ind = (np.argwhere(start_cycle == current_state).ravel()[0] + 1) % L
    next_arm_state = start_cycle[next_ind]
    return next_arm_state

def construct_transition_matrix(seed,
                                p_stay_in_seq = 0.5, # proportional to 4a
                                p_jump_to_other_seq = 0.2, # proportional to 4B * 5
                                p_jump_to_other_seq_type = 0.1, # proportional to 3b * 8
                                p_jump_to_random = 0.2): # proportional to 4c
    """
    The transition matrix looks like:
               Seq1    Seq2    Seq3    Seq4     Seq5    Seq6      Seq7     Seq8    Random
             -------  -------  ------  ------   ------   ------   ------  ------  -------
             2 4 1 3  3 4 2 1 1 2 3 4  3 1 4 2  1 2 4 3  4 3 2 1  1 3 2   1 2 4    1
    Seq1|2     a        B           B      B        B    B                    b    c             
    Seq1|4       a          B B          B      B              B  b       b        c
    Seq1|1
    Seq1|3
    Seq2|3
    Seq2|4
    Seq2|2
    Seq2|1
    Seq3|1
    Seq3|2
    Seq3|3
    Seq3|4
    Seq4|3
    Seq4|1
    Seq4|4
    Seq4|2
    Seq5|1 
    ...
    random|1
    
    """
    A = [] # A is the transition matrix
    
    a = p_stay_in_seq / 4
    b = p_jump_to_other_seq / 20
    B = p_jump_to_other_seq_type / 24
    c = p_jump_to_random / 4
    
    for s_ind in range(len(state_space)): # construct row by row
        s = state_space[s_ind]
        start_cycle_type = cycle_type[s_ind]
        start_cycle_length = cycle_lengths[start_cycle_type]
        
        if start_cycle_type == 14:#14 is random state
            # random state, can transition to any state with equal probability
            A_s = np.ones(len(state_space)) * B
            A_s[-1] = c
            A_s = list(np.array(A_s)/np.sum(A_s))
            A.append(A_s)
            continue
        
        # the next arm of the current cycle
        start_cycle = cycles[start_cycle_type]
        next_arm_state = get_next_arm(start_cycle, s)
        
        A_s = [] #for all other states
        
        for cycle_ind in range(len(cycles)-1): # exclude the last random state
            keys = jr.split(jr.PRNGKey(seed*1000 + s_ind*100 + cycle_ind), 3)
            
            cycle = np.array(cycles[cycle_ind])
            
            L = len(cycle)
            target_cycle_type = cycle_type_by_cycle[cycle_ind]
            target_cycle_length = cycle_lengths[target_cycle_type]
            
            A_s_cycle = np.zeros(L)
            s_ind_cycle = np.argwhere(cycle == next_arm_state).ravel()
            #assert 0 == 1

            if start_cycle_type == target_cycle_type:
                # transition within the same cycle type
                A_s_cycle[s_ind_cycle] = a + jr.normal(keys[0])*a/10
            elif start_cycle_length == target_cycle_length:
                A_s_cycle[s_ind_cycle] = B + jr.normal(keys[0])*B/10
            else:
                A_s_cycle[s_ind_cycle] = b + jr.normal(keys[0])*B/10
            A_s += list(A_s_cycle)

        #now we deal with the random state
        A_s += [c + jr.normal(keys[1])*a/10]
            
        #if start_cycle_type == 14: # #14 is random state, it is treated slightly differently
        A_s = list(np.array(A_s)/np.sum(A_s))
            
        A.append(A_s)

    A = np.array(A)
    return A


def return_A_seq_level(A):
    A_seq = []
    
    start_row = 0
    for block_ind_row in range(len(cycle_lengths)): # row
        end_row = start_row + cycle_lengths[block_ind_row]
        A_sub_row = np.sum(A[start_row:end_row,:], axis = 0)
    
        start_col = 0
        A_sub_col = []
        for block_ind_col in range(len(cycle_lengths)): # row
             end_col = start_col + cycle_lengths[block_ind_col]
             A_sub_col.append(np.sum(A_sub_row[start_col:end_col]))
             start_col = end_col
        
        A_seq.append(A_sub_col/np.sum(A_sub_col))
    
        start_row =  end_row
    A_seq = np.array(A_seq)
    return A_seq

def construct_emission_matrix(noise_level = 0.6):
    E = np.array([state_space == i for i in [1,2,3,4]]).astype("float")
    E += np.random.rand(np.shape(E)[0],np.shape(E)[1]) * noise_level

    # the last state (random state) will be different.
    # It is equally likely to emit any observation.
    E[:,-1] = np.ones(4)

    E = E/np.sum(E, axis = 0).reshape((1,-1))
    # up to this point the emission matrix is shape (4, num_states)
    # we need to transpose it to be (num_states, 4)
    E = E.T
    return E

def initialize_state_space(A, E):
    """_summary_

    Args:
        A (numpy array): of size (num_states, num_states), transition matrix, A.sum(axis = 1) == 1
        E (numpy array): of size (num_states, num_observation_states), emission matrix, E.sum(axis = 1) == 1
        emission_all (numpy array): (num_batch x num_time), observed emissions
    """
    
    # convert to jax numpy
    A_jnp = jnp.array(A)
    E_jnp = jnp.array(E)

    num_states = A.shape[0]  
    num_emissions = 1 # some extra functionality of dynamax that we do not need
    num_classes = 4
    
    # initially all sequences are equally likely
    initial_np = np.array([1/cycle_lengths[cycle_type[s]] for s in range(num_states)])
    initial_jnp = jnp.array(initial_np/np.sum(initial_np))

    # Construct the HMM
    hmm = CategoricalHMM(num_states, num_emissions, num_classes)

    # Initialize the parameters struct with known values
    init_params, props = hmm.initialize(initial_probs=initial_jnp,
                            transition_matrix = A_jnp,
                            emission_probs=E_jnp.reshape(num_states, num_emissions, num_classes))
    return hmm, init_params, props

def em_state_space(hmm, init_params, props, emission_all):
    
    # ensure no nan in emissions
    for emission in emission_all:
        if np.isnan(emission).any():
            print("Error, emission contains NaN values. Aborting initialization.")
            return None
    emissions = jnp.array(emission_all)
    num_emissions = 1 # some extra functionality of dynamax that we do not need
    
    num_batches, timesteps = emissions.shape[0], emissions.shape[1]
    emissions_ = emissions.reshape(num_batches, timesteps, num_emissions)
    
    # Fit the HMM using EM
    em_params, em_losses = hmm.fit_em(init_params, 
                                props, 
                                emissions_, 
                                num_iters=50)
    return em_params, em_losses
    

def get_day_observations(animal, day):

    nwb_file_name = animal.lower() + day + '.nwb'
    nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)

    print(nwb_copy_file_name)
    session_interval, position_interval = runSessionNames(nwb_copy_file_name)
                
    q = {"nwb_file_name": nwb_copy_file_name,
        "proportion": 0.1,
        "delta_t_minus":5, "delta_t_plus":5,
        "max_flag":1}

    emission_all = []
    df_all =[]
    for session_name in session_interval:
        q["epoch"] = session_name[:2]
        df = ChangeofMindRemoteTheta().fetch1_dataframe(q)   
        OuterWellIndex = df.OuterWellIndex[:80]
        if np.sum(np.isnan(OuterWellIndex)) > 0:
            print(f"Warning: NaN values found in emissions for session {session_name}. Skipping this session.")
            continue
        emission_all.append(np.array(OuterWellIndex).astype("int") - 1)
        df_all.append(df)
    return emission_all, df_all

### the following function returns the posterior probabilities of being in cycle4, cycle3, random
def return_cycle_type_posterior(hmm, em_params, outers, causal = False):
    posterior = hmm.smoother(em_params, outers)
    
    if causal:
        posterior_prob = posterior.filtered_probs
    else:
        posterior_prob = posterior.smoothed_probs
    
    cycle4_posterior = np.sum(posterior_prob[:,np.array(cycle_type) <= 5], axis = 1)
    cycle3_posterior = np.sum(posterior_prob[:,np.logical_and(np.array(cycle_type) > 5, np.array(cycle_type) < 14)], axis = 1)
    random_posterior = np.sum(posterior_prob[:,np.array(cycle_type) == 14], axis = 1)
    return cycle3_posterior, cycle4_posterior, random_posterior

def return_target_seq_posterior(hmm, em_params, outers, target_seq, causal = False):
    if target_seq == "seq2":
        target_type = 1
    elif target_seq == "rev2":
        target_type = 4
    confound_type1 = 0
    confound_type2 = 3
    
    posterior = hmm.smoother(em_params, outers)
    if causal:
        posterior_prob = posterior.filtered_probs
    else:
        posterior_prob = posterior.smoothed_probs
        
    target_posterior = np.sum(posterior_prob[:,np.array(cycle_type) == target_type], axis = 1)
    confound1_posterior = np.sum(posterior_prob[:,np.array(cycle_type) == confound_type1], axis = 1)
    confound2_posterior = np.sum(posterior_prob[:,np.array(cycle_type) == confound_type2], axis = 1)
    
    return target_posterior, confound1_posterior, confound2_posterior

##############
#### Below are for large scale per animal analysis
# change of mind trials

def get_trained_model(animal, day, init_num = 10,
                      p_stay_in_seq = 0.8, p_jump_to_other_seq = 0.2,
                      p_jump_to_other_seq_type = 0.1, p_jump_to_random = 0.5, noise_level = 1):
    emission_all, df_all = get_day_observations(animal, day)
    
    em_log = 0
    em_params = None
    input_params = None
    
    params = {
        "p_stay_in_seq": p_stay_in_seq,
        "p_jump_to_other_seq": p_jump_to_other_seq,
        "p_jump_to_other_seq_type": p_jump_to_other_seq_type,
        "p_jump_to_random": p_jump_to_random}
    for num in range(init_num):
    
        seed = num
        A = construct_transition_matrix(seed, **params)
        E = construct_emission_matrix(noise_level = noise_level)
     
        hmm, init_params, props = initialize_state_space(A, E)
        #props.emissions.probs.trainable = False
        em_params_k, em_log_k = em_state_space(hmm, init_params, props, emission_all)
        print("em log ", em_log_k[-1])
        if em_log_k[-1] > em_log:
            print("found a better solution")
            em_params = em_params_k
            em_log = em_log_k[-1]
            input_params = init_params

    return emission_all, df_all, hmm, em_params, input_params

def return_triggered_matrix(trialIDs, target_posterior, t_minus = 4, t_plus = 4):
    # trialID is index-ed!!!
    # trials in rows
    # 
    max_num = len(target_posterior)
    triggers = np.zeros((len(trialIDs), t_minus + t_plus + 1)) + np.nan
    for row_ind in range(len(trialIDs)):
        trialID = trialIDs[row_ind]
        trial_index = trialID - 1
    
        index = np.arange(np.max([trial_index - t_minus, 0]), np.min([trial_index + t_plus + 1, max_num]))
        triggers[row_ind, index - trial_index + t_minus] = target_posterior[index]

    return triggers

def return_triggered_day(animal, day, seq, p_stay_in_seq = 0.5, # proportional to 4a
                                     p_jump_to_other_seq = 0.2, # proportional to 4B * 5
                                     p_jump_to_other_seq_type = 0.1, # proportional to 3b * 8
                                     p_jump_to_random = 0.5, noise_level = 1, init_num = 5):
    
    params = {
        "p_stay_in_seq": p_stay_in_seq,
        "p_jump_to_other_seq": p_jump_to_other_seq,
        "p_jump_to_other_seq_type": p_jump_to_other_seq_type,
        "p_jump_to_random": p_jump_to_random,
        "noise_level": noise_level}
    emission_all, df_all, hmm, em_params, input_params = get_trained_model(animal, day, init_num = init_num, **params)

    min_num = 0
    max_num = 80
    t_minus = 4
    t_plus = 4
    # input

    com_triggers_all = []
    notcom_triggers_all = []
    for session_ind in range(len(emission_all)):

        df = df_all[session_ind]
        all_outers = np.array(df.OuterWellIndex)-1
        all_outers = jnp.array(all_outers[min_num:max_num])
        if (all_outers == np.nan).any():
            print("nan encountered in outer well visits, skipping this session.")
            continue
        
        target_posterior, confound1_posterior, confound2_posterior = return_target_seq_posterior(hmm, em_params, all_outers, seq)

        # do triggering

        subset_df = df.loc[min_num:max_num]
        subset_df_com = subset_df[subset_df.change_of_mind] #has_remote_interval] ######### need to change
        subset_df_notcom = subset_df[~subset_df.change_of_mind]
        
        com_trialID = np.array(subset_df_com.index)
        notcom_trialID = np.array(subset_df_notcom.index)
        

        com_triggers = return_triggered_matrix(com_trialID, target_posterior, t_minus = t_minus, t_plus = t_plus)
        notcom_triggers = return_triggered_matrix(notcom_trialID, target_posterior, t_minus = t_minus, t_plus = t_plus)
        com_triggers_all.append(com_triggers)
        notcom_triggers_all.append(notcom_triggers)

    com_triggers_day = np.vstack(com_triggers_all)
    notcom_triggers_day = np.vstack(notcom_triggers_all)
    
    return com_triggers_day, notcom_triggers_day
