import os
import pickle
import pandas as pd
import statsmodels.api as sm
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import logging
import multiprocessing
from spyglass.shijiegu.Analysis_SGU import TrialChoice, ChangeofMind
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename
from spyglass.shijiegu.changeOfMind import color_by_rat


seq1=[1,3,2,4]
seq2=[1,3,4,2]
seq3=[1,2,3,4]

rev1=[1,4,2,3]
rev2=[1,2,4,3]
rev3=[1,4,3,2]

seqs=np.vstack((seq1,seq2,seq3,rev1,rev2,rev3))
orders=['Seq1','Seq2','Seq3','Rev1','Rev2','Rev3'] #'1st repeat','2nd repeat','3rd repeat'

P_task=np.zeros((4,4,6))
for s in range(6):
    for a in range(4):
        P_task[seqs[s,a%4]-1,seqs[s,(a+1)%4]-1,s]=1

# for each session,
# - calculate percentage of conforming transitions according to each sequence
def return_sd(p,n):
    return np.sqrt(p*(1-p)/n)

def correctPairSessions(index, outers):
    pctCorrect = np.zeros(6)

    P_behavior_count = find_behavior_transitions(index,outers)
        
    for seq_ind in range(6):

        pctCorrect[seq_ind] = np.sum(np.multiply(P_behavior_count,P_task[:,:,seq_ind]))/np.sum(P_behavior_count)
        
    return pctCorrect

def rewardTrialsSessions(index, outers, reward):
    pctCorrect = np.sum(reward == 2)/len(outers)
    return pctCorrect

def determine_if_include_session(log_df):
    nan_index = np.argwhere(np.isnan(log_df['timestamp_O']))
    if len(nan_index) == 0:
        nan_index = np.array([[len(log_df)]])
    last_row = int(nan_index.ravel()[0])
    log_df = log_df[:last_row]
    if len(log_df) < 20:
        return False, log_df
    else:
        return True, log_df

def return_pctCorrect(animal, dates_to_plot):
    pctCorrect_sessions, numTrials_sessions = {}, {}
    pctCorrect_days = {}
    pctCorrect_days_sd = {}
    pctReward_days = {}
    pctReward_days_sd = {}

    for day in dates_to_plot:
    
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)

        numTrials_day = []
        pctCorrect_day = []
        pctReward_day = []
        for session_name in session_interval:
            q = (TrialChoice() & {"nwb_file_name": nwb_copy_file_name, "epoch":int(session_name[:2])})
            assert q.fetch1("epoch_name") == session_name
            log_df = pd.DataFrame(q.fetch1("choice_reward"))
            include_session, log_df = determine_if_include_session(log_df)
            if not include_session: # do not include short sessions
                continue
            log_df = log_df[~np.isnan(log_df.timestamp_O)]
            
            index = np.array(log_df.index)
            outers = np.array(log_df.OuterWellIndex)
            reward = np.array(log_df.rewardNum).astype("int")

            pctCorrect = correctPairSessions(index, outers)
            pctReward = rewardTrialsSessions(index, outers, reward)
                
            numTrials = len(log_df)
            
            info = (day, session_name)
            pctCorrect_sessions[info] = pctCorrect
            numTrials_sessions[info] = numTrials
            
            numTrials_day.append(numTrials)
            pctCorrect_day.append(pctCorrect)
            pctReward_day.append(pctReward)
        
        numTrials_day = np.array(numTrials_day)
        pctCorrect_days[day] = np.array(pctCorrect_day).T @ numTrials_day/np.sum(numTrials_day)
        pctCorrect_days_sd[day] = return_sd(pctCorrect_days[day],np.sum(numTrials_day))
        
        pctReward_days[day] = np.array(pctReward_day).T @ numTrials_day/np.sum(numTrials_day)
        pctReward_days_sd[day] = return_sd(pctReward_days[day],np.sum(numTrials_day))
    return pctCorrect_days, pctCorrect_days_sd, pctReward_days, pctReward_days_sd

def find_behavior_transitions(index,outers): #P(xi|xj)
    '''returns numbers of transition'''
    T=np.zeros((4,4))
    for t in range(len(outers)-1):
        if (index[t+1]-index[t])==1: #adjacent trials
            T[int(outers[t])-1,int(outers[t+1])-1] += 1 # minus 1 due to python indexing
    #for ti in range(4):
    #    T[ti]=T[ti]/np.sum(T[ti])
    return T


def find_COM_trials(animal, dates_to_plot, proportion_threshold = 0.1):
    """return the number of COM trials for each day in dates_to_plot"""
    #trials_days = find_trials_animal(animal,dates_to_plot,proportion_threshold = proportion_threshold)

    trial_number = []
    for day in dates_to_plot:
        
        trial_number_d = 0
        total_trial_number_d = 0
        
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        
        for session in session_interval:
            q = {"nwb_file_name":nwb_copy_file_name,
                 "epoch":int(session[:2]), "proportion": proportion_threshold}
            q_result = ChangeofMind() & q
            if len(q_result) == 0:
                continue
            info_table = ChangeofMind().fetch1_dataframe(q)
            include_session, info_table = determine_if_include_session(info_table)
            if not include_session: # do not include short sessions
                continue
            info_table = info_table[~np.isnan(info_table.timestamp_O)]
            
            total_trial_number_d += (len(info_table)-1)
            trial_number_d += len(np.argwhere(np.array(info_table.change_of_mind) == True).ravel())
        #print(f"{day} has {total_trial_number_d} trials.")
        if trial_number_d == 0:
            trial_number.append(0)
        else:
            trial_number.append(trial_number_d / total_trial_number_d)
    return trial_number

def find_COM_trial_number(animal, dates_to_plot, proportion_threshold = 0.1):
    """return the number of COM trials for each day in dates_to_plot"""
    #trials_days = find_trials_animal(animal,dates_to_plot,proportion_threshold = proportion_threshold)

    trial_number = []
    com_number = []
    reward_number = []
    for day in dates_to_plot:
        
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)
        
        for session in session_interval:
            q = {"nwb_file_name":nwb_copy_file_name,
                 "epoch":int(session[:2]), "proportion": proportion_threshold}
            q_result = ChangeofMind() & q
            if len(q_result) == 0:
                continue
            info_table = ChangeofMind().fetch1_dataframe(q)
            include_session, info_table = determine_if_include_session(info_table)
            if not include_session: # do not include short sessions
                continue
            info_table = info_table[~np.isnan(info_table.timestamp_O)]
            #info_table = pd.read_pickle(info)
            
            total_trial_number_d = len(info_table)
            com_number_d = len(np.argwhere(np.array(info_table.change_of_mind) == True).ravel())

            trial_number.append(total_trial_number_d)
            com_number.append(com_number_d)
            reward_number.append(len(info_table[info_table.rewardNum == 2]))
    return trial_number, com_number, reward_number
        

def findExactSequence(outers,level=3):
    # outers: outer arm visit, 1, 2, 3, 4
    # level 5 is a complete sequence with 5 correct in a row/unbroken.
    # level 4 is 4 correct vists in a row.
    # level 3 is 3 correct vists in a row, etc
    
    # seq 1-3, rev 1-3
    SEQN = 6*4; #6 sequence, each 4 triplets
    occurance = np.zeros((len(outers),SEQN))+np.nan
    runningsum_outer=signal.convolve(outers**2,np.ones(level))
    
    for se in range(6):
        conv=np.zeros(len(outers))
        for p in range(4): #permutation
            seq_tmp=np.roll(np.hstack((seqs[se],seqs[se])),-p)[:level]
            seq_norm=np.linalg.norm(seq_tmp)
            conv_tmp=signal.convolve(outers,np.flip(seq_tmp))/(seq_norm*np.sqrt(runningsum_outer))
            conv = np.concatenate((np.zeros(level - 1),np.abs(conv_tmp[level-1:-(level-1)] - 1)<=np.exp(-10)));
            conv =np.array(conv>0).astype('int')
        
            occurance[np.arange(len(outers)),se*4+p]=conv
    
    return occurance




def bootstrap_behavior(outer):
    '''
    bootstrap the behavior
    '''

    totalTrialNum = len(outer)
    
    # sample with replacement
    outerBoot = np.random.choice(outer,totalTrialNum)

    return outerBoot

def plot_learning_curve(animal, mean_data, sd_data, COM_rate_animals1, COM_rate_animals2, COM_rate_animals3,
                        peak_day,
                        target_sequence_animal, savename = None):
    """
    plot learning curve for one animal
    
    target_sequence_animal: is the index of the target sequence for the animal
    (0-5 for Seq1-3, Rev1-3)
    """

    (trial_number1, trial_number2, trial_number3) = (COM_rate_animals1,#[animal],
                                                     COM_rate_animals2,#[animal],
                                                     COM_rate_animals3)#[animal]) # C-o-M trials
    
    days = list(mean_data.keys())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 2),sharex=True)

    colors = [[0.6,0.6,0.6] for k in range(6)]
    colors[target_sequence_animal] = 'C1'
    print(colors)
    
    for seq_ind in range(6):
        # if seq_ind == target_sequence_animal:
        #     linewidth, alpha = 3, 1
        #     color = 'C1'
        # else:
        #     linewidth, alpha = 1, 0.1
        #     color = 'grey'
        color = f"C{seq_ind}"
        y = np.array( [mean_data[k][seq_ind] for k in mean_data])
        sd = np.array( [sd_data[k][seq_ind] for k in sd_data] )
        ax1.plot(days, y * 100, color = color, alpha = 0.5, linewidth = 3)
        ax1.fill_between(days, (y - sd) * 100, (y + sd) * 100, color = color, alpha = 0.4)
        
        
    ax2.plot(days, np.array(trial_number1) * 100, label = "5 cm in", color = 'k',linestyle = "dashed")
    ax2.plot(days, np.array(trial_number2) * 100, label = "10 cm in", linewidth = 1, color = 'k')
    ax2.plot(days, np.array(trial_number3) * 100, label = "20 cm in",color = 'k',linestyle = "dotted")
    #ax2.legend(bbox_to_anchor=(1.1, 1.05))
    

    ax1.text(0,70,animal,fontsize = 12)
    ax1.set_ylabel('% of trials \n conforming')
    ax2.set_ylabel('ch. of m. \n pct of trials')
    #ax3.set_ylabel('$\|P - P_{target}\|$');
    
    dates_to_plot = days
    peak_day_index = dates_to_plot.index(peak_day)
    # shift x axis to align peak performance days
    ax2.set_xlabel('day, relative to peak performance day')
    
    # choose every other day to be in xticks
    #if len(dates_to_plot) % 2 == 0:
    #    xticks = np.arange(len(dates_to_plot))[1::2]
    #else:
    xticks = np.arange(len(dates_to_plot))
    ax2.set_xticks(xticks)
    daylabels = xticks - peak_day_index
    ax2.set_xticklabels(daylabels, rotation = 0) #rotation = 45

    ax1.spines[['right', 'top']].set_visible(False)
    ax2.spines[['right', 'top']].set_visible(False)
    
    if savename is not None:
        plt.savefig(savename,bbox_inches='tight',dpi = 300)
        
def plot_learning_curve_main(animal, mean_data, sd_data, COM_rate_animals1,
                        peak_day,target_seq_ind,
                        savename = None):
    """
    plot learning curve for one animal, simplified, for paper main text.
    """

    trial_number1 = COM_rate_animals1
    
    days = list(mean_data.keys())
    
    fig, ax1 = plt.subplots(1, 1, figsize=(3, 1.5),sharex=True)
    color = "k"#color_by_rat[animal]
    y = np.array( [mean_data[k][target_seq_ind] for k in mean_data])
    sd = np.array( [sd_data[k][target_seq_ind] for k in sd_data] )
    ax1.scatter(days, y * 100, color = color)
    ax1.plot(days, y * 100, color = color, alpha = 0.5, linewidth = 3, label = "learning curve")
    
    ax1.axhline(y = 1/4 * 100, color = "k", linewidth = 1, linestyle = ":")
    
    ax1.fill_between(days, (y - sd) * 100, (y + sd) * 100, color = color, alpha = 0.2)
    
    # ax2 = ax1.twinx()
    # ax2.plot(days, np.array(trial_number1) * 100, label = "change of mind",
    #          alpha = 0.5, linewidth = 3,
    #          color = [0.5,0.5,0.5])
    # ax2.legend(bbox_to_anchor=(1.1, 1.5))
    ax1.legend(bbox_to_anchor=(0.1, 1.5))
    ax1.set_ylim(15,65)
    

    ax1.text(0,80,f"Rat {animal[0].upper()}",fontsize = 12)
    ax1.set_ylabel('% conforming',color = color)
    #ax2.set_ylabel('chng.of.mind %')
    #ax3.set_ylabel('$\|P - P_{target}\|$');
    
    dates_to_plot = days
    peak_day_index = dates_to_plot.index(peak_day)
    # shift x axis to align peak performance days
    ax1.set_xlabel('day, relative to peak performance day')
    
    # choose every other day to be in xticks
    #if len(dates_to_plot) % 2 == 0:
    #    xticks = np.arange(len(dates_to_plot))[1::2]
    #else:
    xticks = np.arange(len(dates_to_plot))
    ax1.set_xticks(xticks)
    daylabels = xticks - peak_day_index
    ax1.set_xticklabels(daylabels, rotation = 0) #rotation = 45

    ax1.spines[['right', 'top']].set_visible(False)
    #ax2.spines[['top']].set_visible(False)
    
    if savename is not None:
        plt.savefig(savename,bbox_inches='tight',dpi = 300)
        
def return_neighbor_nonneighor(P_behavior):
    neighors = [(0,1),(1,0),(2,3),(3,2)] #same side
    nonneighbors = [(0,2),(0,3),
                    (1,2),(1,3),
                    (2,0),(2,1),
                    (3,0),(3,1)] #switch side

    P_neighors = np.zeros(P_behavior.shape[2])
    P_nonneighors = np.zeros(P_behavior.shape[2])
    for n in neighors:
        P_neighors += P_behavior[n[0],n[1]]

    for n in nonneighbors:
        P_nonneighors += P_behavior[n[0],n[1]]
    return P_neighors, P_nonneighors
        
def get_transition_matrix_animal(animal, animal_save_name, dates_to_plot):
    """all behavior"""
    
    T_average_days = {}
    for day in dates_to_plot:
    
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)

        numTrials_day = []
        pctCorrect_day = []
        T_day = []
        for session_name in session_interval:
            q = (TrialChoice() & {"nwb_file_name": nwb_copy_file_name,
                                  "epoch":int(session_name[:2])})
            assert q.fetch1("epoch_name") == session_name
            log_df = pd.DataFrame(q.fetch1("choice_reward"))
            include_session, log_df = determine_if_include_session(log_df)
            if not include_session: # do not include short sessions
                continue
            index = np.array(log_df.index)
            outers = np.array(log_df.OuterWellIndex)
            T = find_behavior_transitions(index,outers)
            numTrials = len(log_df)
            
            numTrials_day.append(numTrials)
            T_day.append(T)
            
        numTrials_day = np.array(numTrials_day)
        numTrials_day = numTrials_day/np.sum(numTrials_day)
        T_average_day = np.zeros((4,4))
        for t in range(len(T_day)):
            T_average_day += T_day[t]*numTrials_day[t]
            
        T_average_days[day] = T_average_day
        
    return T_average_days

def get_com_num_animal(animal, days_to_plot, proportion_threshold = 0.1):
    """all behavior"""
    
    com_num = []
    for day in days_to_plot:
    
        nwb_file_name = animal.lower() + day + '.nwb'
        nwb_copy_file_name = get_nwb_copy_filename(nwb_file_name)
        session_interval, position_interval = runSessionNames(nwb_copy_file_name)

        com_num_day = 0
        for session_name in session_interval:
            q = {"nwb_file_name":nwb_copy_file_name,
                 "epoch":int(session_name[:2]), "proportion": proportion_threshold}
            q_result = ChangeofMind() & q
            if len(q_result) == 0:
                continue
            info_table = ChangeofMind().fetch1_dataframe(q)
            include_session, info_table = determine_if_include_session(info_table)
            if not include_session: # do not include short sessions
                continue
            
            CoMNum_by_arm = np.array(info_table.CoMNum_by_arm)
            com_num_day = CoMNum_by_arm[CoMNum_by_arm > 0]
            com_num.extend(com_num_day.tolist())
        
    return com_num
        


