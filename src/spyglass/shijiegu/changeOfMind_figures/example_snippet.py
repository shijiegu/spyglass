import numpy as np
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from spyglass.shijiegu.ripple_add_replay import select_subset_helper, select_subset_helper_pd
from spyglass.shijiegu.Analysis_SGU import get_linearization_map
from spyglass.common.common_position import TrackGraph, IntervalLinearizedPosition, IntervalPositionInfo
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.load import load_run_sessions, load_position
from spyglass.shijiegu.Analysis_SGU import ChangeofMind
from spyglass.shijiegu.changeOfMind_helper import findProportion
from spyglass.shijiegu.changeOfMind import load_epoch_data_wrapper, findDirectionPlot
from spyglass.shijiegu.changeOfMind import find_direction_dot_product

# plot_decode_spiking() in ripple_add_replay.py, but with only
# 1 - decode
# 2 - broad band band
# 2 - theta LFP
# 3 - MUA
# 4 - animal head speed

def save_data(example, proportion_threshold, output_folder,
              long_data = False,
              likelihood = False, causal = False, use_1d_decode = True, classifier_param_name = 'default_decoding_gpu_4armMaze', variant = None):
      nwb_copy_file_name, epoch_num, session_name, t, ind, (tx, ty) = example
      #day, epoch_num, session_name, trial number, number of change of mind, -tx, +ty
      
      animal = nwb_copy_file_name[:5]
      run_session_ids, run_session_names, pos_session_names = load_run_sessions(nwb_copy_file_name)
      session_name = run_session_names[epoch_num]
      position_name = pos_session_names[epoch_num]
      print(nwb_copy_file_name, session_name, position_name)

    
      # 2. load stateScript
      key={'nwb_file_name':nwb_copy_file_name,'epoch':int(session_name[:2]),
             "proportion":str(proportion_threshold)}
      log_df = ChangeofMind().fetch1_dataframe(key)
    

      # 3. load data
      decode_options = {}
      if len(decode_options.keys()) == 0:
            decode_options["encoding_set"] = '2Dheadspeed_above_4'
            decode_options["classifier_param_name"] = classifier_param_name #"default_decoding_gpu_4armMaze_W40msO20ms"
            decode_options["causal"] = causal
            decode_options["likelihood"] = likelihood
      
        
      (_,decode,head_speed,head_orientation,
       linear_position_xr,lfp_xr,theta_xr,
       ripple_df,neural_df,mua_xr,mua_mean,mua_sd,spikeColInd) = load_epoch_data_wrapper(
             nwb_copy_file_name, session_name, position_name, decode_options,
             load_ripple_flag = False, load_spike_flag = False, use_1d_decode = use_1d_decode)
       

      linear_position_df=(IntervalLinearizedPosition() & {
        'nwb_file_name':nwb_copy_file_name,
        'interval_list_name':position_name,
        'position_info_param_name':'default_decoding'}).fetch1_dataframe()

      position_info = (IntervalPositionInfo() & {
        'nwb_file_name':nwb_copy_file_name,
        'interval_list_name':position_name,
        'position_info_param_name':'default_decoding'}).fetch1_dataframe()
      
      # find turn around times
      camera_frequency = 1/np.mean(np.diff(linear_position_df.index))

      # for each trial
      start = log_df.loc[t,'timestamp_H']
      end = log_df.loc[t,'timestamp_O']

      # restrict to this trial's position info
      trialInd = (linear_position_df.index >= start) &(linear_position_df.index <= end)
      trialPosInfo = linear_position_df.loc[trialInd,:]
      trialPosInfo = trialPosInfo.tail(int(120*camera_frequency)) #use at most xx seconds prior to nose poke at the outer well.
      proportion, track_segment_id, max_proportion, turnaround_time = findProportion(trialPosInfo, camera_frequency)
      turnaround_t = turnaround_time[ind]
       
      # 4. turnaround_t, mua_xr_zscore,
      #turnaround_t = log_df.loc[t,"initial_time"] 
      t0 = turnaround_t
    
      plottimes = [turnaround_t + tx, turnaround_t + ty]
      plot_filename = f"{animal}_{nwb_copy_file_name}_{session_name}_trial{t}_{np.round(plottimes[0],2)}_{np.round(plottimes[1],2)}"
      # if likelihood:
      #       plot_data_filename = f"data_{animal}_{nwb_copy_file_name}_{session_name}_trial{t}_likelihood.pkl"
      # elif causal:
      #       plot_data_filename = f"data_{animal}_{nwb_copy_file_name}_{session_name}_trial{t}_causal.pkl"
      # else:
      #       plot_data_filename = f"data_{animal}_{nwb_copy_file_name}_{session_name}_trial{t}_acausal.pkl"
      if long_data:
            plot_data_filename = f"longdata_variant{variant}_{animal}_{nwb_copy_file_name}_{session_name}_trial{t}.pkl"
      else:
            plot_data_filename = f"data_variant{variant}_{animal}_{nwb_copy_file_name}_{session_name}_trial{t}.pkl"
            
      turnaround_t = np.array(turnaround_t).reshape((-1,1))
      turnaround_t = np.hstack((turnaround_t - 0.01, turnaround_t + 0.01))
      
      # arm_direction_t, arm_direction, _ = findDirectionPlot(t,log_df,
      #                                                       linear_position_df,position_info)
      # head_direction_sign = pd.Series(arm_direction, index = arm_direction_t)
    
      mua_xr_zscore = (mua_xr - mua_mean)/mua_sd

      # 5. select subset
      """select subsets of data"""
      time_slice = slice(plottimes[0], plottimes[1])
      if decode_options["likelihood"]:
            posterior_position_subset=decode.sel(time=time_slice).likelihood.sum(
                  dim='state')
            posterior_position_subset = posterior_position_subset/posterior_position_subset.sum(dim='position')
            #posterior_state_subset=decode.sel(time=time_slice).likelihood.sum('position')
      elif decode_options["causal"]:
            results_subset = select_subset_helper(decode,plottimes)
            posterior_position_subset = results_subset.causal_posterior.sum(dim='state')
            #posterior_state_subset=results_subset.causal_posterior.sum('position')
      else:
            results_subset = select_subset_helper(decode,plottimes)
            posterior_position_subset=results_subset.acausal_posterior.sum(
                  dim='state')
            #posterior_state_subset=results_subset.acausal_posterior.sum('position')
      
      linear_position_subset = select_subset_helper(linear_position_xr,plottimes)
      position_subset = select_subset_helper_pd(position_info,plottimes)
      theta_subset = select_subset_helper(theta_xr, plottimes)
      lfp_subset = select_subset_helper(lfp_xr, plottimes)
      mua_subset = select_subset_helper(mua_xr_zscore, plottimes)
    
      head_speed_subset=head_speed.sel(
        time=head_speed.time[
            np.logical_and(head_speed.time>=plottimes[0],head_speed.time<=plottimes[1])])
      
      head_direction, rightward = find_direction_dot_product(linear_position_subset.to_dataframe(),
                                                  position_subset)

      data = {"plottimes":plottimes,
            "decode":posterior_position_subset,
            "linear_position_subset":linear_position_subset,
            "position_subset": position_subset,
            "theta_subset": theta_subset,
            "lfp_subset": lfp_subset,
            "mua_subset": mua_subset,
            "head_speed_subset": head_speed_subset,
            "output_folder":output_folder,
            "filename":plot_filename,
            "head_direction":head_direction, #dot product with the arm the animal is currently in
            "rightward":rightward, #how aligned is the animal towards 90deg clockwise of the arm
            "t0":t0,
           }
      
      output_path = os.path.join(output_folder,plot_data_filename) #os.join(output_folder,plot_data_filename)
      with open(output_path, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
      print(f"Data successfully pickled and saved to {plot_data_filename}")


def plot_decode_spiking(plottimes,linear_position_subset,posterior_position_subset,theta_subset,
                        lfp_subset,
                        mua_subset, head_speed_subset,
                        savefolder=[], savename=[],
                        turnaround = None, likelihood = False,
                        t0 = None, plot_len = None,
                        rightward = None,
                        head_direction = None, vmax = 0.05):

      if plot_len is None:
            duration = posterior_position_subset.time[-1] - posterior_position_subset.time[0]
            plot_len = duration * 3 + 3

      if (posterior_position_subset.time[-1] - posterior_position_subset.time[0]) > 4:
            plot_len = 10
            
      fig, axes = plt.subplots(5, 1, figsize=(plot_len, 8), sharex=True,
                             constrained_layout=True, gridspec_kw={"height_ratios": [3,0.3,0.5,0.5,0.5]},)
      
      
      
      """plotting"""
      
      '''theta band LFP'''
      if theta_subset is not None:
            
            theta_d=np.array(theta_subset.to_array()).astype('float32').T
            theta_t=np.array(theta_subset.time)
            xticks = theta_t[theta_d[:,0] == 1]
            axes[1].plot(theta_t,theta_d[:,0],linewidth = 2, alpha = 0.5,color = 'C1')
            axes[1].set_title('theta band LFP')
      else:
            xticks = None
            
      # Decode and Position data
      
      #if likelihood:
      posterior_position_subset.plot(
             x='time', y='position', ax=axes[0],vmax = vmax,xticks=xticks,
             rasterized=True, robust=True, cmap='bone_r')
      #axes[0].imshow(posterior_position_subset.T, extent=(
      #      plottimes[0],plottimes[1],0,np.array(posterior_position_subset.position)[-1]),vmax=0.02,cmap='bone_r',rasterized=True,origin='lower')
        
      axes[0].axis('off')
      
      if likelihood:
            axes[0].scatter(linear_position_subset.time,
                        np.array(linear_position_subset.linear_position),
                        s=1, color='magenta', zorder=10, alpha = 0.1)
      else:
            axes[0].scatter(linear_position_subset.time,
                        np.array(linear_position_subset.linear_position),
                        s=1, color='magenta', zorder=10)
      
      if t0 is None:
            t0 = 0.5 * (plottimes[0] + plottimes[1])
      
      axes[0].set_aspect('auto')
      axes[0].set_xlim(plottimes[0],plottimes[1])
      
      # add 50 ms scale bar
      ymin,ymax = axes[0].get_ylim()
      axes[0].plot([plottimes[0]+0.01,plottimes[0]+0.01+0.05],[ymax,ymax], linewidth=2, color='firebrick', alpha=0.5)
      axes[0].text(plottimes[0]+0.01,ymax + 30,'50 ms',fontsize=20)
      
      """head direction"""
      head_direction_axe = 3
      set1_t = np.array(linear_position_subset.time).copy()
      set1_d = head_direction.copy()
      set1_t[rightward < 0.5] = np.nan
      set1_d[rightward < 0.5] = np.nan
      set1_d[set1_d > 0.8] = np.nan
      
      set2_t = np.array(linear_position_subset.time).copy()
      set2_d = head_direction.copy()
      set2_t[rightward >= -0.5] = np.nan
      set2_d[rightward >= -0.5] = np.nan
      set2_d[set2_d > 0.8] = np.nan
      
      set3_t = np.array(linear_position_subset.time).copy()
      set3_d = head_direction.copy()
      set3_d[set3_d <= 0.8] = np.nan
      
      axes[head_direction_axe].plot(set1_t, set1_d, color = 'C0',label = "facing right")
      axes[head_direction_axe].plot(set2_t, set2_d, color = 'C1',label = "facing left")
      axes[head_direction_axe].plot(set3_t, set3_d, color = 'k',label = "aligned with arm")
      axes[head_direction_axe].set_title("head direction dot product maze arm direction")
      axes[head_direction_axe].legend(bbox_to_anchor=(1.2, 1))
      
      '''broad band Power'''
      if lfp_subset is not None:
            lfp_d=np.array(lfp_subset[0]).astype('float32').T
            artifact_time = np.argwhere(lfp_d <= -1200).ravel()
            # lfp_d[artifact_time] = np.nan
            # lfp_d = lfp_d / np.max(np.abs(lfp_d))
            # lfp_t=np.array(lfp_subset.time)
            # axes[1].plot(lfp_t,lfp_d,'k')
            # axes[1].set_title('broad band LFP')
            
            
      
      
      
      
      '''MUA'''
      if mua_subset is not None:
            mua_d=np.array(mua_subset.to_array()).astype('float32').T
            mua_t=np.array(mua_subset.time)
            mua_d[artifact_time] = np.nan
            #axes[2].plot(mua_t,mua_d)
            axes[2].fill_between(mua_t,np.zeros_like(mua_d).squeeze(),mua_d.squeeze(), color = [0.1,0.1,0.1])
            axes[2].set_title('MUA')
            axes[2].set_ylabel("zscore")
            axes[2].set_ylim([-1,4])
            
      
      
      '''speed information'''
      speed_axe = 4
      head_speed_plot = np.array(head_speed_subset.to_array()).ravel()
      axes[speed_axe].plot(np.array(head_speed_subset.time),head_speed_plot)
      axes[speed_axe].spines[['right', 'top']].set_visible(False)
      axes[speed_axe].set_ylim([0, np.nanmax(head_speed_plot) + 4])
      axes[speed_axe].set_title(f'animal head speed')
      axes[speed_axe].set_ylabel("cm/s")
      
      axes[speed_axe].set_xlabel("ms")
      
      if turnaround is not None:
            for turn_ind in range(np.shape(turnaround)[0]):
                  for axes_id in [0]:
                        axes[axes_id].axvspan(turnaround[turn_ind][0], turnaround[turn_ind][1], color = "red", alpha = 0.1)
                        axes[axes_id].axvspan(turnaround[turn_ind][0], turnaround[turn_ind][1], color = "red", alpha = 0.1)
                        axes[axes_id].axvspan(turnaround[turn_ind][0], turnaround[turn_ind][1], color = "red", alpha = 0.1)
            
      """well locations"""
      linear_map,node_location=get_linearization_map()
      for axes_id in [0]:
            for junctions in linear_map[:,1]:
                  axes[axes_id].axhline(junctions, linewidth = 0.2, linestyle = '-.',color = [0.2,0.2,0.2])
            for n in node_location.keys():
                  axes[axes_id].axhline(node_location[n], linewidth = 0.2, linestyle = '-.',color = [0.2,0.2,0.2])
      
      for axes_id in [0,1,2,3]:
            axes[axes_id].spines[['right', 'top']].set_visible(False)
      
      xticklabels = ["" for x in xticks]
      ind_first = np.argwhere((xticks-t0) > 0).ravel()[0]
      xticklabels[ind_first] = int(np.round(xticks[ind_first] - t0,3) * 1000)
      xticklabels[ind_first+1] = int(np.round(xticks[ind_first+1] - t0,3) * 1000)
      
      for axes_id in [0,1,2,3]:
            axes[axes_id].spines[['bottom']].set_visible(False)
            if axes_id == 1:
                  axes[1].tick_params(axis='x', top=True, bottom=True) 
            if xticks is not None:
                  axes[axes_id].set_xticks(xticks)
                  axes[axes_id].set_xticklabels(xticklabels,fontsize = 7, rotation = 45)
      
      #x_axis2 = np.concatenate((np.arange(t0, plottimes[0],-0.2), np.arange(t0,plottimes[1],0.2)))
      #axes[3].set_xticks(x_axis2)
      #axes[3].set_xticklabels(np.round(x_axis2 - t0,2))
            
      # axes_id = 0
      # graph = TrackGraph() & {'track_graph_name': '4 arm lumped 2023'}
      # graph.plot_track_graph_as_1D(
      #       ax=axes[axes_id],
      #       axis='y',
      #       other_axis_start = plottimes[0]-0.1)
      
      if len(savefolder)>0:
            plt.savefig(os.path.join(savefolder,savename+'.pdf'),format="pdf",bbox_inches='tight',dpi=200,transparent=True)
        #plt.savefig(os.path.join(exampledir,'ripple_'+str(ripple_num)+'.png'),bbox_inches='tight',dpi=300)
      
      return fig, axes