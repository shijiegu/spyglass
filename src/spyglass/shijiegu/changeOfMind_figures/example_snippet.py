import numpy as np
import os
import pandas as pd
import xarray as xr
import pickle
import matplotlib.pyplot as plt
from spyglass.shijiegu.ripple_add_replay import select_subset_helper, select_subset_helper_pd
from spyglass.shijiegu.Analysis_SGU import get_linearization_map, Imu
from spyglass.common.common_position import TrackGraph, IntervalLinearizedPosition, IntervalPositionInfo
from spyglass.shijiegu.decodeHelpers import runSessionNames
from spyglass.shijiegu.load import load_run_sessions, load_position
from spyglass.shijiegu.Analysis_SGU import ChangeofMind
from spyglass.shijiegu.changeOfMind_helper import findProportion
from spyglass.shijiegu.changeOfMind import load_epoch_data_wrapper, findDirectionPlot
from spyglass.shijiegu.changeOfMind import find_direction_dot_product, find_direction_dot_product_single_arm
from ripple_detection.core import segment_boolean_series

from position_tools import (
    get_angle,
    get_distance,
    get_speed,
    get_velocity,
    interpolate_nan,
)

# plot_decode_spiking() in ripple_add_replay.py, but with only
# 1 - decode
# 2 - broad band band
# 2 - theta LFP
# 3 - MUA
# 4 - animal head speed

def save_data(example, proportion_threshold, output_folder,
              likelihood = False, causal = False, use_1d_decode = True,
              imu_name = "big_acc_bias",
              classifier_param_name = 'default_decoding_gpu_4armMaze', variant = None):
      nwb_copy_file_name, epoch_num, session_name, t, ind, (tx, ty), long_data, arm = example
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
        'track_graph_name': '4 arm lumped 2023',
        'position_info_param_name':'default_decoding'}).fetch1_dataframe()

      position_info = (IntervalPositionInfo() & {
        'nwb_file_name':nwb_copy_file_name,
        'interval_list_name':position_name,
        'position_info_param_name':'default_decoding'}).fetch1_dataframe()
      
      # load IMU
      if imu_name is not None:
            key_imu={'nwb_file_name':nwb_copy_file_name,
                  'epoch':int(session_name[:2]),
                  'trial':t,
                  "parameter":imu_name}
            postion_info_gyro = Imu().fetch1_dataframe(key_imu)
      else:
            postion_info_gyro = position_info
      
      
      # find turn around times
      camera_frequency = 1/np.mean(np.diff(linear_position_df.index))

      # for each trial
      start = log_df.loc[t,'timestamp_H']
      end = log_df.loc[t,'timestamp_O']

      # restrict to this trial's position info
      # trialInd = (linear_position_df.index >= start) &(linear_position_df.index <= end)
      # trialPosInfo = linear_position_df.loc[trialInd,:]
      # trialPosInfo = trialPosInfo.tail(int(120*camera_frequency)) #use at most xx seconds prior to nose poke at the outer well.
      # proportion, track_segment_id, max_proportion, turnaround_time = findProportion(trialPosInfo, camera_frequency)
      # turnaround_t = turnaround_time[ind]
      turnaround_t =  log_df.loc[t,'initial_time']
      
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
    
      mua_xr_zscore = mua_xr#(mua_xr - mua_mean)/mua_sd

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
      
      postion_info_gyro_subset = select_subset_helper_pd(postion_info_gyro,plottimes)
      linear_position_subset = linear_position_xr.interp(time=np.array(postion_info_gyro_subset.index),
                                               method="nearest")
      
      #linear_position_subset = select_subset_helper(linear_position_xr,plottimes_updated)
      #position_subset = select_subset_helper_pd(position_info,plottimes_updated)
      
      theta_subset = select_subset_helper(theta_xr, plottimes)
      lfp_subset = select_subset_helper(lfp_xr, plottimes)
      mua_subset = select_subset_helper(mua_xr_zscore, plottimes)
      
      if imu_name is not None:
            head_speed_subset = get_speed(
                  np.array(postion_info_gyro_subset)[:,[0,1]],
                  postion_info_gyro_subset.index,
                  sigma=0.001,
                  sampling_frequency=1/np.mean(np.diff(np.array(postion_info_gyro_subset.index))),
            )
      else:
            head_speed_subset = np.array(postion_info_gyro_subset.head_speed)
      # pack into xarray
      head_speed_subset = pd.DataFrame(data= head_speed_subset, index= postion_info_gyro_subset.index)
      head_speed_subset.index.name='time'
      head_speed_subset= xr.Dataset.from_dataframe(head_speed_subset)
      
      
      angular_speed_subset = get_velocity(
            np.array(postion_info_gyro_subset)[:,2],
            postion_info_gyro_subset.index,
            sigma=0.001,
            sampling_frequency=1/np.mean(np.diff(np.array(postion_info_gyro_subset.index))),
      )
      # pack into xarray
      angular_speed_subset = pd.DataFrame(data= angular_speed_subset, index= postion_info_gyro_subset.index)
      angular_speed_subset.index.name='time'
      angular_speed_subset= xr.Dataset.from_dataframe(angular_speed_subset)
    
      # head_speed_subset=head_speed.sel(
      #   time=head_speed.time[
      #       np.logical_and(head_speed.time>=plottimes[0],head_speed.time<=plottimes[1])])
      
      if np.isnan(arm):
            head_direction, rightward = find_direction_dot_product(linear_position_subset.to_dataframe(),
                                                        postion_info_gyro_subset)
      else:
            head_direction, rightward = find_direction_dot_product_single_arm(arm, #find_direction_dot_product(linear_position_subset.to_dataframe(),
                                                        postion_info_gyro_subset)

      data = {"plottimes":plottimes,
            "decode":posterior_position_subset,
            "linear_position_subset":linear_position_subset,
            "position_subset": postion_info_gyro_subset,
            "postion_info_gyro_subset":postion_info_gyro_subset,
            "theta_subset": theta_subset,
            "lfp_subset": lfp_subset,
            "mua_subset": mua_subset,
            "head_speed_subset": head_speed_subset,
            "angular_speed_subset": angular_speed_subset,
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


def plot_behavior(plottimes,linear_position_subset,posterior_position_subset,
                        head_speed_subset, head_direction,
                        head_direction_lim = 8,
                        savefolder=[], savename=[],
                        turnaround = None,
                        t0 = None, plot_len = None,
):

      if plot_len is None:
            if (posterior_position_subset.time[-1] - posterior_position_subset.time[0]) > 4:
                  plot_len = 10
            else:
                  duration = posterior_position_subset.time[-1] - posterior_position_subset.time[0]
                  plot_len = duration * 3 + 3

      decode_subpanel_lengths = np.array([43, 44, 44, 44, 16, 29])
      decode_subpanel_lengths = decode_subpanel_lengths / np.sum(decode_subpanel_lengths) * 2
      height_ratios = np.concatenate((decode_subpanel_lengths, np.array([0.5,0.5])))
            
      fig, axes = plt.subplots(8, 1, figsize=(plot_len, 5), sharex=True,
                             constrained_layout=True, gridspec_kw={"height_ratios": height_ratios},)
      
      xticks1 = np.arange(t0, posterior_position_subset.time[0], -1)[::-1]
      xticks2 = np.arange(t0, posterior_position_subset.time[-1], 1)
      xticks = np.concatenate((xticks1, xticks2[1:]))
      xticklabels = [np.round(x-t0,0) for x in xticks]
      
      """plotting"""
            
      # Decode and Position data
      positions, posterior_position_subset, axes_ind = decode_by_regions(posterior_position_subset, region)
      plot_decode_subpanel(axes_ind, posterior_position_subset, axes, False)

      for ind in axes_ind:
            axes[ind].scatter(linear_position_subset.time,
                        np.array(linear_position_subset.linear_position),
                        s=1, color='magenta', zorder=10)
            axes[ind].set_xlim(plottimes[0],plottimes[1])
            
      
      if t0 is None:
            t0 = 0.5 * (plottimes[0] + plottimes[1])
      
      #axes[0].set_aspect('auto')
      
      
      """head direction"""
      head_direction_axe = 7
      angular_speed_plot = np.abs(np.array(head_direction.to_array()).ravel())
      
      axes[head_direction_axe].plot(np.array(head_direction.time), angular_speed_plot, color = 'k')
      axes[head_direction_axe].set_title("head angular speed")
      axes[head_direction_axe].set_ylabel("rad/s")
      axes[head_direction_axe].set_ylim([0,head_direction_lim])
      if head_direction_lim > 50:
            axes[head_direction_axe].set_yticks([50, 70])
      axes[head_direction_axe].spines[['right', 'top']].set_visible(False)
      axes[head_direction_axe].set_xlabel("seconds")
      
      '''speed information'''
      speed_axe = 6
      head_speed_plot = np.array(head_speed_subset.to_array()).ravel()
      axes[speed_axe].plot(np.array(head_speed_subset.time),head_speed_plot, color = 'k')
      axes[speed_axe].spines[['right', 'top']].set_visible(False)
      axes[speed_axe].set_ylim([0, np.nanmax(head_speed_plot) + 4])
      axes[speed_axe].set_title(f'head speed')
      axes[speed_axe].set_ylabel("cm/s")
      
  
      colors = ["grey","steelblue","C3","C1","C2","C4"]
      for axes_id in [0,1,2,3,4,5]:
            axes[axes_id].axvspan(xticks[0]-1, xticks[-1]+1, color = colors[::-1][axes_id], alpha = 0.1, zorder = -10)
      
      if turnaround is not None:
            for turn_ind in range(np.shape(turnaround)[0]):
                  for axes_id in [0,1,2,3,4,5]:
                        axes[axes_id].axvspan(turnaround[turn_ind][0], turnaround[turn_ind][1], color = "red", alpha = 0.1)
                        axes[axes_id].axvspan(turnaround[turn_ind][0], turnaround[turn_ind][1], color = "red", alpha = 0.1)
                        axes[axes_id].axvspan(turnaround[turn_ind][0], turnaround[turn_ind][1], color = "red", alpha = 0.1)
            
      """well locations"""
      # linear_map,node_location=get_linearization_map()
      # for axes_id in [0]:
      #       for junctions in linear_map[:,1]:
      #             axes[axes_id].axhline(junctions, linewidth = 0.5, linestyle = '-.',color = [0.2,0.2,0.2])
      #       for n in node_location.keys():
      #             axes[axes_id].axhline(node_location[n], linewidth = 0.5, linestyle = '-.',color = [0.2,0.2,0.2])
      
      
      
      
      for axes_id in [6]:
            axes[axes_id].spines[['right', 'top']].set_visible(False)
            axes[axes_id].spines[['bottom']].set_visible(False)
            if axes_id == 1:
                  axes[1].tick_params(axis='x', top=True, bottom=True) 
            if xticks is not None:
                  axes[axes_id].set_xticks(xticks)
                  axes[axes_id].set_xticklabels(xticklabels,fontsize = 10)

      
      if len(savefolder)>0:
            plt.savefig(os.path.join(savefolder,savename+'.pdf'),format="pdf",bbox_inches='tight',dpi=200,transparent=True)
        #plt.savefig(os.path.join(exampledir,'ripple_'+str(ripple_num)+'.png'),bbox_inches='tight',dpi=300)
      
      return fig, axes

from scipy.signal import find_peaks
SQRT2 = 0.5 * np.sqrt(2)
def plot_decode_spiking(plottimes,linear_position_subset,posterior_position_subset,theta_subset,
                        lfp_subset,
                        mua_subset, head_speed_subset,
                        savefolder=[], savename=[],
                        turnaround = None, likelihood = False,
                        t0 = None, plot_len = None,
                        rightward = None,
                        head_direction = None, location_size = 5, vmin = 0.01, vmax = 0.06,
                        local_extended_theta_interval = None, remote_theta_interval = None):

      if plot_len is None:
            if (posterior_position_subset.time[-1] - posterior_position_subset.time[0]) > 4:
                  plot_len = 10
            else:
                  duration = posterior_position_subset.time[-1] - posterior_position_subset.time[0]
                  plot_len = duration * 3 + 3

      decode_subpanel_lengths = np.array([43, 44, 44, 44, 16, 29])
      decode_subpanel_lengths = decode_subpanel_lengths / np.sum(decode_subpanel_lengths) * 3
      height_ratios = np.concatenate((decode_subpanel_lengths, np.array([0.3,0.5,0.5,0.5,0.5])))
            
      fig, axes = plt.subplots(11, 1, figsize=(plot_len, 8), sharex=True,
                             constrained_layout=True, gridspec_kw={"height_ratios": height_ratios},)
      
      
      
      """plotting"""
      
      '''theta band LFP'''
      theta_axe_ind = 2+5
      if theta_subset is not None:
            
            theta_d=np.array(theta_subset.to_array()).astype('float32').T
            theta_t=np.array(theta_subset.time)
            xticks_ind, _ = find_peaks(theta_d[:,1])
            xticks = theta_t[xticks_ind]
            axes[theta_axe_ind].plot(theta_t,theta_d[:,1],linewidth = 2, alpha = 0.5,color = 'C1')
            axes[theta_axe_ind].set_title('theta band LFP')
      else:
            xticks = None
            
      # Decode and Position data
      positions, posterior_position_subset, axes_ind = decode_by_regions(posterior_position_subset, region)
      plot_decode_subpanel(axes_ind, posterior_position_subset, axes, vmin = vmin, vmax = vmax)
      
      #if likelihood:
      # posterior_position_subset.plot(
      #        x='time', y='position', ax=axes[0],vmax = vmax,xticks=xticks,
      #        rasterized=True, robust=True, cmap='bone_r')
      #axes[0].imshow(posterior_position_subset.T, extent=(
      #      plottimes[0],plottimes[1],0,np.array(posterior_position_subset.position)[-1]),vmax=0.02,cmap='bone_r',rasterized=True,origin='lower')


      for ind in axes_ind:
            axes[ind].scatter(linear_position_subset.time,
                        np.array(linear_position_subset.linear_position),
                        s=location_size, color='magenta', zorder=10)
            # position_ = np.array(linear_position_subset.linear_position)
            # subset_ind = np.logical_and(position_ >= np.min(positions[5-ind]), position_ <= np.max(positions[5-ind]))
            
            # if np.sum(subset_ind) >= 0:
            #       t_ = np.array(linear_position_subset.time)
            #       t_[~subset_ind] = np.nan
            #       position_[~subset_ind] = np.nan
            #       axes[ind].plot(t_,
            #             position_,
            #             linewidth=3, color='magenta', zorder=15, alpha =0.5)

            axes[ind].set_xlim(plottimes[0],plottimes[1])
            
      
      if t0 is None:
            t0 = 0.5 * (plottimes[0] + plottimes[1])
      
      #axes[0].set_aspect('auto')
      
      
      # add 50 ms scale bar
      ymin,ymax = axes[0].get_ylim()
      axes[0].plot([plottimes[0]+0.01,plottimes[0]+0.01+0.1],[ymax,ymax], linewidth=2, color='firebrick', alpha=0.5)
      axes[0].text(plottimes[0]+0.01,ymax + 30,'100 ms',fontsize=20)
      
      """head direction"""
      head_direction_axe = 9
      set1_t = np.array(linear_position_subset.time).copy()
      set1_d = head_direction.copy()
      #set1_t[rightward < 0.5] = np.nan
      #set1_d[rightward < 0.5] = np.nan
      #set1_d[set1_d > 0.8] = np.nan
      
      is_perpendicular = pd.Series(np.abs(rightward) > SQRT2, index = set1_t)
      is_perpendicular_segments = np.array(segment_boolean_series(
            is_perpendicular, minimum_duration=0.1))

      
      # set2_t = np.array(linear_position_subset.time).copy()
      # set2_d = head_direction.copy()
      # set2_t[rightward >= -0.5] = np.nan
      # set2_d[rightward >= -0.5] = np.nan
      # set2_d[set2_d > 0.8] = np.nan
      
      set3_t = np.array(linear_position_subset.time).copy()
      set3_d = head_direction.copy()
      #set3_d[set3_d <= 0.8] = np.nan
      
      is_aligned = pd.Series(np.abs(head_direction) > SQRT2, index = set1_t)
      is_aligned_segments = np.array(segment_boolean_series(
            is_aligned, minimum_duration=0.1))

      
      #axes[head_direction_axe].plot(set1_t, set1_d, color = 'C0',label = "facing right")
      #axes[head_direction_axe].plot(set2_t, set2_d, color = 'C1',label = "facing left")
      # for seg in is_aligned_segments:
      #       axes[head_direction_axe].axvspan(seg[0], seg[1], color = 'k', alpha = 0.1)
      #for seg in is_perpendicular_segments:
      #      axes[head_direction_axe].axvspan(seg[0], seg[1], color = 'C1', alpha = 0.1)
      axes[head_direction_axe].plot(set3_t, set3_d, color = 'k')
      axes[head_direction_axe].set_title("head direction dot product maze arm direction")
      axes[head_direction_axe].spines[['right', 'top']].set_visible(False)
      axes[head_direction_axe].set_ylim([-1,1.1])
      axes[head_direction_axe].set_yticks([-1,0,1])
      #axes[head_direction_axe].legend(bbox_to_anchor=(1.2, 1))
      
      '''broad band Power'''
      lfp_axe = 1+5
      if lfp_subset is not None:
            lfp_d=np.array(lfp_subset[0]).astype('float32').T
            artifact_time = np.argwhere(lfp_d <= -1200).ravel()
            lfp_d[artifact_time] = np.nan
            #lfp_d = lfp_d / np.max(np.abs(lfp_d))
            lfp_t=np.array(lfp_subset.time)
            axes[lfp_axe].plot(lfp_t,lfp_d,'k')
            axes[lfp_axe].set_title('broad band LFP')
      
      
      '''MUA'''
      if mua_subset is not None:
            mua_ind = 3 + 5
            mua_d=np.array(mua_subset.to_array()).astype('float32').T
            mua_t=np.array(mua_subset.time)
            mua_d[artifact_time] = np.nan
            #axes[2].plot(mua_t,mua_d)
            axes[mua_ind].fill_between(mua_t,np.zeros_like(mua_d).squeeze(),mua_d.squeeze(), color = [0.1,0.1,0.1])
            axes[mua_ind].set_title('MUA')
            axes[mua_ind].set_ylabel("spikes")
            #axes[mua_ind].set_ylim([-1,4])
            
      
      
      '''speed information'''
      speed_axe = 5 + 5
      head_speed_plot = np.array(head_speed_subset.to_array()).ravel()
      axes[speed_axe].plot(np.array(head_speed_subset.time),head_speed_plot,color = 'k')
      axes[speed_axe].spines[['right', 'top']].set_visible(False)
      axes[speed_axe].set_ylim([0, np.nanmax(head_speed_plot) + 4])
      axes[speed_axe].set_title(f'animal head speed')
      axes[speed_axe].set_ylabel("cm/s")
      
      #axes[speed_axe].set_xlabel("ms")
      
      if turnaround is not None:
            for turn_ind in range(np.shape(turnaround)[0]):
                  for axes_id in [0,1,2,3,4,5]:
                        x_= (turnaround[turn_ind][0] + turnaround[turn_ind][1]) / 2
                        axes[axes_id].axvline(x = x_, color = "red", linestyle = ":", linewidth = 2)
                        #axes[axes_id].axvspan(turnaround[turn_ind][0], turnaround[turn_ind][1], color = "red", alpha = 0.1)
                        
      if local_extended_theta_interval is not None:
            for turn_ind in range(np.shape(local_extended_theta_interval)[0]):
                  for axes_id in [0,1,2,3,4,5]:
                        axes[axes_id].axvspan(local_extended_theta_interval[turn_ind][0], local_extended_theta_interval[turn_ind][1], color = "C0", alpha = 0.1)
                        axes[axes_id].axvspan(local_extended_theta_interval[turn_ind][0], local_extended_theta_interval[turn_ind][1], color = "C0", alpha = 0.1)
                        axes[axes_id].axvspan(local_extended_theta_interval[turn_ind][0], local_extended_theta_interval[turn_ind][1], color = "C0", alpha = 0.1)
      
      if remote_theta_interval is not None:
            for turn_ind in range(np.shape(remote_theta_interval)[0]):
                  for axes_id in [0,1,2,3,4,5]:
                        axes[axes_id].axvspan(remote_theta_interval[turn_ind][0], remote_theta_interval[turn_ind][1], color = "C1", alpha = 0.1)
                        axes[axes_id].axvspan(remote_theta_interval[turn_ind][0], remote_theta_interval[turn_ind][1], color = "C1", alpha = 0.1)
                        axes[axes_id].axvspan(remote_theta_interval[turn_ind][0], remote_theta_interval[turn_ind][1], color = "C1", alpha = 0.1)
            
            
      """well locations"""
      # linear_map,node_location=get_linearization_map()
      # for axes_id in [0]:
      #       for junctions in linear_map[:,1]:
      #             axes[axes_id].axhline(junctions, linewidth = 0.5, linestyle = '-.',color = [0.2,0.2,0.2])
      #       for n in node_location.keys():
      #             axes[axes_id].axhline(node_location[n], linewidth = 0.5, linestyle = '-.',color = [0.2,0.2,0.2])
      
      for axes_id in [1+5,2+5,3+5]:
            axes[axes_id].spines[['right', 'top']].set_visible(False)
      
      xticklabels = ["" for x in xticks]
      # ind_first = np.argwhere((xticks-t0) > 0).ravel()[0]
      # xticklabels[ind_first] = int(np.round(xticks[ind_first] - t0,3) * 1000)
      # xticklabels[ind_first+1] = int(np.round(xticks[ind_first+1] - t0,3) * 1000)
      
      for axes_id in np.array([0,1,2,3,4]):
            axes[axes_id].spines[['bottom']].set_visible(False)
      # home arm
      axes[5].spines[['bottom']].set_visible(True)
      axes[5].spines[['top']].set_visible(False)
            
      for axes_id in np.array([1,2,3]) + 5:
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

from spyglass.shijiegu.changeOfMind_remote_location import region, center_region
region["center"] = center_region

def merge_center_platform(center_region,decode_input_subset):
    # Step 1: find all platform segments
    center_platform = []
    for segment in center_region:
        start_end = segment
    
        position = np.array(decode_input_subset.position)
        ind = np.argwhere(np.logical_and(position>start_end[0],
                                         position<start_end[1])).ravel()
        sel = decode_input_subset.isel(position = slice(ind[0], ind[-1]))
        center_platform.append(sel)

    # Step 2: interpolate
    center_platform_interp = []
    position_query = np.array(center_platform[0].position)
    for sel in center_platform:
        sel['position'] = np.linspace(position_query[0],position_query[-1],len(sel.position))
        center_platform_interp.append(sel.interp(position = position_query))
        #subset = sel.isel(position = slice(0,16))
        #subset["position"] = position_query
        #center_platform_interp.append(subset)
        
    # Step 3: sum
    center_platform_summed = center_platform_interp[0] + center_platform_interp[1] + center_platform_interp[2] + center_platform_interp[3] + center_platform_interp[4]

    return center_platform_summed, center_platform_interp

def decode_by_regions(decode, region):
    # return a decode xarray object, with only position bins in the region
    
    position = np.array(decode.position)
    positions = []
    decode_outs = []
    for name in [5,"center", 6, 7, 8, 9]: #home, center, arm1, arm2, arm3, arm4
        start_end = region[name]
        if len(start_end) > 2:
            # need to sum all areas
            center_platform_summed, _ = merge_center_platform(center_region,decode)
            positions.append(np.array(center_platform_summed.position))
            decode_outs.append(center_platform_summed)
        else:
            ind = np.argwhere(np.logical_and(position>start_end[0],
                                             position<start_end[1])).ravel()
            positions.append(position[ind])
        
            decode_out = decode.sel(position = position[ind])
            decode_outs.append(decode_out)

    axes_ind = [5, 4, 3, 2, 1, 0]
    return positions, decode_outs, axes_ind

def plot_decode_subpanel(axes_ind, decode_outs, axes, plot_decode = True, vmin = 0, vmax = 0.06):
    
      for ind in range(6):
            ax_ind = axes_ind[ind]
            decode_out = decode_outs[ind]
            if plot_decode:
                  decode_out.plot(
                              x='time', y='position', ax=axes[ax_ind],vmin = vmin, vmax = vmax,
                              rasterized=True, cmap='bone_r',infer_intervals = False,
                              add_colorbar=False, add_labels=False)
            axes[ax_ind].set_xticks([])
            axes[ax_ind].set_yticks([])
            axes[ax_ind].set_xlabel("")
            axes[ax_ind].set_ylabel("")

            axes[ax_ind].spines['top'].set_linewidth(0.5)
            axes[ax_ind].spines['right'].set_visible(False)
            # Keep the left and bottom borders visible
            axes[ax_ind].spines['left'].set_visible(False)
            axes[ax_ind].spines['bottom'].set_linewidth(0.5)
            
            position_bins = np.array(decode_out.position)
            #if ax_ind == 4:
            axes[ax_ind].set_ylim([position_bins[0]-3,position_bins[-1]+3])
            #else:
            #      axes[ax_ind].set_ylim([position_bins[0]-1,position_bins[-1] + 1])
            
def plot_behavior_only(plottimes,linear_position_subset,
                        head_speed_subset, head_angle_speed_subset,
                        savefolder=[], savename=[],
                        turnaround = None, likelihood = False,
                        t0 = None, #plot_len = 2,
                        rightward = None,
                        head_direction = None):

      decode_subpanel_lengths = np.array([43, 44, 44, 44, 16, 29])
      decode_subpanel_lengths = decode_subpanel_lengths / np.sum(decode_subpanel_lengths) * 1.5
      height_ratios = np.concatenate((decode_subpanel_lengths, np.array([0.7, 0.4])))
            
      fig, axes = plt.subplots(8, 1, figsize=(4, 6), sharex=True,
                               gridspec_kw={"height_ratios": height_ratios})
      plt.subplots_adjust(hspace=0.6)
      
      
      
      """plotting"""
      axes_list = [5,4,3,2,1,0]
      axes_ind = 0
      for name in [5,"center", 6, 7, 8, 9]: #home, center, arm1, arm2, arm3, arm4
            start_end = region[name]
            if name == "center":
                  start_end = start_end[0]

            axes[axes_list[axes_ind]].set_ylim([start_end[0], start_end[1]])
            axes[axes_list[axes_ind]].spines['left'].set_bounds(start_end[0], start_end[1])
            #axes[axes_ind].set_xlim(plottimes[0],plottimes[1])
            axes_ind += 1

      for ind in [0,1,2,3,4,5]:
            axes[ind].scatter(linear_position_subset.time,
                        np.array(linear_position_subset.linear_position),
                        s=1, color='magenta', zorder=10)
            axes[ind].set_xlim(plottimes[0],plottimes[1])
      
            axes[ind].set_yticks([])
            axes[ind].set_yticklabels([])
            
      
      if t0 is None:
            t0 = 0.5 * (plottimes[0] + plottimes[1])
      
      
      '''speed information'''
      speed_axe = 6
      head_speed_plot = np.array(head_speed_subset.to_array()).ravel()
      axes[speed_axe].plot(np.array(head_speed_subset.time),head_speed_plot,color = 'k')
      axes[speed_axe].spines[['right', 'top']].set_visible(False)
      axes[speed_axe].set_ylim([0, np.nanmax(head_speed_plot) + 35])
      axes[speed_axe].spines['left'].set_bounds(0, np.nanmax(head_speed_plot))
      axes[speed_axe].set_yticks([0,25,50])
      #axes[speed_axe].set_title(f'animal head speed')
      axes[speed_axe].set_ylabel("cm/s")

      
      """angular speed information"""
      speed_axe = 7
      head_angular_plot = np.abs(np.clip(np.array(head_angle_speed_subset.to_array()).ravel(), -10, 10))
      axes[speed_axe].plot(np.array(head_angle_speed_subset.time),head_angular_plot,color = 'k')
      axes[speed_axe].spines[['right', 'top']].set_visible(False)
      axes[speed_axe].set_ylim([0, 10])#([0, np.nanmax(head_speed_plot) + 4])
      #axes[speed_axe].set_title(f'animal head angular speed')
      axes[speed_axe].set_ylabel("rad/s")
      
      """head direction"""
      head_direction_axe = 6 #8
      head_direction_y = np.nanmax(head_speed_plot) + 30
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
      
      set1_t_region = get_nonnan_regions(set1_d)
      
      for r in set1_t_region:
            start, end = set1_t[r[0]], set1_t[r[-1]]
            axes[head_direction_axe].plot([start, end],[head_direction_y,head_direction_y],color = 'C0',
                                          linewidth = 5,
                                          label = "facing right")

      set2_t_region = get_nonnan_regions(set2_d)
      for r in set2_t_region:
            start, end = set2_t[r[0]], set2_t[r[-1]]
            axes[head_direction_axe].plot([start, end],[head_direction_y,head_direction_y],color = 'C1',
                                          linewidth = 5,
                                          label = "facing left")
      
      set3_t_region = get_nonnan_regions(set3_d)
      for r in set3_t_region:
            start, end = set3_t[r[0]], set3_t[r[-1]]
            axes[head_direction_axe].plot([start, end],[head_direction_y,head_direction_y],color = 'grey',
                                          linewidth = 5,
                                          label = "aligned with arm")
            
            
      #axes[head_direction_axe].set_ylim([-1,4])
      #axes[head_direction_axe].set_xlabel("s")
            
      
      
      
      if turnaround is not None:
            for turn_ind in range(np.shape(turnaround)[0]):
                  for axes_id in [0,1,2,3,4,5]:
                        axes[axes_id].axvspan(turnaround[turn_ind][0], turnaround[turn_ind][1], color = "red", alpha = 0.1)
                        axes[axes_id].axvspan(turnaround[turn_ind][0], turnaround[turn_ind][1], color = "red", alpha = 0.1)
                        axes[axes_id].axvspan(turnaround[turn_ind][0], turnaround[turn_ind][1], color = "red", alpha = 0.1)
            
      """well locations"""
      # linear_map,node_location=get_linearization_map()
      # for axes_id in [0]:
      #       for junctions in linear_map[:,1]:
      #             axes[axes_id].axhline(junctions, linewidth = 0.5, linestyle = '-.',color = [0.2,0.2,0.2])
      #       for n in node_location.keys():
      #             axes[axes_id].axhline(node_location[n], linewidth = 0.5, linestyle = '-.',color = [0.2,0.2,0.2])
      
      for axes_id in [1+5,2+5]:
            axes[axes_id].spines[['right', 'top']].set_visible(False)
      
      xticks = np.arange(plottimes[0], plottimes[1], 1)
      xticklabels = [np.round(x - t0,2)  for x in xticks]
      
      
      for axes_id in np.arange(6):
            axes[axes_id].spines[['bottom',"right"]].set_visible(False)
      
      for axes_id in np.arange(8):

            if xticks is not None:
                  axes[axes_id].set_xticks(xticks)
                  axes[axes_id].set_xticklabels(xticklabels)#fontsize = 7, rotation = 45)
      
      #x_axis2 = np.concatenate((np.arange(t0, plottimes[0],-0.2), np.arange(t0,plottimes[1],0.2)))
      #axes[3].set_xticks(x_axis2)
      #axes[3].set_xticklabels(np.round(x_axis2 - t0,2))
            
      # axes_id = 0
      # graph = TrackGraph() & {'track_graph_name': '4 arm lumped 2023'}
      # graph.plot_track_graph_as_1D(
      #       ax=axes[axes_id],
      #       axis='y',
      #       other_axis_start = plottimes[0]-0.1)
      #fig.subplots_adjust(wspace=2)
      if len(savefolder)>0:
            plt.savefig(os.path.join(savefolder,savename+'.pdf'),format="pdf",bbox_inches='tight',dpi=200,transparent=True)
        #plt.savefig(os.path.join(exampledir,'ripple_'+str(ripple_num)+'.png'),bbox_inches='tight',dpi=300)
      
      return fig, axes

def get_nonnan_regions(data):
    """
    Parses a list for start/end (inclusive/exclusive) indices
    of contiguous non-NaN segments.
    """
    regions = []
    start = None
    
    for i, val in enumerate(data):
        # Check if current value is NOT NaN
        if not np.isnan(val):
            if start is None:
                start = i
        else:
            # If we were tracking a segment, it ends here
            if start is not None:
                regions.append((start, i-1))
                start = None
                
    # Append final segment if list ended while tracking
    if start is not None:
        regions.append((start, len(data)-1))
        
    return regions
