import numpy as np
from spyglass.shijiegu.changeOfMind_triggered import form_null_model_full

def get_prepost_normalized_histogram(animal, list_of_days_animals,
                                     loaded_data_all_days, SD_threshold = 3):
    # first estimate the standard decoding error for each position bin, then we can normalize the decoding error by the distance to the end of the track.
    # diff = [] 
    # for d in list_of_days_animals[animal]:
    #     for ind in range(len(loaded_data_all_days[d]['pos_post'])):
    #         pos = loaded_data_all_days[d]['pos_post'][ind]
    #         decode = loaded_data_all_days[d]['decode_post'][ind]
    #         diff.append(decode - pos)
    # diff = np.concatenate(diff)
    # std = np.nanstd(diff)
    print("working on animal ", animal, " with SD threshold ", SD_threshold)
            
    counts_forward = []
    counts_behind = []

    histogram_bins = np.linspace(0,1,6)
    histogram_bins[-1] = 1.1

    

    for d in list_of_days_animals[animal]:
        print(d)
        triggered_positions = loaded_data_all_days[d]['pos_post_control']
        triggered_decodes = loaded_data_all_days[d]['decode_post_control']
        gaussian_process, _, _, gaussian_process_CI = form_null_model_full(triggered_positions, triggered_decodes)
    
        for ind in range(len(loaded_data_all_days[d]['pos_post'])):
            pos = loaded_data_all_days[d]['pos_post'][ind]
            decode = loaded_data_all_days[d]['decode_post'][ind]
            
            pos_query = pos.reshape(-1, 1)
            # replace NaN values in pos_query to 0
            if np.isnan(pos_query).any():
                pos_query = np.nan_to_num(pos_query)
            decode_mean_null = gaussian_process.predict(pos_query, return_std=False)[0]
            decode_CI_null = gaussian_process_CI.predict(pos_query, return_std=False)[0]
            decode_null_u = decode_mean_null + SD_threshold * decode_CI_null
            decode_null_l = decode_mean_null - SD_threshold * decode_CI_null
        
            # in front of the animal
            forward = decode > decode_null_u
            pos_to_end = np.clip(80-pos, a_min=0, a_max=None)[forward]
            decode_to_end = np.clip(decode - pos, a_min=0, a_max=None)[forward]
            normalized_decode = decode_to_end / pos_to_end
            normalized_decode = np.clip(normalized_decode, a_min=0, a_max=1)
            weights = np.ones_like(normalized_decode)/len(normalized_decode) if normalized_decode.size else None
            count, bins = np.histogram(normalized_decode, bins=histogram_bins, weights=weights)
            counts_forward.append(count)
    
            # back of the animal
            behind = decode < decode_null_l
            pos_to_end = pos[behind]
            decode_to_end = np.clip(pos-decode, a_min=0, a_max=None)[behind]
            normalized_decode = decode_to_end / pos_to_end
            normalized_decode = np.clip(normalized_decode, a_min=0, a_max=1)
            weights = np.ones_like(normalized_decode)/len(normalized_decode) if normalized_decode.size else None
            count, bins = np.histogram(normalized_decode, bins=histogram_bins, weights=weights)
            counts_behind.append(count)

    counts_forward = np.array(counts_forward)
    counts_behind = np.array(counts_behind)

    mean_forward = np.nanmean(counts_forward, axis = 0 )
    sd_forward = np.nanstd(counts_forward, axis = 0 ) / np.sqrt(counts_forward.shape[0])

    mean_behind = np.nanmean(counts_behind, axis = 0 )
    sd_behind = np.nanstd(counts_behind, axis = 0 ) / np.sqrt(counts_behind.shape[0])
    
    return bins, counts_forward, counts_behind, mean_forward, sd_forward, mean_behind, sd_behind

def plot_prepost_normalized_histogram(animal, bins, mean_forward, sd_forward, mean_behind, sd_behind, ax):
    

    # Calculate bin centers
    bins[-1] = 1.0
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    
    ax.plot(bin_centers, mean_forward, linewidth=2, label='animal to well', alpha=0.5, color = 'tab:blue')
    ax.plot(bin_centers, mean_behind, linewidth=2, label='arm start to animal', alpha=0.5, color = 'tab:orange')
    ax.scatter(bin_centers, mean_forward, color='tab:blue', s=50, alpha=0.7)
    ax.scatter(bin_centers, mean_behind, color='tab:orange', s=50, alpha=0.7)
    
    # Add shaded regions for standard deviation
    ax.fill_between(bin_centers, mean_forward - sd_forward, mean_forward + sd_forward, alpha=0.2)
    ax.fill_between(bin_centers, mean_behind - sd_behind, mean_behind + sd_behind, alpha=0.2)
    
    # set xlim from 0 to 1
    ax.set_xlim(0, 1)
    bin_centers_include_zero_one = np.concatenate(([0], bin_centers, [1]))
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f'{bc:.2f}' for bc in bin_centers], fontsize=12)
    ax.set_xlabel('normalized dist \nto end of arm', fontsize = 15)
    if animal == "molly":
        ax.set_ylabel('proportion of \ntime bins', fontsize = 15)
    ax.legend(loc='upper left', fontsize=12)
    # set yticklabels fontsize to 12
    ax.set_yticklabels(np.round(ax.get_yticks(),1), fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_title(f'Rat {animal[0].upper()}', fontsize = 15)