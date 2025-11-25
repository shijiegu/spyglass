import numpy as np
import pandas as pd
import pickle
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import ranksums
import seaborn as sns


from spyglass.shijiegu.changeOfMind_remote import find_remote_theta_animal_new


output_folder = '/stelmo/shijie/change_of_mind_analysis/'
def return_save_name(animal, encoding_set, classifier_param_name, d1, d2, proportion = 0.1, use_1d = 1):
    save_name = f'{animal.lower()}_figure7e_{encoding_set}_{classifier_param_name}_{d1}_{d2}_p{proportion}_use1d{use_1d}'
    return save_name

def save_remote_animal_figure7c(animal, list_of_days, encoding_set, classifier_param_name,params_pre,
                                posterior_arm_pre, posterior_arm_control_pre, arm_position,
                                posterior_arm_post, posterior_arm_control_post,
                                ):
    
    d1= list_of_days[0]
    d2= list_of_days[-1]
    proportion = params_pre["proportion"]
    use_1d = int(params_pre["use_1d"])
    save_name = return_save_name(animal, encoding_set, classifier_param_name, d1, d2, proportion, use_1d)
    file_path = output_folder + save_name + '.pkl'
    
    data = {}
    (data["posterior_arm_pre"], data["posterior_arm_control_pre"],
     data["arm_position"], data["posterior_arm_post"],
     data["posterior_arm_control_post"]) = (
                posterior_arm_pre, posterior_arm_control_pre,
                arm_position,
                posterior_arm_post, posterior_arm_control_post)
        
    # Open the file in binary write mode and dump the data
    with open(file_path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data successfully pickled and saved to {file_path}")
    return 1

def load_remote_animal_figure7c(animal, list_of_days, encoding_set, classifier_param_name,
                                proportion = 0.1, use_1d = 1):
    d1, d2 = list_of_days[0], list_of_days[-1]
    save_name = return_save_name(animal, encoding_set, classifier_param_name, d1, d2, proportion, use_1d)
    file_path = output_folder + save_name + '.pkl'
    
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
        print(f"Successfully loaded data from '{file_path}':")
    return loaded_data