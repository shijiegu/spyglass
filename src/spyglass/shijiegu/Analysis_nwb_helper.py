from pynwb import NWBFile, NWBHDF5IO
from datetime import datetime, timezone
from hdmf.common import DynamicTable
import pandas as pd
from spyglass.common.common_nwbfile import AnalysisNwbfile

def write_data_to_analyis_nwb(data, scratch_name, nwb_file_name_copy, table_name):
    """
    table_name = ChangeofMind().full_table_name
    """
    key = {}
    key["nwb_file_name"] = nwb_file_name_copy
    
    nwb_analysis_file = AnalysisNwbfile() # Initialize the AnalysisNwbfile object
    key["analysis_file_name"] = AnalysisNwbfile().create(nwb_file_name_copy)
    
    nwb_scratch_object = DynamicTable.from_dataframe(name=scratch_name, df = pd.DataFrame(data))

    nwb_object_id = nwb_analysis_file.add_nwb_object(
        analysis_file_name = key["analysis_file_name"],
        nwb_object= nwb_scratch_object,
        table_name = table_name,
        )
    #key["nwb_object_id"] = nwb_object_id
    
    nwb_analysis_file.add(
        nwb_file_name=key["nwb_file_name"],
            analysis_file_name=key["analysis_file_name"],
        )
        
    AnalysisNwbfile().log(key, table=table_name)
    return key

def write_df_to_analyis_nwb(df, scratch_name, nwb_file_name_copy, table_name):
    """
    table_name = ChangeofMind().full_table_name
    """
    key = {}
    key["nwb_file_name"] = nwb_file_name_copy
    
    nwb_analysis_file = AnalysisNwbfile() # Initialize the AnalysisNwbfile object
    key["analysis_file_name"] = AnalysisNwbfile().create(nwb_file_name_copy)
    
    nwb_analysis_file.add_nwb_object(key["analysis_file_name"], df, scratch_name)

    # REGISTER 
    nwb_analysis_file.add(
        nwb_file_name=key["nwb_file_name"],
            analysis_file_name=key["analysis_file_name"],
        )
    
    # nwb_scratch_object = DynamicTable.from_dataframe(name=scratch_name, df = pd.DataFrame(data))

    # nwb_object_id = nwb_analysis_file.add_nwb_object(
    #     analysis_file_name = key["analysis_file_name"],
    #     nwb_object= nwb_scratch_object,
    #     table_name = table_name,
    #     )
    #key["nwb_object_id"] = nwb_object_id
    
    
        
    AnalysisNwbfile().log(key, table=table_name)
    return key
    

"""
with NWBHDF5IO("example.nwb", "r") as io:
    nwbfile_in = io.read()
    # get_scratch automatically converts DynamicTable to a dataframe
    df_in = nwbfile_in.get_scratch("my_table")
    print(df_in)
"""
