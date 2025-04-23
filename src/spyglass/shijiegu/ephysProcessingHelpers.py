from spyglass.common import Electrode, IntervalList, TaskEpoch
from spyglass.spikesorting.v0 import SortGroup, SortInterval

def set_reference(nwb_copy_file_name, canula_1_tet_list, canula_1_ref):
    """set reference for each electrode in canula_1_tet_list
    by inserting into SortGroup

    Parameters
    ----------
    canula_1_tet_list : list
        list of electrode_group_name to set reference. 0 indexed.
    canula_1_ref : int
        electrode_id
    """

    for tetrode in canula_1_tet_list:
        key = {'nwb_file_name' : nwb_copy_file_name,
               'electrode_group_name' : tetrode,
               'bad_channel' : 'False'}

        # information for this tetrode
        electrode_group_list = (Electrode() & key).fetch('electrode_group_name').astype(int).tolist()
        sort_group_id = electrode_group_list[0]

        # insert into SortGroup: all channels of this electrode in one SortGroup
        SortGroup.insert1({'nwb_file_name' : nwb_copy_file_name,
                           'sort_group_id' : sort_group_id,
                           'sort_reference_electrode_id' : canula_1_ref}, skip_duplicates=True)

        # insert into SortGroupElectrode
        # information for all channels in this tetrode
        electrode_list = (Electrode() & key).fetch('electrode_id').tolist()

        for ndx in range(len(electrode_list)):
            electrode_group_name = electrode_group_list[ndx]
            electrode_id = electrode_list[ndx]
            SortGroup.SortGroupElectrode.insert1({'nwb_file_name' : nwb_copy_file_name,
                                            'sort_group_id' : sort_group_id,
                                            'electrode_group_name' : electrode_group_name,
                                            'electrode_id' : electrode_id}, skip_duplicates=True)
    return None


def insert_lick_artifact(nwb_copy_file_name, canula_1_tet_list, canula_1_ref, target):
    """
    this is a way to make a new sort group that includes several tetrodes that will be used for artifact detection
    we want to make one for each canula
    for example, for a 32 tetrode drive:
     tetrodes 1-32 > sort_group_id = 100
     tetrodes 33-64 > sort_group_id = 101
    use the values above for the sort_reference_electrode_id
    """


    electrode_list = []
    electrode_group_list = []
    for tetrode in canula_1_tet_list:
        key = {'nwb_file_name' : nwb_copy_file_name,
               'electrode_group_name' : tetrode,
               'bad_channel' : 'False'}
        electrode_list = electrode_list + (Electrode() & key).fetch('electrode_id').tolist()
        electrode_group_list = electrode_group_list + (Electrode() & key).fetch('electrode_group_name').astype(int).tolist()

    # SortGroup first, its name is target
    SortGroup.insert1({'nwb_file_name' : nwb_copy_file_name,
                        'sort_group_id' : target,
                        'sort_reference_electrode_id' : canula_1_ref}, skip_duplicates=True)

    # subtable now, all members are good electrode in one cannula
    for ndx in range(len(electrode_list)):
        electrode_group_name = electrode_group_list[ndx]
        electrode_id = electrode_list[ndx]
        SortGroup.SortGroupElectrode.insert1({'nwb_file_name' : nwb_copy_file_name,
                                            'sort_group_id' : target,
                                            'electrode_group_name' : electrode_group_name,
                                            'electrode_id' : electrode_id}, skip_duplicates=True)


def insert_SortInterval(nwb_copy_file_name):
    session_interval = (TaskEpoch() & {"nwb_file_name": nwb_copy_file_name}).fetch("interval_list_name")

    # get times
    for ses_intvl in session_interval:
        intvl=(IntervalList & {'nwb_file_name' : nwb_copy_file_name,
                                'interval_list_name' : ses_intvl}).fetch1('valid_times')

        SortInterval.insert1({'nwb_file_name' : nwb_copy_file_name,
                        'sort_interval_name' : ses_intvl,
                        'sort_interval' : intvl}, skip_duplicates=True)
    return session_interval
