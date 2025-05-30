import json
import os
import shutil
import time
import uuid
import warnings
from pathlib import Path
from typing import List

import datajoint as dj
import numpy as np
import spikeinterface as si
from packaging import version

if version.parse(si.__version__) < version.parse("0.99.1"):
    raise ImportError(
        "SpikeInterface version must updated. "
        + "Please run `pip install spikeinterface==0.99.1` to update."
    )
import spikeinterface.preprocessing as sip
import spikeinterface.qualitymetrics as sq

from spyglass.common import BrainRegion, Electrode
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.settings import waveforms_dir
from spyglass.spikesorting.v0.merged_sorting_extractor import (
    MergedSortingExtractor,
)
from spyglass.spikesorting.v0.spikesorting_recording import (
    SortInterval,
    SpikeSortingRecording,
)
from spyglass.utils import SpyglassMixin, logger

from .spikesorting_recording import SortGroup
from .spikesorting_sorting import SpikeSorting

schema = dj.schema("spikesorting_curation")

valid_labels = ["reject", "noise", "artifact", "mua", "accept"]


def apply_merge_groups_to_sorting(
    sorting: si.BaseSorting, merge_groups: List[List[int]]
):
    """Apply merge groups to a sorting extractor."""
    # return a new sorting where the units are merged according to merge_groups
    # merge_groups is a list of lists of unit_ids.
    # for example: merge_groups = [[1, 2], [5, 8, 4]]]

    return MergedSortingExtractor(
        parent_sorting=sorting, merge_groups=merge_groups
    )


@schema
class Curation(SpyglassMixin, dj.Manual):
    definition = """
    # Stores each spike sorting; similar to IntervalList
    curation_id: int # a number corresponding to the index of this curation
    -> SpikeSorting
    ---
    parent_curation_id=-1: int
    curation_labels: blob # a dictionary of labels for the units
    merge_groups: blob # a list of merge groups for the units
    quality_metrics: blob # a list of quality metrics for the units (if available)
    description='': varchar(1000) #optional description for this curated sort
    time_of_creation: int   # in Unix time, to the nearest second
    """

    _nwb_table = AnalysisNwbfile

    @staticmethod
    def insert_curation(
        sorting_key: dict,
        parent_curation_id: int = -1,
        labels=None,
        merge_groups=None,
        metrics=None,
        description="",
    ):
        """Given a SpikeSorting key and the parent_sorting_id (and optional
        arguments) insert an entry into Curation.


        Parameters
        ----------
        sorting_key : dict
            The key for the original SpikeSorting
        parent_curation_id : int, optional
            The id of the parent sorting
        labels : dict or None, optional
        merge_groups : dict or None, optional
        metrics : dict or None, optional
            Computed metrics for sorting
        description : str, optional
            text description of this sort

        Returns
        -------
        curation_key : dict

        """
        if parent_curation_id == -1:
            # check to see if this sorting with a parent of -1 has already been
            # inserted and if so, warn the user
            inserted_curation = (Curation & sorting_key).fetch("KEY")
            if len(inserted_curation) > 0:
                Warning(
                    "Sorting has already been inserted, returning key to previously"
                    "inserted curation"
                )
                return inserted_curation[0]

        if labels is None:
            labels = {}
        if merge_groups is None:
            merge_groups = []
        if metrics is None:
            metrics = {}

        # generate a unique number for this curation
        id = (Curation & sorting_key).fetch("curation_id")
        curation_id = parent_curation_id + 1
        #if len(id) > 0:
        #    curation_id = max(id) + 1
        #else:
        #    curation_id = 0

        # convert unit_ids in labels to integers for labels from sortingview.
        new_labels = {int(unit_id): labels[unit_id] for unit_id in labels}

        sorting_key.update(
            {
                "curation_id": curation_id,
                "parent_curation_id": parent_curation_id,
                "description": description,
                "curation_labels": new_labels,
                "merge_groups": merge_groups,
                "quality_metrics": metrics,
                "time_of_creation": int(time.time()),
            }
        )

        # mike: added skip duplicates
        Curation.insert1(sorting_key, skip_duplicates=True)

        # get the primary key for this curation
        curation_key = {
            item: sorting_key[item] for item in Curation.primary_key
        }

        return curation_key

    @staticmethod
    def get_recording(key: dict):
        """Returns the recording extractor for the recording related to this curation

        Parameters
        ----------
        key : dict
            SpikeSortingRecording key

        Returns
        -------
        recording_extractor : spike interface recording extractor

        """
        recording_path = (SpikeSortingRecording & key).fetch1("recording_path")
        return si.load_extractor(recording_path)

    @staticmethod
    def get_curated_sorting(key: dict):
        """Returns the sorting extractor related to this curation,
        with merges applied.

        Parameters
        ----------
        key : dict
            Curation key

        Returns
        -------
        sorting_extractor: spike interface sorting extractor

        """
        sorting_path = (SpikeSorting & key).fetch1("sorting_path")
        sorting = si.load_extractor(sorting_path)
        merge_groups = (Curation & key).fetch1("merge_groups")
        # TODO: write code to get merged sorting extractor
        if len(merge_groups) != 0:
            return MergedSortingExtractor(
                parent_sorting=sorting, merge_groups=merge_groups
            )
        else:
            return sorting

    @staticmethod
    def save_sorting_nwb(
        key,
        sorting,
        timestamps,
        sort_interval_list_name,
        sort_interval,
        labels=None,
        metrics=None,
        unit_ids=None,
    ):
        """Store a sorting in a new AnalysisNwbfile

        Parameters
        ----------
        key : dict
            key to SpikeSorting table
        sorting : si.Sorting
            sorting
        timestamps : array_like
            Time stamps of the sorted recoridng;
            used to convert the spike timings from index to real time
        sort_interval_list_name : str
            name of sort interval
        sort_interval : list
            interval for start and end of sort
        labels : dict, optional
            curation labels, by default None
        metrics : dict, optional
            quality metrics, by default None
        unit_ids : list, optional
            IDs of units whose spiketrains to save, by default None

        Returns
        -------
        analysis_file_name : str
        units_object_id : str

        """
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])

        sort_interval_valid_times = (
            IntervalList & {"interval_list_name": sort_interval_list_name}
        ).fetch1("valid_times")

        units = dict()
        units_valid_times = dict()
        units_sort_interval = dict()

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()

        for unit_id in unit_ids:
            spike_times_in_samples = sorting.get_unit_spike_train(
                unit_id=unit_id
            )
            units[unit_id] = timestamps[spike_times_in_samples]
            units_valid_times[unit_id] = sort_interval_valid_times
            units_sort_interval[unit_id] = [sort_interval]

        object_ids = AnalysisNwbfile().add_units(
            analysis_file_name,
            units,
            units_valid_times,
            units_sort_interval,
            metrics=metrics,
            labels=labels,
        )
        AnalysisNwbfile().add(key["nwb_file_name"], analysis_file_name)

        if object_ids == "":
            logger.warning(
                "Sorting contains no units."
                "Created an empty analysis nwb file anyway."
            )
            units_object_id = ""
        else:
            units_object_id = object_ids[0]

        return analysis_file_name, units_object_id


@schema
class WaveformParameters(SpyglassMixin, dj.Manual):
    """Parameters for extracting waveforms from a sorting extractor

    Parameters
    ----------
    waveform_params_name : str
        Name of the waveform extraction parameters
    waveform_params : dict
        Dictionary of waveform extraction parameters, including...
        ms_before : float
            Number of milliseconds before the spike time to include
        ms_after : float
            Number of milliseconds after the spike time to include
        max_spikes_per_unit : int
            Maximum number of spikes to extract for each unit
        n_jobs : int
            Number of parallel jobs to use for extraction
        total_memory : str
            Total memory to use for extraction (e.g. "5G")
        whiten : bool
            Whether to whiten the waveforms or not
    """

    definition = """
    waveform_params_name: varchar(80) # name of waveform extraction parameters
    ---
    waveform_params: blob # a dict of waveform extraction parameters
    """

    def insert_default(self):
        """Inserts default waveform parameters"""
        default = {
            "ms_before": 0.5,
            "ms_after": 0.5,
            "max_spikes_per_unit": 5000,
            "n_jobs": 5,
            "total_memory": "5G",
        }
        self.insert(
            (
                ["default_whitened", dict(**default, whiten=True)],
                ["default_not_whitened", dict(**default, whiten=False)],
            ),
            skip_duplicates=True,
        )


@schema
class WaveformSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> Curation
    -> WaveformParameters
    ---
    """


@schema
class Waveforms(SpyglassMixin, dj.Computed):
    _use_transaction, _allow_insert = False, True

    definition = """
    -> WaveformSelection
    ---
    waveform_extractor_path: varchar(400)
    -> AnalysisNwbfile
    waveforms_object_id: varchar(40)   # Object ID for the waveforms in NWB file
    """

    def make(self, key):
        """Populate Waveforms table with waveform extraction results

        1. Fetches ...
            - Recording and sorting from Curation table
            - Parameters from WaveformParameters table
        2. Uses spikeinterface to extract waveforms
        3. Generates an analysis NWB file with the waveforms
        4. Inserts the key into Waveforms table
        """
        key["analysis_file_name"] = AnalysisNwbfile().create(  # logged
            key["nwb_file_name"]
        )
        recording = Curation.get_recording(key)
        if recording.get_num_segments() > 1:
            recording = si.concatenate_recordings([recording])

        sorting = Curation.get_curated_sorting(key)

        logger.info("Extracting waveforms...")
        waveform_params = (WaveformParameters & key).fetch1("waveform_params")
        if "whiten" in waveform_params:
            if waveform_params.pop("whiten"):
                recording = sip.whiten(recording, dtype="float32")

        waveform_extractor_name = self._get_waveform_extractor_name(key)
        key["waveform_extractor_path"] = str(
            Path(waveforms_dir) / Path(waveform_extractor_name)
        )
        if os.path.exists(key["waveform_extractor_path"]):
            shutil.rmtree(key["waveform_extractor_path"])
        waveforms = si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=key["waveform_extractor_path"],
            **waveform_params,
        )

        object_id = AnalysisNwbfile().add_units_waveforms(
            key["analysis_file_name"], waveform_extractor=waveforms
        )
        key["waveforms_object_id"] = object_id
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])

        AnalysisNwbfile().log(key, table=self.full_table_name)
        self.insert1(key)

    def load_waveforms(self, key: dict):
        """Returns a spikeinterface waveform extractor specified by key

        Parameters
        ----------
        key : dict
            Could be an entry in Waveforms, or some other key that uniquely defines
            an entry in Waveforms

        Returns
        -------
        we : spikeinterface.WaveformExtractor
        """
        we_path = (self & key).fetch1("waveform_extractor_path")
        we = si.WaveformExtractor.load_from_folder(we_path)
        return we

    def fetch_nwb(self, key):
        """Fetches the NWB file path for the waveforms. NOT YET IMPLEMENTED."""
        # TODO: implement fetching waveforms from NWB
        return NotImplementedError

    def _get_waveform_extractor_name(self, key):
        waveform_params_name = (WaveformParameters & key).fetch1(
            "waveform_params_name"
        )

        return (
            f'{key["nwb_file_name"]}_{str(uuid.uuid4())[0:8]}_'
            f'{key["curation_id"]}_{waveform_params_name}_waveforms'
        )


@schema
class MetricParameters(SpyglassMixin, dj.Manual):
    """Parameters for computing quality metrics of sorted units

    See MetricParameters.get_available_metrics() for a list of available metrics
    """

    definition = """
    # Parameters for computing quality metrics of sorted units
    metric_params_name: varchar(64)
    ---
    metric_params: blob
    """

    # NOTE: See #630, #664. Excessive key length.

    metric_default_params = {
        "snr": {
            "peak_sign": "neg",
            "random_chunk_kwargs_dict": {
                "num_chunks_per_segment": 20,
                "chunk_size": 10000,
                "seed": 0,
            },
        },
        "isi_violation": {"isi_threshold_ms": 1.5, "min_isi_ms": 0.0},
        "nn_isolation": {
            "max_spikes": 1000,
            "min_spikes": 10,
            "n_neighbors": 5,
            "n_components": 7,
            "radius_um": 100,
            "seed": 0,
        },
        "nn_noise_overlap": {
            "max_spikes": 1000,
            "min_spikes": 10,
            "n_neighbors": 5,
            "n_components": 7,
            "radius_um": 100,
            "seed": 0,
        },
        "peak_channel": {"peak_sign": "neg"},
        "num_spikes": {},
    }
    # Example of peak_offset parameters 'peak_offset': {'peak_sign': 'neg'}
    available_metrics = [
        "snr",
        "isi_violation",
        "nn_isolation",
        "nn_noise_overlap",
        "peak_offset",
        "peak_channel",
        "num_spikes",
    ]

    def get_metric_default_params(self, metric: str):
        "Returns default params for the given metric"
        return self.metric_default_params(metric)

    def insert_default(self) -> None:
        """Inserts default metric parameters"""
        self.insert1(
            ["franklab_default3", self.metric_default_params],
            skip_duplicates=True,
        )

    def get_available_metrics(self):
        """Log available metrics and their descriptions"""
        for metric in _metric_name_to_func:
            if metric in self.available_metrics:
                metric_doc = _metric_name_to_func[metric].__doc__.split("\n")[0]
                metric_string = ("{metric_name} : {metric_doc}").format(
                    metric_name=metric, metric_doc=metric_doc
                )
                logger.info(metric_string + "\n")

    # TODO
    def _validate_metrics_list(self, key):
        """Checks whether a row to be inserted contains only available metrics"""
        # get available metrics list
        # get metric list from key
        # compare
        return NotImplementedError


@schema
class MetricSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> Waveforms
    -> MetricParameters
    """

    def insert1(self, key, **kwargs):
        """Overriding insert1 to add warnings for peak_offset and peak_channel"""
        waveform_params = (WaveformParameters & key).fetch1("waveform_params")
        metric_params = (MetricParameters & key).fetch1("metric_params")
        if "peak_offset" in metric_params:
            if waveform_params["whiten"]:
                warnings.warn(
                    "Calculating 'peak_offset' metric on "
                    "whitened waveforms may result in slight "
                    "discrepancies"
                )
        if "peak_channel" in metric_params:
            if waveform_params["whiten"]:
                Warning(
                    "Calculating 'peak_channel' metric on "
                    "whitened waveforms may result in slight "
                    "discrepancies"
                )
        super().insert1(key, **kwargs)


@schema
class QualityMetrics(SpyglassMixin, dj.Computed):
    _use_transaction, _allow_insert = False, True

    definition = """
    -> MetricSelection
    ---
    quality_metrics_path: varchar(500)
    -> AnalysisNwbfile
    object_id: varchar(40) # Object ID for the metrics in NWB file
    """

    def make(self, key):
        """Populate QualityMetrics table with quality metric results.

        1. Fetches ...
            - Waveform extractor from Waveforms table
            - Parameters from MetricParameters table
        2. Computes metrics, including SNR, ISI violation, NN isolation,
            NN noise overlap, peak offset, peak channel, and number of spikes.
        3. Generates an analysis NWB file with the metrics.
        4. Inserts the key into QualityMetrics table
        """
        analysis_file_name = AnalysisNwbfile().create(  # logged
            key["nwb_file_name"]
        )
        waveform_extractor = Waveforms().load_waveforms(key)
        key["analysis_file_name"] = (
            analysis_file_name  # add to key here to prevent fetch errors
        )
        qm = {}
        params = (MetricParameters & key).fetch1("metric_params")
        for metric_name, metric_params in params.items():
            metric = self._compute_metric(
                waveform_extractor, metric_name, **metric_params
            )
            qm[metric_name] = metric
        qm_name = self._get_quality_metrics_name(key)
        key["quality_metrics_path"] = str(
            Path(waveforms_dir) / Path(qm_name + ".json")
        )
        # save metrics dict as json
        logger.info(f"Computed all metrics: {qm}")
        self._dump_to_json(qm, key["quality_metrics_path"])

        key["object_id"] = AnalysisNwbfile().add_units_metrics(
            key["analysis_file_name"], metrics=qm
        )
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])
        AnalysisNwbfile().log(key, table=self.full_table_name)

        self.insert1(key)

    def _get_quality_metrics_name(self, key):
        wf_name = Waveforms()._get_waveform_extractor_name(key)
        qm_name = wf_name + "_qm"
        return qm_name

    def _compute_metric(self, waveform_extractor, metric_name, **metric_params):
        metric_func = _metric_name_to_func[metric_name]

        peak_sign_metrics = ["snr", "peak_offset", "peak_channel"]
        if metric_name == "isi_violation":
            return metric_func(waveform_extractor, **metric_params)
        elif metric_name in peak_sign_metrics:
            if "peak_sign" not in metric_params:
                raise Exception(
                    f"{peak_sign_metrics} metrics require peak_sign",
                    "to be defined in the metric parameters",
                )
            return metric_func(
                waveform_extractor,
                peak_sign=metric_params.pop("peak_sign"),
                **metric_params,
            )

        metric = {}
        num_spikes = sq.compute_num_spikes(waveform_extractor)

        is_nn_iso = metric_name == "nn_isolation"
        is_nn_overlap = metric_name == "nn_noise_overlap"
        min_spikes = metric_params.get("min_spikes", 10)

        for unit_id in waveform_extractor.sorting.get_unit_ids():
            # checks to avoid bug in spikeinterface 0.98.2
            if num_spikes[unit_id] < min_spikes and (
                is_nn_iso or is_nn_overlap
            ):
                if is_nn_iso:
                    metric[str(unit_id)] = (np.nan, np.nan)
                elif is_nn_overlap:
                    metric[str(unit_id)] = np.nan

            else:
                metric[str(unit_id)] = metric_func(
                    waveform_extractor,
                    this_unit_id=int(unit_id),
                    **metric_params,
                )
            # nn_isolation returns tuple with isolation and unit number.
            # We only want isolation.
            if is_nn_iso:
                metric[str(unit_id)] = metric[str(unit_id)][0]
        return metric

    def _dump_to_json(self, qm_dict, save_path):
        new_qm = {}
        for key, value in qm_dict.items():
            m = {}
            for unit_id, metric_val in value.items():
                m[str(unit_id)] = np.float64(metric_val)
            new_qm[str(key)] = m
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(new_qm, f, ensure_ascii=False, indent=4)


def _compute_isi_violation_fractions(waveform_extractor, **metric_params):
    """Computes the per unit fraction of interspike interval violations to total spikes."""
    isi_threshold_ms = metric_params["isi_threshold_ms"]
    min_isi_ms = metric_params["min_isi_ms"]

    # Extract the total number of spikes that violated the isi_threshold for each unit
    isi_violation_counts = sq.compute_isi_violations(
        waveform_extractor,
        isi_threshold_ms=isi_threshold_ms,
        min_isi_ms=min_isi_ms,
    ).isi_violations_count

    # Extract the total number of spikes from each unit. The number of ISIs is one less than this
    num_spikes = sq.compute_num_spikes(waveform_extractor)

    # Calculate the fraction of ISIs that are violations
    isi_viol_frac_metric = {
        str(unit_id): isi_violation_counts[unit_id] / (num_spikes[unit_id] - 1)
        for unit_id in waveform_extractor.sorting.get_unit_ids()
    }
    return isi_viol_frac_metric


def _get_peak_offset(
    waveform_extractor: si.WaveformExtractor, peak_sign: str, **metric_params
):
    """Computes the shift of the waveform peak from center of window."""
    if "peak_sign" in metric_params:
        del metric_params["peak_sign"]
    peak_offset_inds = si.core.get_template_extremum_channel_peak_shift(
        waveform_extractor=waveform_extractor,
        peak_sign=peak_sign,
        **metric_params,
    )
    peak_offset = {key: int(abs(val)) for key, val in peak_offset_inds.items()}
    return peak_offset


def _get_peak_channel(
    waveform_extractor: si.WaveformExtractor, peak_sign: str, **metric_params
):
    """Computes the electrode_id of the channel with the extremum peak for each unit."""
    if "peak_sign" in metric_params:
        del metric_params["peak_sign"]
    peak_channel_dict = si.core.get_template_extremum_channel(
        waveform_extractor=waveform_extractor,
        peak_sign=peak_sign,
        **metric_params,
    )
    peak_channel = {key: int(val) for key, val in peak_channel_dict.items()}
    return peak_channel


def _get_num_spikes(
    waveform_extractor: si.WaveformExtractor, this_unit_id: int
):
    """Computes the number of spikes for each unit."""
    all_spikes = sq.compute_num_spikes(waveform_extractor)
    cluster_spikes = all_spikes[this_unit_id]
    return cluster_spikes


_metric_name_to_func = {
    "snr": sq.compute_snrs,
    "isi_violation": _compute_isi_violation_fractions,
    "nn_isolation": sq.nearest_neighbors_isolation,
    "nn_noise_overlap": sq.nearest_neighbors_noise_overlap,
    "peak_offset": _get_peak_offset,
    "peak_channel": _get_peak_channel,
    "num_spikes": _get_num_spikes,
}


@schema
class AutomaticCurationParameters(SpyglassMixin, dj.Manual):
    """Parameters for automatic curation of spike sorting

    Parameters
    ----------
    auto_curation_params_name : str
        Name of the automatic curation parameters
    merge_params : dict, optional
        Dictionary of parameters for merging units. May include nn_noise_overlap
        List[comparison operator: str, threshold: float, labels: List[str]]
    label_params : dict, optional
        Dictionary of parameters for labeling units
    """

    definition = """
    auto_curation_params_name: varchar(36)   # name of this parameter set
    ---
    merge_params: blob   # dictionary of params to merge units
    label_params: blob   # dictionary params to label units
    """

    # NOTE: No existing entries impacted by this change

    def insert1(self, key, **kwargs):
        """Overriding insert1 to validats label_params and merge_params"""
        # validate the labels and then insert
        # TODO: add validation for merge_params
        for metric in key["label_params"]:
            if metric not in _metric_name_to_func:
                raise Exception(f"{metric} not in list of available metrics")
            comparison_list = key["label_params"][metric]
            if comparison_list[0] not in _comparison_to_function:
                raise Exception(
                    f'{metric}: "{comparison_list[0]}" '
                    f"not in list of available comparisons"
                )
            if not isinstance(comparison_list[1], (int, float)):
                raise Exception(
                    f"{metric}: {comparison_list[1]} is of type "
                    f"{type(comparison_list[1])} and not a number"
                )
            for label in comparison_list[2]:
                if label not in valid_labels:
                    raise Exception(
                        f'{metric}: "{label}" '
                        f"not in list of valid labels: {valid_labels}"
                    )
        super().insert1(key, **kwargs)

    def insert_default(self):
        """Inserts default automatic curation parameters"""
        # label_params parsing: Each key is the name of a metric,
        # the contents are a three value list with the comparison, a value,
        # and a list of labels to apply if the comparison is true
        default_params = {
            "auto_curation_params_name": "default",
            "merge_params": {},
            "label_params": {
                "nn_noise_overlap": [">", 0.1, ["noise", "reject"]]
            },
        }
        self.insert1(default_params, skip_duplicates=True)

        # Second default parameter set for not applying any labels,
        # or merges, but adding metrics
        no_label_params = {
            "auto_curation_params_name": "none",
            "merge_params": {},
            "label_params": {},
        }
        self.insert1(no_label_params, skip_duplicates=True)


@schema
class AutomaticCurationSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> QualityMetrics
    -> AutomaticCurationParameters
    """


_comparison_to_function = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "==": np.equal,
}


@schema
class AutomaticCuration(SpyglassMixin, dj.Computed):
    definition = """
    -> AutomaticCurationSelection
    ---
    auto_curation_key: blob # the key to the curation inserted by make
    """

    def make(self, key):
        """Populate AutomaticCuration table with automatic curation results.

        1. Fetches ...
            - Quality metrics from QualityMetrics table
            - Parameters from AutomaticCurationParameters table
            - Parent curation/sorting from Curation table
        2. Curates the sorting based on provided merge and label parameters
        3. Inserts IDs into  AutomaticCuration and Curation tables
        """
        metrics_path = (QualityMetrics & key).fetch1("quality_metrics_path")
        with open(metrics_path) as f:
            quality_metrics = json.load(f)

        # get the curation information and the curated sorting
        parent_curation = (Curation & key).fetch(as_dict=True)[0]
        parent_merge_groups = parent_curation["merge_groups"]
        parent_labels = parent_curation["curation_labels"]
        parent_curation_id = parent_curation["curation_id"]
        parent_sorting = Curation.get_curated_sorting(key)

        merge_params = (AutomaticCurationParameters & key).fetch1(
            "merge_params"
        )
        merge_groups, units_merged = self.get_merge_groups(
            parent_sorting, parent_merge_groups, quality_metrics, merge_params
        )

        label_params = (AutomaticCurationParameters & key).fetch1(
            "label_params"
        )
        labels = self.get_labels(
            parent_sorting, parent_labels, quality_metrics, label_params
        )

        # keep the quality metrics only if no merging occurred.
        metrics = quality_metrics if not units_merged else None

        # insert this sorting into the CuratedSpikeSorting Table
        # first remove keys that aren't part of the Sorting (the primary key of curation)
        c_key = (SpikeSorting & key).fetch("KEY")[0]
        curation_key = {item: key[item] for item in key if item in c_key}
        key["auto_curation_key"] = Curation.insert_curation(
            curation_key,
            parent_curation_id=parent_curation_id,
            labels=labels,
            merge_groups=merge_groups,
            metrics=metrics,
            description="auto curated",
        )

        self.insert1(key)

    @staticmethod
    def get_merge_groups(
        sorting, parent_merge_groups, quality_metrics, merge_params
    ):
        """Identifies units to be merged based on the quality_metrics and
        merge parameters and returns an updated list of merges for the curation.

        Parameters
        ---------
        sorting : spikeinterface.sorting
        parent_merge_groups : list
            Information about previous merges
        quality_metrics : list
        merge_params : dict

        Returns
        -------
        merge_groups : list of lists
        merge_occurred : bool

        """

        # overview:
        # 1. Use quality metrics to determine merge groups for units
        # 2. Combine merge groups with current merge groups to produce union of merges

        if not merge_params:
            return parent_merge_groups, False
        else:
            # TODO: use the metrics to identify clusters that should be merged
            # new_merges should then reflect those merges and the line below should be deleted.
            new_merges = []
            # append these merges to the parent merge_groups
            for new_merge in new_merges:
                # check to see if the first cluster listed is in a current merge group
                for previous_merge in parent_merge_groups:
                    if new_merge[0] == previous_merge[0]:
                        # add the additional units in new_merge to the identified merge group.
                        previous_merge.extend(new_merge[1:])
                        previous_merge.sort()
                        break
                else:
                    # append this merge group to the list if no previous merge
                    parent_merge_groups.append(new_merge)
            return parent_merge_groups.sort(), True

    @staticmethod
    def get_labels(sorting, parent_labels, quality_metrics, label_params):
        """Returns a dictionary of labels using quality_metrics and label
        parameters.

        Parameters
        ---------
        sorting : spikeinterface.sorting
        parent_labels : list
            Information about previous merges
        quality_metrics : list
        label_params : dict

        Returns
        -------
        parent_labels : list

        """
        # overview:
        # 1. Use quality metrics to determine labels for units
        # 2. Append labels to current labels, checking for inconsistencies
        if not label_params:
            return parent_labels

        for metric in label_params:
            if metric not in quality_metrics:
                Warning(f"{metric} not found in quality metrics; skipping")
                continue

            compare = _comparison_to_function[label_params[metric][0]]

            for unit_id in quality_metrics[metric]:

                # compare the quality metric to the threshold with the
                # specified operator note that label_params[metric] is a three
                # element list with a comparison operator as a string, the
                # threshold value, and a list of labels to be applied if the
                # comparison is true

                label = label_params[metric]

                if compare(quality_metrics[metric][unit_id], label[1]):
                    if unit_id not in parent_labels:
                        parent_labels[unit_id] = label[2]
                    # check if the label is already there, and if not, add it
                    elif label[2] not in parent_labels[unit_id]:
                        parent_labels[unit_id].extend(label[2])

            return parent_labels


@schema
class CuratedSpikeSortingSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> Curation
    """


@schema
class CuratedSpikeSorting(SpyglassMixin, dj.Computed):
    definition = """
    -> CuratedSpikeSortingSelection
    ---
    -> AnalysisNwbfile
    units_object_id: varchar(40)
    """
    _use_transaction, _allow_insert = False, True

    class Unit(SpyglassMixin, dj.Part):
        definition = """
        # Table for holding sorted units
        -> CuratedSpikeSorting
        unit_id: int   # ID for each unit
        ---
        label='': varchar(200)   # optional set of labels for each unit
        nn_noise_overlap=-1: float   # noise overlap metric for each unit
        nn_isolation=-1: float   # isolation score metric for each unit
        isi_violation=-1: float   # ISI violation score for each unit
        snr=0: float            # SNR for each unit
        firing_rate=-1: float   # firing rate
        num_spikes=-1: int   # total number of spikes
        peak_channel=null: int # channel of maximum amplitude for each unit
        """

    def make(self, key):
        """Populate CuratedSpikeSorting table with curated sorting results.

        1. Fetches metrics and sorting from the Curation table
        2. Saves the sorting in an analysis NWB file
        3. Inserts key into CuratedSpikeSorting table and units into part table.
        """
        AnalysisNwbfile()._creation_times["pre_create_time"] = time.time()
        unit_labels_to_remove = ["reject"]
        # check that the Curation has metrics
        metrics = (Curation & key).fetch1("quality_metrics")
        if metrics == {}:
            logger.warning(
                f"Metrics for Curation {key} should normally be calculated "
                + "before insertion here"
            )

        sorting = Curation.get_curated_sorting(key)
        unit_ids = sorting.get_unit_ids()

        # Get the labels for the units, add only those units that do not have
        # 'reject' or 'noise' labels
        unit_labels = (Curation & key).fetch1("curation_labels")
        accepted_units = []
        for unit_id in unit_ids:
            if unit_id in unit_labels:
                if (
                    len(set(unit_labels_to_remove) & set(unit_labels[unit_id]))
                    == 0
                ):
                    accepted_units.append(unit_id)
            else:
                accepted_units.append(unit_id)

        # get the labels for the accepted units
        labels = {}
        for unit_id in accepted_units:
            if unit_id in unit_labels:
                labels[unit_id] = ",".join(unit_labels[unit_id])

        # convert unit_ids in metrics to integers, including only accepted units.
        #  TODO: convert to int this somewhere else
        final_metrics = {}
        for metric in metrics:
            final_metrics[metric] = {
                int(unit_id): metrics[metric][unit_id]
                for unit_id in metrics[metric]
                if int(unit_id) in accepted_units
            }

        logger.info(f"Found {len(accepted_units)} accepted units")

        # get the sorting and save it in the NWB file
        sorting = Curation.get_curated_sorting(key)
        recording = Curation.get_recording(key)

        # get the sort_interval and sorting interval list
        sort_interval = (SortInterval & key).fetch1("sort_interval")
        sort_interval_list_name = (SpikeSorting & key).fetch1(
            "artifact_removed_interval_list_name"
        )

        timestamps = SpikeSortingRecording._get_recording_timestamps(recording)

        (
            key["analysis_file_name"],
            key["units_object_id"],
        ) = Curation().save_sorting_nwb(
            key,
            sorting,
            timestamps,
            sort_interval_list_name,
            sort_interval,
            metrics=final_metrics,
            unit_ids=accepted_units,
            labels=labels,
        )

        AnalysisNwbfile().log(key, table=self.full_table_name)
        self.insert1(key)

        # now add the units
        # Remove the non primary key entries.
        del key["units_object_id"]
        del key["analysis_file_name"]

        metric_fields = self.metrics_fields()
        for unit_id in accepted_units:
            key["unit_id"] = unit_id
            if unit_id in labels:
                key["label"] = labels[unit_id]
            for field in metric_fields:
                if field in final_metrics:
                    key[field] = final_metrics[field][unit_id]
                else:
                    Warning(
                        f"No metric named {field} in computed unit quality "
                        + "metrics; skipping"
                    )
            CuratedSpikeSorting.Unit.insert1(key)

    def metrics_fields(self):
        """Returns a list of the metrics that are currently in the Units table."""
        unit_info = self.Unit().fetch(limit=1, format="frame")
        unit_fields = [column for column in unit_info.columns]
        unit_fields.remove("label")
        return unit_fields

    @classmethod
    def get_recording(cls, key):
        """Returns the recording related to this curation. Useful for operations downstream of merge table"""
        # expand the key
        recording_key = (cls & key).fetch1("KEY")
        return SpikeSortingRecording()._get_filtered_recording(recording_key)

    @classmethod
    def get_sorting(cls, key):
        """Returns the sorting related to this curation. Useful for operations downstream of merge table"""
        # expand the key
        sorting_key = (cls & key).fetch1("KEY")
        return Curation.get_curated_sorting(sorting_key)

    @classmethod
    def get_sort_group_info(cls, key):
        """Returns the sort group information for the curation
        (e.g. brain region, electrode placement, etc.)

        Parameters
        ----------
        key : dict
            restriction on CuratedSpikeSorting table

        Returns
        -------
        sort_group_info : Table
            Table with information about the sort groups
        """
        table = cls & key

        electrode_restrict_list = []
        for entry in table:
            # Just take one electrode entry per sort group
            electrode_restrict_list.extend(
                ((SortGroup.SortGroupElectrode() & entry) * Electrode).fetch(
                    limit=1
                )
            )
        # Run joins with the tables with info and return
        sort_group_info = (
            (Electrode & electrode_restrict_list)
            * table
            * SortGroup.SortGroupElectrode()
        ) * BrainRegion()
        return sort_group_info
