from functools import cached_property
from pathlib import Path

import datajoint as dj
from tqdm import tqdm

from spyglass.common import AnalysisNwbfile
from spyglass.utils.database_settings import SHARED_MODULES

schema = dj.schema("cbroz_check_files")


@schema
class AnalysisFileIssues(dj.Computed):
    definition = """
    -> AnalysisNwbfile
    ---
    exists=1   : bool # whether the analysis file exists
    readable=1 : bool # whether the analysis file is readable
    issue=NULL : varchar(255) # description of the issue
    table=NULL : varchar(64) # name of the table that created the analysis file
    """

    @cached_property
    def analysis_children(self):
        banned = [
            "`common_nwbfile`.`analysis_nwbfile_log`",
            "`cbroz_check_files`.`__analysis_file_issues`",
        ]
        return [
            c
            for c in AnalysisNwbfile().children(as_objects=True)
            if c.full_table_name not in banned
        ]

    def get_tbl(self, key):
        ret = []
        f_key = dict(analysis_file_name=key["analysis_file_name"])
        for child in self.analysis_children:
            if child & f_key:
                ret.append(child.full_table_name)
        if len(ret) != 1:
            raise ValueError(
                f"{len(ret)} tables for {key['analysis_file_name']}: {ret}"
            )
        return ret[0]

    def make(self, key):
        """
        Check if the analysis file exists and is readable.
        """
        insert = key.copy()
        not_exist = dict(exists=False, readable=False)
        checksum = dict(exists=True, readable=False)
        fname = None
        try:
            fname = AnalysisNwbfile().get_abs_path(key["analysis_file_name"])
        except FileNotFoundError as e:
            insert = dict(key, **not_exist, issue=e.args[0])
        except dj.DataJointError as e:
            insert = dict(
                key, **checksum, issue=e.args[0], table=self.get_tbl(key)
            )
        if fname is not None and not Path(fname).exists():
            insert.update(**not_exist, issue=f"path not found: {fname}")
        self.insert1(insert, skip_duplicates=True)

    def show_downstream(self, restriction=True):
        entries = (self & "readable=0" & restriction).fetch("KEY", as_dict=True)
        if not entries:
            print("No issues found.")
            return
        ret = [(c & entries) for c in self.analysis_children if (c & entries)]
        if not ret:
            print("No issues found.")
            return
        return ret if len(ret) > 1 else ret[0]


if __name__ == "__main__":
    afi = AnalysisFileIssues()
    afi.show_downstream(restriction=True)
