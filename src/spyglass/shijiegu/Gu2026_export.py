import os
import sys
import datajoint as dj

from spyglass.common.common_usage import Export, ExportSelection
from spyglass.common.common_ephys import Electrode
from spyglass.position.v1 import TrodesPosV1
from spyglass.spikesorting.v1.curation import CurationV1
from spyglass.shijiegu.Analysis_SGU import *
import logging

# Configure logging to write to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("export_output_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)

def my_function():
    try:
        logging.info("Starting the function...")
        # Simulate normal output
        logging.info("Exporting data...")
        
        # Simulate an error (division by zero)
        Export().populate_paper(**paper_key)
        
    except Exception as e:
        # Capture error message and traceback
        logging.error("An error occurred", exc_info=True)

if __name__ == "__main__":
    paper_key = {"paper_id": "Gu2026"}
    my_function()