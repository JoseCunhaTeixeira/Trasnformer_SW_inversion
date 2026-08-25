"""Grand_Est site utility: median RMS/NRMS misfit (invert_qc.py's
DCs-rms.txt outputs) across profiles for a date range.

Site-specific -- light cleanup only (folders.py replaced by config.py),
not held to the same bar as the core pipeline.
Run from the repo root: python experiments/grand_est/print_rms.py

Author : Jose CUNHA TEIXEIRA
License : SNCF Reseau, UMR 7619 METIS, Sorbonne Universite
"""

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from silex.config import Paths

### PARAMS ----------------------------------------------------------------------------------------
paths = Paths.from_env()
model_id = "[202407170928]"
periods = [["2022-07-01", "2022-07-31"]]
site = "Grand_Est"
### -----------------------------------------------------------------------------------------------


RMSs = []
NRMSs = []

rms_pattern = r"RMS: ([\d.]+) m/s"
nrms_pattern = r"NRMS: ([\d.]+) %"

existing_results = sorted(os.listdir(f"{paths.output}/{model_id}/{site}/"))
if "results" in existing_results:
    existing_results.remove("results")

profiles = ["P1", "P2", "P3", "P4", "P5"]
Nprofiles = len(profiles)

for period in periods:
    dates = pd.date_range(start=datetime.strptime(period[0], "%Y-%m-%d"), end=datetime.strptime(period[1], "%Y-%m-%d"), freq="D")
    dates = dates[dates.isin(existing_results)]
    N_dates = len(dates)

    for date in dates:
        for profile in profiles:
            path = f"{paths.output}/{model_id}/{site}/{date.strftime('%Y-%m-%d')}/{profile}/DCs-rms.txt"
            with open(path) as file:
                string = file.read()

                # Extract values using regular expressions
                rms_match = re.search(rms_pattern, string)
                nrms_match = re.search(nrms_pattern, string)

                rms = float(rms_match.group(1)) if rms_match else np.nan
                nrms = float(nrms_match.group(1)) if nrms_match else np.nan

            RMSs.append(rms)
            NRMSs.append(nrms)


print(f"RMS = {np.median(RMSs)}")
print(f"NRMS = {np.median(NRMSs)}")
