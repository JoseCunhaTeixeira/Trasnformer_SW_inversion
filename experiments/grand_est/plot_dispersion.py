"""Grand_Est site figure: per-profile grid of observed vs. recomputed
dispersion curves (invert_qc.py's Vr_fx.txt against the raw field picks),
one subplot per xmid, annotated with RMS misfit.

Site-specific -- light cleanup only (folders.py/misc.py replaced by
config.py/_legacy_utils.py), not held to the same bar as the core
pipeline. Run from the repo root: python experiments/grand_est/plot_dispersion.py

Author : Jose CUNHA TEIXEIRA
License : SNCF Reseau, UMR 7619 METIS, Sorbonne Universite
"""

import json
import os
import sys

import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm

import matplotlib.pyplot as plt

from _legacy_utils import resample_legacy as resamp
from silex.config import Paths

plt.rcParams.update({"font.size": 12})
CM = 1 / 2.54


### PARAMS ----------------------------------------------------------------------------------------
paths = Paths.from_env()
model_id = "[202407170928]"
site = "Grand_Est"
### -----------------------------------------------------------------------------------------------


### FORMATS ---------------------------------------------------------------------------------------
# silex_params.json is the new checkpoint.py format (see invert_qc.py) --
# dz_soil/Ns/soils aren't needed here (they were loaded but never used by
# this script even in the original), so unlike invert_qc.py this doesn't
# also need the training data's params.json.
with open(f"{paths.models}/{model_id}/silex_params.json") as f:
    params = json.load(f)
    min_freq = params["min_freq"]
    max_freq = params["max_freq"]
    min_vel = params["min_vel"]
    max_vel = params["max_vel"]
### -----------------------------------------------------------------------------------------------


### PROFILES TO DISPLAY ---------------------------------------------------------------------------
if not os.path.exists(f"{paths.output}/{model_id}/{site}/"):
    print(f"\033[1;31mERROR: Model {model_id} has no inversion data for site {site}.\033[0m")
    sys.exit()

dates = os.listdir(f"{paths.output}/{model_id}/{site}")
dates = sorted(dates)
if "results" in dates:
    dates.remove("results")
profiles = []
for date in dates:
    year, month, day = date.split("-")
    if year in ["2022", "2023"] and month in ["07"]:  ### <- Change this line to select the desired dates
        list_profiles = os.listdir(f"{paths.output}/{model_id}/{site}/{date}/")
        list_profiles = sorted(list_profiles)
        profiles += [f"{site}/{date}/{profile}" for profile in list_profiles]
### -----------------------------------------------------------------------------------------------


for PROFILE in tqdm(profiles, total=len(profiles)):
    ### Load data ---------------------------------------------------------------------------------
    xs = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/xs.txt")
    x_min = np.min(xs)
    x_max = np.max(xs)

    fs = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/fs.txt")

    Vr_fx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/Vr_fx.txt")
    ### -------------------------------------------------------------------------------------------

    ### FIELD DISPERSION DATA FILES ---------------------------------------------------------------
    PROFILE_NAME = PROFILE.split("/")[-1]

    files = os.listdir(f"{paths.input}/real_data/{PROFILE}/")
    files = sorted(files, key=lambda x: float(x.split("_")[0]))

    xmids = [float(x.split("_")[0]) for x in files]
    xmids = sorted(xmids)

    if not os.path.exists(f"{paths.output}/{model_id}/{PROFILE}/"):
        os.makedirs(f"{paths.output}/{model_id}/{PROFILE}/")
    ### -------------------------------------------------------------------------------------------

    ### DISP --------------------------------------------------------------------------------------
    num_plots = len(xmids)
    subplot_width = int(np.ceil(np.sqrt(num_plots)))
    subplot_height = int(np.ceil(num_plots / subplot_width))
    fig, ax = plt.subplots(subplot_height, subplot_width, figsize=(19 * CM, 19 * CM), dpi=600)
    plt_line = 0
    plt_col = 0

    for Vr_f, file, xmid in zip(Vr_fx.T, files, xmids, strict=True):
        db = np.loadtxt(f"{paths.input}/real_data/{PROFILE}/{file}")

        fs_obs_raw, Vr_obs_raw = db[:, 0], db[:, 1]

        wl = len(Vr_obs_raw) / 4 + 1 if (len(Vr_obs_raw) / 4) % 2 == 0 else len(Vr_obs_raw) / 4
        Vr_obs_raw = savgol_filter(Vr_obs_raw, window_length=wl, polyorder=2, mode="nearest")

        axis_resamp = np.arange(min_freq, max_freq + 1, 1)
        fs_obs_comp, Vr_obs_comp = resamp(fs_obs_raw, Vr_obs_raw, axis_resamp=axis_resamp, type="frequency")
        rms = np.sqrt(np.mean((Vr_obs_comp - Vr_f) ** 2))
        nrms = rms / (np.max(Vr_obs_comp) - np.min(Vr_obs_comp))

        ax[plt_line, plt_col].plot(fs_obs_raw, Vr_obs_raw, linestyle="dotted", color="black", linewidth=1)
        ax[plt_line, plt_col].plot(fs_obs_comp, Vr_obs_comp, color="black", linewidth=1)
        ax[plt_line, plt_col].plot(fs, Vr_f, color="blue", linestyle="--", linewidth=1)
        ax[plt_line, plt_col].set_ylim([200, 400])
        ax[plt_line, plt_col].set_xlim([5, 50])
        ax[plt_line, plt_col].set_xticks([10, 30, 50])
        ax[plt_line, plt_col].set_title(
            f"x={xmid:.0f} m\nRMS={rms:.0f} m/s", fontsize=5, y=0.65, loc="right", color=(0, 0, 0, 0.5)
        )

        if plt_line == subplot_height - 1:
            if plt_col == 0:
                ax[plt_line, plt_col].tick_params(
                    top=True, labeltop=False, bottom=True, labelbottom=True, left=True, labelleft=True, right=False, labelright=False
                )
                ax[plt_line, plt_col].set_xlabel("Frequency\n[Hz]")
                ax[plt_line, plt_col].set_ylabel("${V_R}$ [m/s]")
            else:
                ax[plt_line, plt_col].tick_params(
                    top=True, labeltop=False, bottom=True, labelbottom=True, left=True, labelleft=False, right=True, labelright=False
                )
                ax[plt_line, plt_col].set_xlabel("Frequency [Hz]")
                ax[plt_line, plt_col].set_ylabel("")
        else:
            if plt_col == 0:
                ax[plt_line, plt_col].tick_params(
                    top=True, labeltop=False, bottom=True, labelbottom=False, left=True, labelleft=True, right=True, labelright=False
                )
                ax[plt_line, plt_col].set_xlabel("")
                ax[plt_line, plt_col].set_ylabel("${V_R}$ [m/s]")
            else:
                ax[plt_line, plt_col].tick_params(
                    top=True, labeltop=False, bottom=True, labelbottom=False, left=True, labelleft=False, right=True, labelright=False
                )
                ax[plt_line, plt_col].set_xlabel("")
                ax[plt_line, plt_col].set_ylabel("")

        if plt_col == subplot_width - 1:
            plt_line += 1
            plt_col = 0
        else:
            plt_col += 1
    ### -------------------------------------------------------------------------------------------

    while plt_col < subplot_width and plt_line < subplot_height:
        ax[plt_line, plt_col].axis("off")
        plt_col += 1
    fig.legend(["Observed not used as input", "Observed used as input", "Recomputed from inferred outputs"], loc="lower center")
    p = PROFILE.replace("/", "_")
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/DCs-comp.svg", bbox_inches="tight", transparent=True)
    plt.close("all")
