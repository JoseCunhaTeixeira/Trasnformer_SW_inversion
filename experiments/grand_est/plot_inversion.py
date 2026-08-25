"""Grand_Est site figures: per-profile depth sections (soil type, N, and
the rock-physics/seismic fields invert_qc.py derives -- h, Sw, Swe, Kf,
rhof, rhob, Km, mum, Vs, Vp, Vr), each also as a mode-filtered smoothed
variant.

Site-specific -- light cleanup only (folders.py/misc.py replaced by
config.py/_legacy_utils.py), not held to the same bar as the core
pipeline. Run from the repo root: python experiments/grand_est/plot_inversion.py

Author : Jose CUNHA TEIXEIRA
License : SNCF Reseau, UMR 7619 METIS, Sorbonne Universite
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.ndimage import generic_filter
from scipy.signal import savgol_filter
from tqdm import tqdm

import matplotlib.pyplot as plt

from _legacy_utils import mode_filter_count, mode_filter_mean
from silex.config import Paths

plt.rcParams.update({"font.size": 12})
CM = 1 / 2.54


### PARAMS ----------------------------------------------------------------------------------------
paths = Paths.from_env()
model_id = "[202407170928]"
site = "Grand_Est"
### -----------------------------------------------------------------------------------------------


### FORMATS ---------------------------------------------------------------------------------------
# The generation-time params (soils/d_thickness/Ns), not part of
# checkpoint.py's trimmed inference-only silex_params.json -- see
# invert_qc.py, which reads this same training_data/<site>/params.json.
with open(f"{paths.input}/training_data/{site}/params.json") as f:
    data_params = json.load(f)
    dz_soil = data_params["d_thickness"]
    Ns = data_params["Ns"]
    soils = data_params["soils"]
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
    z_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/z_zx.txt")
    zs_lengths_x = [len(zs) - np.isnan(zs).sum() for zs in z_zx.T]
    longest_col_index = np.argmax(zs_lengths_x)
    zs = z_zx[:, longest_col_index]

    max_z = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/max_z.txt")

    xs = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/xs.txt")
    x_min = np.min(xs)
    x_max = np.max(xs)

    fs = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/fs.txt")

    soil_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/soil_zx.txt", dtype=str)
    thick_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/thick_zx.txt")
    N_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/N_zx.txt")
    WT_x = -np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/WT_x.txt")

    h_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/h_zx.txt")
    Sw_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/Sw_zx.txt")
    Swe_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/Swe_zx.txt")

    mus_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/mus_zx.txt")
    Ks_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/Ks_zx.txt")
    rhos_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/rhos_zx.txt")
    nus_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/nus_zx.txt")

    Kf_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/Kf_zx.txt")
    rhof_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/rhof_zx.txt")
    rhob_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/rhob_zx.txt")

    Km_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/Km_zx.txt")
    mum_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/mum_zx.txt")

    Vs_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/Vs_zx.txt")
    Vp_zx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/Vp_zx.txt")
    Vr_fx = np.loadtxt(f"{paths.output}/{model_id}/{PROFILE}/Vr_fx.txt")

    zs = zs[0 : np.where(zs == max_z)[0][0] + 1]
    h_zx = h_zx[: len(zs), :]
    Sw_zx = Sw_zx[: len(zs), :]
    Swe_zx = Swe_zx[: len(zs), :]
    mus_zx = mus_zx[: len(zs), :]
    Ks_zx = Ks_zx[: len(zs), :]
    rhos_zx = rhos_zx[: len(zs), :]
    nus_zx = nus_zx[: len(zs), :]
    Kf_zx = Kf_zx[: len(zs), :]
    rhof_zx = rhof_zx[: len(zs), :]
    rhob_zx = rhob_zx[: len(zs), :]
    Km_zx = Km_zx[: len(zs), :]
    mum_zx = mum_zx[: len(zs), :]
    Vs_zx = Vs_zx[: len(zs), :]
    Vp_zx = Vp_zx[: len(zs), :]
    ### -------------------------------------------------------------------------------------------

    ### Smooth water table profile -----------------------------------------------------------------
    wl = len(WT_x) / 4 + 1 if (len(WT_x) / 4) % 2 == 0 else len(WT_x) / 4
    smoothed_WT_x = savgol_filter(WT_x, window_length=wl, polyorder=2, mode="nearest")
    ### -------------------------------------------------------------------------------------------

    ### SOILS -------------------------------------------------------------------------------------
    soil_to_int = {"None": 0}
    for i, soil in enumerate(soils):
        soil_to_int[soil] = i + 1
    int_to_soil = {val: key for key, val in soil_to_int.items()}

    soil_int_zx = []
    for i, (soil_z, thick_z) in enumerate(zip(soil_zx.T, thick_zx.T, strict=True)):
        log = []
        for soil, thick in zip(soil_z, thick_z, strict=True):
            if soil != "None":
                log.extend([soil_to_int[soil]] * int(thick / dz_soil))
        soil_int_zx.append(log)
    soil_int_zx = pd.DataFrame(soil_int_zx).to_numpy().T
    soil_int_zx = soil_int_zx[: int(abs(max_z) / dz_soil), :]

    zs_soil = np.linspace(0, max_z, len(soil_int_zx))

    cmap = ListedColormap(["mediumblue", "dodgerblue", "limegreen", "yellow"])

    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs_soil, soil_int_zx, cmap=cmap, vmin=1, vmax=len(int_to_soil) - 1, alpha=0.5)
    ax.plot(xs, smoothed_WT_x, color="k", linewidth=1, linestyle="--")
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    colorbar = fig.colorbar(im, ax=ax, label="Soil type", aspect=10)
    colorbar.set_ticks(list(soil_to_int.values())[1:])
    colorbar.set_ticklabels(list(soil_to_int.keys())[1:])
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/soil_zx.png", bbox_inches="tight")

    soil_int_zx = np.where(np.isnan(soil_int_zx), 0, soil_int_zx)
    smoothed_soil_int_zx = generic_filter(soil_int_zx, mode_filter_count, size=(1, 3))
    smoothed_soil_int_zx = generic_filter(smoothed_soil_int_zx, mode_filter_count, size=(1, 3))
    soil_int_zx = np.where(soil_int_zx == 0, np.nan, soil_int_zx)
    smoothed_soil_int_zx = np.where(smoothed_soil_int_zx == 0, np.nan, smoothed_soil_int_zx)

    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs_soil, smoothed_soil_int_zx, cmap=cmap, vmin=1, vmax=len(int_to_soil) - 1, alpha=0.5)
    ax.plot(xs, smoothed_WT_x, color="k", linewidth=1, linestyle="--")
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    colorbar = fig.colorbar(im, ax=ax, label="Soil type", aspect=10)
    colorbar.set_ticks(list(soil_to_int.values())[1:])
    colorbar.set_ticklabels(list(soil_to_int.keys())[1:])
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/soil-smoothed_zx.png", bbox_inches="tight")
    ### -------------------------------------------------------------------------------------------

    ### N -----------------------------------------------------------------------------------------
    if len(Ns) == 5:
        cmap = ListedColormap(["tab:blue", "tab:cyan", "tab:green", "tab:orange", "tab:red"])
        figsize = (19 * CM, 2.5 * CM)
        cb_aspect = 10
        cb_loc = "right"
        cb_fontsize = 12
    else:
        cmap = "hot_r"
        figsize = (19 * CM, 3.5 * CM)
        cb_aspect = 51
        cb_loc = "bottom"
        cb_fontsize = 4

    N_int_zx = []
    for i, (N_z, thick_z) in enumerate(zip(N_zx.T, thick_zx.T, strict=True)):
        log = []
        for N, thick in zip(N_z, thick_z, strict=True):
            if ~np.isnan(N):
                log.extend([int(N)] * int(thick / dz_soil))
        N_int_zx.append(log)
    N_int_zx = pd.DataFrame(N_int_zx).to_numpy().T
    N_int_zx = N_int_zx[: int(abs(max_z) / dz_soil), :]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    im = ax.pcolormesh(xs, zs_soil, N_int_zx, cmap=cmap, vmin=min(Ns), vmax=max(Ns))
    ax.plot(xs, smoothed_WT_x, color="k", linewidth=1, linestyle="--")
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    cb = fig.colorbar(im, ax=ax, label="N", aspect=cb_aspect, location=cb_loc)
    cb.set_ticks(Ns)
    cb.set_ticklabels(Ns, fontsize=cb_fontsize)
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/N_zx.png", bbox_inches="tight")

    N_int_zx = np.where(np.isnan(N_int_zx), 0, N_int_zx)
    smoothed_N_int_zx = generic_filter(N_int_zx, mode_filter_count, size=(1, 3))
    smoothed_N_int_zx = generic_filter(smoothed_N_int_zx, mode_filter_count, size=(1, 3))
    N_int_zx = np.where(N_int_zx == 0, np.nan, N_int_zx)
    smoothed_N_int_zx = np.where(smoothed_N_int_zx == 0, np.nan, smoothed_N_int_zx)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    im = ax.pcolormesh(xs, zs_soil, smoothed_N_int_zx, cmap=cmap, vmin=min(Ns), vmax=max(Ns))
    ax.plot(xs, smoothed_WT_x, color="k", linewidth=1, linestyle="--")
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    cb = fig.colorbar(im, ax=ax, label="N", aspect=cb_aspect, location=cb_loc)
    cb.set_ticks(Ns)
    cb.set_ticklabels(Ns, fontsize=cb_fontsize)
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/N-smoothed_zx.png", bbox_inches="tight")
    ### -------------------------------------------------------------------------------------------

    ### h -----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs, h_zx, cmap="terrain")
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    cb = fig.colorbar(im, ax=ax, label="${h}$ [m/s]", aspect=10)
    cb.minorticks_on()
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/h_zx.png", bbox_inches="tight")
    ### -------------------------------------------------------------------------------------------

    ### Sw ----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs, Sw_zx, cmap="terrain")
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    cb = fig.colorbar(im, ax=ax, label="${Sw}$ [m/s]", aspect=10)
    cb.minorticks_on()
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/Sw_zx.png", bbox_inches="tight")
    ### -------------------------------------------------------------------------------------------

    ### Swe ---------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs, Swe_zx, cmap="terrain")
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    cb = fig.colorbar(im, ax=ax, label="${Swe}$ [m/s]", aspect=10)
    cb.minorticks_on()
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/Swe_zx.png", bbox_inches="tight")
    ### -------------------------------------------------------------------------------------------

    ### Kf ----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs, Kf_zx, cmap="terrain")
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    cb = fig.colorbar(im, ax=ax, label="${K_f}$ [m/s]", aspect=10)
    cb.minorticks_on()
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/Kf_zx.png", bbox_inches="tight")
    ### -------------------------------------------------------------------------------------------

    ### rhof --------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs, rhof_zx, cmap="terrain")
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    cb = fig.colorbar(im, ax=ax, label="${rho_f}$ [m/s]", aspect=10)
    cb.minorticks_on()
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/rhof_zx.png", bbox_inches="tight")
    ### -------------------------------------------------------------------------------------------

    ### rhob --------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs, rhob_zx, cmap="terrain")
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    cb = fig.colorbar(im, ax=ax, label="${rho_b}$ [m/s]", aspect=10)
    cb.minorticks_on()
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/rhob_zx.png", bbox_inches="tight")
    ### -------------------------------------------------------------------------------------------

    ### Km ----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs, Km_zx / 1_000_000_000, cmap="terrain")
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    cb = fig.colorbar(im, ax=ax, label="${K_m}$ [GPa]", aspect=10)
    cb.minorticks_on()
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/Km_zx.png", bbox_inches="tight")

    smoothed_Km_zx = generic_filter(Km_zx, mode_filter_mean, size=(3, 3))

    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs, smoothed_Km_zx / 1_000_000_000, cmap="terrain")
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    ax.set_ylim([max_z, 0])
    cb = fig.colorbar(im, ax=ax, label="${K_m}$ [GPa]", aspect=10)
    cb.minorticks_on()
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/Km-smoothed_zx.png", bbox_inches="tight")
    ### -------------------------------------------------------------------------------------------

    ### mum ---------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs, mum_zx / 1_000_000_000, cmap="terrain")
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    cb = fig.colorbar(im, ax=ax, label="${mu_m}$ [GPa]", aspect=10)
    cb.minorticks_on()
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/mum_zx.png", bbox_inches="tight")

    smoothed_mum_zx = generic_filter(mum_zx, mode_filter_mean, size=(3, 3))

    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs, smoothed_mum_zx / 1_000_000_000, cmap="terrain")
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    ax.set_ylim([max_z, 0])
    cb = fig.colorbar(im, ax=ax, label="${mu_m}$ [GPa]", aspect=10)
    cb.minorticks_on()
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/mum-smoothed_zx.png", bbox_inches="tight")
    ### -------------------------------------------------------------------------------------------

    ### Vs ----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs, Vs_zx, cmap="terrain", vmin=200, vmax=600)
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    cb = fig.colorbar(im, ax=ax, label="${V_S}$ [m/s]", aspect=10)
    cb.minorticks_on()
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/Vs.png", bbox_inches="tight")

    smoothed_Vs_zx = generic_filter(Vs_zx, mode_filter_mean, size=(3, 3))

    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs, smoothed_Vs_zx, cmap="terrain", vmin=200, vmax=600)
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    ax.set_ylim([max_z, 0])
    cb = fig.colorbar(im, ax=ax, label="${V_S}$ [m/s]", aspect=10)
    cb.minorticks_on()
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/Vs-smooth.png", bbox_inches="tight")
    ### -------------------------------------------------------------------------------------------

    ### Vp ----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(19 * CM, 2.5 * CM), dpi=300)
    im = ax.pcolormesh(xs, zs, Vp_zx, cmap="terrain")
    ax.set_xlabel("Position [m]")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Depth [m]")
    ax.set_ylim([max_z, 0])
    ax.tick_params(which="both", labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
    cb = fig.colorbar(im, ax=ax, label="${V_P}$ [m/s]", aspect=10)
    cb.minorticks_on()
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/Vp.png", bbox_inches="tight")
    ### -------------------------------------------------------------------------------------------

    ### DISP --------------------------------------------------------------------------------------
    cmap = plt.get_cmap("nipy_spectral")
    colors = [cmap(i / (Vr_fx.shape[1] - 1)) for i in range(Vr_fx.shape[1])]

    fig, ax = plt.subplots(figsize=(19 * CM, 6 * CM), dpi=300)
    for i, Vr_f in enumerate(Vr_fx.T):
        ax.plot(fs, Vr_f, color=colors[i])
    ax.set_ylim([100, 500])
    ax.set_xlim([fs[0], fs[-1]])
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("${V_R}$ [m/s]")
    ax.minorticks_on()
    fig.savefig(f"{paths.output}/{model_id}/{PROFILE}/Vr.png", bbox_inches="tight")
    ### -------------------------------------------------------------------------------------------

    plt.close("all")
