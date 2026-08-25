"""Grand_Est site QC pipeline: run the trained model on real field
dispersion curves, then Santiludo rock physics + GPDC re-forward-model the
decoded soil profile to check how well it reproduces the observed curve
(RMS/NRMS), writing per-profile depth sections (soil, N, WT, h, Sw, Swe,
mus, Ks, rhos, nus, Kf, rhof, rhob, Km, mum, Vp, Vs, Vr) that
plot_inversion.py/plot_dispersion.py/plot_WT.py/print_rms.py read.

Was run_invertion.py. The AI-inference call itself is genuinely general
infrastructure and already lives in sigpipe's silex.py (SilexModel) for
production use -- this script's remaining job is Grand_Est-specific
rock-physics QC, so it stays here rather than in the core silex package,
light cleanup only (not held to the same bar). Concrete fixes made while
porting: the checkpoint is loaded via silex.checkpoint.load_checkpoint +
silex.decoding.decode instead of unpickling a `Transformer` instance
(pickle.load can't deserialize that class any more -- it doesn't exist in
this codebase; a checkpoint saved by scripts/2_train.py or migrated with
sigpipe's experiments/silex/export_legacy_model.py is required now); the
gpdc subprocess call no longer uses shell=True; a missing `rhob_z =
rhob_z[::factor]` (every other per-depth array on either side of it in
the original was resliced, this one silently wasn't) and a bare `sys.exit`
(no parens -- a no-op, not an exit) are fixed; the TensorFlow-specific
"run on CPUs" block is dropped (this repo runs on a torch backend, and
TensorFlow isn't a dependency here at all -- see silex/__init__.py).

Run from the repo root: python experiments/grand_est/invert_qc.py

Author : Jose CUNHA TEIXEIRA
License : SNCF Reseau, UMR 7619 METIS, Sorbonne Universite
"""

import json
import os
import sys
from io import StringIO
from subprocess import CalledProcessError, run

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

from _legacy_utils import resample_legacy as resamp
from silex.checkpoint import load_checkpoint
from silex.config import Paths
from silex.decoding import decode

# santiludo's compiled extension modules need a C++ toolchain to build
# (see generation.py's module docstring) -- an environment without one
# still fails to resolve this import even though it's a normal optional
# dependency (`uv sync --extra generation`), hence the ignore rather than
# that being a real error here.
from santiludo.RPfunctions import (  # pyright: ignore[reportMissingImports]
    biotGassmann,
    effFluid,
    hertzMindlin,
    hillsAverage,
)
from santiludo.TTDSPfunctions import (  # pyright: ignore[reportMissingImports]
    readDispersion,
    writeVelocityModel,
)
from santiludo.VGfunctions import vanGen  # pyright: ignore[reportMissingImports]

cm = 1 / 2.54


### PARAMS ----------------------------------------------------------------------------------------
paths = Paths.from_env()
model_id = "[202407170928]"
site = "Grand_Est"
dx = 3  # Distance between the dispersion curves [m]
### -----------------------------------------------------------------------------------------------


### LOAD MODEL ------------------------------------------------------------------------------------
print(f"\nLoading checkpoint: {paths.models}/{model_id}")
model, checkpoint_params = load_checkpoint(paths.models / model_id)
### -----------------------------------------------------------------------------------------------


### PROFILES TO INVERT ----------------------------------------------------------------------------
dates = os.listdir(f"{paths.input}/real_data/{site}/")
dates = sorted(dates)
profiles = []
for date in dates:
    year, month, day = date.split("-")
    if year in ["2022", "2023"] and month in ["07"]:  ### <- Change this line to select the desired dates
        list_profiles = os.listdir(f"{paths.input}/real_data/{site}/{date}/")
        list_profiles = sorted(list_profiles)
        profiles += [f"{site}/{date}/{profile}" for profile in list_profiles]
dxs = [dx] * len(profiles)
### -----------------------------------------------------------------------------------------------


### FORMATS ---------------------------------------------------------------------------------------
min_freq = checkpoint_params.min_freq
max_freq = checkpoint_params.max_freq
min_vel = checkpoint_params.min_vel
max_vel = checkpoint_params.max_vel
N_freqs = checkpoint_params.n_freqs
print(f"\n{min_freq = }, {max_freq = }, {min_vel = }, {max_vel = }")

word_to_index = checkpoint_params.word_to_index
index_to_word = checkpoint_params.index_to_word
len_output_seq = checkpoint_params.output_seq_length
start_id = word_to_index["[START]"]
end_id = word_to_index["[END]"]
pad_id = word_to_index["[PAD]"]

# Generation-time params (rock physics/gpdc), not part of checkpoint.py's
# trimmed inference-only format -- read from the training data's own
# params.json instead (produced by generation.py's save_dataset).
with open(f"{paths.input}/training_data/{site}/params.json") as f:
    data_params = json.load(f)

# Under layers
under_layers = data_params["under_layers"]
N_under_layers = data_params["N_under_layers"]

# Geometry and discretisation of the medium
dz_origin = data_params["dz"]  # Depth sample interval [m]
dz = dz_origin
top_surface_level_origin = data_params["top_surface_level"]  # Altitude of the soil surface[m]
top_surface_level = top_surface_level_origin

n_modes = data_params["n_modes"]  # Number of modes to compute
### -----------------------------------------------------------------------------------------------


### ROCK PHYSICS CONSTANTS ------------------------------------------------------------------------
rhow = 1000.0  # Water density [Kg/m3]
rhoa = 1.0  # Air density [Kg/m3]
kw = 2.3e9  # Water bulk modulus [Pa]
ka = 1.01e5  # Air bulk modulus [Pa]
g = 9.82  # Gravity acceleration [m/s2]

# Grains/agregate mechanical properties
mu_clay = 6.8  # Shear moduli [GPa]
mu_silt = 45.0
mu_sand = 45.0
k_clay = 25.0  # Bulk moduli [GPa]
k_silt = 37.0
k_sand = 37.0
rho_clay = 2580.0  # Density [kg/m3]
rho_silt = 2600.0
rho_sand = 2600.0

# Three possible RP models:
kk = 3  # Pe with suction (cf. Solazzi et al. 2021)
### -----------------------------------------------------------------------------------------------


### SEISMIC CONSTANTS -----------------------------------------------------------------------------
s = "frequency"  # Over frequencies mode
wave = "R"  # Rayleigh (PSV) fundamental mode
### -----------------------------------------------------------------------------------------------


for profile, dx in tqdm(zip(profiles, dxs, strict=True), total=len(profiles), desc="Profiles", colour="green"):
    ### FIELD DISPERSION DATA FILES ---------------------------------------------------------------
    PROFILE_NAME = profile.split("/")[-1]

    files = os.listdir(f"{paths.input}/real_data/{profile}/")
    files = sorted(files, key=lambda x: float(x.split("_")[0]))

    xmids = [float(x.split("_")[0]) for x in files]
    xmids = sorted(xmids)

    if not os.path.exists(f"{paths.output}/{model_id}/{profile}/"):
        os.makedirs(f"{paths.output}/{model_id}/{profile}/")
    ### -------------------------------------------------------------------------------------------

    ### INVERSION ---------------------------------------------------------------------------------
    z_zx = []

    soil_zx = []
    thick_zx = []
    N_zx = []
    WT_x = []

    h_zx = []
    Sw_zx = []
    Swe_zx = []

    mus_zx = []
    Ks_zx = []
    rhos_zx = []
    nus_zx = []

    Kf_zx = []
    rhof_zx = []
    rhob_zx = []

    Km_zx = []
    mum_zx = []

    Vs_zx = []
    Vp_zx = []
    disp_db = []

    rms_x = []

    for file in tqdm(files, total=len(files), leave=False, desc="Xmids"):
        computed = False
        flag = False
        if dz != dz_origin:
            dz = dz_origin
            top_surface_level = top_surface_level_origin
            print(f"INFO : dz reset at {dz_origin}\n")

        while computed is False:
            ### OBSERVED DATA -------------------------------------------------------------------------
            db = np.loadtxt(f"{paths.input}/real_data/{profile}/{file}")
            fs_obs_raw, Vr_obs_raw = db[:, 0], db[:, 1]

            wl = len(Vr_obs_raw) / 4 + 1 if (len(Vr_obs_raw) / 4) % 2 == 0 else len(Vr_obs_raw) / 4
            Vr_obs_raw = savgol_filter(Vr_obs_raw, window_length=wl, polyorder=2, mode="nearest")

            axis_resamp = np.arange(min_freq, max_freq + 1, 1)
            fs_obs, Vr_obs = resamp(fs_obs_raw, Vr_obs_raw, axis_resamp=axis_resamp, type="frequency")

            Vr_obs_comp = np.copy(Vr_obs)

            Vr_obs = (Vr_obs - min_vel) / (max_vel - min_vel)

            Vr_obs = Vr_obs.reshape(1, Vr_obs.shape[0], 1).astype(np.float32)
            ### ---------------------------------------------------------------------------------------

            ### INFERENCE -----------------------------------------------------------------------------
            input_seq = Vr_obs

            decoded_ids = decode(
                model, input_seq, checkpoint_params.forbidden_tokens, start_id, end_id, pad_id, len_output_seq
            )

            decoded_GM = [index_to_word[idx] for idx in decoded_ids]

            soil_types = decoded_GM[3::6]
            soil_types = [soil for soil in soil_types if soil not in ["[PAD]", "[END]"]]

            GM_thicknesses = decoded_GM[5::6]
            GM_thicknesses = [float(thickness) for thickness in GM_thicknesses if thickness not in ["[PAD]", "[END]"]]

            Ns = decoded_GM[7::6]
            Ns = [float(N) for N in Ns if N not in ["[PAD]", "[END]"]]

            WT = float(decoded_GM[1])

            fracs = [0.3] * len(soil_types)

            depth = np.sum(GM_thicknesses)

            for soil in soil_types:
                if soil not in data_params["soils"]:
                    print("Error: Soil type not in soil vocabulary.")
                    sys.exit()
            ### ---------------------------------------------------------------------------------------

            ### ROCK PHYSICS CONSTANTS ----------------------------------------------------------------
            zs = -np.arange(top_surface_level, depth + dz, dz)  # Depth positions (negative downward) [m]

            NbCells = len(zs) - 1  # Number of exploration points in depth [#]
            ### ---------------------------------------------------------------------------------------

            ### SEISMIC CONSTANTS ---------------------------------------------------------------------
            VM_thicknesses = np.diff(np.abs(zs))  # thickness vector [m]
            ### ---------------------------------------------------------------------------------------

            #### ROCK PHYSICS -------------------------------------------------------------------------
            # Saturation profile with depth
            h_z, Sw_z, Swe_z = vanGen(zs, WT, soil_types, GM_thicknesses)

            # Effective Grain Properties (constant with depth)
            mus_z, Ks_z, rhos_z, nus_z = hillsAverage(
                mu_clay, mu_silt, mu_sand, rho_clay, rho_silt, rho_sand, k_clay, k_silt, k_sand, soil_types
            )

            # Effective Fluid Properties
            Kf_z, rhof_z, rhob_z = effFluid(Sw_z, kw, ka, rhow, rhoa, rhos_z, soil_types, GM_thicknesses, dz)

            # Hertz Mindlin Frame Properties
            Km_z, mum_z = hertzMindlin(
                Swe_z, zs, h_z, rhob_z, g, rhoa, rhow, Ns, mus_z, nus_z, fracs, kk, soil_types, GM_thicknesses
            )

            # Saturated Properties
            Vp_z, Vs_z = biotGassmann(Km_z, mum_z, Ks_z, Kf_z, rhob_z, soil_types, GM_thicknesses, dz)
            ### ---------------------------------------------------------------------------------------

            #### SEISMIC FWD MODELING -----------------------------------------------------------------
            # Velocity model in string format for GPDC
            velocity_model_string = writeVelocityModel(
                VM_thicknesses, Vp_z, Vs_z, rhob_z, under_layers, N_under_layers
            )

            # Dispersion curves computing with GPDC
            velocity_model_RAMfile = StringIO(
                velocity_model_string
            )  # Keep velocity model string in the RAM in a file format alike to trick GPDC which expects a file
            gpdc_args = [
                "gpdc",
                f"-{wave}",
                str(n_modes),
                "-n",
                str(N_freqs),
                "-min",
                str(min_freq),
                "-max",
                str(max_freq),
                "-s",
                s,
            ]

            try:
                process = run(
                    gpdc_args,
                    input=velocity_model_RAMfile.getvalue(),
                    text=True,
                    shell=False,
                    capture_output=True,
                    check=True,
                )  # Raw output string from GPDC
            except CalledProcessError as e:
                print(f"\nERROR during GPDC computation. Returned:\n{e.stdout}")
                print("Used parameters:")
                print(f"{soil_types = }")
                print(f"{GM_thicknesses = }")
                print(f"{Ns = }")
                print(f"{fracs = }")
                print(f"{WT = }")
                print(f"{dz = }\n")
                dz /= 10
                top_surface_level /= 10
                print(f"INFO : dz reduced at {dz}\n")
                if dz > 0.001:
                    continue
                else:
                    dispersion_data = disp_db[-1]
                    rms = rms_x[-1]
                    flag = True

            computed = True

            if not flag:
                gpdc_output_string = process.stdout  # Raw output string from GPDC
                dispersion_data, n_modes = readDispersion(
                    gpdc_output_string
                )  # Reads GPDC output and converts dispersion data to a list of numpy arrays for each mode
                # Updates number of computed modes (can be lower than what was defined if frequency range too small)
            flag = False

            rms = np.sqrt(np.mean((Vr_obs_comp - dispersion_data[0][:, 1]) ** 2))
            nrms = rms / (np.max(Vr_obs_comp) - np.min(Vr_obs_comp))

            factor = int(dz_origin / dz)
            zs = zs[::factor]
            h_z = h_z[::factor]
            Sw_z = Sw_z[::factor]
            Swe_z = Swe_z[::factor]
            Kf_z = Kf_z[::factor]
            rhof_z = rhof_z[::factor]
            rhob_z = rhob_z[::factor]
            Km_z = Km_z[::factor]
            mum_z = mum_z[::factor]
            Vp_z = Vp_z[::factor]
            Vs_z = Vs_z[::factor]

            thick_zx.append(GM_thicknesses)
            soil_zx.append(soil_types)
            WT_x.append(WT)
            N_zx.append(Ns)
            z_zx.append(zs)
            h_zx.append(h_z)
            Sw_zx.append(Sw_z)
            Swe_zx.append(Swe_z)
            mus_zx.append(mus_z)
            Ks_zx.append(Ks_z)
            rhos_zx.append(rhos_z)
            nus_zx.append(nus_z)
            Kf_zx.append(Kf_z)
            rhof_zx.append(rhof_z)
            rhob_zx.append(rhob_z)
            Km_zx.append(Km_z)
            mum_zx.append(mum_z)
            Vp_zx.append(Vp_z)
            Vs_zx.append(Vs_z)
            disp_db.append(dispersion_data)
            rms_x.append(rms)
            ### ---------------------------------------------------------------------------------------

    ### SAVE DATA ---------------------------------------------------------------------------------
    z_zx = pd.DataFrame(z_zx).to_numpy().T
    xmids = np.array(xmids)
    xs = xmids

    soil_zx = pd.DataFrame(soil_zx).to_numpy().T
    if soil_zx.shape[0] < 4:
        soil_zx = np.pad(soil_zx, ((0, 4 - soil_zx.shape[0]), (0, 0)), "constant", constant_values="None")
    thick_zx = pd.DataFrame(thick_zx).to_numpy().T
    if thick_zx.shape[0] < 4:
        thick_zx = np.pad(thick_zx, ((0, 4 - thick_zx.shape[0]), (0, 0)), "constant", constant_values=np.nan)
    N_zx = pd.DataFrame(N_zx).to_numpy().T
    if N_zx.shape[0] < 4:
        N_zx = np.pad(N_zx, ((0, 4 - N_zx.shape[0]), (0, 0)), "constant", constant_values=np.nan)
    WT_x = pd.array(WT_x).reshape(len(WT_x))

    h_zx = pd.DataFrame(h_zx).to_numpy().T
    Sw_zx = pd.DataFrame(Sw_zx).to_numpy().T
    Swe_zx = pd.DataFrame(Swe_zx).to_numpy().T

    mus_zx = pd.DataFrame(mus_zx).to_numpy().T
    if mus_zx.shape[0] < 4:
        mus_zx = np.pad(mus_zx, ((0, 4 - mus_zx.shape[0]), (0, 0)), "constant", constant_values=np.nan)
    Ks_zx = pd.DataFrame(Ks_zx).to_numpy().T
    if Ks_zx.shape[0] < 4:
        Ks_zx = np.pad(Ks_zx, ((0, 4 - Ks_zx.shape[0]), (0, 0)), "constant", constant_values=np.nan)
    rhos_zx = pd.DataFrame(rhos_zx).to_numpy().T
    if rhos_zx.shape[0] < 4:
        rhos_zx = np.pad(rhos_zx, ((0, 4 - rhos_zx.shape[0]), (0, 0)), "constant", constant_values=np.nan)
    nus_zx = pd.DataFrame(nus_zx).to_numpy().T
    if nus_zx.shape[0] < 4:
        nus_zx = np.pad(nus_zx, ((0, 4 - nus_zx.shape[0]), (0, 0)), "constant", constant_values=np.nan)

    Kf_zx = pd.DataFrame(Kf_zx).to_numpy().T
    rhof_zx = pd.DataFrame(rhof_zx).to_numpy().T
    rhob_zx = pd.DataFrame(rhob_zx).to_numpy().T

    Km_zx = pd.DataFrame(Km_zx).to_numpy().T
    mum_zx = pd.DataFrame(mum_zx).to_numpy().T

    Vs_zx = pd.DataFrame(Vs_zx).to_numpy().T
    Vp_zx = pd.DataFrame(Vp_zx).to_numpy().T
    disp_db = pd.DataFrame(disp_db).to_numpy()

    # zs
    np.savetxt(f"{paths.output}/{model_id}/{profile}/z_zx.txt", z_zx, fmt="%.2f")

    # max_depth
    with open(f"{paths.output}/{model_id}/{profile}/max_z.txt", "w") as f:
        f.write(f"{-data_params['max_depth']}")

    # xs
    np.savetxt(f"{paths.output}/{model_id}/{profile}/xs.txt", xs.reshape(1, len(xs)), fmt="%.3f")

    # xmids
    np.savetxt(f"{paths.output}/{model_id}/{profile}/xmids.txt", xmids.reshape(1, len(xmids)), fmt="%.3f")

    # freqs
    np.savetxt(f"{paths.output}/{model_id}/{profile}/fs.txt", disp_db[0][0][:, 0], fmt="%.2f")

    # GMs
    np.savetxt(f"{paths.output}/{model_id}/{profile}/soil_zx.txt", soil_zx, fmt="%s")

    # thicks
    np.savetxt(f"{paths.output}/{model_id}/{profile}/thick_zx.txt", thick_zx, fmt="%.2f")

    # Ns
    np.savetxt(f"{paths.output}/{model_id}/{profile}/N_zx.txt", N_zx, fmt="%.2f")

    # WT
    np.savetxt(f"{paths.output}/{model_id}/{profile}/WT_x.txt", WT_x, fmt="%.2f")

    # h
    np.savetxt(f"{paths.output}/{model_id}/{profile}/h_zx.txt", h_zx, fmt="%.2f")

    # Sw
    np.savetxt(f"{paths.output}/{model_id}/{profile}/Sw_zx.txt", Sw_zx, fmt="%.2f")

    # Swe
    np.savetxt(f"{paths.output}/{model_id}/{profile}/Swe_zx.txt", Swe_zx, fmt="%.2f")

    # mus
    np.savetxt(f"{paths.output}/{model_id}/{profile}/mus_zx.txt", mus_zx, fmt="%.2f")

    # Ks
    np.savetxt(f"{paths.output}/{model_id}/{profile}/Ks_zx.txt", Ks_zx, fmt="%.2f")

    # rhos
    np.savetxt(f"{paths.output}/{model_id}/{profile}/rhos_zx.txt", rhos_zx, fmt="%.2f")

    # nus
    np.savetxt(f"{paths.output}/{model_id}/{profile}/nus_zx.txt", nus_zx, fmt="%.2f")

    # Kf
    np.savetxt(f"{paths.output}/{model_id}/{profile}/Kf_zx.txt", Kf_zx, fmt="%.2f")

    # rhof
    np.savetxt(f"{paths.output}/{model_id}/{profile}/rhof_zx.txt", rhof_zx, fmt="%.2f")

    # rhob
    np.savetxt(f"{paths.output}/{model_id}/{profile}/rhob_zx.txt", rhob_zx, fmt="%.2f")

    # Km
    np.savetxt(f"{paths.output}/{model_id}/{profile}/Km_zx.txt", Km_zx, fmt="%.2f")

    # mum
    np.savetxt(f"{paths.output}/{model_id}/{profile}/mum_zx.txt", mum_zx, fmt="%.2f")

    # Vs
    np.savetxt(f"{paths.output}/{model_id}/{profile}/Vs_zx.txt", Vs_zx, fmt="%.2f")

    # Vp
    np.savetxt(f"{paths.output}/{model_id}/{profile}/Vp_zx.txt", Vp_zx, fmt="%.2f")

    # disp
    Vr_fx = []
    for disp in disp_db:
        Vr_fx.append(disp[0][:, 1])
    Vr_fx = np.array(Vr_fx).T
    np.savetxt(f"{paths.output}/{model_id}/{profile}/Vr_fx.txt", Vr_fx, fmt="%.2f")
    ### -------------------------------------------------------------------------------------------

    ### RMS ---------------------------------------------------------------------------------------
    with open(f"{paths.output}/{model_id}/{profile}/DCs-rms.txt", "w") as f:
        f.write(f"Model ID: {model_id}\n\n")
        f.write(f"Profile: {profile}\n\n")
        f.write("Average root mean square error on the dispersion curves\n\n")
        f.write(f"RMS: {np.mean(rms_x)} m/s\n")
        f.write(f"NRMS: {np.mean(rms_x) / (np.nanmax(Vr_fx) - np.nanmin(Vr_fx)) * 100} %\n")
    ### -------------------------------------------------------------------------------------------
