"""Grand_Est paper figures: 3D soil/N/mum cubes and channel slices (pyvista)
plus the 2D soil+N/entropy/mum/Vs summary panels, aggregated over one or
two date periods.

Site-specific, paper-panel-layout-specific -- light cleanup only
(folders.py/misc.py replaced by config.py/_legacy_utils.py), not held to
the same bar as the core pipeline.
Run from the repo root: python experiments/grand_est/plot_paper_figures.py

Author : Jose CUNHA TEIXEIRA
License : SNCF Reseau, UMR 7619 METIS, Sorbonne Universite
"""


import os
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import savgol_filter
from scipy.ndimage import generic_filter
from datetime import datetime
from json import load
from scipy.interpolate import interp1d, RegularGridInterpolator
from cmcrameri import cm


from _legacy_utils import mode_filter_count, mode_filter_mean
from silex.config import Paths

plt.rcParams.update({'font.size': 8})
CM = 1/2.54




### PARAMS ----------------------------------------------------------------------------------------
paths = Paths.from_env()
model_id = '[202407170928]'
periods = [['2022-07-01', '2022-07-31'],
           ['2023-07-01', '2023-07-31']] # From 2023-07-01 to 2023-07-31 and from 2024-07-01 to 2024-07-31
site = 'Grand_Est'
### -----------------------------------------------------------------------------------------------




### FORMATS ---------------------------------------------------------------------------------------
# Generation-time params, not part of checkpoint.py's trimmed inference-only
# silex_params.json -- see invert_qc.py, which reads this same
# training_data/<site>/params.json.
with open(f'{paths.input}/training_data/{site}/params.json', 'r') as f:
    data_params = load(f)
    dz_soil = data_params['d_thickness']
    dz_vel = data_params['dz']
    max_z = data_params['max_depth']
    soils = data_params['soils']
    Ns = data_params['Ns']
    WTs = data_params['WTs']
### -----------------------------------------------------------------------------------------------




### CMAPS -----------------------------------------------------------------------------------------
colors_soil = np.array([
    (79, 136, 183),
    (115, 186, 203),
    (178, 223, 200),
    (195, 164, 84),
    ]) / 255
cmap_soil = ListedColormap(colors_soil)


colors_N = np.array([
        (251, 249, 206),
        (239, 181, 93),
        (227, 114, 82),
        (146, 64, 62),
        (51, 35, 19),
    ]) / 255
cmap_N = ListedColormap(colors_N)

cmap_entropy = cm.romaO_r
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
if len(periods) == 1:
    fig_soils_Ns, axs_soils_Ns = plt.subplots(5, 2, figsize=(18.4*CM, 28/2*CM), dpi=600, gridspec_kw={'hspace': 0.5, 'wspace': 0.1})
    fig_mums, axs_mums = plt.subplots(5, 1, figsize=(18.4/2.1*CM, 13*CM), dpi=600, gridspec_kw={'hspace': 0.5, 'wspace': 0.1})
    fig_VSs, axs_VSs = plt.subplots(5, 2, figsize=(18.4*CM, 28/2*CM), dpi=600, gridspec_kw={'hspace': 0.5, 'wspace': 0.1})
    fig_soil_N_entropy, axs_soil_N_entropy = plt.subplots(5, 1, figsize=(18.4/2.1*CM, 13*CM), dpi=600, gridspec_kw={'hspace': 0.5, 'wspace': 0.1})
elif len(periods) == 2:
    fig_soils_Ns, axs_soils_Ns = plt.subplots(11, 2, figsize=(18.4*CM, 28*CM), dpi=600, gridspec_kw={'hspace': 0.5, 'wspace': 0.1})
    fig_mums, axs_mums = plt.subplots(5, 2, figsize=(18.4*CM, 13*CM), dpi=600, gridspec_kw={'hspace': 0.5, 'wspace': 0.1})
    fig_VSs, axs_VSs = plt.subplots(11, 2, figsize=(18.4*CM, 28*CM), dpi=600, gridspec_kw={'hspace': 0.5, 'wspace': 0.1})
    fig_soil_N_entropy, axs_soil_N_entropy = plt.subplots(5, 2, figsize=(18.4*CM, 13*CM), dpi=600, gridspec_kw={'hspace': 0.5, 'wspace': 0.1})

existing_results = sorted(os.listdir(f'{paths.output}/{model_id}/{site}/'))
if 'results' in existing_results:
    existing_results.remove('results')

if not os.path.exists(f"{paths.output}/{model_id}/{site}/results/"):
    os.makedirs(f"{paths.output}/{model_id}/{site}/results/")

if len(periods) == 1:
    name = f"[{periods[0][0]}_{periods[0][-1]}]"
elif len(periods) == 2:
    name = f"[{periods[0][0]}_{periods[0][-1]}]_[{periods[1][0]}_{periods[1][-1]}]"
else:
    raise ValueError('Only 1 or 2 periods are allowed.')
    
if not os.path.exists(f"{paths.output}/{model_id}/{site}/results/{name}/"):
    os.makedirs(f"{paths.output}/{model_id}/{site}/results/{name}/")
    
profiles = ['P1', 'P2', 'P3', 'P4', 'P5']
Nprofiles = len(profiles)


nums = []

for i_period, period in enumerate(periods):

    Nz_soil = 5000
    Nz_vel = 500

    if i_period == 0:
        shift = 0
    else:
        shift = 1
    
    dates = pd.date_range(start=datetime.strptime(period[0], '%Y-%m-%d'), end=datetime.strptime(period[1], '%Y-%m-%d'), freq='D')
    dates = dates[dates.isin(existing_results)]
    N_dates = len(dates)
    
    xs = np.loadtxt(f"{paths.output}/{model_id}/{site}/{dates[0].strftime('%Y-%m-%d')}/{profiles[0]}/xs.txt")
    Nx = len(xs)

    soil_xyzt = np.full((Nx, Nprofiles, Nz_soil, N_dates), np.nan)
    N_xyzt = np.full((Nx, Nprofiles, Nz_soil, N_dates), np.nan)
    WT_xyt = np.full((Nx, Nprofiles, N_dates), np.nan)
    mum_xyzt = np.full((Nx, Nprofiles, Nz_vel, N_dates), np.nan)
    Vs_xyzt = np.full((Nx, Nprofiles, Nz_vel, N_dates), np.nan)

    for i_date, date in enumerate(dates):

        for i_profile, profile in enumerate(profiles):

            path = f"{paths.output}/{model_id}/{site}/{date.strftime('%Y-%m-%d')}/{profile}/"
            if not os.path.exists(path):
                SystemError(f'ERROR: {path} does not exist.')

            z_zx = np.loadtxt(f'{path}/z_zx.txt')
            zs_lengths_x = [len(zs) - np.isnan(zs).sum() for zs in z_zx.T]
            longest_col_index = np.argmax(zs_lengths_x)
            zs = z_zx[:, longest_col_index]

            # max_z = np.loadtxt(f'{path}/max_z.txt')

            xs = np.loadtxt(f'{path}/xs.txt')

            x_min = np.min(xs)
            x_max = np.max(xs)

            fs = np.loadtxt(f'{path}/fs.txt')

            thick_zx = np.loadtxt(f'{path}/thick_zx.txt')

            soil_zx = np.loadtxt(f'{path}/soil_zx.txt', dtype=str)
            soil_int_zx = []

            soil_to_int = {'None': 0}
            for i, soil in enumerate(soils):
                soil_to_int[soil] = i+1
            int_to_soil = {val:key for key, val in soil_to_int.items()}

            for i, (soil_z, thick_z) in enumerate(zip(soil_zx.T, thick_zx.T)):
                log = []
                for soil, thick in zip(soil_z, thick_z):
                    if soil != 'None':
                        log.extend([soil_to_int[soil]]*int(thick/dz_soil))
                soil_int_zx.append(log)
            soil_int_zx = pd.DataFrame(soil_int_zx)
            soil_int_zx = soil_int_zx.fillna(0)
            soil_int_zx = soil_int_zx.to_numpy().T

            thick_zx = np.loadtxt(f'{path}/thick_zx.txt')

            N_zx = np.loadtxt(f'{path}/N_zx.txt')
            N_int_zx = []
            for i, (N_z, thick_z) in enumerate(zip(N_zx.T, thick_zx.T)):
                log = []
                for N, thick in zip(N_z, thick_z):
                    if ~ np.isnan(N):
                        log.extend([int(N)]*int(thick/dz_soil))
                N_int_zx.append(log)
            N_int_zx = pd.DataFrame(N_int_zx)
            N_int_zx = N_int_zx.to_numpy().T

            WT_x = -np.loadtxt(f'{path}/WT_x.txt')

            mum_zx = np.loadtxt(f'{path}/mum_zx.txt')

            Vs_zx = np.loadtxt(f'{path}/Vs_zx.txt')


            soil_xyzt[:, i_profile, 0:soil_int_zx.shape[0], i_date] = soil_int_zx.T
            N_xyzt[:, i_profile, 0:N_int_zx.shape[0], i_date] = N_int_zx.T
            WT_xyt[:, i_profile, i_date] = WT_x
            mum_xyzt[:, i_profile, 0:mum_zx.shape[0], i_date] = mum_zx.T
            Vs_xyzt[:, i_profile, 0:Vs_zx.shape[0], i_date] = Vs_zx.T

    
    soil_xyzt = soil_xyzt[:, : , 0:int(max_z/dz_soil ), :]
    N_xyzt = N_xyzt[:, :, 0:int(max_z/dz_soil ), :]
    mum_xyzt = mum_xyzt[:, :, 0:int(max_z/dz_vel) , :]
    Vs_xyzt = Vs_xyzt[:, :, 0:int(max_z/dz_vel) , :]
    Nz_soil = soil_xyzt.shape[2]
    Nz_vel = mum_xyzt.shape[2]

    soil_xyz = np.full(soil_xyzt.shape[0:3], np.nan)
    N_xyz = np.full(N_xyzt.shape[0:3], np.nan)
    WT_xy = np.full(WT_xyt.shape[0:2], np.nan)
    mum_xyz = np.full(mum_xyzt.shape[0:3], np.nan)
    Vs_xyz = np.full(mum_xyzt.shape[0:3], np.nan)
   
    
    soil_int_zx = np.where(np.isnan(soil_int_zx), 0, soil_int_zx)
    smoothed_soil_int_zx = generic_filter(soil_int_zx, mode_filter_count, size=(1,3))
    smoothed_soil_int_zx = generic_filter(smoothed_soil_int_zx, mode_filter_count, size=(1,3))
    soil_int_zx = np.where(soil_int_zx == 0, np.nan, soil_int_zx)
    smoothed_soil_int_zx = np.where(smoothed_soil_int_zx == 0, np.nan, smoothed_soil_int_zx) 
    
    
    
    # Entropy computing
    comb_to_int = {}
    i_comb = 1
    for soil in list(soil_to_int.values())[1:]:
        for N in Ns:
            for WT in WTs:
                comb_to_int[(soil, N, -WT)] = i_comb
                i_comb += 1
            
    soil_N_xyzp = np.full((soil_xyzt.shape[0], soil_xyzt.shape[1], soil_xyzt.shape[2], len(comb_to_int)), 0.0)
    soil_N_entropy_xyz = np.full((soil_xyzt.shape[0], soil_xyzt.shape[1], soil_xyzt.shape[2]), np.nan)
    
    for i_profile in range(Nprofiles):
        for i_x in range(Nx):
            
            WT_t = WT_xyt[i_x, i_profile, :]
            
            for i_z in range(Nz_soil):
                
                soil_t = soil_xyzt[i_x, i_profile, i_z, :]
                N_t = N_xyzt[i_x, i_profile, i_z, :]
                
                soil_N_t = [(soil,N,WT) for soil, N, WT in zip(soil_t, N_t, WT_t) if soil!=0 and not np.isnan(N)]
                soil_N_t_comb = [comb_to_int[soil_N] for soil_N in soil_N_t]

                unique, counts = np.unique(soil_N_t_comb, return_counts=True)
                dic = dict(zip(unique, counts))
                total_counts = np.sum(counts)
                for i_key, key in enumerate(comb_to_int.values()):
                    if key in dic.keys():
                        soil_N_xyzp[i_x, i_profile, i_z, i_key] = dic[key]/total_counts
                
                soil_N_entropy_xyz[i_x, i_profile, i_z] = -np.sum([p*np.log2(p) for p in soil_N_xyzp[i_x, i_profile, i_z, :] if p != 0])
                
    soil_N_entropy_xyz /= np.log2(len(comb_to_int))
    
    
    
    # Temporal smoothing
    for i_profile in range(Nprofiles):
        for i_x in range(Nx):
            for i_z in range(Nz_soil):
                soil_t = soil_xyzt[i_x, i_profile, i_z, :]
                soil_t = soil_t[~np.isnan(soil_t)]
                if soil_t.size != 0:
                    soil_xyz[i_x, i_profile, i_z] = mode_filter_count(soil_t)
                else:
                    soil_xyz[i_x, i_profile, i_z] = 0

    for i_profile in range(Nprofiles):
        for i_x in range(Nx):
            for i_z in range(Nz_soil):
                N_t = N_xyzt[i_x, i_profile, i_z, :]
                N_t = N_t[~np.isnan(N_t)]
                if N_t.size != 0:
                    N_xyz[i_x, i_profile, i_z] = mode_filter_count(N_t)
                else:
                    N_xyz[i_x, i_profile, i_z] = 0

    for i_profile in range(Nprofiles):
        for i_x in range(Nx):
            WT_t = WT_xyt[i_x, i_profile, :]
            WT_xy[i_x, i_profile] = np.mean(WT_t)

    for i_profile in range(Nprofiles):
        for i_x in range(Nx):
            for i_z in range(Nz_vel):
                mum_t = mum_xyzt[i_x, i_profile, i_z, :]
                mum_t = mum_t[~np.isnan(mum_t)]
                if mum_t.size != 0:
                    mum_xyz[i_x, i_profile, i_z] = np.mean(mum_t)
                else:
                    mum_xyz[i_x, i_profile, i_z] = 0

    for i_profile in range(Nprofiles):
        for i_x in range(Nx):
            for i_z in range(Nz_vel):
                Vs_t = Vs_xyzt[i_x, i_profile, i_z, :]
                Vs_t = Vs_t[~np.isnan(Vs_t)]
                if Vs_t.size != 0:
                    Vs_xyz[i_x, i_profile, i_z] = np.mean(Vs_t)
                else:
                    Vs_xyz[i_x, i_profile, i_z] = 0



    # Small spatial smoothing over x
    for i_profile in range(Nprofiles):
        soil_xz = soil_xyz[:, i_profile, :]
        soil_xz = generic_filter(soil_xz, mode_filter_count, size=(3,1))
        soil_xz = generic_filter(soil_xz, mode_filter_count, size=(3,1))
        soil_xyz[:, i_profile, :] = soil_xz

    for i_profile in range(Nprofiles):
        N_xz = N_xyz[:, i_profile, :]
        N_xz = generic_filter(N_xz, mode_filter_count, size=(3,1))
        N_xz = generic_filter(N_xz, mode_filter_count, size=(3,1))
        N_xyz[:, i_profile, :] = N_xz

    for i_profile in range(Nprofiles):
        if (Nx/4) % 2 == 0:
            wl = Nx/4 + 1
        else:
            wl = Nx/4
        WT_xy[:, i_profile] = savgol_filter(WT_xy[:, i_profile], window_length=wl, polyorder=2, mode="nearest")

    for i_profile in range(Nprofiles):
        mum_xz = mum_xyz[:, i_profile, :]
        mum_xz = generic_filter(mum_xz, mode_filter_mean, size=(3,3))
        mum_xz = generic_filter(mum_xz, mode_filter_mean, size=(3,3))
        mum_xyz[:, i_profile, :] = mum_xz

    for i_profile in range(Nprofiles):
        Vs_xz = Vs_xyz[:, i_profile, :]
        Vs_xz = generic_filter(Vs_xz, mode_filter_mean, size=(3,3))
        Vs_xz = generic_filter(Vs_xz, mode_filter_mean, size=(3,3))
        Vs_xyz[:, i_profile, :] = Vs_xz


    soil_xyz = np.where(soil_xyz == 0, np.nan, soil_xyz)
    N_xyz = np.where(N_xyz == 0, np.nan, N_xyz)
    
    
    
    
    
    
    for i_profile, profile in enumerate(profiles):

        # Plot soil and N
        ax_soil_N = axs_soils_Ns[i_profile+(5*i_period)+shift, :]
        zs = np.linspace(0, -Nz_soil*dz_soil, Nz_soil)
        im1 = ax_soil_N[0].pcolormesh(xs, zs, soil_xyz[:, i_profile, :].T, cmap=cmap_soil, vmin=1, vmax=len(int_to_soil)-1, rasterized=True)
        ax_soil_N[0].plot(xs, WT_xy[:, i_profile], color='k', linewidth=1, linestyle='--')
        ax_soil_N[0].set_ylim([-max_z, 0])
        ax_soil_N[0].set_ylabel('Depth [m]')
        ax_soil_N[0].set_yticks([0, -10, -20])
        ax_soil_N[0].set_yticklabels([0, -10, -20])
        ax_soil_N[0].set_xticks([0, 25, 50, 75, 100, 125])
        ax_soil_N[0].set_title(fr'$\mathbf{{L_{{{i_profile+1}}}}}$', rotation='vertical', x=-0.22, y=0.2, fontsize=8)
        ax_soil_N[0].minorticks_on()
        if i_profile == 0:
            ax_soil_N[0].set_xlabel('x [m]', labelpad=0)
            ax_soil_N[0].set_xticklabels(['SW\n0', 25, 50, 75, 100, 'NE\n125'])
            ax_soil_N[0].xaxis.set_label_position('top')
            ax_soil_N[0].tick_params(which='both', labeltop=True, labelbottom=False, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
        elif i_profile == 4:
            ax_soil_N[0].set_xticklabels(['$G_{1}$', '', '', '', '', '$G_{42}$'])
            ax_soil_N[0].xaxis.set_label_position('bottom')
            ax_soil_N[0].tick_params(which='both', labeltop=False, labelbottom=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
        else:
            ax_soil_N[0].tick_params(which='both', labeltop=False, labelbottom=False, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
        im2 = ax_soil_N[1].pcolormesh(xs, zs, N_xyz[:, i_profile, :].T, cmap=cmap_N, vmin=6, vmax=10, rasterized=True)
        ax_soil_N[1].plot(xs, WT_xy[:, i_profile], color='k', linewidth=1, linestyle='--')
        ax_soil_N[1].tick_params(which='both', labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
        ax_soil_N[1].set_ylim([-max_z, 0])
        ax_soil_N[1].set_ylabel('')
        ax_soil_N[1].set_yticks([0, -10, -20])
        ax_soil_N[1].set_xticks([0, 25, 50, 75, 100, 125])
        ax_soil_N[1].minorticks_on()
        if i_profile == 0:
            ax_soil_N[1].set_xlabel('x [m]', labelpad=0)
            ax_soil_N[1].set_xticklabels(['SW\n0', 25, 50, 75, 100, 'NE\n125'])
            ax_soil_N[1].xaxis.set_label_position('top')
            ax_soil_N[1].tick_params(which='both', labeltop=True, labelbottom=False, labelleft=False, labelright=False, bottom=True, top=True, left=True, right=True)
        elif i_profile == 4:
            ax_soil_N[1].set_xticklabels(['$G_{1}$', '', '', '', '', '$G_{42}$'])
            ax_soil_N[1].xaxis.set_label_position('bottom')
            ax_soil_N[1].tick_params(which='both', labeltop=False, labelbottom=True, labelleft=False, labelright=False, bottom=True, top=True, left=True, right=True)
        else:
            ax_soil_N[1].tick_params(which='both', labeltop=False, labelbottom=False, labelleft=False, labelright=False, bottom=True, top=True, left=True, right=True)
            
            
        
        # Plot entropies
        if len(periods) == 1:
            ax_soil_N_entropy = axs_soil_N_entropy[i_profile]
        elif len(periods) == 2:
            ax_soil_N_entropy = axs_soil_N_entropy[i_profile, i_period]
        zs = np.linspace(0, -Nz_soil*dz_soil, Nz_soil)
        im8 =  ax_soil_N_entropy.pcolormesh(xs, zs, soil_N_entropy_xyz[:, i_profile, :].T, cmap=cmap_entropy, vmin=0, vmax=1, rasterized=True)
        ax_soil_N_entropy.set_ylim([-max_z, 0])
        ax_soil_N_entropy.set_yticks([0, -10, -20])
        ax_soil_N_entropy.set_xticks([0, 25, 50, 75, 100, 125])
        ax_soil_N_entropy.minorticks_on()
        if i_period == 0:
            ax_soil_N_entropy.set_ylabel('Depth [m]')
            ax_soil_N_entropy.set_yticklabels([0, -10, -20])
            ax_soil_N_entropy.set_title(fr'$\mathbf{{L_{{{i_profile+1}}}}}$', rotation='vertical', x=-0.22, y=0.2, fontsize=8)
            if i_profile == 0:
                ax_soil_N_entropy.set_xlabel('x [m]', labelpad=0)
                ax_soil_N_entropy.set_xticklabels(['SW\n0', 25, 50, 75, 100, 'NE\n125'])
                ax_soil_N_entropy.xaxis.set_label_position('top')
                ax_soil_N_entropy.tick_params(which='both', labeltop=True, labelbottom=False, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
            elif i_profile == 4:
                ax_soil_N_entropy.set_xticklabels(['$G_{1}$', '', '', '', '', '$G_{42}$'])
                ax_soil_N_entropy.xaxis.set_label_position('bottom')
                ax_soil_N_entropy.tick_params(which='both', labeltop=False, labelbottom=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
            else:
                ax_soil_N_entropy.tick_params(which='both', labeltop=False, labelbottom=False, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
        else:
            if i_profile == 0:
                ax_soil_N_entropy.set_xlabel('x [m]', labelpad=0)
                ax_soil_N_entropy.set_xticklabels(['SW\n0', 25, 50, 75, 100, 'NE\n125'])
                ax_soil_N_entropy.xaxis.set_label_position('top')
                ax_soil_N_entropy.tick_params(which='both', labeltop=True, labelbottom=False, labelleft=False, labelright=False, bottom=True, top=True, left=True, right=True)
            elif i_profile == 4:
                ax_soil_N_entropy.set_xticklabels(['$G_{1}$', '', '', '', '', '$G_{42}$'])
                ax_soil_N_entropy.xaxis.set_label_position('bottom')
                ax_soil_N_entropy.tick_params(which='both', labeltop=False, labelbottom=True, labelleft=False, labelright=False, bottom=True, top=True, left=True, right=True)
            else:
                ax_soil_N_entropy.tick_params(which='both', labeltop=False, labelbottom=False, labelleft=False, labelright=False, bottom=True, top=True, left=True, right=True)



        # Plot mum
        if len(periods) == 1:
            ax_mums = axs_mums[i_profile]
        elif len(periods) == 2:
            ax_mums = axs_mums[i_profile, i_period]
        zs = np.linspace(0, -Nz_vel*dz_vel, Nz_vel)
        cmap = 'terrain'
        im3 = ax_mums.pcolormesh(xs, zs, mum_xyz[:, i_profile, :].T/1_000_000_000, cmap=cmap, vmin=0.1, vmax=0.4, rasterized=True)
        ax_mums.set_ylim([-max_z, 0])
        ax_mums.set_yticks([0, -10, -20])
        ax_mums.set_xticks([0, 25, 50, 75, 100, 125])
        ax_mums.minorticks_on()
        if i_period == 0:
            ax_mums.set_ylabel('Depth [m]')
            ax_mums.set_yticklabels([0, -10, -20])
            ax_mums.set_title(fr'$\mathbf{{L_{{{i_profile+1}}}}}$', rotation='vertical', x=-0.22, y=0.2, fontsize=8)
            if i_profile == 0:
                ax_mums.set_xlabel('x [m]', labelpad=0)
                ax_mums.set_xticklabels(['SW\n0', 25, 50, 75, 100, 'NE\n125'])
                ax_mums.xaxis.set_label_position('top')
                ax_mums.tick_params(which='both', labeltop=True, labelbottom=False, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
            elif i_profile == 4:
                ax_mums.set_xticklabels(['$G_{1}$', '', '', '', '', '$G_{42}$'])
                ax_mums.xaxis.set_label_position('bottom')
                ax_mums.tick_params(which='both', labeltop=False, labelbottom=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
            else:
                ax_mums.tick_params(which='both', labeltop=False, labelbottom=False, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
        else:
            if i_profile == 0:
                ax_mums.set_xlabel('x [m]', labelpad=0)
                ax_mums.set_xticklabels(['SW\n0', 25, 50, 75, 100, 'NE\n125'])
                ax_mums.xaxis.set_label_position('top')
                ax_mums.tick_params(which='both', labeltop=True, labelbottom=False, labelleft=False, labelright=False, bottom=True, top=True, left=True, right=True)
            elif i_profile == 4:
                ax_mums.set_xticklabels(['$G_{1}$', '', '', '', '', '$G_{42}$'])
                ax_mums.xaxis.set_label_position('bottom')
                ax_mums.tick_params(which='both', labeltop=False, labelbottom=True, labelleft=False, labelright=False, bottom=True, top=True, left=True, right=True)
            else:
                ax_mums.tick_params(which='both', labeltop=False, labelbottom=False, labelleft=False, labelright=False, bottom=True, top=True, left=True, right=True)
                


        # Plot Vs
        ax_VS = axs_VSs[i_profile+(5*i_period)+shift, :]
        zs = np.linspace(0, -Nz_vel*dz_vel, Nz_vel)
        cmap = 'terrain'
        im4 = ax_VS[0].pcolormesh(xs, zs, Vs_xyz[:, i_profile, :].T, cmap=cmap, vmin=200, vmax=600, rasterized=True)
        ax_VS[0].set_ylim([-max_z, 0])
        ax_VS[0].set_ylabel('Depth [m]')
        ax_VS[0].set_yticks([0, -10, -20])
        ax_VS[0].set_yticklabels([0, -10, -20])
        ax_VS[0].set_xticks([0, 25, 50, 75, 100, 125])
        ax_VS[0].set_title(fr'$\mathbf{{L_{{{i_profile+1}}}}}$', rotation='vertical', x=-0.22, y=0.2, fontsize=8)
        ax_VS[0].minorticks_on()
        if i_profile == 0:
            ax_VS[0].set_xlabel('x [m]', labelpad=0)
            ax_VS[0].set_xticklabels(['SW\n0', 25, 50, 75, 100, 'NE\n125'])
            ax_VS[0].xaxis.set_label_position('top')
            ax_VS[0].tick_params(which='both', labeltop=True, labelbottom=False, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
        elif i_profile == 4:
            ax_VS[0].set_xticklabels(['$G_{1}$', '', '', '', '', '$G_{42}$'])
            ax_VS[0].xaxis.set_label_position('bottom')
            ax_VS[0].tick_params(which='both', labeltop=False, labelbottom=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
        else:
            ax_VS[0].tick_params(which='both', labeltop=False, labelbottom=False, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
        cmap = 'terrain'
        im5 = ax_VS[1].pcolormesh(xs, zs, np.full_like(Vs_xyz[:, i_profile, :].T, np.nan), cmap=cmap, vmin=200, vmax=600, rasterized=True)
        ax_VS[1].tick_params(which='both', labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
        ax_VS[1].set_ylim([-max_z, 0])
        ax_VS[1].set_ylabel('')
        ax_VS[1].set_yticks([0, -10, -20])
        ax_VS[1].set_xticks([0, 25, 50, 75, 100, 125])
        ax_VS[1].minorticks_on()
        if i_profile == 0:
            ax_VS[1].set_xlabel('x [m]', labelpad=0)
            ax_VS[1].set_xticklabels(['SW\n0', 25, 50, 75, 100, 'NE\n125'])
            ax_VS[1].xaxis.set_label_position('top')
            ax_VS[1].tick_params(which='both', labeltop=True, labelbottom=False, labelleft=False, labelright=False, bottom=True, top=True, left=True, right=True)
        elif i_profile == 4:
            ax_VS[1].set_xticklabels(['$G_{1}$', '', '', '', '', '$G_{42}$'])
            ax_VS[1].xaxis.set_label_position('bottom')
            ax_VS[1].tick_params(which='both', labeltop=False, labelbottom=True, labelleft=False, labelright=False, bottom=True, top=True, left=True, right=True)
        else:
            ax_VS[1].tick_params(which='both', labeltop=False, labelbottom=False, labelleft=False, labelright=False, bottom=True, top=True, left=True, right=True)
 
 

    soil_xyz = soil_xyz[:, :, ::-1]
      
    dpi = 300
    width_inches = 16
    height_inches = 9
    window_size = (width_inches * dpi, height_inches * dpi)
       
    data = pv.ImageData()
    data.dimensions = np.array(soil_xyz.shape) + 1
    data.origin = (0, 0, -20)
    data.spacing = (3.0, 5, dz_soil)
    data.cell_data["Soil type"] = soil_xyz.flatten(order="F")
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(data, scalars="Soil type", cmap=cmap_soil, edge_color='k', line_width=1, clim=[1, 4])
    plotter.camera_position = [
        (176.65843001947934, -93.5849138205549, 42.515273570861936),
        (50.83957070484356, 32.233945494080956, -32.97604201791967),
        (0.0, 0.0, 1.0)
    ]
    light = pv.Light(intensity=0.75)
    light.set_camera_light()
    plotter.add_light(light)
    plotter.show_grid(n_ylabels=6, n_xlabels=2, n_zlabels=5, axes_ranges=[0, 129, 0, 25, -20, 0], bounds=[0, 129, 0, 25, -20, 0])
    plotter.show_axes()
    plotter.set_background(None, top=None)
    plotter.save_graphic(f'{paths.output}/{model_id}/{site}/results/{name}/[{period[0]}_{period[-1]}]_soil_cube.svg')
    
    for i_soil, soil in enumerate(soils):
        channels = data.threshold([i_soil+1, i_soil+1])
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.add_mesh(channels, scalars="Soil type", cmap=ListedColormap(colors_soil[i_soil:]), edge_color='k', line_width=1)
        plotter.camera_position = [
            (176.65843001947934, -93.5849138205549, 42.515273570861936),
            (50.83957070484356, 32.233945494080956, -32.97604201791967),
            (0.0, 0.0, 1.0)
        ]
        light = pv.Light(intensity=0.75)
        light.set_camera_light()
        plotter.add_light(light)
        plotter.show_grid(n_ylabels=6, n_xlabels=2, n_zlabels=5, axes_ranges=[0, 129, 0, 25, -20, 0], bounds=[0, 129, 0, 25, -20, 0])
        plotter.show_axes()
        plotter.save_graphic(f'{paths.output}/{model_id}/{site}/results/{name}/[{period[0]}_{period[-1]}]_soil_channel_{[soil]}.svg')
    
    # slices = data.slice_orthogonal()
    slices = data.slice_along_axis(n=5, axis="y")
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(slices, scalars="Soil type", cmap=cmap_soil, edge_color='k', line_width=1, clim=[1, 4])
    
    
    d = 0.05
    for i_profile, profile in enumerate(profiles):
        new_xs = np.arange(np.min(xs), np.max(xs)+d, d)
        interfunc = interp1d(xs, WT_xy[:, i_profile], kind='cubic')
        new_WT = interfunc(new_xs)
        new_WT = np.pad(new_WT, (int(1.5/d), int(1.5/d)), mode='constant', constant_values=(np.nan, np.nan))
        
        chunk_size = 10
        for start in range(chunk_size, len(new_WT), 2 * chunk_size):
            new_WT[start:start + chunk_size] = np.nan
            
        new_xs = np.arange(0, 129+d, d)
        points = np.c_[new_xs, np.full_like(new_xs, i_profile*5+0.2+1*i_profile), new_WT]
        curve = pv.PolyData(points)
        plotter.add_mesh(curve, color='black', line_width=10)    
    
    
    plotter.camera_position = [
        (176.65843001947934, -93.5849138205549, 42.515273570861936),
        (50.83957070484356, 32.233945494080956, -32.97604201791967),
        (0.0, 0.0, 1.0)
    ]
    light = pv.Light(intensity=0.75)
    light.set_camera_light()
    plotter.add_light(light)
    plotter.show_grid(n_ylabels=5, n_xlabels=2, n_zlabels=5, axes_ranges=[0, 129, 0, 20, -20, 0])
    plotter.show_axes()
    plotter.save_graphic(f'{paths.output}/{model_id}/{site}/results/{name}/[{period[0]}_{period[-1]}]_soil_slices.svg')
    
    
    
    N_xyz = N_xyz[:, :, ::-1]
    
    colors = ["tab:blue", "tab:cyan", "tab:green", "tab:orange", "tab:red"]
    
    dpi = 300
    width_inches = 16
    height_inches = 9
    window_size = (width_inches * dpi, height_inches * dpi)
       
    data = pv.ImageData()
    data.dimensions = np.array(N_xyz.shape) + 1
    data.origin = (0, 0, -20)
    data.spacing = (3, 5, dz_soil)
    data.cell_data["N"] = N_xyz.flatten(order="F")
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(data, scalars="N", cmap=cmap_N, edge_color='k', line_width=1, clim=[6, 10])
    plotter.camera_position = [
        (176.65843001947934, -93.5849138205549, 42.515273570861936),
        (50.83957070484356, 32.233945494080956, -32.97604201791967),
        (0.0, 0.0, 1.0)
    ]
    light = pv.Light(intensity=0.75)
    light.set_camera_light()
    plotter.add_light(light)
    plotter.show_grid(n_ylabels=6, n_xlabels=2, n_zlabels=5, axes_ranges=[0, 129, 0, 25, -20, 0], bounds=[0, 129, 0, 25, -20, 0])
    plotter.show_axes()
    plotter.save_graphic(f'{paths.output}/{model_id}/{site}/results/{name}/[{period[0]}_{period[-1]}]_N_cube.svg')
    
    for i_N, N in enumerate(Ns):
        channels = data.threshold([N, N])
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.add_mesh(channels, scalars="N", cmap=ListedColormap(colors_N[i_N]), edge_color='k', line_width=1)
        plotter.camera_position = [
            (176.65843001947934, -93.5849138205549, 42.515273570861936),
            (50.83957070484356, 32.233945494080956, -32.97604201791967),
            (0.0, 0.0, 1.0)
        ]
        light = pv.Light(intensity=0.75)
        light.set_camera_light()
        plotter.add_light(light)
        plotter.show_grid(n_ylabels=6, n_xlabels=2, n_zlabels=5, axes_ranges=[0, 129, 0, 25, -20, 0], bounds=[0, 129, 0, 25, -20, 0])
        plotter.show_axes()
        plotter.save_graphic(f'{paths.output}/{model_id}/{site}/results/{name}/[{period[0]}_{period[-1]}]_N_channel_[{N}].svg')
    
    slices = data.slice_along_axis(n=5, axis="y")
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(slices, scalars="N", cmap=cmap_N, edge_color='k', line_width=1, clim=[6, 10])
    
    d = 0.05
    for i_profile, profile in enumerate(profiles):
        new_xs = np.arange(np.min(xs), np.max(xs)+d, d)
        interfunc = interp1d(xs, WT_xy[:, i_profile], kind='cubic')
        new_WT = interfunc(new_xs)
        new_WT = np.pad(new_WT, (int(1.5/d), int(1.5/d)), mode='constant', constant_values=(np.nan, np.nan))
        
        chunk_size = 10  # Number of consecutive values to replace
        for start in range(chunk_size, len(new_WT), 2 * chunk_size):
            new_WT[start:start + chunk_size] = np.nan
            
        new_xs = np.arange(0, 129+d, d)
        points = np.c_[new_xs, np.full_like(new_xs, i_profile*5+0.2+1*i_profile), new_WT]
        curve = pv.PolyData(points)
        plotter.add_mesh(curve, color='black', line_width=10)    
        
    plotter.camera_position = [
        (176.65843001947934, -93.5849138205549, 42.515273570861936),
        (50.83957070484356, 32.233945494080956, -32.97604201791967),
        (0.0, 0.0, 1.0)
    ]
    light = pv.Light(intensity=0.75)
    light.set_camera_light()
    plotter.add_light(light)
    plotter.show_grid(n_ylabels=5, n_xlabels=2, n_zlabels=5, axes_ranges=[0, 129, 0, 20, -20, 0])
    plotter.show_axes()
    plotter.save_graphic(f'{paths.output}/{model_id}/{site}/results/{name}/[{period[0]}_{period[-1]}]_N_slices.svg')
    
 
    mum_xyz = mum_xyz[:, :, ::-1]/1_000_000_000
    
    ys = np.arange(0, 19+4.75, 4.75)

    d = 0.1
    new_xs = np.arange(np.min(xs), np.max(xs)+d, d)
    new_ys = np.arange(np.min(ys), np.max(ys)+d, d)
    new_zs = np.arange(np.min(zs), np.max(zs)+d, d)
    new_xs_g, new_ys_g, new_zs_g = np.meshgrid(new_xs, new_ys, new_zs)
    new_points = (new_xs_g, new_ys_g, new_zs_g)
    
    interfunc = RegularGridInterpolator((xs, ys, zs), mum_xyz, method='cubic', bounds_error=False, fill_value=None)
    new_mum_xyz = interfunc(new_points)
    print('Interpolation done')
    
    new_mum_xyz = new_mum_xyz.swapaxes(0, 1)
    new_mum_xyz = np.flip(new_mum_xyz, axis=2)
    mum_xyz = new_mum_xyz
    
    nums.append(mum_xyz)
    
    data = pv.ImageData()
    data.dimensions = np.array(mum_xyz.shape) + 1
    data.origin = (0, 0, -20)
    data.spacing = (d, d, d)
    data.cell_data["mum"] = mum_xyz.flatten(order="F")
    cmap = 'terrain'
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(data, scalars="mum", cmap=cmap, clim=[0.1, 0.4])
    plotter.camera_position = [
        (176.65843001947934, -93.5849138205549, 42.515273570861936),
        (50.83957070484356, 32.233945494080956, -32.97604201791967),
        (0.0, 0.0, 1.0)
    ]
    light = pv.Light(intensity=0.75)
    light.set_camera_light()
    plotter.add_light(light)
    plotter.show_grid(n_ylabels=5, n_xlabels=2, n_zlabels=5, axes_ranges=[0, 126, 0, 19, -20, 0], bounds=[0, 126, 0, 19, -20, 0])
    plotter.show_axes()
    plotter.save_graphic(f'{paths.output}/{model_id}/{site}/results/{name}/[{period[0]}_{period[-1]}]_mum_cube.svg')
    
    min = 0.1
    max = 0.16
    channels = data.threshold([min, max])
    cmap = 'terrain'
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(channels, scalars="mum", cmap=cmap, clim=[0.1, 0.4])
    plotter.camera_position = [
        (176.65843001947934, -93.5849138205549, 42.515273570861936),
        (50.83957070484356, 32.233945494080956, -32.97604201791967),
        (0.0, 0.0, 1.0)
    ]
    light = pv.Light(intensity=0.75)
    light.set_camera_light()
    plotter.add_light(light)
    plotter.show_grid(n_ylabels=5, n_xlabels=2, n_zlabels=5, axes_ranges=[0, 126, 0, 19, -20, 0], bounds=[0, 126, 0, 19, -20, 0])
    plotter.show_axes()
    plotter.save_graphic(f'{paths.output}/{model_id}/{site}/results/{name}/[{period[0]}_{period[-1]}]_mum_channels_[{min}-{max}].svg')
    
    min = 0.27
    max = 0.4
    channels = data.threshold([min, max])
    cmap = 'terrain'
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(channels, scalars="mum", cmap=cmap, clim=[0.1, 0.4])
    plotter.camera_position = [
        (176.65843001947934, -93.5849138205549, 42.515273570861936),
        (50.83957070484356, 32.233945494080956, -32.97604201791967),
        (0.0, 0.0, 1.0)
    ]

    light = pv.Light(intensity=0.75)
    light.set_camera_light()
    plotter.add_light(light)
    plotter.show_grid(n_ylabels=5, n_xlabels=2, n_zlabels=5, axes_ranges=[0, 126, 0, 19, -20, 0], bounds=[0, 126, 0, 19, -20, 0])
    plotter.show_axes()
    plotter.save_graphic(f'{paths.output}/{model_id}/{site}/results/{name}/[{period[0]}_{period[-1]}]_mum_channels_[{min}-{max}].svg')
    
    slices = data.slice_along_axis(n=5, axis="y")
    cmap = 'terrain'
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(slices, scalars="mum", cmap=cmap, clim=[0.1, 0.4])
    plotter.camera_position = [
        (176.65843001947934, -93.5849138205549, 42.515273570861936),
        (50.83957070484356, 32.233945494080956, -32.97604201791967),
        (0.0, 0.0, 1.0)
    ]
    light = pv.Light(intensity=0.75)
    light.set_camera_light()
    plotter.add_light(light)
    plotter.show_grid(n_ylabels=5, n_xlabels=2, n_zlabels=5, axes_ranges=[0, 126, 0, 19, -20, 0])
    plotter.show_axes()
    plotter.save_graphic(f'{paths.output}/{model_id}/{site}/results/{name}/[{period[0]}_{period[-1]}]_mum_slices.svg')


if len(periods) == 1:
    cb_soil = fig_soils_Ns.colorbar(im1, ax=axs_soils_Ns[:,0].ravel(), orientation='horizontal', aspect=15, label='Soil type', shrink=0.4, pad=0.05)
elif len(periods) == 2:
    cb_soil = fig_soils_Ns.colorbar(im1, ax=axs_soils_Ns[:,0].ravel(), orientation='horizontal', aspect=15, label='Soil type', shrink=0.4, pad=0.02)
cb_soil.set_ticks(list(int_to_soil.keys())[1:])
cb_soil.set_ticklabels(list(int_to_soil.values())[1:])
if len(periods) == 1:
    cb_N = fig_soils_Ns.colorbar(im2, ax=axs_soils_Ns[:,1].ravel(), orientation='horizontal', aspect=15, label='N', shrink=0.4, pad=0.05)
elif len(periods) == 2:
    cb_N = fig_soils_Ns.colorbar(im2, ax=axs_soils_Ns[:,1].ravel(), orientation='horizontal', aspect=15, label='N', shrink=0.4, pad=0.03)
cb_N.set_ticks([6, 7, 8, 9, 10])
cb_N.set_ticklabels([6, 7, 8, 9, 10])
axs_soils_Ns[0, 1].set_title(f"{datetime.strptime(periods[0][0], '%Y-%m-%d').strftime('%b %d, %Y')} - {datetime.strptime(periods[0][-1], '%Y-%m-%d').strftime('%b %d, %Y')}", fontsize=8, weight='bold', x=-0.055, y=1.8)
if len(periods) == 2:
    axs_soils_Ns[6, 1].set_title(f"{datetime.strptime(periods[1][0], '%Y-%m-%d').strftime('%b %d, %Y')} - {datetime.strptime(periods[1][-1], '%Y-%m-%d').strftime('%b %d, %Y')}", fontsize=8, weight='bold', x=-0.055, y=1.8)
    axs_soils_Ns[5, 0].set_axis_off()
    axs_soils_Ns[5, 1].set_axis_off()
fig_soils_Ns.savefig(f'{paths.output}/{model_id}/{site}/results/{name}/{name}_soil_N.svg', bbox_inches='tight')


if len(periods) == 1:
    cb_mums = fig_mums.colorbar(im8, ax=axs_soil_N_entropy.ravel(), orientation='horizontal', aspect=15, label='Relative entropy [-]', shrink=0.35, pad=0.05)
elif len(periods) == 2:
    cb_mums = fig_mums.colorbar(im8, ax=axs_soil_N_entropy.ravel(), orientation='horizontal', aspect=15, label='Relative entropy [-]', shrink=0.2, pad=0.07)
if len(periods) == 1:
    fig_mums.suptitle(f"{datetime.strptime(periods[0][0], '%Y-%m-%d').strftime('%b %d, %Y')} - {datetime.strptime(periods[0][-1], '%Y-%m-%d').strftime('%b %d, %Y')}",
                            fontsize=8, weight='bold')
elif len(periods) == 2:
    axs_mums[0, 1].set_title(f"{datetime.strptime(periods[0][0], '%Y-%m-%d').strftime('%b %d, %Y')} - {datetime.strptime(periods[0][-1], '%Y-%m-%d').strftime('%b %d, %Y')}                              {datetime.strptime(periods[1][0], '%Y-%m-%d').strftime('%b %d, %Y')} - {datetime.strptime(periods[1][-1], '%Y-%m-%d').strftime('%b %d, %Y')}",
                          fontsize=8, weight='bold', x=-0.06, y=1.9)
fig_soil_N_entropy.savefig(f'{paths.output}/{model_id}/{site}/results/{name}/{name}_soil_N_WT_entropy.svg', bbox_inches='tight')


if len(periods) == 1:
    cb_mums = fig_mums.colorbar(im3, ax=axs_mums.ravel(), orientation='horizontal', aspect=15, label="${μ_m}$ [GPa]", shrink=0.35, pad=0.05)
elif len(periods) == 2:
    cb_mums = fig_mums.colorbar(im3, ax=axs_mums.ravel(), orientation='horizontal', aspect=15, label="${μ_m}$ [GPa]", shrink=0.2, pad=0.07)
cb_mums.set_ticks([0.1, 0.2, 0.3, 0.4])
cb_mums.set_ticklabels([0.1, 0.2, 0.3, 0.4])
if len(periods) == 1:
    fig_mums.suptitle(f"{datetime.strptime(periods[0][0], '%Y-%m-%d').strftime('%b %d, %Y')} - {datetime.strptime(periods[0][-1], '%Y-%m-%d').strftime('%b %d, %Y')}",
                            fontsize=8, weight='bold')
elif len(periods) == 2:
    axs_mums[0, 1].set_title(f"{datetime.strptime(periods[0][0], '%Y-%m-%d').strftime('%b %d, %Y')} - {datetime.strptime(periods[0][-1], '%Y-%m-%d').strftime('%b %d, %Y')}                              {datetime.strptime(periods[1][0], '%Y-%m-%d').strftime('%b %d, %Y')} - {datetime.strptime(periods[1][-1], '%Y-%m-%d').strftime('%b %d, %Y')}",
                          fontsize=8, weight='bold', x=-0.06, y=1.9)
fig_mums.savefig(f'{paths.output}/{model_id}/{site}/results/{name}/{name}_mum.svg', bbox_inches='tight')


if len(periods) == 1:
    cb_Vs = fig_VSs.colorbar(im5, ax=axs_VSs.ravel(), orientation='horizontal', aspect=15, label="${V_S}$ [m/s]", shrink=0.2, pad=0.07)
elif len(periods) == 2:
    cb_Vs = fig_VSs.colorbar(im5, ax=axs_VSs.ravel(), orientation='horizontal', aspect=15, label="${V_S}$ [m/s]", shrink=0.2, pad=0.032)
cb_Vs.set_ticks([200, 400, 600])
cb_Vs.set_ticklabels([200, 400, 600])
# cb_Vs.set_ticks([200, 550, 900])
# cb_Vs.set_ticklabels([200, 550, 900])
axs_VSs[0, 1].set_title(f"{datetime.strptime(periods[0][0], '%Y-%m-%d').strftime('%b %d, %Y')} - {datetime.strptime(periods[0][-1], '%Y-%m-%d').strftime('%b %d, %Y')}", fontsize=8, weight='bold', x=-0.055, y=1.8)
if len(periods) == 2:
    axs_VSs[6, 1].set_title(f"{datetime.strptime(periods[1][0], '%Y-%m-%d').strftime('%b %d, %Y')} - {datetime.strptime(periods[1][-1], '%Y-%m-%d').strftime('%b %d, %Y')}", fontsize=8, weight='bold', x=-0.055, y=1.8)
    axs_VSs[5, 0].set_axis_off()
    axs_VSs[5, 1].set_axis_off()
fig_VSs.savefig(f'{paths.output}/{model_id}/{site}/results/{name}/{name}_Vs.svg', bbox_inches='tight')


plt.close('all')
