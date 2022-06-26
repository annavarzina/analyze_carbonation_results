
import utils as ut
import numpy as np
np.set_printoptions(precision=5, threshold=np.inf)

import pathlib
current_dir = pathlib.Path().parent.absolute()
shared_dir = current_dir.parents[3]/'VirtualBox VMs' / 'shared' / 'results' / 'output_csh' / '06_port'
print(shared_dir)
# prev_results_dir = parent_dir /'previous_results' / 'output_csh' / '01_default'
figures_dir =  current_dir.parents[0] /'figures' / 'output_csh' / '06_port'  # / '01_csh_default'
print(figures_dir)
fname = 'compare'
# ut.make_output_dir(parent_dir /'figures' / 'output_csh')
# ut.make_output_dir(parent_dir /'figures' / 'output_csh' / '01_default')
ut.make_output_dir(figures_dir)


import matplotlib.pylab as plt

save_figures = False
# %% SETTINGS
Ts = 1000.
names = np.array(['01_csh_default', '02_ch_default']) #

label = np.array(['CSH', 'CH'])
scale = 100
linetype = np.array(['-', '--', '-.', ':', '-'])
results = {}
for nn in names:
    file_name = nn + '_results.pkl'
    results[nn] = ut.load_obj(shared_dir / nn / file_name)
# %% Keys
keys = ['time', 'calcite', 'Ca', 'C']
        #, 'Si','CSHQ_TobD', 'CSHQ_TobH', 'CSHQ_JenD', 'CSHQ_JenH']
sres = {}
n = names
for i in range(0, len(n)):
    temp = {}
    s = np.size(results[n[i]]['time'])
    for k in keys:
        if (np.size(results[n[i]][k]) == s):
            temp[k] = np.array(results[n[i]][k])
    temp['time'] *= scale
    temp['calcite'] *= scale
    # temp['CSHQ_TobD'] *= scale
    # temp['CSHQ_TobH'] *= scale
    # temp['CSHQ_JenD'] *= scale
    # temp['CSHQ_JenH'] *= scale
    sres[n[i]] = temp
# %% Plot total concentrations
titles = ['Calcite', 'Calcium', 'Carbon']
comp = ['calcite', 'Ca', 'C']
suffix = ['_calcite', '_calcium', '_carbon']
ylabel = [r'Calcite  $\cdot 10^{-15}$ mol',
          r'Dissolved Ca $\cdot 10^{-15}$ (mol)',
          r'Dissolved C $\cdot 10^{-15}$ (mol)' ]
if(save_figures):
    for k in range(0, len(comp)):
        plt.figure(figsize=(8, 4), dpi = 200)
        # for i in range(0, len(names)):
        for i in range(0, len(names)):
            plt.plot(sres[names[i]]['time'] / 3600, sres[names[i]][comp[k]],
                     ls=linetype[i], label=label[i])
        plt.ylabel(ylabel[k], fontsize=14)
        # plt.title(titles[k])
        plt.xlabel('Time (h)', fontsize=14)
        plt.legend(fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=12)
        file_name = fname + suffix[k] + '.png'
        plt.savefig(figures_dir / file_name, bbox_inches='tight')
        plt.close()
# %% Plot CSH density
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(results[names[0]]['ratio_CaSi (1, 5)'], results[names[0]]['density_CSHQ'])
    plt.ylabel('C-S-H density (g/l)', fontsize=16)
    plt.xlabel('Ca/Si', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.margins(-0.1,-0.1)
    fig.savefig(figures_dir / 'csh_casi_density.png', bbox_inches='tight')
    plt.close()
# %% CSH: plot Ca/Si
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600,
             results[names[0]]['ratio_CaSi (1, 5)'],
             label='Voxel 6', ls='-')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600,
             results[names[0]]['ratio_CaSi (1, 6)'],
             label='Voxel 7', ls='--')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600,
             results[names[0]]['ratio_CaSi (1, 7)'],
             label='Voxel 8', ls='-.')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600,
             results[names[0]]['ratio_CaSi (1, 11)'],
             label='Voxel 12', ls=':')
    plt.legend(fontsize=10)
    plt.ylabel('Ca/Si', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_sg_casi_time.png', bbox_inches='tight')
    plt.close()
# %% CSH: plot Ca
if (save_figures):
    w = 2 # window
    i = int(w/2)
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i] * scale / 3600,
             np.convolve(results[names[0]]['Ca (1, 3)'], np.ones(w), 'valid') / w,
             label='Voxel 4', ls='-.')
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i] * scale / 3600,
             np.convolve(results[names[0]]['Ca (1, 4)'], np.ones(w), 'valid') / w,
             label='Voxel 5', ls=':')
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i]  * scale / 3600,
             np.convolve(results[names[0]]['Ca (1, 5)'], np.ones(w), 'valid') / w,
             label='Voxel 6', ls='-')
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i]  * scale / 3600,
             np.convolve(results[names[0]]['Ca (1, 6)'], np.ones(w), 'valid') / w,
             label='Voxel 7', ls='--')
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i]  * scale / 3600,
             np.convolve(results[names[0]]['Ca (1, 7)'], np.ones(w), 'valid') / w,
             label='Voxel 8', ls='-.')
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i]  * scale / 3600,
             np.convolve(results[names[0]]['Ca (1, 8)'], np.ones(w), 'valid') / w,
             label='Voxel 9', ls=':')
    plt.legend(fontsize=10)
    plt.ylabel('Ca (mol/l)', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_Ca_time.png', bbox_inches='tight')
    plt.close()
# %% CH: plot Ca
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600, results[names[1]]['Ca (1, 3)'],
             label='Voxel 4', ls='-.')
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600, results[names[1]]['Ca (1, 4)'],
             label='Voxel 5', ls=':')
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600, results[names[1]]['Ca (1, 5)'],
             label='Voxel 6', ls='-')
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600, results[names[1]]['Ca (1, 6)'],
             label='Voxel 7', ls='--')
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600, results[names[1]]['Ca (1, 7)'],
             label='Voxel 8', ls='-.')
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600, results[names[1]]['Ca (1, 8)'],
             label='Voxel 9', ls=':')
    plt.legend(fontsize=10)
    plt.ylabel('Ca (mol/l)', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'ch_Ca_time.png', bbox_inches='tight')
    plt.close()
# %% CSH: plot C
if (save_figures):
    w = 2 # window
    i = int(w/2)
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i] * scale / 3600,
             np.convolve(results[names[0]]['C (1, 3)'], np.ones(w), 'valid') / w,
             label='Voxel 4',ls='-.')
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i] * scale / 3600,
             np.convolve(results[names[0]]['C (1, 4)'], np.ones(w), 'valid') / w,
             label='Voxel 5', ls=':')
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i] * scale / 3600,
             np.convolve(results[names[0]]['C (1, 5)'], np.ones(w), 'valid') / w,
             label='Voxel 6',ls='-')
    plt.legend(fontsize=10)
    plt.ylabel('C (mol/l)', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_C_time.png', bbox_inches='tight')
    plt.close()
# %% CH: plot C
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600,
             results[names[1]]['C (1, 3)'],
             label='Voxel 4', ls='-.')
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600,
             results[names[1]]['C (1, 4)'],
             label='Voxel 5', ls=':')
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600,
             results[names[1]]['C (1, 5)'],
             label='Voxel 6', ls='-')
    plt.legend(fontsize=10)
    plt.ylabel('C (mol/l)', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'ch_C_time.png', bbox_inches='tight')
    plt.close()
# %% CSH: plot Si
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['Si (1, 3)'],
             label='Voxel 4', ls='-.')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['Si (1, 4)'],
             label='Voxel 5', ls=':')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['Si (1, 5)'],
             label='Voxel 6', ls='-')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['Si (1, 6)'],
             label='Voxel 7', ls='--')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['Si (1, 7)'],
             label='Voxel 8', ls='-.')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['Si (1, 8)'],
             label='Voxel 9', ls=':')
    plt.legend(fontsize=10)
    plt.ylabel('Si (mol/l)', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_Si_time.png', bbox_inches='tight')
    plt.close()
# %% CSH: plot pH
if (save_figures):
    w=2 #window
    i = int(w/2)
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i] * scale / 3600, np.convolve(results[names[0]]['pH (1, 3)'], np.ones(w), 'valid') / w,
             label='Voxel 4', ls='-.')
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i] * scale / 3600, np.convolve(results[names[0]]['pH (1, 4)'], np.ones(w), 'valid') / w,
             label='Voxel 5', ls=':')
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i] * scale / 3600, np.convolve(results[names[0]]['pH (1, 5)'], np.ones(w), 'valid') / w,
             label='Voxel 6', ls='-')
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i] * scale / 3600, np.convolve(results[names[0]]['pH (1, 6)'], np.ones(w), 'valid') / w,
             label='Voxel 7', ls='--')
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i] * scale / 3600, np.convolve(results[names[0]]['pH (1, 7)'], np.ones(w), 'valid') / w,
             label='Voxel 8', ls='-.')
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i] * scale / 3600, np.convolve(results[names[0]]['pH (1, 8)'], np.ones(w), 'valid') / w,
             label='Voxel 9', ls=':')
    plt.legend(fontsize=10)
    plt.ylabel('pH', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_ph_time.png', bbox_inches='tight')
    plt.close()
# %% CH: plot pH
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600,
             results[names[1]]['pH (1, 3)'],
             label='Voxel 4', ls='-.')
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600,
             results[names[1]]['pH (1, 4)'],
             label='Voxel 5', ls=':')
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600,
             results[names[1]]['pH (1, 5)'],
             label='Voxel 6', ls='-')
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600,
             results[names[1]]['pH (1, 6)'],
             label='Voxel 7', ls='--')
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600,
             results[names[1]]['pH (1, 7)'],
             label='Voxel 8', ls='-.')
    plt.plot(np.array(results[names[1]]['time']) * scale / 3600,
             results[names[1]]['pH (1, 8)'],
             label='Voxel 9', ls=':')
    plt.legend(fontsize=10)
    plt.ylabel('pH', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'ch_ph_time.png', bbox_inches='tight')
    plt.close()
# %% plot vol_CSHQ
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['vol_CSHQ (1, 5)'],
             label='Voxel 6', ls='-')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['vol_CSHQ (1, 6)'],
             label='Voxel 7', ls='--')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['vol_CSHQ (1, 7)'],
             label='Voxel 8', ls='-.')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['vol_CSHQ (1, 8)'],
             label='Voxel 9', ls=':')
    plt.legend(fontsize=10)
    plt.ylabel(r'C-S-H volume ($\mu m^3$)', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_vol_CSHQ_time.png', bbox_inches='tight')
    plt.close()
# %% CSHQ phases
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['CSHQ_JenD'], label='JenD',
             ls='-')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['CSHQ_JenH'], label='JenH',
             ls='--')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['CSHQ_TobD'], label='TobD',
             ls='-.')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['CSHQ_TobH'], label='TobH',
             ls=':')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['calcite'],
             label='$C\overline{C}$', ls='--')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600, results[names[0]]['sio2am'], label='$SiO_2$',
             ls='--')
    plt.xlabel('Time (h)', fontsize=16)
    plt.ylabel(r'C-S-H $\cdot 10^{-15}$ $(mol)$', fontsize=16)
    plt.legend(fontsize=10, loc='upper left')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_phases.png', bbox_inches='tight')
    plt.close()
# %% Ca profile
calcium_csh = np.load(shared_dir / names[0] / 'Ca.npy')
calcium_ch = np.load(shared_dir / names[1] / 'Ca.npy')
print('Ca:')
print(calcium_csh)
print(calcium_ch)
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.imshow([calcium_csh[1:-1], calcium_ch[1:-1]])
    plt.xlabel(r'Distance ($\mu m$)', fontsize=12)
    plt.title('Ca concentration')
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    ax.set_yticks([0,1])
    ax.set_yticklabels(['C-S-H', 'CH'])
    clb = plt.colorbar(orientation="horizontal", pad=0.4)
    clb.ax.set_title(r'Colorbar: Ca concentration ($mol/l$)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.savefig(figures_dir / 'Ca_profile.png', bbox_inches='tight')
    plt.close()
# %% C profile
carbon_csh = np.load(shared_dir / names[0] / 'C.npy')
carbon_ch = np.load(shared_dir / names[1] / 'C.npy')
    # correction
    # carbon_csh[7] = (carbon_csh[6] + carbon_csh[8] )/2
    # carbon_csh[11] = (carbon_csh[10] + carbon_csh[12] )/2
    # carbon_csh[13] = (carbon_csh[12] + carbon_csh[14] )/2
print('C:')
print(carbon_csh)
print(carbon_ch)
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.imshow([carbon_csh[1:-1], carbon_ch[1:-1]])
    plt.xlabel(r'Distance ($\mu m$)', fontsize=12)
    plt.title('C concentration')
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    ax.set_yticks([0,1])
    ax.set_yticklabels(['C-S-H', 'CH'])
    clb = plt.colorbar(orientation="horizontal", pad=0.4)
    clb.ax.set_title(r'Colorbar: C concentration ($mol/l$)', fontsize=12)
    clb.ax.locator_params(nbins=5)
    # clb.ax.set_xticklabels(clb.ax.get_xticklabels(), rotation=-45)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.savefig(figures_dir / 'C_profile.png', bbox_inches='tight')
    plt.close()
# %% CC profile
cc_csh = np.load(shared_dir / names[0] / 'CC.npy')
cc_ch = np.load(shared_dir / names[1] / 'CC.npy')
print('CC:')
print(cc_csh)
print(cc_ch)
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.imshow([cc_csh[1:-1], cc_ch[1:-1]])
    plt.xlabel(r'Distance ($\mu m$)', fontsize=12)
    plt.title('Calcite ')
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    ax.set_yticks([0,1])
    ax.set_yticklabels(['C-S-H', 'CH'])
    clb = plt.colorbar(orientation="horizontal", pad=0.4)
    clb.ax.set_title(r'Colorbar: Calcite amount ($mol/dm^3$)', fontsize=12)
    # clb.ax.locator_params(nbins=5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.savefig(figures_dir / 'CC_profile.png', bbox_inches='tight')
    plt.close()
# %% Poros profile
poros_csh = np.load(shared_dir / names[0] / 'poros.npy')
poros_ch = np.load(shared_dir / names[1] / 'poros.npy')
print('Porosity:')
print(poros_csh)
print(poros_ch)
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.imshow([poros_csh[1:-1], poros_ch[1:-1]])
    plt.xlabel(r'Distance ($\mu m$)', fontsize=12)
    plt.title(r'Free volume ($\mu m^3$)')
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    ax.set_yticks([0,1])
    ax.set_yticklabels(['C-S-H', 'CH'])
    clb = plt.colorbar(orientation="horizontal", pad=0.4)
    clb.ax.set_title(r'Colorbar: Free volume ($\mu m^3$)', fontsize=12)
    # clb.ax.locator_params(nbins=5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.savefig(figures_dir / 'poros_profile.png', bbox_inches='tight')
    plt.close()
# %% ph
ph_csh = np.load(shared_dir / names[0]/ 'pH.npy')
ph_ch = np.load(shared_dir / names[1]/ 'pH.npy')
    # correction
    # ph_csh[7] = (ph_csh[6] + ph_csh[8] )/2
    # ph_csh[11] = (ph_csh[10] + ph_csh[12] )/2
    # ph_csh[13] = (ph_csh[12] + ph_csh[14] )/2
print('pH:')
print(ph_csh)
print(ph_ch)
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.imshow([ph_csh[1:-1], ph_ch[1:-1]])
    plt.xlabel(r'Distance ($\mu m$)', fontsize=12)
    plt.title('pH')
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    ax.set_yticks([0,1])
    ax.set_yticklabels(['C-S-H', 'CH'])
    clb = plt.colorbar(orientation="horizontal", pad=0.4)
    clb.ax.set_title(r'Colorbar: pH', fontsize=12)
    # clb.ax.locator_params(nbins=5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.savefig(figures_dir / 'ph_profile.png', bbox_inches='tight')
    plt.close()
# %% De
de_csh = np.load(shared_dir / names[0] / 'De.npy')
de_ch = np.load(shared_dir / names[1] / 'De.npy')
print('De:')
print(de_csh[1:-1])
print(de_ch[1:-1])
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.imshow([de_csh[1:-1], ph_ch[1:-1]])  #
    plt.xlabel(r'Distance ($\mu m$)', fontsize=12)
    plt.title('De')
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['C-S-H', 'CH'])
    clb = plt.colorbar(orientation="horizontal", pad=0.4)
    clb.ax.set_title(r'Colorbar: De', fontsize=12)
    # clb.ax.locator_params(nbins=5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.savefig(figures_dir / 'de_profile.png', bbox_inches='tight')
    plt.close()