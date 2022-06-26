import utils as ut
import numpy as np
import matplotlib.pylab as plt
np.set_printoptions(precision=5, threshold=np.inf)

import pathlib
current_dir = pathlib.Path().parent.absolute()
shared_dir = current_dir.parents[3]/'VirtualBox VMs' / 'shared' / 'results' / 'output_csh' / '07_co2'
print(shared_dir)
# prev_results_dir = parent_dir /'previous_results' / 'output_csh' / '01_default'
figures_dir =  current_dir.parents[0] /'figures' / 'output_csh' / '07co2'  # / '01_csh_default'
print(figures_dir)
fname = 'compare'
# ut.make_output_dir(parent_dir /'figures' / 'output_csh')
# ut.make_output_dir(parent_dir /'figures' / 'output_csh' / '01_default')
ut.make_output_dir(figures_dir)

save_figures = True
#%% SETTINGS
Ts = 1000.
names = np.array(['01_co2_003_352','02_co2_01_3', '03_co2_1_2']) #

label = np.array(['0.03% CO2', '0.1% CO2', '1% CO2'])
scale = 100
linetype = np.array(['-', '--', '-.', ':', '-'])
results = {}
for nn in names:
    file_name = nn + '_results.pkl'
    results[nn] = ut.load_obj(shared_dir / nn / file_name)

# %% Scaling
print("Scaling...")
keys = ['time', 'calcite', 'Ca', 'C','Si',
        'CSHQ_TobD', 'CSHQ_TobH', 'CSHQ_JenD', 'CSHQ_JenH']
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
    temp['CSHQ_TobD'] *= scale
    temp['CSHQ_TobH'] *= scale
    temp['CSHQ_JenD'] *= scale
    temp['CSHQ_JenH'] *= scale
    sres[n[i]] = temp

#%% Plot totals
print("Plot totals...")
titles = ['Calcite', 'Calcium', 'Carbon', 'Silicium',
          'CSHQ_TobD', 'CSHQ_TobH', 'CSHQ_JenD', 'CSHQ_JenH']
comp = ['calcite', 'Ca', 'C', 'Si',
        'CSHQ_TobD', 'CSHQ_TobH', 'CSHQ_JenD', 'CSHQ_JenH']
suffix = ['_calcite', '_calcium', '_carbon', '_silicium',
          '_CSHQ_TobD', '_CSHQ_TobH', '_CSHQ_JenD', '_CSHQ_JenH']
ylabel = [r'Calcite  $\cdot 10^{-15}$ mol',
          r'Dissolved Ca $\cdot 10^{-15}$ (mol)',
          r'Dissolved C $\cdot 10^{-15}$ (mol)',
          r'Dissolved Si $\cdot 10^{-15}$ (mol)',
          r'CSHQ_TobD  $\cdot 10^{-15}$ mol',
          r'CSHQ_TobH  $\cdot 10^{-15}$ mol',
          r'CSHQ_JenD  $\cdot 10^{-15}$ mol',
          r'CSHQ_JenH  $\cdot 10^{-15}$ mol', ]
if (save_figures):
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
print("Plot CSH density...")
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
print("Plot Ca/Si...")
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600,
             results[names[0]]['ratio_CaSi (1, 5)'],
             label='Voxel 6', ls='-')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600,
             results[names[0]]['ratio_CaSi (1, 7)'],
             label='Voxel 8', ls='-.')
    plt.plot(np.array(results[names[0]]['time']) * scale / 3600,
             results[names[0]]['ratio_CaSi (1, 9)'],
             label='Voxel 10', ls=':')
    plt.legend(fontsize=10)
    plt.ylabel('Ca/Si', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_sg_casi_time.png', bbox_inches='tight')
    plt.close()
# %% CSH: plot Ca
print("Plot Ca 0.03% CO2...")
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
    fig.savefig(figures_dir / 'csh_0.03CO2_Ca_time.png', bbox_inches='tight')
    plt.close()
# %% CSH: plot Ca 1% CO2
print("Plot Ca 1% CO2...")
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['Ca (1, 3)'],
             label='Voxel 4', ls='-.')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['Ca (1, 4)'],
             label='Voxel 5', ls=':')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['Ca (1, 5)'],
             label='Voxel 6', ls='-')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['Ca (1, 6)'],
             label='Voxel 7', ls='--')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['Ca (1, 7)'],
             label='Voxel 8', ls='-.')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['Ca (1, 8)'],
             label='Voxel 9', ls=':')
    plt.legend(fontsize=10)
    plt.ylabel('Ca (mol/l)', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_1CO2_Ca_time.png', bbox_inches='tight')
    plt.close()
# %% CSH 0.03 CO2: plot C
print("Plot C 0.03% CO2...")
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
    plt.plot(np.array(results[names[0]]['time'])[i-1:-i] * scale / 3600,
             np.convolve(results[names[0]]['C (1, 6)'], np.ones(w), 'valid') / w,
             label='Voxel 7',ls='-')
    plt.legend(fontsize=10)
    plt.ylabel('C (mol/l)', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_0.03CO2_C_time.png', bbox_inches='tight')
    plt.close()
# %% CH 1% CO2: plot C
print("Plot C 1% CO2...")
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600,
             results[names[2]]['C (1, 3)'],
             label='Voxel 4', ls='-.')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600,
             results[names[2]]['C (1, 4)'],
             label='Voxel 5', ls=':')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600,
             results[names[2]]['C (1, 5)'],
             label='Voxel 6', ls='-')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600,
             results[names[2]]['C (1, 6)'],
             label='Voxel 7', ls='-')
    plt.legend(fontsize=10)
    plt.ylabel('C (mol/l)', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_1CO2_C_time.png', bbox_inches='tight')
    plt.close()
# %% CSH: plot Si
print("Plot Si 1% CO2...")
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['Si (1, 3)'],
             label='Voxel 4', ls=':')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['Si (1, 4)'],
             label='Voxel 5', ls=':')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['Si (1, 5)'],
             label='Voxel 6', ls='-')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['Si (1, 6)'],
             label='Voxel 7', ls='--')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['Si (1, 7)'],
             label='Voxel 8', ls='-.')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['Si (1, 8)'],
             label='Voxel 9', ls=':')
    plt.legend(fontsize=10)
    plt.ylabel('Si (mol/l)', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_Si_time.png', bbox_inches='tight')
    plt.close()
# %% CSH: plot pH
print("Plot pH 1% CO2...")
if (save_figures):
    w=2 #window
    i = int(w/2)
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[2]]['time'])[i-1:-i] * scale / 3600, np.convolve(results[names[2]]['pH (1, 3)'], np.ones(w), 'valid') / w,
             label='Voxel 4', ls='-.')
    plt.plot(np.array(results[names[2]]['time'])[i-1:-i] * scale / 3600, np.convolve(results[names[2]]['pH (1, 4)'], np.ones(w), 'valid') / w,
             label='Voxel 5', ls=':')
    plt.plot(np.array(results[names[2]]['time'])[i-1:-i] * scale / 3600, np.convolve(results[names[2]]['pH (1, 5)'], np.ones(w), 'valid') / w,
             label='Voxel 6', ls='-')
    plt.plot(np.array(results[names[2]]['time'])[i-1:-i] * scale / 3600, np.convolve(results[names[2]]['pH (1, 6)'], np.ones(w), 'valid') / w,
             label='Voxel 7', ls='--')
    plt.plot(np.array(results[names[2]]['time'])[i-1:-i] * scale / 3600, np.convolve(results[names[2]]['pH (1, 7)'], np.ones(w), 'valid') / w,
             label='Voxel 8', ls='-.')
    plt.plot(np.array(results[names[2]]['time'])[i-1:-i] * scale / 3600, np.convolve(results[names[2]]['pH (1, 8)'], np.ones(w), 'valid') / w,
             label='Voxel 9', ls=':')
    plt.legend(fontsize=10)
    plt.ylabel('pH', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_ph_time.png', bbox_inches='tight')
    plt.close()
# %% plot vol_CSHQ
print("Plot CSH colume 1% CO2...")
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['vol_CSHQ (1, 5)'],
             label='Voxel 6', ls='-')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['vol_CSHQ (1, 6)'],
             label='Voxel 7', ls='--')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['vol_CSHQ (1, 7)'],
             label='Voxel 8', ls='-.')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['vol_CSHQ (1, 8)'],
             label='Voxel 9', ls=':')
    plt.legend(fontsize=10)
    plt.ylabel(r'C-S-H volume ($\mu m^3$)', fontsize=16)
    plt.xlabel('Time (h)', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_vol_CSHQ_time.png', bbox_inches='tight')
    plt.close()
# %% CSHQ phases
print("Plot CSH phases 1% CO2...")
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['CSHQ_JenD'], label='JenD',
             ls='-')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['CSHQ_JenH'], label='JenH',
             ls='--')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['CSHQ_TobD'], label='TobD',
             ls='-.')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['CSHQ_TobH'], label='TobH',
             ls=':')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['calcite'],
             label='$C\overline{C}$', ls='--')
    plt.plot(np.array(results[names[2]]['time']) * scale / 3600, results[names[2]]['sio2am'], label='$SiO_2$',
             ls='--')
    plt.xlabel('Time (h)', fontsize=16)
    plt.ylabel(r'C-S-H $\cdot 10^{-15}$ $(mol)$', fontsize=16)
    plt.legend(fontsize=10, loc='upper left')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(figures_dir / 'csh_phases.png', bbox_inches='tight')
    plt.close()
# %% Ca profile
print("Ca profile...")
csh_1 = np.load(shared_dir / names[0] / 'Ca.npy')
csh_2 = np.load(shared_dir / names[1] / 'Ca.npy')
csh_3 = np.load(shared_dir / names[2] / 'Ca.npy')
# print('Ca:')
# print(csh_1)
# print(csh_2)
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.imshow([csh_1[1:-1], csh_2[1:-1], csh_3[1:-1]])
    plt.xlabel(r'Distance ($\mu m$)', fontsize=12)
    plt.title('Ca concentration')
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    ax.set_yticks([0,1,2])
    ax.set_yticklabels(['C-S-H 0.03% CO2', 'C-S-H 0.1% CO2', 'C-S-H 1% CO2'])
    clb = plt.colorbar(orientation="horizontal", pad=0.4)
    clb.ax.set_title(r'Colorbar: Ca concentration ($mol/l$)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.savefig(figures_dir / 'Ca_profile.png', bbox_inches='tight')
    plt.close()
# %% C profile
print("C profile...")
csh_1 = np.load(shared_dir / names[0] / 'C.npy')
csh_2 = np.load(shared_dir / names[1] / 'C.npy')
csh_3 = np.load(shared_dir / names[2] / 'C.npy')
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.imshow([csh_1[1:-1], csh_2[1:-1], csh_3[1:-1]])
    plt.xlabel(r'Distance ($\mu m$)', fontsize=12)
    plt.title('C concentration')
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    ax.set_yticks([0,1,2])
    ax.set_yticklabels(['C-S-H 0.03% CO2', 'C-S-H 0.1% CO2', 'C-S-H 1% CO2'])
    clb = plt.colorbar(orientation="horizontal", pad=0.4)
    clb.ax.set_title(r'Colorbar: C concentration ($mol/l$)', fontsize=12)
    clb.ax.locator_params(nbins=5)
    # clb.ax.set_xticklabels(clb.ax.get_xticklabels(), rotation=-45)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.savefig(figures_dir / 'C_profile.png', bbox_inches='tight')
    plt.close()
# %% CC profile
print("CC profile...")
csh_1 = np.load(shared_dir / names[0] / 'CC.npy')
csh_2 = np.load(shared_dir / names[1] / 'CC.npy')
csh_3 = np.load(shared_dir / names[2] / 'CC.npy')
print('CC:')
print(csh_1)
print(csh_2)
print(csh_3)
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.imshow([csh_1[1:-1], csh_2[1:-1], csh_3[1:-1]])
    plt.xlabel(r'Distance ($\mu m$)', fontsize=12)
    plt.title('Calcite ')
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    ax.set_yticks([0,1,2])
    ax.set_yticklabels(['C-S-H 0.03% CO2', 'C-S-H 0.1% CO2', 'C-S-H 1% CO2'])
    clb = plt.colorbar(orientation="horizontal", pad=0.4)
    clb.ax.set_title(r'Colorbar: Calcite amount ($mol/dm^3$)', fontsize=12)
    # clb.ax.locator_params(nbins=5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.savefig(figures_dir / 'CC_profile.png', bbox_inches='tight')
    plt.close()
# %% Poros profile
print("Porosity profile...")
csh_1 = np.load(shared_dir / names[0] / 'poros.npy')
csh_2 = np.load(shared_dir / names[1] / 'poros.npy')
csh_3 = np.load(shared_dir / names[2] / 'poros.npy')
print('Porosity:')
print(csh_1)
print(csh_2)
print(csh_3)
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.imshow([csh_1[1:-1], csh_2[1:-1], csh_3[1:-1]])
    plt.xlabel(r'Distance ($\mu m$)', fontsize=12)
    plt.title(r'Free volume ($\mu m^3$)')
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    ax.set_yticks([0,1,2])
    ax.set_yticklabels(['C-S-H 0.03% CO2', 'C-S-H 0.1% CO2', 'C-S-H 1% CO2'])
    clb = plt.colorbar(orientation="horizontal", pad=0.4)
    clb.ax.set_title(r'Colorbar: Free volume ($\mu m^3$)', fontsize=12)
    # clb.ax.locator_params(nbins=5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.savefig(figures_dir / 'poros_profile.png', bbox_inches='tight')
    plt.close()
# %% ph
print("pH profile...")
csh_1 = np.load(shared_dir / names[0] / 'pH.npy')
csh_2 = np.load(shared_dir / names[1] / 'pH.npy')
csh_3 = np.load(shared_dir / names[2] / 'pH.npy')
print('pH:')
print(csh_1)
print(csh_2)
print(csh_3)
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.imshow([csh_1[1:-1], csh_2[1:-1], csh_3[1:-1]])
    plt.xlabel(r'Distance ($\mu m$)', fontsize=12)
    plt.title('pH')
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    ax.set_yticks([0,1,2])
    ax.set_yticklabels(['C-S-H 0.03% CO2', 'C-S-H 0.1% CO2', 'C-S-H 1% CO2'])
    clb = plt.colorbar(orientation="horizontal", pad=0.4)
    clb.ax.set_title(r'Colorbar: pH', fontsize=12)
    # clb.ax.locator_params(nbins=5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.savefig(figures_dir / 'ph_profile.png', bbox_inches='tight')
    plt.close()
# %% De
print("De profile...")
csh_1 = np.load(shared_dir / names[0] / 'De.npy')
csh_2 = np.load(shared_dir / names[1] / 'De.npy')
csh_3 = np.load(shared_dir / names[2] / 'De.npy')
print('De:')
print(csh_1)
print(csh_2)
print(csh_3)
if (save_figures):
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.imshow([csh_1[1:-1], csh_2[1:-1], csh_3[1:-1]])
    plt.xlabel(r'Distance ($\mu m$)', fontsize=12)
    plt.title('De')
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    ax.set_yticks([0,1,2])
    ax.set_yticklabels(['C-S-H 0.03% CO2', 'C-S-H 0.1% CO2', 'C-S-H 1% CO2'])
    clb = plt.colorbar(orientation="horizontal", pad=0.4)
    clb.ax.set_title(r'Colorbar: De', fontsize=12)
    # clb.ax.locator_params(nbins=5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.savefig(figures_dir / 'de_profile.png', bbox_inches='tight')
    plt.close()