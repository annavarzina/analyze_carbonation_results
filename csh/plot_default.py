
import utils as ut
import numpy as np
np.set_printoptions(precision=5, threshold=np.inf)

import pathlib
current_dir = pathlib.Path().parent.absolute()
parent_dir = current_dir.parents[0]
prev_results_dir = parent_dir /'previous_results' / 'output_csh' / '01_default'
figures_dir =  parent_dir /'figures' / 'output_csh' / '01_default' / '02_CSH_5vox'

fname = 'compare'
# ut.make_output_dir(parent_dir /'figures' / 'output_csh')
# ut.make_output_dir(parent_dir /'figures' / 'output_csh' / '01_default')
ut.make_output_dir(figures_dir)


import matplotlib.pylab as plt

# %% SETTINGS
Ts = 1000.

names = np.array(['02_CSH_5vox'])

label = np.array(['default'])
scale = 100
linetype = np.array(['-', '--', '-.', ':', '-'])
results = {}
for nn in names:
    file_name = nn + '_results.pkl'
    results[nn] = ut.load_obj(prev_results_dir / nn / file_name)
# %% Scaling
keys = ['time', 'calcite', 'Ca', 'C', 'Si',
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

# %%
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
for k in range(0, len(comp)):
    plt.figure(figsize=(8, 4), dpi = 200)
    # for i in range(0, len(names)):
    for i in range(0, len(names)):
        plt.plot(sres[names[i]]['time'], sres[names[i]][comp[k]],
                 ls=linetype[i], label=label[i])
    plt.ylabel(ylabel[k], fontsize=14)
    # plt.title(titles[k])
    plt.xlabel('Time (s)', fontsize=14)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    file_name = fname + suffix[k] + '.png'
    plt.savefig(figures_dir / file_name)
    plt.close()
# %% ph
    ph_csh = np.load(prev_results_dir / names[0]/ 'pH.npy')
    # ph_ch = np.load(prev_results_dir / names[1]/ 'pH.npy')
    # correction
    # ph_csh[7] = (ph_csh[6] + ph_csh[8] )/2
    # ph_csh[11] = (ph_csh[10] + ph_csh[12] )/2
    # ph_csh[13] = (ph_csh[12] + ph_csh[14] )/2
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.imshow([ph_csh[1:-1]]) # , ph_ch[1:-1]
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
    de_csh = np.load(prev_results_dir / names[0] / 'De.npy')
    # ph_ch = np.load(prev_results_dir / names[1]/ 'pH.npy')
    # correction
    # ph_csh[7] = (ph_csh[6] + ph_csh[8] )/2
    # ph_csh[11] = (ph_csh[10] + ph_csh[12] )/2
    # ph_csh[13] = (ph_csh[12] + ph_csh[14] )/2
    print(de_csh[1:-1])
    poros_csh = np.load(prev_results_dir / names[0] / 'poros.npy')
    print(poros_csh[1:-1])