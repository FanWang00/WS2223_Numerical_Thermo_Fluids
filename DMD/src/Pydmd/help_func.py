import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
# sys.path.append('../../src')
# print(os.path.dirname(os.path.abspath(__file__)))
from .dmd import DMD
from .cdmd import CDMD
from .mrdmd import MrDMD
from .kdmd import KDMD
from .sdmd import SDMD
from scipy.integrate import trapz

# def dmd_cri16(Psi, Phi):
    
#     Phi = np.linalg.norm(Phi, axis=0)**2
#     # Psi = np.sum(np.absolute(Psi), axis=1)
#     Psi = trapz(np.absolute(Psi), axis=1)
#     cri_idx = Psi*Phi
#     return np.argsort(cri_idx)
def plot_total_err_vs_rank(L1_err, L2_err, rank):

    fig, ax1 = plt.subplots(figsize=(16, 10))
    # for r, l1, l2 in zip(rank, L1_err, L2_err):
    p1= ax1.plot(rank, L1_err, color='b', marker='o', label=f"total L1 error")
    
    ax1.set_title("Total error")
    # ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    p2 = ax2.plot(rank, L2_err, color='g', marker='x', label=f"total L2 error")
    pp = p1 + p2
    labs = [l.get_label() for l in pp]
    ax1.set_ylabel('total L1 error') 
    ax1.legend(pp, labs, loc=0)

        # color = 'tab:blue'
    ax2.set_ylabel('total L2 error')  # we already handled the x-label with ax
    # ax2.tick_params(axis='y', labelcolor=color)
    plt.show()



def plot_recons_err_rank(snapshots, dmd_str='dmd', svd_rank=[0, -1], **kwargs):

    switcher={
                'dmd':DMD,
                'cdmd':CDMD,
                'mrdmd':MrDMD,
                'sdmd':SDMD,
                'kdmd':KDMD
            }
    dmd_obj = switcher[dmd_str]
    print(f"dmd mode: {dmd_str} with setup {kwargs}")
    L1_snap = np.linalg.norm(snapshots, axis=0, ord=1)
    L2_snap = np.linalg.norm(snapshots, axis=0, ord=2)
    # marker = [j for i, j in enumerate(mpl.markers.MarkerStyle.markers.keys()) if i < 25 and j != '_' and j != '|' ]
    colors = plt.cm.jet(np.linspace(0, 1, len(svd_rank)))
    L1_err = []
    L2_err = []
    L1_err_tot = []
    L2_err_tot = []
    # svd_rank = np.array(svd_rank)
    # if len(svd_rank) > len(marker):
    #     raise ValueError('too many values for svd_rank!')
    x = [i for i in range(snapshots.shape[-1])]

    for i, rank in enumerate(svd_rank):
        
        dmd = dmd_obj(svd_rank=rank, **kwargs)
        print(f"current rank: {rank}")
        dmd.fit(snapshots)
        # reocns_data = self.reconstructed_data_save('no_reshape')
        # if isinstance(dmd.reconstructed_data, np.ndarray):
        reocns_data = dmd.reconstructed_data
        # else:
        #     reocns_data = dmd.reconstructed_data()
        
        if  reocns_data.shape != snapshots.shape:
            raise ValueError(f"shape of snapshots ({snapshots.shape}) and recons data ({reocns_data.shape}) does not match!")
        # print(reocns_data.shape)

        L1_err_tot_tmp = np.linalg.norm((snapshots - reocns_data), ord=1, axis=0)
        L1_err_tot.append(L1_err_tot_tmp)
        L1_err_tmp = L1_err_tot_tmp/L1_snap

        # print(L1_err_tmp.shape)
        L1_err.append(L1_err_tmp)
        L2_err_tot_tmp = np.linalg.norm((snapshots - reocns_data), ord=2, axis=0)
        L2_err_tot.append(L2_err_tot_tmp)
        L2_err_tmp = L2_err_tot_tmp/L2_snap
        L2_err.append(L2_err_tmp.flatten())
        print(dmd.r_svd)
        if rank < 1:
            svd_rank[i] = dmd.r_svd
        # dmd = None
    
    legend_idx = np.argsort(svd_rank)
    
    fig, axs = plt.subplots(figsize=(16, 10))
    for i, (l1, l2) in enumerate(zip(L1_err, L2_err)):
        axs.plot(x, l1, color=colors[i], marker='o', label=f"L1 norm with rank {svd_rank[i]}", markevery=50)
        axs.plot(x, l2, color=colors[i], marker='x', label=f"L2 norm with rank {svd_rank[i]}", markevery=50)
    # print(legend_idx)

    axs.set_ylabel(r'$\frac{\left\Vert X-X_{recons}  \right\Vert_p}{\left\Vert X \right \Vert_p}$', 
                    fontsize=30)
    axs.set_xlabel('Time step', fontsize=30)
    handles, labels = axs.get_legend_handles_labels()
    handles = np.reshape(handles, (-1, 2))
    labels = np.reshape(labels, (-1, 2))
    
    # print(f"handle: {handles.shape}")
    # print(f"labels: {labels.shape}")
    # handles_sort = [handles[i] for i in legend_idx]
    handles_sort = np.ravel([handles[i] for i in legend_idx])
    labels_sort = np.ravel([labels[i] for i in legend_idx])
    # print(handles_sort)
    axs.legend(handles_sort, labels_sort, loc='upper left', bbox_to_anchor=(1, 1))
    # axs.legend(loc='upper left')
    plt.yscale('log')
    plt.title('reconstruction error')
    plt.show()
    return  L1_err_tot, L2_err_tot, svd_rank
# def colormap_dmd():
#     path = os.path.dirname(os.path.abspath(__file__))
#     cmap = np.load(os.path.join(path, '../CCool.npy'))
#     n_bins = cmap.shape[0]
#     name_cmap = 'DMD'
#     cm = LinearSegmentedColormap.from_list(name=name_cmap, colors=cmap, N=n_bins)
#     return cm