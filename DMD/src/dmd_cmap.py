from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import os 
import numpy as np

def colormap_dmd():
    path = os.path.dirname(os.path.abspath(__file__))
    cmap = np.load(os.path.join(path, 'CCool.npy'))
    n_bins = cmap.shape[0]
    name_cmap = 'DMD'
    cm = LinearSegmentedColormap.from_list(name=name_cmap, colors=cmap, N=n_bins)
    return cm

if __name__ == '__main__':
    colormap_dmd() 

def r_Spectral():
    color_map = plt.cm.get_cmap('Spectral')
    return color_map.reversed()