import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
import comm_tools
from dmd_cmap import colormap_dmd
from multiprocessing import Pool
from functools import partial
from PIL import Image
import copy
from scipy.integrate import trapz
from pvpy.img2video import ffmpeg_convert
# sys.path.append('../../src')
# print(os.path.dirname(os.path.abspath(__file__)))
from sklearn.preprocessing import MinMaxScaler

def ball_2d(X, Y, r):

    theta = np.asarray([i for i in range(1000)]) / 100*2*np.pi
    x = X + r * np.sin(theta)
    y = Y + r * np.cos(theta)
    return x, y

def St_number(f, D=0.0048, U=318.31588):
    """Strouhal number"""
    # for Re =100
    # dt = 0.2
    # T = 300 * dt
    # Su = 0.16
    # f = Su * U / D
    return f * D / U

# def plot_recons_error(recons_data, snaps):
        
#         fig, axs = plt.subplots(figsize=(8, 5))
#         axs.plot([i for i in range(recons_data.shape[-1])], 
#                     np.linalg.norm(snaps - recons_data, axis=0)/np.linalg.norm(snaps, axis=0),
#                     marker='v', label="L2 norm")
#         axs.plot([i for i in range(recons_data.shape[-1])], 
#                     np.linalg.norm(snaps - recons_data, axis=0, ord=1)/np.linalg.norm(snaps, axis=0, ord=1),
#                     marker='o', label="L1 norm")
#         axs.set_ylabel(r'$\dfrac{\left\Vert X-X_{recons}  \right\Vert_p}{\left\Vert X \right \Vert_p}$', 
#                         fontsize=20)
#         axs.set_xlabel('Time step', fontsize=20)
#         plt.legend(loc='upper right')
#         plt.xticks(size=14)
#         plt.yticks(size=14)
#         plt.yscale('log')
#         plt.title('reconstruction error')
#         return axs
    
def plot_recons_error(recons_data, snaps, predict=False):
        

        L2_err_norm = np.linalg.norm(snaps - recons_data, axis=0)/np.linalg.norm(snaps, axis=0)
        L1_err_norm =  np.linalg.norm(snaps - recons_data, axis=0, ord=1)/np.linalg.norm(snaps, axis=0, ord=1)

        fig, axs = plt.subplots(figsize=(12, 5))
        axs.plot([i for i in range(recons_data.shape[-1])], 
                    L2_err_norm,
                    marker='v', label="L2 norm")
        axs.plot([i for i in range(recons_data.shape[-1])], 
                    L1_err_norm,
                    marker='o', label="L1 norm")
        axs.set_ylabel(r'$\dfrac{\left\Vert X-X_{recons}  \right\Vert_p}{\left\Vert X \right \Vert_p}$', 
                        fontsize=20)
        axs.set_xlabel('Time step', fontsize=14)
        # plt.ticklabel_format(axis="y", style="sci")
        plt.legend(loc='upper right')
        plt.xticks(size=20)
        plt.yticks(size=20)
        print(f'max L1 err: {L1_err_norm.max()}')
        print(f'max L2 err: {L2_err_norm.max()}')
        
        if not predict:
            plt.yscale('log')

        # if plot_max_err:
       
        #     L1_err_norm_max = L1_err_norm.max()
        #     L2_err_norm_max = L2_err_norm.max()
        #     locsx, _ = plt.xticks()
        #     xmin = locsx.min()
        #     xmax = locsx.max()
        #     locs, labels = plt.yticks()
        #     # locsx, labels = plt.yticks()
        #     locs = np.append(locs, [L1_err_norm_max, L2_err_norm_max], axis=0)
        #     sort_arg = np.argsort(locs)
        #     locs = np.sort(locs)
        #     labels = np.asarray(list(labels)+[f'{L1_err_norm_max:.2f}', f'{L2_err_norm_max:.2f}'])
        #     plt.yticks(locs[sort_arg], labels[sort_arg])
        #     print(labels[sort_arg])
            
            # plt.hlines(L1_err_norm_max, xmin=xmin, xmax=xmax, colors='r', linestyles='--')
            # plt.hlines(L2_err_norm_max, xmin=xmin, xmax=xmax, colors='r', linestyles='--')
        plt.title('Reconstruction Error')
        return axs

def load_npz(file_path, item):
    dataset = np.load(file_path)
    nx = dataset['nx']
    ny = dataset['ny']
    snaps = dataset[item]
    file_name = dataset['file_name']
    time_series = dataset['time_series']
    return snaps, nx, ny, file_name, time_series

def skl_scaler(x, min=0, max=1):
    scaler = MinMaxScaler((min, max), copy=False)
    x_shape = x.shape 
    x_res = scaler.fit_transform(x.flatten()[:, np.newaxis])
    x_res = x_res.squeeze().reshape(x_shape)
    return x_res

def nameByPath(filepath, pos=-4):
  
    # s_all = {
    #         'WithShock': 'W',
    #         'WithoutShock': 'Wo',
    #         'velocityU': 'U',
    #         'pressure': 'p',
    #         'velocityX': 'X',
    #         'velocityY': 'Y',
    #         'linear': 'Lin',
    #         'cubic': 'Cub'
    #     }
        
    s_level = ['L'+str(i) for i in range(4)]
    s_level = dict(zip(s_level, s_level))
    
    s_dmd = ['dmd', 'spdmd', 'mrdmd']
    s_dmd = dict(zip(s_dmd, s_dmd))

    s_shock_mode = {
                    'WithShock': 'W',
                    'WithoutShock': 'Wo'
                    }
    s_item = {
                'velocityU': 'U',
                'pressure': 'p',
                'velocityX': 'X',
                'velocityY': 'Y',
                'density'  : 'd'
                }
    s_interp_m = {
                'linear': 'Lin',
                'cubic': 'Cub'
                 }

    s_all = {**s_shock_mode, **s_item, **s_interp_m, **s_level, **s_dmd}
    # print(s_all)
    p_split = list(filter(None, filepath.split(os.sep)[pos:]))
    # print(filepath)
    # print(p_split)
    name = []
    for i in p_split:
        name.append(s_all[i])
        # print(name)
    name = '_'.join(name)
    # print(s_all)
    # name = f'L{r_level}{s_shock_mode[shock_mode]}{s_item[item]}{dmd_mode}{s_interp_m[interp_m]}'
    return name 

def combFileName(filepath, r_level, shock_mode, interp_m, item, base_spec_name='', dmd_mode='', pos=-4):

    s_shock_mode = {
                    'WithShock': 'W',
                    'WithoutShock': 'Wo'
                    }
    s_item = {
                'velocityU': 'U',
                'pressure': 'p',
                'velocityX': 'X',
                'velocityY': 'Y'
                }
    s_interp_m = {
                'linear': 'Lin',
                'cubic': 'Cub'
                 }
    
    name = f'L{r_level}{s_shock_mode[shock_mode]}{s_item[item]}{dmd_mode}{s_interp_m[interp_m]}'
    return name 
    
def time_step_str(tp):
    tp_str = [f'time step={i:.4E}' for i in tp]
    return tp_str

def file_name_gen(N, base=None, prefix='test', suffix=''):
    digs = comm_tools.count_digits(N)
    name = [f'{prefix}_{i:0{digs}}{suffix}' for i in range(N)]
    if base is not None:
        name = [os.path.join(base, n) for n in name]
    return name

def img_simple_plot(x, sym_norm=False, norm=None, w=14, h=6, show_cmap=True, cmap=colormap_dmd()):
    fig, axs = plt.subplots(figsize=(w, h),
                            subplot_kw={'xticks': [], 'yticks': []},
                            sharex='col', sharey='row',
                            # gridspec_kw={'hspace': 0, 'wspace': 0},
                            gridspec_kw={'wspace': 0})

    if sym_norm:
        v = np.absolute(x).max()
        # vmin = -v
        # vmax = v
        norm_p = plt.Normalize(-v, v)
    else: 
        norm_p = norm
    im = axs.pcolormesh(x, norm=norm_p, cmap=cmap)
    if show_cmap:
        cbar = fig.colorbar(im, ax=axs, shrink=1)
    # im = axs[i].pcolormesh(value, vmin=vmin, vmax=vmax, cmap=cmap)
    axs.label_outer()
    axs.set_title(f"mode")
    
    # notice that here we use ax param of figure.colorbar method instead of

    # the cax param as the above example

    # cbar = fig.colorbar(images[-1], ax=axs.tolist(), shrink=0.75)

    # cbar.set_ticks(np.arange(0, 1.1, 0.5))
    # cbar.set_ticklabels(['low', 'medium', 'high'])
    # fig.suptitle(title, fontsize=30)
    # fig.tight_layout()
    plt.show()


def _png_grid_comb_save_mpi(n_ele, nrow, new_width, new_height, bg_color, save_dir, 
                            png_path, fn):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    mpi_save = partial(_png_grid_comb_save, n_ele, nrow, new_width, new_height, bg_color, save_dir)
    if rank == 0:
        p_f = list(zip(png_path, fn))
        idx_glb = np.asarray([i for i in range(len(png_path))])
        sendbuf = idx_glb
        ave, res = divmod(len(sendbuf), nprocs)
        count = [ave + 1 if p < res else ave for p in range(nprocs)]
        # idx = np.asarray([i for i in range(len(png_path))])
        count = np.array(count)
        # print(count)
        # displacement: the starting index of each sub-task
        displ = [sum(count[:p]) for p in range(nprocs)]
        displ = np.array(displ)
    else:
        sendbuf = None
        # initialize count on worker processes
        count = np.zeros(nprocs, dtype=np.int).tolist()
        displ = None
    # broadcast count
    comm.Bcast(count, root=0)
    comm.bcast(p_f, root=0)
    recvbuf = np.zeros(count[rank])

    comm.Scatterv([sendbuf, count, displ], recvbuf, root=0)

    print('After Scatterv, process {} has data:'.format(rank), recvbuf)
    # print(type(recvbuf))
    comm.Barrier()
    # idx = [displ + i for i in range(count)]
    # new_sendbuf = recvbuf ** 2 
    for i in range(len(recvbuf)):
        mpi_save(i[0], i[1])
    
    # print(new_sendbuf)
    # comm.Barrier()
    # new_rev = np.zeros((N, M))
    # comm.Gatherv(new_sendbuf, [new_rev, count, displ, MPI.DOUBLE], root=0)
    # # comm.Barrier()
    # if rank == 0:
    #     print(f"result:{new_rev}")
    #     return new_rev

def _png_grid_comb_save(n_ele, nrow, new_width, new_height, bg_color, save_dir, 
                        png_path, fn):
    x = 0
    y = 0
    new_img = Image.new('RGBA', (new_width*(n_ele), new_height*(nrow)), color=bg_color)
    images = [Image.open(p) for p in png_path]
    for img in images:
        
        # print(img)
        # img = img.resize((eachSize, eachSize), Image.ANTIALIAS)
        new_img.paste(img, (x*new_width, y*new_height))
        # print(x*new_width, y*new_height)
        # print(n_ele, nrow)
        x += 1
        
        if x == n_ele:
            x = 0
            y += 1
    new_img.save(os.path.join(save_dir, fn), format='png')
        
        
def png_grid_comb(img_path,  file_name=None, save_dir='test_comb', n_ele=None, MP=False, nproc=2,
                  to_video=False, video_path='test.mp4', remove_png_source=False):
    '''combine pngs to grid-like
    images: Img obj 2D list size=(T, N) T: time step, N: items for each time step 
    n_ele:  numer of element for one row, default N
    save_dir: 
    video_path : abs path
    '''    
    comm_tools.make_dir(save_dir)
    if len(comm_tools.dim_list(img_path)) == 1:
        img_path = [[i] for i in img_path]

    bg_color = (255, 255, 255)
    # images = np.empty(tuple(img_size), dtype=object)
    img_size = comm_tools.dim_list(img_path)
    N = img_size[1]
    T = img_size[0]
    # T, N = images.shape    
    if n_ele is None:
        n_ele = img_size[1]
    nrow = int(np.ceil(N/n_ele))
    if file_name is None:
        file_name = file_name_gen(T, base='test', suffix='.png')
# print(nrow)
    # print(nrow)
    # if direction=='horizontal':
    # img_flat = []
    # img_flat = [i for j in images for i in j]
    # # print(img_flat)
    # widths, heights = zip(*(i.size for i in img_flat))
    # new_width = max(widths)
    # new_height = max(heights) 
    with Image.open(img_path[0][0]) as get_new_size:

        new_width, new_height = get_new_size.size
    # new_width = widths
    # new_height = heights 
    # print(new_width, new_height)
    # print(nrow, n_ele)
    if MP:
        MP_save = partial(_png_grid_comb_save, n_ele, nrow, new_width, new_height, bg_color, save_dir)
        comm_tools.func_MP(MP_save, zip(img_path, file_name), nproc)
    else:  
        for path, fn in zip(img_path, file_name):
            _png_grid_comb_save(save_dir, fn, n_ele, nrow, path, new_width, new_height, bg_color)
    if to_video:
        ffmpeg_convert(save_dir+'/*.png', outfile=video_path, remove_png_source=remove_png_source)

    
def png_grid_comb_save(Img_obj, save_path):
    '''save multiple pngs in grid
    Img_obj:   input file path, str list, size = (m, n), m, n are same size (row, column) of output grid pic 
    save_dir:   outoput dir
    # h :         high of each single  
    '''
    # img_size = comm_tools.dim_list(Img_obj)
    # w, h = Img_obj.size
    # new_img = 
    img_row = []
    [img_row.append(comm_tools.append_images(f, direction='horizontal') for f in Img_obj)]
    new_img = comm_tools.append_images(img_row, direction='vertical')
    new_img.save(save_path, format='png')

def _combine_mode_dy_save(img_1_resize, save_dir, h, fn, r_path):   
    '''save func for combine_mode_dy, separate for MP 
    '''
# for fn, r_path in zip(f_name, path_recons):
# def save(f_name, path_recons, img_1_resize):
    # print(fn, r_path[0])
    img = copy.deepcopy(img_1_resize)
    img_recons = list(map(Image.open, r_path))
#     r_recons = img_recons[0].shape[0] / img_recons[0].shape[1]
    r_recons = img_recons[0].size[0] / img_recons[0].size[1]
    img_recons_resize = []
    [img_recons_resize.append(i.resize((int(h*r_recons), h), Image.ANTIALIAS)) for i in img_recons]
    img.append(img_recons_resize) 
    img = list(map(list, zip(*img)))
    new_img_tmp = []
    for j in range(len(img)):
        new_img_tmp.append(comm_tools.append_images(img[j]))
    new_img = comm_tools.append_images(new_img_tmp, direction='vertical')
    save_path = os.path.join(save_dir, f"{fn}.png")
    # print(save_path)
    new_img.save(save_path, format='png')
    

def combine_mode_dy(path_real, path_img, path_dy, path_recons, save_dir, f_name=None, h=100, prefix=None, 
                    MP=False, nproc=2, 
                    to_video=False, video_path='test.mp4', remove_png_source=False):
    
    '''save real, img part of mode , dynamics, and each mode corresponding recons  
    
    path_real:      1D list of real part mode of path 
    path_img:       1D list of img part mode of path 
    path_dy:        1D list of real part mode of path 
    path_recons:    path of recons 2D list, size=(T, N) T: time steps, N: number of modes    
    save_dir:       save dir of png output
    f_name:         outfile name, 1D list, if is None creates by number, starts with 0
    h=100 :         hight of single pics
    '''
    if f_name is None:
        if prefix is None:
            raise ValueError('Please input prefix!')
        N = len(path_recons)
        digs = comm_tools.count_digits(N)
        f_name = [f'{prefix}_{i:0{digs}}' for i in range(N)]
    # resize of pics
    img_1 = [list(map(Image.open, path_real)), list(map(Image.open, path_img)), list(map(Image.open, path_dy))]
    # raise ValueError('')
    # img = [ ]
    r_s = [img_1[0][0].size, img_1[1][0].size, img_1[2][0].size]
    ratio = [i[0]/i[1] for i in r_s]
    img_1_resize = []
    comm_tools.make_dir(save_dir)
    for i in range(3):
        img_tmp = []
        for j in img_1[i]:
            img_tmp.append(j.resize((int(h*ratio[i]), h), Image.ANTIALIAS))
        img_1_resize.append(img_tmp)
    ## combine real and img mode to one pic
    
    if MP:
        _save_par = partial(_combine_mode_dy_save, img_1_resize, save_dir, h)
        pool = Pool(nproc)
        pool.starmap(_save_par, list(zip(f_name, path_recons)))
        pool.terminate()
        pool.join()
    else:
        for fn, r_path in zip(f_name, path_recons):
            _combine_mode_dy_save(img_1_resize, save_dir, h, fn, r_path)
    if to_video:
        ffmpeg_convert(save_dir+"/./*.png", outfile=video_path, remove_png_source=remove_png_source)
        
        
        

def check_dim(*params):
    d = []
    for item in params:
        if isinstance(item, np.ndarray):
            d.append(item.shape[0])
        elif isinstance(item, list):
            d.append(len(item))
        else:
            raise TypeError(f'Please check the type of parameter {comm_tools.retrieve_name(item)}')
    dc = d[0]
    check = [d[i] == dc for i in range(len(d))]
    if not all(check):
        raise ValueError(f'somthing wrong in dim of parameters. dim={d}')
    else:
        print(f"dim check check successful! dim={d[0]}")
        return True

def mode_dynamics_video(nx, ny, save_img, modes, dynamics, MP, cmap,
                        time_step, savepath, recons):
    '''
    r: idx of snapshot
    mn: (m, n) flatten snapshot one frame
    t: time step 
    param: modes: dim = (mn, r)
    param: dynamics: dim = (t, r)
    idx: idx to plot    
    '''

    if MP:
        # recons = [recons]
        recons = recons[np.newaxis, :]
        time_step = [time_step]
        savepath = [savepath]        
    # if modes.ndim == 1:
    #     modes = modes[:, np.newaxis]
    if dynamics.ndim == 1:
        dynamics = dynamics[np.newaxis, :]
    p = dynamics.shape[0]
    checked = check_dim(time_step, savepath, recons)
    print(checked)
    # print(modes.shape)
    if not checked:
        print("program did not run, check input!")
    else:
        # print(dynamics.shape)
        # print(len(savepath))
        # print(modes[:, 0].dot(dynamics[0, :]).shape)
        # print(np.reshape(modes[:, [i]].dot(dynamics[[i], :]).real, (nx, ny, -1)).T.shape)
        # recons = [np.reshape(modes[:, [i]].dot(dynamics[[i], :]).real, (nx, ny, -1)).T for i in range(p)]
        modes = modes.reshape((nx, ny, -1)).T
        # dy_p = dynamics[idx, :]
        # print(recons[0].shape)
        print(comm_tools.dim_list(time_step))
        print(comm_tools.dim_list(savepath))
        print(recons.shape)
        iter_args = zip(time_step, savepath, recons)
        
        # tt = dynamics[-1]
        for step, path, data in iter_args:
            # print(f"data dim={data.shape}")
            fig, axs = plt.subplots(p, 4, subplot_kw={'xticks': [], 'yticks': []}, figsize=(4*20, p*6))
            
            for i in range(p):
                axs[i, 0].pcolormesh(modes[i].real, cmap=cmap)
                axs[i, 1].pcolormesh(modes[i].imag, cmap=cmap)
                axs[i, 2].plot(time_step, dynamics[i, :].real, label='Re')
                axs[i, 2].plot(time_step, dynamics[i, :].imag, label='Imag')
                # axs[i].set_title(f"dynamics of mode {i}")
                # axs[i, 1].legend(loc='upper right')
                axs[i, 3].pcolormesh(data[i], cmap=cmap)
            if save_img:
                plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close()
        

# def colormap_dmd():
#     path = os.path.dirname(os.path.abspath(__file__))
#     cmap = np.load(os.path.join(path, 'CCool.npy'))
#     n_bins = cmap.shape[0]
#     name_cmap = 'DMD'
#     cm = LinearSegmentedColormap.from_list(name=name_cmap, colors=cmap, N=n_bins)
#     return cm

def _save_multi_png_par(data, save_path, time_step, tp2, 
                        cm=colormap_dmd(), vmin=None, vmax=None, Nc=1, save_img=False, title='', 
                        sym_norm=False, nproc=2):
    
    par_func = partial(_save_multi_png, cm, vmin, vmax, Nc, save_img, title, sym_norm)
    param_iter = zip(data, save_path, time_step, tp2)
    func_MP(par_func, param_iter, nproc=nproc)
    

def _save_multi_png(cm, vmin, vmax, Nc, save_img, title, sym_norm,
                     data, save_path, time_step, tp2):

    if isinstance(data, str):
        d_szie = comm_tools.dim_list(data)
    elif isinstance(data, np.ndarray):
        d_size = data.shape
    else:
        raise TypeError('please check data type!')
    n = d_size[0]
    if tp2 is None:
        tp2 = [None]*n
    elif not isinstance(tp2, list):
        tp2 = [tp2]
    if time_step is None:
        time_Step = [None]*n
    elif not isinstance(time_step, list):
        time_step = [time_step] 
    nr = int(np.ceil(n/Nc))
    # ele_zip = zip(data, save_path, time_step, tp2)
    fig, axs = plt.subplots(nr, Nc, subplot_kw={'xticks': [], 'yticks': []}, figsize=(Nc*14, nr*6))   
    # if n != 1:
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
        # data = [data]
    # print(d.shape)
    # print('done')
    # print(d.shape)
    axs = axs.flatten()
    # print(nn)
    # print(axs.shape[0])
    for i in range(min(axs.shape[0], n)):
        # print(d[i])
        # print(min(axs.shape[0], nn))
        # print(comm_tools.dim_list(d))
        # print(sym_norm, vmin, vmax)
        if sym_norm:
            v = np.absolute(data[i]).max()
            # print(v)
        # vmin = -v
        # vmax = v
            norm_p = plt.Normalize(-v, v)
        elif vmin is not None or vmax is not None:
            norm_p = plt.Normalize(vmin, vmax)
        else:
            norm_p = None
        # image = axs[i].pcolormesh(d[i],  vmax=vmax[i], vmin=vmin[i], cmap=cm)
        # print(d.shape)
        image = axs[i].pcolormesh(data[i],  norm=norm_p, cmap=cm)
        # axs[i].set_title(title[i])
        # if i%2 ==0:  
        fig.colorbar(image, ax=axs[i], orientation='vertical', pad=0, fraction=0.05, shrink=1)
        # print('done')
        # print(n)
        # if MP:
        axs[i].text(0.01, 0.01, s=time_step[i], fontsize=16, fontweight='bold', ha='left', va='bottom', transform = axs[i].transAxes,)
        axs[i].text(0.5, 0.99, s=tp2[i], fontsize=16, fontweight='bold', ha='center', va='top', transform = axs[i].transAxes,)
            # axs[i].text(0.01, 0.99, s=step, fontsize=16, fontweight='bold', ha='left', va='top', transform = axs[i].transAxes,)
            # axs[i].text(0.5, 0.99, s=tp2[k], fontsize=14, fontweight='bold', ha='center', va='top', transform = axs[i].transAxes,)
        if save_img:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_save_png(data=None, save_path=None, tp2=None, time_step=None, cm=colormap_dmd(), 
                    sym_norm=False, vmin=None, vmax=None, Nc=1, nproc=2,
                    save_img=False, title='', MP=False):
    '''
    :param data: input data to save
    '''
    if isinstance(data, str):
        d_szie = comm_tools.dim_list(data)
    elif isinstance(data, np.ndarray):
        d_size = data.shape
    else:
        raise TypeError('please check data type!')

    n = d_size[0]
    if not isinstance(save_path, list):
        save_path = [save_path]
    if tp2 is None:
        tp2 = [None]*n
    # elif tp2
    if time_step is None:
        time_Step = [None]*n
    if MP:
        print('Multiprocessing saving mode...')
        _save_multi_png_par(data=data, save_path=save_path, time_step=time_step, tp2=tp2, 
                            cm=cm, vmin=vmin, vmax=vmax, Nc=Nc, save_img=save_img, title=title, 
                            sym_norm=sym_norm, nproc=nproc)
    else:
        for d, path, ts, p in zip(data, save_path, time_step, tp2):
            _save_multi_png(cm=cm, vmin=vmin,vmax=vmax, Nc=Nc, save_img=save_img, title=title, 
                            sym_norm=sym_norm, data=d, save_path=path, time_step=ts, tp2=p)


def func_MP(func, param_iter, nproc):
    pool = Pool(processes=nproc)
    pool.starmap(func, param_iter)
    pool.terminate()
    pool.join()
    
def save_png_plt_par_std(param_tuple, 
                         sym_norm=False, vmin=None, vmax=None, Nc=1, tp2=None, title='',
                         cm=colormap_dmd(), save_img=False, nproc=1, MP=True):

    par_MP = partial(save_png_plt_std, cm, sym_norm, vmin, vmax, Nc, save_img, tp2, title, MP)
    # print(len(param_tuple))
    func_MP(par_MP, param_tuple, nproc)
    
# def save_png_plt_mpi_std(param_tuple, 
#                          sym_norm=False, vmin=None, vmax=None, Nc=1, tp2=None, title='',
#                          cm=colormap_dmd(), save_img=False, nproc=1, MP=True):
#     from mpi4py import MPI

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     nprocs = comm.Get_size()
    
    
#     par_MP = partial(save_png_plt_std, cm, sym_norm, vmin, vmax, Nc, save_img, tp2, title, MP)
#     # print(len(param_tuple))
#     func_MP(par_MP, param_tuple, nproc)
    
    
def save_png_plt_std(cm=colormap_dmd(), sym_norm=False, vmin=None, vmax=None, Nc=1, save_img=False, tp2=None, title='', MP=False,
                    data=None, save_path=None, time_step=None):
    '''
    :param data: input data to save
    '''
    # if isinstance(data, list):
        # nr = len(data)
        # print('list')
    # if isinstance(data[0], np.ndarray):
    #     n = data[0].shape[0]
    # print(f'{save_path}')
    # print(f'{time_step}')
    # print(tp2)
    print(len(time_step))
    if isinstance(data, list):
        n = len(data)
    elif isinstance(data, np.ndarray):
        n = data.shape[0]
    else:
        raise ValueError('Wrong data type! data must be list or np.ndarray')
    # print(data.shape)
    if not MP:
        n = 1 
    if not save_img:
        save_path = [None]*n
    if vmin is None:
        # nn = len(data[0])
        vmin = [None]*n
    
    if vmax is None:
        # nn = len(data[0])
        vmax = [None]*n
    if title=='':
        # nn = len(data[0])
        title = ['']*n
    if tp2 is None:
        # nn = len(data[0])
        tp2 = ['']*n

    if time_step is None:
        time_step_str = ['']
    elif not isinstance(time_step, list):
        time_step_str = [time_step]
    # elif not isinstance(time_step, list):
    #     time_step_str = [f'timestep={time_step}']
    # else:
    #     time_step_str = [f'timestep={t}' for t in time_step]
    # print(len(save_path))
    # print(data.shape)
    if isinstance(save_path, str):
        # print('extend dim to data adn save path')
        data = [data]
        save_path = [save_path]
    # print(f'data {len(data)}')
    # print(f'save_path {len(save_path)}')
    # print(f'time_step_str {len(time_step_str)}')
    # print(len(save_path))
    # print(time_step)
    # print(data)
    # print(len(data))

    ele_zip = zip(data, save_path, time_step_str)
    # print(len(list(ele_zip)))
    im_p = []
    # i = 0
    # print(data.shape)
    for k, (d, path, step) in enumerate(ele_zip):
        # print(d.shape)
        # print(dim_list(d))
        # print(i)
        # i = i+1
        # nn = len(d)
        if isinstance(d, list):
            nn = len(d)
        elif isinstance(d, np.ndarray):
            nn = d.shape[0]
        # print(nn)
        nr = int(np.ceil(n/Nc))
        # print(nr)
        fig, axs = plt.subplots(nr, Nc, subplot_kw={'xticks': [], 'yticks': []}, figsize=(Nc*14, nr*6))   
        # if n != 1:
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
            # d = [d]
        # print(d.shape)
        # print('done')
        # print(d.shape)
        axs = axs.flatten()
        # print(nn)
        # print(axs.shape[0])
        for i in range(min(axs.shape[0], nn)):
            # print(d[i])
            # print(min(axs.shape[0], nn))
            # print(comm_tools.dim_list(d))
            # print(sym_norm, vmin, vmax)
            if sym_norm and vmin[i] is None and vmax[i] is None:
                v = np.absolute(d[i]).max()
                # print(v)
            # vmin = -v
            # vmax = v
                norm_p = plt.Normalize(-v, v)
            else: 
                norm_p = plt.Normalize(vmin, vmax)
            # image = axs[i].pcolormesh(d[i],  vmax=vmax[i], vmin=vmin[i], cmap=cm)
            # print(d.shape)
            image = axs[i].pcolormesh(d[i],  norm=norm_p, cmap=cm)
            axs[i].set_title(title[i])
            # if i%2 ==0:  
            fig.colorbar(image, ax=axs[i], orientation='vertical', pad=0, fraction=0.05, shrink=1)
            # print('done')
            # print(n)
            if MP:
                axs[i].text(0.01, 0.99, s=step, fontsize=16, fontweight='bold', ha='left', va='top', transform = axs[i].transAxes,)
                axs[i].text(0.5, 0.99, s=tp2[i], fontsize=16, fontweight='bold', ha='center', va='top', transform = axs[i].transAxes,)
            else:
                axs[i].text(0.01, 0.99, s=step, fontsize=16, fontweight='bold', ha='left', va='top', transform = axs[i].transAxes,)
                axs[i].text(0.5, 0.99, s=tp2[k], fontsize=14, fontweight='bold', ha='center', va='top', transform = axs[i].transAxes,)
        
        # axins1 = inset_axes(axs[0],
        #             width="80%",  # width = 50% of parent_bbox width
        #             height="5%",  # height : 5%
        #             loc='upper center')

        # cbar = fig.colorbar(image, ax=axs.tolist(), orientation='vertical', pad=0, fraction=0.05, shrink=0.75)
        # fig.colorbar(image, cax=axins1)
        if save_img:
            # print('done')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()
            # print(axes_list)
        # else:
        #     print('done')
        #     for ax in axs:
        #         image = ax.pcolormesh(d[i],  vmax=vmax[i], vmin=vmin[i], cmap=cm)
        #         print('done')
        #         ax.set_title(title[i])
        #         fig.colorbar(image, ax=ax, shrink=1)
        #         ax.text(0.05, 0.95, s=step, fontsize=40, fontweight='bold', va='top', transform = ax.transAxes,)
        #     if r_axes:
        #         axes_list.append(axs)
        #     if save_img:
        #         plt.savefig(path, bbox_inches='tight', pad_inches=0)
        #     plt.close()
        
def dmd_cri16(Psi, Phi):
    
    Phi = np.linalg.norm(Phi, axis=0)**2
    # Psi = np.sum(np.absolute(Psi), axis=1)
    Psi = trapz(np.absolute(Psi), axis=1)
    cri_idx = Psi*Phi
    
    return np.argsort(cri_idx)

# def save_dynamics(save_path_dir, prefix, sort_mode=None, save_num=None):
    
    
def plot_save_dynamics(save_path_dir, dy_data, prefix='test', save_num=None, time_step=None, W=4, H=3):
  
    r, x = dy_data.shape
    # if save_num is None:
    #     save_num = r
    # elif save_num >= r :
    #     save_num = r
    if time_step is None:
        t = [i for i in range(x)]
    else:
        t = time_step
    digs = comm_tools.count_digits(save_num)
    for i in range(save_num):
        fig, ax = plt.subplots(1, 1, figsize=(W, H))
        ax.plot(t, dy_data[i].real, label='Re')
        ax.plot(t, dy_data[i].imag, label='Img')
        ax.legend(loc='upper right')
        path = os.path.join(save_path_dir, f'{prefix}_{i:0{digs}}')
        plt.savefig(path, pad_inches=0)
        plt.close()
                

def plot_dynamics(v, Nr, Nc, x=None):
    '''
    x : x axis if plot
    '''
        
        # idx = self.item_sort_idx(m=sort_mode)
        # tt = self.dmd_timesteps
        # dynamics_real = self.dynamics[idx[:Nr*Nc], :].real
        # dynamics_imag = self.dynamics[idx[:Nr*Nc], :].imag
    if x is None:
        x = np.array([i for i in range(v.shape[1])])
    dy_r = v.real
    dy_im = v.imag
    # fig, axs = plt.subplots(Nr, Nc, figsize=(Nr*6, Nc*4))
    fig, axs = plt.subplots(nrows=Nr, ncols=Nc, figsize=(Nc*4, Nr*3),
                            # subplot_kw={'xticks': [], 'yticks': []},
                            # sharex='col',
                            # sharey='row',
                            # gridspec_kw={'hspace': 0, 'wspace': 0},
                            # gridspec_kw={'wspace': 0}
                            )
    axs = axs.flatten()
    # axs = [axs]
    handles = []
    labels = []
    h = min(Nc*Nr, v.shape[0])
    for i in range(h):
        # plt.xticks([])
        # plt.yticks([])
        # plot_idx = amp_idx[i]
        # print(plot_idx)
        # pos = axs[0].imshow(value_real, cmap=cm)

        # for ax, value in zip(axs, [dynamics_real, dynamics_imag]):

        axs[i].plot(x, dy_r[i], label='Re')
        axs[i].plot(x, dy_im[i], label='Imag')
        axs[i].set_title(f"dynamics of mode {i}")
        axs[0].legend(loc='upper right')
        #     ax.legend()    #     print(value)
            # axs[1].imshow(value_imag, cmap=cm)
            # axs[0].set_title(f"real part of dyanmics mode {plot_idx:03d}")
            # axs[0].set_title(f"imaginary part dynamics of mode {plot_idx:03d}")
            # plt.contour(value, levels=1, colors='k', linewidths=1.2)
            # # plt.colorbar(pos)
            # fig.subplots_adjust(right=0.8)
            # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            # fig.colorbar(im, cax=cbar_ax, extend='max')
        #     handle_tmp, label_tmp = ax.get_legend_handles_labels()
        #     handles.append(handle_tmp)
        #     labels.append(label_tmp)
        # handles, labels = axs[-1].get_legend_handles_labels()
        # axs[0].legend(handles, labels, bbox_to_anchor=(0, 1.35), loc='upper left')
        # Create the legend
        # fig.legend([l1, l2, l3, l4],     # The line objects
        #            labels=line_labels,   # The labels for each line
        #            loc="center right",   # Position of legend
        #            borderaxespad=0.1,    # Small spacing around legend box
        #            title="Legend Title"  # Title for the legend
        #            )
        # plt.label()
    fig.suptitle('dynamics of modes sorted by amplitudes', fontsize=30)
    # plt.tight_layout()
    plt.show()
        
def plot_dmd_item(values, Nr=1, Nc=5, vmin=None, vmax=None,
                    sym_norm=False, sort_mode=None,
                    cmap=colormap_dmd(), n_plot=None, title='',
                    save_fig=False, png_name='test.png'):
    # marker_size, col, num_marker  = self._plot_setup(n_plot=n_plot)
    # amp_max = self.item_sort[0]
    # amp_magnitude = np.absolute(self.amplitudes)
    # amp_idx = self.item_sort_idx()
            
    # value_real_amp =np.angle(dmd.modes[:, amp_idx[:nrows*ncols]])
    
    # for i, (ax, value) in enumerate (zip(axs.flat, np.reshape(values))):
    #     ax.imshow(value, cm=cmap)
    #     ax.set_title(f"dynamics of mode {i}")
    # if sort_mode is not None:
    #     idx = self.item_sort_idx(m=sort_mode)
    #     plot_idx = idx[:Nr*Nc]
    #     values = values[:, :, plot_idx]
    # else:
    #     values = values[:, :, Nr*Nc]
    # print(values.shape)


    images = []
        
    fig, axs = plt.subplots(nrows=Nr, ncols=Nc, figsize=(Nc*5, Nr*2),
                            subplot_kw={'xticks': [], 'yticks': []},
                            # sharex='col', sharey='row',
                            # gridspec_kw={'hspace': 0, 'wspace': 0},
                            # gridspec_kw={'wspace': 0}
                            )
    axs = axs.flatten()
    # print(values.shape)
    # print(axs.shape)
    # value_real_amp = copy.deepcopy(abs(dmd.modes[:, amp_idx[:Nc*Nr]]))
    # for ax, value in zip(axs.flat, np.reshape(value_real_amp, (nx, ny, -1)).T):
    #     images.append(ax.imshow(value, cmap=cm))
    #     ax.label_outer()
    for i, value in enumerate(values):

        if sym_norm and vmin is None and vmax is None:
            v = np.absolute(value).max()
            # vmin = -v
            # vmax = v
            norm_p = plt.Normalize(-v, v)
        else: 
            norm_p = plt.Normalize(vmin, vmax)
        images.append(axs[i].pcolormesh(value, norm=norm_p, cmap=cmap))
        # im = axs[i].pcolormesh(value, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[i].label_outer()
        axs[i].set_title(f"mode {i}", fontsize=20)
        cbar = fig.colorbar(images[i], ax=axs[i], pad=0, shrink=1)
        
        
    plt.subplots_adjust(top = 0.99, bottom = 0, right = 1, left = 0, 
                    hspace = 0.2, wspace = 0)
    plt.margins(0,0)
    # notice that here we use ax param of figure.colorbar method instead of

    # the cax param as the above example

    # cbar = fig.colorbar(images[-1], ax=axs.tolist(), shrink=0.75)

    # cbar.set_ticks(np.arange(0, 1.1, 0.5))
    # cbar.set_ticklabels(['low', 'medium', 'high'])
    # fig.suptitle(title, fontsize=30)
    # fig.tight_layout()
    if save_fig:
        fig1 = fig1 = plt.gcf()
        fig1.savefig(png_name)
        
    plt.show()

# class DMDRecons(object):
#     def __init__(self, modes, eigs, snapshots):
#         self.modes = modes
#         self.eigs = eigs 
#         self.snapshots = snapshots
    
#     @property
#     def amplitudes(self):
#         return _b
    
#     @property
#     def dynamics(self):
#         return Psi
#     def _b(self):
#      b = np.linalg.lstsq(self.modes, self.snapshots.T[0], rcond=None)[0]
#     return b
    
#     def Psi(self, dmd_timesteps):
#     """
#     Get the time evolution of each mode.
#     :return: the matrix that contains all the time evolution, stored by
#         row.
#     :rtype: numpy.ndarray
#     """
#     dt = dmd_timesteps[1] - dmd_timesteps[0]
#     omega = np.log(self.eigs) / dt
#     assert np.allclose(old_div(np.log(self.eigs), dt), np.log(self.eigs) / dt)
#     vander = np.exp(
#         np.outer(omega, dmd_timesteps - dmd_timesteps[0]))
#     return vander * self._b[:, None]

#     @property
#     def recons_data(self):
#     """
#     Get the reconstructed data.
#     :return: the matrix that contains the reconstructed snapshots.
#     :rtype: numpy.ndarray
#     """
#     return self.modes.dot(self.Psi).real

#     def recons_error(self):
        
#     #     reocns_data = self.reconstructed_data
#         # print(reocns_data.shape)
#         fig, axs = plt.subplots(figsize=(16, 10))
#         axs.plot([i for i in range(self.recons_data.shape[-1])], 
#                     np.linalg.norm(self.snapshots - self.recons_data, axis=0)/np.linalg.norm(self.snapshots, axis=0),
#                     marker='v', label="L2 norm")
#         axs.plot([i for i in range(self.recons_data.shape[-1])], 
#                     np.linalg.norm(self.snapshots - self.recons_data, axis=0, ord=1)/np.linalg.norm(self.snapshots, axis=0, ord=1),
#                     marker='o', label="L1 norm")
#         axs.set_ylabel(r'$\dfrac{\left\Vert X-X_{recons}  \right\Vert_p}{\left\Vert X \right \Vert_p}$', 
#                         fontsize=20)
#         axs.set_xlabel('Time step', fontsize=20)
#         plt.legend(loc='upper right')
#         plt.xticks(size=14)
#         plt.yticks(size=14)
#         plt.yscale('log')
#         plt.title('reconstruction error')
    