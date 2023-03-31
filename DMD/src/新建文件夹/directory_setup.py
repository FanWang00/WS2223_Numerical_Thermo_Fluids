import os 
import sys
src_path = ['python/src', '../../src']
for path in src_path:
    if os.path.exists(path):
        sys.path.append(path)
        print(f"src path: {path}")
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import copy
import pickle
plt.style.use('ggplot')
import h5py
# import dmd_tools
import utils_dmd
# from dmd_dev import plot_EigVal
import comm_tools
import dmd_tools
# import dmd_plot
from get_dir_base import (dir_base_list, getBaseDir, InterpBasedir, dmd_fit_path,
                          DatasetBase, pvpython_path, mpiexec_pv_path, VORLL_path)

# def _key2item():
#     param_lst = ['velocityU', 'pressure', 'velocityX', 'velocityY', 'density']
#     interp_lst = ['linear', 'cubic']
#     shock_mode_lst = ['WithShock', 'WithoutShock']
    
#     read_param_idx = dict(zip(param_lst, [i for i in range(len(param_lst))]))
#     save_param_idx = dict(zip(param_lst, [i for i in  range(len(param_lst))]))
#     interp_idx = dict(zip(interp_lst, [i for i in range(len(interp_lst))]))
#     shock_mode_idx = dict(zip(shock_mode_lst, [i for i in range(len(shock_mode_lst))]))

def dir_setup(r_level=3, interpm='linear', readparam='velocityU', saveparam='velocityU', shockmode='WithShock', 
              dmd_mode='dmd', N=299, test_mode=True, pre_load=False, save_dmd_fit=False):
    
    param_lst = ['velocityU', 'pressure', 'velocityX', 'velocityY', 'density']
    interp_lst = ['linear', 'cubic']
    shock_mode_lst = ['WithShock', 'WithoutShock']
    
    read_param_idx = dict(zip(param_lst, [i for i in range(len(param_lst))]))
    save_param_idx = dict(zip(param_lst, [i for i in  range(len(param_lst))]))
    interp_m_idx = dict(zip(interp_lst, [i for i in range(len(interp_lst))]))
    shock_mode_idx = dict(zip(shock_mode_lst, [i for i in range(len(shock_mode_lst))]))
    # save_recons = False
    # save_modes = True
    # save_sep_modes = False
    # dmd_mode = 'dmd'
    # plot_modes = False
    # mid_name = ''
    # test_mode = False
    # write_multip = True
    dir_base = dir_base_list()

    # pre_load= False
    # save_dmd_fit = True
    if pre_load and save_dmd_fit:
        raise ValueError("pre load and save fit obj can not be True at same time!")
    read_param_list = [['velocityU'], ['pressure'], ['velocityX'], ['velocityY'], ['density']]
    # save_param = ['velocityX', 'velocityY', 'velocityU', 'density','pressure']
    save_param_list = [['velocityU'], ['pressure'], ['velocityX'], ['velocityY'], ['density']]
    # r_level = 3
    shock_mode_list = ['WithShock', 'WithoutShock']
    interp_m_list = ['linear', 'cubic']
    # base_path = '/local/disk1/fanwang/ALPACA_CWD/data_gen_temp/PreProcess/interp/cubic/'

    # read_param = read_param_list[0]
    # # save_param = save_param_list[-1]
    # save_param = read_param
    # shock_mode = shock_mode_list[0]
    # interp_m = interp_m_list[0]
    read_param = read_param_list[read_param_idx[readparam]]
    # save_param = save_param_list[-1]
    save_param = save_param_list[save_param_idx[saveparam]]
    shock_mode = shock_mode_list[shock_mode_idx[shockmode]]
    interp_m = interp_m_list[interp_m_idx[interpm]]
    
    input_file_base = InterpBasedir(r_level, shock_mode, interp_m=interp_m)
    input_file_path = input_file_base.raw_path
    # dmd_npy_path = '/media/overflow/Volume/ALPACA_CWD/Postprocess/dmd_fit/L3/WithShock'
    # i = 0
    # print(f"interp_base_path: {interp_base_path}")
    # print(f"fit dmd class save path: {dmd_save_path()}")

    groupname_dict = dict(zip(['points', 'cells', 'point_data', 'cell_data'],
                            ['domain/vertex_coordinates', 'domain/cell_vertices',
                            'simulation', 'simulation']))  

    modes_sep_base = getBaseDir('sep_modes', dir_base=dir_base, r_level=r_level, interp_m=interp_m, shock_mode=shock_mode, dmd_mode=dmd_mode, 
                                item=save_param[0])
    recons_base = getBaseDir('recons', dir_base=dir_base, r_level=r_level, shock_mode=shock_mode, interp_m=interp_m, dmd_mode=dmd_mode, 
                            item=save_param[0])
    dy_base = getBaseDir('dynamics', dir_base=dir_base, r_level=r_level, shock_mode=shock_mode, interp_m=interp_m, dmd_mode=dmd_mode, 
                            item=save_param[0])
    modes_base = getBaseDir('modes', dir_base=dir_base, r_level=r_level, interp_m=interp_m, shock_mode=shock_mode, dmd_mode=dmd_mode, 
                            item=save_param[0])
    dmd_fit_save_path = dmd_fit_path(dir_base=dir_base, r_level=r_level, interp_m=interp_m, shock_mode=shock_mode, dmd_mode='dmd', 
                                    item=save_param[0], extr='')
    
    print(f'save fit object: {save_dmd_fit}')
    if save_dmd_fit:
        print(f'fit object savepath: {dmd_fit_save_path}')
    print(f'preload dmd_fit obj: {pre_load}')
    if pre_load:
        print(f'preload dmd_fit obj path: {dmd_fit_save_path}')
        
    modes_base_dir = modes_base.path
    recons_base_dir = recons_base.path
    modes_sep_base_dir = modes_sep_base.path
    dy_base_dir = dy_base.path
    print(f'input data param: L{r_level}, {interp_m}, {shock_mode}, {save_param[0]}')
    print(f"recons_base_path: {recons_base_dir}")
    print(f"modes_base_path: {modes_base_dir}")
    print(f"modes_sep_base_path: {modes_sep_base_dir}")
    print(f"dynamics_base_path: {dy_base_dir}")
    print(f"dmd_fit_save_path: {dmd_fit_save_path}")
    
    time_ref_path = comm_tools.dir_list(input_file_path, 'aero*.xdmf')
    time_series = utils_dmd.extract_time_xdmf(time_ref_path[0])
    cell_info = np.load(os.path.join(input_file_path, 'aerobreakup_interp_cell.npz'))
    pattern = '*.h5'
    file_name = []
    # file_path = glob.glob(os.path.join(work_path, input_file_dir, pattern))
    file_path = comm_tools.dir_list(input_file_path, pattern)  

    file_path = np.sort(file_path)
    file_path = file_path[N:]
    time_series = time_series[N:]
    if test_mode:
        n = 10
        file_path = file_path[:n]
        time_series = time_series[:n]
    # print(file_path)

    file_name = [os.path.splitext(os.path.basename(path))[0] for path in file_path]
    print(f"file path check: {file_path[0]}")
    print(f"read param: {read_param}")
    print(f"numbers of data to load: {len(file_path)}")
    print(f"file name check: {file_name[0]}")
    VORLL_base = VORLL_path()
    return file_path, recons_base, dy_base, modes_sep_base, dmd_fit_save_path, VORLL_base, cell_info

