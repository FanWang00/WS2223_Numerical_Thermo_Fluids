


import numpy as np
import matplotlib.pyplot as plt
import h5py
import os 
import comm_tools
import combine_pics
import moviepy_video_func
from os import path as osp

## combin multi pngs to grid pic

# file_paths = [r'I:/LBM_RBC/test/test_new_500Ra5.0E+06_dt1.0E+00_Th1.0001647257468422/png/T',
#            r'I:/LBM_RBC/test/test_new_500Ra5.0E+06_dt1.0E+00_Th1.0003834415482804/png/T',
#            r'I:/LBM_RBC/test/test_new_500Ra5.0E+06_dt1.0E+00_Th1.0004016182315303/png/T',
#            r'I:/LBM_RBC/test/test_new_500Ra5.0E+06_dt1.0E+00_Th1.000073124723441/png/T'
#            ]
# T_chaos = []
# U_chaos = []
# file_paths = [r'I:\LBM_RBC\loop_rho_rand_1e-1\data\500Ra3.0E+06_dt1.0E+00_SD1830312345\png/T', 
#               r'I:\LBM_RBC\loop_rho_rand_1e-1\data\500Ra3.0E+06_dt1.0E+00_SD1830312345\png/u']

dir_root = r'I:\LBM_RBC\Ras'
# mode_num = [0,2,4,6,8,10]
file_paths = []
Ras_dirs_sort = []
Ras_dirs = os.listdir(dir_root)
dir_Ras_save = r'I:\LBM_RBC\Raspng'

if False:

    Ras_idx = 0
    for j, i in enumerate([0,5,1,6,2,3,4]):
        # Ras_dirs_sort.append(Ras_dirs[i])
        subfolder_name = f'Ras{j:02}'
        f_paths = []
        combine_files_path = []
        Rasp = Ras_dirs[i]
        Ras_T = osp.join(Rasp, 'png', 'T')
        Ras_U = osp.join(Rasp, 'png', 'U')
        T_paths = osp.join(dir_root, Ras_T)
        U_paths = osp.join(dir_root, Ras_U)
        # f_paths.append(T_paths)
        # f_paths.append(U_paths)
        
        for dp in [T_paths, U_paths]:
            f_p = comm_tools.dir_list_glob(dp, '*.png')
            # for p in path_list:
            combine_files_path.append(f_p)
        
        combine_files_path = comm_tools.list_T(combine_files_path)
        combine_names = [f'combine_T{i:04}.png' for i in range(len(combine_files_path))] 
        combine_sub_folder = osp.join(dir_Ras_save, subfolder_name)
        combine_pics.png_grid_comb(combine_files_path,  file_name=combine_names, save_dir=combine_sub_folder, n_ele=1, MP=False, nproc=4,
                        to_video=False, video_path='test.mp4', remove_png_source=False)

TU_paths = []
Rascombine_folder = r'I:\LBM_RBC\Raspng'
TU_dirs = os.listdir(Rascombine_folder)

for m_dir in TU_dirs:
    # for num in mode_num:
    TU_paths.append(os.path.join(Rascombine_folder, m_dir))

combine_files_path = []
# data_idx = [2, 250]
for dp in TU_paths:
    path_list = comm_tools.dir_list_glob(dp, '*.png')
    # for p in path_list:
    combine_files_path.append(path_list)


combine_files_path = comm_tools.list_T(combine_files_path)


combine_names = [f'combine_T{i:04}.png' for i in range(len(combine_files_path))] 
    # for num in mode_num:
    #     fn = f'mode{num:04}'
    #     if fn in m_dir:
    #         file_paths.append(os.path.join(dir_root, m_dir))
    
# file_paths = [r'D:\GoogleDrive\NTF_project\dy_modesT3\mode0000_0112',
#             r'D:\GoogleDrive\NTF_project\dy_modesT3\mode0001_0108',
#             r'D:\GoogleDrive\NTF_project\dy_modesT3\mode0003_0103',
#             r'D:\GoogleDrive\NTF_project\dy_modesT3\mode0005_0111',
#             r'D:\GoogleDrive\NTF_project\dy_modesT3\mode0007_0082',
#             r'D:\GoogleDrive\NTF_project\dy_modesT3\mode0009_0117', 
#             ]

# file_paths = [r'D:\GoogleDrive\NTF_project\dy_modesT3T_1\mode0000_0070',
#               r'D:\GoogleDrive\NTF_project\dy_modesT3T_1\mode0002_0069', 
#               r'D:\GoogleDrive\NTF_project\dy_modesT3T_1\mode0004_0073',
#               r'D:\GoogleDrive\NTF_project\dy_modesT3T_1\mode0006_0085',
#               r'D:\GoogleDrive\NTF_project\dy_modesT3T_1\mode0008_0075',
#               r'D:\GoogleDrive\NTF_project\dy_modesT3T_1\mode0010_0077'
#               ]


if __name__ == '__main__':

    save_dir = r'I:\LBM_RBC\Rascomb'
    out_path = r'I:\LBM_RBC\combRas.mp4'
    # save_dir = r'D:\GoogleDrive\NTF_project/dy_modesT3T_1_combine'
    # out_path = r'D:\GoogleDrive\NTF_project\combT31modes_v.mp4'
    combine_pics.png_grid_comb(combine_files_path,  file_name=combine_names, save_dir=save_dir, n_ele=3, MP=True, nproc=4,
                    to_video=False, video_path='test.mp4', remove_png_source=False)
    
    png_dir = save_dir
   
    
    duration=0.07
    moviepy_video_func.png_to_movie(out_path, png_dir=png_dir, duration=duration)





