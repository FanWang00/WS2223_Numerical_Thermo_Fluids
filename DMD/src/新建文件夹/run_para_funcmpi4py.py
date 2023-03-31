#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:39:30 2020

@author: fan
"""
import sys
# sys.path.append("src")
import subprocess
# from utils_dmd import dir_base_list
# from utils_dmd.comm_tools import Timer, dir_list, make_dir,  run_command_realtime_output
import get_dir_base
from comm_tools import Timer, dir_list, make_dir, run_command_realtime_output
import os 
import numpy as np
from pvpy import img2video
from functools import partial
from toolz.functoolz import curry
from multiprocessing import Pool

def xdmf2png(pvpy_path, py_script, xdmf_path, actual_dis_item, png_save_path, 
             realtime,
             time_step_a, time_step_b):
    # for key, value in kwargs.items():
    #     if key == 'time_step_a':
    #         time_step_a = kwargs['time_step_a']
            
    #     if key == 'time_step_b':
    #         time_step_b = kwargs['time_step_b']
    try: 
        range_ab = zip(time_step_a, time_step_b)
    except TypeError:
        range_ab =zip([time_step_a], [time_step_b])
    
    for a, b in range_ab:
        a = time_step_a
        b = time_step_b
        
        cmd =  [pvpy_path, py_script, xdmf_path, actual_dis_item, png_save_path, 
            str(a), str(b)]
        # run_command_realtime_output(cmd)
        print(f"converting frame {a} >>> {b} / total  {max_frame}")

        if realtime:
            run_command_realtime_output(cmd)
        else:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            if err:
                print("error:\n", err.decode('UTF-8'))
            else:
                # while True:
                #     line = process.stdout.readline()
                #     stdout.append(line)
                #     print (line)
                #     if line == '' and process .poll() != None:
                #         break
                # return ''.join(stdout)
            
                # print("{} convert is done".format(outfile))
                print(out.decode('UTF-8'))

def xdmf2png_par(pvpy_path, py_script, xdmf_path, actual_dis_item,
                  png_save_path, realtime):
    # print(kwargs)
    # return curry(xdmf2png)(pvpy_path=pvpy_path)(py_script=py_script)(xdmf_path=py_script)\
    #                       (actual_dis_item=actual_dis_item)(png_save_path=png_save_path)\
    #                       (realtime=realtime)
    return partial(xdmf2png, pvpy_path, py_script, xdmf_path, actual_dis_item, png_save_path, 
             realtime)
             
def xdmf2png_video(pvpy_path, py_script, actual_dis_item, max_frame, xdmf_path,
                    png_save_path, video_save_path, nproc=1,
                    convert_video=True, realtime=False):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    do_func = xdmf2png_par(pvpy_path, py_script, xdmf_path,
                            actual_dis_item, png_save_path, 
                            realtime)
    if rank == 0:
        time_step = np.arange(0, max_frame, 10)
        time_step = np.append(time_step, max_frame)
        time_step_a = list(time_step[:-1])
        time_step_b = list(time_step[1:]-1)
        
        time_step_buff = list(zip(time_step_a, time_step_b))
        
    else:
        time_step_buff=None
# time_step = None
    time_step_buff = comm.scatter(time_step_buff, root=0)   
    do_func(*time_step_buff)
        # time_setp_kw = {'time_step_a':time_step_a, 'time_step_b':time_step_b}
    # else:
    #     time_step_a, time_step_b = None, None
    #     with Timer('pvpy'):
    #         if nproc > 1:
                
    #             # do_func(1, 2)
    #             # pool = Pool(nproc)
            
    #             # pool.starmap(do_func, zip(time_step_a, time_step_b))
    #         else:
        # xdmf2png(pvpy_path, py_script, xdmf_path, actual_dis_item, png_save_path, 
        #         False,
        #         time_step_a, time_step_b)
    # video_name = 'interp_L3_WithoutShock.mp4'
    if convert_video:
        with Timer('video converting'):
            img_dir = os.path.join(os.path.dirname(png_save_path), './*.png')
            outfile = os.path.join(dir_base, 'video', 'test', video_save_path)
            make_dir(os.path.dirname(outfile))
            img2video.ffmpeg_convert(img_dir, outfile, fulloutput=False)

if __name__ == '__main__':

    import sys
    # sys.path.append('src') 
    import get_dir_base

    dir_base = get_dir_base.dir_base_list()
    pvpy_mpi_path = '/home/overflow/Desktop/ParaView-5.8.0-osmesa-MPI-Linux-Python3.7-64bit/bin/mpiexec'
    pvpy_path = '/home/overflow/Desktop/ParaView-5.8.0-osmesa-MPI-Linux-Python3.7-64bit/bin/pvpython'
    nproc=1
    # pvpy_path = '/home/overflow/Desktop/ParaView-5.8.0-osmesa-MPI-Linux-Python3.7-64bit/bin/pvserver'
    # py_script = 'pv_save_png_compare.py'
    # py_script = 'pvpy_compare_new.py'
    # py_script = 'pvpy_single_test_subprocess.py'
    # py_script = 'input_format_test.py'
    # py_script = 'pv_2_png_test.py'
    # py_script = 'run_paraview2_old.py'
    # py_script = 'pv_single_new.py'

    ############# pvpy single file test ######################
    # xdmf_path = '/media/fan/SeagateExpansionDrive/ALPACA_CWD/DataSet/L3/WithShock/aerobreakup_data_index.xdmf'
    r_level = 3
    dmd_mode = ""
    shock_mode_list = ['WithShock', 'WithoutShock']
    shock_mode = shock_mode_list[0]
    # xdmf_path = '/media/overflow/Volume/ALPACA_CWD/PreProcess/interp/L3/WithShock/aerobreakup_data_index.xdmf'
    xdmf_path = os.path.join(dir_base, 'PreProcess', 'interp', f"L{r_level}", shock_mode, 'aerobreakup_data_index.xdmf')

    actual_dis_item = 'pressure'
    png_save_path = os.path.join(dir_base, 'PreProcess', 'interp', 'png', f"L{r_level}", f"L{r_level}_{actual_dis_item}.png")
    video_save_path = os.path.join(dir_base, 'PreProcess', 'interp', 'video', f"L{r_level}", f"L{r_level}_{shock_mode}_{actual_dis_item}.mp4")
    
    [make_dir(os.path.dirname(path)) for path in [png_save_path, video_save_path]]
    
    
    num_files = len(dir_list(os.path.dirname(xdmf_path), pattern='*.h5'))
    max_frame = 100
    input_frame = [0, max_frame]
    py_script = 'pvpy/pv_single_new.py'
    xdmf2png_video(pvpy_path, py_script, actual_dis_item, max_frame, xdmf_path,
                    png_save_path, video_save_path, nproc=nproc)
    # cmd =  [pvpy_path, py_script, xdmf_path, actual_dis_item, png_save_path, 
    #        str(input_frame[0]), str(input_frame[1])]
    ############################################################ 

    ###################### pvpy compare file ####################
    # r_level_1 = 3
    # r_level_2 = 3
    # dmd_mode = ""
    # shock_mode_list = ['WithShock', 'WithoutShock']
    # shock_mode_1 = shock_mode_list[0]
    # shock_mode_2 = shock_mode_list[1]
    # # xdmf_path = os.path.join(dir_base, 'PreProcess', 'interp', f"L{r_level}", shock_mode, 'aerobreakup_data_index.xdmf')
    
    # xdmf_path =[os.path.join(dir_base, 'PreProcess', 'interp', f"L{r_level_1}", shock_mode_list[0], 'aerobreakup_data_index.xdmf'),
    #             os.path.join(dir_base, 'PreProcess', 'interp', f"L{r_level_2}", shock_mode_list[1], 'aerobreakup_data_index.xdmf')]
    # # xdmf_path = ['/media/fan/SeagateExpansionDrive/ALPACA_CWD/PreProcess/interp/L3/WithShock/aerobreakup_data_index.xdmf',
    # #              '/media/fan/SeagateExpansionDrive/ALPACA_CWD/DataSet/L3/WithShock/aerobreakup_data_index']
    # # xdmf_path =[os.path.join(dir_base, "output/L5/aerobreakup_l5_40/DMD_Recons/data_index.xdmf")]
    # # input variables 
    # input_item = ['pressure', 'pressure']
    # actual_dis_item = [input_item[0], input_item[1]]
    
    # png_save_path = os.path.join(dir_base, 'PreProcess', 'interp', 'png',
    #                             f"L{r_level_1}{shock_mode_1}{actual_dis_item[0]}"\
    #                             f"_L{r_level_2}{shock_mode_2}{actual_dis_item[1]}.png")
    # video_save_path = os.path.join(dir_base, 'PreProcess', 'interp', 'video', 
    #                             f"L{r_level_1}{shock_mode_1}{actual_dis_item[0]}"\
    #                             f"_L{r_level_2}{shock_mode_2}{actual_dis_item[1]}.mp4")
    
    
    # [make_dir(os.path.dirname(path)) for path in [png_save_path, video_save_path]]


    # num_files = len(dir_list(os.path.dirname(xdmf_path), pattern='*.h5'))
    # max_frame = 100
    # input_frame = [0, max_frame]


    # py_script = 'pv_vertical_2png_new2.py'
    # cmd = [pvpy_path, py_script, xdmf_path[0], xdmf_path[1], input_item[0], 
    #        input_item[1], png_save_path, str(input_frame[0]), str(input_frame[1])]
    # xdmf2png_video(pvpy_path, py_script, actual_dis_item, max_frame, xdmf_path,
    #                 png_save_path, video_save_path, nproc=nproc)
    #########################################################

    # # cmd = [pvpy_path, py_script, xdmf_path[0], xdmf_path[1], input_item, input_CellArr, png_save_path]
    # cmd = [pvpy_path, py_script]
    # # cmd = ['python', 'input_format_test.py', xdmf_path[0], xdmf_path[1], input_item, input_CellArr, png_save_path]