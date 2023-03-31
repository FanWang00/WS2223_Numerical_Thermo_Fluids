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

def xdmf2png_video(pvpy_path, py_script, actual_dis_item, max_frame, 
                    png_save_path, video_save_path,
                    convert_video=True, realtime=False):

    time_step = np.arange(0, max_frame, 10)
    time_step = np.append(time_step, max_frame)
    time_step_a = time_step[:-1]
    time_step_b = time_step[1:]-1
    with Timer('pvpy'):
        
        for a, b in zip(time_step_a, time_step_b):
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
    max_frame = 10
    input_frame = [0, max_frame]
    py_script = 'pvpy/pv_single_new.py'
    xdmf2png_video(pvpy_path, py_script, actual_dis_item, max_frame, 
                    png_save_path, video_save_path)
    # cmd =  [pvpy_path, py_script, xdmf_path, actual_dis_item, png_save_path, 
    #        str(input_frame[0]), str(input_frame[1])]
    ############################################################ 

    ###################### pvpy compare file ####################

    # file_path =[os.path.join(dir_base, "output/data_index.xdmf"),
    #                   os.path.join(dir_base, "PostProcess/interp/L5/data_index.xdmf")]
    # file_path = ['/media/fan/SeagateExpansionDrive/ALPACA_CWD/PreProcess/interp/L3/WithShock/aerobreakup_data_index.xdmf',
    #              '/media/fan/SeagateExpansionDrive/ALPACA_CWD/DataSet/L3/WithShock/aerobreakup_data_index']
    # # file_path =[os.path.join(dir_base, "output/L5/aerobreakup_l5_40/DMD_Recons/data_index.xdmf")]
    # # input variables
    # xdmf_path = ['/media/fan/SeagateExpansionDrive/ALPACA_CWD/DataSet/L3/WithShock/aerobreakup_data_index.xdmf',
    #               '/media/fan/SeagateExpansionDrive/ALPACA_CWD/PreProcess/interp/L3/WithShock/aerobreakup_data_index.xdmf'] 
    # input_item = ['pressure', 'pressure']
    # png_save_path = '/media/fan/SeagateExpansionDrive/ALPACA_CWD/test/test_file.png'
    # input_frame =[0, 9]

    # py_script = 'pv_vertical_2png_new2.py'
    # cmd = [pvpy_path, py_script, xdmf_path[0], xdmf_path[1], input_item[0], 
    #        input_item[1], png_save_path, str(input_frame[0]), str(input_frame[1])]
    #########################################################

    # # cmd = [pvpy_path, py_script, xdmf_path[0], xdmf_path[1], input_item, input_CellArr, png_save_path]
    # cmd = [pvpy_path, py_script]
    # # cmd = ['python', 'input_format_test.py', xdmf_path[0], xdmf_path[1], input_item, input_CellArr, png_save_path]