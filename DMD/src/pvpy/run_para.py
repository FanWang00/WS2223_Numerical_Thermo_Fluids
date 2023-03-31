#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:39:30 2020

@author: fan
"""
import sys
sys.path.append("..")
import subprocess
# from utils_dmd import dir_base_list
# from utils_dmd.comm_tools import Timer, dir_list, make_dir,  run_command_realtime_output
import dir_base_list
from comm_tools import Timer, dir_list, make_dir,  run_command_realtime_output
import os 
import numpy as np
import img2video

print(os.getcwd())
dir_base = dir_base_list.dir_base_list()
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
# input_file = '/media/fan/SeagateExpansionDrive/ALPACA_CWD/DataSet/L3/WithShock/aerobreakup_data_index.xdmf'
# shock_mode = ['WithShock', 'WithoutShock']
# # input_file = '/media/overflow/Volume/ALPACA_CWD/PreProcess/interp/L3/WithShock/aerobreakup_data_index.xdmf'
# input_file = os.path.join(dir_base, 'PreProcess', 'interp', "L3", shock_mode[0], 'aerobreakup_data_index.xdmf')
# actual_dis_item = 'pressure'
# input_save_path = os.path.join(dir_base, 'PreProcess', 'video', 'interp', 'L3', 'L3_single.png')
# make_dir(os.path.dirname(input_save_path))
# num_files = len(dir_list(os.path.dirname(input_file), pattern='*.h5'))
# input_frame = [0, num_files]
# py_script = 'pv_single_new.py'
# cmd =  [pvpy_path, py_script, input_file, actual_dis_item, input_save_path, 
#        str(input_frame[0]), str(input_frame[1])]
############################################################ 

###################### pvpy compare file ####################

file_path =[os.path.join(dir_base, "output/data_index.xdmf"),
                  os.path.join(dir_base, "PostProcess/interp/L5/data_index.xdmf")]
file_path = ['/media/fan/SeagateExpansionDrive/ALPACA_CWD/PreProcess/interp/L3/WithShock/aerobreakup_data_index.xdmf',
             '/media/fan/SeagateExpansionDrive/ALPACA_CWD/DataSet/L3/WithShock/aerobreakup_data_index']
# file_path =[os.path.join(dir_base, "output/L5/aerobreakup_l5_40/DMD_Recons/data_index.xdmf")]
# input variables
input_file = ['/media/fan/SeagateExpansionDrive/ALPACA_CWD/DataSet/L3/WithShock/aerobreakup_data_index.xdmf',
              '/media/fan/SeagateExpansionDrive/ALPACA_CWD/PreProcess/interp/L3/WithShock/aerobreakup_data_index.xdmf'] 
input_item = ['pressure', 'pressure']
input_save_path = '/media/fan/SeagateExpansionDrive/ALPACA_CWD/test/test_file.png'
input_frame =[0, 9]

py_script = 'pv_vertical_2png_new2.py'
cmd = [pvpy_path, py_script, input_file[0], input_file[1], input_item[0], 
       input_item[1], input_save_path, str(input_frame[0]), str(input_frame[1])]
num_files = 100
#########################################################

# # cmd = [pvpy_path, py_script, input_file[0], input_file[1], input_item, input_CellArr, input_save_path]
# cmd = [pvpy_path, py_script]
# # cmd = ['python', 'input_format_test.py', input_file[0], input_file[1], input_item, input_CellArr, input_save_path]

time_step = np.arange(0, num_files, 10)
time_step = np.append(time_step, num_files)
time_step_a = time_step[:-1]
time_step_b = time_step[1:]-1
# time_step = np.array([0, 500, 1000, num_files], dtype='int')
# time_step_a = time_step[:-1]
# time_step_b = time_step[1:]-1
with Timer('pvpy'):
    for a, b in zip(time_step_a, time_step_b):
        cmd =  [pvpy_mpi_path, pvpy_path, py_script, input_file, actual_dis_item, input_save_path, 
                str(a), str(b)]
        # run_command_realtime_output(cmd)
        print(f"convertin frame {a} >>> {b} / total {num_files}")
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
# with Timer('video'):
#     img_dir = os.path.join(os.path.dirname(input_save_path), './*.png')
#     outfile = os.path.join(dir_base, 'video', 'test', video_name)
#     make_dir(os.path.dirname(outfile))
    # img2video.ffmpeg_convert(img_dir, outfile, fulloutput=False)