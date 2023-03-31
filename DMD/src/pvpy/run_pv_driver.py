#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:39:30 2020

@author: fan
"""
import sys
sys.path.append("../..")
import subprocess
from utils_dmd import dir_base_list
from utils_dmd.comm_tools import Timer, run_command_realtime_output
import os 

dir_base = dir_base_list.dir_base_list()

pvpy_path = '/home/fan/Desktop/ParaView-5.8.0-MPI-Linux-Python3.7-64bit/bin/pvbatch'
# py_script = 'pv_save_png_compare.py'
py_script = 'pvpy_compare_new.py'
# py_script = 'pvpy_single_test_subprocess.py'
# py_script = 'input_format_test.py'
# py_script = 'pv_2_png_test.py'
# py_script = 'run_paraview2_old.py'
py_script = 'pv_single_new.py'

############# pvpy single file test ######################
# input_file = '/media/fan/SeagateExpansionDrive/ALPACA_CWD/DataSet/L3/WithShock/aerobreakup_data_index.xdmf'
input_file = '/media/fan/SeagateExpansionDrive/ALPACA_CWD/PostProcess/L3/dmd/DMDrecons_test/aerobreakup_data.xdmf'
actual_dis_item = 'pressure'
input_save_path = '/media/fan/SeagateExpansionDrive/ALPACA_CWD/test/test.png'
input_frame = [0, 5]
py_script = 'pv_single_new.py'
cmd = [pvpy_path, py_script, input_file, actual_dis_item, input_save_path, 
       str(input_frame[0]), str(input_frame[1])]
############################################################ 



# file_path =[os.path.join(dir_base, "output/data_index.xdmf"),
#                   os.path.join(dir_base, "PostProcess/interp/L5/data_index.xdmf")]
# file_path = ['/media/fan/SeagateExpansionDrive/ALPACA_CWD/PreProcess/interp/L3/WithShock/aerobreakup_data_index.xdmf',
#              '/media/fan/SeagateExpansionDrive/ALPACA_CWD/DataSet/L3/WithShock/aerobreakup_data_index']
# # file_path =[os.path.join(dir_base, "output/L5/aerobreakup_l5_40/DMD_Recons/data_index.xdmf")]
# # input variables
# input_file = file_path
# input_item = 'velocityU'
# input_CellArr = 'velocityU'
# input_save_path = os.path.join(dir_base, "test/test_file.png")


# # cmd = [pvpy_path, py_script, input_file[0], input_file[1], input_item, input_CellArr, input_save_path]
# cmd = [pvpy_path, py_script]
# # cmd = ['python', 'input_format_test.py', input_file[0], input_file[1], input_item, input_CellArr, input_save_path]


with Timer('pvpy'):
    run_command_realtime_output(cmd)
    # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # out, err = process.communicate()
    # if err:
    #     print("error:\n", err.decode('UTF-8'))
    # else:
    #     # while True:
    #     #     line = process.stdout.readline()
    #     #     stdout.append(line)
    #     #     print (line)
    #     #     if line == '' and process .poll() != None:
    #     #         break
    #     # return ''.join(stdout)
    
    #     # print("{} convert is done".format(outfile))
    #     print(out.decode('UTF-8'))