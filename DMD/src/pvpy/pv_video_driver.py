#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:39:30 2020

@author: fan
"""
import sys
sys.path.append("..")
import subprocess
import dir_base_list
import img2video
import os 
import comm_tools

png_save_done = False
dir_base = dir_base_list.dir_base_list()

pvpy_path = '/home/fan/Desktop/ParaView-5.8.0-MPI-Linux-Python3.7-64bit/bin/pvpython'
py_script = 'pv_save_png_compare.py'
# py_script = 'pvpy_single_test_subprocess.py'
# py_script = 'input_format_test.py'
# py_script = 'pv_2_png_test.py'
# py_script = 'run_paraview2_old.py'
py_script = 'run_pv_par.py'
file_path =[os.path.join(dir_base, "output/L5/aerobreakup_l5_40/DMD_Recons/data_index.xdmf"),
                  os.path.join(dir_base, "PostProcess/interp/L5/data_index.xdmf")]

# file_path =[os.path.join(dir_base, "output/L5/aerobreakup_l5_40/DMD_Recons/data_index.xdmf")]
# input variables
input_file = file_path
input_item = 'velocityU'
input_CellArr = 'velocityU'
input_save_path = os.path.join(dir_base, "test/test_file.png")
input_fram = [str(0), str(5)]

cmd = [pvpy_path, py_script, input_file[0], input_file[1], input_item, input_CellArr, 
       input_save_path, input_fram[0], input_fram[1]]
# cmd = [pvpy_path, py_script]
# cmd = ['python', 'input_format_test.py', input_file[0], input_file[1], input_item, input_CellArr, input_save_path]
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
    png_save_done = True
video_name = 'test.mp4'
img_dir = os.path.join(os.path.dirname(input_save_path), './*.png')
outfile = os.path.join(dir_base, 'video', 'test', video_name)
comm_tools.make_dir(os.path.dirname(outfile))
img2video.ffmpeg_convert(img_dir, outfile, fulloutput=False)