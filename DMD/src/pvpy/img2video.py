#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:49:35 2020

@author: overflow
"""
import subprocess
import sys
import os
sys.path.append('..')
import comm_tools
from comm_tools import run_command_realtime_output
from socket import gethostname


def stack_video(input_path_lst, stack, out_path, realtime=False, fulloutput=False):

    if stack not in ('hstack', 'vstack'):
        raise ValueError("parameter of stack are 'hstack' and 'vstack'!")
    n = len(input_path_lst)


    # file_dir = '/local/disk1/fanwang/ALPACA_CWD/data_gen_temp/PostProcess/video/single'
    file_name_lst = ['cubic_L2_WithoutShock_.mp4', 'cubic_L2_Without_pressure.mp4']
    # input_path_lst = [os.path.join(file_dir, f) for f in file_name_lst]
    # input_path_lst


    stack_cmd = [f'{stack}={n}']
    in_file_cmd = [f'-i {name}' for name in input_path_lst]
    print(gethostname())
    if gethostname() in ('mieko.aer.mw.tum.de', 'miyuko.aer.mw.tum.de'):
        cmd = ['/global/ffmpeg/ffmpeg-3.2.4-64bit-static/ffmpeg'] + in_file_cmd + ['-filter_complex'] + stack_cmd + [out_path]
    else:
        cmd = ['ffmpeg'] + in_file_cmd + ['-filter_complex'] + stack_cmd + [out_path]

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
            if fulloutput:
                print(out.decode('UTF-8'))
            else:
                print("{} convert is done".format(outfile))
        # if remove_png_source:
        #     r_path = comm_tools.dir_list(*os.path.split(img_dir))
        #     for f in r_path:
        #         os.remove(f)

def py_ffmpeg(input_png_dir, out_path):
    import ffmpeg
    (
    ffmpeg
    .input(input_png_dir, pattern_type='glob', framerate=10)
    # .filter('deflicker', mode='pm', size=10)
    .filter('scale', force_original_aspect_ratio='increase')
    .output(out_path, crf=24, preset='veryfast', vcodec='h264', movflags='faststart', pix_fmt='yuv420p')
    .overwrite_output()
    # .view(filename='filter_graph')
    .run()
    )

def ffmpeg_convert(img_dir, outfile, remove_png_source=False, fulloutput=False, realtime=False):
    
# outfile = "test.mp4"
# img_dir = './*.png'

    if gethostname() in ('mieko.aer.mw.tum.de', 'miyuko.aer.mw.tum.de'):
        ffmpeg_cmd = '/global/ffmpeg/ffmpeg-3.2.4-64bit-static/ffmpeg'
    else:
        ffmpeg_cmd = 'ffmpeg'
    cmd = [ffmpeg_cmd, '-y',
           '-framerate',
           '10',
           '-pattern_type', 
           'glob',
           '-i',
           img_dir,
           '-vf',
           "pad=ceil(iw/2)*2:ceil(ih/2)*2",
           '-vcodec',
           'h264',
           '-acodec',
           'aac',
           '-strict',
           '-2',
           '-pix_fmt',
           'yuv420p',
           '-preset',
           'veryslow',
           outfile] 
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
            if fulloutput:
                print(out.decode('UTF-8'))
            else:
                print("{} convert is done".format(outfile))
        if remove_png_source:
            r_path = comm_tools.dir_list(*os.path.split(img_dir))
            for f in r_path:
                os.remove(f)

    

# cmd = [ffmpeg,
#        -framerate,
#        10,
#        -pattern_type glob,
#        -i,
#        './.png',
#        -vf,
#        "pad=ceil(iw/2)2:ceil(ih/2)*2",
#        -vcodec,
#        h264,
#        -acodec,
#        aac,
#        -strict,
#        -2,
#        -pix_fmt,
#        yuv420p,
#        -preset,
#        veryslow,

# ffmpeg -framerate 10 -pattern_type glob -i './.png' -vf "pad=ceil(iw/2)2:ceil(ih/2)*2" -vcodec h264 -acodec aac -strict -2 -pix_fmt yuv420p -preset veryslow test.mp4] 

def cv2_img_video(image_folder, ):
    
    import cv2
    import os
    
    image_folder= '/media/overflow/Seagate Expansion Drive/PostProcess/paraview/Animation/L5_Ori_Recons_Pressure'
    # video_name = '/media/overflow/Seagate Expansion Drive/PostProcess/paraview/Animation/video/L5_Ori_Interp_U.avi'
    video_name_base = '/media/overflow/Seagate Expansion Drive/PostProcess/paraview/Animation/video/'
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
    images = sorted(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    for i in range(len(images)):
        
        print("fps: {}".format(i))
        video_path_tmp = os.path.join(video_name_base, "L5_Ori_Recons_Pressure" + str(i)+ ".avi")
        video = cv2.VideoWriter(video_path_tmp, fourcc, i, (width, height))
        
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
            
        # cv2.destoryAllWindows()
        video.release()
if __name__ == '__main__':
    
    import os
    import sys
    # out_dir = "/media/fan/Seagate Expansion Drive/PostProcess/paraview/Animation/video/"
    # out_name = "test3.mp4"
    # dir_base = "/media/fan/Seagate Expansion Drive/PostProcess/paraview/Animation/copy_L5_Ori_Interp_U/"
    # img_parttern = './*.png'
    out_dir = sys.argv[1]
    out_name = sys.argv[2]
    dir_base = sys.argv[3]
    img_pattern = sys.argv[4]
    
    outfile = os.path.join(out_dir, out_name)
    img_dir = os.path.join(dir_base, img_parttern)
    ffmpeg_convert(img_dir, outfile)
    