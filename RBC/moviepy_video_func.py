import numpy as np
from os import path as osp
import matplotlib.pyplot as plt
from moviepy.editor import *

## encode pngs to video

def png_to_movie(output_path, png_dir=None, png_paths=None, duration=0.1, fps=24, verbose=True):


    if png_dir is not None:
        data_p_list = [os.path.join(png_dir, f) for f in os.listdir(png_dir)]
        data_p_list = sorted(data_p_list)
    else:
        data_p_list = png_paths
    clips = [ImageClip(m).set_duration(duration)
            for m in data_p_list]
    concat_clip = concatenate_videoclips(clips, method="compose")
    if verbose:
        print(f'pics are conveting to videos with {fps} fps, {duration} duration')
        concat_clip.write_videofile(output_path, fps=fps)
        print(f'converting done, video saved in {output_path}')
    else:
        concat_clip.write_videofile(output_path, fps=fps)
    
if __name__ == '__main__':
    path = r'I:\LBM_RBC\data\500Ra5.0E+06_dt1.0E+00\data'
    out_path = r'I:\LBM_RBC\data\500Ra5.0E+06_dt1.0E+00\data\RBC_T.mp4'
    duration=0.07
    png_to_movie(out_path, png_dir=path, duration=duration)
    

