import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# import xml.etree.ElementTree as ET
import os
import glob
import time
# from dmd_tools import colormap_dmd
# from dmd_cmap import colormap_dmd
import inspect
# from scipy.interpolate import griddata
# import h5py
# import meshio
# from numba import jit
# from utils_dmd.xdmfWriter import Xdmf_Index_Writer, Xdmf_Writer
from PIL import Image
from multiprocessing import Pool
# # from functools import wraps
from functools import partial
import math
from PIL import Image
import logging
# def d_print(var):
#     print(f"{var=}")

# def combine_pic(file):
def find_dir(folder='dev', path=''):
# if not os.path.isdir(os.path.join(path, folder)):
    
    # print(path)
    if not os.path.isdir(os.path.join(path, folder)):
        path = os.path.join(path, '..')      
        # print(path)
        return find_dir(folder='dev', path=path)

    else:
        return os.path.join(path, folder)
        
def list_T(x):
    '''transpose 2d list 
    '''
    return [list(i) for i in zip(*x)]

def logger(filename='mylog.log'):
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    # filename='myapp.log',
                    filename=filename,
                    filemode='w')

    #################################################################################################
    # define StreamHandler，put INFO level or higher in to consle and save
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    #################################################################################################
    return logging

def rescale_mode(input, min=0, max=1):
    res = np.asarray([rescale(value/np.linalg.norm(value), min, max) for value in input])
    return res

def rescale(input, min=0, max=1):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input
    
def image_compose(image_names, IMAGES_PATH, IMAGES_FORMAT, IMAGE_SIZE, IMAGE_ROW, IMAGE_COLUMN, IMAGE_SAVE_PATH ):
    
    # IMAGES_PATH = 'D:\Mafengwo\photo\五月坦桑的暖风，非洲原野的呼唤\\'  # 图片集地址
    # IMAGES_FORMAT = ['.jpg', '.JPG']  # 图片格式
    # IMAGE_SIZE = 256  # 每张小图片的大小
    # IMAGE_ROW = 5  # 图片间隔，也就是合并成一张图后，一共有几行
    # IMAGE_COLUMN = 4  # 图片间隔，也就是合并成一张图后，一共有几列
    # IMAGE_SAVE_PATH = 'final.jpg'  # 图片转换后的地址

    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) #创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH) # 保存新图


def prev_dir(split_str, deep=0):
    
    if deep <= 0:
#         print(split_str)
        return split_str
    else:
        deep -= 1
        new_str = os.path.dirname(split_str)
#         print(new_str)
        return prev_dir(new_str, deep)
    
def append_images(images, direction='horizontal',
                  bg_color=(255,255,255), aligment='center'):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)


    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im

def count_digits(n):
    '''
    n : intger to count 
    '''
    if n > 0:
        digits = int(math.log10(n))+1
    elif n == 0:
        digits = 1
    else:
        digits = int(math.log10(-n))+2 # +1 if you don't count the '-' 
    return digits
    
def save_png_assemble(arr_list, minmax=False, clean=False, clean_threshold=None):
    # if arr_list.ndim == 3:
    
    print(f"clean: {clean}")
    if clean:
        print('clean')
        arr_list = np.where(np.abs(arr_list)<clean_threshold, 0, arr_list)
    arr_assemble = np.stack(arr_list, axis=1)
    # arr_assemble = [list(a) for a in zip(*[j.tolist() for j in arr_list])]
    vmax = np.zeros(len(arr_list))
    vmin = np.zeros(len(arr_list))
    for idx, i in enumerate(arr_list):
        max_tmp = i.max()
        min_tmp = i.min()
        if np.sign(max_tmp) == np.sign(min_tmp):
            vmax[idx] = max_tmp
            vmin[idx] = min_tmp
        else:
            vmax[idx] = max(abs(max_tmp), abs(min_tmp))
            vmin[idx] = -vmax[idx]

    # vmax = np.asarray([i.max() for i in arr_list])
    # vmin = np.asarray([i.min() for i in arr_list])
    if minmax:
        return arr_assemble, vmin, vmax
    else:
        return arr_assemble

def dim_list(a):
    '''
    get dimensiotn of list
    '''
    if not type(a) == list:
        return []
    return [len(a)] + dim_list(a[0])

def func_MP(func, param_iter, nproc):
    pool = Pool(processes=nproc)
    pool.starmap(func, param_iter)
    pool.terminate()
    pool.join()
    
    # pool.starmap_async(func, param_iter)

# def func_MP(func, param_iter, nproc):
#     pool = Pool(processes=nproc)
#     pool.starmap_async(func, param_iter)
    # pool.terminate()
    # return ax

def retrieve_name(var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]
            
def contains_explicit_return(f):
    import ast
    import inspect
    return any(isinstance(node, ast.Return) for node in ast.walk(ast.parse(inspect.getsource(f))))

def find_src(path='src'):
    newpath = os.path.join('../', path)
#     print(newpath)
    if not os.path.exists(newpath):
#         print(newpath)
        newpath = find_src(newpath)
    if os.path.exists(newpath):
        return newpath
        
def run_command_realtime_output(command):
    import subprocess
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    # while True:
    #     line = process.stdout.readline()
    #     process.stdin.flush()
    #     if not line: break
    while True:
        output = process.stdout.readline()
        process.stdout.flush()
        if output == '' and process.poll() is not None:
            break
        if output:
            print (output.strip())
        
    rc = process.poll()
    return rc

def make_dir(path):

    if not os.path.exists(path):
        os.makedirs(path)

def sort_idx_arr(arr):
    
    idx = np.argsort(arr)
    return idx, np.sort(arr)

def dir_list(path, pattern='*', sort=False):
    
    file_path = glob.glob(os.path.join(path, pattern))
    if sort:
        return sorted(file_path)
    else:
        return file_path

def load_data(input_path, pattern):
    
    pass


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))