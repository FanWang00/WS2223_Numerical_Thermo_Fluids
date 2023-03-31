import os
import comm_tools
import numpy as np 
from multiprocessing import Pool
from functools import partial
from PIL import Image
import copy

def _png_grid_comb_save_mpi(n_ele, nrow, new_width, new_height, bg_color, save_dir, 
                            png_path, fn):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    mpi_save = partial(_png_grid_comb_save, n_ele, nrow, new_width, new_height, bg_color, save_dir)
    if rank == 0:
        p_f = list(zip(png_path, fn))
        idx_glb = np.asarray([i for i in range(len(png_path))])
        sendbuf = idx_glb
        ave, res = divmod(len(sendbuf), nprocs)
        count = [ave + 1 if p < res else ave for p in range(nprocs)]
        # idx = np.asarray([i for i in range(len(png_path))])
        count = np.array(count)
        # print(count)
        # displacement: the starting index of each sub-task
        displ = [sum(count[:p]) for p in range(nprocs)]
        displ = np.array(displ)
    else:
        sendbuf = None
        # initialize count on worker processes
        count = np.zeros(nprocs, dtype=np.int).tolist()
        displ = None
    # broadcast count
    comm.Bcast(count, root=0)
    comm.bcast(p_f, root=0)
    recvbuf = np.zeros(count[rank])

    comm.Scatterv([sendbuf, count, displ], recvbuf, root=0)

    print('After Scatterv, process {} has data:'.format(rank), recvbuf)
    # print(type(recvbuf))
    comm.Barrier()
    # idx = [displ + i for i in range(count)]
    # new_sendbuf = recvbuf ** 2 
    for i in range(len(recvbuf)):
        mpi_save(i[0], i[1])
    
    # print(new_sendbuf)
    # comm.Barrier()
    # new_rev = np.zeros((N, M))
    # comm.Gatherv(new_sendbuf, [new_rev, count, displ, MPI.DOUBLE], root=0)
    # # comm.Barrier()
    # if rank == 0:
    #     print(f"result:{new_rev}")
    #     return new_rev

def _png_grid_comb_save(n_ele, nrow, new_width, new_height, bg_color, save_dir, 
                        png_path, filename):
    x = 0
    y = 0
    new_img = Image.new('RGBA', (new_width*(n_ele), new_height*(nrow)), color=bg_color)
    images = [Image.open(p) for p in png_path]
    for img in images:
        
        # print(img)
        # img = img.resize((eachSize, eachSize), Image.ANTIALIAS)
        new_img.paste(img, (x*new_width, y*new_height))
        # print(x*new_width, y*new_height)
        # print(n_ele, nrow)
        x += 1
        
        if x == n_ele:
            x = 0
            y += 1
    new_img.save(os.path.join(save_dir, filename), format='png')
        
        
def png_grid_comb(img_path,  file_name=None, save_dir='test_comb', n_ele=None, MP=False, nproc=2,
                  to_video=False, video_path='test.mp4', remove_png_source=False):
    '''combine pngs to grid-like
    images: Img obj 2D list size=(T, N) T: time step, N: items for each time step 
    n_ele:  numer of element for one row, default N
    save_dir: 
    video_path : abs path
    '''    
    comm_tools.make_dir(save_dir)
    if len(comm_tools.dim_list(img_path)) == 1:
        img_path = [[i] for i in img_path]

    bg_color = (255, 255, 255)
    # images = np.empty(tuple(img_size), dtype=object)
    img_size = comm_tools.dim_list(img_path)
    N = img_size[1]
    T = img_size[0]
    # T, N = images.shape    
    if n_ele is None:
        n_ele = img_size[1]
    nrow = int(np.ceil(N/n_ele))
    # if file_name is None:
    #     file_name = file_name_gen(T, base='test', suffix='.png')
# print(nrow)
    # print(nrow)
    # if direction=='horizontal':
    # img_flat = []
    # img_flat = [i for j in images for i in j]
    # # print(img_flat)
    # widths, heights = zip(*(i.size for i in img_flat))
    # new_width = max(widths)
    # new_height = max(heights) 
    with Image.open(img_path[0][0]) as get_new_size:

        new_width, new_height = get_new_size.size
    # new_width = widths
    # new_height = heights 
    # print(new_width, new_height)
    # print(nrow, n_ele)
    if MP:
        MP_save_func = partial(_png_grid_comb_save, n_ele, nrow, new_width, new_height, bg_color, save_dir)
        comm_tools.func_MP(MP_save_func, zip(img_path, file_name), nproc)
    else:  
        for path, fn in zip(img_path, file_name):
            _png_grid_comb_save(n_ele, nrow, new_width, new_height, bg_color, save_dir,path, fn)
    # if to_video:
        # ffmpeg_convert(save_dir+'/*.png', outfile=video_path, remove_png_source=remove_png_source)

def png_grid_comb_save(Img_obj, save_path):
    '''save multiple pngs in grid
    Img_obj:   input file path, str list, size = (m, n), m, n are same size (row, column) of output grid pic 
    save_dir:   outoput dir
    # h :         high of each single  
    '''
    # img_size = comm_tools.dim_list(Img_obj)
    # w, h = Img_obj.size
    # new_img = 
    img_row = []
    [img_row.append(comm_tools.append_images(f, direction='horizontal') for f in Img_obj)]
    new_img = comm_tools.append_images(img_row, direction='vertical')
    new_img.save(save_path, format='png')

def _combine_mode_dy_save(img_1_resize, save_dir, h, fn, r_path):   
    '''save func for combine_mode_dy, separate for MP 
    '''
# for fn, r_path in zip(f_name, path_recons):
# def save(f_name, path_recons, img_1_resize):
    # print(fn, r_path[0])
    img = copy.deepcopy(img_1_resize)
    img_recons = list(map(Image.open, r_path))
#     r_recons = img_recons[0].shape[0] / img_recons[0].shape[1]
    r_recons = img_recons[0].size[0] / img_recons[0].size[1]
    img_recons_resize = []
    [img_recons_resize.append(i.resize((int(h*r_recons), h), Image.ANTIALIAS)) for i in img_recons]
    img.append(img_recons_resize) 
    img = list(map(list, zip(*img)))
    new_img_tmp = []
    for j in range(len(img)):
        new_img_tmp.append(comm_tools.append_images(img[j]))
    new_img = comm_tools.append_images(new_img_tmp, direction='vertical')
    save_path = os.path.join(save_dir, f"{fn}.png")
    # print(save_path)
    new_img.save(save_path, format='png')