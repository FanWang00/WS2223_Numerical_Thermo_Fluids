# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:42:04 2020

@author: AdminF
"""
import os
from socket import gethostname

def video_name(r_level, interp_m, shock_mode, item):
    name = '_'.join([f'{interp_m}', f'L{r_level}', f'{shock_mode}', f'{item}'])+'.mp4'
    return name

def path_gen(base_spec_name, r_level, shock_mode, item, dir_base='', dmd_mode=''):
    path = os.path.join(dir_base, 'PostProcess', base_spec_name, dmd_mode, f"L{r_level}", shock_mode, item)
    return path

def VORLL_path(load=False):
    VORLL = type('VORLL', (object,), {})()
    if gethostname() == 'kanon':
        base = '/local/disk1/fanwang/ALPACA_CWD/VORL'
        # print('kanon')
        # name = 'CYLINDER_ALL.npz'
        # VORLL.path = '/local/disk1/fanwang/ALPACA_CWD/VORLL/CYLINDER_ALL.npz'
        # VORLL.recons_path = '/local/disk1/fanwang/ALPACA_CWD/VORLL/recons'
        # VORLL.sep_mode_path = '/local/disk1/fanwang/ALPACA_CWD/VORLL/sep_mode' 
        # VORLL.dy_path = '/local/disk1/fanwang/ALPACA_CWD/VORLL/dy' 
    elif gethostname() in ('mieko.aer.mw.tum.de', 'miyuko.aer.mw.tum.de') :
        base = '/local/temp/fanwang/tmp/ALPACA_CWD/VORLL'
        # name = 'CYLINDER_ALL.npz'
        # VORLL.path = '/local/temp/fanwang/tmp/ALPACA_CWD/VORLL/CYLINDER_ALL.npz'
        # VORLL.recons_path = '/local/temp/fanwang/tmp/ALPACA_CWD/VORLL/recons'
        # VORLL.sep_mode_path = '/local/temp/fanwang/tmp/ALPACA_CWD/VORLL/sep_mode' 
        # VORLL.dy_path = '/local/temp/fanwang/tmp/ALPACA_CWD/VORLL/dy' 
        # VORLL.parital_recons = '/local/temp/fanwang/tmp/ALPACA_CWD/VORLL/partial_recons'
    else:
        base = '/mnt/Volume/ALPACA_CWD/VORLL'
    VORLL.path = os.path.join(base, 'CYLINDER_ALL.npz')
    VORLL.recons_path = os.path.join(base, 'recons')
    VORLL.sep_mode_path = os.path.join(base, 'sep_mode') 
    VORLL.dy_path = os.path.join(base, 'dy') 
    VORLL.modes_path = os.path.join(base, 'modes') 
    VORLL.parital_recons = os.path.join(base, 'partial_recons')  
    VORLL.original_path = os.path.join(base, 'original')    
    return VORLL

def dir_base_list():
    
    dir_base_dict = {'DESKTOP-3L9FF3M':'G:\\ALPACA_CWD',
                     'fan-TM1613':'/media/fan/SeagateExpansionDrive',
                    #  'overflow-MS-7A37':'/media/overflow/Volume/ALPACA_CWD/',
                    'overflow-MS-7A37':'/mnt/Volume/ALPACA_CWD/',
                     'kanon':'/local/disk1/fanwang/ALPACA_CWD/',
                     'mieko.aer.mw.tum.de':'/local/temp/fanwang/tmp/ALPACA_CWD/',
                     'miyuko.aer.mw.tum.de':'/local/temp/fanwang/tmp/ALPACA_CWD/'
                    #  'kanon':'/local/disk1/fanwang/ALPACA_CWD/data_gen_temp/'
                     }
    
    dir_base_list = ['/media/fan/Seagate Expansion Drive',
                     '/media/overflow/SegateExpansion1',
                     '/media/overflow/Seagate Expansion Drive',
                     '/local/disk1/fanwang/SimulationData/',
                     '/media/overflow/Volume/ALPACA_Dataset/from_remote',
                     'G:\\ALPACA_Dataset\\from_remote\\'
                    ]
    
    # if get_list:
    #     return dir_base_list
    
    dir_base = []
    dir_base.append(dir_base_dict[gethostname()])
    # for path in dir_base_list:
    #     if os.path.exists(path):
    #         dir_base.append(path)
    if len(dir_base) == 1:
        return dir_base[0]
    
    elif len(dir_base) == 0:
        print("no available path! please check")
    
    else:
        print("more than 1 path available! please check")

class InterpBasedir(object):

    def __init__(self, r_level=None, shock_mode=None, dir_base=None, interp_m='linear', item=None, data_mode=''):

        if dir_base is None:
            self.dir_base = dir_base_list()
        else:
            self.dir_base = dir_base

        if r_level is None:
            self.r_level = 3
        else:
            self.r_level = r_level

        if shock_mode is None:
            self.shock_mode = 'WithShock'
        else:
            self.shock_mode = shock_mode
        if item is None:
            self.item = ''
        else:
            self.item = item
        if interp_m is None:
            self.interp_m = ''
        else:
            self.interp_m = interp_m
        # if CoM:
        #     raw_str = 'raw_CoM'
        # else:
        #     raw_str = 'raw'
        if data_mode != '':
            data_mode = '_' + data_mode
        # self.raw_path = os.path.join(self.dir_base, 'PreProcess', 'interp','raw', 'xdmf', self.interp_m, f"L{self.r_level}", self.shock_mode)
        self.raw_path = os.path.join(self.dir_base, 'PreProcess', 'interp', 'raw'+data_mode, self.interp_m, f"L{self.r_level}", self.shock_mode)
        self.png_dir = os.path.join(self.dir_base, 'PreProcess', 'interp', 'png', self.interp_m, f"L{self.r_level}", 
                                        self.shock_mode, self.item)
        self.npz_path = os.path.join(self.dir_base, 'PreProcess', 'interp', 'raw'+data_mode, 'npz', f'{item}_{self.interp_m}_{self.shock_mode}_L{self.r_level}.npz')
        video_dir = os.path.join(self.dir_base, 'PreProcess', 'interp', 'video', self.interp_m, f"L{self.r_level}", 
                                        self.shock_mode, self.item)
        videoname = video_name(r_level=self.r_level, interp_m=self.interp_m, shock_mode=self.shock_mode, item=self.item)
        self.video_path = os.path.join(video_dir, videoname)


class DatasetBase(object):

    def __init__(self, r_level=None, shock_mode=None, dir_base=None):

        if dir_base is None:
            self.dir_base = dir_base_list()
        else:
            self.dir_base = dir_base

        if r_level is None:
            self.r_level = 3
        else:
            self.r_level = r_level

        if shock_mode is None:
            self.shock_mode = 'WithShock'
        else:
            self.shock_mode = shock_mode

        self.path = os.path.join(self.dir_base, 'SimulationData', f"L{self.r_level}", self.shock_mode)
        self.png_path = os.path.join(self.dir_base, 'PostProcess', 'png', 'interp',  f"L{self.r_level}", self.shock_mode)
        self.video_path = os.path.join(self.dir_base, 'PostProcess', 'video', 'interp',  f"L{self.r_level}", self.shock_mode)

class ReconsBase(object):

    def __init__(self, r_level=None, shock_mode=None, dmd_mode=None, item=None, dir_base=None):

        if dir_base is None:
            self.dir_base = dir_base_list()
        else:
            self.dir_base = dir_base

        if r_level is None:
            self.r_level = 3
        else:
            self.r_level = r_level

        if shock_mode is None:
            self.shock_mode = 'WithShock'
        else:
            self.shock_mode = shock_mode

        if dmd_mode is None:
            self.dmd_mode = 'dmd'
        else:
            self.dmd_mode = dmd_mode

        if item is None:
            self.item = 'velocityU'
        else:
            self.item = item

        self.path = os.path.join(dir_base, 'PostProcess', 'Recons',
                                 dmd_mode, f"L{r_level}", shock_mode, item)
        self.png_path = os.path.join(dir_base, 'PostProcess', 'png', 'Recons',
                                     dmd_mode, f"L{r_level}", shock_mode, item)
        self.video_path = os.path.join(dir_base, 'PostProcess', 'video','Recons',
                                       dmd_mode, f"L{r_level}", shock_mode, item)

class SepModesBase(object):

    def __init__(self, r_level=None, shock_mode=None, dmd_mode=None, item=None, dir_base=None):

        if dir_base is None:
            self.dir_base = dir_base_list()
        else:
            self.dir_base = dir_base

        if r_level is None:
            self.r_level = 3
        else:
            self.r_level = r_level

        if shock_mode is None:
            self.shock_mode = 'WithShock'
        else:
            self.shock_mode = shock_mode

        if dmd_mode is None:
            self.dmd_mode = 'dmd'
        else:
            self.dmd_mode = dmd_mode

        if item is None:
            self.item = 'velocityU'
        else:
            self.item = item

        self.path = os.path.join(dir_base, 'PostProcess', 'SepModes',
                                 dmd_mode, f"L{r_level}", shock_mode, item)
        self.png_path = os.path.join(dir_base, 'PostProcess', 'png', 'SepModes',
                                     dmd_mode, f"L{r_level}", shock_mode, item)
        self.video_path = os.path.join(dir_base, 'PostProcess', 'video', 'SepModes',
                                       dmd_mode, f"L{r_level}", shock_mode, item)


class getBaseDir(object):
    
    def __init__(self, base_spec_name=None, interp_m=None, r_level=None, 
                shock_mode=None, dmd_mode=None, item=None, dir_base=None, cm='', data_mode=''):
        '''
        cm : ccool, option: dmd, rspec
        '''
        # if base_spec_name is None:
        #     self.base_spec_name = base_spec_name
        # else:
        self.base_spec_name = base_spec_name
        
        if dir_base is None:
            self.dir_base = dir_base_list()
        else:
            self.dir_base = dir_base

        if r_level is None:
            self.r_level = 3
        else:
            self.r_level = r_level

        if shock_mode is None:
            self.shock_mode = 'WithShock'
        else:
            self.shock_mode = shock_mode

        if dmd_mode is None:
            self.dmd_mode = 'dmd'
        else:
            self.dmd_mode = dmd_mode
        if interp_m is None:
            self.interp_m = ''
        else:
            self.interp_m = interp_m

        if item is None:
            self.item = 'velocityU'
        elif isinstance (item, list) and len(item) == 1:
            item = item[0]
            self.item = item
        elif isinstance(item, str):
            self.item = item
        else:
            raise TypeError("Please check data type of item, it should be str or list fo str with len = 1 !")
        
        # if data_mode == 'CoM':
        #     com_str = 'CoM'
        # elif data_mode == 'crop':
        #     com_str = 'crop'
        # else:
        #     com_str = '' 
        #  # try: 
        #     base_spec_name
        #     raise NameError(f"base_spec_name not defined, please specify!")
        # else: 
        f = os.path.join(self.base_spec_name, self.interp_m, self.dmd_mode, f"L{self.r_level}", self.shock_mode, self.item)
        self.base_spec_name = base_spec_name
        self.path = os.path.join(self.dir_base, 'PostProcess'+data_mode, 'xdmf', f)
        
        self.png_dir = os.path.join(self.dir_base, 'PostProcess'+data_mode, 'png'+cm, f)
                                     
        self.video_path = os.path.join(self.dir_base, 'PostProcess'+data_mode, 'video', f,
                                       f"{self.item}_{self.interp_m}_{self.shock_mode}_L{self.r_level}_{cm}.mp4"
                                       )
                                            
        self.dmd_fit_dir = os.path.join(self.dir_base, 'PostProcess'+data_mode, 'dmd_fit', f
                                        # f"L{self.r_level}_{self.shock_mode}_{self.item}.pkl"
                                       )
                                    

def pvpython_path():
    # pvpy_mpi_path = '/home/overflow/Desktop/ParaView-5.8.0-osmesa-MPI-Linux-Python3.7-64bit/bin/mpiexec'
    pvpy_path = '/home/overflow/Desktop/ParaView-5.8.0-osmesa-MPI-Linux-Python3.7-64bit/bin/pvpython'
    
    pvpython_dict = {'overflow-MS-7A37':'/home/overflow/Desktop/ParaView-5.8.0-osmesa-MPI-Linux-Python3.7-64bit/bin/pvpython',
                     'kanon':'/local/disk1/fanwang/ALPACA_CWD/'
                     }
    
    
    dir_base = pvpython_dict[gethostname()]
    return dir_base

def mpiexec_pv_path():
    pvpy_mpi_path = '/home/overflow/Desktop/ParaView-5.8.0-osmesa-MPI-Linux-Python3.7-64bit/bin/mpiexec'
    pvpy_path = '/home/overflow/Desktop/ParaView-5.8.0-osmesa-MPI-Linux-Python3.7-64bit/bin/pvpython'
    
    mpiexec_pv_dict = {'overflow-MS-7A37':'/home/overflow/Desktop/ParaView-5.8.0-osmesa-MPI-Linux-Python3.7-64bit/bin/mpiexec',
                     'kanon':'/local/disk1/fanwang/ALPACA_CWD/'
                     }
    
    dir_base = mpiexec_pv_dict[gethostname()]
    return dir_base

def dmd_fit_path(dir_base=None, r_level=3, interp_m='cubic', base_spec_name='', shock_mode='WithShock', dmd_mode='dmd', 
                 item=None, extr='', data_mode=''):
    
    if dir_base is None:
        dir_base = dir_base_list()
    if data_mode != '':
        data_mode = '_' + data_mode
    # if CoM:
    #     com_str = 'CoM'
    # else:
    #     com_str = ''
    path = os.path.join(dir_base, base_spec_name, 'PostProcess'+data_mode, 'dmd_fit', base_spec_name, 
                    dmd_mode, interp_m, f"L{r_level}", shock_mode, item, f"L{r_level}_{shock_mode}_{item}{extr}.pkl")
    return path
    # return os.path.join(dir_base, 'PostProcess', "dmd_fit", dmd_mode, f"L{r_level}", shock_mode, filename)
    # return os.path.join(dir_base , 'PostProcess', dmd_mode, 'dmd_fit', 'L'+str(r_level), shock_mode)
if __name__ == '__main__':
    dir_base = dir_base_list()
    print(dir_base)
    print(os.path.exists(dir_base))