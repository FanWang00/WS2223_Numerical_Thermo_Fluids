#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:02:42 2020

@author: fan
"""

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata
from numba import njit
import numpy as np
from numba.core.errors import NumbaWarning
import warnings
warnings.simplefilter("ignore", NumbaWarning)
import h5py
import os 
# from functools import wraps

#from pyevtk.hl import gridToVTK
# def save_xdmf(tmp_xy, cells_tmp, groupname_dict, 
#               xdmf_path, dmd_recons, time_series):
#     print('start')
#     # xdmf_path = os.path.join(save_modes_path, file_name[i] + '.xdmf')
#     dicts = {'velocityU': dmd_recons}
#     # tmp_y = values_interp['velocityY']
#     # cells_tmp = {'quad': tmp_vertex}
#     # dicts = {'velocityU': reconstructed_data}
#     print("start writing")
#     write_xdmf(time_series, xdmf_path, tmp_xy, cells_tmp, 
#                dicts, groupname_dict)
#     print(os.getpid())
def xp_agnostic_func(X):
    
    try:
            xp = cp.get_array_module(X)
    except Exception:
            xp = np
    return xp

def xp_agenostic(func):
    def wrapper(*args, **kwargs):
        global xp
        try:
            xp = cp.get_array_module(args[0])
        except Exception:
            xp = np
        print(xp)
        try:
            return func(*args, **kwargs)
        except Exception:
            return None
    return wrapper

class HDF5Dataset_():
    
    """
    adopt from https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5

    Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive=False, load_data=True, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            
    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('data'))
    
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path, 'r') as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds[()], file_path)
                    
                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds[()].shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds[()], file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]
    
def load_data(file_path, read_param):
    
    print('loading data...')
    # snapshots = np.load(file_path[0]).flatten()
    # for path in file_path[1:]:
    #     snapshots_tmp = np.load(path).flatten()
    #     snapshots = np.vstack((snapshots, snapshots_tmp))
    # snapshots = snapshots.T
    '''
    file_path: list of all input files
    read_param: list of parameters to read
    '''
    num_file = len(file_path)
    values_read_param = {}
    for item in read_param:
        values_read_param[item] = []

    for i in range(num_file):
        path = file_path[i]
        # timestep = '{:.15e}'.format(time_series[i])
        # timestep = time_series[i]
        data = h5py.File(path, 'r')
        f_name =  os.path.split(path)[-1][:-3]
        ## ['density', 'interface_velocity', 'levelset', 'partition',
        ## 'pressure', 'velocityX', 'velocityY']
        tmp_xy = data['domain']['vertex_coordinates'][()][:, 0:2]
        # tmp_xy3d = data['domain']['vertex_coordinates'][()]
        tmp_vertex = data['domain']['cell_vertices'][()]
        cell_center = vtx_center(tmp_xy, tmp_vertex)
        cells_tmp = {'quad': tmp_vertex}
        # tmp_data = [data['simulation'][item][()] for item in read_param]
        # values_read_param_tmp = dict(zip(read_param, tmp_data))
        [values_read_param[item].append(data['simulation'][item][()]) for item in read_param]
    print("data loaded")
    return values_read_param, tmp_xy, cells_tmp
    
def mesh_split(XY, Vertex, num_files, file_name, mesh_plot=False):

    max_level = 7
    C_mesh = 1.250000000e-02
    F_mesh = 9.765625000e-05
    mesh_size = [F_mesh * 2 ** i for i in range(max_level, -1,-1)]

    idx_mesh_set = []
    for j in range(num_files):
        vtx  = Vertex[j] 
        xy = XY[j]
        idx_mesh = []
        sum_check = 0
        idx_mesh = []
        for i in range(8):
            mesh_size_point = abs(xy[vtx[:, 1], 0] - xy[vtx[:, 0], 0])
            # tmp_idx = np.nonzero(np.isclose(mesh_size_point, [mesh_size[i]], rtol=1e-7 ))
            tmp_idx = np.nonzero(abs(mesh_size_point - [mesh_size[i]]) <  1e-6 )
            sum_check = sum_check + np.shape(tmp_idx)[1]
            idx_mesh.append(tmp_idx)
        idx_mesh_set.append(idx_mesh)
        #     plt.figure(i)
        #     plt.xlim([0, 0.2])
        #     plt.ylim([0, 0.2])
        #     plt.scatter(xy[vtx[tmp_idx], 0], xy[vtx[tmp_idx], 1])
        if sum_check != vtx.shape[0]:
            print('Somthing Wrong!!!')
    
    ## plot 1 level over snapshots single plot Ver.
    # level_plot = 1
    # for j in range(num_files):
    #     vtx = Vertex[j]
    #     xy = XY[j]
    #     print(vtx.shape)
    #     idx_mesh = idx_mesh_set[j]
    #     plt.figure(j)
    #     plt.xlim([0, 0.2])
    #     plt.ylim([0, 0.2])
    #     plt.title(file_name[j])
    #     plt.scatter(xy[vtx[idx_mesh[level_plot]], 0], xy[vtx[idx_mesh[level_plot]], 1])
            
    if mesh_plot:
        
        xlim = [0, 0.2]
        ylim = [0, 0.2]
        fig, axes = plt.subplots(num_files, max_level)
        fig.set_size_inches(20, 15)
        for level_plot in range(max_level): 
        
            for j in range(num_files):
                
                vtx = Vertex[j]
                xy = XY[j]
                # print(vtx.shape)
                idx_mesh = idx_mesh_set[j]
                # plt.figure(j)
                # plt.xlim([0, 0.2])
                # plt.ylim([0, 0.2])
                axes[j, level_plot].set_title(file_name[j])
                axes[j, level_plot].scatter(xy[vtx[idx_mesh[level_plot]], 0], xy[vtx[idx_mesh[level_plot]], 1])
                
        plt.setp(axes, xlim=xlim, ylim=ylim)
        plt.tight_layout()
        plt.show()
        
    return idx_mesh_set

def vtx_center(xy, vtx):
    
    return (xy[vtx[:, 2]] + xy[vtx[:, 0]]) / 2

def mesh_interp(mesh_x, mesh_y, mesh_k, values, interp_m='linear'):
    # print(f'interp method: {interp_m}')
    values_interp = griddata(mesh_k, values, (mesh_x, mesh_y), method=interp_m)
    return values_interp

def extract_time_xdmf(path, ver=2):
    import xml.etree.ElementTree as ET
    # file_path = os.path.join('F:', os.sep, 'ALPACA','aerobreakup_l4_40', 'domain',
    #                      'aerobreakup_l4_40.xdmf')
    root = ET.parse(path).getroot()
    
    if ver == 2:
        for type_tag in root.findall('Domain/Grid/Time/DataItem'):
            time_text = type_tag.text.split()
            # time_text = time_text.strip('\t')
        time_seris = np.genfromtxt(time_text, delimiter=',')       
        return time_seris
    elif ver==3:
        t_tag = root.findall('.//Time')
        time_series = [float(t.attrib['Value']) for t in iter(t_tag)]
        return time_series

def write_h5(values, groups, save_path):
    
    hf = h5py.File(save_path, 'w')
    for i in range(len(groups)):
        
        hf.create_dataset(groups[i], data=values[i])
    hf.close()
    
@njit
def vertex_info_cal(vertex_num_y, cell_num_x, cell_num_y):

    vertex = np.zeros((cell_num_y, cell_num_x, 4), dtype=np.uint32)
    for i in range(cell_num_y):
        for j in range(cell_num_x):
            vertex[i, j] = [int(j*vertex_num_y + i),
                            int(j*vertex_num_y + i+1),
                            int((j+1)*vertex_num_y + i+1),
                            int((j+1)*vertex_num_y + i)]
    return vertex


def vertex_info(mesh_x_interp, mesh_y_interp):
    
    vertex_num_x = mesh_x_interp.shape[0]
    vertex_num_y = mesh_x_interp.shape[1]
    cell_num_x = vertex_num_x -1
    cell_num_y = vertex_num_y -1 
    # vertex = np.zeros((cell_num_y, cell_num_x, 4), dtype=np.uint32)
    xy = []
    
    for i in range(vertex_num_x):
        for j in range(vertex_num_y):
            xy.append([mesh_x_interp[i, j], mesh_y_interp[i, j]])
    xy = np.array(xy)
    
    # for i in range(cell_num_y):
    #     for j in range(cell_num_x):
    #         vertex[i, j] = [int(j*vertex_num_y + i),
    #                         int(j*vertex_num_y + i+1),
    #                         int((j+1)*vertex_num_y + i+1),
    #                         int((j+1)*vertex_num_y + i)]

    vertex = vertex_info_cal(vertex_num_y, cell_num_x, cell_num_y)
    vertex = np.reshape(vertex, (-1, 4), order='C')
    # mesh_points_interp = xy[vertex]
    # plt.scatter(xy[vertex[:10], 0], xy[vertex[:10], 1])
    return xy, vertex


def saveVTK(values, vtk_file, dim):
    import pyvista as pv
    '''
    sample code:
    grid = pv.UniformGrid()
    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = np.array([nx, ny, nz]) + 1

    # Edit the spatial reference
    temp = np.reshape(snapshots[:,i], (nx, ny, 1), order='C')

    # Add the data values to the cell data
    grid.cell_arrays["values"] = dmd_check_recons[:,i].real.flatten(order='C')
    # vtk_file = 'VTKOutput_py/recons_test_pv/'
    vtk_file = vtk_dir+ 'Flow' + str(i) + '.vtk'
    grid.save(vtk_file)
    # grid.plot(show_edges=True)
    '''
    grid = pv.UniformGrid()
    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = dim + 1
    # Edit the spatial reference
    # temp = np.reshape(snapshots[:,i], (nx, ny, 1), order='C')
    # Add the data values to the cell data
    grid.cell_arrays["values"] = values
    # vtk_file = 'VTKOutput_py/recons_test_pv/'
    grid.save(vtk_file)
    # print('file saved')
    # grid.save(vtk_file)
    # grid.plot(show_edges=True)


# def saveVTK_pyevtk(save_format):
#
#    '''
#     x = np.arange(0, ny+1)
#     y = np.arange(0, nx+1)
#     z = np.array([0])
#     t = [[1]]
#     save_format = {'filename': None,
#                'x': None, 'y': None, 'z': None,
#                'cellData': None,
#                'pointData': None
#                 }
#
#     save_format['x'] = x
#     save_format['y'] = y
#     save_format['z'] = z
#
#
#     noSlices = 1
#     for i in range(len(t)):
#        vtk_str = fn_DMD + 'FF' + str(t[i]) + '.vtk'
#        Xdmd_tmp = np.dstack([np.real(np.reshape(Xdmd[:,i], (ny, nx), order='C'))] * 1)
#        gridToVTK(vtk_str, x, y, z, cellData = {'ff':Xdmd_tmp})
#        save_format['filename'] = vtk_str
#        save_format['cellData'] = {'Flow_Field':Xdmd_tmp}
#        saveVTK(save_format)
#
#
#        cellData = None
#        pointData = None
#        x, y, z = save_format['x'], save_format['y'], save_format['z']
#        filename = save_format['filename']
# '''
#
#    cellData = None
#    pointData = None
#    x, y, z = save_format['x'], save_format['y'], save_format['z']
#    filename = save_format['filename']
#    if save_format['cellData'] != None:
#        cellData = save_format['cellData']
#    if save_format['pointData'] != None:
#        pointData = save_format['pointData']
#
#    gridToVTK(filename, x, y, z, cellData=cellData, pointData=pointData)

if __name__ == '__main__':
    
    import numpy as np
    import h5py
    # import meshio
    import os
    import glob
    

    ## list h5 file 
    work_path = os.getcwd()
    h5_path = 'L4'
    pattern = 'data*.h5'
    file_name = []
    C_mesh = 1.250000000e-02
    F_mesh = 9.765625000e-05
    for file in os.listdir(h5_path):
        # if file.endswith("h5"):
        # if .glob('data*.h5'):
        file_path = glob.glob(os.path.join(work_path, h5_path, pattern))
        # file_name = list(filter(lambda pattern: pattern[0:] == "s",seq))
        # file_name.append(file_path)
            # print(os.path.join("", file))
    file_path = np.sort(file_path)
    file_name = [os.path.split(path)[-1] for path in file_path]
    print(file_path)
    print(file_name)
    
    # hdf5_path = 'data_0.002156.h5'
    ## read h5 file
    vX = []
    vY = []
    U = []
    XY = []
    Vertex = []
    # file_name = ['D://GitHub//ALPACA_Test//aerobreakup//domain//data_0.000000.h5']
    # file_name = ['aerobreakup//domain//data_0.000000.h5']
    num_files = len(file_path)
    for path in file_path:
        
        data = h5py.File(path, 'r')
        tmp_vx = data['simulation']['velocityX'][()]
        tmp_vy = data['simulation']['velocityY'][()]
        tmp_xy = data['domain']['vertex_coordinates'][()][:,0:2]
        tmp_vertex = data['domain']['cell_vertices'][()]
        XY.append(tmp_xy)
        Vertex.append(tmp_vertex)
        vX.append(tmp_vx)
        vY.append(tmp_vy)
    # idx_mesh = mesh_split(XY, Vertex, num_files, file_name, mesh_plot=False)
    xy = XY[0]
    vtx = Vertex[0]
    vx = vX[0]
    vy = vY[0]
    c = ['r', 'b','y','c']
    mesh_size = F_mesh * 2**4
    
    # for i in range(4):
    # plt.scatter(xy[vtx,0], xy[vtx,1])
    node = vtx_center(xy, vtx)
    plt.scatter(node[:, 0], node[:, 1], marker='.')
    
    mesh_X = np.arange(0, 0.2, mesh_size)
    mesh_x, mesh_y = np.meshgrid(mesh_X, mesh_X)
    # mesh_vx, mesh_vy = np.meshgrid(vx, vy)
    plt.quiver(node[:, 0], node[:, 1], vx, vy)
    
    mesh_X_interp = np.arange(mesh_size, 0.2, mesh_size)
    mesh_x_interp, mesh_y_interp = np.meshgrid(mesh_X_interp, mesh_X_interp)
    values_interp_x = mesh_interp(mesh_x_interp, mesh_y_interp, node, vx)
    values_interp_y = mesh_interp(mesh_x_interp, mesh_y_interp, node, vy)
    plt.figure(2)
    plt.quiver(mesh_x_interp, mesh_y_interp, values_interp_x, values_interp_y)
    