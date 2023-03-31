import h5py
# import helpers
import numpy as np
from pathlib import Path
# import torch
# from torch.utils import data

class HDF5Dataset():
    """Represents an abstract HDF5 dataset.
    
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
    

#  class HDF5Dataset_pytorch(data.Dataset):
#     """Represents an abstract HDF5 dataset.
    
#     Input params:
#         file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
#         recursive: If True, searches for h5 files in subdirectories.
#         load_data: If True, loads all the data immediately into RAM. Use this if
#             the dataset is fits into memory. Otherwise, leave this at false and 
#             the data will load lazily.
#         data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
#         transform: PyTorch transform to apply to every data instance (default=None).
#     """
#     def __init__(self, file_path, recursive=False, load_data=True, data_cache_size=3, transform=None):
#         super().__init__()
#         self.data_info = []
#         self.data_cache = {}
#         self.data_cache_size = data_cache_size
#         self.transform = transform

#         # Search for all h5 files
#         p = Path(file_path)
#         assert(p.is_dir())
#         if recursive:
#             files = sorted(p.glob('**/*.h5'))
#         else:
#             files = sorted(p.glob('*.h5'))
#         if len(files) < 1:
#             raise RuntimeError('No hdf5 datasets found')

#         for h5dataset_fp in files:
#             self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            
#     def __getitem__(self, index):
#         # get data
#         x = self.get_data("data", index)
#         if self.transform:
#             x = self.transform(x)
#         else:
#             x = torch.from_numpy(x)

#         # get label
#         y = self.get_data("label", index)
#         y = torch.from_numpy(y)
#         return (x, y)

#     def __len__(self):
#         return len(self.get_data_infos('data'))
    
#     def _add_data_infos(self, file_path, load_data):
#         with h5py.File(file_path, 'r') as h5_file:
#             # Walk through all groups, extracting datasets
#             for gname, group in h5_file.items():
#                 for dname, ds in group.items():
#                     # if data is not loaded its cache index is -1
#                     idx = -1
#                     if load_data:
#                         # add data to the data cache
#                         idx = self._add_to_cache(ds[()], file_path)
                    
#                     # type is derived from the name of the dataset; we expect the dataset
#                     # name to have a name such as 'data' or 'label' to identify its type
#                     # we also store the shape of the data in case we need it
#                     self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds[()].shape, 'cache_idx': idx})

#     def _load_data(self, file_path):
#         """Load data to the cache given the file
#         path and update the cache index in the
#         data_info structure.
#         """
#         with h5py.File(file_path) as h5_file:
#             for gname, group in h5_file.items():
#                 for dname, ds in group.items():
#                     # add data to the data cache and retrieve
#                     # the cache index
#                     idx = self._add_to_cache(ds[()], file_path)

#                     # find the beginning index of the hdf5 file we are looking for
#                     file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

#                     # the data info should have the same index since we loaded it in the same way
#                     self.data_info[file_idx + idx]['cache_idx'] = idx

#         # remove an element from data cache if size was exceeded
#         if len(self.data_cache) > self.data_cache_size:
#             # remove one item from the cache at random
#             removal_keys = list(self.data_cache)
#             removal_keys.remove(file_path)
#             self.data_cache.pop(removal_keys[0])
#             # remove invalid cache_idx
#             self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

#     def _add_to_cache(self, data, file_path):
#         """Adds data to the cache and returns its index. There is one cache
#         list for every file_path, containing all datasets in that file.
#         """
#         if file_path not in self.data_cache:
#             self.data_cache[file_path] = [data]
#         else:
#             self.data_cache[file_path].append(data)
#         return len(self.data_cache[file_path]) - 1

#     def get_data_infos(self, type):
#         """Get data infos belonging to a certain type of data.
#         """
#         data_info_type = [di for di in self.data_info if di['type'] == type]
#         return data_info_type

#     def get_data(self, type, i):
#         """Call this function anytime you want to access a chunk of data from the
#             dataset. This will make sure that the data is loaded in case it is
#             not part of the data cache.
#         """
#         fp = self.get_data_infos(type)[i]['file_path']
#         if fp not in self.data_cache:
#             self._load_data(fp)
        
#         # get new cache_idx assigned by _load_data_info
#         cache_idx = self.get_data_infos(type)[i]['cache_idx']
#         return self.data_cache[fp][cache_idx]
    
if __name__ == '__main__':
    import sys
    sys.path.append('src')
    # import HDF5Dataset
    import numpy as np
    import os 
    import h5py
    import comm_tools
    import utils_dmd
    from xdmfWriter import save_xdmf, xdmf_index_ind
    from get_dir_base import (dir_base_list, getBaseDir, InterpBasedir, dmd_save_path,
                            DatasetBase, pvpython_path, mpiexec_pv_path)
    # from torch.utils.data import Dataset, DataLoader
    # from torch.utils import data
    
    save_recons = True
    # save_modes = True
    save_sep_modes = False
    dmd_mode = 'dmd'
    plot_modes = False
    mid_name = ''
    test_mode = False
    write_multip = True
    shock_mode_list = ['WithShock', 'WithoutShock']
    shock_mode = shock_mode_list[0]
    dir_base = dir_base_list()
    r_level = 3
    nproc = 4
    pre_load= False
    save_dmd_fit = True
    input_file_base = InterpBasedir(r_level, shock_mode)
    input_file_path = input_file_base.path
    # dmd_npy_path = '/media/overflow/Volume/ALPACA_CWD/Postprocess/dmd_fit/L3/WithShock'
    read_param_list = [['velocityU'], ['pressure']]
    # save_param = ['velocityX', 'velocityY', 'velocityU', 'density','pressure']
    save_param_list = [['velocityU'], ['pressure']]
    i = 0
    # print(f"interp_base_path: {interp_base_path}")
    # print(f"fit dmd class save path: {dmd_save_path()}")
    groupname_dict = dict(zip(['points', 'cells', 'point_data', 'cell_data'],
                          ['domain/vertex_coordinates', 'domain/cell_vertices',
                           'simulation', 'simulation']))     
    read_param = read_param_list[0]
    save_param = save_param_list[0]

    recons_base = getBaseDir('recons', r_level, shock_mode, dmd_mode, item=save_param[0] )
    recons_base_path = recons_base.path
    modes_base = getBaseDir('modes', r_level=r_level, shock_mode=shock_mode, dmd_mode=dmd_mode, item=save_param[0])
    modes_base_path = modes_base.path
    modes_sep_base = getBaseDir('sep_modes', r_level=r_level, shock_mode=shock_mode, dmd_mode=dmd_mode,item=save_param[0])
    modes_sep_base_path = modes_sep_base.path

    print(f"recons_base_path: {recons_base_path}")
    print(f"modes_base_path: {modes_base_path}")
    print(f"modes_sep_base_path: {modes_sep_base_path}")
    time_ref_path = comm_tools.dir_list(input_file_path, 'aero*.xdmf')
    time_series = utils_dmd.extract_time_xdmf(time_ref_path[0])
    cell_info = np.load(os.path.join(input_file_path, 'aerobreakup_interp_cell.npz'))
    pattern = '*.h5'
    file_name = []
    # file_path = glob.glob(os.path.join(work_path, input_file_dir, pattern))
    file_path = comm_tools.dir_list(input_file_path, pattern)  
    file_path = np.sort(file_path)
    if test_mode:
        n = 10
        file_path = file_path[:n]
        time_series = time_series[:n]
    file_path = file_path[:500]
    file_name = [os.path.splitext(os.path.basename(path))[0] for path in file_path]
    print(f"file path check: {file_path[0]}")
    print(f"read param: {read_param}")
    print(f"numbers of data to load: {len(file_path)}")
    print(f"file name check: {file_name[0]}")
    
    hdf5_loader = HDF5Dataset(recons_base_path)
    # hdf5_loader = torch.utils.data.DataLoader(HDF5Dataset(input_file_path))
    x = []
    xx = hdf5_loader.get_data('velocityU', 0)
    for i in range(500):
        x.append(hdf5_loader.get_data('velocityU', i))
    x = np.array(x)
    print('debug')