import os 
import h5py
import numpy as np

def count_digits(num):
    count = 0
    while num != 0:
        num //= 10
        count += 1
    return count


def save_hdf5(filename, data_dict):
    with h5py.File(filename, 'w') as f:
        for k in data_dict.keys():
            dset = f.create_dataset(k, data=data_dict[k])