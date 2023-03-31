# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 23:02:02 2020

@author: AdminF
"""
from multiprocessing import Process, Manager

def init():
    
    global old_root
    old_root = Manager().dict()

    global idx_ele